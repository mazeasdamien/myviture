// Service worker: fans the dashboard's feed out to the sites using WebXR, and when one
// starts a session, switches the glasses to 3D and moves that tab fullscreen onto them.

const SBS_MIN_WIDTH = 3000;
let app = null;
const sites = new Set();
/** tabId -> where the tab came from, and whether we switched the glasses to 3D */
const placed = new Map();

chrome.runtime.onConnect.addListener(port => {
  if (port.name === 'app') {
    app = port;
    port.onMessage.addListener(m => { if (m.type === 'feed') for (const s of sites) s.postMessage(m); });
    port.onDisconnect.addListener(() => { if (app === port) app = null; });
  } else if (port.name === 'site') {
    sites.add(port);
    port.onMessage.addListener(m => {
      if (m.type === 'session') (m.active ? enter : leave)(port.sender.tab);
      else if (m.type === 'rays') app?.postMessage(m);
    });
    port.onDisconnect.addListener(() => {
      sites.delete(port);
      if (port.sender.tab) leave(port.sender.tab);
    });
  }
});

const portOf = tabId => [...sites].find(p => p.sender.tab?.id === tabId);
const reply = (tabId, ok, reason) => portOf(tabId)?.postMessage({ type: 'placed', ok, reason });

/** The glasses' display: named VITURE, else the first non-primary one. */
async function glassesDisplay() {
  const all = await chrome.system.display.getInfo();
  return all.find(d => /viture|luma/i.test(d.name)) ?? all.find(d => !d.isPrimary && !d.mirroringSourceId) ?? null;
}

async function waitWide(ms) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    const d = await glassesDisplay();
    if (d && d.bounds.width >= SBS_MIN_WIDTH) return d;
    await new Promise(r => setTimeout(r, 200));
  }
  return null;
}

async function enter(tab) {
  if (!tab || placed.has(tab.id)) return reply(tab?.id, true);
  try {
    let d = await glassesDisplay();
    if (!d) return reply(tab.id, false, 'écran des lunettes introuvable (Win+P → Étendre)');
    let switched = false;
    if (d.bounds.width < SBS_MIN_WIDTH) {
      if (!app) return reply(tab.id, false, 'ouvre le tableau de bord myViture, ou passe les lunettes en 3D (bouton 2D/3D)');
      app.postMessage({ type: 'display3d', on: true });
      switched = true;
      d = await waitWide(6000);
      if (!d) return reply(tab.id, false, 'les lunettes ne sont pas passées en 3D : appuie sur leur bouton 2D/3D');
    }
    const { left, top, width, height } = d.bounds;
    placed.set(tab.id, { windowId: tab.windowId, index: tab.index, switched, newWindowId: null });
    const win = await chrome.windows.create({ tabId: tab.id, left, top, width, height, focused: true });
    placed.get(tab.id).newWindowId = win.id;
    await chrome.windows.update(win.id, { state: 'fullscreen' });
    // Chrome slows a background tab's GPU: the dashboard's depth (hit test) would stop.
    // The moved tab often leaves another tab in front in the dashboard's window.
    const dash = app?.sender?.tab;
    if (dash && dash.id !== tab.id) chrome.tabs.update(dash.id, { active: true }).catch(() => {});
    // Let the page see its new size before the session starts.
    setTimeout(() => reply(tab.id, true), 400);
  } catch (e) {
    reply(tab.id, false, String(e?.message ?? e));
  }
}

async function leave(tab) {
  const info = placed.get(tab.id);
  if (!info) return;
  placed.delete(tab.id);
  if (info.switched) app?.postMessage({ type: 'display3d', on: false });
  try {
    await chrome.tabs.move(tab.id, { windowId: info.windowId, index: info.index });
    await chrome.tabs.update(tab.id, { active: true });
    await chrome.windows.update(info.windowId, { focused: true });
  } catch {
    // Original window gone (or the tab closed): leave the tab where it is, windowed.
    if (info.newWindowId) chrome.windows.update(info.newWindowId, { state: 'normal' }).catch(() => {});
  }
}
