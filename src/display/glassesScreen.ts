// Opens a dedicated window on the glasses' screen, in side-by-side 3D.
//
// Order matters: switching the glasses to 3D changes their screen to 3840x1200, so
// switch first, wait for Windows to report the new size, then place a window
// exactly over that screen (Window Management API) and ask for fullscreen.

type ScreenInfo = Screen & { label: string; isPrimary: boolean; availLeft: number; availTop: number; left: number; top: number };
type ScreenDetails = EventTarget & { screens: ScreenInfo[] };

const SBS_MIN_WIDTH = 3000;

async function screenDetails(): Promise<ScreenDetails | null> {
  const w = window as Window & { getScreenDetails?: () => Promise<ScreenDetails> };
  if (!w.getScreenDetails) return null;
  try { return await w.getScreenDetails(); } catch { return null; }
}

/** The glasses' screen: labelled VITURE, else the first non-primary screen. */
function glassesScreen(d: ScreenDetails): ScreenInfo | null {
  return d.screens.find(s => /viture/i.test(s.label)) ?? d.screens.find(s => !s.isPrimary) ?? null;
}

export type GlassesWindow =
  | { ok: true; win: Window; label: string; switched: boolean }
  | { ok: false; reason: 'no-api' | 'mirrored' | 'not-3d' | 'blocked' };

/** switchTo3d is called only if the glasses' screen is not already 3840 wide. */
export async function openGlassesWindow(switchTo3d: () => Promise<void> | void, page = '/glasses.html'): Promise<GlassesWindow> {
  const d = await screenDetails();
  if (!d) return { ok: false, reason: 'no-api' };
  let target = glassesScreen(d);
  if (!target) return { ok: false, reason: 'mirrored' };

  let switched = false;
  if (target.width < SBS_MIN_WIDTH) {
    await switchTo3d();
    switched = true;
    target = await waitForWidth(d, SBS_MIN_WIDTH, 6000);
    if (!target) return { ok: false, reason: 'not-3d' };
  }
  const geometry = `left=${target.left},top=${target.top},width=${target.width},height=${target.height}`;
  // "fullscreen" asks Chrome for a fullscreen popup; it is ignored where unsupported.
  const win = window.open(page, 'myviture-glasses', `popup,${geometry},fullscreen`)
    ?? window.open(page, 'myviture-glasses', `popup,${geometry}`);
  // window.open needs the click's user activation, which can expire while the screen switches mode.
  if (!win) return { ok: false, reason: 'blocked' };
  return { ok: true, win, label: target.label || 'lunettes', switched };
}

/** Resolves with the dedicated window's canvas once its page has loaded. */
export function glassesCanvas(win: Window, timeoutMs = 5000): Promise<HTMLCanvasElement | null> {
  return new Promise(resolve => {
    const t0 = performance.now();
    const poll = () => {
      if (win.closed) return resolve(null);
      try {
        const c = win.document.getElementById('view') as HTMLCanvasElement | null;
        if (c && win.document.readyState === 'complete') return resolve(c);
      } catch { /* still about:blank on another origin */ }
      if (performance.now() - t0 > timeoutMs) return resolve(null);
      setTimeout(poll, 50);
    };
    poll();
  });
}

function waitForWidth(d: ScreenDetails, width: number, timeoutMs: number): Promise<ScreenInfo | null> {
  return new Promise(resolve => {
    let poll = 0, timer = 0;
    const done = (s: ScreenInfo | null) => {
      clearInterval(poll);
      clearTimeout(timer);
      d.removeEventListener('screenschange', check);
      resolve(s);
    };
    function check() {
      const s = glassesScreen(d);
      if (s && s.width >= width) done(s);
    }
    d.addEventListener('screenschange', check);
    poll = window.setInterval(check, 200);
    timer = window.setTimeout(() => done(null), timeoutMs);
    check();
  });
}
