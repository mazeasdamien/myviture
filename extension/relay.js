// Every site: relays between the injected WebXR runtime (page, via window.postMessage)
// and the extension's service worker. Connects only once the page asks for WebXR.

const TAG = '__myvitureXr';
let port = null;

function connect() {
  if (port) return port;
  port = chrome.runtime.connect({ name: 'site' });
  port.onMessage.addListener(m => window.postMessage({ [TAG]: m.type, ...m }, '*'));
  port.onDisconnect.addListener(() => { port = null; });
  return port;
}

window.addEventListener('message', e => {
  if (e.source !== window || typeof e.data?.[TAG] !== 'string') return;
  const type = e.data[TAG];
  if (type === 'hello') connect();
  else if (type === 'session') connect().postMessage({ type: 'session', active: !!e.data.active });
  else if (type === 'rays') connect().postMessage({ type: 'rays', rays: e.data.rays });
});
