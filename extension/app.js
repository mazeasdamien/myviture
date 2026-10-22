// The myViture dashboard (localhost): forwards its WebXR feed (pose, eyes, hands) to the
// service worker, and the worker's requests (display mode, hit-test rays) back to it.

const isDashboard = !!document.getElementById('btn-webxr');
if (isDashboard) {
  const feed = new BroadcastChannel('myviture-xr');
  const control = new BroadcastChannel('myviture-xr-control');
  let port = null;

  const connect = () => {
    port = chrome.runtime.connect({ name: 'app' });
    port.onMessage.addListener(m => control.postMessage(m));
    port.onDisconnect.addListener(() => { port = null; });
  };

  feed.onmessage = e => {
    // Extension reloaded or removed: this copy of the script is orphaned, stop.
    if (!chrome.runtime?.id) { feed.close(); control.close(); return; }
    try {
      if (!port) connect();
      port.postMessage({ type: 'feed', feed: e.data });
    } catch {
      port = null;
    }
  };
}
