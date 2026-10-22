// The dashboard's glasses window (3D scene, WebXR page): fullscreen as soon as it opens.
// A page can only ask for fullscreen after a click in it; the extension needs none.

chrome.runtime.sendMessage({ type: 'fullscreen' }).catch(() => {});
