# myViture WebXR (Chrome extension)

Gives any WebXR site the glasses as an `immersive-ar` device. The site's own
"Enter AR / VR" button then starts a session on the glasses.

- `xr-runtime.js`: our WebXR runtime (`src/webxr/runtime.ts`), injected before the site's scripts. Rebuild with `npm run build:extension`.
- `relay.js`: page ↔ service worker, on every site.
- `app.js`: on the myViture dashboard, forwards its feed (pose, eyes, hands) and takes the 3D requests.
- `background.js`: fans the feed out; on session start, switches the glasses to 3D and moves the tab fullscreen onto their screen; on end, puts it back.

Supported: `hit-test` (rays walked against the dashboard's depth map), `anchors` (kept in
VITURE's tracking frame, and persistent anchors in the room map, so they come back after a
restart once the room is recognised), hands. Not supported: layers, depth sensing, DOM overlay.

The 3D switch goes through the bridge, which sends our MCU frame on the engine's USB handle
(the engine's own `switch_display_mode` targets the camera board and returns -1 on the Luma Ultra).

Install: `chrome://extensions` → Developer mode → Load unpacked → this folder.
Needs the dashboard open and connected. Escape ends the session.
