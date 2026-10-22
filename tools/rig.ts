// Sanity check of the stereo rig built from the factory calibration (run after `npm run probe`).
import { readFileSync } from 'node:fs';
import { parseCalibration } from '../src/device/calibration';
import { stereoRig, triangulate } from '../src/hands/stereoGeometry';
const c = parseCalibration(new Uint8Array(readFileSync(process.argv[2] ?? 'calibration.bin')));
const [l, r] = stereoRig(c)!;
console.log('origins (head frame, m)', l.origin.map(v => v.toFixed(3)), r.origin.map(v => v.toFixed(3)), 'baseline', Math.hypot(l.origin[0] - r.origin[0], l.origin[1] - r.origin[1], l.origin[2] - r.origin[2]).toFixed(3));
console.log('principal rays', l.ray(l.cx, l.cy).map(v => v.toFixed(3)), r.ray(r.cx, r.cy).map(v => v.toFixed(3)));
// Project a point 40 cm ahead into both cameras and triangulate it back.
const P: [number, number, number] = [0.05, -0.1, -0.4];
const toPix = (cam: typeof l) => {
  // invert toHead: p_cam = R^T (P - o); then distort with the FOV model
  const d = [P[0] - cam.origin[0], P[1] - cam.origin[1], P[2] - cam.origin[2]];
  const R = cam.R; const pc = [R[0] * d[0] + R[3] * d[1] + R[6] * d[2], R[1] * d[0] + R[4] * d[1] + R[7] * d[2], R[2] * d[0] + R[5] * d[1] + R[8] * d[2]];
  const xu = pc[0] / pc[2], yu = pc[1] / pc[2], ru = Math.hypot(xu, yu), w = c.cameras[cam === l ? 0 : 1].distortion[0];
  const rd = Math.atan(2 * ru * Math.tan(w / 2)) / w, s = rd / ru;
  return [cam.fx * xu * s + cam.cx, cam.fy * yu * s + cam.cy];
};
const [ul, vl] = toPix(l), [ur, vr] = toPix(r);
console.log('pixels', [ul, vl].map(v => v.toFixed(1)), [ur, vr].map(v => v.toFixed(1)));
const t = triangulate(l.origin, l.ray(ul, vl), r.origin, r.ray(ur, vr))!;
console.log('triangulated', t.point.map(v => v.toFixed(4)), 'gap', t.gap.toExponential(2));
