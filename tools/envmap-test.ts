// Synthetic check of the environment map: a textured room seen in one session, then a
// second session whose frame is turned 40 degrees and shifted. Relocalisation must find
// that transform.
import * as THREE from 'three';
import { EnvMap } from '../src/vio/envMap';
import { DEPTH_H, DEPTH_W, depthFrameFrom, type RectAxes } from '../src/depth/stereoDepth';

const f = 150;
const axes: RectAxes = { ex: [1, 0, 0], ey: [0, -1, 0], ez: [0, 0, -1], origin: [0, 0, 0], baseline: 0.136, f };
// Room: random blobs on 4 walls, as an intensity function of a world point.
const blobs = Array.from({ length: 1200 }, () => {
  const wall = (Math.random() * 4) | 0, a = (Math.random() - 0.5) * 6, b = Math.random() * 2.5 - 1;
  const p = wall === 0 ? [a, b, -3] : wall === 1 ? [a, b, 3] : wall === 2 ? [-3, b, a] : [3, b, a];
  return { p, r: 0.03 + Math.random() * 0.08, v: Math.random() * 200 };
});
function render(q: THREE.Quaternion, pos: THREE.Vector3) {
  const img = new Uint8Array(DEPTH_W * DEPTH_H).fill(60), depth = new Float32Array(DEPTH_W * DEPTH_H);
  const inv = q.clone().invert();
  for (let v = 0; v < DEPTH_H; v++) for (let u = 0; u < DEPTH_W; u++) {
    const dir = new THREE.Vector3((u + 0.5 - DEPTH_W / 2) / f, -(v + 0.5 - DEPTH_H / 2) / f, -1).applyQuaternion(q);
    // Intersect the box room |x|,|z| <= 3.
    let s = Infinity;
    for (const [k, w] of [[0, 3], [0, -3], [2, 3], [2, -3]] as const) { const d = dir.getComponent(k); if (Math.abs(d) > 1e-6) { const t = (w - pos.getComponent(k)) / d; if (t > 0) s = Math.min(s, t); } }
    const P = pos.clone().addScaledVector(dir, s);
    let val = 60;
    for (const bl of blobs) { const dd = Math.hypot(P.x - bl.p[0], P.y - bl.p[1], P.z - bl.p[2]); if (dd < bl.r) { val = bl.v; break; } }
    img[v * DEPTH_W + u] = val;
    depth[v * DEPTH_W + u] = P.clone().sub(pos).applyQuaternion(inv).dot(new THREE.Vector3(0, 0, -1));
  }
  return { img: { w: DEPTH_W, h: DEPTH_H, px: img }, d: depthFrameFrom({ width: DEPTH_W, height: DEPTH_H, depth, left: img, validRatio: 1, axes }) };
}
const map = new EnvMap();
// Session 1: look around from the centre.
for (let k = 0; k < 8; k++) {
  const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), (k / 8) * 2 * Math.PI);
  const pos = new THREE.Vector3(0.2 * Math.sin(k), 0, 0.2 * Math.cos(k));
  const { img, d } = render(q, pos);
  const st = map.process(img, d, q, pos);
  console.log('session 1 frame', k, 'points', st.points, 'inliers', st.inliers, `${st.ms.toFixed(0)} ms`);
}
// Session 2: its frame = map frame turned by 40 deg and shifted (0.5, 0, -0.3).
const yaw = THREE.MathUtils.degToRad(40), shift = new THREE.Vector3(0.5, 0, -0.3);
const Rs = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
map.newSession();
for (let k = 0; k < 3; k++) {
  const qWorld = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.3 + k * 0.4);
  const pWorld = new THREE.Vector3(0.3, 0, 0.1 * k);
  const { img, d } = render(qWorld, pWorld);
  // The same view, expressed in the session frame: x_s = Rs x_m + shift.
  const qS = Rs.clone().multiply(qWorld), pS = pWorld.clone().applyQuaternion(Rs).add(shift);
  const st = map.process(img, d, qS, pS);
  console.log('session 2 frame', k, 'relocalised', st.relocalized, 'inliers', st.inliers, '/', st.matches,
    'yaw', st.T ? THREE.MathUtils.radToDeg(st.T.yaw).toFixed(1) : '-', '(40)', 't', st.T?.t.map(v => v.toFixed(2)), '(0.5 0 -0.3)', `${st.ms.toFixed(0)} ms`);
}
