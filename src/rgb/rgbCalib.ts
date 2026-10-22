// RGB camera calibration from hand joints.
//
// The stereo hand tracker gives metric 3D joints; MediaPipe on the RGB frame gives
// the same joints in pixels. A few hundred such correspondences, taken while the
// hand is still at several places, fix the RGB camera's focal length, principal
// point and pose relative to the head (Levenberg-Marquardt on reprojection error).

import * as THREE from 'three';
import type { Vec3 } from '../hands/stereoGeometry';

/** Pinhole model of the RGB camera in the head frame. Camera axes: x right, y down, z forward. */
export interface RgbCamera {
  width: number;
  height: number;
  f: number;
  cx: number;
  cy: number;
  /** camera -> head rotation, quaternion [x, y, z, w] */
  q: [number, number, number, number];
  /** camera centre in the head frame, metres */
  t: Vec3;
  /** reprojection RMS of the fit, pixels */
  rms?: number;
}

export interface Correspondence {
  /** joint in the head frame at the RGB image time */
  P: Vec3;
  /** pixel in the RGB image */
  u: number;
  v: number;
}

/** Camera axes (x right, y down, z forward) to head axes (x right, y up, z back). */
const CAM_TO_HEAD = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);

/** Model used before any calibration: at the head origin, looking forward, given field. */
export function defaultRgbCamera(width: number, height: number, hfovDeg: number): RgbCamera {
  const f = width / 2 / Math.tan(THREE.MathUtils.degToRad(hfovDeg) / 2);
  const q = CAM_TO_HEAD;
  return { width, height, f, cx: width / 2, cy: height / 2, q: [q.x, q.y, q.z, q.w], t: [0, 0, 0] };
}

/** Head-frame point -> pixel (null when behind the camera). */
export function project(cam: RgbCamera, P: Vec3, qInv = new THREE.Quaternion(...cam.q).invert()): [number, number] | null {
  const p = new THREE.Vector3(P[0] - cam.t[0], P[1] - cam.t[1], P[2] - cam.t[2]).applyQuaternion(qInv);
  if (p.z < 0.05) return null;
  return [cam.f * p.x / p.z + cam.cx, cam.f * p.y / p.z + cam.cy];
}

/** Undistorted normalised ray (x, y, 1) of a pixel, in camera axes. */
export function pixelRay(cam: RgbCamera, u: number, v: number): [number, number] {
  return [(u - cam.cx) / cam.f, (v - cam.cy) / cam.f];
}

/** Camera-frame point -> head frame. */
export function cameraToHead(cam: RgbCamera, p: Vec3): Vec3 {
  const v = new THREE.Vector3(...p).applyQuaternion(new THREE.Quaternion(...cam.q));
  return [v.x + cam.t[0], v.y + cam.t[1], v.z + cam.t[2]];
}

// ------------------------------------------------------------------ fit

type Params = number[]; // [rx, ry, rz, tx, ty, tz, f, cx, cy]: rotation as a correction to the start

function toCamera(start: RgbCamera, x: Params): RgbCamera {
  const dr = new THREE.Vector3(x[0], x[1], x[2]);
  const angle = dr.length();
  const dq = angle > 1e-12 ? new THREE.Quaternion().setFromAxisAngle(dr.normalize(), angle) : new THREE.Quaternion();
  const q = new THREE.Quaternion(...start.q).multiply(dq);
  return { ...start, q: [q.x, q.y, q.z, q.w], t: [x[3], x[4], x[5]], f: x[6], cx: x[7], cy: x[8] };
}

// Hands cover a small depth range, so focal length, principal point, rotation and
// translation trade off against each other: a fit of the pixels alone can land 25 cm
// and 60 degrees away with a small error (seen 2026-10-22). Priors keep the camera
// near the glasses' bridge, looking forward, with its principal point near the centre.
const PRIOR = { rotRad: THREE.MathUtils.degToRad(4), tM: 0.02, centrePx: 0.03, fRel: 0.15 };

function residuals(start: RgbCamera, x: Params, data: Correspondence[], prior?: RgbCamera): number[] {
  const cam = toCamera(start, x);
  const qInv = new THREE.Quaternion(...cam.q).invert();
  const r: number[] = [];
  for (const c of data) {
    const p = project(cam, c.P, qInv);
    if (!p) { r.push(200, 200); continue; }
    r.push(p[0] - c.u, p[1] - c.v);
  }
  if (prior) {
    // Each prior counts like a fraction of the data, in pixel units (8 px per sigma).
    const k = 8 * Math.sqrt(data.length / 20);
    const dq = new THREE.Quaternion(...prior.q).invert().multiply(new THREE.Quaternion(...cam.q));
    const ang = 2 * Math.acos(Math.min(1, Math.abs(dq.w)));
    r.push(k * ang / PRIOR.rotRad);
    for (let i = 0; i < 3; i++) r.push(k * (cam.t[i] - prior.t[i]) / PRIOR.tM);
    r.push(k * (cam.cx - prior.cx) / (PRIOR.centrePx * cam.width), k * (cam.cy - prior.cy) / (PRIOR.centrePx * cam.width));
    r.push(k * (cam.f / prior.f - 1) / PRIOR.fRel);
  }
  return r;
}

/** False for a model no real Luma Ultra RGB camera can have (a degenerate fit). */
export function plausibleRgbCamera(cam: RgbCamera): boolean {
  const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(new THREE.Quaternion(...cam.q));
  const offAxis = THREE.MathUtils.radToDeg(fwd.angleTo(new THREE.Vector3(0, 0, -1)));
  return offAxis < 20 && Math.hypot(...cam.t) < 0.08
    && Math.abs(cam.cx - cam.width / 2) < 0.15 * cam.width && Math.abs(cam.cy - cam.height / 2) < 0.15 * cam.height;
}

/** Huber weights keep a few bad matches (wrong hand, motion) from pulling the fit. */
function huber(r: number[], k = 8) {
  return r.map(v => (Math.abs(v) <= k ? 1 : k / Math.abs(v)));
}

/** `prior`: the uncalibrated model the fit is pulled towards (null = pixels only). */
export function fitRgbCamera(start: RgbCamera, data: Correspondence[], prior: RgbCamera | null = start, iterations = 40): RgbCamera {
  const res = (xx: Params) => residuals(start, xx, data, prior ?? undefined);
  let x: Params = [0, 0, 0, start.t[0], start.t[1], start.t[2], start.f, start.cx, start.cy];
  let lambda = 1e-2;
  const steps = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 0.5, 0.5, 0.5];
  let r = res(x);
  let w = huber(r);
  let cost = r.reduce((s, v, i) => s + w[i] * v * v, 0);
  for (let it = 0; it < iterations; it++) {
    // Numeric Jacobian, then weighted normal equations (J^T W J + lambda diag) dx = -J^T W r.
    const J: number[][] = steps.map((h, k) => {
      const xp = x.slice(); xp[k] += h;
      const rp = res(xp);
      return rp.map((v, i) => (v - r[i]) / h);
    });
    const n = x.length;
    const A = Array.from({ length: n }, () => new Array(n).fill(0));
    const g = new Array(n).fill(0);
    for (let i = 0; i < r.length; i++) {
      for (let a = 0; a < n; a++) {
        g[a] += J[a][i] * w[i] * r[i];
        for (let b = 0; b < n; b++) A[a][b] += J[a][i] * w[i] * J[b][i];
      }
    }
    let improved = false;
    for (let tries = 0; tries < 8 && !improved; tries++) {
      const M = A.map((row, a) => row.map((v, b) => (a === b ? v * (1 + lambda) + 1e-9 : v)));
      const dx = solve(M, g.map(v => -v));
      if (!dx) { lambda *= 10; continue; }
      const xn = x.map((v, k) => v + dx[k]);
      const rn = res(xn);
      const wn = huber(rn);
      const cn = rn.reduce((s, v, i) => s + wn[i] * v * v, 0);
      if (cn < cost) { x = xn; r = rn; w = wn; cost = cn; lambda = Math.max(1e-6, lambda / 3); improved = true; }
      else lambda *= 10;
    }
    if (!improved) break;
  }
  const cam = toCamera(start, x);
  const final = residuals(start, x, data);  // pixels only
  cam.rms = Math.sqrt(final.reduce((s, v) => s + v * v, 0) / final.length) || 0;
  return cam;
}

/** Gaussian elimination with partial pivoting. */
function solve(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const k = M[r][c] / M[c][c];
      for (let j = c; j <= n; j++) M[r][j] -= k * M[c][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let j = r + 1; j < n; j++) s -= M[r][j] * x[j];
    x[r] = s / M[r][r];
  }
  return x;
}

export function loadRgbCamera(): RgbCamera | null {
  try {
    const raw = localStorage.getItem('rgbCamera');
    const cam = raw ? (JSON.parse(raw) as RgbCamera) : null;
    // A degenerate fit saved by an earlier version: forget it.
    if (cam && !plausibleRgbCamera(cam)) { localStorage.removeItem('rgbCamera'); return null; }
    return cam;
  } catch { return null; }
}

export function saveRgbCamera(cam: RgbCamera | null) {
  try {
    if (cam) localStorage.setItem('rgbCamera', JSON.stringify(cam));
    else localStorage.removeItem('rgbCamera');
  } catch { /* storage blocked */ }
}
