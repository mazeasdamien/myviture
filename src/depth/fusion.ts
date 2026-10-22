// Temporal depth fusion: each new stereo map is merged with the previous fused map,
// re-projected into the current view with the head motion (IMU rotation + odometry
// position). Static surfaces accumulate into a dense, stable map; moving things are
// replaced by the new measurement. Then short gaps are bridged along rows and columns
// where both ends agree (never across depth edges).

import * as THREE from 'three';
import { DEPTH_H, DEPTH_W, rectPoint, type DepthFrame, type RectAxes } from './stereoDepth';
import type { Vec3 } from '../hands/stereoGeometry';

const MAX_WEIGHT = 8;
const DECAY = 0.8;
const AGREE = 0.06; // relative depth difference treated as the same surface
const MAX_GAP = 6; // px bridged by hole filling

export class DepthFusion {
  private F = new Float32Array(DEPTH_W * DEPTH_H);
  private W = new Float32Array(DEPTH_W * DEPTH_H);
  private warpedF = new Float32Array(DEPTH_W * DEPTH_H);
  private warpedW = new Float32Array(DEPTH_W * DEPTH_H);
  private prev: { q: THREE.Quaternion; p: THREE.Vector3 } | null = null;

  reset() {
    this.F.fill(0);
    this.W.fill(0);
    this.prev = null;
  }

  /** q: head -> world at image time; p: head position (null = rotation only). Returns the fused depth. */
  update(d: DepthFrame, q: THREE.Quaternion, p: THREE.Vector3 | null): Float32Array {
    const pos = p ?? this.prev?.p ?? new THREE.Vector3();
    if (this.prev) this.warp(d.axes, this.prev.q, this.prev.p, q, pos);
    else { this.warpedF.fill(0); this.warpedW.fill(0); }

    const F = this.F, W = this.W, wf = this.warpedF, ww = this.warpedW, D = d.depth;
    for (let i = 0; i < F.length; i++) {
      const z = D[i], zw = wf[i];
      if (z > 0 && zw > 0) {
        if (Math.abs(z - zw) < AGREE * z) {
          const w = ww[i];
          F[i] = (zw * w + z) / (w + 1);
          W[i] = Math.min(w + 1, MAX_WEIGHT);
        } else {
          F[i] = z; W[i] = 1; // the scene changed here (moving hand, new object)
        }
      } else if (z > 0) {
        F[i] = z; W[i] = 1;
      } else if (zw > 0 && ww[i] * DECAY >= 0.5) {
        F[i] = zw; W[i] = ww[i] * DECAY;
      } else {
        F[i] = 0; W[i] = 0;
      }
    }
    this.prev = { q: q.clone(), p: pos.clone() };

    const out = F.slice();
    fillGaps(out, DEPTH_W, DEPTH_H);
    return out;
  }

  /** Forward-projects the fused map from the previous view into the current one (z-buffered). */
  private warp(a: RectAxes, qPrev: THREE.Quaternion, pPrev: THREE.Vector3, qCur: THREE.Quaternion, pCur: THREE.Vector3) {
    const wf = this.warpedF, ww = this.warpedW;
    wf.fill(0);
    ww.fill(0);
    // head_cur = Rcur^T (Rprev head_prev + pPrev - pCur)
    const M = new THREE.Matrix4().makeRotationFromQuaternion(qCur.clone().invert().multiply(qPrev));
    const tr = pPrev.clone().sub(pCur).applyQuaternion(qCur.clone().invert());
    const m = M.elements; // column-major
    const pt: Vec3 = [0, 0, 0];
    const { ex, ey, ez, origin, f } = a;
    for (let v = 0; v < DEPTH_H; v++) {
      for (let u = 0; u < DEPTH_W; u++) {
        const i = v * DEPTH_W + u;
        const z0 = this.F[i];
        if (!(z0 > 0)) continue;
        rectPoint(a, u, v, z0, pt);
        const hx = m[0] * pt[0] + m[4] * pt[1] + m[8] * pt[2] + tr.x - origin[0];
        const hy = m[1] * pt[0] + m[5] * pt[1] + m[9] * pt[2] + tr.y - origin[1];
        const hz = m[2] * pt[0] + m[6] * pt[1] + m[10] * pt[2] + tr.z - origin[2];
        const z = ez[0] * hx + ez[1] * hy + ez[2] * hz;
        if (z < 0.15) continue;
        const x = ex[0] * hx + ex[1] * hy + ex[2] * hz, y = ey[0] * hx + ey[1] * hy + ey[2] * hz;
        const u2 = Math.round((f * x) / z + DEPTH_W / 2 - 0.5), v2 = Math.round((f * y) / z + DEPTH_H / 2 - 0.5);
        if (u2 < 0 || v2 < 0 || u2 >= DEPTH_W || v2 >= DEPTH_H) continue;
        const j = v2 * DEPTH_W + u2;
        if (wf[j] === 0 || z < wf[j]) { wf[j] = z; ww[j] = this.W[i]; }
      }
    }
    // Forward splatting leaves one-pixel cracks: fill them from the nearest neighbours.
    for (let v = 1; v < DEPTH_H - 1; v++) {
      for (let u = 1; u < DEPTH_W - 1; u++) {
        const i = v * DEPTH_W + u;
        if (wf[i] > 0) continue;
        let sum = 0, n = 0, w = 0;
        for (const j of [i - 1, i + 1, i - DEPTH_W, i + DEPTH_W]) if (wf[j] > 0) { sum += wf[j]; w += ww[j]; n++; }
        if (n >= 3) { wf[i] = sum / n; ww[i] = (w / n) * 0.8; }
      }
    }
  }
}

/** Bridges gaps of up to MAX_GAP pixels along rows then columns when both ends agree. */
function fillGaps(d: Float32Array, w: number, h: number) {
  const pass = (count: number, len: number, idx: (a: number, b: number) => number) => {
    for (let a = 0; a < count; a++) {
      let last = -1;
      for (let b = 0; b < len; b++) {
        const z = d[idx(a, b)];
        if (!(z > 0)) continue;
        if (last >= 0 && b - last > 1 && b - last <= MAX_GAP + 1) {
          const z0 = d[idx(a, last)];
          if (Math.abs(z - z0) < AGREE * Math.min(z, z0)) {
            for (let k = last + 1; k < b; k++) d[idx(a, k)] = z0 + ((z - z0) * (k - last)) / (b - last);
          }
        }
        last = b;
      }
    }
  };
  pass(h, w, (y, x) => y * w + x);
  pass(w, h, (x, y) => y * w + x);
}
