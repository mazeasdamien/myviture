// Environment memory: a map of 3D landmarks (stereo depth) with binary descriptors,
// kept from one session to the next, and the relocalisation that finds where the
// current session's frame sits in it.
//
// Every session tracks in its own frame (the odometry's, gravity-aligned: y up). The map
// has its own frame, fixed when it was created. Matching the current landmarks to the
// map gives `T` (map -> session): a yaw about the vertical and a translation, since
// gravity already fixes the other two angles. Anchors are stored in the map frame and
// shown through T, so they come back where they were, and T's slow refinement during a
// session also cancels the odometry's drift around them.
//
// Descriptors: BRIEF (256 intensity comparisons in a 25x25 patch of the smoothed
// rectified image), steered by the gravity direction seen in the image, which keeps them
// valid when the head rolls. Rotation-steered BRIEF is the core of ORB (Rublee et al.,
// ICCV 2011); gravity replaces ORB's intensity-centroid angle here because the IMU gives
// it for free.

import * as THREE from 'three';
import type { DepthFrame } from '../depth/stereoDepth';
import { corners, type Gray } from './klt';

const WORDS = 8; // 256 bits
const CAPACITY = 12000;
const MATCH_MAX = 64; // Hamming distance
const RATIO = 0.8;
const MIN_INLIERS = 14;

/** map -> session: x_s = R_y(yaw) x_m + t */
export interface MapTransform {
  yaw: number;
  t: [number, number, number];
}

export interface MapData {
  pos: Float32Array;
  desc: Uint32Array;
  seen: Uint16Array;
}

export interface MapStatus {
  points: number;
  relocalized: boolean;
  /** inliers of the last alignment, matches tried */
  inliers: number;
  matches: number;
  T: MapTransform | null;
  ms: number;
}

// Sampling pattern: pairs of offsets, Gaussian (sigma 5 px) within the patch, fixed seed.
const PATTERN = (() => {
  let s = 0x2545f491;
  const rnd = () => ((s = (Math.imul(s ^ (s >>> 15), 0x2c1b3c6d) + 0x9e3779b9) | 0) >>> 0) / 4294967296;
  const g = () => {
    for (;;) {
      const v = Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd()) * 5;
      if (Math.abs(v) <= 12) return v;
    }
  };
  const p = new Float32Array(256 * 4);
  for (let i = 0; i < p.length; i++) p[i] = g();
  return p;
})();
const R_PATCH = 18; // keypoints this far from the border at least (rotated patch)

function popcount(v: number) {
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

function hamming(a: Uint32Array, ai: number, b: Uint32Array, bi: number) {
  let d = 0;
  for (let k = 0; k < WORDS; k++) d += popcount(a[ai + k] ^ b[bi + k]);
  return d;
}

/** 5x5 box blur: BRIEF compares single pixels, so noise must go first. */
function smooth(g: Gray): Uint8Array {
  const { w, h, px } = g;
  const tmp = new Uint16Array(w * h), out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    let s = 0;
    for (let x = -2; x <= 2; x++) s += px[y * w + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = s;
      s += px[y * w + Math.min(w - 1, x + 3)] - px[y * w + Math.max(0, x - 2)];
    }
  }
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = -2; y <= 2; y++) s += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = s / 25;
      s += tmp[Math.min(h - 1, y + 3) * w + x] - tmp[Math.max(0, y - 2) * w + x];
    }
  }
  return out;
}

/** Median depth of the 3x3 block, 0 on edges or without enough samples. */
function depthAt(d: DepthFrame, x: number, y: number) {
  const vals: number[] = [];
  for (let j = y - 1; j <= y + 1; j++) for (let i = x - 1; i <= x + 1; i++) {
    const z = d.depth[j * d.width + i];
    if (z > 0.3 && z < 5) vals.push(z);
  }
  if (vals.length < 6) return 0;
  vals.sort((a, b) => a - b);
  const m = vals[vals.length >> 1];
  return vals[vals.length - 1] - vals[0] < 0.06 * m ? m : 0;
}

export class EnvMap {
  private pos = new Float32Array(CAPACITY * 3);
  private desc = new Uint32Array(CAPACITY * WORDS);
  private seen = new Uint16Array(CAPACITY);
  private count = 0;
  T: MapTransform | null = null;
  private lastInsert: { p: THREE.Vector3; q: THREE.Quaternion } | null = null;
  /** set when the map is new in this session (nothing to relocalise against) */
  private fresh = true;
  status: MapStatus = { points: 0, relocalized: false, inliers: 0, matches: 0, T: null, ms: 0 };

  get size() { return this.count; }

  /** New session (reconnect): the session frame changed, T must be found again. */
  newSession() {
    this.T = null;
    this.lastInsert = null;
    this.fresh = this.count === 0;
    this.status = { ...this.status, relocalized: false, T: null };
  }

  clear() {
    this.count = 0;
    this.newSession();
    this.fresh = true;
  }

  load(m: MapData) {
    const n = Math.min(CAPACITY, m.seen.length);
    this.pos.set(m.pos.subarray(0, n * 3));
    this.desc.set(m.desc.subarray(0, n * WORDS));
    this.seen.set(m.seen.subarray(0, n));
    this.count = n;
    this.newSession();
  }

  data(): MapData {
    return { pos: this.pos.slice(0, this.count * 3), desc: this.desc.slice(0, this.count * WORDS), seen: this.seen.slice(0, this.count) };
  }

  /** Landmark positions in the session frame (for display), null before relocalisation. */
  sessionPoints(): Float32Array | null {
    if (!this.T) return null;
    const out = new Float32Array(this.count * 3);
    const c = Math.cos(this.T.yaw), s = Math.sin(this.T.yaw), [tx, ty, tz] = this.T.t;
    for (let i = 0; i < this.count; i++) {
      const x = this.pos[i * 3], y = this.pos[i * 3 + 1], z = this.pos[i * 3 + 2];
      out[i * 3] = c * x + s * z + tx;
      out[i * 3 + 1] = y + ty;
      out[i * 3 + 2] = -s * x + c * z + tz;
    }
    return out;
  }

  /**
   * One frame: rectified left image and its depth, head orientation and position in the
   * session frame at the image time.
   */
  process(img: Gray, depth: DepthFrame, q: THREE.Quaternion, p: THREE.Vector3): MapStatus {
    const t0 = performance.now();
    const feats = this.features(img, depth, q, p);
    let inliers = 0, matches = 0;
    let found: { feat: number; map: number }[] = [];

    if (this.fresh && !this.T) {
      // Nothing to find: this session's frame becomes the map frame.
      this.T = { yaw: 0, t: [0, 0, 0] };
    }
    if (this.count >= MIN_INLIERS && feats.n >= MIN_INLIERS) {
      const m = this.match(feats);
      found = m;
      matches = m.length;
      const fit = this.align(feats, m);
      if (fit) {
        inliers = fit.inliers.length;
        // First fit of the session: take it. Afterwards: drift correction, eased in.
        this.T = this.T && this.status.relocalized ? blend(this.T, fit.T, 0.3) : fit.T;
        this.status.relocalized = true;
        // Matched landmarks: refine their position (running mean in the map frame).
        for (const k of fit.inliers) this.refine(m[k].map, feats, m[k].feat);
      }
    }
    if (this.fresh) this.status.relocalized = true;

    // New landmarks only once the frame is placed in the map, and only after enough motion.
    if (this.T && this.status.relocalized && this.moved(p, q)) {
      const matched = new Set(found.map(mm => mm.feat));
      for (let i = 0; i < feats.n; i++) if (!matched.has(i)) this.insert(feats, i);
      this.lastInsert = { p: p.clone(), q: q.clone() };
    }
    this.status = { points: this.count, relocalized: this.status.relocalized, inliers, matches, T: this.T, ms: performance.now() - t0 };
    return this.status;
  }

  // ------------------------------------------------------------ features

  private features(img: Gray, depth: DepthFrame, q: THREE.Quaternion, p: THREE.Vector3) {
    const sm = smooth(img);
    const w = img.w, h = img.h;
    // Gravity's direction in the image steers the pattern.
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(q.clone().invert());
    const { ex, ey } = depth.axes;
    let gx = -(ex[0] * up.x + ex[1] * up.y + ex[2] * up.z), gy = -(ey[0] * up.x + ey[1] * up.y + ey[2] * up.z);
    const gl = Math.hypot(gx, gy) || 1;
    gx /= gl; gy /= gl;
    const pts = corners(img, 20, 15, (x, y) => x >= R_PATCH && y >= R_PATCH && x < w - R_PATCH && y < h - R_PATCH && depthAt(depth, x, y) > 0);
    const n = pts.length;
    const desc = new Uint32Array(n * WORDS);
    const S = new Float32Array(n * 3);
    const pt: [number, number, number] = [0, 0, 0];
    const v = new THREE.Vector3();
    for (let i = 0; i < n; i++) {
      const [x, y] = pts[i];
      for (let b = 0; b < 256; b++) {
        const ax = PATTERN[b * 4], ay = PATTERN[b * 4 + 1], bx = PATTERN[b * 4 + 2], by = PATTERN[b * 4 + 3];
        const ia = Math.round(y + ax * -gx + ay * gy) * w + Math.round(x + ax * gy + ay * gx);
        const ib = Math.round(y + bx * -gx + by * gy) * w + Math.round(x + bx * gy + by * gx);
        if (sm[ia] < sm[ib]) desc[i * WORDS + (b >> 5)] |= 1 << (b & 31);
      }
      depth.point(x, y, depthAt(depth, x, y), pt);
      v.set(pt[0], pt[1], pt[2]).applyQuaternion(q).add(p);
      S[i * 3] = v.x; S[i * 3 + 1] = v.y; S[i * 3 + 2] = v.z;
    }
    return { n, desc, S };
  }

  /** Best map match per feature (Hamming, ratio test). Near the head only once relocalised. */
  private match(f: { n: number; desc: Uint32Array; S: Float32Array }) {
    const out: { feat: number; map: number }[] = [];
    const near = this.T && this.status.relocalized ? this.sessionPoints() : null;
    for (let i = 0; i < f.n; i++) {
      let best = 999, second = 999, bj = -1;
      for (let j = 0; j < this.count; j++) {
        // Guided matching: a known pose only needs landmarks within 40 cm of the feature.
        if (near) {
          const dx = near[j * 3] - f.S[i * 3], dy = near[j * 3 + 1] - f.S[i * 3 + 1], dz = near[j * 3 + 2] - f.S[i * 3 + 2];
          if (dx * dx + dy * dy + dz * dz > 0.16) continue;
        }
        const d = hamming(f.desc, i * WORDS, this.desc, j * WORDS);
        if (d < best) { second = best; best = d; bj = j; } else if (d < second) second = d;
      }
      if (bj >= 0 && best <= MATCH_MAX && best < RATIO * second) out.push({ feat: i, map: bj });
    }
    return out;
  }

  /** 4-DoF RANSAC (yaw + translation) between map and session coordinates, then least squares. */
  private align(f: { S: Float32Array }, m: { feat: number; map: number }[]) {
    if (m.length < MIN_INLIERS) return null;
    const M = m.map(k => [this.pos[k.map * 3], this.pos[k.map * 3 + 1], this.pos[k.map * 3 + 2]]);
    const S = m.map(k => [f.S[k.feat * 3], f.S[k.feat * 3 + 1], f.S[k.feat * 3 + 2]]);
    let best: number[] = [];
    for (let it = 0; it < 300; it++) {
      const a = (Math.random() * m.length) | 0, b = (Math.random() * m.length) | 0;
      if (a === b) continue;
      const mx = M[b][0] - M[a][0], mz = M[b][2] - M[a][2], sx = S[b][0] - S[a][0], sz = S[b][2] - S[a][2];
      if (mx * mx + mz * mz < 0.01) continue;
      const T = fromYaw(Math.atan2(mx * sz - mz * sx, mx * sx + mz * sz) * -1, M[a], S[a]);
      const inl = inliersOf(T, M, S);
      if (inl.length > best.length) best = inl;
    }
    if (best.length < MIN_INLIERS || best.length < 0.3 * m.length) return null;
    const T = leastSquares(best.map(i => M[i]), best.map(i => S[i]));
    const inl = inliersOf(T, M, S);
    if (inl.length < MIN_INLIERS) return null;
    return { T: leastSquares(inl.map(i => M[i]), inl.map(i => S[i])), inliers: inl };
  }

  private refine(j: number, f: { S: Float32Array; desc: Uint32Array }, i: number) {
    if (!this.T) return;
    const m = toMap(this.T, f.S[i * 3], f.S[i * 3 + 1], f.S[i * 3 + 2]);
    const n = Math.min(this.seen[j], 20);
    for (let k = 0; k < 3; k++) this.pos[j * 3 + k] = (this.pos[j * 3 + k] * n + m[k]) / (n + 1);
    if (this.seen[j] < 65535) this.seen[j]++;
  }

  private insert(f: { S: Float32Array; desc: Uint32Array }, i: number) {
    if (!this.T) return;
    let j = this.count;
    if (j >= CAPACITY) {
      // Full: replace a landmark seen only once, starting from a random place.
      j = -1;
      for (let k = 0, s = (Math.random() * CAPACITY) | 0; k < 2000; k++) {
        const c = (s + k) % CAPACITY;
        if (this.seen[c] <= 1) { j = c; break; }
      }
      if (j < 0) return;
    } else this.count++;
    const m = toMap(this.T, f.S[i * 3], f.S[i * 3 + 1], f.S[i * 3 + 2]);
    this.pos.set(m, j * 3);
    this.desc.set(f.desc.subarray(i * WORDS, (i + 1) * WORDS), j * WORDS);
    this.seen[j] = 1;
  }

  private moved(p: THREE.Vector3, q: THREE.Quaternion) {
    if (!this.lastInsert) return true;
    return p.distanceTo(this.lastInsert.p) > 0.15 || q.angleTo(this.lastInsert.q) > THREE.MathUtils.degToRad(12);
  }
}

function fromYaw(yaw: number, m: number[], s: number[]): MapTransform {
  const c = Math.cos(yaw), sn = Math.sin(yaw);
  return { yaw, t: [s[0] - (c * m[0] + sn * m[2]), s[1] - m[1], s[2] - (-sn * m[0] + c * m[2])] };
}

function apply(T: MapTransform, m: number[]) {
  const c = Math.cos(T.yaw), s = Math.sin(T.yaw);
  return [c * m[0] + s * m[2] + T.t[0], m[1] + T.t[1], -s * m[0] + c * m[2] + T.t[2]];
}

function toMap(T: MapTransform, x: number, y: number, z: number): [number, number, number] {
  const c = Math.cos(T.yaw), s = Math.sin(T.yaw);
  const dx = x - T.t[0], dy = y - T.t[1], dz = z - T.t[2];
  return [c * dx - s * dz, dy, s * dx + c * dz];
}

function inliersOf(T: MapTransform, M: number[][], S: number[][]) {
  const out: number[] = [];
  for (let i = 0; i < M.length; i++) {
    const p = apply(T, M[i]);
    const e = Math.hypot(p[0] - S[i][0], p[1] - S[i][1], p[2] - S[i][2]);
    // Stereo depth error grows with distance.
    const dist = Math.hypot(S[i][0], S[i][2]);
    if (e < 0.04 + 0.03 * Math.min(dist, 4)) out.push(i);
  }
  return out;
}

/** Closed-form 4-DoF fit: centroids, then the yaw of the horizontal cross-covariance. */
function leastSquares(M: number[][], S: number[][]): MapTransform {
  const n = M.length;
  const cm = [0, 0, 0], cs = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) { cm[k] += M[i][k] / n; cs[k] += S[i][k] / n; }
  let sxx = 0, sxz = 0;
  for (let i = 0; i < n; i++) {
    const mx = M[i][0] - cm[0], mz = M[i][2] - cm[2], sx = S[i][0] - cs[0], sz = S[i][2] - cs[2];
    sxx += mx * sx + mz * sz;
    sxz += mz * sx - mx * sz;
  }
  return fromYaw(Math.atan2(sxz, sxx), cm, cs);
}

/** Moves T a fraction `k` towards U (drift correction without jumps). */
function blend(T: MapTransform, U: MapTransform, k: number): MapTransform {
  let d = U.yaw - T.yaw;
  d = Math.atan2(Math.sin(d), Math.cos(d));
  return { yaw: T.yaw + k * d, t: [0, 1, 2].map(i => T.t[i] + k * (U.t[i] - T.t[i])) as [number, number, number] };
}
