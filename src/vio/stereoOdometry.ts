// Stereo visual odometry with an IMU rotation prior, plus a position filter.
//
// Keyframe scheme: corners with a valid stereo depth are picked on a keyframe and
// tracked (KLT, forward-backward checked) in every following rectified left image.
// With the head orientations from the IMU filter at both image times, each tracked
// point votes for the head position:
//     p_cur = p_kf + R_kf * P_kf - R_cur * P_cur      (P = point in the head frame)
// The votes are combined robustly. Tracking against the keyframe (rather than the
// previous image) keeps the position from drifting while the scene stays in view.

import * as THREE from 'three';
import type { DepthFrame } from '../depth/stereoDepth';
import type { Vec3 } from '../hands/stereoGeometry';
import { corners, pyramid, trackPoint, type Gray, type Pyramid } from './klt';

const LEVELS = 3;
const MIN_TRACKS = 25;
const INLIER = 0.04; // m

interface Keyframe {
  pyr: Pyramid;
  pts: [number, number][];
  /** head-frame 3D point of each keypoint at the keyframe */
  P: Vec3[];
  R: THREE.Quaternion;
  pos: THREE.Vector3;
}

export interface OdometryResult {
  /** world position of the head at the image time, metres */
  position: THREE.Vector3;
  inliers: number;
  tracked: number;
  newKeyframe: boolean;
}

export class StereoOdometry {
  private kf: Keyframe | null = null;
  private last: [number, number][] = [];
  private lastPyr: Pyramid | null = null;
  readonly position = new THREE.Vector3();
  lost = true;

  reset(pos = new THREE.Vector3()) {
    this.kf = null;
    this.position.copy(pos);
    this.lost = true;
  }

  /**
   * img: rectified left image of the depth frame; q: head orientation (head -> world)
   * at the image time.
   */
  update(img: Gray, depth: DepthFrame, q: THREE.Quaternion): OdometryResult | null {
    const pyr = pyramid(img, LEVELS);
    let result: OdometryResult | null = null;

    if (this.kf && this.lastPyr) {
      const kf = this.kf;
      const votes: THREE.Vector3[] = [];
      const keep: number[] = [];
      const cur: [number, number][] = [];
      const pt: Vec3 = [0, 0, 0];
      const Rkf = kf.R, Rcur = q;
      for (let i = 0; i < kf.pts.length; i++) {
        const [lx, ly] = this.last[i];
        if (lx < 0) { cur.push([-1, -1]); continue; }
        const fwd = trackPoint(this.lastPyr, pyr, lx, ly);
        const back = fwd && trackPoint(pyr, this.lastPyr, fwd[0], fwd[1]);
        if (!fwd || !back || Math.hypot(back[0] - lx, back[1] - ly) > 0.8) { cur.push([-1, -1]); continue; }
        cur.push(fwd);
        const z = depthAt(depth, fwd[0], fwd[1]);
        if (!z) continue;
        depth.point(fwd[0], fwd[1], z, pt);
        const a = new THREE.Vector3(...kf.P[i]).applyQuaternion(Rkf);
        const b = new THREE.Vector3(...pt).applyQuaternion(Rcur);
        votes.push(kf.pos.clone().add(a).sub(b));
        keep.push(i);
      }
      this.last = cur;
      const tracked = cur.filter(c => c[0] >= 0).length;
      const est = robustMean(votes);
      if (est && est.inliers >= MIN_TRACKS / 2) {
        this.position.copy(est.mean);
        this.lost = false;
        result = { position: this.position.clone(), inliers: est.inliers, tracked, newKeyframe: false };
      } else {
        this.lost = true;
      }
      // New keyframe when too few points survive.
      if (tracked < Math.max(MIN_TRACKS, kf.pts.length * 0.5)) this.kf = null;
    }

    if (!this.kf) {
      const accept = (x: number, y: number) => depthAt(depth, x, y) > 0;
      const pts = corners(img, 16, 12, accept);
      const P: Vec3[] = [];
      const good: [number, number][] = [];
      for (const [x, y] of pts) {
        const z = depthAt(depth, x, y);
        if (!z) continue;
        P.push(depth.point(x, y, z, [0, 0, 0]));
        good.push([x, y]);
      }
      if (good.length >= MIN_TRACKS) {
        this.kf = { pyr, pts: good, P, R: q.clone(), pos: this.position.clone() };
        this.last = good.map(p => [p[0], p[1]] as [number, number]);
        if (result) result.newKeyframe = true;
      }
    }
    this.lastPyr = pyr;
    return result;
  }
}

/** Median depth of the 3x3 block around (x, y), 0 if not enough valid samples. */
function depthAt(d: DepthFrame, x: number, y: number): number {
  const cx = Math.round(x), cy = Math.round(y);
  const vals: number[] = [];
  for (let j = cy - 1; j <= cy + 1; j++) for (let i = cx - 1; i <= cx + 1; i++) {
    if (i < 0 || j < 0 || i >= d.width || j >= d.height) continue;
    const z = d.depth[j * d.width + i];
    if (z > 0.25 && z < 5) vals.push(z);
  }
  if (vals.length < 6) return 0;
  vals.sort((a, b) => a - b);
  const m = vals[vals.length >> 1];
  // Reject depth discontinuities (corner on an object edge).
  return vals[vals.length - 1] - vals[0] < 0.1 * m ? m : 0;
}

/** Component-wise median, then the mean of votes within INLIER of it. */
function robustMean(v: THREE.Vector3[]) {
  if (v.length < 5) return null;
  const med = (k: 'x' | 'y' | 'z') => v.map(p => p[k]).sort((a, b) => a - b)[v.length >> 1];
  const m = new THREE.Vector3(med('x'), med('y'), med('z'));
  const inl = v.filter(p => p.distanceTo(m) < INLIER);
  if (!inl.length) return null;
  const mean = inl.reduce((a, p) => a.add(p), new THREE.Vector3()).divideScalar(inl.length);
  return { mean, inliers: inl.length };
}

/**
 * Head position at IMU rate: a Kalman filter on position, velocity and accelerometer
 * bias (one 3x3 covariance shared by the three world axes), driven by the world
 * acceleration and corrected by the delayed odometry fixes, each compared with the
 * filter's own position at the image time. A still head adds a zero-velocity measurement.
 *
 * `display` is what to render: corrections are not shown as steps but folded in over
 * ~0.15 s (an offset that decays), so a fix every 150 ms does not make the world shake,
 * while the IMU's own motion shows at once.
 */
const ACCEL_NOISE = 0.2; // m/s^2, white acceleration (sensor noise + orientation error)
const BIAS_WALK = 0.02; // m/s^2 per sqrt(s)
const ZUPT_SIGMA = 0.01; // m/s
const BLEND_MOVING = 0.15; // s
const BLEND_STILL = 0.35; // s

type M3 = number[]; // row-major 3x3

export class PositionFilter {
  readonly p = new THREE.Vector3();
  readonly v = new THREE.Vector3();
  /** accelerometer bias, world frame */
  readonly b = new THREE.Vector3();
  /** position to render: p plus a decaying share of the recent corrections */
  readonly display = new THREE.Vector3();
  private offset = new THREE.Vector3();
  private P: M3 = [1, 0, 0, 0, 1, 0, 0, 0, 0.01];
  private history: { t: number; p: THREE.Vector3 }[] = [];
  private lastT = -1;
  private lastFixT = -1;
  private rejected = 0;
  private stationary = false;
  private tmp = new THREE.Vector3();

  /** accel: calibrated specific force in the head frame; q: head -> world; t: device time (s) */
  predict(accel: THREE.Vector3, q: THREE.Quaternion, t: number, stationary: boolean) {
    const dt = this.lastT < 0 ? 0 : t - this.lastT;
    this.lastT = t;
    if (!(dt > 0 && dt < 0.05)) return;
    const a = this.tmp.copy(accel).applyQuaternion(q);
    a.y -= 9.80665;
    if (a.length() > 30) a.set(0, 0, 0);
    a.sub(this.b);
    this.stationary = stationary;
    this.p.addScaledVector(this.v, dt).addScaledVector(a, 0.5 * dt * dt);
    this.v.addScaledVector(a, dt);
    // Without optical fixes for a while, bound the inertial drift.
    if (this.lastFixT < 0 || t - this.lastFixT > 0.4) this.v.multiplyScalar(0.995);

    // P = F P F^T + Q, F = [[1, dt, -dt^2/2], [0, 1, -dt], [0, 0, 1]]
    const F = [1, dt, -0.5 * dt * dt, 0, 1, -dt, 0, 0, 1];
    const qa = ACCEL_NOISE * ACCEL_NOISE, qb = BIAS_WALK * BIAS_WALK;
    const Pn = mul(mul(F, this.P), transpose(F));
    Pn[0] += (qa * dt * dt * dt) / 3; Pn[1] += (qa * dt * dt) / 2; Pn[3] += (qa * dt * dt) / 2;
    Pn[4] += qa * dt; Pn[8] += qb * dt;
    this.P = Pn;
    if (stationary) this.update(1, this.tmp.copy(this.v).negate(), ZUPT_SIGMA, 0);

    const tau = stationary ? BLEND_STILL : BLEND_MOVING;
    this.offset.multiplyScalar(Math.exp(-dt / tau));
    this.display.copy(this.p).add(this.offset);
    this.history.push({ t, p: this.p.clone() });
    if (this.history.length > 1500) this.history.splice(0, this.history.length - 1500);
  }

  /**
   * Scalar-per-axis measurement of state `k` (0 position, 1 velocity) with innovation
   * `innov`, taken `lag` seconds ago: the correction is propagated to now.
   */
  private update(k: 0 | 1, innov: THREE.Vector3, sigma: number, lag: number) {
    const P = this.P;
    const S = P[k * 3 + k] + sigma * sigma;
    const K = [P[k] / S, P[3 + k] / S, P[6 + k] / S];
    // Correction at the measurement time, carried forward with the motion model.
    const dp = innov.clone().multiplyScalar(K[0] + K[1] * lag - 0.5 * K[2] * lag * lag);
    this.p.add(dp);
    this.offset.sub(dp);
    this.v.addScaledVector(innov, K[1] - K[2] * lag);
    this.b.addScaledVector(innov, K[2]);
    const row = [P[k * 3], P[k * 3 + 1], P[k * 3 + 2]];
    this.P = P.map((v, i) => v - K[Math.floor(i / 3)] * row[i % 3]);
    return K[0];
  }

  /**
   * Optical position measured at device time t. inliers: points agreeing on it; the
   * fewer, the less it is trusted.
   */
  correct(measured: THREE.Vector3, t: number, inliers = 30) {
    const past = this.at(t) ?? this.p;
    const innov = measured.clone().sub(past);
    const n = innov.length();
    if (n > 0.5) {
      // Large jump (re-initialisation): snap.
      this.p.add(innov);
      this.display.copy(this.p);
      this.offset.set(0, 0, 0);
      this.v.set(0, 0, 0);
      for (const h of this.history) h.p.add(innov);
      this.lastFixT = t;
      return;
    }
    const sigma = (this.stationary ? 0.012 : 0.006) + 0.04 / Math.sqrt(Math.max(inliers, 1));
    // Outlier gate (4 sigma per axis); several in a row mean the filter is the one off.
    if (n * n > 16 * 3 * (this.P[0] + sigma * sigma) && ++this.rejected < 4) return;
    this.rejected = 0;
    const kp = this.update(0, innov, sigma, Math.max(0, this.lastT - t));
    for (const h of this.history) if (h.t >= t) h.p.addScaledVector(innov, kp);
    this.lastFixT = t;
  }

  private at(t: number) {
    const h = this.history;
    if (!h.length || t < h[0].t) return null;
    let lo = 0, hi = h.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (h[mid].t < t) lo = mid + 1; else hi = mid; }
    return h[lo].p;
  }

  reset() {
    this.p.set(0, 0, 0);
    this.v.set(0, 0, 0);
    this.b.set(0, 0, 0);
    this.display.set(0, 0, 0);
    this.offset.set(0, 0, 0);
    this.P = [1, 0, 0, 0, 1, 0, 0, 0, 0.01];
    this.history = [];
    this.lastFixT = -1;
    this.rejected = 0;
  }
}

function mul(A: M3, B: M3): M3 {
  const C = new Array(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
  return C;
}

function transpose(A: M3): M3 {
  return [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
}

/** Orientation history at IMU rate, to look up the head orientation at image time. */
export class OrientationHistory {
  private t: Float64Array;
  private q: Float32Array;
  private n = 0;
  private head = 0;
  constructor(private size = 2048) {
    this.t = new Float64Array(size);
    this.q = new Float32Array(size * 4);
  }
  push(t: number, q: THREE.Quaternion) {
    this.t[this.head] = t;
    this.q.set([q.x, q.y, q.z, q.w], this.head * 4);
    this.head = (this.head + 1) % this.size;
    this.n = Math.min(this.n + 1, this.size);
  }
  at(t: number, out: THREE.Quaternion): THREE.Quaternion | null {
    let best = -1, bestDt = Infinity;
    for (let k = 0; k < this.n; k++) {
      const i = (this.head - 1 - k + this.size) % this.size;
      const dt = Math.abs(this.t[i] - t);
      if (dt < bestDt) { bestDt = dt; best = i; }
      if (this.t[i] < t - 0.01) break;
    }
    if (best < 0 || bestDt > 0.05) return null;
    return out.set(this.q[best * 4], this.q[best * 4 + 1], this.q[best * 4 + 2], this.q[best * 4 + 3]);
  }
}
