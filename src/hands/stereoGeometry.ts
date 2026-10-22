// Geometry of the two tracking cameras, from the factory calibration:
// FOV-model undistortion, camera -> head transform, ray triangulation.

import type { Calibration, CameraCalib } from '../device/calibration';

export type Vec3 = [number, number, number];

export class TrackingCamera {
  readonly fx: number; readonly fy: number; readonly cx: number; readonly cy: number;
  private w: number;
  private twoTanHalfW: number;
  /** head-frame rotation (row-major) and position of the camera */
  readonly R: number[];
  readonly origin: Vec3;

  constructor(cam: CameraCalib, R_gl_imu: number[]) {
    [this.fx, this.fy, this.cx, this.cy] = cam.intrinsics;
    this.w = cam.distortion[0] ?? 0;
    this.twoTanHalfW = 2 * Math.tan(this.w / 2);
    // T_cam_imu: p_cam = Rci p_imu + t  =>  p_imu = Rci^T (p_cam - t)
    const T = cam.T_cam_imu;
    const Rci = [T[0], T[1], T[2], T[4], T[5], T[6], T[8], T[9], T[10]];
    const t: Vec3 = [T[3], T[7], T[11]];
    const Ric = transpose(Rci);
    this.R = mul(R_gl_imu, Ric);
    const o = apply(Ric, t);
    this.origin = apply(R_gl_imu, [-o[0], -o[1], -o[2]]);
  }

  /** Pixel -> undistorted normalised image coordinates (pinhole, z = 1). */
  undistort(u: number, v: number): [number, number] {
    const xd = (u - this.cx) / this.fx, yd = (v - this.cy) / this.fy;
    const rd = Math.hypot(xd, yd);
    if (this.w < 1e-6 || rd < 1e-9) return [xd, yd];
    const ang = Math.min(rd * this.w, Math.PI / 2 - 1e-3);
    const s = Math.tan(ang) / this.twoTanHalfW / rd;
    return [xd * s, yd * s];
  }

  /** Undistorted normalised coordinates -> pixel (FOV model). */
  distort(x: number, y: number): [number, number] {
    const ru = Math.hypot(x, y);
    const s = this.w < 1e-6 || ru < 1e-9 ? 1 : Math.atan(ru * this.twoTanHalfW) / this.w / ru;
    return [this.fx * x * s + this.cx, this.fy * y * s + this.cy];
  }

  /** Camera-frame point (x right, y down, z forward) -> head frame. */
  toHead(p: Vec3): Vec3 {
    const r = apply(this.R, p);
    return [r[0] + this.origin[0], r[1] + this.origin[1], r[2] + this.origin[2]];
  }

  /** Unit ray direction in the head frame for a pixel. */
  ray(u: number, v: number): Vec3 {
    const [x, y] = this.undistort(u, v);
    const d = apply(this.R, [x, y, 1]);
    const n = Math.hypot(d[0], d[1], d[2]);
    return [d[0] / n, d[1] / n, d[2] / n];
  }
}

export function stereoRig(c: Calibration): [TrackingCamera, TrackingCamera] | null {
  if (c.cameras.length < 2 || c.cameras.some(cam => cam.intrinsics.length < 4 || cam.T_cam_imu.length < 16)) return null;
  return [new TrackingCamera(c.cameras[0], c.R_gl_imu), new TrackingCamera(c.cameras[1], c.R_gl_imu)];
}

/** Closest point between two rays; returns the midpoint, the gap and both ray parameters. */
export function triangulate(o1: Vec3, d1: Vec3, o2: Vec3, d2: Vec3) {
  const w0 = sub(o1, o2);
  const a = dot(d1, d1), b = dot(d1, d2), c = dot(d2, d2), d = dot(d1, w0), e = dot(d2, w0);
  const den = a * c - b * b;
  if (Math.abs(den) < 1e-9) return null;
  const s = (b * e - c * d) / den, t = (a * e - b * d) / den;
  const p1 = add(o1, scale(d1, s)), p2 = add(o2, scale(d2, t));
  return { point: scale(add(p1, p2), 0.5), gap: Math.hypot(...sub(p1, p2)), s, t };
}

const transpose = (m: number[]) => [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
const apply = (m: number[], v: Vec3): Vec3 => [
  m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
  m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
  m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
];
function mul(a: number[], b: number[]) {
  const r = new Array(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) r[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
  return r;
}
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
