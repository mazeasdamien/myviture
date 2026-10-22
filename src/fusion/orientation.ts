// 3DoF head orientation from the raw IMU: calibration, Mahony complementary filter,
// stationary gyro-bias tracking, recentering and levelling.
//
// Frames: samples are rotated into the OpenGL head frame (x right, y up, z back)
// with the factory R_gl_imu, so the output quaternion drives a three.js camera directly.

import { Quaternion, Vector3 } from 'three';
import type { Calibration } from '../device/calibration';
import { DEFAULT_R_GL_IMU } from '../device/calibration';
import type { ImuSample } from '../device/protocol';

const KP = 1.2;
const KI = 0.02;
const STILL_GYRO = 0.02; // rad/s
const STILL_ACCEL = 0.15; // m/s² from 1 g
const BIAS_RATE = 0.002;
const G = 9.80665;

export class OrientationFilter {
  /** head -> world */
  readonly q = new Quaternion();
  readonly gyro = new Vector3();
  readonly accel = new Vector3();
  stationary = false;

  private R = DEFAULT_R_GL_IMU;
  private accelBias = [0, 0, 0];
  private accelScale = [1, 1, 1];
  private gyroBias = new Vector3();
  private integral = new Vector3();
  private lastT = -1;
  private initialised = false;
  private stillFor = 0;
  private yawRef = new Quaternion();
  private level = new Quaternion();

  setCalibration(c: Calibration) {
    this.R = c.R_gl_imu;
    this.accelBias = c.accelBias;
    this.accelScale = c.accelScale;
    this.gyroBias.copy(this.toGl(c.gyroBias));
  }

  private toGl(v: ArrayLike<number>, out = new Vector3()) {
    const R = this.R;
    return out.set(
      R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
      R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
      R[6] * v[0] + R[7] * v[1] + R[8] * v[2],
    );
  }

  private tmpA = [0, 0, 0];
  private up = new Vector3();
  private err = new Vector3();
  private dq = new Quaternion();

  update(s: ImuSample) {
    for (let i = 0; i < 3; i++) this.tmpA[i] = (s.accel[i] - this.accelBias[i]) * this.accelScale[i];
    this.toGl(this.tmpA, this.accel);
    this.toGl(s.gyro, this.gyro).sub(this.gyroBias);

    let dt = this.lastT < 0 ? 0.001 : s.t - this.lastT;
    this.lastT = s.t;
    if (!(dt > 0 && dt < 0.05)) dt = 0.001;

    const aNorm = this.accel.length();
    if (!this.initialised && aNorm > 1) {
      // Start level with gravity so the filter does not need to converge.
      this.q.setFromUnitVectors(this.accel.clone().normalize(), new Vector3(0, 1, 0));
      this.initialised = true;
    }

    // Track gyro bias while the glasses are still.
    const still = this.gyro.length() < STILL_GYRO && Math.abs(aNorm - G) < STILL_ACCEL;
    this.stillFor = still ? this.stillFor + dt : 0;
    this.stationary = this.stillFor > 0.5;
    if (this.stationary) this.gyroBias.addScaledVector(this.gyro, BIAS_RATE);

    const w = this.gyro.clone();
    if (aNorm > 0.5 * G && aNorm < 1.5 * G) {
      // Measured up (body) x predicted up (body).
      this.up.set(0, 1, 0).applyQuaternion(this.dq.copy(this.q).invert());
      this.err.copy(this.accel).divideScalar(aNorm).cross(this.up);
      this.integral.addScaledVector(this.err, KI * dt);
      w.addScaledVector(this.err, KP).add(this.integral);
    }
    const angle = w.length() * dt;
    if (angle > 0) {
      this.dq.setFromAxisAngle(w.normalize(), angle);
      this.q.multiply(this.dq).normalize();
    }
  }

  /** Orientation to render: yaw re-zeroed by recenter(), constant tilt removed by setLevel(). */
  display(out = new Quaternion()) {
    return out.copy(this.yawRef).multiply(this.q).multiply(this.level);
  }

  /** Head-frame angular velocity in the display frame's head axes (display() = ... * level). */
  displayRate(w: Vector3, out = new Vector3()) {
    return out.copy(w).applyQuaternion(new Quaternion().copy(this.level).invert());
  }

  /** display() for an orientation from the past (e.g. at an image timestamp). */
  displayOf(q: Quaternion, out = new Quaternion()) {
    return out.copy(this.yawRef).multiply(q).multiply(this.level);
  }

  /** Heading correction set by recenter(); world positions get the same rotation. */
  yaw(out = new Quaternion()) {
    return out.copy(this.yawRef);
  }

  /** Makes the current heading the forward direction. */
  recenter() {
    const fwd = new Vector3(0, 0, -1).applyQuaternion(this.display());
    const yaw = Math.atan2(-fwd.x, -fwd.z);
    this.yawRef.premultiply(new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), -yaw));
  }

  /** Treats the current head tilt as level (look straight ahead, then call). */
  setLevel() {
    const q = this.q;
    // Swing-twist: q = twist(yaw about world Y) * swing(tilt); keep the inverse swing.
    const twist = new Quaternion(0, q.y, 0, q.w).normalize();
    const swing = twist.clone().invert().multiply(q);
    this.level.copy(swing).invert();
    this.recenter();
  }

  resetLevel() {
    this.level.identity();
  }

  eulerDeg() {
    const fwd = new Vector3(0, 0, -1).applyQuaternion(this.display());
    const right = new Vector3(1, 0, 0).applyQuaternion(this.display());
    return {
      yaw: (Math.atan2(-fwd.x, -fwd.z) * 180) / Math.PI,
      pitch: (Math.asin(Math.max(-1, Math.min(1, fwd.y))) * 180) / Math.PI,
      roll: (Math.asin(Math.max(-1, Math.min(1, right.y))) * -180) / Math.PI,
    };
  }
}
