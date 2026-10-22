// Messages between the page and the two workers.
//
//   page <-> device worker : WebUSB driver, IMU fusion (orientation + position), exposure
//   page <-> vision worker : stereo depth and odometry results for display
//   device worker -> vision worker (MessagePort): stereo pairs with the head pose
//   vision worker -> device worker (MessagePort): odometry fixes

import type { Calibration } from '../device/calibration';
import type { DeviceInfo } from '../device/lumaUltra';
import type { RectAxes } from '../depth/stereoDepth';
import type { MapData, MapStatus } from '../vio/envMap';

export type Quat = [number, number, number, number]; // x, y, z, w
export type V3 = [number, number, number];

// ------------------------------------------------------------------ page -> device worker

/** 'viture': the VITURE VIO engine through bridge/viture_bridge.exe; 'usb': our own driver and tracking. */
export type TrackingSource = 'viture' | 'usb';

export type ToDevice =
  | { type: 'connect'; source: TrackingSource; visionPort: MessagePort; handsPort: MessagePort }
  | { type: 'disconnect' }
  | { type: 'exposure'; value: number }
  | { type: 'gain'; value: number }
  | { type: 'autoExposure'; on: boolean }
  | { type: 'display'; mode: number }
  | { type: 'recenter' }
  | { type: 'level' }
  | { type: 'sixDof'; on: boolean }
  | { type: 'csv'; on: boolean };

// ------------------------------------------------------------------ device worker -> page

export interface PoseState {
  /** head -> world, recentred and levelled, at `stamp` */
  q: Quat;
  /** angular velocity (rad/s) about the display head axes, to extrapolate q to render time */
  w: V3;
  /** absolute ms (performance.timeOrigin + now) of q */
  stamp: number;
  /** recentred world position (m), null in 3DoF */
  p: V3 | null;
  gyro: V3;
  accel: V3;
  stationary: boolean;
  euler: { yaw: number; pitch: number; roll: number };
  tracking: boolean;
  /**
   * Tracking frame -> scene: scene = yaw * (x - origin) + eye height. The tracking frame
   * (VITURE's, or our odometry's) does not move on recenter; the environment map and its
   * anchors live in it.
   */
  frame: { yaw: Quat; origin: V3 };
  source: TrackingSource;
}

/** Jitter of each tracker while the head is still: RMS around the 1.5 s mean. */
export interface TrackerJitter {
  posMm: number;
  rotDeg: number;
}
export interface TrackerComparison {
  viture: TrackerJitter | null;
  ours: TrackerJitter | null;
  /** angle between the two trackers' gravity directions, degrees (they should agree) */
  tiltDiffDeg: number;
}

export type FromDevice =
  | { type: 'connected'; info: DeviceInfo; calibration: Calibration | null }
  | { type: 'disconnected'; error?: string }
  | { type: 'pose'; state: PoseState }
  /** decimated gyro/accel (head frame) for the charts: [gx, gy, gz, ax, ay, az] * n */
  | { type: 'imu'; samples: Float32Array }
  | { type: 'frame'; camera: 0 | 1; t: number; width: number; height: number; pixels: Uint8Array }
  | { type: 'stats'; imuHz: number; camFps: [number, number]; vsyncHz: number; tempC: number; exposure: number; compare: TrackerComparison | null }
  | { type: 'csv'; text: string };

// ------------------------------------------------------------------ device worker -> vision worker

export type ToVision =
  | { type: 'calibration'; calibration: Calibration }
  | {
    type: 'pair';
    t: number;
    left: Uint8Array;
    right: Uint8Array;
    /** raw filter orientation at image time (odometry frame) */
    q: Quat;
    /** reference pose at image time in the tracking frame (VITURE's), when not our own */
    ref?: { q: Quat; p: V3 };
  };

export interface VisionStats {
  depthMs: number;
  voMs: number;
  maps: number;
  inliers: number;
  tracked: number;
}

export type FromVision =
  | { type: 'vo'; t: number; position: V3; inliers: number; tracked: number }
  | { type: 'depth'; width: number; height: number; depth: Float32Array; left: Uint8Array; validRatio: number; axes: RectAxes; stats: VisionStats }
  | { type: 'error'; message: string }
  /** environment memory: status after each processed frame */
  | { type: 'map'; status: MapStatus }
  /** landmarks in the tracking frame, for display */
  | { type: 'mapPoints'; points: Float32Array }
  | { type: 'mapData'; data: MapData };

export type ToVisionFromPage =
  | { type: 'port'; port: MessagePort }
  | { type: 'options'; depth: boolean; sixDof: boolean }
  | { type: 'mapLoad'; data: MapData }
  | { type: 'mapSave' }
  | { type: 'mapClear' };
