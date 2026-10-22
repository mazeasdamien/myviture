// Display (optical see-through) calibration: eye positions relative to the head
// frame and projection, with user fine-tuning persisted in localStorage.
//
// The Luma Ultra panels are 1920x1200 per eye; the factory frustum is about
// 43° x 25° and 28.147° is the vertical field used by the vendor's own renderer.
// The glasses carry no per-eye display extrinsics, so eyes start symmetric around
// the head origin and are adjusted with the anchoring test.

import * as THREE from 'three';

export const LUMA_ULTRA_VFOV = 28.147;
const DEFAULT_IPD = 0.063;

export interface DisplaySettings {
  /** vertical field of view, degrees */
  vfov: number;
  /** metres between the eyes */
  ipd: number;
  /** eye position relative to the head origin (IMU), metres, head frame */
  eyeOffset: [number, number, number];
  /** extra pitch of the displays, degrees (positive = looking up) */
  pitch: number;
}

export const DEFAULT_SETTINGS: DisplaySettings = { vfov: LUMA_ULTRA_VFOV, ipd: DEFAULT_IPD, eyeOffset: [0, 0, 0], pitch: 0 };

/** Eye poses in the head frame for the current settings. */
export function eyePoses(s: DisplaySettings) {
  const offset = new THREE.Vector3(...s.eyeOffset);
  return {
    left: new THREE.Vector3(-s.ipd / 2, 0, 0).add(offset),
    right: new THREE.Vector3(s.ipd / 2, 0, 0).add(offset),
    centre: offset.clone(),
    rotation: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), THREE.MathUtils.degToRad(s.pitch)),
  };
}

export function loadSettings(): DisplaySettings {
  try {
    const raw = localStorage.getItem('display');
    if (raw) return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch { /* storage blocked or bad JSON */ }
  return { ...DEFAULT_SETTINGS, eyeOffset: [...DEFAULT_SETTINGS.eyeOffset] };
}

export function saveSettings(s: DisplaySettings) {
  try { localStorage.setItem('display', JSON.stringify(s)); } catch { /* storage blocked */ }
}
