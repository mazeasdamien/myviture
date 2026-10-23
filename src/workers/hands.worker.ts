/// <reference lib="webworker" />
// Hands worker: MediaPipe Hand Landmarker off the page's main thread. Stereo pairs
// come straight from the device worker (MessagePort), RGB frames from the page as
// ImageBitmaps. Only the latest image is processed; older ones are dropped.

import { RgbHandTracker, StereoHandTracker, landmarkerDelegate, type Hand } from '../hands/handTracker';
import { stereoRig, type TrackingCamera } from '../hands/stereoGeometry';
import type { Calibration } from '../device/calibration';
import { defaultRgbCamera, type RgbCamera } from '../rgb/rgbCalib';
import * as THREE from 'three';
import { headMatrix } from '../scene/headPose';
import type { Vec3 } from '../hands/stereoGeometry';

/** Head pose at image time, so hands are placed in the world, not carried by the head. */
export interface ImagePose { q: [number, number, number, number]; p: [number, number, number] | null; stamp: number }

function worldTransform(pose: ImagePose) {
  const m = headMatrix(new THREE.Quaternion(...pose.q), pose.p ? new THREE.Vector3(...pose.p) : null);
  const v = new THREE.Vector3();
  return (j: Vec3): Vec3 => { v.set(j[0], j[1], j[2]).applyMatrix4(m); return [v.x, v.y, v.z]; };
}

// MediaPipe loads its wasm glue with importScripts(), which module workers lack.
(self as unknown as { importScripts: (...urls: string[]) => void }).importScripts = (...urls: string[]) => {
  for (const url of urls) {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', url, false);
    xhr.send();
    (0, eval)(xhr.responseText);
  }
};

export type ToHands =
  | { type: 'port'; port: MessagePort }
  | { type: 'options'; enabled: boolean; source: 'stereo' | 'rgb'; rgbCam: RgbCamera | null; hfov: number; calibrating: boolean }
  | { type: 'rgb'; bitmap: ImageBitmap; pose: ImagePose };

export type FromHands =
  | { type: 'hands'; hands: Hand[]; rate: number; source: 'stereo' | 'rgb'; ms?: { waited: number; convert: number; detect: number; posted: number } }
  | { type: 'status'; text: string }
  | { type: 'idle' };

const post = (m: FromHands) => (self as DedicatedWorkerGlobalScope).postMessage(m);

const stereoHands = new StereoHandTracker();
const rgbHands = new RgbHandTracker();
let rig: [TrackingCamera, TrackingCamera] | null = null;
let device: MessagePort | null = null;
const options = { enabled: true, source: 'stereo' as 'stereo' | 'rgb', hfov: 78, rgbCam: null as RgbCamera | null, calibrating: false };
let loading: Promise<void> | null = null;

let nextStereo: { left: Uint8Array; right: Uint8Array; w: number; h: number; pose: ImagePose } | null = null;
let nextRgb: { bitmap: ImageBitmap; pose: ImagePose } | null = null;
let scheduled = false;

const images = [new ImageData(640, 480), new ImageData(640, 480)];
function toImage(i: number, px: Uint8Array) {
  const out = images[i].data;
  for (let k = 0, j = 0; k < px.length; k++, j += 4) {
    out[j] = out[j + 1] = out[j + 2] = px[k];
    out[j + 3] = 255;
  }
  return images[i];
}

function ensureLoaded() {
  loading ??= (async () => {
    post({ type: 'status', text: 'chargement du modèle…' });
    await stereoHands.init();
    await rgbHands.init();
    post({ type: 'status', text: `prêt (${landmarkerDelegate})` });
  })().catch(e => {
    post({ type: 'status', text: `modèle indisponible : ${(e as Error).message}` });
    loading = null;
  });
}

function schedule() {
  if (scheduled) return;
  scheduled = true;
  setTimeout(run, 0);
}

function run() {
  scheduled = false;
  if (!options.enabled) { nextRgb?.bitmap.close(); nextRgb = null; nextStereo = null; return; }
  const wantStereo = options.source === 'stereo' || options.calibrating;
  const wantRgb = options.source === 'rgb' || options.calibrating;
  if (wantStereo && nextStereo && (!stereoHands.ready || !rig)) {
    // Cannot use it yet: drop it and ask for a fresh one later.
    nextStereo = null;
    device?.postMessage('ready');
    ensureLoaded();
    return;
  }
  if (wantStereo && nextStereo && rig) {
    const s = nextStereo;
    nextStereo = null;
    const t0 = performance.now();
    // Time the pair spent between its arrival and here (absolute clocks).
    const waited = performance.timeOrigin + t0 - s.pose.stamp;
    const li = toImage(0, s.left), ri = toImage(1, s.right);
    const t1 = performance.now();
    const hands = stereoHands.detect(li, ri, { w: s.w, h: s.h }, rig, worldTransform(s.pose));
    const t2 = performance.now();
    for (const h of hands) h.stamp = s.pose.stamp;
    post({ type: 'hands', hands, rate: stereoHands.rate.value, source: 'stereo', ms: { waited, convert: t1 - t0, detect: t2 - t1, posted: performance.timeOrigin + performance.now() } });
    device?.postMessage('ready');
  }
  if (wantRgb && nextRgb) {
    if (!rgbHands.ready) { ensureLoaded(); nextRgb.bitmap.close(); nextRgb = null; post({ type: 'idle' }); return; }
    const { bitmap: b, pose } = nextRgb;
    nextRgb = null;
    const cam = options.rgbCam ?? defaultRgbCamera(b.width, b.height, options.hfov);
    const hands = rgbHands.detect(b, b.width, b.height, cam, worldTransform(pose));
    for (const h of hands) h.stamp = pose.stamp;
    b.close();
    post({ type: 'hands', hands, rate: rgbHands.rate.value, source: 'rgb' });
    post({ type: 'idle' });
  }
}

function onDevice(e: MessageEvent<{ type: 'calibration'; calibration: Calibration } | { type: 'pair'; left: Uint8Array; right: Uint8Array; w: number; h: number; pose: ImagePose }>) {
  const m = e.data;
  if (m.type === 'calibration') { rig = stereoRig(m.calibration); return; }
  nextStereo = m;
  if (options.enabled && (options.source === 'stereo' || options.calibrating) && stereoHands.ready && rig) schedule();
  else device?.postMessage('ready'); // not used: let the device worker keep sending
}

self.onmessage = (e: MessageEvent<ToHands>) => {
  const m = e.data;
  if (m.type === 'port') {
    device = m.port;
    m.port.onmessage = onDevice;
  } else if (m.type === 'options') {
    Object.assign(options, { enabled: m.enabled, source: m.source, hfov: m.hfov, rgbCam: m.rgbCam, calibrating: m.calibrating });
    if (m.enabled) ensureLoaded();
  } else if (m.type === 'rgb') {
    nextRgb?.bitmap.close();
    nextRgb = { bitmap: m.bitmap, pose: m.pose };
    schedule();
  }
};
