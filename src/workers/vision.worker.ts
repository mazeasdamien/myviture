/// <reference lib="webworker" />
// Vision worker: stereo depth (WebGL2 on an OffscreenCanvas) and visual odometry.
// Pairs come from the device worker with the head orientation at image time;
// odometry fixes go back to it, depth maps go to the page.

import * as THREE from 'three';
import { StereoDepth } from '../depth/stereoDepth';
import { stereoRig } from '../hands/stereoGeometry';
import { StereoOdometry } from '../vio/stereoOdometry';
import { DepthFusion } from '../depth/fusion';
import { EnvMap } from '../vio/envMap';
import { DEPTH_H, DEPTH_W } from '../depth/stereoDepth';
import type { FromVision, ToVision, ToVisionFromPage, VisionStats } from './messages';

const page = (m: FromVision, transfer: Transferable[] = []) => (self as DedicatedWorkerGlobalScope).postMessage(m, transfer);

let device: MessagePort | null = null;
let depth: StereoDepth | null = null;
let failed = false;
const odometry = new StereoOdometry();
const fusion = new DepthFusion();
const options = { depth: true, sixDof: true };
const stats: VisionStats = { depthMs: 0, voMs: 0, maps: 0, inliers: 0, tracked: 0 };
let maps = 0;
const envMap = new EnvMap();
let mapFrames = 0;

async function onPair(m: Extract<ToVision, { type: 'pair' }>) {
  if (!depth) return;
  const t0 = performance.now();
  const d = await depth.compute(m.left, m.right);
  stats.depthMs = performance.now() - t0;
  maps++;

  const q = new THREE.Quaternion(...m.q);
  if (options.sixDof) {
    const t1 = performance.now();
    const res = odometry.update({ w: DEPTH_W, h: DEPTH_H, px: d.left }, d, q);
    stats.voMs = performance.now() - t1;
    if (res) {
      stats.inliers = res.inliers;
      stats.tracked = res.tracked;
      device?.postMessage({ type: 'vo', t: m.t, position: res.position.toArray(), inliers: res.inliers, tracked: res.tracked } satisfies FromVision);
    }
  }
  // Environment memory, in the tracking frame: VITURE's when it is the reference, else
  // our odometry's. Every third map (~2 Hz) is plenty to recognise the room.
  const sq = m.ref ? new THREE.Quaternion(...m.ref.q) : q;
  const sp = m.ref ? new THREE.Vector3(...m.ref.p) : options.sixDof && !odometry.lost ? odometry.position : null;
  if (sp && ++mapFrames % 3 === 0) {
    const status = envMap.process({ w: DEPTH_W, h: DEPTH_H, px: d.left }, d, sq, sp);
    page({ type: 'map', status });
    if (mapFrames % 15 === 0) {
      const points = envMap.sessionPoints();
      if (points) page({ type: 'mapPoints', points }, [points.buffer]);
    }
  }

  if (options.depth) {
    // Dense map for display and occlusion: fused over time with the head motion.
    // The odometry keeps using the raw measurements.
    // The reference tracker's pose (VITURE's) when there is one, else our odometry.
    const fused = m.ref
      ? fusion.update(d, new THREE.Quaternion(...m.ref.q), new THREE.Vector3(...m.ref.p))
      : fusion.update(d, q, options.sixDof && !odometry.lost ? odometry.position : null);
    let valid = 0;
    for (let i = 0; i < fused.length; i++) if (fused[i] > 0) valid++;
    // The odometry keeps the rectified image for tracking: send the page a copy.
    const left = d.left.slice();
    page({ type: 'depth', width: d.width, height: d.height, depth: fused, left, validRatio: valid / fused.length, axes: d.axes, stats: { ...stats } }, [fused.buffer, left.buffer]);
  }
}

async function onDevice(e: MessageEvent<ToVision>) {
  const m = e.data;
  if (m.type === 'calibration') {
    const rig = stereoRig(m.calibration);
    depth = null;
    failed = false;
    odometry.reset();
    fusion.reset();
    envMap.newSession();
    if (rig) {
      try { depth = new StereoDepth(rig); } catch (err) { failed = true; page({ type: 'error', message: (err as Error).message }); }
    }
    return;
  }
  try {
    if (!failed) await onPair(m);
  } finally {
    device?.postMessage({ type: 'idle' });
  }
}

self.onmessage = (e: MessageEvent<ToVisionFromPage>) => {
  const m = e.data;
  if (m.type === 'port') {
    device?.close();
    device = m.port;
    device.onmessage = onDevice;
  } else if (m.type === 'mapLoad') {
    envMap.load(m.data);
  } else if (m.type === 'mapSave') {
    page({ type: 'mapData', data: envMap.data() });
  } else if (m.type === 'mapClear') {
    envMap.clear();
  } else if (m.type === 'options') {
    if (m.sixDof && !options.sixDof) odometry.reset();
    Object.assign(options, { depth: m.depth, sixDof: m.sixDof });
  }
};

setInterval(() => { stats.maps = maps; maps = 0; }, 1000);
