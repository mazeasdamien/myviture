/// <reference lib="webworker" />
// Device worker: owns the WebUSB link so that heavy work on the page (rendering,
// hand tracking) never starves the 1 kHz IMU and camera transfers. Runs the IMU
// fusion (orientation + inertial position) and the tracking-camera auto-exposure,
// and hands stereo pairs to the vision worker.

import * as THREE from 'three';
import { LumaUltra } from '../device/lumaUltra';
import { VitureBridge, type ViturePose } from '../device/vitureBridge';
import { PID_LUMA_ULTRA, VID, type ImuSample, type StereoFrame } from '../device/protocol';
import { parseCalibration } from '../device/calibration';
import { OrientationFilter } from '../fusion/orientation';
import { OrientationHistory, PositionFilter } from '../vio/stereoOdometry';
import { autoExposureStep } from '../ui/autoExposure';
import type { FromDevice, FromVision, PoseState, Quat, ToDevice, ToVision, TrackerComparison, TrackerJitter, TrackingSource, V3 } from './messages';

const post = (m: FromDevice, transfer: Transferable[] = []) => (self as DedicatedWorkerGlobalScope).postMessage(m, transfer);

let glasses: LumaUltra | VitureBridge | null = null;
let source: TrackingSource = 'usb';
let vision: MessagePort | null = null;
let hands: MessagePort | null = null;
let filter = new OrientationFilter();
const position = new PositionFilter();
const history = new OrientationHistory();
const origin = new THREE.Vector3();
let sixDof = true;
let lastFix = -1;
let autoExposure = true;
let exposure = 2048;
let csv: string[] | null = null;

const counts = { imu: 0, cams: [0, 0] as [number, number], vsync: 0, temp: NaN };
let chart: number[] = [];
let decim = 0;

// ------------------------------------------------------------------ IMU

function onImu(s: ImuSample) {
  counts.imu++;
  counts.temp = s.tempC;
  filter.update(s);
  history.push(s.t, filter.q);
  position.predict(filter.accel, filter.q, s.t, filter.stationary);
  if (csv) csv.push(`${s.t.toFixed(4)},${s.gyro.join(',')},${s.accel.join(',')},${s.tempC.toFixed(2)}`);
  if (++decim % 5 === 0) chart.push(filter.gyro.x, filter.gyro.y, filter.gyro.z, filter.accel.x, filter.accel.y, filter.accel.z);
}

const tracking = () => sixDof && lastFix >= 0 && performance.now() - lastFix < 700;

// Displayed position: the filter's display output (corrections blended in, no steps).
const smoothed = new THREE.Vector3();

function displayPosition(out: THREE.Vector3): V3 | null {
  if (!tracking()) return null;
  out.copy(smoothed).sub(origin).applyQuaternion(filter.yaw());
  return [out.x, out.y, out.z];
}

/** Advances the position smoothing; called at the pose output rate. */
function smoothPosition() {
  smoothed.copy(position.display);
}

// ------------------------------------------------------------------ VITURE pose (bridge)
//
// The engine's pose, in its own frame (gravity-aligned, fixed for the session). Our
// recenter is applied on top (vYaw, vOrigin), like our own tracker's yawRef and origin.

let vPose: ViturePose | null = null;
const vYaw = new THREE.Quaternion();
const vOrigin = new THREE.Vector3();
/** recent VITURE poses (host ms) to find the pose at an image's time */
const vHistory: { t: number; q: THREE.Quaternion; p: THREE.Vector3 }[] = [];
const vm = new THREE.Matrix4();

function onViturePose(p: ViturePose) {
  vPose = p;
  vm.fromArray(p.m);
  const q = new THREE.Quaternion().setFromRotationMatrix(vm);
  const pos = new THREE.Vector3().setFromMatrixPosition(vm);
  vHistory.push({ t: p.hostMs, q, p: pos });
  if (vHistory.length > 1200) vHistory.splice(0, vHistory.length - 1200);
  jitter.viture.push(p.hostMs, q, pos, filter.stationary);
}

/** VITURE pose at absolute time t (ms), interpolated. */
function vitureAt(t: number): { q: THREE.Quaternion; p: THREE.Vector3 } | null {
  const h = vHistory;
  if (!h.length || t < h[0].t - 50) return null;
  let i = h.length - 1;
  while (i > 0 && h[i - 1].t > t) i--;
  const b = h[i], a = h[Math.max(0, i - 1)];
  const k = b.t > a.t ? Math.min(1, Math.max(0, (t - a.t) / (b.t - a.t))) : 1;
  return { q: a.q.clone().slerp(b.q, k), p: a.p.clone().lerp(b.p, k) };
}

function vitureRecenter() {
  const last = vHistory[vHistory.length - 1];
  if (!last) return;
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(last.q);
  vYaw.setFromAxisAngle(new THREE.Vector3(0, 1, 0), -Math.atan2(-fwd.x, -fwd.z));
  vOrigin.copy(last.p);
}

// ------------------------------------------------------------------ tracker comparison
//
// Both trackers run on the same IMU and images (the engine's, and ours fed with what the
// bridge relays): their jitter while the head is still says which one holds virtual
// content steadier.

class JitterMeter {
  // Tremor, not motion: each sample's deviation from the mean of its neighbours within
  // +-60 ms, RMS over the last 2 s. Smooth head motion averages out; shake does not.
  private s: { t: number; q: THREE.Quaternion; p: THREE.Vector3 }[] = [];
  value: TrackerJitter | null = null;
  private n = 0;
  push(t: number, q: THREE.Quaternion, p: THREE.Vector3 | null, _still: boolean) {
    if (!p) return;
    this.s.push({ t, q: q.clone(), p: p.clone() });
    while (this.s.length && this.s[0].t < t - 2000) this.s.shift();
    if (++this.n % 25 || this.s.length < 60) return;
    let pe = 0, re = 0, k = 0;
    const mp = new THREE.Vector3(), mq = new THREE.Quaternion();
    for (let i = 0; i < this.s.length; i++) {
      const c = this.s[i];
      if (c.t < this.s[0].t + 60 || c.t > t - 60) continue;
      mp.set(0, 0, 0);
      let qx = 0, qy = 0, qz = 0, qw = 0, m = 0;
      for (let j = i; j >= 0 && this.s[j].t >= c.t - 60; j--) acc(this.s[j]);
      for (let j = i + 1; j < this.s.length && this.s[j].t <= c.t + 60; j++) acc(this.s[j]);
      function acc(x: { q: THREE.Quaternion; p: THREE.Vector3 }) {
        mp.add(x.p);
        const sg = x.q.dot(c.q) < 0 ? -1 : 1;
        qx += sg * x.q.x; qy += sg * x.q.y; qz += sg * x.q.z; qw += sg * x.q.w; m++;
      }
      if (m < 5) continue;
      mp.divideScalar(m);
      mq.set(qx, qy, qz, qw).normalize();
      pe += c.p.distanceToSquared(mp);
      re += mq.angleTo(c.q) ** 2;
      k++;
    }
    if (k > 20) this.value = { posMm: Math.sqrt(pe / k) * 1000, rotDeg: THREE.MathUtils.radToDeg(Math.sqrt(re / k)) };
  }
}
const jitter = { viture: new JitterMeter(), ours: new JitterMeter() };

function comparison(): TrackerComparison | null {
  if (source !== 'viture' || !vHistory.length) return null;
  const up = new THREE.Vector3(0, 1, 0);
  const a = up.clone().applyQuaternion(vHistory[vHistory.length - 1].q.clone().invert());
  const b = up.clone().applyQuaternion(filter.q.clone().invert());
  return { viture: jitter.viture.value, ours: jitter.ours.value, tiltDiffDeg: THREE.MathUtils.radToDeg(a.angleTo(b)) };
}

const qd = new THREE.Quaternion();
const pd = new THREE.Vector3();
const wd = new THREE.Vector3();

function eulerOf(q: THREE.Quaternion) {
  const e = new THREE.Euler().setFromQuaternion(q, 'YXZ');
  const deg = THREE.MathUtils.radToDeg;
  return { yaw: deg(e.y), pitch: deg(e.x), roll: deg(e.z) };
}

function poseState(): PoseState {
  // Not extrapolated here: the page extrapolates to its own render time with `w`, so the
  // age of this message (timer and main-thread delays) does not show as judder.
  const gyro: V3 = [filter.gyro.x, filter.gyro.y, filter.gyro.z];
  const accel: V3 = [filter.accel.x, filter.accel.y, filter.accel.z];
  if (source === 'viture') {
    const last = vHistory[vHistory.length - 1];
    if (last) {
      qd.copy(vYaw).multiply(last.q);
      pd.copy(last.p).sub(vOrigin).applyQuaternion(vYaw);
    } else { qd.identity(); pd.set(0, 0, 0); }
    return {
      q: [qd.x, qd.y, qd.z, qd.w], w: gyro, stamp: vPose?.hostMs ?? absNow(),
      p: last ? [pd.x, pd.y, pd.z] : null, gyro, accel, stationary: filter.stationary,
      euler: eulerOf(qd), tracking: vPose?.rc === 0,
      frame: { yaw: [vYaw.x, vYaw.y, vYaw.z, vYaw.w], origin: [vOrigin.x, vOrigin.y, vOrigin.z] }, source,
    };
  }
  filter.display(qd);
  filter.displayRate(filter.gyro, wd);
  const yaw = filter.yaw();
  return {
    q: [qd.x, qd.y, qd.z, qd.w],
    w: [wd.x, wd.y, wd.z],
    stamp: absNow(),
    p: displayPosition(pd),
    gyro,
    accel,
    stationary: filter.stationary,
    euler: filter.eulerDeg(),
    tracking: tracking(),
    frame: { yaw: [yaw.x, yaw.y, yaw.z, yaw.w], origin: [origin.x, origin.y, origin.z] },
    source,
  };
}

// ------------------------------------------------------------------ stereo frames

const pending: (StereoFrame | null)[] = [null, null];
let visionBusy = false;
let brightness = 0;

function onFrame(f: StereoFrame) {
  counts.cams[f.camera]++;
  if (f.camera === 0) {
    let sum = 0;
    for (let i = 0; i < f.pixels.length; i += 16) sum += f.pixels[i];
    brightness = sum / (f.pixels.length / 16);
  }
  // Copies for the vision and hands workers; the original goes to the page.
  pending[f.camera] = { ...f, pixels: f.pixels.slice() };
  handsPending[f.camera] = { ...f, pixels: f.pixels.slice(), arrived: absNow() };
  sendHands();
  post({ type: 'frame', camera: f.camera, t: f.t, width: f.width, height: f.height, pixels: f.pixels }, [f.pixels.buffer]);
  sendPair();
}

const handsPending: ((StereoFrame & { arrived?: number }) | null)[] = [null, null];
const qHands = new THREE.Quaternion();
/** Absolute ms, comparable between the page and workers. */
const absNow = () => performance.timeOrigin + performance.now();
// The hands worker says when it is free: only the newest pair is ever sent, so it
// never works on a backlog of old images.
let handsBusy = false;
function sendHands() {
  const [l, r] = handsPending;
  if (!hands || handsBusy || !l || !r || Math.abs(l.t - r.t) > 0.02) return;
  // Head pose when the image was taken: hands are then placed in the world.
  let pose: { q: number[]; p: V3 | null; stamp: number };
  const v = source === 'viture' && glasses instanceof VitureBridge ? vitureAt(glasses.hostMs(l.t)) : null;
  if (v) {
    const q = vYaw.clone().multiply(v.q), p = v.p.clone().sub(vOrigin).applyQuaternion(vYaw);
    pose = { q: [q.x, q.y, q.z, q.w], p: [p.x, p.y, p.z], stamp: l.arrived ?? absNow() };
  } else {
    const q = history.at(l.t, qHands) ? filter.displayOf(qHands, qHands) : filter.display(qHands);
    pose = { q: [q.x, q.y, q.z, q.w], p: displayPosition(new THREE.Vector3()), stamp: l.arrived ?? absNow() };
  }
  hands.postMessage({ type: 'pair', left: l.pixels, right: r.pixels, w: l.width, h: l.height, pose }, [l.pixels.buffer, r.pixels.buffer]);
  handsPending[0] = handsPending[1] = null;
  handsBusy = true;
}

const qImg = new THREE.Quaternion();
/** Depth + odometry rate: ~7 Hz leaves the integrated GPU to hand tracking; the
 * temporal fusion keeps the depth dense between maps. */
const VISION_PERIOD = 0.14;
let lastPairT = -1;
function sendPair() {
  const [l, r] = pending;
  if (!vision || visionBusy || !l || !r || Math.abs(l.t - r.t) > 0.02) return;
  if (l.t - lastPairT < VISION_PERIOD - 0.005 && l.t > lastPairT) return;
  if (!history.at(l.t, qImg)) return;
  const v = source === 'viture' && glasses instanceof VitureBridge ? vitureAt(glasses.hostMs(l.t)) : null;
  const msg: ToVision = {
    type: 'pair', t: l.t, left: l.pixels, right: r.pixels,
    q: [qImg.x, qImg.y, qImg.z, qImg.w] as Quat,
    ...(v ? { ref: { q: [v.q.x, v.q.y, v.q.z, v.q.w] as Quat, p: [v.p.x, v.p.y, v.p.z] as V3 } } : {}),
  };
  vision.postMessage(msg, [l.pixels.buffer, r.pixels.buffer]);
  lastPairT = l.t;
  pending[0] = pending[1] = null;
  visionBusy = true;
}

function onVision(e: MessageEvent<FromVision | { type: 'idle' }>) {
  const m = e.data;
  if (m.type === 'idle') { visionBusy = false; sendPair(); return; }
  if (m.type === 'vo' && sixDof) {
    position.correct(new THREE.Vector3(...m.position), m.t, m.inliers);
    // First optical fix of the session: start the displayed position at zero.
    if (lastFix < 0) origin.copy(position.p);
    lastFix = performance.now();
  }
}

// ------------------------------------------------------------------ connection

async function connect(port: MessagePort, handsPort: MessagePort, src: TrackingSource) {
  source = src;
  hands?.close();
  hands = handsPort;
  handsBusy = false;
  hands.onmessage = () => { handsBusy = false; sendHands(); };
  vision?.close();
  vision = port;
  vision.onmessage = onVision;
  visionBusy = false;
  try {
    let g: LumaUltra | VitureBridge;
    let info;
    if (src === 'viture') {
      g = new VitureBridge();
      info = await g.open();
    } else {
      const dev = (await navigator.usb.getDevices()).find(d => d.vendorId === VID && d.productId === PID_LUMA_ULTRA);
      if (!dev) throw new Error('lunettes non autorisées ou débranchées');
      g = new LumaUltra(dev);
      info = await g.open();
    }
    filter = new OrientationFilter();
    position.reset();
    origin.set(0, 0, 0);
    lastFix = -1;
    vPose = null;
    vHistory.length = 0;
    vYaw.identity();
    vOrigin.set(0, 0, 0);
    let calibration = null;
    try {
      // The VIO engine does not expose the factory calibration: the dev server keeps a copy.
      const raw = g instanceof VitureBridge
        ? new Uint8Array(await (await fetch('/api/calibration')).arrayBuffer())
        : await g.readCalibration();
      calibration = parseCalibration(raw);
      filter.setCalibration(calibration);
      vision.postMessage({ type: 'calibration', calibration } satisfies ToVision);
      hands.postMessage({ type: 'calibration', calibration });
    } catch (e) {
      console.warn('calibration read failed', e);
    }
    g.events = {
      imu: onImu,
      frame: onFrame,
      vsync: () => counts.vsync++,
      pose: onViturePose,
      error: e => {
        glasses = null;
        post({ type: 'disconnected', error: String((e as Error)?.message ?? e) });
      },
    };
    await g.start();
    await g.setGain(64);
    await g.setExposure(exposure);
    glasses = g;
    post({ type: 'connected', info, calibration });
  } catch (e) {
    post({ type: 'disconnected', error: String((e as Error)?.message ?? e) });
  }
}

async function disconnect() {
  const g = glasses;
  glasses = null;
  await g?.close();
  post({ type: 'disconnected' });
}

self.onmessage = async (e: MessageEvent<ToDevice>) => {
  const m = e.data;
  switch (m.type) {
    case 'connect': await connect(m.visionPort, m.handsPort, m.source); break;
    case 'disconnect': await disconnect(); break;
    case 'exposure': autoExposure = false; exposure = m.value; await glasses?.setExposure(m.value); break;
    case 'gain': autoExposure = false; await glasses?.setGain(m.value); break;
    case 'autoExposure': autoExposure = m.on; break;
    case 'display': await glasses?.setDisplayMode(m.mode); break;
    case 'recenter': filter.recenter(); origin.copy(smoothed); vitureRecenter(); break;
    case 'level': filter.setLevel(); break;
    case 'sixDof': sixDof = m.on; break;
    case 'csv':
      if (m.on) csv = [];
      else if (csv) {
        post({ type: 'csv', text: 't_s,gx_rad_s,gy_rad_s,gz_rad_s,ax_m_s2,ay_m_s2,az_m_s2,temp_c\n' + csv.join('\n') });
        csv = null;
      }
      break;
  }
};

// ------------------------------------------------------------------ periodic output

setInterval(() => {
  if (!glasses) return;
  smoothPosition();
  const state = poseState();
  if (source === 'viture') {
    // Our own tracker, on the same data, for the comparison.
    filter.display(qd);
    jitter.ours.push(absNow(), qd, displayPosition(pd) ? pd : null, filter.stationary);
  }
  post({ type: 'pose', state });
  if (chart.length) {
    const samples = new Float32Array(chart);
    chart = [];
    post({ type: 'imu', samples }, [samples.buffer]);
  }
}, 8);

setInterval(() => {
  if (!glasses) return;
  post({ type: 'stats', imuHz: counts.imu, camFps: [...counts.cams] as [number, number], vsyncHz: counts.vsync, tempC: counts.temp, exposure, compare: comparison() });
  counts.imu = 0; counts.cams = [0, 0]; counts.vsync = 0;
}, 1000);

setInterval(() => {
  if (!glasses || !autoExposure) return;
  const next = autoExposureStep(brightness, exposure);
  if (next === null) return;
  exposure = next;
  glasses.setExposure(next);
}, 250);
