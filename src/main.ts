// Page: dashboard, hand tracking and the immersive scene. USB and IMU fusion run in
// the device worker, stereo depth and odometry in the vision worker, so the
// rendering never starves the glasses' data streams.

import './style.css';
import * as THREE from 'three';
import { USB_FILTERS } from './device/lumaUltra';
import { PID_LUMA_ULTRA, VID, type StereoFrame } from './device/protocol';
import type { Calibration } from './device/calibration';
import { DisplayMode } from './device/mcu';
import { StripChart } from './ui/stripChart';
import { StereoView } from './ui/stereoView';
import { Recorder, download, findGlassesCamera, listCameras, openCamera, snapshot } from './rgb/rgbCamera';
import { Preview } from './scene/preview';
import { EYE_HEIGHT, XrView, XrWorld, headMatrix, type EyeSetup } from './scene/immersive';
import { HUMAN_ANIMS } from './scene/human';
import { HandPredictor, absNow } from './hands/prediction';
import { defaultRgbCamera, fitRgbCamera, loadRgbCamera, plausibleRgbCamera, saveRgbCamera, type Correspondence, type RgbCamera } from './rgb/rgbCalib';
import { DEFAULT_SETTINGS, eyePoses, loadSettings, saveSettings } from './display/displayCalib';
import { glassesCanvas, openGlassesWindow } from './display/glassesScreen';
import { XR_CHANNEL, XR_CONTROL_CHANNEL, type XrFeed, type XrRay } from './webxr/runtime';
import { DEPTH_H, DEPTH_W, depthFrameFrom, type DepthFrame } from './depth/stereoDepth';
import type { Hand } from './hands/handTracker';
import type { FromHands, ToHands } from './workers/hands.worker';
import { TrackingCamera, stereoRig } from './hands/stereoGeometry';
import { drawHands } from './hands/handOverlay';
import type { FromDevice, FromVision, PoseState, ToDevice, ToVisionFromPage, TrackerJitter, TrackingSource } from './workers/messages';
import { MODELS, chat, costs, detect, transcribe, visionModels, type Detection, type ModelInfo } from './assistant/orange';
import { Speaker, VoiceInput, type VoiceMode } from './assistant/voice';
import { hitDepth, locateRgbPixel } from './assistant/locate';
import type { MapStatus, MapTransform } from './vio/envMap';
import { clearMap, loadMap, saveMap, type Anchor } from './vio/mapStore';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const setText = (id: string, v: string) => { $(id).textContent = v; };
const chip = (id: string, state: 'on' | 'off' | 'wait' | 'err') => { $(id).dataset.state = state; };
const checked = (id: string) => ($(id) as HTMLInputElement).checked;
const fmt3 = (v: ArrayLike<number>, d = 3) => Array.from(v, x => (x >= 0 ? ' ' : '') + x.toFixed(d)).join(' ');
const store = {
  get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* storage blocked */ } },
};

function banner(msg: string | null) {
  const b = $('banner');
  b.hidden = !msg;
  b.textContent = msg ?? '';
  b.title = msg ?? '';
}

// ---------------------------------------------------------------- workers

const deviceWorker = new Worker(new URL('./workers/device.worker.ts', import.meta.url), { type: 'module' });
const visionWorker = new Worker(new URL('./workers/vision.worker.ts', import.meta.url), { type: 'module' });
const toDevice = (m: ToDevice, transfer: Transferable[] = []) => deviceWorker.postMessage(m, transfer);
const toVision = (m: ToVisionFromPage, transfer: Transferable[] = []) => visionWorker.postMessage(m, transfer);

// ---------------------------------------------------------------- state

let connected = false;
let rig: [TrackingCamera, TrackingCamera] | null = null;
let pose: PoseState | null = null;
let depthFrame: DepthFrame | null = null;
let paused = false;

// ---------------------------------------------------------------- views

const preview = new Preview($('preview') as HTMLCanvasElement);
const axes = [{ label: 'x', color: '#ff5d6c' }, { label: 'y', color: '#2fd39a' }, { label: 'z', color: '#4c8dff' }];
const gyroChart = new StripChart($('chart-gyro') as HTMLCanvasElement, axes, 1000);
const accelChart = new StripChart($('chart-accel') as HTMLCanvasElement, axes, 1000);
const stereo = new StereoView([$('cam-left') as HTMLCanvasElement, $('cam-right') as HTMLCanvasElement]);
const video = $('rgb') as HTMLVideoElement;

/** Head prediction for rendering (display latency), ms; set by the "Prédiction" slider. */
let predictMs = 30;
const dq = new THREE.Quaternion();
const wAxis = new THREE.Vector3();

/**
 * Head orientation extrapolated with the gyro to now + `leadMs`. Extrapolating from the
 * pose's own timestamp, at render time, removes the judder of a pose that is 0 to 15 ms
 * old depending on when its message got through.
 */
function displayOrientation(out: THREE.Quaternion, leadMs = predictMs) {
  if (!pose) return out.identity();
  out.set(...pose.q);
  const dt = Math.min(0.1, Math.max(0, (absNow() - pose.stamp + leadMs) / 1000));
  const w = wAxis.set(...pose.w).length();
  if (w > 1e-6) out.multiply(dq.setFromAxisAngle(wAxis.divideScalar(w), w * dt));
  return out;
}

function displayPosition(out: THREE.Vector3): THREE.Vector3 | null {
  return pose?.p ? out.set(...pose.p) : null;
}

function recenter() {
  toDevice({ type: 'recenter' });
  preview.clearTrail();
}

// ---------------------------------------------------------------- device worker messages

deviceWorker.onmessage = (e: MessageEvent<FromDevice>) => {
  const m = e.data;
  switch (m.type) {
    case 'connected':
      connected = true;
      setText('d-product', m.info.product);
      setText('d-serial', m.info.serial);
      setText('d-fw', m.info.firmware);
      if (m.calibration) setCalibration(m.calibration);
      chip('chip-usb', 'on');
      setConnectedUi(true);
      store.set('autoConnect', '1');
      banner(null);
      restoreMap();
      if (!rgbStream) startRgb();
      break;
    case 'disconnected':
      connected = false;
      pose = null;
      chip('chip-usb', m.error ? 'err' : 'off');
      chip('chip-6dof', 'off');
      setConnectedUi(false);
      if (m.error) {
        banner(/claim|Access denied/i.test(m.error)
          ? 'Les lunettes sont déjà utilisées (autre onglet, SpaceWalker ou service VITURE XR). Ferme-les puis reconnecte.'
          : `Lunettes déconnectées : ${m.error}`);
      }
      break;
    case 'pose':
      pose = m.state;
      // Also lets a webxr.html opened by hand (any screen, any tab) follow the glasses.
      feedXr(m.state);
      break;
    case 'imu':
      if (paused) break;
      for (let i = 0; i + 5 < m.samples.length; i += 6) {
        gyroChart.push(m.samples.subarray(i, i + 3));
        accelChart.push(m.samples.subarray(i + 3, i + 6));
      }
      break;
    case 'frame': {
      const f: StereoFrame = { camera: m.camera, t: m.t, width: m.width, height: m.height, pixels: m.pixels };
      if (!paused) stereo.push(f);
      break;
    }
    case 'stats': {
      const j = (v: TrackerJitter | null) => (v ? `${v.posMm.toFixed(2)} mm · ${v.rotDeg.toFixed(3)}°` : 'mesure…');
      setText('v-cmp-v', m.compare ? j(m.compare.viture) : '-');
      setText('v-cmp-o', m.compare ? `${j(m.compare.ours)}${m.compare.tiltDiffDeg > 3 ? ` (écart ${m.compare.tiltDiffDeg.toFixed(0)}°)` : ''}` : '-');
      setText('d-imu-rate', `${m.imuHz} Hz`);
      setText('d-cam-rate', `${m.camFps[0]} / ${m.camFps[1]} i/s`);
      setText('d-vsync', `${m.vsyncHz} Hz`);
      if (!Number.isNaN(m.tempC)) setText('d-temp', `${m.tempC.toFixed(1)} °C`);
      if (checked('chk-ae')) { expInput.value = String(m.exposure); setText('o-exp', String(m.exposure)); }
      break;
    }
    case 'csv':
      download(new Blob([m.text], { type: 'text/csv' }), `imu-${stamp()}.csv`);
      break;
  }
};

// ---------------------------------------------------------------- vision worker messages

const depthCtx = ($('depth-map') as HTMLCanvasElement).getContext('2d')!;
const depthImage = new ImageData(DEPTH_W, DEPTH_H);
let visionInfo = '';

visionWorker.onmessage = (e: MessageEvent<FromVision>) => {
  const m = e.data;
  if (m.type === 'error') { visionInfo = `profondeur indisponible : ${m.message}`; return; }
  if (m.type === 'map' || m.type === 'mapPoints' || m.type === 'mapData') { onMapMessage(m); return; }
  if (m.type !== 'depth') return;
  depthFrame = depthFrameFrom(m);
  if (checked('chk-depth')) drawDepthMap(depthFrame);
  visionInfo = `${m.stats.maps} cartes/s · stéréo ${m.stats.depthMs.toFixed(0)} ms · odométrie ${m.stats.voMs.toFixed(0)} ms · ${(m.validRatio * 100).toFixed(0)} % mesurés`;
};

function drawDepthMap(d: DepthFrame) {
  const px = depthImage.data;
  for (let i = 0; i < d.depth.length; i++) {
    const z = d.depth[i];
    const j = i * 4;
    if (!(z > 0)) { px[j] = px[j + 1] = px[j + 2] = 0; px[j + 3] = 255; continue; }
    // Near = warm, far = cold, 0.2 m .. 4 m.
    const t = Math.min(1, Math.max(0, (Math.log(z) - Math.log(0.2)) / (Math.log(4) - Math.log(0.2))));
    px[j] = 255 * (1 - t) ** 0.8;
    px[j + 1] = 255 * Math.sin(Math.PI * Math.min(1, t * 1.1)) * 0.85;
    px[j + 2] = 255 * t;
    px[j + 3] = 255;
  }
  depthCtx.putImageData(depthImage, 0, 0);
}

function syncVisionOptions() {
  toVision({ type: 'options', depth: checked('chk-depth'), sixDof: checked('chk-6dof') });
  toDevice({ type: 'sixDof', on: checked('chk-6dof') });
}

// ---------------------------------------------------------------- connection (WebUSB, in the device worker)

function setConnectedUi(on: boolean) {
  $('btn-connect').textContent = on ? 'Déconnecter' : 'Connecter les lunettes';
  $('btn-connect').classList.toggle('primary', !on);
}

const sourceSel = $('sel-source') as HTMLSelectElement;
sourceSel.value = store.get('source') ?? (location.hostname === 'localhost' ? 'viture' : 'usb');
const source = () => sourceSel.value as TrackingSource;

async function bridge(action: 'start' | 'stop'): Promise<{ ok: boolean; error?: string }> {
  try {
    return await (await fetch(`/api/bridge/${action}`, { method: 'POST' })).json();
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
}

async function connect() {
  banner(null);
  if (source() === 'viture') {
    // The VITURE engine (bridge process) owns the glasses' USB link.
    chip('chip-usb', 'wait');
    $('btn-connect').textContent = 'Démarrage du suivi VITURE…';
    const res = await bridge('start');
    if (!res.ok) {
      chip('chip-usb', 'err');
      setConnectedUi(false);
      banner(`Suivi VITURE indisponible : ${res.error}. Ferme SpaceWalker si elle est ouverte, ou passe en « Suivi maison ».`);
      return;
    }
    openWorkers();
    return;
  }
  await bridge('stop');
  if (!('usb' in navigator)) {
    chip('chip-usb', 'err');
    banner('WebUSB n’est pas disponible dans ce navigateur. Ouvre la page dans Chrome ou Edge (http://localhost).');
    return;
  }
  chip('chip-usb', 'wait');
  // The permission prompt needs the page; the worker then finds the device itself.
  const known = (await navigator.usb.getDevices()).some(d => d.vendorId === VID && d.productId === PID_LUMA_ULTRA);
  if (!known) {
    try { await navigator.usb.requestDevice({ filters: USB_FILTERS }); } catch { chip('chip-usb', 'off'); return; }
  }
  openWorkers();
}

function openWorkers() {
  const vision = new MessageChannel(), handsCh = new MessageChannel();
  toVision({ type: 'port', port: vision.port2 }, [vision.port2]);
  toHands({ type: 'port', port: handsCh.port2 }, [handsCh.port2]);
  syncVisionOptions();
  syncHandOptions();
  toDevice({ type: 'connect', source: source(), visionPort: vision.port1, handsPort: handsCh.port1 }, [vision.port1, handsCh.port1]);
}

function disconnect() {
  store.set('autoConnect', '0');
  toDevice({ type: 'disconnect' });
  // Frees the glasses for other applications.
  if (source() === 'viture') setTimeout(() => bridge('stop'), 500);
}

sourceSel.onchange = async () => {
  store.set('source', sourceSel.value);
  if (!connected) return;
  toDevice({ type: 'disconnect' });
  await bridge('stop');
  setTimeout(connect, 800);
};

function setCalibration(c: Calibration) {
  rig = stereoRig(c);
  setText('d-gbias', fmt3(c.gyroBias, 4));
  c.cameras.forEach((cam, i) => {
    const [fx, fy, cx, cy] = cam.intrinsics;
    setText(`d-cam${i}`, `f ${fx?.toFixed(0)}/${fy?.toFixed(0)} c ${cx?.toFixed(0)},${cy?.toFixed(0)} ${cam.distortionModel} ${cam.distortion.map(d => d.toFixed(3)).join(' ')}`);
  });
}

// ---------------------------------------------------------------- RGB camera

let rgbStream: MediaStream | null = null;
let rgbRetry = 0;
const recorder = new Recorder();

async function startRgb(deviceId?: string) {
  try {
    chip('chip-rgb', 'wait');
    rgbStream?.getTracks().forEach(t => t.stop());
    rgbStream = null;
    if (!deviceId) {
      // Labels are only exposed once camera permission is granted.
      let cams = await listCameras();
      if (!cams.some(c => c.label)) {
        (await openCamera()).getTracks().forEach(t => t.stop());
        cams = await listCameras();
      }
      const glassesCam = findGlassesCamera(cams);
      fillCameraList(cams, glassesCam?.deviceId);
      if (!glassesCam) {
        // No fallback to the laptop webcam: hand tracking would use the wrong camera.
        chip('chip-rgb', 'err');
        setText('rgb-info', 'caméra des lunettes introuvable : débranche puis rebranche les lunettes');
        scheduleRgbRetry();
        return;
      }
      deviceId = glassesCam.deviceId;
    }
    const stream = await openCamera(deviceId);
    rgbStream = stream;
    video.srcObject = stream;
    const track = stream.getVideoTracks()[0];
    const s = track.getSettings();
    setText('rgb-info', `${track.label} · ${s.width}×${s.height} @ ${Math.round(s.frameRate ?? 0)} i/s`);
    chip('chip-rgb', 'on');
    rgbRetry = 0;
    // The camera can be taken by another page or re-enumerated: try to get it back.
    track.onended = () => { if (rgbStream === stream) { rgbStream = null; scheduleRgbRetry(); } };
  } catch (e) {
    chip('chip-rgb', 'err');
    const name = (e as DOMException).name;
    setText('rgb-info', name === 'NotReadableError' ? 'caméra occupée par une autre page ou application' : `indisponible : ${(e as Error).message}`);
    scheduleRgbRetry();
  }
}

function scheduleRgbRetry() {
  if (++rgbRetry > 40) return;
  setTimeout(() => { if (!rgbStream) startRgb(); }, 3000);
}

function fillCameraList(cams: MediaDeviceInfo[], selected?: string) {
  const sel = $('sel-rgb') as HTMLSelectElement;
  sel.replaceChildren(...cams.map(c => new Option(c.label || 'Caméra', c.deviceId, false, c.deviceId === selected)));
}

navigator.mediaDevices?.addEventListener('devicechange', () => { if (!rgbStream && connected) startRgb(); });

// ---------------------------------------------------------------- hand tracking (hands worker)

const handsWorker = new Worker(new URL('./workers/hands.worker.ts', import.meta.url), { type: 'module' });
const toHands = (m: ToHands, transfer: Transferable[] = []) => handsWorker.postMessage(m, transfer);
let hands: Hand[] = [];
const handPredictor = new HandPredictor();
let handsRate = 0;
/** Smoothed hand latency parts (ms): waiting for the worker, MediaPipe, page queue. */
const handLatency = { waited: 0, detect: 0, queue: 0 };
let handsStatus = '';
let rgbBusy = false;
const handSource = $('sel-hand-source') as HTMLSelectElement;
/** Calibrated RGB camera (null: assumed at the head origin with the "Champ RGB" field). */
let rgbCamera: RgbCamera | null = loadRgbCamera();
/** Calibration session in progress. */
let calib: { until: number; data: Correspondence[]; stereo: Hand[][]; poses: Map<number, THREE.Matrix4> } | null = null;
const hfovInput = $('rng-hfov') as HTMLInputElement;
hfovInput.value = store.get('rgbHfov') ?? hfovInput.value;
handSource.value = store.get('handSource') ?? 'stereo';
setText('o-hfov', `${hfovInput.value}°`);
$('hfov-wrap').hidden = handSource.value !== 'rgb';

function syncHandOptions() {
  $('hfov-wrap').hidden = handSource.value !== 'rgb';
  setText('o-hfov', `${hfovInput.value}°`);
  store.set('rgbHfov', hfovInput.value);
  store.set('handSource', handSource.value);
  if (!checked('chk-hands')) hands = [];
  toHands({
    type: 'options', enabled: checked('chk-hands') || !!calib, source: handSource.value as 'stereo' | 'rgb',
    rgbCam: rgbCamera, hfov: +hfovInput.value, calibrating: !!calib,
  });
}
hfovInput.oninput = syncHandOptions;
handSource.onchange = syncHandOptions;
$('chk-hands').addEventListener('change', syncHandOptions);
syncHandOptions();

handsWorker.onmessage = (e: MessageEvent<FromHands>) => {
  const m = e.data;
  if (m.type === 'hands') {
    if (m.ms) {
      // Where the latency goes: before MediaPipe, in it, and queued for the page's thread.
      const k = 0.1, l = handLatency;
      l.waited += (m.ms.waited - l.waited) * k;
      l.detect += (m.ms.convert + m.ms.detect - l.detect) * k;
      l.queue += (absNow() - m.ms.posted - l.queue) * k;
    }
    if (calib) collectCalibration(m.hands, m.source);
    if (m.source === handSource.value && checked('chk-hands')) {
      hands = m.hands;
      handPredictor.push(m.hands);
    }
    handsRate = m.rate;
  } else if (m.type === 'status') {
    handsStatus = m.text;
  } else if (m.type === 'idle') {
    rgbBusy = false;
  }
};

/** RGB source: hand the newest video frame to the worker when it is free. */
function feedRgbHands() {
  const wanted = calib || (handSource.value === 'rgb' && checked('chk-hands'));
  if (!wanted || rgbBusy || !rgbStream || video.readyState < 2) return;
  rgbBusy = true;
  const q = displayOrientation(new THREE.Quaternion(), 0);
  const p = displayPosition(new THREE.Vector3());
  const pose = { q: [q.x, q.y, q.z, q.w] as [number, number, number, number], p: p ? [p.x, p.y, p.z] as [number, number, number] : null, stamp: absNow() };
  // Kept so calibration can bring stereo joints into this frame's head pose.
  if (calib) calib.poses.set(pose.stamp, headMatrix(q.clone(), p ? p.clone() : null));
  // Half resolution: the landmarker works at 224 px anyway, and the copy costs GPU time
  // the glasses view needs.
  createImageBitmap(video, { resizeWidth: 960, resizeHeight: Math.round(960 * video.videoHeight / video.videoWidth), resizeQuality: 'low' }).then(
    bitmap => toHands({ type: 'rgb', bitmap, pose }, [bitmap]),
    () => { rgbBusy = false; },
  );
}

// ---------------------------------------------------------------- RGB camera calibration
//
// For ~20 s both trackers run: the stereo one gives metric 3D joints, the RGB one the
// same joints in pixels. Pairs are kept when both saw the same hand at nearly the
// same time and the hand was still; a least-squares fit then gives the RGB camera.

const CALIB_SECONDS = 20;

function startCalibration() {
  if (!rig || !rgbStream) { banner('Calibration RGB : connecte les lunettes et la caméra RGB d\u2019abord.'); return; }
  calib = { until: performance.now() + CALIB_SECONDS * 1000, data: [], stereo: [], poses: new Map() };
  syncHandOptions();
  banner('Calibration RGB : montre une main ouverte, immobile une seconde à chaque endroit (gauche, droite, haut, bas, près, loin), devant la caméra RGB.');
}

function collectCalibration(found: Hand[], source: 'stereo' | 'rgb') {
  if (!calib) return;
  if (source === 'stereo') {
    calib.stereo.push(found.filter(h => h.depth === 'stéréo'));
    if (calib.stereo.length > 12) calib.stereo.shift();
    return;
  }
  const W = video.videoWidth, H = video.videoHeight;
  for (const h of found) {
    const px = h.views.rgb;
    const pose = h.stamp !== undefined ? calib.poses.get(h.stamp) : undefined;
    if (!px || !pose || !h.stamp) continue;
    // Closest stereo measurement of the same hand, and the one before it to check stillness.
    let best: Hand | null = null, prev: Hand | null = null;
    for (let k = calib.stereo.length - 1; k >= 0; k--) {
      const s = calib.stereo[k].find(x => x.side === h.side);
      if (!s?.stamp) continue;
      if (!best) { if (Math.abs(s.stamp - h.stamp) < 80) best = s; continue; }
      prev = s;
      break;
    }
    if (!best || !prev) continue;
    const moved = Math.hypot(...best.joints[0].map((v, i) => v - prev!.joints[0][i]));
    if (moved > 0.012) continue;
    const toHead = pose.clone().invert();
    best.joints.forEach((j, i) => {
      const P = new THREE.Vector3(...j).applyMatrix4(toHead);
      calib!.data.push({ P: [P.x, P.y, P.z], u: px[i].x * W, v: px[i].y * H });
    });
  }
  for (const t of [...calib.poses.keys()]) if (t < absNow() - 2000) calib.poses.delete(t);
}

function finishCalibration() {
  const c = calib!;
  calib = null;
  syncHandOptions();
  const W = video.videoWidth, H = video.videoHeight;
  if (c.data.length < 21 * 8) {
    banner(`Calibration RGB : pas assez de mesures (${Math.floor(c.data.length / 21)} positions de main). Recommence en gardant la main immobile plus longtemps.`);
    return;
  }
  const prior = defaultRgbCamera(W, H, +hfovInput.value);
  const cam = fitRgbCamera(rgbCamera ?? prior, c.data, prior);
  if (!(cam.rms! < 25) || !plausibleRgbCamera(cam)) {
    banner(`Calibration RGB rejetée : erreur ${cam.rms!.toFixed(1)} px${plausibleRgbCamera(cam) ? '' : ', position de caméra impossible'}. Recommence.`);
    return;
  }
  rgbCamera = cam;
  saveRgbCamera(cam);
  syncHandOptions();
  const hfov = THREE.MathUtils.radToDeg(2 * Math.atan(W / 2 / cam.f));
  banner(`Caméra RGB calibrée : champ ${hfov.toFixed(1)}°, décalage ${cam.t.map(v => (v * 100).toFixed(1)).join(' / ')} cm, erreur ${cam.rms!.toFixed(1)} px sur ${Math.floor(c.data.length / 21)} positions.`);
}

setInterval(() => {
  if (!calib) return;
  const left = Math.ceil((calib.until - performance.now()) / 1000);
  setText('btn-rgb-calib', `${left} s · ${Math.floor(calib.data.length / 21)}`);
  if (left <= 0) { setText('btn-rgb-calib', 'Calibrer'); finishCalibration(); }
}, 250);

function drawHandOverlays() {
  const src = handSource.value;
  drawHands($('hand-overlay') as HTMLCanvasElement, video.videoWidth, video.videoHeight, src === 'rgb' ? hands : [], 'rgb');
  drawHands($('overlay-left') as HTMLCanvasElement, 640, 480, src === 'stereo' ? hands : [], 'left');
  drawHands($('overlay-right') as HTMLCanvasElement, 640, 480, src === 'stereo' ? hands : [], 'right');
}

setInterval(() => {
  if (!checked('chk-hands')) { setText('hands-info', 'désactivé'); return; }
  if (handSource.value === 'stereo' && !rig) { setText('hands-info', 'en attente de la calibration (connecte les lunettes)'); return; }
  if (!handsStatus.startsWith('prêt')) { setText('hands-info', handsStatus || 'chargement du modèle…'); return; }
  const headPos = new THREE.Vector3(0, 0, 0).applyMatrix4(world.head.matrixWorld);
  const desc = hands.map(h => `${h.side === 'left' ? 'G' : 'D'} ${(headPos.distanceTo(new THREE.Vector3(...h.joints[0])) * 100).toFixed(0)} cm ${h.depth}${h.pinching ? ' · pince' : ''}`).join('\n');
  const l = handLatency;
  setText('hands-info', `${handsRate.toFixed(0)} détections/s, ${handsStatus.slice(6, -1)}, latence ${handPredictor.latencyMs.toFixed(0)} ms (compensée)\n` +
    `= attente ${l.waited.toFixed(0)} + MediaPipe ${l.detect.toFixed(0)} + page ${l.queue.toFixed(0)} ms\n${desc || 'aucune main'}`);
}, 250);

// ---------------------------------------------------------------- display calibration

const display = loadSettings();

const displayInputs: [string, string, (v: number) => void, () => number, (v: number) => string][] = [
  ['rng-vfov', 'o-vfov', v => (display.vfov = v), () => display.vfov, v => `${v.toFixed(2)}°`],
  ['rng-ipd', 'o-ipd', v => (display.ipd = v / 1000), () => display.ipd * 1000, v => `${v.toFixed(1)} mm`],
  ['rng-eye-y', 'o-eye-y', v => (display.eyeOffset[1] = v / 1000), () => display.eyeOffset[1] * 1000, v => `${v.toFixed(0)} mm`],
  ['rng-eye-z', 'o-eye-z', v => (display.eyeOffset[2] = v / 1000), () => display.eyeOffset[2] * 1000, v => `${v.toFixed(0)} mm`],
  ['rng-pitch', 'o-pitch', v => (display.pitch = v), () => display.pitch, v => `${v.toFixed(1)}°`],
];

function syncDisplayUi() {
  for (const [id, out, , get, fmt] of displayInputs) {
    ($(id) as HTMLInputElement).value = String(get());
    setText(out, fmt(get()));
  }
}

for (const [id, out, set, , fmt] of displayInputs) {
  const input = $(id) as HTMLInputElement;
  input.oninput = () => {
    set(+input.value);
    setText(out, fmt(+input.value));
    saveSettings(display);
  };
}
$('btn-display-reset').onclick = () => {
  Object.assign(display, { ...DEFAULT_SETTINGS, eyeOffset: [...DEFAULT_SETTINGS.eyeOffset] });
  saveSettings(display);
  syncDisplayUi();
};
syncDisplayUi();

const predictInput = $('rng-predict') as HTMLInputElement;
predictInput.value = store.get('predictMs') ?? predictInput.value;
function syncPredict() {
  setText('o-predict', `${predictInput.value} ms`);
  store.set('predictMs', predictInput.value);
  predictMs = +predictInput.value;
}
predictInput.oninput = syncPredict;
syncPredict();

function eyeSetup(): EyeSetup {
  return { ...eyePoses(display), vfov: display.vfov };
}

// ---------------------------------------------------------------- mixed-reality views
//
// One world, three views: the dedicated window on the glasses' screen (per eye,
// side-by-side 3D), the in-page fallback overlay, and the RGB overlay on the
// dashboard (the scene seen from the RGB camera, composited over its video).

const world = new XrWorld({
  orientation: displayOrientation,
  position: displayPosition,
  hands: () => handPredictor.predict(hands),
  eyes: eyeSetup,
  depth: () => (checked('chk-depth') ? depthFrame : null),
  rgbCamera: () => rgbCamera ?? (video.videoWidth ? defaultRgbCamera(video.videoWidth, video.videoHeight, +hfovInput.value) : null),
});
const immersiveEl = $('immersive');
const pageView = new XrView($('immersive-canvas') as HTMLCanvasElement, world, 'glasses');
const mrCanvas = $('mr-overlay') as HTMLCanvasElement;
const mrView = new XrView(mrCanvas, world, 'rgb', () => (video.videoWidth && video.videoHeight ? video.videoWidth / video.videoHeight : 16 / 9));
// The RGB overlay only needs the camera's 30 i/s (15 while the glasses window renders).
mrView.minGapMs = 30;
let glassesView: XrView | null = null;
let glassesWin: Window | null = null;

function syncWorldOptions() {
  Object.assign(world.options, {
    anchorTest: checked('chk-anchor'),
    occlusion: checked('chk-occlusion') && checked('chk-depth'),
    showDepth: checked('chk-show-depth') && checked('chk-depth'),
  });
}
for (const id of ['chk-anchor', 'chk-occlusion', 'chk-show-depth', 'chk-depth']) $(id).addEventListener('change', syncWorldOptions);
syncWorldOptions();

function syncMr() {
  const on = checked('chk-mr');
  mrCanvas.hidden = !on;
  if (on) mrView.start(); else mrView.stop();
}
$('chk-mr').addEventListener('change', syncMr);
syncMr();

let in3d = false;

function setGlasses3d(on: boolean) {
  in3d = on;
  if (connected) toDevice({ type: 'display', mode: on ? DisplayMode.MODE_3D : DisplayMode.MODE_2D });
}

function setSbs(on: boolean) {
  pageView.sbs = on;
  if (glassesView) glassesView.sbs = on;
  setText('btn-sbs', `SBS 3D : ${on ? 'oui' : 'non'}`);
  const open = !!glassesView || !immersiveEl.hidden;
  $('btn-immersive').classList.toggle('active', open);
  setText('btn-immersive', open ? 'Quitter la 3D' : 'Scène 3D');
}

async function openImmersive() {
  if (glassesWin && !glassesWin.closed) { glassesWin.focus(); return; }
  const res = await openGlassesWindow(() => setGlasses3d(true));
  if (!res.ok) {
    if (res.reason === 'no-api') return openPageScene();
    banner(`Scène 3D : ${{
      'mirrored': 'les lunettes ne sont pas un écran séparé. Win+P → Étendre, puis relance la scène 3D.',
      'not-3d': 'l’écran des lunettes n’est pas passé en 3840×1200. Appuie sur le bouton 2D/3D des lunettes puis relance la scène 3D.',
      'blocked': 'lunettes passées en 3D. Clique encore sur « Scène 3D » pour ouvrir la fenêtre.',
    }[res.reason]}`);
    return;
  }
  if (res.switched) in3d = true;
  const win = res.win;
  const canvas = await glassesCanvas(win);
  if (!canvas) { banner('Scène 3D : la fenêtre des lunettes ne s’est pas chargée.'); return; }
  glassesWin = win;
  glassesView = new XrView(canvas, world, 'glasses');
  mrView.minGapMs = 66;
  setSbs(true);
  glassesView.start();
  recenter();
  win.addEventListener('keydown', e => { if (!onTalkKey(e, true)) onSceneKey(e); });
  win.addEventListener('keyup', e => onTalkKey(e, false));
  // The window closing ends the session and restores 2D.
  const watch = setInterval(() => { if (win.closed) { clearInterval(watch); closeImmersive(); } }, 300);
  banner(null);
}

/** Fallback when the browser cannot place windows: the scene over the dashboard. */
function openPageScene() {
  immersiveEl.hidden = false;
  setSbs(false);
  pageView.start();
  banner('Ce navigateur ne sait pas viser l’écran des lunettes : glisse la fenêtre dessus puis clique « Plein écran ».');
}

function closeImmersive() {
  if (glassesView) {
    mrView.minGapMs = 30;
    glassesView.dispose();
    glassesView = null;
    if (glassesWin && !glassesWin.closed) glassesWin.close();
    glassesWin = null;
  }
  if (!immersiveEl.hidden) {
    pageView.stop();
    immersiveEl.hidden = true;
    if (document.fullscreenElement) document.exitFullscreen();
  }
  if (in3d) setGlasses3d(false);
  setSbs(false);
}

function onSceneKey(e: KeyboardEvent) {
  const k = e.key.toLowerCase();
  if (k === 'r') recenter();
  else if (k === 's') setSbs(!(glassesView ?? pageView).sbs);
  else if (k === 'escape') closeImmersive();
  // 1..7: the human's clips (idle, talk, listen, wave, shrug, think, clap)
  else if (/^[1-9]$/.test(k) && HUMAN_ANIMS[+k - 1]) world.human.play(HUMAN_ANIMS[+k - 1]);
  else if (k === 'l') world.human.studio = !world.human.studio;
}

// ---------------------------------------------------------------- WebXR (step 6)
//
// webxr.html is an ordinary three.js WebXR page; src/webxr/runtime.ts gives it a WebXR
// device for the glasses, fed from here: pose, display calibration and hands.

const xrChannel = new BroadcastChannel(XR_CHANNEL);
let xrWin: Window | null = null;
// The extension (extension/) asks for 3D when a site starts a WebXR session, 2D after.
// WebXR pages (through the extension, or webxr.html) send their hit-test rays here.
new BroadcastChannel(XR_CONTROL_CHANNEL).onmessage = e => {
  if (e.data?.type === 'display3d') setGlasses3d(!!e.data.on);
  else if (e.data?.type === 'rays') xrHits = hitTestRays(e.data.rays);
};
let xrHits: XrFeed['hits'] = {};

/**
 * Scene rays against the depth map: hit point and surface normal (from two neighbour
 * rays 3 cm off), or null. The normal faces the ray's origin; up when it cannot be told.
 */
function hitTestRays(rays: XrRay[]): XrFeed['hits'] {
  const out: NonNullable<XrFeed['hits']> = {};
  const depth = checked('chk-depth') ? depthFrame : null;
  const head = headMatrix(displayOrientation(new THREE.Quaternion(), 0), displayPosition(new THREE.Vector3()));
  const inv = head.clone().invert();
  for (const r of rays.slice(0, 16)) {
    const o = new THREE.Vector3(...r.o).applyMatrix4(inv);
    const d = new THREE.Vector3(...r.d).transformDirection(inv);
    const hit = hitDepth(o, d, depth);
    if (!hit.measured) { out[r.id] = null; continue; }
    const side = new THREE.Vector3(0, 1, 0).cross(d);
    if (side.lengthSq() < 1e-6) side.set(1, 0, 0);
    side.normalize();
    const up = d.clone().cross(side).normalize();
    const near = (off: THREE.Vector3) => {
      const target = hit.point.clone().add(off);
      const h = hitDepth(o, target.sub(o).normalize(), depth);
      return h.measured ? h.point : null;
    };
    const a = near(side.multiplyScalar(0.03)), b = near(up.multiplyScalar(0.03));
    let n = a && b ? a.sub(hit.point).cross(b.sub(hit.point)).normalize() : null;
    if (n && n.dot(d) > 0) n.negate();
    const p = hit.point.applyMatrix4(head);
    const ns = n ? n.transformDirection(head) : new THREE.Vector3(0, 1, 0);
    out[r.id] = { p: p.toArray(), n: ns.toArray() };
  }
  return out;
}

function feedXr(state: PoseState) {
  const e = eyeSetup();
  const msg: XrFeed = {
    pose: state,
    eyes: { left: e.left.toArray() as [number, number, number], right: e.right.toArray() as [number, number, number], rotation: e.rotation.toArray() as [number, number, number, number], vfov: e.vfov },
    hands: handPredictor.predict(hands).map(h => ({ side: h.side, joints: h.joints, pinching: h.pinching })),
    predictMs,
    eyeHeight: EYE_HEIGHT,
    map: mapStatus?.relocalized ? mapStatus.T : null,
    hits: xrHits,
  };
  xrChannel.postMessage(msg);
}

async function openWebXr() {
  if (xrWin && !xrWin.closed) { xrWin.close(); return; }
  if (glassesView || !immersiveEl.hidden) closeImmersive();
  const res = await openGlassesWindow(() => setGlasses3d(true), '/webxr.html');
  if (!res.ok) {
    banner(res.reason === 'no-api' ? 'WebXR : ce navigateur ne sait pas viser l\u2019écran des lunettes.' : 'WebXR : passe les lunettes en 3D (bouton 2D/3D) puis réessaie.');
    return;
  }
  if (res.switched) in3d = true;
  xrWin = res.win;
  $('btn-webxr').classList.add('active');
  const watch = setInterval(() => {
    if (!xrWin || xrWin.closed) {
      clearInterval(watch);
      xrWin = null;
      $('btn-webxr').classList.remove('active');
      if (in3d && !glassesView) setGlasses3d(false);
    }
  }, 300);
}
$('btn-webxr').onclick = openWebXr;

// Leaving fullscreen on the in-page fallback ends it.
document.addEventListener('fullscreenchange', () => {
  if (!document.fullscreenElement && !immersiveEl.hidden) closeImmersive();
});
window.addEventListener('beforeunload', () => { if (glassesWin && !glassesWin.closed) glassesWin.close(); });

// ---------------------------------------------------------------- assistant (Orange gateway)
//
// Questions by voice (push-to-talk, or hands-free starting with « lunettes »), answered
// by Gemini looking at a camera frame; the answer shows in the scene and is spoken.
// « Objets » labels what Gemini finds, placed in 3D with the depth map. The default
// camera is the left tracking camera: factory-calibrated and the depth map's own
// camera, so labels land where the objects are without any RGB calibration.

const ASSISTANT_SYSTEM =
  'Tu es l’assistant de lunettes de réalité mixte. L’image jointe est la vue de la caméra frontale : ' +
  'ce que l’utilisateur a devant les yeux. Réponds en français, en une à trois phrases courtes, sans markdown ni liste.';
const WAKE = /^\s*(?:ok\s+|hey\s+|dis\s+)?(?:lunettes?|viture)\b[\s,.!?:]*/i;

const voiceIn = new VoiceInput();
const speaker = new Speaker();
const voiceSel = $('sel-voice') as HTMLSelectElement;
voiceSel.value = store.get('voice') ?? 'orange';
speaker.mode = voiceSel.value as VoiceMode;
voiceSel.onchange = () => { speaker.mode = voiceSel.value as VoiceMode; store.set('voice', voiceSel.value); };
speaker.onSpeaking = on => { voiceIn.muted = on; };
const seeCam = $('sel-see-cam') as HTMLSelectElement;
seeCam.value = store.get('seeCam') ?? 'stereo';
seeCam.onchange = () => { store.set('seeCam', seeCam.value); objectBoxes = null; drawObjectBoxes(); };
let objectBoxes: { dets: Detection[]; until: number; cam: 'stereo' | 'rgb' } | null = null;

function assistantLog(text: string) {
  setText('assistant-log', text);
}

interface Shot {
  image: string;
  head: THREE.Matrix4;
  /** Head-frame point seen at image position (x, y), 0..1. */
  locate(x: number, y: number): { point: THREE.Vector3; measured: boolean };
  /** Added to the system prompt for this camera. */
  note: string;
}

/** The selected camera's frame now (JPEG) and the head pose at that moment. */
function grabShot(): Shot | null {
  const head = headMatrix(displayOrientation(new THREE.Quaternion(), 0), displayPosition(new THREE.Vector3()));
  const depth = checked('chk-depth') ? depthFrame : null;
  if (seeCam.value === 'stereo') {
    const left = $('cam-left') as HTMLCanvasElement;
    const cam = rig?.[0];
    if (!cam || !stereo.version[0]) return null;
    return {
      image: left.toDataURL('image/jpeg', 0.85),
      head,
      locate: (x, y) => hitDepth(new THREE.Vector3(...cam.origin), new THREE.Vector3(...cam.ray(x * left.width, y * left.height)), depth),
      note: ' L’image vient d’une caméra grand angle en niveaux de gris : ne parle ni des couleurs ni de la qualité de l’image.',
    };
  }
  const cam = world.src.rgbCamera();
  if (!rgbStream || !video.videoWidth || !cam) return null;
  const c = document.createElement('canvas');
  c.width = 1024;
  c.height = Math.round((video.videoHeight / video.videoWidth) * 1024);
  c.getContext('2d')!.drawImage(video, 0, 0, c.width, c.height);
  return { image: c.toDataURL('image/jpeg', 0.8), head, locate: (x, y) => locateRgbPixel(cam, x * cam.width, y * cam.height, depth), note: '' };
}

/**
 * Where a detected object is: the depth map is sparse, so a 5x5 grid over the middle of
 * the box is tried and the median measured distance wins; the centre ray otherwise.
 */
function locateBox(shot: Shot, b: Detection['box']): { point: THREE.Vector3; measured: boolean } {
  const hits: { point: THREE.Vector3; d: number }[] = [];
  for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
    const x = b.x0 + (b.x1 - b.x0) * (0.2 + 0.15 * i), y = b.y0 + (b.y1 - b.y0) * (0.2 + 0.15 * j);
    const at = shot.locate(x, y);
    if (at.measured) hits.push({ point: at.point, d: at.point.length() });
  }
  if (hits.length < 2) return shot.locate((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2);
  hits.sort((a, c) => a.d - c.d);
  return { point: hits[hits.length >> 1].point, measured: true };
}

async function ask(question: string) {
  const shot = grabShot();
  speaker.stop();
  assistantLog(`❓ ${question}\n…`);
  const t0 = performance.now();
  try {
    const answer = await chat(ASSISTANT_SYSTEM + (shot?.note ?? ''), question, shot?.image);
    assistantLog(`❓ ${question}\n💬 ${answer}\n(${((performance.now() - t0) / 1000).toFixed(1)} s${shot ? '' : ', sans image'})`);
    world.showMessage(answer);
    speaker.say(answer);
  } catch (e) {
    assistantLog(`❓ ${question}\n⚠ ${(e as Error).message}`);
  }
}

voiceIn.onUtterance = async (audio, pushed) => {
  assistantLog('🎙 transcription…');
  try {
    let text = await transcribe(audio);
    if (!pushed) {
      const m = text.match(WAKE);
      if (!m) { assistantLog(`(entendu, sans « lunettes » : « ${text} »)`); return; }
      text = text.slice(m[0].length);
    }
    if (text.trim().length < 2) { assistantLog('🎙 rien compris'); return; }
    await ask(text.trim());
  } catch (e) {
    assistantLog(`⚠ ${(e as Error).message}`);
  }
};

async function findObjects() {
  const shot = grabShot();
  const cam = seeCam.value as 'stereo' | 'rgb';
  if (!shot) { assistantLog(cam === 'stereo' ? 'Objets : connecte les lunettes d’abord.' : 'Objets : la caméra RGB n’est pas active.'); return; }
  assistantLog('🔎 recherche des objets…');
  const t0 = performance.now();
  try {
    const dets = await detect(shot.image);
    world.clearLabels();
    let measured = 0;
    for (const d of dets) {
      const at = locateBox(shot, d.box);
      if (at.measured) measured++;
      const p = at.point.applyMatrix4(shot.head);
      // Remembered with the room when it is recognised; shown once otherwise.
      if (!at.measured || !addAnchor(d.label, p)) world.addLabel(d.label, p);
    }
    objectBoxes = { dets, until: performance.now() + 10000, cam };
    drawObjectBoxes();
    assistantLog(`🔎 ${dets.length} objets (${measured} placés avec la profondeur, ${((performance.now() - t0) / 1000).toFixed(1)} s) :\n${dets.map(d => d.label).join(', ') || 'aucun'}`);
  } catch (e) {
    assistantLog(`⚠ ${(e as Error).message}`);
  }
}

function drawObjectBoxes() {
  for (const id of ['obj-overlay', 'obj-overlay-left']) {
    const c = $(id) as HTMLCanvasElement;
    c.getContext('2d')!.clearRect(0, 0, c.width, c.height);
  }
  if (!objectBoxes || performance.now() > objectBoxes.until) { objectBoxes = null; return; }
  const onLeft = objectBoxes.cam === 'stereo';
  const canvas = $(onLeft ? 'obj-overlay-left' : 'obj-overlay') as HTMLCanvasElement;
  const dpr = devicePixelRatio || 1;
  const cw = Math.round(canvas.clientWidth * dpr), ch = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
  const ctx = canvas.getContext('2d')!;
  const sw = onLeft ? 640 : video.videoWidth, sh = onLeft ? 480 : video.videoHeight;
  if (!sw) return;
  const scale = Math.min(cw / sw, ch / sh);
  const ox = (cw - sw * scale) / 2, oy = (ch - sh * scale) / 2;
  ctx.font = `${12 * dpr}px system-ui, sans-serif`;
  ctx.lineWidth = 2 * dpr;
  for (const d of objectBoxes.dets) {
    const x = ox + d.box.x0 * sw * scale, y = oy + d.box.y0 * sh * scale;
    const w = (d.box.x1 - d.box.x0) * sw * scale, h = (d.box.y1 - d.box.y0) * sh * scale;
    ctx.strokeStyle = ctx.fillStyle = '#ffd2b0';
    ctx.strokeRect(x, y, w, h);
    ctx.fillText(d.label, x + 4 * dpr, y - 4 * dpr);
  }
}
setInterval(() => { if (objectBoxes) drawObjectBoxes(); }, 500);

const talkBtn = $('btn-talk');
const talkStart = async () => {
  talkBtn.classList.add('active');
  speaker.stop();
  try { await voiceIn.pushStart(); } catch (e) { assistantLog(`⚠ micro : ${(e as Error).message}`); talkBtn.classList.remove('active'); }
};
const talkEnd = () => { talkBtn.classList.remove('active'); voiceIn.pushEnd(); };
talkBtn.addEventListener('pointerdown', talkStart);
talkBtn.addEventListener('pointerup', talkEnd);
talkBtn.addEventListener('pointerleave', () => { if (talkBtn.classList.contains('active')) talkEnd(); });
$('chk-listen').addEventListener('change', async () => {
  try {
    await voiceIn.setListening(checked('chk-listen'));
    if (checked('chk-listen')) assistantLog(`Écoute active (${voiceIn.device || 'micro par défaut'}) : commence par « lunettes ».`);
  } catch (e) {
    ($('chk-listen') as HTMLInputElement).checked = false;
    assistantLog(`⚠ micro : ${(e as Error).message}`);
  }
});
$('btn-see').onclick = () => ask('Que vois-tu devant moi ?');
$('btn-objects').onclick = findObjects;
$('btn-labels-clear').onclick = () => { world.clearLabels(); objectBoxes = null; drawObjectBoxes(); };
setInterval(() => { ($('mic-level') as HTMLElement).style.width = `${(voiceIn.level * 100).toFixed(0)}%`; }, 60);

const usd = (v: number) => (v < 0.01 ? `${(v * 100).toFixed(3)} c$` : `${v.toFixed(3)} $`);

// Vision model: any of the gateway's, cheapest first, with what each call cost and took.
const modelSel = $('sel-vision-model') as HTMLSelectElement;
let visionList: ModelInfo[] = [];
MODELS.vision = store.get('visionModel') ?? MODELS.vision;
function modelLabel(m: ModelInfo) {
  const s = costs.byModel[m.id];
  const measured = s?.n ? ` · ${usd(s.total / s.n)}/appel · ${(s.ms / s.n / 1000).toFixed(1)} s` : '';
  return `${m.id.replace(/^[^/]+\//, '')} · ${m.inCost.toFixed(2)} / ${m.outCost.toFixed(2)} $${measured}`;
}
function fillModels() {
  const list = visionList.length ? visionList : [{ id: MODELS.vision, inCost: NaN, outCost: NaN }];
  modelSel.replaceChildren(...list.map(m => new Option(visionList.length ? modelLabel(m) : m.id, m.id, false, m.id === MODELS.vision)));
}
fillModels();
visionModels().then(list => { visionList = list; fillModels(); }, e => assistantLog(`⚠ liste des modèles : ${(e as Error).message}`));
modelSel.onchange = () => { MODELS.vision = modelSel.value; store.set('visionModel', modelSel.value); };
let labelledCalls = 0;

setInterval(() => {
  if (!costs.calls) return;
  // New measurements: refresh the labels (not while the list is open).
  if (costs.calls !== labelledCalls && document.activeElement !== modelSel) { labelledCalls = costs.calls; if (visionList.length) fillModels(); }
  const k = costs.byKind;
  const key = Number.isFinite(costs.keySpend) ? `
clé du labo : ${costs.keySpend.toFixed(2)} / ${costs.keyBudget.toFixed(0)} $` : '';
  setText('ai-cost', `Coût IA : ${usd(costs.session)} (${costs.calls} appels, dernier ${usd(costs.last)})
vision ${usd(k.vision)} · transcription ${usd(k.transcription)} · voix ${usd(k.voix)}${key}`);
}, 1000);

/** Space bar: push-to-talk, from the dashboard or the glasses window. */
function onTalkKey(e: KeyboardEvent, down: boolean) {
  if (e.code !== 'Space' || e.repeat || e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return false;
  e.preventDefault();
  if (down) talkStart(); else talkEnd();
  return true;
}
window.addEventListener('keyup', e => onTalkKey(e, false));

// ---------------------------------------------------------------- environment memory and anchors (step 5)
//
// The vision worker keeps a map of the room's landmarks in its own frame and finds, in
// each session, where the current tracking frame sits in it (T: map -> tracking frame).
// Anchors are stored in the map frame: they come back at the same place after a restart,
// once the room is recognised. Map frame -> tracking frame (T) -> scene (pose.frame).

let mapStatus: MapStatus | null = null;
let anchors: Anchor[] = [];
const anchorViews = new Map<string, THREE.Object3D>();
let mapLoaded = false;
let mapDirty = false;
let lastMapPoints: Float32Array | null = null;

/** Tracking frame -> scene, from the pose's frame (recenter included). */
function trackingToScene(x: THREE.Vector3): THREE.Vector3 {
  const f = pose?.frame;
  if (!f) return x;
  return x.sub(new THREE.Vector3(...f.origin)).applyQuaternion(new THREE.Quaternion(...f.yaw)).add(new THREE.Vector3(0, EYE_HEIGHT, 0));
}

function sceneToTracking(x: THREE.Vector3): THREE.Vector3 {
  const f = pose?.frame;
  if (!f) return x;
  return x.sub(new THREE.Vector3(0, EYE_HEIGHT, 0)).applyQuaternion(new THREE.Quaternion(...f.yaw).invert()).add(new THREE.Vector3(...f.origin));
}

function mapToTracking(m: [number, number, number], T: MapTransform): THREE.Vector3 {
  const c = Math.cos(T.yaw), s = Math.sin(T.yaw);
  return new THREE.Vector3(c * m[0] + s * m[2] + T.t[0], m[1] + T.t[1], -s * m[0] + c * m[2] + T.t[2]);
}

function trackingToMap(x: THREE.Vector3, T: MapTransform): [number, number, number] {
  const c = Math.cos(T.yaw), s = Math.sin(T.yaw);
  const dx = x.x - T.t[0], dy = x.y - T.t[1], dz = x.z - T.t[2];
  return [c * dx - s * dz, dy, s * dx + c * dz];
}

/** Stores a scene point as an anchor; false until the room is recognised. */
function addAnchor(label: string, scenePos: THREE.Vector3): boolean {
  const T = mapStatus?.relocalized ? mapStatus.T : null;
  if (!T) return false;
  anchors.push({ id: crypto.randomUUID(), label, m: trackingToMap(sceneToTracking(scenePos.clone()), T), created: Date.now() });
  mapDirty = true;
  return true;
}

/** Keeps anchor labels where their map position says, as T and the recenter change. */
function updateAnchors() {
  const T = mapStatus?.relocalized ? mapStatus.T : null;
  for (const [id, g] of anchorViews) if (!anchors.some(a => a.id === id)) { world.removeLabel(g); anchorViews.delete(id); }
  for (const a of anchors) {
    let g = anchorViews.get(a.id);
    if (!T) { if (g) g.visible = false; continue; }
    const p = trackingToScene(mapToTracking(a.m, T));
    if (!g) { g = world.addLabel(a.label, p, '#9dffcf'); anchorViews.set(a.id, g); }
    g.visible = true;
    g.position.copy(p);
  }
}
setInterval(updateAnchors, 50);

function showMapPoints() {
  if (!checked('chk-map') || !lastMapPoints) { world.setMapPoints(null); return; }
  const out = new Float32Array(lastMapPoints.length);
  const v = new THREE.Vector3();
  for (let i = 0; i < out.length; i += 3) {
    trackingToScene(v.set(lastMapPoints[i], lastMapPoints[i + 1], lastMapPoints[i + 2]));
    out[i] = v.x; out[i + 1] = v.y; out[i + 2] = v.z;
  }
  world.setMapPoints(out);
}
$('chk-map').addEventListener('change', showMapPoints);

function onMapMessage(m: FromVision) {
  if (m.type === 'map') {
    if (mapStatus && m.status.points !== mapStatus.points) mapDirty = true;
    mapStatus = m.status;
    const s = m.status;
    setText('map-info', `mémoire : ${s.points} points · ${s.relocalized ? `pièce reconnue (${s.inliers}/${s.matches})` : 'recherche de la pièce…'} · ${anchors.length} ancre${anchors.length > 1 ? 's' : ''}`);
  } else if (m.type === 'mapPoints') {
    lastMapPoints = m.points;
    showMapPoints();
  } else if (m.type === 'mapData') {
    saveMap({ map: m.data, anchors, saved: Date.now() });
  }
}

/** Loads the stored room into the vision worker, once per page. */
async function restoreMap() {
  if (mapLoaded) return;
  mapLoaded = true;
  const stored = await loadMap();
  if (!stored) return;
  anchors = stored.anchors ?? [];
  toVision({ type: 'mapLoad', data: stored.map });
}

// Saved every 20 s when something changed, and when the page is hidden or closed.
const persistMap = () => { if (mapDirty) { mapDirty = false; toVision({ type: 'mapSave' }); } };
setInterval(persistMap, 20000);
document.addEventListener('visibilitychange', () => { if (document.hidden) persistMap(); });

$('btn-anchor').onclick = () => {
  // Where the head looks: the centre ray against the depth map.
  const hit = hitDepth(new THREE.Vector3(), new THREE.Vector3(0, 0, -1), checked('chk-depth') ? depthFrame : null);
  const head = headMatrix(displayOrientation(new THREE.Quaternion(), 0), displayPosition(new THREE.Vector3()));
  if (!addAnchor(`Ancre ${anchors.length + 1}`, hit.point.applyMatrix4(head))) banner('Ancre : la pièce n’est pas encore reconnue, regarde autour de toi quelques secondes.');
  else { banner(null); persistMap(); }
};
$('btn-map-clear').onclick = () => {
  anchors = [];
  toVision({ type: 'mapClear' });
  clearMap();
  mapDirty = false;
  lastMapPoints = null;
  showMapPoints();
};

// ---------------------------------------------------------------- first-person recording (RGB + scene)

/** Composites the RGB video and the MR overlay into one stream for MediaRecorder. */
function mixedStream(): MediaStream {
  const c = document.createElement('canvas');
  c.width = video.videoWidth || 1920;
  c.height = video.videoHeight || 1080;
  const ctx = c.getContext('2d')!;
  const draw = () => {
    if (!recorder.recording) return;
    ctx.drawImage(video, 0, 0, c.width, c.height);
    if (checked('chk-mr')) {
      // The overlay letterboxes like the video: crop it to the video area.
      const ow = mrCanvas.width, oh = mrCanvas.height, a = c.width / c.height;
      const vw = Math.min(ow, oh * a), vh = vw / a;
      ctx.drawImage(mrCanvas, (ow - vw) / 2, (oh - vh) / 2, vw, vh, 0, 0, c.width, c.height);
    }
    requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
  return c.captureStream(30);
}

// ---------------------------------------------------------------- wiring

$('btn-connect').onclick = () => (connected ? disconnect() : connect());
$('btn-rgb').onclick = () => { rgbRetry = 0; startRgb(); };
$('btn-rgb-calib').onclick = () => { if (!calib) startCalibration(); };
($('sel-rgb') as HTMLSelectElement).onchange = e => startRgb((e.target as HTMLSelectElement).value);
$('btn-recenter').onclick = recenter;
$('btn-level').onclick = () => toDevice({ type: 'level' });
$('btn-clear-trail').onclick = () => preview.clearTrail();
$('btn-immersive').onclick = () => (glassesView || !immersiveEl.hidden ? closeImmersive() : openImmersive());
$('btn-exit').onclick = closeImmersive;
$('btn-sbs').onclick = () => setSbs(!pageView.sbs);
$('btn-im-recenter').onclick = recenter;
$('btn-fs').onclick = () => immersiveEl.requestFullscreen();
($('chk-pause') as HTMLInputElement).onchange = e => { paused = (e.target as HTMLInputElement).checked; };
for (const id of ['chk-6dof', 'chk-depth']) $(id).addEventListener('change', syncVisionOptions);

let csvOn = false;
$('btn-imu-rec').onclick = () => {
  csvOn = !csvOn;
  toDevice({ type: 'csv', on: csvOn });
  $('btn-imu-rec').textContent = csvOn ? 'Stop CSV' : 'CSV';
  $('btn-imu-rec').classList.toggle('active', csvOn);
};

$('btn-snap').onclick = async () => {
  if (!video.videoWidth) return;
  const b = await snapshot(video);
  if (b) download(b, `rgb-${stamp()}.png`);
};
$('btn-rec').onclick = async () => {
  if (!rgbStream) return;
  if (!recorder.recording) {
    recorder.start(checked('chk-mr') ? mixedStream() : rgbStream);
    $('btn-rec').classList.add('active');
  } else {
    download(await recorder.stop(), `rgb-${stamp()}.webm`);
    $('btn-rec').classList.remove('active');
  }
};
$('btn-stereo-snap').onclick = () => {
  for (const [id, side] of [['cam-left', 'gauche'], ['cam-right', 'droite']] as const)
    ($(id) as HTMLCanvasElement).toBlob(b => b && download(b, `stereo-${side}-${stamp()}.png`));
};

const expInput = $('rng-exp') as HTMLInputElement;
expInput.oninput = () => {
  ($('chk-ae') as HTMLInputElement).checked = false;
  setText('o-exp', expInput.value);
  toDevice({ type: 'exposure', value: +expInput.value });
};
const gainInput = $('rng-gain') as HTMLInputElement;
gainInput.oninput = () => {
  // Auto-exposure would immediately cancel a gain change.
  ($('chk-ae') as HTMLInputElement).checked = false;
  setText('o-gain', gainInput.value);
  toDevice({ type: 'gain', value: +gainInput.value });
};
$('chk-ae').addEventListener('change', () => toDevice({ type: 'autoExposure', on: checked('chk-ae') }));

window.addEventListener('keydown', (e: KeyboardEvent) => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  if (onTalkKey(e, true)) return;
  if (glassesView || !immersiveEl.hidden) onSceneKey(e);
  else if (e.key.toLowerCase() === 'r') recenter();
});

// ---------------------------------------------------------------- loops

const q = new THREE.Quaternion();
const p = new THREE.Vector3();
function render() {
  // While the glasses render, the dashboard's 3D preview and charts would only take
  // GPU time from them: skip them.
  if (immersiveEl.hidden && !glassesView) {
    preview.render(displayOrientation(q), displayPosition(p) ?? undefined);
    if (!paused) { gyroChart.draw(); accelChart.draw(); }
  }
  stereo.draw();
  feedRgbHands();
  drawHandOverlays();
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

setInterval(() => {
  if (pose) {
    setText('v-yaw', `${pose.euler.yaw.toFixed(1)}°`);
    setText('v-pitch', `${pose.euler.pitch.toFixed(1)}°`);
    setText('v-roll', `${pose.euler.roll.toFixed(1)}°`);
    setText('v-still', pose.stationary ? 'oui' : 'non');
    setText('v-gyro', fmt3(pose.gyro));
    setText('v-accel', fmt3(pose.accel, 2));
  }
  setText('v-pos', pose?.p ? `${fmt3(pose.p, 2)} m` : '3DoF');
  setText('v-track', !connected ? '-'
    : pose?.source === 'viture' ? (pose.tracking ? 'VITURE (SDK)' : 'VITURE, initialisation…')
    : !checked('chk-6dof') ? 'IMU (3DoF)'
    : pose?.tracking ? 'stéréo + IMU'
    : 'IMU, recherche de points');
  chip('chip-6dof', !connected || (pose?.source !== 'viture' && !checked('chk-6dof')) ? 'off' : pose?.tracking ? 'on' : 'wait');
  setText('depth-info', !connected ? '-' : !rig ? 'en attente de la calibration' : visionInfo || 'en attente des images');
}, 100);

function stamp() {
  return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
}

// Reconnect on load if the last session was connected (no prompt needed then).
if (store.get('autoConnect') === '1') {
  if (source() === 'viture') connect();
  else navigator.usb?.getDevices().then(d => { if (d.some(x => x.vendorId === VID && x.productId === PID_LUMA_ULTRA)) connect(); });
}

// Glasses plugged back in: reconnect without a click.
navigator.usb?.addEventListener('connect', (e: USBConnectionEvent) => {
  if (!connected && source() === 'usb' && store.get('autoConnect') === '1' && e.device.vendorId === VID && e.device.productId === PID_LUMA_ULTRA) {
    setTimeout(connect, 1500);
  }
});

// Development hook: lets the console (or an automated check) sample the live pose.
(window as unknown as { myviture: unknown }).myviture = {
  pose: () => pose, hands: () => hands, rgbCamera: () => rgbCamera, depth: () => depthFrame, head: () => headMatrix(displayOrientation(new THREE.Quaternion(), 0), displayPosition(new THREE.Vector3())).toArray(),
  timing: () => ({ glasses: glassesView?.timing, page: pageView.timing, mr: mrView.timing }),
};
