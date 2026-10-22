// Hand tracking with MediaPipe Hand Landmarker (GPU, in the browser).
//
// Preferred source: the two tracking cameras. MediaPipe runs on each image, the
// landmarks are undistorted with the factory calibration and triangulated, which
// gives metric 3D joints in the head frame. When a hand is seen by one camera only
// (or with the RGB camera), the depth comes from MediaPipe's metric hand shape
// fitted to the image landmarks.

import { FilesetResolver, HandLandmarker, type HandLandmarkerResult, type ImageSource, type NormalizedLandmark, type Landmark } from '@mediapipe/tasks-vision';
import { TrackingCamera, triangulate, type Vec3 } from './stereoGeometry';
import { PointSmoother } from './smoothing';
import { cameraToHead, type RgbCamera } from '../rgb/rgbCalib';

const WASM = '/mediapipe/wasm';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task';

export const TIP = { thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20 } as const;
export const BONES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

const PINCH_ON = 0.025; // m between thumb and index tips
const PINCH_OFF = 0.04;
const MAX_STEREO_GAP = 0.03; // m, mean ray gap accepted for a left/right match

export type View = 'rgb' | 'left' | 'right';

export interface Hand {
  /** from the wearer's point of view */
  side: 'left' | 'right';
  score: number;
  /** normalised image landmarks (0..1) per camera that saw the hand */
  views: Partial<Record<View, { x: number; y: number }[]>>;
  /** metres: head frame (x right, y up, z back), or world frame once given a toWorld */
  joints: Vec3[];
  /** absolute ms (performance.timeOrigin based) when the image arrived */
  stamp?: number;
  depth: 'stéréo' | 'mono';
  pinching: boolean;
  /** metres between thumb and index tips */
  pinchDistance: number;
}

let fileset: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>> | null = null;
/** 'GPU' or 'CPU', whichever the last landmarker could use. */
export let landmarkerDelegate = '';

/** Which delegate to ask for first. GPU measured faster than WASM even with the depth
 * and the rendering sharing the integrated GPU (~20 vs ~5 detections/s). */
export let preferredDelegate: 'GPU' | 'CPU' = 'GPU';
export function setPreferredDelegate(d: 'GPU' | 'CPU') { preferredDelegate = d; }

async function createLandmarker(): Promise<HandLandmarker> {
  fileset ??= await FilesetResolver.forVisionTasks(WASM);
  const options = (delegate: 'GPU' | 'CPU') => ({
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: 'VIDEO' as const,
    numHands: 2,
    minHandDetectionConfidence: 0.45,
    minHandPresenceConfidence: 0.45,
    minTrackingConfidence: 0.45,
  });
  const order: ('GPU' | 'CPU')[] = preferredDelegate === 'GPU' ? ['GPU', 'CPU'] : ['CPU', 'GPU'];
  try {
    const lm = await HandLandmarker.createFromOptions(fileset, options(order[0]));
    landmarkerDelegate = order[0];
    return lm;
  } catch {
    landmarkerDelegate = order[1];
    return await HandLandmarker.createFromOptions(fileset, options(order[1]));
  }
}

class Rate {
  value = 0;
  private n = 0;
  private t0 = performance.now();
  tick() {
    this.n++;
    const now = performance.now();
    if (now - this.t0 > 1000) { this.value = (this.n * 1000) / (now - this.t0); this.n = 0; this.t0 = now; }
  }
}

/** Hysteresis on the thumb-index distance, per hand side. */
class PinchFilter {
  private state = new Map<string, boolean>();
  update(side: string, distance: number) {
    const was = this.state.get(side) ?? false;
    const now = was ? distance < PINCH_OFF : distance < PINCH_ON;
    this.state.set(side, now);
    return now;
  }
  keep(sides: string[]) {
    for (const k of [...this.state.keys()]) if (!sides.includes(k)) this.state.delete(k);
  }
}

/** One Euro smoothing of joints (m) and image landmarks, per hand side and view. */
class HandSmoothing {
  private joints = new Map<string, PointSmoother>();
  private images = new Map<string, PointSmoother>();

  /** toWorld: head-frame point -> world (head pose at image time), applied before smoothing. */
  apply(hands: Hand[], toWorld?: (v: Vec3) => Vec3): Hand[] {
    const t = performance.now() / 1000;
    for (const h of hands) {
      if (toWorld) h.joints = h.joints.map(toWorld);
      const js = this.get(this.joints, h.side, 2.5, 8.0).apply(h.joints.flat(), t);
      h.joints = h.joints.map((_, i) => [js[i * 3], js[i * 3 + 1], js[i * 3 + 2]]);
      for (const [view, pts] of Object.entries(h.views)) {
        const f = this.get(this.images, `${h.side}/${view}`, 2.5, 4.0).apply(pts!.flatMap(p => [p.x, p.y]), t);
        h.views[view as View] = pts!.map((_, i) => ({ x: f[i * 2], y: f[i * 2 + 1] }));
      }
    }
    // Forget hands that left the view so they do not glide in from their old position.
    const keys = new Set(hands.map(h => h.side));
    for (const k of [...this.joints.keys()]) if (!keys.has(k as Hand['side'])) this.joints.delete(k);
    for (const k of [...this.images.keys()]) if (!keys.has(k.split('/')[0] as Hand['side'])) this.images.delete(k);
    return hands;
  }

  private get(m: Map<string, PointSmoother>, key: string, minCutoff: number, beta: number) {
    let s = m.get(key);
    if (!s) m.set(key, (s = new PointSmoother(minCutoff, beta)));
    return s;
  }
}

// MediaPipe labels assume a mirrored selfie image; these cameras look away from the user.
const sideOf = (r: HandLandmarkerResult, i: number): Hand['side'] => (r.handedness[i]?.[0]?.categoryName === 'Left' ? 'right' : 'left');
const scoreOf = (r: HandLandmarkerResult, i: number) => r.handedness[i]?.[0]?.score ?? 0;
const pinchDist = (w: Landmark[]) => Math.hypot(w[4].x - w[8].x, w[4].y - w[8].y, w[4].z - w[8].z);

// ------------------------------------------------------------------ RGB camera

export class RgbHandTracker {
  private lm: HandLandmarker | null = null;
  private pinch = new PinchFilter();
  private smoothing = new HandSmoothing();
  private lastTime = -1;
  readonly rate = new Rate();
  hands: Hand[] = [];

  async init() { this.lm = await createLandmarker(); }
  get ready() { return !!this.lm; }

  /** image: a video frame (ImageBitmap in the hands worker); w, h: its size; cam: the RGB camera model. */
  detect(image: ImageSource, w: number, h: number, cam: RgbCamera, toWorld?: (v: Vec3) => Vec3): Hand[] {
    if (!this.lm) return this.hands;
    const now = performance.now();
    if (now <= this.lastTime) return this.hands;
    this.lastTime = now;
    const res = this.lm.detectForVideo(image, now);
    // The camera model may have been calibrated at another resolution.
    const s = w / cam.width;
    const f = cam.f * s, cx = cam.cx * s, cy = cam.cy * s;
    const out: Hand[] = [];
    res.landmarks.forEach((lm, i) => {
      const rays = lm.map(p => [(p.x * w - cx) / f, (p.y * h - cy) / f] as [number, number]);
      const t = fitTranslation(rays, res.worldLandmarks[i]);
      if (!t) return;
      const joints = res.worldLandmarks[i].map(p => cameraToHead(cam, [p.x + t[0], p.y + t[1], p.z + t[2]]));
      const side = sideOf(res, i), d = pinchDist(res.worldLandmarks[i]);
      out.push({ side, score: scoreOf(res, i), views: { rgb: lm }, joints, depth: 'mono', pinching: this.pinch.update(side, d), pinchDistance: d });
    });
    this.pinch.keep(out.map(o => o.side));
    this.rate.tick();
    return (this.hands = this.smoothing.apply(out, toWorld));
  }
}

// ------------------------------------------------------------------ stereo tracking cameras

interface MonoHand {
  side: Hand['side'];
  score: number;
  image: NormalizedLandmark[];
  world: Landmark[];
  /** per-landmark head-frame rays */
  rays: Vec3[];
  /** single-camera 3D estimate, head frame */
  joints: Vec3[] | null;
}

export class StereoHandTracker {
  private lm: [HandLandmarker, HandLandmarker] | null = null;
  private pinch = new PinchFilter();
  private smoothing = new HandSmoothing();
  readonly rate = new Rate();
  hands: Hand[] = [];

  async init() { this.lm = [await createLandmarker(), await createLandmarker()]; }
  get ready() { return !!this.lm; }

  detect(left: ImageSource, right: ImageSource, size: { w: number; h: number }, cams: [TrackingCamera, TrackingCamera], toWorld?: (v: Vec3) => Vec3): Hand[] {
    if (!this.lm) return this.hands;
    const t = performance.now();
    // With no hand tracked, MediaPipe runs its palm detector on every image: search
    // the left image only, and add the right one as soon as a hand shows up.
    const resL = this.lm[0].detectForVideo(left, t);
    const searchRight = resL.landmarks.length > 0 || this.hands.length > 0;
    const res = [resL, searchRight ? this.lm[1].detectForVideo(right, t) : { landmarks: [], worldLandmarks: [], handedness: [], handednesses: [] } as unknown as HandLandmarkerResult];
    const mono = res.map((r, c) => r.landmarks.map((lm, i) => this.monoHand(r, i, lm, cams[c], size)));

    const out: Hand[] = [];
    const usedRight = new Set<number>();
    for (const a of mono[0]) {
      // Best right-camera match for this hand: same side, smallest triangulation gap.
      let best: { j: number; joints: Vec3[]; gap: number } | null = null;
      mono[1].forEach((b, j) => {
        if (usedRight.has(j) || b.side !== a.side) return;
        const tri = this.triangulateHand(a, b, cams);
        if (tri && (!best || tri.gap < best.gap)) best = { j, ...tri };
      });
      const match = best as { j: number; joints: Vec3[]; gap: number } | null;
      if (match && match.gap < MAX_STEREO_GAP) {
        usedRight.add(match.j);
        out.push(this.toHand(a, match.joints, 'stéréo', { left: a.image, right: mono[1][match.j].image }));
      } else if (a.joints) {
        out.push(this.toHand(a, a.joints, 'mono', { left: a.image }));
      }
    }
    mono[1].forEach((b, j) => {
      if (!usedRight.has(j) && b.joints && !out.some(h => h.side === b.side)) out.push(this.toHand(b, b.joints, 'mono', { right: b.image }));
    });
    // One hand per side: keep the stereo one if both cameras produced a candidate.
    const bySide = new Map<string, Hand>();
    for (const h of out) {
      const prev = bySide.get(h.side);
      if (!prev || (prev.depth === 'mono' && h.depth === 'stéréo')) bySide.set(h.side, h);
    }
    this.hands = this.smoothing.apply([...bySide.values()], toWorld);
    this.pinch.keep(this.hands.map(h => h.side));
    this.rate.tick();
    return this.hands;
  }

  private monoHand(r: HandLandmarkerResult, i: number, lm: NormalizedLandmark[], cam: TrackingCamera, size: { w: number; h: number }): MonoHand {
    const undist = lm.map(p => cam.undistort(p.x * size.w, p.y * size.h));
    const tr = fitTranslation(undist, r.worldLandmarks[i]);
    const joints = tr ? r.worldLandmarks[i].map(p => cam.toHead([p.x + tr[0], p.y + tr[1], p.z + tr[2]])) : null;
    return { side: sideOf(r, i), score: scoreOf(r, i), image: lm, world: r.worldLandmarks[i], rays: lm.map(p => cam.ray(p.x * size.w, p.y * size.h)), joints };
  }

  private triangulateHand(a: MonoHand, b: MonoHand, cams: [TrackingCamera, TrackingCamera]) {
    const joints: Vec3[] = [];
    let gap = 0;
    for (let k = 0; k < 21; k++) {
      const tri = triangulate(cams[0].origin, a.rays[k], cams[1].origin, b.rays[k]);
      if (!tri || tri.s <= 0.05 || tri.t <= 0.05) return null;
      joints.push(tri.point);
      gap += tri.gap;
    }
    return { joints, gap: gap / 21 };
  }

  private toHand(m: MonoHand, joints: Vec3[], depth: Hand['depth'], views: Hand['views']): Hand {
    // Pinch from the metric MediaPipe shape: steadier than triangulated fingertips.
    const d = pinchDist(m.world);
    return { side: m.side, score: m.score, views, joints, depth, pinching: this.pinch.update(m.side, d), pinchDistance: d };
  }
}

/**
 * Translation T so that MediaPipe's metric hand (world landmarks, camera-aligned)
 * plus T projects onto the undistorted rays (x, y, 1). Linear least squares on
 *   (Wx + Tx) - x (Wz + Tz) = 0,  (Wy + Ty) - y (Wz + Tz) = 0.
 */
function fitTranslation(rays: [number, number][], world: Landmark[]): Vec3 | null {
  const A = new Array(9).fill(0), b = [0, 0, 0];
  for (let i = 0; i < rays.length; i++) {
    const [x, y] = rays[i], W = world[i];
    const r1 = [1, 0, -x], c1 = x * W.z - W.x;
    const r2 = [0, 1, -y], c2 = y * W.z - W.y;
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) A[r * 3 + c] += r1[r] * r1[c] + r2[r] * r2[c];
      b[r] += r1[r] * c1 + r2[r] * c2;
    }
  }
  const det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
  if (Math.abs(det) < 1e-12) return null;
  const inv = [
    A[4] * A[8] - A[5] * A[7], A[2] * A[7] - A[1] * A[8], A[1] * A[5] - A[2] * A[4],
    A[5] * A[6] - A[3] * A[8], A[0] * A[8] - A[2] * A[6], A[2] * A[3] - A[0] * A[5],
    A[3] * A[7] - A[4] * A[6], A[1] * A[6] - A[0] * A[7], A[0] * A[4] - A[1] * A[3],
  ].map(v => v / det);
  const T: Vec3 = [
    inv[0] * b[0] + inv[1] * b[1] + inv[2] * b[2],
    inv[3] * b[0] + inv[4] * b[1] + inv[5] * b[2],
    inv[6] * b[0] + inv[7] * b[1] + inv[8] * b[2],
  ];
  return T[2] > 0.05 && T[2] < 3 ? T : null;
}
