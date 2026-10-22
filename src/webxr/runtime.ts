// A WebXR runtime for the glasses, in the page: navigator.xr, XRSession, XRWebGLLayer,
// XRFrame, XRView, XRRigidTransform and tracked hands (XRHand), backed by the dashboard's
// head pose, display calibration and hand tracking (received on a BroadcastChannel).
//
// Chrome has no WebXR device for these glasses, so a page importing this module before
// its own code gets one: an 'immersive-ar' session (additive blend: see-through optics)
// rendered side by side into the page's canvas, which fills the glasses' 3D screen.
// Written against what three.js's WebXRManager and WebXRController use; other engines
// using the same core API should work too. Anchors (persistent ones too, kept in the
// room map) and hit test (against the dashboard's depth map) are supported. Not
// implemented: layers, depth sensing, DOM overlay.
//
// Two ways in: webxr.html imports it (feed on the dashboard's BroadcastChannel, same
// origin), or the browser extension (extension/) injects it into any site; the feed
// then comes through the extension, which also moves the tab onto the glasses' screen
// when a session starts.

import * as THREE from 'three';
import type { PoseState, V3 } from '../workers/messages';

export const XR_CHANNEL = 'myviture-xr';
/** Extension -> dashboard requests (glasses display mode). */
export const XR_CONTROL_CHANNEL = 'myviture-xr-control';

/** What the dashboard broadcasts at its pose rate. */
export interface XrFeed {
  pose: PoseState;
  eyes: { left: V3; right: V3; rotation: [number, number, number, number]; vfov: number };
  /** tracked hands, scene (local-floor) coordinates, MediaPipe's 21 joints */
  hands: { side: 'left' | 'right'; joints: V3[]; pinching: boolean }[];
  predictMs: number;
  eyeHeight: number;
  /** room map -> tracking frame, once the room is recognised (persistent anchors) */
  map?: { yaw: number; t: V3 } | null;
  /** answers to the hit-test rays this page sent, scene coordinates */
  hits?: Record<string, { p: V3; n: V3 } | null>;
}

/** A hit-test ray sent to the dashboard (scene coordinates, unit direction). */
export interface XrRay { id: string; o: V3; d: V3 }

// ------------------------------------------------------------------ geometry types

class RigidTransform {
  readonly position: DOMPointReadOnly;
  readonly orientation: DOMPointReadOnly;
  readonly matrix: Float32Array;
  private _inverse: RigidTransform | null = null;

  constructor(position?: DOMPointInit, orientation?: DOMPointInit) {
    const p = new THREE.Vector3(position?.x ?? 0, position?.y ?? 0, position?.z ?? 0);
    const q = new THREE.Quaternion(orientation?.x ?? 0, orientation?.y ?? 0, orientation?.z ?? 0, orientation?.w ?? 1).normalize();
    this.position = new DOMPointReadOnly(p.x, p.y, p.z, 1);
    this.orientation = new DOMPointReadOnly(q.x, q.y, q.z, q.w);
    this.matrix = new Float32Array(new THREE.Matrix4().compose(p, q, new THREE.Vector3(1, 1, 1)).elements);
  }

  static fromMatrix(m: THREE.Matrix4) {
    const p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
    m.decompose(p, q, s);
    return new RigidTransform(p, q);
  }

  get inverse(): RigidTransform {
    return (this._inverse ??= RigidTransform.fromMatrix(new THREE.Matrix4().fromArray(this.matrix).invert()));
  }
}

class Space extends EventTarget {}

class ReferenceSpace extends Space {
  /** reference space -> scene (local-floor); a function for spaces that move (viewer) */
  constructor(private readonly at: THREE.Matrix4 | (() => THREE.Matrix4)) { super(); }
  get toScene(): THREE.Matrix4 { return typeof this.at === 'function' ? this.at() : this.at.clone(); }
  getOffsetReferenceSpace(t: RigidTransform) {
    const offset = new THREE.Matrix4().fromArray(t.matrix);
    return new ReferenceSpace(() => this.toScene.multiply(offset));
  }
}

/** XRRay: origin and direction, from points or from a transform (its -Z axis). */
class Ray {
  readonly origin: DOMPointReadOnly;
  readonly direction: DOMPointReadOnly;
  readonly matrix: Float32Array;
  constructor(a?: DOMPointInit | RigidTransform, b?: DOMPointInit) {
    let o = new THREE.Vector3(), d = new THREE.Vector3(0, 0, -1);
    if (a instanceof RigidTransform) {
      const m = new THREE.Matrix4().fromArray(a.matrix);
      o.setFromMatrixPosition(m);
      d.transformDirection(m);
    } else {
      if (a) o = new THREE.Vector3(a.x ?? 0, a.y ?? 0, a.z ?? 0);
      if (b) d = new THREE.Vector3(b.x ?? 0, b.y ?? 0, b.z ?? -1).normalize();
    }
    this.origin = new DOMPointReadOnly(o.x, o.y, o.z, 1);
    this.direction = new DOMPointReadOnly(d.x, d.y, d.z, 0);
    // Matrix taking (0, 0, 0) to the origin and -Z to the direction.
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, -1), d);
    this.matrix = new Float32Array(new THREE.Matrix4().compose(o, q, new THREE.Vector3(1, 1, 1)).elements);
  }
}

/** A space whose scene pose is read every frame (hand joints, target ray, grip). */
class LiveSpace extends Space {
  constructor(readonly sceneMatrix: () => THREE.Matrix4 | null, readonly jointName?: string) { super(); }
}

// ------------------------------------------------------------------ hands

// WebXR's 25 joints from MediaPipe's 21: metacarpal bases are interpolated from the wrist.
const FINGERS: [string, number][] = [['index-finger', 5], ['middle-finger', 9], ['ring-finger', 13], ['pinky-finger', 17]];
const JOINTS: [string, (j: THREE.Vector3[]) => THREE.Vector3][] = [
  ['wrist', j => j[0]],
  ['thumb-metacarpal', j => j[1]],
  ['thumb-phalanx-proximal', j => j[2]],
  ['thumb-phalanx-distal', j => j[3]],
  ['thumb-tip', j => j[4]],
  ...FINGERS.flatMap(([name, b]): [string, (j: THREE.Vector3[]) => THREE.Vector3][] => [
    [`${name}-metacarpal`, j => j[0].clone().lerp(j[b], 0.3)],
    [`${name}-phalanx-proximal`, j => j[b]],
    [`${name}-phalanx-intermediate`, j => j[b + 1]],
    [`${name}-phalanx-distal`, j => j[b + 2]],
    [`${name}-tip`, j => j[b + 3]],
  ]),
];

class HandInput {
  readonly handedness: 'left' | 'right';
  readonly targetRayMode = 'tracked-pointer';
  readonly profiles = ['generic-hand'];
  readonly gamepad = null;
  readonly hand: Map<string, LiveSpace>;
  readonly targetRaySpace: LiveSpace;
  readonly gripSpace: LiveSpace;
  joints: THREE.Vector3[] = [];
  pinching = false;

  constructor(side: 'left' | 'right', private head: () => THREE.Matrix4) {
    this.handedness = side;
    const at = (p: THREE.Vector3) => new THREE.Matrix4().setPosition(p);
    this.hand = new Map(JOINTS.map(([name, f]) => [name, new LiveSpace(() => (this.joints.length ? at(f(this.joints)) : null), name)]));
    this.gripSpace = new LiveSpace(() => (this.joints.length ? at(this.joints[0]) : null));
    // Ray from the head through the pinch point.
    this.targetRaySpace = new LiveSpace(() => {
      if (!this.joints.length) return null;
      const pinch = this.joints[4].clone().lerp(this.joints[8], 0.5);
      const eye = new THREE.Vector3().setFromMatrixPosition(this.head());
      const m = new THREE.Matrix4().lookAt(eye, pinch, new THREE.Vector3(0, 1, 0));
      return m.setPosition(pinch);
    });
  }
}

// ------------------------------------------------------------------ anchors
//
// An anchor is kept in the tracking frame (VITURE's, which does not move on recenter),
// and in the room map once it is recognised: a persistent anchor comes back in the map,
// so in the same place of the room after a restart.

/** Tracking frame -> scene: scene = yaw * (x - origin) + eye height. */
function trackingToScene(): THREE.Matrix4 | null {
  const f = feed?.pose.frame;
  if (!f) return null;
  return new THREE.Matrix4().makeTranslation(0, feed!.eyeHeight, 0)
    .multiply(new THREE.Matrix4().makeRotationFromQuaternion(new THREE.Quaternion(...f.yaw)))
    .multiply(new THREE.Matrix4().makeTranslation(-f.origin[0], -f.origin[1], -f.origin[2]));
}

/** Room map -> tracking frame, null until the room is recognised. */
function mapToTracking(): THREE.Matrix4 | null {
  const T = feed?.map;
  return T ? new THREE.Matrix4().makeTranslation(...T.t).multiply(new THREE.Matrix4().makeRotationY(T.yaw)) : null;
}

const ANCHOR_STORE = 'myviture-xr-anchors';
function anchorStore(): Record<string, number[]> {
  try { return JSON.parse(localStorage.getItem(ANCHOR_STORE) ?? '{}'); } catch { return {}; }
}
function writeAnchorStore(s: Record<string, number[]>) {
  try { localStorage.setItem(ANCHOR_STORE, JSON.stringify(s)); } catch { /* storage blocked */ }
}

class Anchor {
  readonly anchorSpace: LiveSpace;
  deleted = false;
  constructor(private session: Session, private tracking: THREE.Matrix4 | null, private map: THREE.Matrix4 | null) {
    this.anchorSpace = new LiveSpace(() => this.sceneMatrix());
  }

  /** Anchor -> scene now; null while it cannot be placed (room not recognised yet). */
  sceneMatrix(): THREE.Matrix4 | null {
    const toScene = trackingToScene();
    if (!toScene || this.deleted) return null;
    const M = mapToTracking();
    if (!this.tracking && this.map && M) this.tracking = M.clone().multiply(this.map);
    if (!this.map && this.tracking && M) this.map = M.clone().invert().multiply(this.tracking);
    return this.tracking ? toScene.multiply(this.tracking) : null;
  }

  delete() {
    this.deleted = true;
    this.session.anchors.delete(this);
  }

  async requestPersistentHandle(): Promise<string> {
    this.sceneMatrix();
    if (!this.map) throw new DOMException('myViture : la pièce n’est pas encore reconnue', 'InvalidStateError');
    const id = crypto.randomUUID();
    const store = anchorStore();
    store[id] = [...this.map.elements];
    writeAnchorStore(store);
    return id;
  }

  /** Restored persistent anchor: placed once the room is recognised. */
  static fromMap(session: Session, elements: number[]) {
    return new Anchor(session, null, new THREE.Matrix4().fromArray(elements));
  }
}

// ------------------------------------------------------------------ hit test
//
// The rays go to the dashboard, which walks them against its depth map; the answers
// come back in the feed, a frame or two later.

let hitIds = 0;

class HitTestSource {
  readonly id = `h${++hitIds}-${Math.random().toString(36).slice(2, 7)}`;
  constructor(private session: Session, readonly space: Space | null, readonly offsetRay: Ray, readonly transient: boolean) {}
  cancel() { this.session.hitSources.delete(this); }
}

class HitTestResult {
  constructor(private frame: Frame, private hit: { p: V3; n: V3 }, readonly inputSource?: HandInput) {}
  getPose(base: ReferenceSpace) {
    // Y along the surface normal, as the spec asks.
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...this.hit.n).normalize());
    const m = new THREE.Matrix4().compose(new THREE.Vector3(...this.hit.p), q, new THREE.Vector3(1, 1, 1));
    return { transform: RigidTransform.fromMatrix(base.toScene.invert().multiply(m)), emulatedPosition: false };
  }
  createAnchor() {
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...this.hit.n).normalize());
    return this.frame.createAnchor(new RigidTransform(new THREE.Vector3(...this.hit.p), q), new ReferenceSpace(new THREE.Matrix4()));
  }
}

/** Sends this page's hit-test rays to the dashboard (through the extension, or same origin). */
let control: BroadcastChannel | null = null;
function sendRays(rays: XrRay[]) {
  if (external) toExtension('rays', { rays });
  else (control ??= new BroadcastChannel(XR_CONTROL_CHANNEL)).postMessage({ type: 'rays', rays });
}

// ------------------------------------------------------------------ session

let feed: XrFeed | null = null;
let feedAt = -Infinity;
function onFeed(f: XrFeed) { feed = f; feedAt = performance.now(); }
const fresh = () => performance.now() - feedAt < 1000;

/** Set by installRuntime({ external: true }): fed and placed by the extension. */
let external = false;
/** Page <-> extension relay messages, tagged so the site's own messages are left alone. */
const TAG = '__myvitureXr';
const toExtension = (type: string, data: object = {}) => window.postMessage({ [TAG]: type, ...data }, '*');
let onPlaced: ((r: { ok: boolean; reason?: string }) => void) | null = null;

function listen() {
  if (!external) {
    new BroadcastChannel(XR_CHANNEL).onmessage = e => onFeed(e.data as XrFeed);
    return;
  }
  window.addEventListener('message', e => {
    if (e.source !== window || typeof e.data?.[TAG] !== 'string') return;
    if (e.data[TAG] === 'feed') onFeed(e.data.feed);
    else if (e.data[TAG] === 'placed') { onPlaced?.(e.data); onPlaced = null; }
  });
}

/** Asks the extension for the feed and waits for it (the dashboard may not be open). */
async function waitFeed(ms: number) {
  if (fresh()) return true;
  toExtension('hello');
  const t0 = performance.now();
  while (performance.now() - t0 < ms) {
    await new Promise(r => setTimeout(r, 50));
    if (fresh()) return true;
  }
  return false;
}

const absNow = () => performance.timeOrigin + performance.now();
/** ?preview: head held level and facing forward, to look at a WebXR page on a monitor. */
const PREVIEW = new URLSearchParams(location.search).has('preview');

/** Head -> scene at the expected display time (gyro extrapolation, as the dashboard does). */
function headMatrix(): THREE.Matrix4 {
  const m = new THREE.Matrix4();
  if (!feed) return m.setPosition(0, 1.6, 0);
  const { pose } = feed;
  const q = new THREE.Quaternion(...pose.q);
  const w = new THREE.Vector3(...pose.w);
  const dt = Math.min(0.1, Math.max(0, (absNow() - pose.stamp + feed.predictMs) / 1000));
  const a = w.length();
  if (a > 1e-6) q.multiply(new THREE.Quaternion().setFromAxisAngle(w.divideScalar(a), a * dt));
  if (PREVIEW) q.identity();
  const p = new THREE.Vector3(0, feed.eyeHeight, 0);
  if (pose.p) p.add(new THREE.Vector3(...pose.p));
  return m.compose(p, q, new THREE.Vector3(1, 1, 1));
}

type FrameCallback = (time: number, frame: Frame) => void;

class Session extends EventTarget {
  readonly environmentBlendMode = 'additive';
  readonly interactionMode = 'world-space';
  readonly visibilityState = 'visible';
  readonly enabledFeatures: string[];
  readonly inputSources: HandInput[] = [];
  renderState: { baseLayer: GlLayer | null; depthNear: number; depthFar: number; inlineVerticalFieldOfView: null; layers?: undefined } =
    { baseLayer: null, depthNear: 0.1, depthFar: 1000, inlineVerticalFieldOfView: null };
  private callbacks = new Map<number, FrameCallback>();
  private nextId = 1;
  private rafId = 0;
  private ended = false;
  private hands: Record<'left' | 'right', HandInput>;
  private lastFrame: Frame | null = null;
  onend: ((e: Event) => void) | null = null;
  /** Undoes what the session changed on the page. */
  readonly cleanups: (() => void)[] = [];
  readonly anchors = new Set<Anchor>();
  readonly hitSources = new Set<HitTestSource>();
  private lastRays = 0;

  constructor(readonly mode: string, features: string[]) {
    super();
    this.enabledFeatures = features;
    this.hands = { left: new HandInput('left', headMatrix), right: new HandInput('right', headMatrix) };
    this.tick = this.tick.bind(this);
    this.rafId = requestAnimationFrame(this.tick);
  }

  updateRenderState(s: Partial<Session['renderState']>) {
    Object.assign(this.renderState, s);
  }

  async requestReferenceSpace(type: string) {
    const h = feed?.eyeHeight ?? 1.6;
    if (type === 'local-floor' || type === 'bounded-floor' || type === 'unbounded') return new ReferenceSpace(new THREE.Matrix4());
    if (type === 'local') return new ReferenceSpace(new THREE.Matrix4().makeTranslation(0, h, 0));
    if (type === 'viewer') return new ReferenceSpace(headMatrix);
    throw new DOMException(`reference space ${type} not supported`, 'NotSupportedError');
  }

  requestAnimationFrame(cb: FrameCallback) {
    const id = this.nextId++;
    this.callbacks.set(id, cb);
    return id;
  }

  cancelAnimationFrame(id: number) {
    this.callbacks.delete(id);
  }

  async requestHitTestSource(o: { space: Space; offsetRay?: Ray }) {
    const s = new HitTestSource(this, o.space, o.offsetRay ?? new Ray(), false);
    this.hitSources.add(s);
    return s;
  }

  async requestHitTestSourceForTransientInput(o: { profile?: string; offsetRay?: Ray }) {
    const s = new HitTestSource(this, null, o.offsetRay ?? new Ray(), true);
    this.hitSources.add(s);
    return s;
  }

  get persistentAnchors(): string[] { return Object.keys(anchorStore()); }

  async restorePersistentAnchor(uuid: string) {
    const m = anchorStore()[uuid];
    if (!m) throw new DOMException(`anchor ${uuid} unknown`, 'InvalidStateError');
    const a = Anchor.fromMap(this, m);
    this.anchors.add(a);
    return a;
  }

  async deletePersistentAnchor(uuid: string) {
    const store = anchorStore();
    delete store[uuid];
    writeAnchorStore(store);
  }

  /** Scene rays of every hit-test source now (the viewer's follow the head). */
  rays(): XrRay[] {
    const out: XrRay[] = [];
    const push = (id: string, base: THREE.Matrix4 | null, ray: Ray) => {
      if (!base) return;
      const m = base.multiply(new THREE.Matrix4().fromArray(ray.matrix));
      out.push({ id, o: new THREE.Vector3().setFromMatrixPosition(m).toArray(), d: new THREE.Vector3(0, 0, -1).transformDirection(m).toArray() });
    };
    for (const s of this.hitSources) {
      if (s.transient) for (const h of this.inputSources) push(`${s.id}:${h.handedness}`, h.targetRaySpace.sceneMatrix(), s.offsetRay);
      else push(s.id, s.space instanceof LiveSpace ? s.space.sceneMatrix() : s.space instanceof ReferenceSpace ? s.space.toScene : null, s.offsetRay);
    }
    return out;
  }

  async end() {
    if (this.ended) return;
    this.ended = true;
    cancelAnimationFrame(this.rafId);
    for (const f of this.cleanups.splice(0)) f();
    const e = new Event('end');
    Object.assign(e, { session: this });
    this.dispatchEvent(e);
    this.onend?.(e);
  }

  // Frame rate of the window's display.
  get frameRate() { return undefined; }
  updateTargetFrameRate() { return Promise.resolve(); }

  private tick(time: number) {
    if (this.ended) return;
    this.rafId = requestAnimationFrame(this.tick);
    this.syncHands();
    // Hit-test rays at 30 Hz: the depth map itself changes at ~7 Hz.
    if (this.hitSources.size && time - this.lastRays > 33) {
      this.lastRays = time;
      sendRays(this.rays());
    }
    const frame = new Frame(this, time);
    this.lastFrame = frame;
    const cbs = [...this.callbacks.values()];
    this.callbacks.clear();
    frame.active = true;
    for (const cb of cbs) cb(time, frame);
    frame.active = false;
  }

  private syncHands() {
    const seen = new Set<string>();
    for (const h of feed?.hands ?? []) {
      const input = this.hands[h.side];
      input.joints = h.joints.map(j => new THREE.Vector3(...j));
      seen.add(h.side);
      if (!this.inputSources.includes(input)) {
        this.inputSources.push(input);
        this.emit('inputsourceschange', { added: [input], removed: [] });
      }
      if (h.pinching !== input.pinching) {
        input.pinching = h.pinching;
        this.emit(h.pinching ? 'selectstart' : 'selectend', { inputSource: input, frame: this.lastFrame });
        if (!h.pinching) this.emit('select', { inputSource: input, frame: this.lastFrame });
      }
    }
    for (const input of [...this.inputSources]) {
      if (seen.has(input.handedness)) continue;
      input.joints = [];
      this.inputSources.splice(this.inputSources.indexOf(input), 1);
      this.emit('inputsourceschange', { added: [], removed: [input] });
    }
  }

  private emit(type: string, props: object) {
    const e = new Event(type);
    Object.assign(e, { session: this }, props);
    this.dispatchEvent(e);
  }
}

class View {
  constructor(readonly eye: 'left' | 'right', readonly projectionMatrix: Float32Array, readonly transform: RigidTransform) {}
  readonly recommendedViewportScale = null;
  requestViewportScale() {}
}

class Frame {
  active = false;
  readonly predictedDisplayTime: number;
  constructor(readonly session: Session, time: number) { this.predictedDisplayTime = time + (feed?.predictMs ?? 0); }

  getViewerPose(space: ReferenceSpace) {
    const head = headMatrix();
    const toSpace = space.toScene.clone().invert();
    const views: View[] = [];
    const layer = this.session.renderState.baseLayer;
    const aspect = layer ? layer.framebufferWidth / 2 / layer.framebufferHeight : 16 / 10;
    const { depthNear, depthFar } = this.session.renderState;
    const eyes = feed?.eyes;
    for (const eye of ['left', 'right'] as const) {
      const e = eyes ? new THREE.Vector3(...eyes[eye]) : new THREE.Vector3(eye === 'left' ? -0.0315 : 0.0315, 0, 0);
      const r = eyes ? new THREE.Quaternion(...eyes.rotation) : new THREE.Quaternion();
      const m = toSpace.clone().multiply(head).multiply(new THREE.Matrix4().compose(e, r, new THREE.Vector3(1, 1, 1)));
      const cam = new THREE.PerspectiveCamera(eyes?.vfov ?? 28, aspect, depthNear, depthFar);
      cam.updateProjectionMatrix();
      views.push(new View(eye, new Float32Array(cam.projectionMatrix.elements), RigidTransform.fromMatrix(m)));
    }
    return { transform: RigidTransform.fromMatrix(toSpace.multiply(head)), views, emulatedPosition: !feed?.pose.p };
  }

  getPose(space: Space, base: ReferenceSpace) {
    const m = space instanceof LiveSpace ? space.sceneMatrix() : space instanceof ReferenceSpace ? space.toScene : null;
    if (!m) return null;
    return { transform: RigidTransform.fromMatrix(base.toScene.clone().invert().multiply(m)), emulatedPosition: false };
  }

  getJointPose(joint: LiveSpace, base: ReferenceSpace) {
    const pose = this.getPose(joint, base);
    if (!pose) return null;
    return { ...pose, radius: joint.jointName?.endsWith('tip') ? 0.006 : 0.009 };
  }

  get trackedAnchors(): Set<Anchor> {
    return new Set([...this.session.anchors].filter(a => a.sceneMatrix()));
  }

  createAnchor(pose: RigidTransform, space: Space): Promise<Anchor> {
    const base = space instanceof LiveSpace ? space.sceneMatrix() : space instanceof ReferenceSpace ? space.toScene : null;
    const toScene = trackingToScene();
    if (!base || !toScene) return Promise.reject(new DOMException('anchor: no pose for this space now', 'InvalidStateError'));
    const tracking = toScene.invert().multiply(base).multiply(new THREE.Matrix4().fromArray(pose.matrix));
    const a = new Anchor(this.session, tracking, null);
    this.session.anchors.add(a);
    return Promise.resolve(a);
  }

  getHitTestResults(source: HitTestSource): HitTestResult[] {
    const h = feed?.hits?.[source.id];
    return h ? [new HitTestResult(this, h)] : [];
  }

  getHitTestResultsForTransientInput(source: HitTestSource) {
    return this.session.inputSources.map(input => {
      const h = feed?.hits?.[`${source.id}:${input.handedness}`];
      return { inputSource: input, results: h ? [new HitTestResult(this, h, input)] : [] };
    });
  }

  fillPoses(spaces: LiveSpace[], base: ReferenceSpace, out: Float32Array) {
    spaces.forEach((s, i) => { const p = this.getPose(s, base); if (p) out.set(p.transform.matrix, i * 16); });
    return true;
  }

  fillJointRadii(joints: LiveSpace[], out: Float32Array) {
    joints.forEach((j, i) => { out[i] = j.jointName?.endsWith('tip') ? 0.006 : 0.009; });
    return true;
  }
}

/** The default framebuffer of the page's canvas, split in two eye viewports. */
class GlLayer {
  readonly framebuffer = null;
  readonly antialias: boolean;
  readonly ignoreDepthValues = false;
  fixedFoveation = 0;

  constructor(session: Session, readonly context: WebGL2RenderingContext | WebGLRenderingContext, init: { antialias?: boolean } = {}) {
    this.antialias = init.antialias ?? true;
    const c = context.canvas as HTMLCanvasElement;
    // The canvas fills the window (the glasses' screen) during the session, over a black
    // backdrop: black is see-through on the glasses, the rest of the page would not be.
    const saved = c.style.cssText;
    const backdrop = document.createElement('div');
    backdrop.style.cssText = 'position:fixed;inset:0;background:#000;z-index:2147483646';
    document.body.appendChild(backdrop);
    Object.assign(c.style, { position: 'fixed', left: '0', top: '0', width: '100vw', height: '100vh', zIndex: '2147483647' });
    session.cleanups.push(() => { c.style.cssText = saved; backdrop.remove(); });
  }

  get framebufferWidth() { return Math.round(innerWidth * devicePixelRatio); }
  get framebufferHeight() { return Math.round(innerHeight * devicePixelRatio); }

  getViewport(view: View) {
    const w = this.framebufferWidth / 2;
    return { x: view.eye === 'left' ? 0 : w, y: 0, width: w, height: this.framebufferHeight };
  }

  static getNativeFramebufferScaleFactor() { return 1; }
}

// ------------------------------------------------------------------ install

class XrSystem extends EventTarget {
  private active: Session | null = null;
  async isSessionSupported(mode: string) {
    if (mode === 'inline') return true;
    if (mode !== 'immersive-ar' && mode !== 'immersive-vr') return false;
    // Through the extension, only while the dashboard is tracking the glasses.
    return !external || waitFeed(1500);
  }
  async requestSession(mode: string, init: { requiredFeatures?: string[]; optionalFeatures?: string[] } = {}) {
    const supported = new Set(['local', 'local-floor', 'viewer', 'hand-tracking', 'bounded-floor', 'unbounded', 'anchors', 'hit-test']);
    for (const f of init.requiredFeatures ?? []) if (!supported.has(f)) throw new DOMException(`feature ${f} not supported`, 'NotSupportedError');
    if (this.active) throw new DOMException('a session is already active', 'InvalidStateError');
    const features = ['viewer', 'local', ...(init.requiredFeatures ?? []), ...(init.optionalFeatures ?? []).filter(f => supported.has(f))];
    if (external && mode !== 'inline') await this.placeOnGlasses();
    const s = new Session(mode, [...new Set(features)]);
    this.active = s;
    s.addEventListener('end', () => { this.active = null; });
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') s.end(); };
    addEventListener('keydown', onKey);
    s.cleanups.push(() => removeEventListener('keydown', onKey));
    if (external && mode !== 'inline') s.cleanups.push(() => toExtension('session', { active: false }));
    return s;
  }

  /** The extension switches the glasses to 3D and moves this tab fullscreen onto them. */
  private async placeOnGlasses() {
    if (!(await waitFeed(1500))) throw new DOMException('myViture : ouvre le tableau de bord et connecte les lunettes', 'NotSupportedError');
    const r = await new Promise<{ ok: boolean; reason?: string }>(resolve => {
      onPlaced = resolve;
      toExtension('session', { active: true });
      setTimeout(() => resolve({ ok: false, reason: 'pas de réponse de l’extension' }), 10000);
    });
    if (!r.ok) throw new DOMException(`myViture : ${r.reason}`, 'NotSupportedError');
  }
}

export function installRuntime(opts: { external?: boolean } = {}) {
  external = !!opts.external;
  listen();
  const g = globalThis as Record<string, unknown>;
  Object.defineProperty(navigator, 'xr', { value: new XrSystem(), configurable: true });
  g.XRWebGLLayer = GlLayer;
  g.XRRigidTransform = RigidTransform;
  g.XRSession = Session;
  g.XRFrame = Frame;
  g.XRView = View;
  g.XRReferenceSpace = ReferenceSpace;
  g.XRSpace = Space;
  g.XRRay = Ray;
  g.XRAnchor = Anchor;
  g.XRHitTestSource = HitTestSource;
  g.XRHitTestResult = HitTestResult;
  // No layers API: engines then use XRWebGLLayer.
  delete g.XRWebGLBinding;
  for (const C of [globalThis.WebGL2RenderingContext, globalThis.WebGLRenderingContext]) {
    const proto = C?.prototype as { makeXRCompatible?: () => Promise<void> } | undefined;
    if (proto) proto.makeXRCompatible = () => Promise.resolve();
  }
}

/** Live dashboard feed, for diagnostics. */
export const currentFeed = () => feed;
