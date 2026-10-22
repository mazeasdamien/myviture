// Mixed-reality scene: one shared world, several views.
//
// XrWorld holds the content, the hands and the head pose, and is updated once per
// display frame. XrView renders it into a canvas, either per eye for the glasses
// (mono or side-by-side 3D) or from the RGB camera's viewpoint with a transparent
// background, to composite over the RGB video (first-person view on the dashboard).
//
// Everything is drawn bright: on see-through glasses black is transparent and dark
// colours barely show.

import * as THREE from 'three';
import type { Hand } from '../hands/handTracker';
import type { DepthFrame } from '../depth/stereoDepth';
import type { Vec3 } from '../hands/stereoGeometry';
import { HandsView } from './handsView';
import { Human } from './human';
import { EYE_HEIGHT, headMatrix } from './headPose';
import type { RgbCamera } from '../rgb/rgbCalib';

export { EYE_HEIGHT, headMatrix };

export interface EyeSetup {
  left: THREE.Vector3;
  right: THREE.Vector3;
  centre: THREE.Vector3;
  rotation: THREE.Quaternion;
  vfov: number;
}

export interface WorldSources {
  /** head -> world orientation to render (predicted) */
  orientation(out: THREE.Quaternion): THREE.Quaternion;
  /** metres, world frame; null when only 3DoF is available */
  position(out: THREE.Vector3): THREE.Vector3 | null;
  /** tracked hands, joints in the world frame */
  hands(): Hand[];
  eyes(): EyeSetup;
  depth(): DepthFrame | null;
  /** RGB camera model (pose in the head frame and intrinsics), null before the video starts */
  rgbCamera(): RgbCamera | null;
}

export interface WorldOptions {
  /** world-locked markers to check the display calibration */
  anchorTest: boolean;
  /** real surfaces hide virtual objects */
  occlusion: boolean;
  /** draw the depth mesh in colour */
  showDepth: boolean;
}

const MESH_STEP = 4;
const HALF_TURN_X = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);

export class XrWorld {
  readonly scene = new THREE.Scene();
  readonly head = new THREE.Object3D();
  /** per-eye cameras and the RGB camera, children of the head */
  readonly cams = {
    left: new THREE.PerspectiveCamera(), right: new THREE.PerspectiveCamera(),
    centre: new THREE.PerspectiveCamera(), rgb: new THREE.PerspectiveCamera(),
  };
  options: WorldOptions = { anchorTest: false, occlusion: false, showDepth: false };
  /** standing 2 m ahead, facing the start position */
  readonly human = new Human();
  private objects: THREE.Object3D[] = [];
  private handsView: HandsView;
  private anchor = new THREE.Group();
  private reticle: THREE.Object3D;
  private depthMesh: THREE.Mesh;
  private depthMats: { hidden: THREE.Material; visible: THREE.Material };
  private lastDepth: DepthFrame | null = null;
  private lastUpdate = -1;
  /** world-locked labels: objects found by the assistant, anchors */
  readonly labels = new THREE.Group();
  private message: { panel: THREE.Mesh; until: number } | null = null;
  /** landmarks of the environment memory (scene frame) */
  private mapPoints = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ color: 0x7fd1ff, size: 0.012, transparent: true, opacity: 0.7 }));
  private q = new THREE.Quaternion();
  private p = new THREE.Vector3();

  constructor(readonly src: WorldSources) {
    for (const cam of Object.values(this.cams)) { cam.near = 0.05; cam.far = 100; this.head.add(cam); }
    this.scene.add(this.head);
    this.build();
    this.handsView = new HandsView(this.scene, this.head, this.objects);
    this.reticle = makeReticle();
    this.cams.centre.add(this.reticle);

    // Depth occluder: written to the depth buffer before everything else.
    this.depthMats = {
      hidden: new THREE.MeshBasicMaterial({ colorWrite: false }),
      visible: new THREE.MeshNormalMaterial({ side: THREE.DoubleSide, transparent: true, opacity: 0.5 }),
    };
    this.depthMesh = new THREE.Mesh(new THREE.BufferGeometry(), this.depthMats.hidden);
    this.depthMesh.frustumCulled = false;
    this.head.add(this.depthMesh);
  }

  private build() {
    const s = this.scene;
    s.add(new THREE.HemisphereLight(0xffffff, 0x445066, 2.4));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(1, 3, 2);
    s.add(key);

    // Floating shapes on an arc within reach of a step, around eye level.
    const colors = [0xff7a45, 0x4c8dff, 0x2fd39a, 0xf0c53a, 0xff5d9e, 0x9b7bff];
    const geos = [
      new THREE.TorusKnotGeometry(0.05, 0.016, 96, 12), new THREE.IcosahedronGeometry(0.065, 0),
      new THREE.BoxGeometry(0.09, 0.09, 0.09), new THREE.OctahedronGeometry(0.07),
      new THREE.TorusGeometry(0.055, 0.02, 12, 32), new THREE.DodecahedronGeometry(0.065),
    ];
    colors.forEach((c, i) => {
      const a = THREE.MathUtils.degToRad(-55 + i * 22);
      const m = new THREE.Mesh(geos[i], glowing(c));
      m.position.set(Math.sin(a) * 1.1, EYE_HEIGHT - 0.1 + (i % 2) * 0.18, -Math.cos(a) * 1.1);
      s.add(m);
      this.objects.push(m);
    });

    // Small cubes within arm's reach, to grab directly with a pinch.
    const cube = new THREE.BoxGeometry(0.05, 0.05, 0.05);
    [-0.15, 0, 0.15].forEach((x, i) => {
      const m = new THREE.Mesh(cube, glowing(colors[i + 1]));
      m.position.set(x, EYE_HEIGHT - 0.28, -0.42);
      s.add(m);
      this.objects.push(m);
    });

    // Compass letters at 2.5 m: heading drift shows at a glance.
    const labels: [string, number][] = [['N', 0], ['E', -Math.PI / 2], ['S', Math.PI], ['O', Math.PI / 2]];
    for (const [text, a] of labels) {
      const sprite = textSprite(text, '#ffd2b0');
      sprite.position.set(-Math.sin(a) * 2.5, EYE_HEIGHT + 0.45, -Math.cos(a) * 2.5);
      sprite.scale.set(0.25, 0.25, 1);
      s.add(sprite);
    }

    // Anchoring test: thin posts every 15° at 2 m and a horizon ring at eye height.
    const postMat = new THREE.MeshBasicMaterial({ color: 0x2fd39a });
    for (let k = 0; k < 24; k++) {
      const a = (k / 24) * Math.PI * 2;
      const post = new THREE.Mesh(new THREE.BoxGeometry(0.008, k % 6 === 0 ? 0.5 : 0.25, 0.008), postMat);
      post.position.set(Math.sin(a) * 2, EYE_HEIGHT, -Math.cos(a) * 2);
      this.anchor.add(post);
    }
    const ring = new THREE.Mesh(new THREE.TorusGeometry(2, 0.003, 4, 180), postMat);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = EYE_HEIGHT;
    this.anchor.add(ring);
    this.anchor.visible = false;
    s.add(this.anchor);

    // The model faces +z: placed on the floor at -z, it looks back at the user.
    this.human.root.position.set(0, 0, -2);
    s.add(this.human.root);
    s.add(this.labels);
    this.mapPoints.frustumCulled = false;
    this.mapPoints.visible = false;
    s.add(this.mapPoints);
  }

  /** Shows the environment memory's landmarks (scene coordinates), or hides them (null). */
  setMapPoints(points: Float32Array | null) {
    this.mapPoints.visible = !!points;
    if (!points) return;
    this.mapPoints.geometry.dispose();
    this.mapPoints.geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(points, 3));
  }

  /** Removes one label made by addLabel. */
  removeLabel(g: THREE.Object3D) {
    this.labels.remove(g);
    g.traverse(o => { if (o instanceof THREE.Mesh || o instanceof THREE.Sprite) { o.geometry.dispose(); (o.material as THREE.SpriteMaterial).map?.dispose(); (o.material as THREE.Material).dispose(); } });
  }

  /** A world-locked label at `pos` (world frame), sized to read the same from where the head is now. */
  addLabel(text: string, pos: THREE.Vector3, color = '#ffd2b0'): THREE.Object3D {
    const g = new THREE.Group();
    g.position.copy(pos);
    const dist = Math.max(0.4, pos.distanceTo(this.head.position));
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.008 * dist, 12, 8), new THREE.MeshBasicMaterial({ color }));
    const sprite = labelSprite(text, color);
    const h = 0.045 * dist;
    sprite.scale.set(h * sprite.userData.aspect, h, 1);
    sprite.position.y = h * 0.9;
    g.add(dot, sprite);
    g.userData.text = text;
    this.labels.add(g);
    return g;
  }

  clearLabels() {
    for (const g of [...this.labels.children]) {
      this.labels.remove(g);
      g.traverse(o => { if (o instanceof THREE.Mesh || o instanceof THREE.Sprite) { o.geometry.dispose(); (o.material as THREE.SpriteMaterial).map?.dispose(); (o.material as THREE.Material).dispose(); } });
    }
  }

  /** Text panel 1.2 m ahead (where the head looks now, kept level), fading after `seconds`. */
  showMessage(text: string, seconds = 14) {
    if (this.message) { this.scene.remove(this.message.panel); disposeMesh(this.message.panel); this.message = null; }
    if (!text) return;
    const tex = textPanel(text);
    const w = 0.6, h = w / tex.userData.aspect;
    const panel = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthTest: false }));
    panel.renderOrder = 10;
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(this.head.quaternion);
    fwd.y = 0;
    if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, -1);
    fwd.normalize();
    panel.position.copy(this.head.position).addScaledVector(fwd, 1.2);
    panel.position.y -= 0.12;
    panel.lookAt(this.head.position.x, panel.position.y, this.head.position.z);
    this.scene.add(panel);
    this.message = { panel, until: performance.now() + seconds * 1000 };
  }

  /** Advances the world once per display frame (several views may ask). */
  update(now: number) {
    if (now - this.lastUpdate < 4) return;
    const dt = this.lastUpdate < 0 ? 0 : Math.min(0.1, (now - this.lastUpdate) / 1000);
    this.lastUpdate = now;
    const held = this.handsView.held();
    for (const m of this.objects) if (!held.has(m)) { m.rotation.y += dt * 0.6; m.rotation.x += dt * 0.25; }
    this.human.update(dt);
    this.anchor.visible = this.reticle.visible = this.options.anchorTest;
    if (this.message) {
      const left = this.message.until - now;
      (this.message.panel.material as THREE.MeshBasicMaterial).opacity = Math.min(1, left / 1500);
      if (left <= 0) { this.scene.remove(this.message.panel); disposeMesh(this.message.panel); this.message = null; }
    }

    headMatrix(this.src.orientation(this.q), this.src.position(this.p)).decompose(this.head.position, this.head.quaternion, this.head.scale);
    const eyes = this.src.eyes();
    for (const [name, pos] of [['left', eyes.left], ['right', eyes.right], ['centre', eyes.centre]] as const) {
      this.cams[name].position.copy(pos);
      this.cams[name].quaternion.copy(eyes.rotation);
      this.cams[name].fov = eyes.vfov;
    }
    // RGB camera: calibrated pose in the head frame. three.js cameras look down -z with
    // y up; the camera model uses z forward, y down: a half turn about x between them.
    const rgb = this.src.rgbCamera();
    if (rgb) {
      this.cams.rgb.position.set(...rgb.t);
      this.cams.rgb.quaternion.set(...rgb.q).multiply(HALF_TURN_X);
    }
    this.head.updateMatrixWorld(true);

    this.updateDepth();
    this.handsView.update(this.src.hands());
  }

  /** Rebuilds the head-frame depth mesh when a new depth frame arrives. */
  private updateDepth() {
    const o = this.options;
    this.depthMesh.visible = o.occlusion || o.showDepth;
    this.depthMesh.material = o.showDepth ? this.depthMats.visible : this.depthMats.hidden;
    this.depthMesh.renderOrder = o.showDepth ? 0 : -1;
    const d = this.depthMesh.visible ? this.src.depth() : null;
    if (!d || d === this.lastDepth) return;
    this.lastDepth = d;
    const cols = Math.floor(d.width / MESH_STEP), rows = Math.floor(d.height / MESH_STEP);
    const pos = new Float32Array(cols * rows * 3);
    const z = new Float32Array(cols * rows);
    const pt: Vec3 = [0, 0, 0];
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const u = i * MESH_STEP + MESH_STEP / 2, v = j * MESH_STEP + MESH_STEP / 2;
        const depth = medianDepth(d, u, v);
        const k = j * cols + i;
        z[k] = depth;
        if (depth > 0) d.point(u, v, depth, pt);
        pos.set(depth > 0 ? pt : [0, 0, 0], k * 3);
      }
    }
    const idx: number[] = [];
    const ok = (a: number, b: number) => z[a] > 0 && z[b] > 0 && Math.abs(z[a] - z[b]) < 0.08 * Math.min(z[a], z[b]);
    for (let j = 0; j < rows - 1; j++) {
      for (let i = 0; i < cols - 1; i++) {
        const a = j * cols + i, b = a + 1, c = a + cols, e = c + 1;
        if (ok(a, b) && ok(a, c) && ok(b, c)) idx.push(a, c, b);
        if (ok(b, e) && ok(c, e) && ok(b, c)) idx.push(b, c, e);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    this.depthMesh.geometry.dispose();
    this.depthMesh.geometry = g;
  }
}

export type ViewMode = 'glasses' | 'rgb';

/**
 * A canvas showing the world. 'glasses': per-eye cameras, mono or side-by-side.
 * 'rgb': the RGB camera's view with a transparent background, for compositing.
 */
export class XrView {
  readonly renderer: THREE.WebGLRenderer;
  sbs = false;
  /** Minimum ms between frames (0 = every display frame). */
  minGapMs = 0;
  /** Recent frame intervals and render times (ms), for diagnostics. */
  readonly timing = { gaps: [] as number[], render: [] as number[] };
  private running = false;
  private lastFrame = 0;

  constructor(private canvas: HTMLCanvasElement, private world: XrWorld, private mode: ViewMode, private rgbAspect = () => 16 / 9) {
    // No MSAA: a 3840x1200 side-by-side frame with MSAA is too much for an integrated GPU.
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: mode === 'rgb', preserveDrawingBuffer: mode === 'rgb', powerPreference: 'high-performance' });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setClearColor(0x000000, mode === 'rgb' ? 0 : 1);
  }

  /** Window that owns the canvas: the page, or the dedicated glasses window. */
  private get win(): Window {
    return this.canvas.ownerDocument.defaultView ?? window;
  }

  start() {
    if (this.running) return;
    this.running = true;
    const win = this.win;
    // rAF of the canvas' window paces rendering on that screen's refresh.
    let last = 0;
    const loop = (now: number) => {
      if (!this.running || win.closed) return;
      if (now - last >= this.minGapMs) { last = now; this.frame(now); }
      win.requestAnimationFrame(loop);
    };
    win.requestAnimationFrame(loop);
  }

  stop() {
    this.running = false;
  }

  dispose() {
    this.stop();
    this.renderer.dispose();
  }

  private frame(now: number) {
    const c = this.canvas;
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    const r = this.renderer;
    const dpr = this.win.devicePixelRatio || 1;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      r.setPixelRatio(dpr);
      r.setSize(w, h, false);
    }
    const t0 = performance.now();
    if (this.lastFrame) push(this.timing.gaps, now - this.lastFrame);
    this.lastFrame = now;
    this.world.update(t0);
    this.draw(w, h);
    push(this.timing.render, performance.now() - t0);
  }

  private draw(w: number, h: number) {
    const r = this.renderer;
    const cams = this.world.cams;

    if (this.mode === 'rgb') {
      const cam = cams.rgb;
      const model = this.world.src.rgbCamera();
      if (!model) return;
      const aspect = model.width / model.height;
      cam.aspect = aspect;
      cam.fov = THREE.MathUtils.radToDeg(2 * Math.atan(model.height / 2 / model.f));
      // Principal point off-centre: shift the view window by the same amount.
      cam.setViewOffset(model.width, model.height, model.width / 2 - model.cx, model.height / 2 - model.cy, model.width, model.height);
      cam.updateProjectionMatrix();
      // Letterbox like the <video> (object-fit: contain).
      const vw = Math.min(w, h * aspect), vh = vw / aspect;
      r.setScissorTest(false);
      r.setViewport(0, 0, w, h);
      r.clear();
      r.setViewport((w - vw) / 2, (h - vh) / 2, vw, vh);
      r.render(this.world.scene, cam);
      return;
    }

    const eyeAspect = (this.sbs ? w / 2 : w) / h;
    for (const cam of [cams.left, cams.right, cams.centre]) { cam.aspect = eyeAspect; cam.updateProjectionMatrix(); }
    if (!this.sbs) {
      r.setScissorTest(false);
      r.setViewport(0, 0, w, h);
      r.render(this.world.scene, cams.centre);
      return;
    }
    // Side-by-side: each eye gets half the width (glasses in 3840x1200 3D mode).
    r.setScissorTest(true);
    r.setScissor(0, 0, w / 2, h);
    r.setViewport(0, 0, w / 2, h);
    r.render(this.world.scene, cams.left);
    r.setScissor(w / 2, 0, w / 2, h);
    r.setViewport(w / 2, 0, w / 2, h);
    r.render(this.world.scene, cams.right);
    r.setScissorTest(false);
  }

}

function push(a: number[], v: number) {
  a.push(v);
  if (a.length > 240) a.shift();
}

function glowing(color: number) {
  // Emissive so shapes stay visible on see-through optics whatever the lighting.
  return new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.45, roughness: 0.4, metalness: 0.1 });
}

/** Median of the valid depths in a small block, to drop isolated outliers. */
function medianDepth(d: DepthFrame, u: number, v: number) {
  const vals: number[] = [];
  for (let y = v - 1; y <= v + 1; y++) for (let x = u - 1; x <= u + 1; x++) {
    const z = d.depth[y * d.width + x];
    if (z > 0.15 && z < 6) vals.push(z);
  }
  if (vals.length < 5) return 0;
  vals.sort((a, b) => a - b);
  return vals[vals.length >> 1];
}

function disposeMesh(m: THREE.Mesh) {
  m.geometry.dispose();
  const mat = m.material as THREE.MeshBasicMaterial;
  mat.map?.dispose();
  mat.dispose();
}

/** Bright text with a thin outline: on see-through optics a dark box would be invisible anyway. */
function labelSprite(text: string, color: string) {
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d')!;
  const font = 'bold 56px system-ui, sans-serif';
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 40;
  c.width = w;
  c.height = 84;
  ctx.font = font;
  ctx.textBaseline = 'middle';
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.roundRect(2, 2, w - 4, 80, 18);
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.fillText(text, 20, 44);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  sprite.renderOrder = 5;
  sprite.userData.aspect = w / 84;
  return sprite;
}

/** Word-wrapped text on a canvas texture, for the assistant's answers. */
function textPanel(text: string) {
  const W = 1024, pad = 36, lineH = 52;
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d')!;
  const font = '600 40px system-ui, sans-serif';
  ctx.font = font;
  const lines: string[] = [];
  for (const para of text.split('\n')) {
    let line = '';
    for (const word of para.split(/\s+/)) {
      const next = line ? `${line} ${word}` : word;
      if (ctx.measureText(next).width > W - 2 * pad && line) { lines.push(line); line = word; } else line = next;
    }
    lines.push(line);
  }
  const shown = lines.slice(0, 9);
  c.width = W;
  c.height = pad * 2 + shown.length * lineH;
  ctx.font = font;
  ctx.strokeStyle = '#7fb4ff';
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.roundRect(2, 2, W - 4, c.height - 4, 24);
  ctx.stroke();
  ctx.fillStyle = '#eaf2ff';
  ctx.textBaseline = 'top';
  shown.forEach((l, i) => ctx.fillText(l, pad, pad + i * lineH));
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.userData.aspect = W / c.height;
  return tex;
}

function makeReticle() {
  const g = new THREE.Group();
  const mat = new THREE.LineBasicMaterial({ color: 0xff7a45 });
  const pts = [[-0.03, 0, 0.03, 0], [0, -0.03, 0, 0.03]];
  for (const [x1, y1, x2, y2] of pts) g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x1, y1, 0), new THREE.Vector3(x2, y2, 0)]), mat));
  g.position.z = -1;
  g.visible = false;
  return g;
}

function textSprite(text: string, color: string) {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = color;
  ctx.font = 'bold 96px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 64, 70);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true }));
}
