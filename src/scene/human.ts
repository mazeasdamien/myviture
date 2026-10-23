// Animated human: Microsoft Rocketbox Male_Adult_01 (MIT), public/models/human.glb.
//
// 80-bone 3ds Max biped (Bip01_*) and 67 face blendshapes: the 52 ARKit ones
// (JawOpen, EyeBlinkLeft, MouthSmileLeft…) and 15 visemes (viseme_aa, viseme_PP…).
// The glTF splits the mesh by material (body, head, hair/lashes), so every
// blendshape is set on all three parts.
//
// Body clips are Rocketbox animations made for the same skeleton (public/models/anims,
// built by tools/avatar/anims.py). Looping ones become the base pose; one-shots play
// once and fade back to it.

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { humanLight, humanShading, shadowBlob, shadowTexture, studioEnvironment } from './humanLook';

const URL = '/models/human.glb';
const ANIMS = '/models/anims/';
/** clip name -> loops (base pose) or plays once */
const CLIPS = {
  idle: true, talk: true, listen: true,
  wave: false, shrug: false, think: false, clap: false,
} as const;
export type HumanAnim = keyof typeof CLIPS;
export const HUMAN_ANIMS = Object.keys(CLIPS) as HumanAnim[];

const BLINK_CLOSE = 0.06, BLINK_HOLD = 0.03, BLINK_OPEN = 0.1; // s

export class Human {
  readonly root = new THREE.Group();
  /** body animation */
  mixer: THREE.AnimationMixer | null = null;
  /** blendshape name -> index, shared by every part */
  morphs: Record<string, number> = {};
  private parts: THREE.SkinnedMesh[] = [];
  private clips = new Map<HumanAnim, THREE.AnimationClip>();
  private action: THREE.AnimationAction | null = null;
  /** looping clip the one-shots return to */
  private base: HumanAnim = 'idle';
  private nextBlink = 2;
  private blinkT = -1;
  /** contact shadows under each foot, following them */
  private feet: { bone: THREE.Object3D; blob: THREE.Mesh; restY: number }[] = [];
  private v = new THREE.Vector3();
  private shadowMap = shadowTexture();
  private _studio = true;

  constructor() {
    // Soft occlusion under the whole body. On the see-through glasses black is
    // transparent, so the shadows only show over other content and in the MR video.
    this.root.add(shadowBlob(this.shadowMap, 0.8, 0.55, 0.45));
    this.load().catch(e => console.warn('human: cannot load', e));
  }

  private async load() {
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(URL);
    const bones = new Set<string>();
    gltf.scene.traverse(o => {
      if (o instanceof THREE.Bone) bones.add(o.name);
      if (!(o instanceof THREE.SkinnedMesh)) return;
      // Bounds are the bind pose: animated limbs would leave them and get culled.
      o.frustumCulled = false;
      this.parts.push(o);
      if (o.morphTargetDictionary) this.morphs = o.morphTargetDictionary;
    });
    this.root.add(gltf.scene);

    const env = studioEnvironment();
    for (const p of this.parts) {
      const m = p.material as THREE.MeshStandardMaterial;
      m.envMap = env;
      // Hair cards lie almost flat on the scalp: seen at a grazing angle, plain mipmapping
      // blurs them into smooth grey sheets. Anisotropic filtering keeps the strands.
      for (const t of [m.map, m.normalMap, m.roughnessMap]) if (t) t.anisotropy = 16;
      humanShading(m, m.alphaTest === 0);
    }
    this.studio = this._studio;
    gltf.scene.updateMatrixWorld(true);
    for (const name of ['Bip01_L_Foot', 'Bip01_R_Foot']) {
      const bone = gltf.scene.getObjectByName(name);
      if (!bone) continue;
      const blob = shadowBlob(this.shadowMap, 0.2, 0.32, 0.55);
      this.root.add(blob);
      this.feet.push({ bone, blob, restY: this.root.worldToLocal(bone.getWorldPosition(this.v)).y });
    }

    const mixer = new THREE.AnimationMixer(gltf.scene);
    mixer.addEventListener('finished', e => { if (e.action === this.action) this.play(this.base); });
    this.mixer = mixer;

    await Promise.all(HUMAN_ANIMS.map(async name => {
      const clip = (await loader.loadAsync(ANIMS + name + '.glb')).animations[0];
      // The animation rig has extra helper bones (nubs, footsteps): keep the avatar's.
      clip.tracks = clip.tracks.filter(t => bones.has(t.name.slice(0, t.name.lastIndexOf('.'))));
      clip.name = name;
      this.clips.set(name, clip);
      if (name === this.base && !this.action) this.play(name, 0);
    }));
  }

  get loaded() {
    return this.mixer !== null;
  }

  /**
   * Studio lighting (default): a quarter of the scene's ambient, nearly twice its key
   * light, stronger reflections. Off: the scene's flat lighting, as the other objects get.
   */
  get studio() {
    return this._studio;
  }

  set studio(on: boolean) {
    this._studio = on;
    humanLight.ambient.value = on ? 0.3 : 1;
    humanLight.key.value = on ? 1.9 : 1;
    for (const p of this.parts) {
      const m = p.material as THREE.MeshStandardMaterial;
      m.envMapIntensity = m.alphaTest > 0 ? (on ? 0.5 : 0.35) : (on ? 1 : 0.6);
    }
  }

  /** Name of the clip playing, null before the clips load. */
  get current(): HumanAnim | null {
    return (this.action?.getClip().name as HumanAnim | undefined) ?? null;
  }

  /** Sets one blendshape (0..1) on every part. */
  setMorph(name: string, value: number) {
    const i = this.morphs[name];
    if (i === undefined) return;
    for (const p of this.parts) if (p.morphTargetInfluences) p.morphTargetInfluences[i] = value;
  }

  /**
   * Cross-fades to a clip. idle, talk and listen loop and become the pose the others
   * return to; wave, shrug, think and clap play once. False if not loaded yet.
   */
  play(name: HumanAnim, fade = 0.4): boolean {
    const clip = this.clips.get(name);
    if (!this.mixer || !clip) return false;
    const loop = CLIPS[name];
    if (loop) this.base = name;
    const next = this.mixer.clipAction(clip);
    if (next === this.action && loop) return true;
    next.setLoop(loop ? THREE.LoopRepeat : THREE.LoopOnce, Infinity);
    next.clampWhenFinished = !loop;
    next.reset().play();
    if (this.action && this.action !== next) this.action.crossFadeTo(next, fade, false);
    this.action = next;
    return true;
  }

  update(dt: number) {
    if (!this.mixer) return;
    this.mixer.update(dt);
    this.blink(dt);
    // Feet shadows: under the foot bones (last frame's pose), fainter and wider as a
    // foot lifts.
    for (const f of this.feet) {
      const p = this.root.worldToLocal(f.bone.getWorldPosition(this.v));
      const lift = Math.max(0, p.y - f.restY);
      f.blob.position.set(p.x, 0.003, p.z + 0.04);
      f.blob.scale.setScalar(1 + lift * 2);
      (f.blob.material as THREE.MeshBasicMaterial).opacity = 0.55 * Math.max(0, 1 - lift / 0.25);
    }
  }

  /** Idle blink every 2 to 6 s, now and then a double one. */
  private blink(dt: number) {
    if (this.blinkT < 0) {
      this.nextBlink -= dt;
      if (this.nextBlink > 0) return;
      this.blinkT = 0;
      this.nextBlink = Math.random() < 0.15 ? 0.25 : 2 + Math.random() * 4;
    }
    this.blinkT += dt;
    const t = this.blinkT;
    const open = BLINK_CLOSE + BLINK_HOLD;
    const v = t < BLINK_CLOSE ? t / BLINK_CLOSE : t < open ? 1 : Math.max(0, 1 - (t - open) / BLINK_OPEN);
    this.setMorph('EyeBlinkLeft', v);
    this.setMorph('EyeBlinkRight', v);
    if (t >= open + BLINK_OPEN) this.blinkT = -1;
  }
}
