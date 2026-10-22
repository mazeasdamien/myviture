// Hands in the immersive scene, plus pinch interaction: grab objects near the hand,
// or point at distant ones with a ray from the eyes through the pinch point and drag them.

import * as THREE from 'three';
import { BONES, TIP, type Hand } from '../hands/handTracker';

const NEAR_GRAB = 0.15; // m

class HandMesh {
  readonly group = new THREE.Group();
  readonly joints: THREE.Mesh[] = [];
  readonly bones: THREE.LineSegments;
  readonly ray: THREE.Line;
  readonly cursor: THREE.Mesh;

  constructor(color: number) {
    const jointGeo = new THREE.SphereGeometry(0.006, 10, 8);
    const jointMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    for (let i = 0; i < 21; i++) {
      const m = new THREE.Mesh(jointGeo, jointMat);
      this.joints.push(m);
      this.group.add(m);
    }
    this.bones = new THREE.LineSegments(
      new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(BONES.length * 6), 3)),
      new THREE.LineBasicMaterial({ color }),
    );
    this.group.add(this.bones);
    this.ray = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.6 }));
    this.ray.frustumCulled = false;
    this.cursor = new THREE.Mesh(new THREE.RingGeometry(0.02, 0.03, 24), new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide }));
    this.group.visible = false;
  }
}

interface Grab {
  object: THREE.Object3D;
  /** grab point in object local frame is kept by a fixed offset to the hand target */
  offset: THREE.Vector3;
  /** distance along the ray for far grabs; 0 for direct grabs */
  distance: number;
}

export class HandsView {
  private meshes = { left: new HandMesh(0x4c8dff), right: new HandMesh(0xff7a45) };
  private grabs: Partial<Record<'left' | 'right', Grab>> = {};
  private wasPinching = { left: false, right: false };
  private raycaster = new THREE.Raycaster();
  private hovered = new Set<THREE.Object3D>();

  /** Hand joints arrive in the world frame; `head` gives the eye position for pointing rays. */
  constructor(private scene: THREE.Scene, private head: THREE.Object3D, private targets: THREE.Object3D[]) {
    for (const m of Object.values(this.meshes)) scene.add(m.group, m.ray, m.cursor);
  }

  update(hands: Hand[]) {
    const tmp = new THREE.Vector3();
    for (const o of this.hovered) setGlow(o, 0);
    this.hovered.clear();

    for (const side of ['left', 'right'] as const) {
      const mesh = this.meshes[side];
      const hand = hands.find(h => h.side === side);
      mesh.group.visible = !!hand;
      mesh.ray.visible = mesh.cursor.visible = false;
      if (!hand) {
        this.release(side);
        continue;
      }

      hand.joints.forEach((j, i) => mesh.joints[i].position.set(j[0], j[1], j[2]));
      const pos = mesh.bones.geometry.getAttribute('position') as THREE.BufferAttribute;
      BONES.forEach(([a, b], k) => {
        pos.setXYZ(k * 2, ...mesh.joints[a].position.toArray());
        pos.setXYZ(k * 2 + 1, ...mesh.joints[b].position.toArray());
      });
      pos.needsUpdate = true;

      // Pinch point and pointing ray (from the eyes through the pinch).
      const pinchWorld = new THREE.Vector3(...hand.joints[TIP.thumb]).add(tmp.set(...hand.joints[TIP.index])).multiplyScalar(0.5);
      const eye = new THREE.Vector3().setFromMatrixPosition(this.head.matrixWorld);
      const dir = pinchWorld.clone().sub(eye).normalize();

      const started = hand.pinching && !this.wasPinching[side];
      const ended = !hand.pinching && this.wasPinching[side];
      this.wasPinching[side] = hand.pinching;
      if (ended) this.release(side);

      const grab = this.grabs[side];
      if (grab) {
        const target = grab.distance > 0 ? eye.clone().addScaledVector(dir, grab.distance) : pinchWorld;
        grab.object.position.copy(target).add(grab.offset);
        setGlow(grab.object, 1);
        this.hovered.add(grab.object);
        this.showRay(mesh, pinchWorld, target, grab.distance > 0);
        continue;
      }

      // Near objects win over the ray.
      const near = this.targets.find(o => o.position.distanceTo(pinchWorld) < NEAR_GRAB);
      this.raycaster.set(pinchWorld, dir);
      const hit = near ? null : this.raycaster.intersectObjects(this.targets, true)[0];
      const hitRoot = hit ? rootOf(hit.object, this.targets) : null;
      const focus = near ?? hitRoot;
      if (focus) { setGlow(focus, 0.5); this.hovered.add(focus); }
      if (hit) this.showRay(mesh, pinchWorld, hit.point, true);
      else this.showRay(mesh, pinchWorld, pinchWorld.clone().addScaledVector(dir, 3), false);

      if (started && focus) {
        const distance = near ? 0 : eye.distanceTo(focus.position);
        const anchor = near ? pinchWorld : eye.clone().addScaledVector(dir, distance);
        this.grabs[side] = { object: focus, offset: focus.position.clone().sub(anchor), distance };
      }
    }
  }

  private showRay(mesh: HandMesh, from: THREE.Vector3, to: THREE.Vector3, cursor: boolean) {
    mesh.ray.geometry.setFromPoints([from, to]);
    mesh.ray.visible = true;
    if (cursor) {
      mesh.cursor.position.copy(to);
      mesh.cursor.lookAt(from);
      mesh.cursor.visible = true;
    }
  }

  private release(side: 'left' | 'right') {
    delete this.grabs[side];
  }

  /** Objects currently held, so the scene can stop animating them. */
  held(): Set<THREE.Object3D> {
    return new Set(Object.values(this.grabs).map(g => g!.object));
  }
}

function rootOf(o: THREE.Object3D, roots: THREE.Object3D[]): THREE.Object3D | null {
  for (let cur: THREE.Object3D | null = o; cur; cur = cur.parent) if (roots.includes(cur)) return cur;
  return null;
}

/** Highlights an object by raising its emissive intensity (0 = normal). */
function setGlow(o: THREE.Object3D, amount: number) {
  o.traverse(c => {
    const m = (c as THREE.Mesh).material as THREE.MeshStandardMaterial | undefined;
    if (!m || !('emissiveIntensity' in m)) return;
    m.userData.baseGlow ??= m.emissiveIntensity;
    m.emissiveIntensity = m.userData.baseGlow + amount * 0.6;
  });
}
