// World placement of the head, shared by the scene and the workers.

import * as THREE from 'three';

export const EYE_HEIGHT = 1.6;

/** Head pose in the world frame used by the scene. */
export function headMatrix(q: THREE.Quaternion, pos: THREE.Vector3 | null, out = new THREE.Matrix4()) {
  const p = new THREE.Vector3(0, EYE_HEIGHT, 0);
  if (pos) p.add(pos);
  return out.compose(p, q, new THREE.Vector3(1, 1, 1));
}
