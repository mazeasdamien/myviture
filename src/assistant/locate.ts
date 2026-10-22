// Puts a point seen in the RGB image into 3D: the pixel's ray from the RGB camera is
// walked until it meets the stereo depth map (first surface along the ray).

import * as THREE from 'three';
import { pixelRay, type RgbCamera } from '../rgb/rgbCalib';
import type { DepthFrame } from '../depth/stereoDepth';

/** Head-frame point seen at RGB pixel (u, v); `fallback` metres along the ray without depth. */
export function locateRgbPixel(cam: RgbCamera, u: number, v: number, depth: DepthFrame | null, fallback = 1.5): { point: THREE.Vector3; measured: boolean } {
  const [x, y] = pixelRay(cam, u, v);
  const q = new THREE.Quaternion(...cam.q);
  const dir = new THREE.Vector3(x, y, 1).normalize().applyQuaternion(q);
  const o = new THREE.Vector3(...cam.t);
  const s = depth ? marchDepth(o, dir, depth) : null;
  return { point: o.clone().addScaledVector(dir, s ?? fallback), measured: s !== null };
}

/** Head-frame point where the ray (o, dir), head frame, first meets the depth map. */
export function hitDepth(o: THREE.Vector3, dir: THREE.Vector3, depth: DepthFrame | null, fallback = 1.5): { point: THREE.Vector3; measured: boolean } {
  const s = depth ? marchDepth(o, dir, depth) : null;
  return { point: o.clone().addScaledVector(dir, s ?? fallback), measured: s !== null };
}

function marchDepth(o: THREE.Vector3, dir: THREE.Vector3, d: DepthFrame): number | null {
  const { ex, ey, ez, origin, f } = d.axes;
  const P = new THREE.Vector3();
  let misses = 0;
  for (let s = 0.25; s < 5; s += 0.01 + s * 0.01) {
    P.copy(o).addScaledVector(dir, s);
    const rx = P.x - origin[0], ry = P.y - origin[1], rz = P.z - origin[2];
    const z = ez[0] * rx + ez[1] * ry + ez[2] * rz;
    if (z < 0.15) continue;
    const u = Math.round((f * (ex[0] * rx + ex[1] * ry + ex[2] * rz)) / z + d.width / 2 - 0.5);
    const v = Math.round((f * (ey[0] * rx + ey[1] * ry + ey[2] * rz)) / z + d.height / 2 - 0.5);
    if (u < 1 || v < 1 || u >= d.width - 1 || v >= d.height - 1) { if (++misses > 40) return null; continue; }
    // Median of the 3x3 neighbourhood: one bad pixel must not stop the ray.
    const vals: number[] = [];
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const zz = d.depth[(v + dy) * d.width + u + dx];
      if (zz > 0) vals.push(zz);
    }
    if (vals.length < 4) continue;
    vals.sort((a, b) => a - b);
    // The ray passed behind the surface: it hit it.
    if (z >= vals[vals.length >> 1] - 0.02) return s;
  }
  return null;
}
