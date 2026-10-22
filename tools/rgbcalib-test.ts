// Synthetic check of the RGB camera fit: known camera, noisy projections, recover it.
import * as THREE from 'three';
import { defaultRgbCamera, fitRgbCamera, project, type Correspondence, type RgbCamera } from '../src/rgb/rgbCalib';

const base = defaultRgbCamera(1920, 1080, 78);
const tilt = new THREE.Quaternion().setFromEuler(new THREE.Euler(THREE.MathUtils.degToRad(-12), THREE.MathUtils.degToRad(3), THREE.MathUtils.degToRad(1)));
const q = new THREE.Quaternion(...base.q).premultiply(tilt);
const truth: RgbCamera = { ...base, f: 1350, cx: 990, cy: 515, q: [q.x, q.y, q.z, q.w], t: [0.005, 0.02, -0.01] };
const data: Correspondence[] = [];
for (let i = 0; i < 400; i++) {
  const P: [number, number, number] = [(Math.random() - 0.5) * 0.5, -0.3 + (Math.random() - 0.5) * 0.4, -0.25 - Math.random() * 0.4];
  const px = project(truth, P);
  if (!px || px[0] < 0 || px[0] > 1920 || px[1] < 0 || px[1] > 1080) continue;
  data.push({ P, u: px[0] + (Math.random() - 0.5) * 6, v: px[1] + (Math.random() - 0.5) * 6 });
}
const fit = fitRgbCamera(base, data);
const angle = (a: RgbCamera, b: RgbCamera) => THREE.MathUtils.radToDeg(new THREE.Quaternion(...a.q).angleTo(new THREE.Quaternion(...b.q)));
console.log('points', data.length, 'rms', fit.rms?.toFixed(2), 'px');
console.log('f', fit.f.toFixed(0), '(1350)  cx', fit.cx.toFixed(0), '(990)  cy', fit.cy.toFixed(0), '(515)');
console.log('t', fit.t.map(v => (v * 100).toFixed(1)), 'cm (0.5, 2, -1)  rotation error', angle(fit, truth).toFixed(2), '°');
