// Synthetic check of the position filter: a head swaying 10 cm at 0.5 Hz, noisy IMU,
// odometry fixes every 140 ms arriving 100 ms late with 5 mm noise. Prints the error
// and the frame-to-frame jitter of the displayed position.
import * as THREE from 'three';
import { PositionFilter } from '../src/vio/stereoOdometry';

const f = new PositionFilter();
const q = new THREE.Quaternion();
const gauss = () => Math.sqrt(-2 * Math.log(Math.random() + 1e-12)) * Math.cos(2 * Math.PI * Math.random());
const truth = (t: number) => new THREE.Vector3(0.1 * Math.sin(Math.PI * t), 0.02 * Math.sin(2 * Math.PI * t * 0.7), 0);
const acc = (t: number) => new THREE.Vector3(-0.1 * Math.PI ** 2 * Math.sin(Math.PI * t), -0.02 * (1.4 * Math.PI) ** 2 * Math.sin(1.4 * Math.PI * t), 0);
const pending: { at: number; t: number; p: THREE.Vector3 }[] = [];
let err = 0, n = 0, jit = 0, jn = 0;
let prevErr: THREE.Vector3 | null = null;
for (let i = 0; i < 20000; i++) {
  const t = i / 1000;
  const a = acc(t).add(new THREE.Vector3(gauss(), gauss(), gauss()).multiplyScalar(0.15)).add(new THREE.Vector3(0.05, 0, 0));
  a.y += 9.80665;
  f.predict(a, q, t, false);
  if (i % 140 === 0) pending.push({ at: t + 0.1, t, p: truth(t).add(new THREE.Vector3(gauss(), gauss(), gauss()).multiplyScalar(0.005)) });
  while (pending.length && pending[0].at <= t) { const m = pending.shift()!; f.correct(m.p, m.t, 60); }
  if (t > 3 && i % 16 === 0) {
    const e = f.display.clone().sub(truth(t));
    err += e.lengthSq(); n++;
    if (prevErr) { jit += e.clone().sub(prevErr).lengthSq(); jn++; }
    prevErr = e;
  }
}
console.log('rms error', (Math.sqrt(err / n) * 1000).toFixed(1), 'mm   frame-to-frame jitter', (Math.sqrt(jit / jn) * 1000).toFixed(2), 'mm');
