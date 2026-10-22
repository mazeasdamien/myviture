// A plain three.js WebXR page, written as for any headset (renderer.xr, requestSession,
// XRHand joints, 'select' events): it runs on the glasses through the runtime imported
// first. Pinch a cube to change its colour.

import { installRuntime } from './runtime';
installRuntime();

import * as THREE from 'three';

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local-floor');
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.add(new THREE.HemisphereLight(0xffffff, 0x445066, 2.5));
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.05, 100);

// Cubes around the user at arm's length and a ring on the floor.
const cubes: THREE.Mesh[] = [];
for (let i = 0; i < 8; i++) {
  const a = (i / 8) * Math.PI * 2;
  const color = new THREE.Color().setHSL(i / 8, 0.8, 0.6);
  const m = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.12, 0.12), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.5 }));
  m.position.set(Math.sin(a) * 0.9, 1.55, -Math.cos(a) * 0.9);
  scene.add(m);
  cubes.push(m);
}
const ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 0.93, 96), new THREE.MeshBasicMaterial({ color: 0x7fd1ff, side: THREE.DoubleSide }));
ring.rotation.x = -Math.PI / 2;
scene.add(ring);

// Tracked hands: one sphere per joint (WebXR's 25), the standard way.
const joints = ['wrist', 'thumb-tip', 'index-finger-tip', 'middle-finger-tip', 'ring-finger-tip', 'pinky-finger-tip',
  'index-finger-phalanx-proximal', 'middle-finger-phalanx-proximal', 'ring-finger-phalanx-proximal', 'pinky-finger-phalanx-proximal'];
for (let i = 0; i < 2; i++) {
  const hand = renderer.xr.getHand(i);
  scene.add(hand);
  const spheres = new Map<string, THREE.Mesh>();
  hand.addEventListener('connected', () => {
    for (const name of joints) {
      const s = new THREE.Mesh(new THREE.SphereGeometry(0.008, 10, 8), new THREE.MeshBasicMaterial({ color: i ? 0xff7a45 : 0x4c8dff }));
      spheres.set(name, s);
      scene.add(s);
    }
  });
  hand.userData.spheres = spheres;
  // 'select' from a pinch: recolour the cube nearest to the pinch.
  const controller = renderer.xr.getController(i);
  scene.add(controller);
  controller.addEventListener('select', () => {
    const j = (hand as unknown as { joints: Record<string, THREE.Object3D> }).joints['index-finger-tip'];
    if (!j) return;
    const p = new THREE.Vector3().setFromMatrixPosition(j.matrixWorld);
    let best = cubes[0];
    for (const c of cubes) if (c.position.distanceTo(p) < best.position.distanceTo(p)) best = c;
    const color = new THREE.Color().setHSL(Math.random(), 0.85, 0.6);
    (best.material as THREE.MeshStandardMaterial).color.copy(color);
    (best.material as THREE.MeshStandardMaterial).emissive.copy(color);
  });
}

renderer.setAnimationLoop(() => {
  for (const c of cubes) { c.rotation.y += 0.01; c.rotation.x += 0.004; }
  for (let i = 0; i < 2; i++) {
    const hand = renderer.xr.getHand(i) as unknown as { joints: Record<string, THREE.Object3D>; userData: { spheres: Map<string, THREE.Mesh> } };
    for (const [name, s] of hand.userData.spheres) {
      const j = hand.joints[name];
      s.visible = !!j?.visible;
      if (j) s.position.setFromMatrixPosition(j.matrix);
    }
  }
  renderer.render(scene, camera);
});

// Start the immersive session as a WebXR page would from its "Enter AR" button.
async function enter() {
  if (!(await navigator.xr?.isSessionSupported('immersive-ar'))) return;
  const session = await navigator.xr!.requestSession('immersive-ar', { requiredFeatures: ['local-floor'], optionalFeatures: ['hand-tracking'] });
  await renderer.xr.setSession(session as unknown as XRSession);
}
enter();
document.addEventListener('dblclick', () => window.close());
document.addEventListener('click', () => { if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {}); });
