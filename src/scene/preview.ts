// Dashboard 3D preview: a pair of glasses following the fused orientation (and position in 6DoF).

import * as THREE from 'three';

export function makeGlasses(): THREE.Group {
  const g = new THREE.Group();
  const frame = new THREE.MeshStandardMaterial({ color: 0x202226, roughness: 0.4, metalness: 0.3 });
  const lens = new THREE.MeshStandardMaterial({ color: 0x3a6df0, roughness: 0.05, metalness: 0.1, transparent: true, opacity: 0.55 });
  const bar = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.012, 0.012), frame);
  bar.position.y = 0.02;
  g.add(bar);
  for (const s of [-1, 1]) {
    const l = new THREE.Mesh(new THREE.BoxGeometry(0.058, 0.038, 0.006), lens);
    l.position.set(s * 0.035, 0, 0);
    g.add(l);
    const temple = new THREE.Mesh(new THREE.BoxGeometry(0.006, 0.01, 0.15), frame);
    temple.position.set(s * 0.073, 0.018, 0.075);
    g.add(temple);
  }
  // Forward marker
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.008, 0.03, 12), new THREE.MeshStandardMaterial({ color: 0xff7a45 }));
  nose.rotation.x = -Math.PI / 2;
  nose.position.set(0, 0.02, -0.03);
  g.add(nose);
  return g;
}

export class Preview {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50);
  readonly glasses = makeGlasses();
  private trail: THREE.Line;
  private trailPts: THREE.Vector3[] = [];

  constructor(private canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.camera.position.set(0.28, 0.2, 0.42);
    this.camera.lookAt(0, 0, 0);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 2.2));
    const dir = new THREE.DirectionalLight(0xffffff, 1.5);
    dir.position.set(1, 2, 1);
    this.scene.add(dir);
    const grid = new THREE.GridHelper(2, 20, 0x5b6170, 0x30343c);
    grid.position.y = -0.12;
    this.scene.add(grid);
    this.scene.add(new THREE.AxesHelper(0.1));
    this.scene.add(this.glasses);
    this.trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xff7a45 }));
    this.scene.add(this.trail);
  }

  render(q: THREE.Quaternion, position?: THREE.Vector3) {
    const c = this.canvas;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * devicePixelRatio)) {
      this.renderer.setPixelRatio(devicePixelRatio);
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
    this.glasses.quaternion.copy(q);
    if (position) {
      this.glasses.position.copy(position);
      const last = this.trailPts[this.trailPts.length - 1];
      if (!last || last.distanceTo(position) > 0.005) {
        this.trailPts.push(position.clone());
        if (this.trailPts.length > 2000) this.trailPts.shift();
        this.trail.geometry.setFromPoints(this.trailPts);
      }
      this.camera.position.set(position.x + 0.5, position.y + 0.35, position.z + 0.75);
      this.camera.lookAt(position);
    }
    this.renderer.render(this.scene, this.camera);
  }

  clearTrail() {
    this.trailPts = [];
    this.trail.geometry.setFromPoints([]);
  }
}
