// Draws the grayscale tracking-camera frames and runs a simple auto-exposure.

import type { StereoFrame } from '../device/protocol';

export class StereoView {
  private images: ImageData[] = [];
  private ctxs: CanvasRenderingContext2D[];
  private pending: (StereoFrame | null)[] = [null, null];
  /** Mean brightness of the last left frame, 0..255 */
  mean = 0;
  /** Incremented every time a camera canvas is redrawn */
  readonly version = [0, 0];

  constructor(readonly canvases: HTMLCanvasElement[]) {
    this.ctxs = canvases.map(c => c.getContext('2d')!);
  }

  push(f: StereoFrame) {
    this.pending[f.camera] = f;
    if (f.camera === 0) {
      let sum = 0;
      // Sample every 16th pixel: plenty for exposure control.
      for (let i = 0; i < f.pixels.length; i += 16) sum += f.pixels[i];
      this.mean = sum / (f.pixels.length / 16);
    }
  }

  /** Call from requestAnimationFrame; converts at most one frame per camera per display frame. */
  draw() {
    for (let cam = 0; cam < 2; cam++) {
      const f = this.pending[cam];
      if (!f) continue;
      this.pending[cam] = null;
      const c = this.canvases[cam];
      if (c.width !== f.width) { c.width = f.width; c.height = f.height; }
      let img = this.images[cam];
      if (!img || img.width !== f.width) img = this.images[cam] = new ImageData(f.width, f.height);
      const px = f.pixels, out = img.data;
      for (let i = 0, j = 0; i < px.length; i++, j += 4) {
        out[j] = out[j + 1] = out[j + 2] = px[i];
        out[j + 3] = 255;
      }
      this.ctxs[cam].putImageData(img, 0, 0);
      this.version[cam]++;
    }
  }
}
