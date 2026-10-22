// Stereo depth from the two tracking cameras, on the GPU (WebGL2).
//
// 1. Rectification: both fisheye images are resampled into a common pinhole frame
//    whose x axis is the baseline (lookup tables built from the factory calibration).
// 2. Census transform (5x5, 24 bits): matching on local intensity order, so the two
//    cameras' different exposure and vignetting do not matter.
// 3. Cost volume: Hamming distance for every disparity, aggregated over a 5x5 box
//    (separable passes on a texture array).
// 4. Winner-takes-all with a uniqueness test and sub-pixel refinement, for the left
//    and the right image, then a left-right consistency check.
// 5. On the CPU: speckle removal and a 3x3 median, then depth = f * baseline / d.

import type { TrackingCamera, Vec3 } from '../hands/stereoGeometry';

export const DEPTH_W = 320;
export const DEPTH_H = 240;
const MAX_DISP = 64; // ~32 cm minimum depth with the 13.6 cm baseline
const RECT_F = 150; // px, rectified focal length (~93° horizontal field)

/** Rectified frame in head coordinates (x along the baseline, y down, z forward). */
export interface RectAxes {
  ex: Vec3; ey: Vec3; ez: Vec3; origin: Vec3;
  baseline: number;
  f: number;
}

/** Head-frame point of rectified pixel (u, v) at depth z. */
export function rectPoint(a: RectAxes, u: number, v: number, z: number, out: Vec3): Vec3 {
  const x = ((u + 0.5 - DEPTH_W / 2) / a.f) * z, y = ((v + 0.5 - DEPTH_H / 2) / a.f) * z;
  out[0] = a.origin[0] + a.ex[0] * x + a.ey[0] * y + a.ez[0] * z;
  out[1] = a.origin[1] + a.ex[1] * x + a.ey[1] * y + a.ez[1] * z;
  out[2] = a.origin[2] + a.ex[2] * x + a.ey[2] * y + a.ez[2] * z;
  return out;
}

/** Rebuilds a DepthFrame received from a worker (functions do not cross threads). */
export function depthFrameFrom(d: { width: number; height: number; depth: Float32Array; left: Uint8Array; validRatio: number; axes: RectAxes }): DepthFrame {
  return { ...d, point: (u, v, z, out) => rectPoint(d.axes, u, v, z, out) };
}

export interface DepthFrame {
  axes: RectAxes;
  width: number;
  height: number;
  /** metres along the rectified optical axis, 0 = no measurement */
  depth: Float32Array;
  /** rectified left image (row 0 = top), the reference grid of `depth` */
  left: Uint8Array;
  /** fills `out` with the head-frame point of pixel (u, v) at depth z */
  point(u: number, v: number, z: number, out: Vec3): Vec3;
  validRatio: number;
}

// ------------------------------------------------------------------ shaders

const VS = `#version 300 es
in vec2 pos; out vec2 uv;
void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }`;

const HEAD = `#version 300 es
precision highp float; precision highp int;
precision highp sampler2D; precision highp usampler2D; precision highp sampler2DArray;
in vec2 uv;
uniform vec2 size;
ivec2 pix() { return ivec2(uv * size); }
`;

// Camera image sampled through a rectification lookup table (pixel coords, <0 = outside).
const RECTIFY_FS = HEAD + `out vec4 o;
uniform sampler2D img; uniform sampler2D lut; uniform vec2 imgSize;
void main() {
  vec2 p = texture(lut, uv).xy;
  if (p.x < 0.0) { o = vec4(0.0); return; }
  // Keep 0 for "outside": real pixels are at least 1/255.
  o = vec4(max(texture(img, (p + 0.5) / imgSize).r, 1.0 / 255.0), 0.0, 0.0, 1.0);
}`;

const CENSUS_FS = HEAD + `out uvec4 o;
uniform sampler2D img;
void main() {
  ivec2 c = pix();
  float ctr = texelFetch(img, c, 0).r;
  uint bits = 0u;
  uint k = 0u;
  for (int y = -2; y <= 2; y++) for (int x = -2; x <= 2; x++) {
    if (x == 0 && y == 0) continue;
    ivec2 q = clamp(c + ivec2(x, y), ivec2(0), ivec2(size) - 1);
    if (texelFetch(img, q, 0).r < ctr) bits |= 1u << k;
    k++;
  }
  if (ctr == 0.0) bits |= 0x80000000u; // outside the rectified field of view
  o = uvec4(bits, 0u, 0u, 1u);
}`;

// The cost volume packs 4 disparities per texel (RGBA) and each draw fills 4 layers
// at once (multiple render targets): 16 disparities per draw instead of 1. Fewer
// draw calls matter more than shader work here: Chrome decodes every WebGL call on
// one GPU-process thread shared with hand tracking and rendering.
const OUTS = `layout(location = 0) out vec4 o0; layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2; layout(location = 3) out vec4 o3;`;

const COST_FS = HEAD + OUTS + `
uniform usampler2D L; uniform usampler2D R; uniform int base;
uint popc(uint v) {
  v = v - ((v >> 1) & 0x55555555u);
  v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
  return (((v + (v >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24;
}
float cost(ivec2 c, uint a, int d) {
  int xr = c.x - d;
  if (xr < 0) return 1.0;
  uint b = texelFetch(R, ivec2(xr, c.y), 0).r;
  if (((a | b) & 0x80000000u) != 0u) return 1.0;
  return float(popc(a ^ b)) / 24.0;
}
vec4 four(ivec2 c, uint a, int d) { return vec4(cost(c, a, d), cost(c, a, d + 1), cost(c, a, d + 2), cost(c, a, d + 3)); }
void main() {
  ivec2 c = pix();
  uint a = texelFetch(L, c, 0).r;
  o0 = four(c, a, base); o1 = four(c, a, base + 4); o2 = four(c, a, base + 8); o3 = four(c, a, base + 12);
}`;

const BOX_FS = HEAD + OUTS + `
uniform sampler2DArray src; uniform int layer0; uniform ivec2 dir;
vec4 box(ivec2 c, int layer) {
  vec4 s = vec4(0.0);
  for (int k = -2; k <= 2; k++) s += texelFetch(src, ivec3(clamp(c + dir * k, ivec2(0), ivec2(size) - 1), layer), 0);
  return s / 5.0;
}
void main() {
  ivec2 c = pix();
  o0 = box(c, layer0); o1 = box(c, layer0 + 1); o2 = box(c, layer0 + 2); o3 = box(c, layer0 + 3);
}`;

const COST_AT = `
float costAt(ivec2 p, int d) {
  vec4 v = texelFetch(agg, ivec3(p, d >> 2), 0);
  int k = d & 3;
  return k == 0 ? v.x : k == 1 ? v.y : k == 2 ? v.z : v.w;
}`;

// Left disparity: best cost along the row, unique, refined to sub-pixel.
const WTA_LEFT_FS = HEAD + `out vec4 o;
uniform sampler2DArray agg;
const int MAXD = ${MAX_DISP};` + COST_AT + `
void main() {
  ivec2 c = pix();
  float best = 1e9; int bd = -1;
  for (int d = 0; d < MAXD; d++) {
    if (c.x - d < 3) break;
    float v = costAt(c, d);
    if (v < best) { best = v; bd = d; }
  }
  if (bd < 1 || bd >= MAXD - 1 || best > 0.4) { o = vec4(0.0); return; }
  float second = 1e9;
  for (int d = 0; d < MAXD; d++) {
    if (c.x - d < 3) break;
    if (abs(d - bd) <= 1) continue;
    second = min(second, costAt(c, d));
  }
  if (best > second * 0.95) { o = vec4(0.0); return; }
  float c0 = costAt(c, bd - 1), c2 = costAt(c, bd + 1);
  float den = c0 - 2.0 * best + c2;
  float sub = den > 1e-6 ? clamp(0.5 * (c0 - c2) / den, -0.5, 0.5) : 0.0;
  o = vec4(float(bd) + sub, best, 0.0, 1.0);
}`;

// Right disparity: for right pixel x, the left pixel x + d.
const WTA_RIGHT_FS = HEAD + `out vec4 o;
uniform sampler2DArray agg;
const int MAXD = ${MAX_DISP};` + COST_AT + `
void main() {
  ivec2 c = pix();
  float best = 1e9; int bd = -1;
  for (int d = 0; d < MAXD; d++) {
    int xl = c.x + d;
    if (xl >= int(size.x)) break;
    float v = costAt(ivec2(xl, c.y), d);
    if (v < best) { best = v; bd = d; }
  }
  o = vec4(float(bd), 0.0, 0.0, 1.0);
}`;

const LR_FS = HEAD + `out vec4 o;
uniform sampler2D dispL; uniform sampler2D dispR;
void main() {
  ivec2 c = pix();
  float dl = texelFetch(dispL, c, 0).r;
  if (dl <= 0.0) { o = vec4(0.0); return; }
  int xr = c.x - int(floor(dl + 0.5));
  float dr = xr >= 0 ? texelFetch(dispR, ivec2(xr, c.y), 0).r : -10.0;
  o = abs(dl - dr) <= 1.0 ? vec4(dl, 0.0, 0.0, 1.0) : vec4(0.0);
}`;

// ------------------------------------------------------------------ engine

type Prog = { p: WebGLProgram; u: (name: string) => WebGLUniformLocation | null };

export class StereoDepth {
  private gl: WebGL2RenderingContext;
  private prog: Record<'rect' | 'census' | 'cost' | 'box' | 'wtaL' | 'wtaR' | 'lr', Prog>;
  private imgTex: WebGLTexture[];
  private lutTex: WebGLTexture[];
  private rectTex: WebGLTexture[];
  private censusTex: WebGLTexture[];
  private costArr: WebGLTexture;
  private tmpArr: WebGLTexture;
  private dispL: WebGLTexture;
  private dispR: WebGLTexture;
  private final: WebGLTexture;
  private fbo: WebGLFramebuffer;
  private mrtFbo!: WebGLFramebuffer;
  private dispPbo!: WebGLBuffer;
  private rectPbo!: WebGLBuffer;
  private readBuf = new Float32Array(DEPTH_W * DEPTH_H * 4);
  private rectBuf = new Uint8Array(DEPTH_W * DEPTH_H * 4);
  private grayBuf = new Uint8Array(640 * 480);
  private axes: RectAxes;

  constructor(private cams: [TrackingCamera, TrackingCamera]) {
    const canvas = new OffscreenCanvas(DEPTH_W, DEPTH_H);
    const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: false });
    if (!gl) throw new Error('WebGL2 indisponible');
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('EXT_color_buffer_float indisponible');
    this.gl = gl;
    const mk = (fs: string): Prog => {
      const p = program(gl, VS, fs);
      const cache = new Map<string, WebGLUniformLocation | null>();
      return { p, u: n => (cache.has(n) ? cache.get(n)! : (cache.set(n, gl.getUniformLocation(p, n)), cache.get(n)!)) };
    };
    this.prog = {
      rect: mk(RECTIFY_FS), census: mk(CENSUS_FS), cost: mk(COST_FS), box: mk(BOX_FS),
      wtaL: mk(WTA_LEFT_FS), wtaR: mk(WTA_RIGHT_FS), lr: mk(LR_FS),
    };
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    this.axes = rectifiedAxes(cams);
    this.imgTex = [0, 1].map(() => tex2d(gl, gl.R8, 640, 480, gl.LINEAR));
    this.lutTex = cams.map(cam => {
      const t = tex2d(gl, gl.RG32F, DEPTH_W, DEPTH_H, gl.NEAREST);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, DEPTH_W, DEPTH_H, gl.RG, gl.FLOAT, this.buildLut(cam));
      return t;
    });
    this.rectTex = [0, 1].map(() => tex2d(gl, gl.R8, DEPTH_W, DEPTH_H, gl.NEAREST));
    this.censusTex = [0, 1].map(() => tex2d(gl, gl.R32UI, DEPTH_W, DEPTH_H, gl.NEAREST));
    this.costArr = texArray(gl, gl.RGBA16F, DEPTH_W, DEPTH_H, MAX_DISP / 4);
    this.tmpArr = texArray(gl, gl.RGBA16F, DEPTH_W, DEPTH_H, MAX_DISP / 4);
    this.mrtFbo = gl.createFramebuffer()!;
    this.dispL = tex2d(gl, gl.RGBA32F, DEPTH_W, DEPTH_H, gl.NEAREST);
    this.dispR = tex2d(gl, gl.R32F, DEPTH_W, DEPTH_H, gl.NEAREST);
    this.final = tex2d(gl, gl.RGBA32F, DEPTH_W, DEPTH_H, gl.NEAREST);
    this.fbo = gl.createFramebuffer()!;
    for (const [name, bytes] of [['dispPbo', DEPTH_W * DEPTH_H * 16], ['rectPbo', DEPTH_W * DEPTH_H * 4]] as const) {
      const b = gl.createBuffer()!;
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, b);
      gl.bufferData(gl.PIXEL_PACK_BUFFER, bytes, gl.STREAM_READ);
      this[name] = b;
    }
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
  }

  /** For rectified pixel (u, v): source pixel in this camera, or -1 when it falls outside. */
  private buildLut(cam: TrackingCamera): Float32Array {
    const { ex, ey, ez } = this.axes;
    const lut = new Float32Array(DEPTH_W * DEPTH_H * 2);
    const Rt = cam.R; // camera -> head (row-major); head -> camera is the transpose
    for (let v = 0; v < DEPTH_H; v++) {
      for (let u = 0; u < DEPTH_W; u++) {
        const x = (u + 0.5 - DEPTH_W / 2) / RECT_F, y = (v + 0.5 - DEPTH_H / 2) / RECT_F;
        const d: Vec3 = [ex[0] * x + ey[0] * y + ez[0], ex[1] * x + ey[1] * y + ez[1], ex[2] * x + ey[2] * y + ez[2]];
        const pc = [Rt[0] * d[0] + Rt[3] * d[1] + Rt[6] * d[2], Rt[1] * d[0] + Rt[4] * d[1] + Rt[7] * d[2], Rt[2] * d[0] + Rt[5] * d[1] + Rt[8] * d[2]];
        // GL texture rows go bottom-up: rectified row 0 (top) is texture row H-1.
        const i = ((DEPTH_H - 1 - v) * DEPTH_W + u) * 2;
        if (pc[2] <= 0.05) { lut[i] = lut[i + 1] = -1; continue; }
        const [pu, pv] = cam.distort(pc[0] / pc[2], pc[1] / pc[2]);
        const inside = pu >= 0 && pu < 640 && pv >= 0 && pv < 480;
        lut[i] = inside ? pu : -1;
        // Camera images are uploaded flipped, so image row pv is texture row 479 - pv.
        lut[i + 1] = inside ? 479 - pv : -1;
      }
    }
    return lut;
  }

  private target(t: WebGLTexture, layer = -1) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    if (layer < 0) gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    else gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, t, 0, layer);
  }

  /** Four consecutive layers of a texture array as the four render targets. */
  private targetLayers(t: WebGLTexture, layer0: number) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.mrtFbo);
    for (let k = 0; k < 4; k++) gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + k, t, 0, layer0 + k);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2, gl.COLOR_ATTACHMENT3]);
  }

  private use(pr: Prog) {
    const gl = this.gl;
    gl.useProgram(pr.p);
    gl.uniform2f(pr.u('size'), DEPTH_W, DEPTH_H);
  }

  private bind(pr: Prog, name: string, unit: number, t: WebGLTexture, array = false) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(array ? gl.TEXTURE_2D_ARRAY : gl.TEXTURE_2D, t);
    gl.uniform1i(pr.u(name), unit);
  }

  private draw() {
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
  }

  /** Runs one depth estimation from two 640x480 grayscale frames. */
  async compute(left: Uint8Array, right: Uint8Array): Promise<DepthFrame> {
    const gl = this.gl;
    const P = this.prog;
    for (const [i, px] of [left, right].entries()) {
      // Flip rows so texture row 0 is the bottom of the image, like the LUT expects.
      for (let y = 0; y < 480; y++) this.grayBuf.set(px.subarray(y * 640, y * 640 + 640), (479 - y) * 640);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.imgTex[i]);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 640, 480, gl.RED, gl.UNSIGNED_BYTE, this.grayBuf);
    }
    gl.viewport(0, 0, DEPTH_W, DEPTH_H);

    this.use(P.rect);
    gl.uniform2f(P.rect.u('imgSize'), 640, 480);
    for (let i = 0; i < 2; i++) {
      this.target(this.rectTex[i]);
      this.bind(P.rect, 'img', 0, this.imgTex[i]);
      this.bind(P.rect, 'lut', 1, this.lutTex[i]);
      this.draw();
    }

    this.use(P.census);
    for (let i = 0; i < 2; i++) {
      this.target(this.censusTex[i]);
      this.bind(P.census, 'img', 0, this.rectTex[i]);
      this.draw();
    }

    // 16 disparities per draw (4 per texel x 4 render targets).
    this.use(P.cost);
    this.bind(P.cost, 'L', 0, this.censusTex[0]);
    this.bind(P.cost, 'R', 1, this.censusTex[1]);
    for (let base = 0; base < MAX_DISP; base += 16) {
      this.targetLayers(this.costArr, base / 4);
      gl.uniform1i(P.cost.u('base'), base);
      this.draw();
    }

    // Separable 5x5 box aggregation: cost -> tmp (horizontal) -> cost (vertical).
    this.use(P.box);
    for (const [src, dst, dir] of [[this.costArr, this.tmpArr, [1, 0]], [this.tmpArr, this.costArr, [0, 1]]] as const) {
      this.bind(P.box, 'src', 0, src, true);
      gl.uniform2i(P.box.u('dir'), dir[0], dir[1]);
      for (let layer0 = 0; layer0 < MAX_DISP / 4; layer0 += 4) {
        this.targetLayers(dst, layer0);
        gl.uniform1i(P.box.u('layer0'), layer0);
        this.draw();
      }
    }

    this.use(P.wtaL);
    this.target(this.dispL);
    this.bind(P.wtaL, 'agg', 0, this.costArr, true);
    this.draw();
    this.use(P.wtaR);
    this.target(this.dispR);
    this.bind(P.wtaR, 'agg', 0, this.costArr, true);
    this.draw();
    this.use(P.lr);
    this.target(this.final);
    this.bind(P.lr, 'dispL', 0, this.dispL);
    this.bind(P.lr, 'dispR', 1, this.dispR);
    this.draw();
    // Asynchronous readback: copy into pixel buffers, then wait on a fence without
    // blocking Chrome's shared GPU queue (a plain readPixels stalls every context).
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.dispPbo);
    gl.readPixels(0, 0, DEPTH_W, DEPTH_H, gl.RGBA, gl.FLOAT, 0);
    this.target(this.rectTex[0]);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.rectPbo);
    gl.readPixels(0, 0, DEPTH_W, DEPTH_H, gl.RGBA, gl.UNSIGNED_BYTE, 0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    const fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)!;
    gl.flush();
    await waitFence(gl, fence);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.dispPbo);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this.readBuf);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.rectPbo);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this.rectBuf);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);

    const rectLeft = new Uint8Array(DEPTH_W * DEPTH_H);
    const disp = new Float32Array(DEPTH_W * DEPTH_H);
    for (let v = 0; v < DEPTH_H; v++) {
      const src = (DEPTH_H - 1 - v) * DEPTH_W;
      for (let u = 0; u < DEPTH_W; u++) {
        rectLeft[v * DEPTH_W + u] = this.rectBuf[(src + u) * 4];
        disp[v * DEPTH_W + u] = this.readBuf[(src + u) * 4];
      }
    }

    removeSpeckles(disp, DEPTH_W, DEPTH_H, 1.0, 25);
    const filtered = median3(disp, DEPTH_W, DEPTH_H);
    const depth = new Float32Array(DEPTH_W * DEPTH_H);
    const fb = RECT_F * this.axes.baseline;
    let valid = 0;
    for (let i = 0; i < depth.length; i++) {
      if (filtered[i] > 1.5) { depth[i] = fb / filtered[i]; valid++; }
    }
    const axes = this.axes;
    return {
      axes, width: DEPTH_W, height: DEPTH_H, depth, left: rectLeft,
      validRatio: valid / depth.length,
      point: (u, v, z, out) => rectPoint(axes, u, v, z, out),
    };
  }
}

/** Resolves once the GPU has passed the fence, polling instead of blocking. */
function waitFence(gl: WebGL2RenderingContext, fence: WebGLSync): Promise<void> {
  return new Promise(resolve => {
    const poll = () => {
      const r = gl.clientWaitSync(fence, 0, 0);
      if (r === gl.ALREADY_SIGNALED || r === gl.CONDITION_SATISFIED || r === gl.WAIT_FAILED) {
        gl.deleteSync(fence);
        resolve();
      } else setTimeout(poll, 2);
    };
    poll();
  });
}

/** Invalidates connected regions (|Δd| <= maxDiff) smaller than minSize pixels. */
function removeSpeckles(d: Float32Array, w: number, h: number, maxDiff: number, minSize: number) {
  const label = new Int32Array(d.length);
  const stack = new Int32Array(d.length);
  const region: number[] = [];
  let next = 1;
  for (let s = 0; s < d.length; s++) {
    if (!(d[s] > 0) || label[s]) continue;
    region.length = 0;
    let sp = 0;
    stack[sp++] = s;
    label[s] = next;
    while (sp) {
      const i = stack[--sp];
      region.push(i);
      const x = i % w, y = (i / w) | 0;
      const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1];
      for (const j of nb) {
        if (j < 0 || label[j] || !(d[j] > 0) || Math.abs(d[j] - d[i]) > maxDiff) continue;
        label[j] = next;
        stack[sp++] = j;
      }
    }
    if (region.length < minSize) for (const i of region) d[i] = 0;
    next++;
  }
}

/** 3x3 median over valid disparities (keeps holes as holes). */
function median3(d: Float32Array, w: number, h: number) {
  const out = new Float32Array(d.length);
  const v: number[] = [];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      if (!(d[i] > 0)) continue;
      v.length = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const z = d[i + dy * w + dx];
        if (z > 0) v.push(z);
      }
      v.sort((a, b) => a - b);
      out[i] = v[v.length >> 1];
    }
  }
  return out;
}

/** Rectified frame in head coordinates: x along the baseline, y down, z forward. */
function rectifiedAxes([l, r]: [TrackingCamera, TrackingCamera]): RectAxes {
  const b: Vec3 = [r.origin[0] - l.origin[0], r.origin[1] - l.origin[1], r.origin[2] - l.origin[2]];
  const baseline = Math.hypot(...b);
  const ex = norm(b);
  const fl = l.ray(l.cx, l.cy), fr = r.ray(r.cx, r.cy);
  let ez: Vec3 = [fl[0] + fr[0], fl[1] + fr[1], fl[2] + fr[2]];
  const k = dot(ez, ex);
  ez = norm([ez[0] - k * ex[0], ez[1] - k * ex[1], ez[2] - k * ex[2]]);
  const ey = cross(ez, ex);
  return { ex, ey, ez, origin: l.origin, baseline, f: RECT_F };
}

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: Vec3): Vec3 => { const n = Math.hypot(...a); return [a[0] / n, a[1] / n, a[2] / n]; };

function program(gl: WebGL2RenderingContext, vs: string, fs: string) {
  const p = gl.createProgram()!;
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]] as const) {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? 'shader');
    gl.attachShader(p, s);
  }
  gl.bindAttribLocation(p, 0, 'pos');
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? 'link');
  return p;
}

function tex2d(gl: WebGL2RenderingContext, internal: number, w: number, h: number, filter: number) {
  const t = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texStorage2D(gl.TEXTURE_2D, 1, internal, w, h);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

function texArray(gl: WebGL2RenderingContext, internal: number, w: number, h: number, layers: number) {
  const t = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, t);
  gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, internal, w, h, layers);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  return t;
}
