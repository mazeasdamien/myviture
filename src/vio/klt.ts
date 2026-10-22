// Pyramidal Lucas-Kanade optical flow and Shi-Tomasi corners on 8-bit grayscale images.

export interface Gray {
  w: number;
  h: number;
  px: Uint8Array;
}

export type Pyramid = Gray[];

export function pyramid(img: Gray, levels: number): Pyramid {
  const out = [img];
  for (let l = 1; l < levels; l++) {
    const s = out[l - 1];
    const w = s.w >> 1, h = s.h >> 1;
    const px = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = 2 * y * s.w + 2 * x;
        px[y * w + x] = (s.px[i] + s.px[i + 1] + s.px[i + s.w] + s.px[i + s.w + 1] + 2) >> 2;
      }
    }
    out.push({ w, h, px });
  }
  return out;
}

function sample(g: Gray, x: number, y: number): number {
  const x0 = Math.floor(x), y0 = Math.floor(y);
  if (x0 < 0 || y0 < 0 || x0 >= g.w - 1 || y0 >= g.h - 1) return NaN;
  const fx = x - x0, fy = y - y0, i = y0 * g.w + x0, p = g.px;
  return (p[i] * (1 - fx) + p[i + 1] * fx) * (1 - fy) + (p[i + g.w] * (1 - fx) + p[i + g.w + 1] * fx) * fy;
}

const HALF = 5; // 11x11 window
const ITERS = 12;

/** Tracks one point from pyramid A to B; returns the position in B or null. */
export function trackPoint(A: Pyramid, B: Pyramid, x: number, y: number, guessX = x, guessY = y): [number, number] | null {
  const levels = A.length;
  let dx = (guessX - x) / 2 ** levels, dy = (guessY - y) / 2 ** levels;
  for (let l = levels - 1; l >= 0; l--) {
    const a = A[l], b = B[l], s = 2 ** l;
    const px = x / s, py = y / s;
    dx *= 2; dy *= 2;
    // Spatial gradient matrix of the template window in A.
    let gxx = 0, gxy = 0, gyy = 0;
    const n = (2 * HALF + 1) ** 2;
    const ix = new Float32Array(n), iy = new Float32Array(n), it = new Float32Array(n);
    let k = 0;
    for (let wy = -HALF; wy <= HALF; wy++) {
      for (let wx = -HALF; wx <= HALF; wx++, k++) {
        const gx = (sample(a, px + wx + 1, py + wy) - sample(a, px + wx - 1, py + wy)) / 2;
        const gy = (sample(a, px + wx, py + wy + 1) - sample(a, px + wx, py + wy - 1)) / 2;
        if (Number.isNaN(gx) || Number.isNaN(gy)) return null;
        ix[k] = gx; iy[k] = gy;
        it[k] = sample(a, px + wx, py + wy);
        gxx += gx * gx; gxy += gx * gy; gyy += gy * gy;
      }
    }
    const det = gxx * gyy - gxy * gxy;
    if (det < 1e-3) return null;
    for (let iter = 0; iter < ITERS; iter++) {
      let bx = 0, by = 0;
      k = 0;
      for (let wy = -HALF; wy <= HALF; wy++) {
        for (let wx = -HALF; wx <= HALF; wx++, k++) {
          const v = sample(b, px + dx + wx, py + dy + wy);
          if (Number.isNaN(v)) return null;
          const diff = it[k] - v;
          bx += diff * ix[k];
          by += diff * iy[k];
        }
      }
      const ux = (gyy * bx - gxy * by) / det, uy = (gxx * by - gxy * bx) / det;
      dx += ux; dy += uy;
      if (ux * ux + uy * uy < 0.0009) break;
    }
  }
  const nx = x + dx, ny = y + dy;
  return nx >= HALF && ny >= HALF && nx < A[0].w - HALF && ny < A[0].h - HALF ? [nx, ny] : null;
}

/**
 * Shi-Tomasi corners, the strongest one per grid cell, where `accept` (e.g. a valid
 * depth) holds.
 */
export function corners(g: Gray, cellsX: number, cellsY: number, accept: (x: number, y: number) => boolean): [number, number][] {
  const out: [number, number][] = [];
  const cw = Math.floor(g.w / cellsX), ch = Math.floor(g.h / cellsY);
  const p = g.px, w = g.w;
  for (let cy = 0; cy < cellsY; cy++) {
    for (let cx = 0; cx < cellsX; cx++) {
      let best = 0, bx = -1, by = -1;
      for (let y = Math.max(HALF + 2, cy * ch); y < Math.min(g.h - HALF - 2, (cy + 1) * ch); y += 2) {
        for (let x = Math.max(HALF + 2, cx * cw); x < Math.min(g.w - HALF - 2, (cx + 1) * cw); x += 2) {
          let sxx = 0, sxy = 0, syy = 0;
          for (let wy = -2; wy <= 2; wy++) {
            for (let wx = -2; wx <= 2; wx++) {
              const i = (y + wy) * w + x + wx;
              const gx = p[i + 1] - p[i - 1], gy = p[i + w] - p[i - w];
              sxx += gx * gx; sxy += gx * gy; syy += gy * gy;
            }
          }
          const tr = sxx + syy, det = sxx * syy - sxy * sxy;
          const minEig = tr / 2 - Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
          if (minEig > best && accept(x, y)) { best = minEig; bx = x; by = y; }
        }
      }
      if (bx >= 0 && best > 400) out.push([bx, by]);
    }
  }
  return out;
}
