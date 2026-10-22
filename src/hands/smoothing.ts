// One Euro filter (Casiez et al., CHI 2012): strong smoothing at rest, little lag in fast motion.

class LowPass {
  private y: number | null = null;
  filter(x: number, alpha: number) {
    this.y = this.y === null ? x : alpha * x + (1 - alpha) * this.y;
    return this.y;
  }
  get last() { return this.y; }
}

export class OneEuro {
  private x = new LowPass();
  private dx = new LowPass();
  private lastT = -1;

  constructor(private minCutoff = 1.5, private beta = 1.0, private dCutoff = 1.0) {}

  filter(value: number, tSec: number) {
    const dt = this.lastT < 0 ? 1 / 30 : Math.max(1e-3, tSec - this.lastT);
    this.lastT = tSec;
    const prev = this.x.last;
    const d = prev === null ? 0 : (value - prev) / dt;
    const ed = this.dx.filter(d, alpha(this.dCutoff, dt));
    const cutoff = this.minCutoff + this.beta * Math.abs(ed);
    return this.x.filter(value, alpha(cutoff, dt));
  }
}

const alpha = (cutoff: number, dt: number) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));

/** Smooths every coordinate of a fixed-size point list (e.g. 21 joints x 3). */
export class PointSmoother {
  private filters: OneEuro[] = [];
  constructor(private minCutoff: number, private beta: number) {}

  apply(values: number[], tSec: number): number[] {
    while (this.filters.length < values.length) this.filters.push(new OneEuro(this.minCutoff, this.beta));
    return values.map((v, i) => this.filters[i].filter(v, tSec));
  }
}
