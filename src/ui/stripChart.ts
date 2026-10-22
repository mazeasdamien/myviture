// Minimal scrolling line chart for high-rate telemetry (canvas 2D, ring buffer).

export interface Series {
  label: string;
  color: string;
}

export class StripChart {
  private ctx: CanvasRenderingContext2D;
  private data: Float32Array[];
  private head = 0;
  private count = 0;

  constructor(
    private canvas: HTMLCanvasElement,
    private series: Series[],
    private capacity = 1000,
    private range: [number, number] | 'auto' = 'auto',
  ) {
    this.ctx = canvas.getContext('2d')!;
    this.data = series.map(() => new Float32Array(capacity));
  }

  push(values: ArrayLike<number>) {
    for (let i = 0; i < this.series.length; i++) this.data[i][this.head] = values[i];
    this.head = (this.head + 1) % this.capacity;
    this.count = Math.min(this.count + 1, this.capacity);
  }

  draw() {
    const { canvas, ctx } = this;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    ctx.clearRect(0, 0, w, h);
    if (!this.count) return;

    let [lo, hi] = this.range === 'auto' ? [Infinity, -Infinity] : this.range;
    if (this.range === 'auto') {
      for (const d of this.data) for (let i = 0; i < this.count; i++) { lo = Math.min(lo, d[i]); hi = Math.max(hi, d[i]); }
      const pad = Math.max((hi - lo) * 0.1, 0.05);
      lo -= pad; hi += pad;
    }
    const y = (v: number) => h - ((v - lo) / (hi - lo)) * h;

    const style = getComputedStyle(canvas);
    ctx.strokeStyle = style.getPropertyValue('--grid').trim() || '#333';
    ctx.lineWidth = 1;
    if (lo < 0 && hi > 0) { ctx.beginPath(); ctx.moveTo(0, y(0)); ctx.lineTo(w, y(0)); ctx.stroke(); }

    ctx.lineWidth = 1.5 * dpr;
    const start = (this.head - this.count + this.capacity) % this.capacity;
    for (let s = 0; s < this.series.length; s++) {
      const d = this.data[s];
      ctx.strokeStyle = this.series[s].color;
      ctx.beginPath();
      for (let i = 0; i < this.count; i++) {
        const px = (i / (this.capacity - 1)) * w, py = y(d[(start + i) % this.capacity]);
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.stroke();
    }
    ctx.fillStyle = style.getPropertyValue('--muted').trim() || '#888';
    ctx.font = `${11 * dpr}px ui-monospace, monospace`;
    ctx.fillText(hi.toFixed(2), 4 * dpr, 12 * dpr);
    ctx.fillText(lo.toFixed(2), 4 * dpr, h - 4 * dpr);
  }
}
