// Draws hand skeletons over a video or camera canvas (content letterboxed with object-fit: contain).

import { BONES, TIP, type Hand, type View } from './handTracker';

const COLORS = { left: '#4c8dff', right: '#ff7a45' } as const;

export function drawHands(canvas: HTMLCanvasElement, srcW: number, srcH: number, hands: Hand[], view: View) {
  const dpr = devicePixelRatio || 1;
  const cw = Math.round(canvas.clientWidth * dpr), ch = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, cw, ch);
  if (!srcW) return;

  const scale = Math.min(cw / srcW, ch / srcH);
  const ox = (cw - srcW * scale) / 2, oy = (ch - srcH * scale) / 2;
  const px = (p: { x: number; y: number }) => [ox + p.x * srcW * scale, oy + p.y * srcH * scale] as const;

  for (const hand of hands) {
    const pts = hand.views[view];
    if (!pts) continue;
    const color = COLORS[hand.side];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5 * dpr;
    ctx.beginPath();
    for (const [a, b] of BONES) {
      const [x1, y1] = px(pts[a]), [x2, y2] = px(pts[b]);
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
    ctx.stroke();
    ctx.fillStyle = '#fff';
    for (const p of pts) {
      const [x, y] = px(p);
      ctx.beginPath();
      ctx.arc(x, y, 2.5 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }
    if (hand.pinching) {
      const [tx, ty] = px(pts[TIP.thumb]), [ix, iy] = px(pts[TIP.index]);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc((tx + ix) / 2, (ty + iy) / 2, 9 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }
    const [wx, wy] = px(pts[0]);
    const dist = Math.hypot(...hand.joints[0]);
    ctx.font = `${12 * dpr}px ui-monospace, monospace`;
    ctx.fillStyle = color;
    ctx.fillText(`${hand.side === 'left' ? 'G' : 'D'} ${(dist * 100).toFixed(0)} cm ${hand.depth}${hand.pinching ? ' · pince' : ''}`, wx + 8 * dpr, wy + 16 * dpr);
  }
}
