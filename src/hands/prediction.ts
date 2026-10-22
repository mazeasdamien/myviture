// Hand latency compensation: hands are measured on images that are already
// ~50-100 ms old when they reach the page. Each joint is extrapolated to "now"
// with a smoothed velocity, capped so a wrong estimate cannot fling the hand away.

import type { Hand } from './handTracker';
import type { Vec3 } from './stereoGeometry';

const MAX_LEAD = 0.12; // s
const VEL_SMOOTH = 0.5;

interface Track {
  stamp: number;
  joints: Vec3[];
  vel: Vec3[];
}

export const absNow = () => performance.timeOrigin + performance.now();

export class HandPredictor {
  private tracks = new Map<string, Track>();
  latencyMs = 0;

  /** New measurement from the hands worker (joints in the world frame). */
  push(hands: Hand[]) {
    const seen = new Set<string>();
    for (const h of hands) {
      seen.add(h.side);
      const stamp = h.stamp ?? absNow();
      const prev = this.tracks.get(h.side);
      if (prev && stamp > prev.stamp) {
        const dt = (stamp - prev.stamp) / 1000;
        if (dt < 0.25) {
          const vel = h.joints.map((j, i) => j.map((v, k) => {
            const raw = (v - prev.joints[i][k]) / dt;
            return prev.vel[i][k] * VEL_SMOOTH + raw * (1 - VEL_SMOOTH);
          }) as Vec3);
          this.tracks.set(h.side, { stamp, joints: h.joints, vel });
          continue;
        }
      }
      this.tracks.set(h.side, { stamp, joints: h.joints, vel: h.joints.map(() => [0, 0, 0] as Vec3) });
    }
    for (const k of [...this.tracks.keys()]) if (!seen.has(k)) this.tracks.delete(k);
    if (hands[0]?.stamp) this.latencyMs = absNow() - hands[0].stamp;
  }

  /** Hands extrapolated to the current time. */
  predict(hands: Hand[]): Hand[] {
    const now = absNow();
    return hands.map(h => {
      const t = this.tracks.get(h.side);
      if (!t) return h;
      const lead = Math.min(MAX_LEAD, Math.max(0, (now - t.stamp) / 1000));
      return { ...h, joints: t.joints.map((j, i) => [j[0] + t.vel[i][0] * lead, j[1] + t.vel[i][1] * lead, j[2] + t.vel[i][2] * lead] as Vec3) };
    });
  }
}
