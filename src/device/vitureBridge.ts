// Client of bridge/viture_bridge.exe: the VITURE VIO engine's 6DoF pose, plus the raw
// IMU, vsync and stereo tracking images it receives from the glasses (the engine owns
// the USB link, so these replace the WebUSB driver's streams). Same events as LumaUltra.

import type { ImuSample, StereoFrame } from './protocol';
import type { DeviceInfo } from './lumaUltra';

export const BRIDGE_URL = 'ws://127.0.0.1:8765';

export interface ViturePose {
  /** head -> world, column-major 4x4, OpenGL axes (x right, y up, z back), metres */
  m: Float32Array;
  /** engine status code (0 = tracking) */
  rc: number;
  /** absolute ms (system clock) when the pose was read */
  hostMs: number;
}

export interface BridgeEvents {
  imu?: (s: ImuSample) => void;
  frame?: (f: StereoFrame) => void;
  vsync?: (t: number) => void;
  pose?: (p: ViturePose) => void;
  error?: (e: unknown) => void;
}

export class VitureBridge {
  events: BridgeEvents = {};
  private ws: WebSocket | null = null;
  private displayWait: ((rc: number) => void) | null = null;
  private counter = 0;
  /** host ms minus device ms, smallest seen (IMU arrives with the least delay) */
  clockOffset = Infinity;

  open(url = BRIDGE_URL, timeoutMs = 8000): Promise<DeviceInfo> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';
      const timer = setTimeout(() => { ws.close(); reject(new Error('le pont VITURE ne répond pas')); }, timeoutMs);
      ws.onerror = () => { clearTimeout(timer); reject(new Error('pont VITURE injoignable')); };
      ws.onclose = () => { if (this.ws === ws) { this.ws = null; this.events.error?.(new Error('pont VITURE fermé')); } };
      ws.onmessage = e => {
        if (typeof e.data === 'string') {
          const m = JSON.parse(e.data);
          if (m.type === 'info') {
            clearTimeout(timer);
            this.ws = ws;
            resolve({ product: 'VITURE Luma Ultra (SDK VITURE)', serial: m.serial, firmware: m.firmware });
          } else if (m.type === 'display') {
            this.displayWait?.(m.rc);
            this.displayWait = null;
          }
          return;
        }
        this.onBinary(e.data as ArrayBuffer);
      };
    });
  }

  private onBinary(buf: ArrayBuffer) {
    const v = new DataView(buf);
    switch (v.getUint8(0)) {
      case 1: {
        const m = new Float32Array(16);
        for (let i = 0; i < 16; i++) m[i] = v.getFloat32(16 + 4 * i, true);
        this.events.pose?.({ m, rc: v.getInt32(4, true), hostMs: v.getFloat64(8, true) });
        break;
      }
      case 2: {
        const n = v.getUint32(4, true);
        for (let k = 0; k < n; k++) {
          const o = 8 + k * 40;
          const t = v.getFloat64(o, true), host = v.getFloat64(o + 8, true);
          this.clockOffset = Math.min(this.clockOffset, host - t * 1000);
          const f = (i: number) => v.getFloat32(o + 16 + 4 * i, true);
          this.events.imu?.({ counter: this.counter++, t, accel: [f(0), f(1), f(2)], gyro: [f(3), f(4), f(5)], tempC: NaN });
        }
        break;
      }
      case 3: {
        const w = v.getUint16(2, true), h = v.getUint16(4, true);
        const pixels = new Uint8Array(buf, 24, w * h).slice();
        this.events.frame?.({ camera: v.getUint8(1) as 0 | 1, t: v.getFloat64(8, true), width: w, height: h, pixels });
        break;
      }
      case 4:
        this.events.vsync?.(v.getFloat64(8, true));
        break;
    }
  }

  /** Device time (s) -> absolute ms on this machine's clock. */
  hostMs(deviceT: number) {
    return deviceT * 1000 + this.clockOffset;
  }

  async start() { /* the engine streams from the start */ }

  // The VIO engine drives the tracking cameras' exposure itself.
  async setExposure(_v: number) {}
  async setGain(_v: number) {}

  /** 0x41 2D, 0x42 3D. Resolves with the engine's return code (0 = done). */
  setDisplayMode(mode: number): Promise<number> {
    if (!this.ws) return Promise.resolve(-1);
    return new Promise(resolve => {
      this.displayWait = resolve;
      this.ws!.send(`display ${mode}`);
      setTimeout(() => { if (this.displayWait === resolve) { this.displayWait = null; resolve(-1); } }, 3000);
    });
  }

  /** Full VIO re-initialisation (tracking lost). */
  resetVio() {
    this.ws?.send('reset');
  }

  async close() {
    const ws = this.ws;
    this.ws = null;
    ws?.close();
  }
}
