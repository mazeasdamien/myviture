// WebUSB driver for the VITURE Luma Ultra. Only depends on the WebUSB `USBDevice`
// interface, so it runs in Chrome/Edge and in Node through the `usb` package.

import {
  Cmd, EP_CMD_IN, EP_CMD_OUT, EP_IMAGE, EP_IMU, EP_VSYNC, PID_LUMA_ULTRA, VID,
  ChunkStream, FrameAssembler, buildCommand, parseImu, parseReply, parseVsync,
  type ImuSample, type Reply, type StereoFrame,
} from './protocol';
import { EP_MCU_IN, EP_MCU_OUT, McuMsg, buildMcuMessage } from './mcu';

export interface DeviceInfo {
  product: string;
  serial: string;
  firmware: string;
}

export interface LumaUltraEvents {
  imu?: (s: ImuSample) => void;
  frame?: (f: StereoFrame) => void;
  vsync?: (t: number) => void;
  /** raw MCU frame from 0x87 */
  mcu?: (v: DataView) => void;
  error?: (e: unknown) => void;
}

const IMU_QUEUE = 8;
const IMAGE_QUEUE = 6;
const IMAGE_TRANSFER = 16384;

export const USB_FILTERS: USBDeviceFilter[] = [{ vendorId: VID, productId: PID_LUMA_ULTRA }];

export class LumaUltra {
  readonly device: USBDevice;
  events: LumaUltraEvents = {};
  private running = false;
  private cmdChain: Promise<unknown> = Promise.resolve();
  private lastCounter = -1;
  private assembler = new FrameAssembler();
  private chunks = new ChunkStream();

  constructor(device: USBDevice) {
    this.device = device;
  }

  static async request(): Promise<LumaUltra> {
    const dev = await navigator.usb.requestDevice({ filters: USB_FILTERS });
    return new LumaUltra(dev);
  }

  /** Devices this origin was already granted, so a reload can reconnect without the chooser. */
  static async known(): Promise<LumaUltra | null> {
    const devs = await navigator.usb.getDevices();
    const dev = devs.find(d => d.vendorId === VID && d.productId === PID_LUMA_ULTRA);
    return dev ? new LumaUltra(dev) : null;
  }

  async open(): Promise<DeviceInfo> {
    const d = this.device;
    if (!d.opened) await d.open();
    if (!d.configuration) await d.selectConfiguration(1);
    await d.claimInterface(0);
    // A previous session may have left the stream running.
    await this.command(Cmd.STOP_STREAM).catch(() => undefined);
    const fw = await this.command(Cmd.FIRMWARE);
    const sn = await this.command(Cmd.SERIAL);
    return {
      product: d.productName ?? 'VITURE Luma Ultra',
      serial: new TextDecoder().decode(sn.data.subarray(0, sn.length)).replace(/\0+$/, ''),
      firmware: Array.from(fw.data.subarray(0, fw.length)).join('.'),
    };
  }

  /** Sends one command and waits for its reply. Commands are serialised. */
  command(cmd: number, payload: number[] = [], flag = 0): Promise<Reply> {
    return this.send(buildCommand(cmd, payload, flag), cmd);
  }

  /** Factory calibration blob (protobuf, ~12 KB): camera intrinsics/extrinsics, IMU noise model. */
  async readCalibration(): Promise<Uint8Array> {
    const total = (await this.command(Cmd.CALIB_SIZE)).length;
    const out = new Uint8Array(total);
    let off = 0;
    while (off < total) {
      const rest = total - off;
      // Full chunks: flag 1 = "256 more bytes". Last chunk: flag 0, length byte = remaining size.
      const frame = rest >= 256
        ? buildCommand(Cmd.CALIB_READ, [], 1)
        : new Uint8Array([0xfa, 0x55, Cmd.CALIB_READ, 0, rest]);
      const r = await this.send(frame, Cmd.CALIB_READ);
      const n = Math.min(r.length, r.data.length, rest);
      if (n === 0) break;
      out.set(r.data.subarray(0, n), off);
      off += n;
    }
    return out.subarray(0, off);
  }

  private send(frame: Uint8Array, cmd: number): Promise<Reply> {
    const run = async () => {
      await this.device.transferOut(EP_CMD_OUT, frame as BufferSource);
      for (let i = 0; i < 4; i++) {
        const v = await this.nextReply(`reply to 0x${cmd.toString(16)}`);
        const reply = v && parseReply(v);
        if (reply && reply.cmd === cmd) return reply;
      }
      throw new Error(`no reply to command 0x${cmd.toString(16)}`);
    };
    const p = this.cmdChain.then(run, run);
    this.cmdChain = p.catch(() => undefined);
    return p;
  }

  // WebUSB transfers cannot be cancelled, so a reader that timed out is kept and
  // reused; otherwise it would swallow the next command's reply.
  private replyRead: Promise<USBInTransferResult> | null = null;

  private async nextReply(what: string): Promise<DataView | undefined> {
    this.replyRead ??= this.device.transferIn(EP_CMD_IN, 512);
    const r = await withTimeout(this.replyRead, 1500, what);
    this.replyRead = null;
    return r.data;
  }

  async start(): Promise<void> {
    await this.command(Cmd.RESET_STREAM);
    await this.command(Cmd.START_STREAM);
    this.running = true;
    this.lastCounter = -1;
    for (let i = 0; i < IMU_QUEUE; i++) this.pump(EP_IMU, 64, v => this.onImuPacket(v));
    for (let i = 0; i < IMAGE_QUEUE; i++) this.pump(EP_IMAGE, IMAGE_TRANSFER, v => this.onImagePacket(v));
    // MCU replies and events (display mode, buttons) arrive on 0x87; drain them.
    this.pump(EP_MCU_IN, 512, v => this.events.mcu?.(v));
    this.pump(EP_VSYNC, 64, v => {
      const t = parseVsync(v);
      if (t !== null) this.events.vsync?.(t);
    });
  }

  /** Camera exposure, raw sensor units (the vendor auto-exposure moves it within ~0x200..0x1000). */
  setExposure(value: number) {
    const v = Math.max(1, Math.min(0xffff, Math.round(value)));
    return this.command(Cmd.SET_EXPOSURE, [v >> 8, v & 0xff]);
  }

  /** Raw MCU message (see mcu.ts). */
  sendMcu(msgId: number, payload: number[] = []) {
    return this.device.transferOut(EP_MCU_OUT, buildMcuMessage(msgId, payload) as BufferSource);
  }

  /** 0x41 = 1920x1200 2D, 0x42 = 3840x1200 side-by-side 3D: the PC sees a new resolution. */
  setDisplayMode(mode: number) {
    return this.sendMcu(McuMsg.DISPLAY_2D_3D, [mode]);
  }

  setGain(value: number) {
    return this.command(Cmd.SET_GAIN, [Math.max(0, Math.min(255, Math.round(value)))]);
  }

  async stop(): Promise<void> {
    if (!this.running) return;
    this.running = false;
    await this.command(Cmd.STOP_STREAM).catch(() => undefined);
  }

  async close(): Promise<void> {
    await this.stop();
    try {
      await this.device.releaseInterface(0);
    } catch { /* already released */ }
    try {
      await this.device.close();
    } catch { /* transfers still draining */ }
  }

  private onImuPacket(v: DataView) {
    const s = parseImu(v);
    // The interrupt endpoint occasionally repeats a packet.
    if (!s || s.counter === this.lastCounter) return;
    this.lastCounter = s.counter;
    this.events.imu?.(s);
  }

  private onImagePacket(v: DataView) {
    for (const chunk of this.chunks.push(v)) {
      const f = this.assembler.push(chunk);
      if (f) this.events.frame?.(f);
    }
  }

  private async pump(ep: number, size: number, handle: (v: DataView) => void) {
    while (this.running) {
      try {
        const r = await this.device.transferIn(ep, size);
        if (r.data && r.data.byteLength) handle(r.data);
      } catch (e) {
        if (!this.running) return;
        this.running = false;
        this.events.error?.(e);
        return;
      }
    }
  }
}

function withTimeout<T>(p: Promise<T>, ms: number, what: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const id = setTimeout(() => reject(new Error(`timeout waiting for ${what}`)), ms);
    p.then(v => { clearTimeout(id); resolve(v); }, e => { clearTimeout(id); reject(e); });
  });
}
