// VITURE Luma Ultra (USB 35ca:1104) wire protocol, decoded from captures.
//
// One vendor interface (#0) with these endpoints:
//   0x81 interrupt IN  - IMU samples, 1 kHz, 28-byte packets
//   0x82 bulk      IN  - stereo tracking camera frames, 16 KiB chunks
//   0x83 interrupt IN  - display vsync, 60 Hz
//   0x04 bulk      OUT - commands
//   0x85 bulk      IN  - command replies
//
// Every stream packet starts with `aa 8f 00 <type>`. Commands are
// `fa 55 <cmd> <flag> <len> <payload...>`, replies `fa 55 <cmd> <status> <lenHi> <lenLo> <data...>`.

export const VID = 0x35ca;
export const PID_LUMA_ULTRA = 0x1104;

export const EP_IMU = 1;
export const EP_IMAGE = 2;
export const EP_VSYNC = 3;
export const EP_CMD_OUT = 4;
export const EP_CMD_IN = 5;

export const Cmd = {
  STATUS: 0x9f,
  FIRMWARE: 0x80,
  SERIAL: 0xed,
  RESET_STREAM: 0x97,
  START_STREAM: 0x95,
  STOP_STREAM: 0x94,
  SET_EXPOSURE: 0x90,
  SET_GAIN: 0x92,
  CALIB_SIZE: 0xe4,
  CALIB_READ: 0xe5,
} as const;

const MAGIC0 = 0xaa;
const MAGIC1 = 0x8f;

export const TYPE_IMU = 0x88;
export const TYPE_VSYNC = 0x91;

// Full-scale ranges of the on-board IMU.
const GYRO_RAD_PER_LSB = (2000 / 32768) * (Math.PI / 180);
const ACCEL_MS2_PER_LSB = (4 * 9.80665) / 32768;

export const CAM_WIDTH = 640;
export const CAM_HEIGHT = 480;
const IMAGE_HEADER = 12;
const IMAGE_CHUNK_PAYLOAD = 16372;

export interface ImuSample {
  counter: number;
  /** Device clock, seconds (100 µs resolution, wraps after ~5 days). */
  t: number;
  /** rad/s, sensor frame */
  gyro: [number, number, number];
  /** m/s², sensor frame */
  accel: [number, number, number];
  tempC: number;
}

export function buildCommand(cmd: number, payload: number[] = [], flag = 0): Uint8Array {
  return new Uint8Array([0xfa, 0x55, cmd, flag, payload.length, ...payload]);
}

export interface Reply {
  cmd: number;
  status: number;
  length: number;
  data: Uint8Array;
}

export function parseReply(buf: DataView): Reply | null {
  if (buf.byteLength < 6 || buf.getUint8(0) !== 0xfa || buf.getUint8(1) !== 0x55) return null;
  const length = buf.getUint16(4, false);
  const data = new Uint8Array(buf.buffer, buf.byteOffset + 6, buf.byteLength - 6);
  return { cmd: buf.getUint8(2), status: buf.getUint8(3), length, data };
}

export function isStreamPacket(v: DataView): boolean {
  return v.byteLength >= 12 && v.getUint8(0) === MAGIC0 && v.getUint8(1) === MAGIC1;
}

export function parseImu(v: DataView): ImuSample | null {
  if (v.byteLength < 26 || !isStreamPacket(v) || v.getUint8(3) !== TYPE_IMU) return null;
  const i16 = (o: number) => v.getInt16(o, true);
  return {
    counter: v.getUint32(4, true),
    t: v.getUint32(8, true) * 1e-4,
    gyro: [i16(12) * GYRO_RAD_PER_LSB, i16(14) * GYRO_RAD_PER_LSB, i16(16) * GYRO_RAD_PER_LSB],
    accel: [i16(18) * ACCEL_MS2_PER_LSB, i16(20) * ACCEL_MS2_PER_LSB, i16(22) * ACCEL_MS2_PER_LSB],
    tempC: i16(24) / 132.48 + 25,
  };
}

export function parseVsync(v: DataView): number | null {
  if (!isStreamPacket(v) || v.getUint8(3) !== TYPE_VSYNC) return null;
  return v.getUint32(8, true) * 1e-4;
}

export interface StereoFrame {
  /** 0 = left, 1 = right */
  camera: 0 | 1;
  t: number;
  width: number;
  height: number;
  pixels: Uint8Array;
}

/**
 * Reassembles camera frames from 0x82 chunks. The type byte's high nibble is the
 * stream (1 = left camera, 2 = right camera, 5..7 = on-device feature data) and
 * the low nibble the position (1 = first, 2 = middle, 5 = last).
 */
export class FrameAssembler {
  private buffers = [new Uint8Array(CAM_WIDTH * CAM_HEIGHT), new Uint8Array(CAM_WIDTH * CAM_HEIGHT)];
  private filled = [0, 0];

  push(v: DataView): StereoFrame | null {
    if (!isStreamPacket(v)) return null;
    const type = v.getUint8(3);
    const stream = type >> 4;
    if (stream !== 1 && stream !== 2) return null;
    const cam = (stream - 1) as 0 | 1;
    const index = v.getUint16(4, true);
    const len = Math.min(v.getUint16(6, true), v.byteLength - IMAGE_HEADER);
    const t = v.getUint32(8, true) * 1e-4;
    const buf = this.buffers[cam];
    const offset = index * IMAGE_CHUNK_PAYLOAD;
    if ((type & 0x0f) === 1) this.filled[cam] = 0;
    if (offset + len > buf.length) return null;
    buf.set(new Uint8Array(v.buffer, v.byteOffset + IMAGE_HEADER, len), offset);
    this.filled[cam] += len;
    if ((type & 0x04) === 0) return null;
    const complete = this.filled[cam] === buf.length;
    this.filled[cam] = 0;
    if (!complete) return null;
    return { camera: cam, t, width: CAM_WIDTH, height: CAM_HEIGHT, pixels: buf.slice() };
  }
}

/** Every chunk on 0x82 occupies exactly this many bytes, whatever its payload length. */
export const IMAGE_CHUNK_WIRE = 16384;

function isChunkHeader(b: Uint8Array, i: number): boolean {
  if (b[i] !== MAGIC0 || b[i + 1] !== MAGIC1 || b[i + 2] !== 0) return false;
  const type = b[i + 3], stream = type >> 4, pos = type & 0x0f;
  const len = b[i + 6] | (b[i + 7] << 8);
  return stream >= 1 && stream <= 7 && (pos === 1 || pos === 2 || pos === 5) && b[i + 5] === 0 && len <= IMAGE_CHUNK_PAYLOAD;
}

/**
 * Re-cuts the 0x82 byte stream into chunks. Transfers normally line up with chunks,
 * but a stream interrupted mid-chunk (page reload, unplug) leaves the pipe shifted
 * by a constant offset; this finds the next header and stays aligned from there.
 */
export class ChunkStream {
  private carry = new Uint8Array(IMAGE_CHUNK_WIRE * 3);
  private n = 0;

  push(v: DataView): DataView[] {
    const bytes = new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
    if (this.n === 0 && bytes.length === IMAGE_CHUNK_WIRE && isChunkHeader(bytes, 0)) return [v];

    if (this.n + bytes.length > this.carry.length) this.n = 0;
    this.carry.set(bytes, this.n);
    this.n += bytes.length;

    const out: DataView[] = [];
    let start = 0;
    while (this.n - start >= IMAGE_HEADER) {
      if (isChunkHeader(this.carry, start)) {
        if (this.n - start < IMAGE_CHUNK_WIRE) break;
        out.push(new DataView(this.carry.slice(start, start + IMAGE_CHUNK_WIRE).buffer));
        start += IMAGE_CHUNK_WIRE;
        continue;
      }
      let next = -1;
      for (let i = start + 1; i <= this.n - IMAGE_HEADER; i++) {
        if (this.carry[i] === MAGIC0 && isChunkHeader(this.carry, i)) { next = i; break; }
      }
      start = next < 0 ? this.n - (IMAGE_HEADER - 1) : next;
    }
    this.carry.copyWithin(0, start, this.n);
    this.n -= start;
    return out;
  }
}
