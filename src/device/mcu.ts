// Glasses MCU messages (display mode, brightness, volume...), carried on the second
// bulk pair of 35ca:1104 (OUT 0x06 / IN 0x87).
//
// 512-byte frame, little-endian:
//   0  u16 header 0xFEFF (host -> glasses; replies use 0xFDFF / 0xFEFF)
//   2  u16 CRC-16/XMODEM over bytes [4, 4 + len + 2)
//   4  u16 len = 12 + payload length
//   6  u32 0, 10 u32 0 (timestamps)
//   14 u16 message id, 16 u16 0
//   18 payload

export const EP_MCU_OUT = 6;
export const EP_MCU_IN = 7;
const FRAME = 512;

export const McuMsg = {
  BRIGHTNESS: 6,
  /** payload: display mode code, 0x41 = 1920x1200 2D, 0x42 = 3840x1200 3D (60 Hz) */
  DISPLAY_2D_3D: 8,
  FILM: 14,
  VOLUME: 51,
} as const;

export const DisplayMode = { MODE_2D: 0x41, MODE_3D: 0x42 } as const;

function crc16(buf: Uint8Array, offset: number, len: number) {
  let crc = 0;
  for (let i = 0; i < len; i++) {
    crc ^= buf[offset + i] << 8;
    for (let b = 0; b < 8; b++) crc = crc & 0x8000 ? ((crc << 1) ^ 0x1021) & 0xffff : (crc << 1) & 0xffff;
  }
  return crc;
}

export function buildMcuMessage(msgId: number, payload: number[] = []): Uint8Array {
  const f = new Uint8Array(FRAME);
  const v = new DataView(f.buffer);
  v.setUint16(0, 0xfeff, true);
  const len = 12 + payload.length;
  v.setUint16(4, len, true);
  v.setUint16(14, msgId, true);
  f.set(payload, 18);
  v.setUint16(2, crc16(f, 4, len + 2), true);
  return f;
}

export const toHex = (b: Uint8Array) => Array.from(b, x => x.toString(16).padStart(2, '0')).join('');
