// Decoder for the factory calibration blob (protobuf without a published schema).
// Field numbers were mapped by comparing the blob with the vendor driver's output.

export interface CameraCalib {
  name: string;
  model: string;
  distortionModel: string;
  /** fx, fy, cx, cy */
  intrinsics: number[];
  distortion: number[];
  /** 4x4 row-major, IMU -> camera */
  T_cam_imu: number[];
  timeshift: number;
}

export interface Calibration {
  serial: string;
  imuRateHz: number;
  cameraRateHz: number;
  /** 3x3 row-major rotation, IMU frame -> OpenGL head frame (x right, y up, z back) */
  R_gl_imu: number[];
  accelBias: number[];
  accelScale: number[];
  gyroBias: number[];
  gyroLsb: number;
  accelLsb: number;
  cameras: CameraCalib[];
}

/** Mounting of the IMU in the Luma Ultra, used until the device calibration is read. */
export const DEFAULT_R_GL_IMU = [-1, 0, 0, 0, 0, 1, 0, 1, 0];

type Field = { no: number; wire: number; value: number | Uint8Array };

function* fields(b: Uint8Array): Generator<Field> {
  const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let i = 0;
  const varint = () => {
    let r = 0, s = 0;
    for (;;) {
      const x = b[i++];
      r += (x & 0x7f) * 2 ** s;
      s += 7;
      if (x < 0x80) return r;
    }
  };
  while (i < b.length) {
    const key = varint();
    const no = Math.floor(key / 8), wire = key & 7;
    if (wire === 0) yield { no, wire, value: varint() };
    else if (wire === 1) { yield { no, wire, value: v.getFloat64(i, true) }; i += 8; }
    else if (wire === 5) { yield { no, wire, value: v.getFloat32(i, true) }; i += 4; }
    else if (wire === 2) { const n = varint(); yield { no, wire, value: b.subarray(i, i + n) }; i += n; }
    else throw new Error(`unsupported wire type ${wire}`);
  }
}

const doubles = (b: Uint8Array) => Array.from(new Float64Array(b.slice().buffer));
const floats = (b: Uint8Array) => Array.from(new Float32Array(b.slice().buffer));
const text = (b: Uint8Array) => new TextDecoder().decode(b);
const bytes = (f: Field) => (f.value instanceof Uint8Array ? f.value : new Uint8Array());

function parseCamera(b: Uint8Array): CameraCalib {
  const cam: CameraCalib = { name: '', model: '', distortionModel: '', intrinsics: [], distortion: [], T_cam_imu: [], timeshift: 0 };
  for (const f of fields(b)) {
    switch (f.no) {
      case 1: cam.T_cam_imu = doubles(bytes(f)); break;
      case 4: cam.model = text(bytes(f)); break;
      case 5: cam.distortion = doubles(bytes(f)); break;
      case 6: cam.distortionModel = text(bytes(f)); break;
      case 7: cam.intrinsics = doubles(bytes(f)); break;
      case 9: cam.name = text(bytes(f)); break;
      case 10: cam.timeshift = f.value as number; break;
    }
  }
  return cam;
}

export function parseCalibration(blob: Uint8Array): Calibration {
  const c: Calibration = {
    serial: '', imuRateHz: 0, cameraRateHz: 0, R_gl_imu: DEFAULT_R_GL_IMU,
    accelBias: [0, 0, 0], accelScale: [1, 1, 1], gyroBias: [0, 0, 0],
    gyroLsb: 0, accelLsb: 0, cameras: [],
  };
  for (const f of fields(blob)) {
    switch (f.no) {
      case 4: c.serial = text(bytes(f)); break;
      case 9: c.cameraRateHz = f.value as number; break;
      case 10: c.imuRateHz = f.value as number; break;
      case 80: {
        // Nested message holding a 4x4 float matrix (row-major).
        for (const g of fields(bytes(f))) {
          if (g.no !== 1) continue;
          const m = floats(bytes(g));
          if (m.length === 16) c.R_gl_imu = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
        }
        break;
      }
      case 82: c.accelBias = doubles(bytes(f)); break;
      case 83: c.accelScale = doubles(bytes(f)); break;
      case 84: c.gyroBias = doubles(bytes(f)); break;
      case 85: c.cameras.push(parseCamera(bytes(f))); break;
      case 89: c.gyroLsb = f.value as number; break;
      case 90: c.accelLsb = f.value as number; break;
    }
  }
  return c;
}
