// Hardware smoke test: runs the browser driver in Node through the `usb` package's WebUSB.
//   npm run probe [seconds]
import { WebUSB } from 'usb';
import { writeFileSync } from 'node:fs';
import { LumaUltra } from '../src/device/lumaUltra';
import { PID_LUMA_ULTRA, VID } from '../src/device/protocol';

const secs = Number(process.argv[2] ?? 5);
const webusb = new WebUSB({ allowAllDevices: true });
const dev = (await webusb.getDevices()).find(d => d.vendorId === VID && d.productId === PID_LUMA_ULTRA);
if (!dev) throw new Error('Luma Ultra not found');

const g = new LumaUltra(dev as unknown as USBDevice);
console.log('info', await g.open());
const calib = await g.readCalibration();
writeFileSync('calibration.bin', calib);
console.log('calibration bytes', calib.length);

let imu = 0, frames = [0, 0], vs = 0, last: any, saved = false;
g.events = {
  imu: s => { imu++; last = s; },
  frame: f => {
    frames[f.camera]++;
    if (!saved && frames[f.camera] > 20) {
      saved = true;
      writeFileSync(`frame${f.camera}.pgm`, Buffer.concat([Buffer.from(`P5 ${f.width} ${f.height} 255\n`), Buffer.from(f.pixels)]));
    }
  },
  vsync: () => vs++,
  error: e => console.error('stream error', e),
};
await g.start();
await g.setExposure(0x0800);
for (let i = 0; i < secs; i++) {
  await new Promise(r => setTimeout(r, 1000));
  const f = (x: number[]) => x.map(v => v.toFixed(3)).join(' ');
  console.log(`t=${i + 1} imu=${imu}/s cams=${frames.join('/')} vsync=${vs}  gyro=[${f(last?.gyro ?? [])}] accel=[${f(last?.accel ?? [])}] ${last?.tempC.toFixed(1)}°C`);
  imu = 0; frames = [0, 0]; vs = 0;
}
await g.close();
process.exit(0);
