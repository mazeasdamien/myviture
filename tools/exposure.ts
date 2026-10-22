// Checks that exposure and gain commands change the tracking-camera brightness.
import { WebUSB } from 'usb';
import { LumaUltra } from '../src/device/lumaUltra';
import { PID_LUMA_ULTRA, VID } from '../src/device/protocol';

const dev = (await new WebUSB({ allowAllDevices: true }).getDevices()).find(d => d.vendorId === VID && d.productId === PID_LUMA_ULTRA);
const g = new LumaUltra(dev as unknown as USBDevice);
await g.open();
let mean = 0;
g.events = { frame: f => { if (f.camera === 0) { let s = 0; for (let i = 0; i < f.pixels.length; i += 16) s += f.pixels[i]; mean = s / (f.pixels.length / 16); } } };
await g.start();
const wait = (ms: number) => new Promise(r => setTimeout(r, ms));
for (const [exp, gain] of [[0x100, 64], [0x800, 64], [0x2000, 64], [0x800, 16], [0x800, 200]]) {
  const r1 = await g.setExposure(exp); const r2 = await g.setGain(gain);
  await wait(800);
  console.log(`exposure ${exp} gain ${gain} -> mean ${mean.toFixed(1)} (status ${r1.status}/${r2.status})`);
}
await g.close();
process.exit(0);
