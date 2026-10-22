// Puts the glasses back in 2D (1920x1200).
import { WebUSB } from 'usb';
import { LumaUltra } from '../src/device/lumaUltra';
import { DisplayMode } from '../src/device/mcu';
import { PID_LUMA_ULTRA, VID } from '../src/device/protocol';
const dev = (await new WebUSB({ allowAllDevices: true }).getDevices()).find(d => d.vendorId === VID && d.productId === PID_LUMA_ULTRA);
const g = new LumaUltra(dev as unknown as USBDevice);
await g.open();
await g.setDisplayMode(DisplayMode.MODE_2D);
await new Promise(r => setTimeout(r, 1500));
await g.close();
process.exit(0);
