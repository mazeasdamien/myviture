// Switches the glasses to side-by-side 3D and back through the MCU channel (0x06/0x87).
import { WebUSB } from 'usb';
import { execSync } from 'node:child_process';
import { LumaUltra } from '../src/device/lumaUltra';
import { DisplayMode } from '../src/device/mcu';
import { PID_LUMA_ULTRA, VID } from '../src/device/protocol';

const screens = () => execSync(`powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Screen]::AllScreens | % { $_.DeviceName + ' ' + $_.Bounds.Width + 'x' + $_.Bounds.Height }"`).toString().trim();
const dev = (await new WebUSB({ allowAllDevices: true }).getDevices()).find(d => d.vendorId === VID && d.productId === PID_LUMA_ULTRA);
const g = new LumaUltra(dev as unknown as USBDevice);
await g.open();
g.events = { mcu: v => console.log('mcu reply', Buffer.from(v.buffer, v.byteOffset, Math.min(24, v.byteLength)).toString('hex')) };
await g.start();
console.log('before:\n' + screens());
await g.setDisplayMode(DisplayMode.MODE_3D);
await new Promise(r => setTimeout(r, 4000));
console.log('3D:\n' + screens());
await g.setDisplayMode(DisplayMode.MODE_2D);
await new Promise(r => setTimeout(r, 4000));
console.log('back to 2D:\n' + screens());
await g.close();
process.exit(0);
