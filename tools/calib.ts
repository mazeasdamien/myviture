// Prints the decoded factory calibration saved by `npm run probe`.
import { readFileSync } from 'node:fs';
import { parseCalibration } from '../src/device/calibration';
console.log(JSON.stringify(parseCalibration(new Uint8Array(readFileSync(process.argv[2] ?? 'calibration.bin'))), (k, v) => typeof v === 'number' ? +v.toFixed(6) : v, 1).replace(/\n\s+(-?[\d.e+-]+)(,?)/g, ' $1$2'));
