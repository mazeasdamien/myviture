// Vite dev-server plugin that lets the page start and stop the VITURE bridge
// (bridge/viture_bridge.exe), so switching to the VITURE SDK needs no second terminal.
//   GET  /api/bridge/status   POST /api/bridge/start   POST /api/bridge/stop
//   GET  /api/calibration     factory calibration (the VIO engine does not expose it)

import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { Plugin } from 'vite';

const ROOT = resolve(import.meta.dirname, '..');
const EXE = resolve(ROOT, 'bridge/viture_bridge.exe');
const BUILD = resolve(ROOT, 'bridge/build.bat');
const VIO_DLL = 'C:\\Program Files\\VITURE\\SpaceWalker\\carina_vio.dll';
const CALIB = resolve(ROOT, 'calibration.bin');

let child: ChildProcess | null = null;
let ready = false;
let log: string[] = [];

function remember(line: string) {
  if (/predict not init|pose not init/.test(line)) return;
  log.push(line);
  if (log.length > 200) log.shift();
}

function run(cmd: string): Promise<number> {
  return new Promise(res => {
    const p = spawn(cmd, [], { cwd: ROOT, shell: true });
    p.stdout.on('data', d => String(d).split(/\r?\n/).forEach(l => l && remember(l)));
    p.stderr.on('data', d => String(d).split(/\r?\n/).forEach(l => l && remember(l)));
    p.on('exit', code => res(code ?? 1));
  });
}

/** Reads the factory calibration over USB once (Node `usb`), before the engine takes the device. */
async function cacheCalibration() {
  if (existsSync(CALIB)) return;
  try {
    const { WebUSB } = await import('usb');
    const { LumaUltra } = await import('../src/device/lumaUltra');
    const { PID_LUMA_ULTRA, VID } = await import('../src/device/protocol');
    const dev = (await new WebUSB({ allowAllDevices: true }).getDevices()).find(d => d.vendorId === VID && d.productId === PID_LUMA_ULTRA);
    if (!dev) return;
    const g = new LumaUltra(dev as unknown as USBDevice);
    await g.open();
    writeFileSync(CALIB, await g.readCalibration());
    await g.close();
  } catch (e) {
    remember(`calibration: ${(e as Error).message}`);
  }
}

async function start(): Promise<{ ok: boolean; error?: string }> {
  if (child && ready) return { ok: true };
  if (!existsSync(VIO_DLL)) return { ok: false, error: 'SpaceWalker (carina_vio.dll) introuvable' };
  if (!existsSync(EXE) && (await run(`"${BUILD}"`)) !== 0) return { ok: false, error: 'compilation du pont impossible' };

  log = [];
  ready = false;
  await cacheCalibration();
  const p = spawn(EXE, [], { cwd: ROOT });
  child = p;
  return new Promise(res => {
    const timer = setTimeout(() => res({ ok: false, error: 'le pont ne répond pas' }), 20000);
    const onLine = (l: string) => {
      remember(l);
      if (l.includes('ws://127.0.0.1')) {
        ready = true;
        clearTimeout(timer);
        res({ ok: true });
      }
    };
    p.stdout!.on('data', d => String(d).split(/\r?\n/).forEach(l => l && onLine(l)));
    p.stderr!.on('data', d => String(d).split(/\r?\n/).forEach(l => l && remember(l)));
    p.on('exit', code => {
      if (child === p) { child = null; ready = false; }
      clearTimeout(timer);
      res({ ok: false, error: log.filter(l => !l.startsWith('[CARINA]')).slice(-2).join(' ') || `code ${code}` });
    });
  });
}

function stop(): Promise<void> {
  const p = child;
  if (!p || p.exitCode !== null) return Promise.resolve();
  return new Promise(res => {
    p.once('exit', () => res());
    // The engine releases the USB interface when the process exits.
    spawn('taskkill', ['/pid', String(p.pid), '/t', '/f']);
  });
}

export function bridgePlugin(): Plugin {
  return {
    name: 'viture-bridge',
    configureServer(server) {
      server.middlewares.use('/api/calibration', (_req, res) => {
        if (!existsSync(CALIB)) { res.statusCode = 404; res.end(); return; }
        res.setHeader('Content-Type', 'application/octet-stream');
        res.end(readFileSync(CALIB));
      });
      server.middlewares.use('/api/bridge', async (req, res) => {
        let body: unknown;
        if (req.method === 'POST' && req.url === '/start') body = await start();
        else if (req.method === 'POST' && req.url === '/stop') { await stop(); body = { ok: true }; }
        else body = { available: existsSync(VIO_DLL), built: existsSync(EXE), running: !!child && ready, log: log.slice(-20) };
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify(body));
      });
      server.httpServer?.on('close', () => { stop(); });
      process.on('exit', () => child?.kill());
    },
  };
}
