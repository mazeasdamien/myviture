import { defineConfig } from 'vite';

// The WebXR runtime as one classic script for the extension's MAIN-world content script.
export default defineConfig({
  build: {
    outDir: 'extension',
    emptyOutDir: false,
    copyPublicDir: false,
    lib: { entry: 'src/webxr/extensionEntry.ts', formats: ['iife'], name: 'myvitureXr', fileName: () => 'xr-runtime.js' },
  },
});
