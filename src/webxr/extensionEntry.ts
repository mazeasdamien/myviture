// Bundled into extension/xr-runtime.js (npm run build:extension) and injected by the
// extension into every page before the page's own scripts.

import { installRuntime } from './runtime';

installRuntime({ external: true });
