import { defineConfig, loadEnv } from 'vite';
import { bridgePlugin } from './tools/bridgePlugin';

// WebUSB and camera access need a secure context: http://localhost qualifies.
//
// /orange/* goes to the lab's orange-gateway worker (Orange LLM gateway: Gemini vision,
// Whisper, Gemini TTS). The worker's shared secret is added here, on the dev server, from
// .env.local (ORANGE_GATEWAY, ORANGE_PROXY_SECRET): it never reaches the page.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'ORANGE_');
  const proxy = env.ORANGE_GATEWAY
    ? {
      '/orange': {
        target: env.ORANGE_GATEWAY,
        changeOrigin: true,
        headers: { 'x-orange-auth': env.ORANGE_PROXY_SECRET ?? '' },
      },
    }
    : undefined;
  return {
    plugins: [bridgePlugin()],
    server: { port: 5180, strictPort: true, host: 'localhost', proxy },
    preview: { proxy },
    build: { rollupOptions: { input: { main: 'index.html', glasses: 'glasses.html', webxr: 'webxr.html' } } },
  };
});
