// viture.agentxr.app: the built dashboard (dist/) behind a password, and the /orange
// proxy to the lab's Orange gateway with its secret added here, never in the page.
//
// Secrets (wrangler secret put): SITE_PASSWORD, ORANGE_PROXY_SECRET. Var: ORANGE_GATEWAY.
// A session is a cookie "<expiry>.<HMAC(SITE_PASSWORD, expiry)>": changing the password
// signs everyone out.

const COOKIE = 'mv_session';
const SESSION_DAYS = 30;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === '/login' && request.method === 'POST') return login(request, env);
    if (url.pathname === '/logout') return new Response(null, { status: 303, headers: { location: '/', 'set-cookie': `${COOKIE}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax` } });

    if (!(await signedIn(request, env))) {
      if (url.pathname.startsWith('/orange/') || url.pathname.startsWith('/api/')) return json({ error: { message: 'non connecté' } }, 401);
      return loginPage(401);
    }

    if (url.pathname.startsWith('/orange/')) return orange(request, url, env);
    // The VITURE engine and its bridge run on the PC next to the glasses: local only.
    if (url.pathname.startsWith('/api/bridge/')) return json({ ok: false, available: false, error: 'suivi VITURE seulement en local (npm run dev sur le PC des lunettes) : choisis « Suivi maison »' });
    if (url.pathname.startsWith('/api/')) return json({ error: 'local only' }, 404);
    return env.ASSETS.fetch(request);
  },
};

// ------------------------------------------------------------------ session

async function hmac(key, text) {
  const k = await crypto.subtle.importKey('raw', new TextEncoder().encode(key), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const sig = await crypto.subtle.sign('HMAC', k, new TextEncoder().encode(text));
  return [...new Uint8Array(sig)].map(b => b.toString(16).padStart(2, '0')).join('');
}

/** Constant-time comparison of two strings. */
function same(a, b) {
  const x = new TextEncoder().encode(a), y = new TextEncoder().encode(b);
  let diff = x.length ^ y.length;
  for (let i = 0; i < Math.max(x.length, y.length); i++) diff |= (x[i] ?? 0) ^ (y[i] ?? 0);
  return diff === 0;
}

async function signedIn(request, env) {
  if (!env.SITE_PASSWORD) return false;
  const m = (request.headers.get('cookie') ?? '').match(new RegExp(`(?:^|;\\s*)${COOKIE}=(\\d+)\\.([0-9a-f]{64})`));
  if (!m || Number(m[1]) < Date.now()) return false;
  return same(m[2], await hmac(env.SITE_PASSWORD, m[1]));
}

async function login(request, env) {
  const form = await request.formData().catch(() => null);
  const password = String(form?.get('password') ?? '');
  if (!env.SITE_PASSWORD || !same(password, env.SITE_PASSWORD)) {
    // Slows down guessing.
    await new Promise(r => setTimeout(r, 1000));
    return loginPage(401, 'Mot de passe incorrect.');
  }
  const expiry = String(Date.now() + SESSION_DAYS * 864e5);
  const cookie = `${COOKIE}=${expiry}.${await hmac(env.SITE_PASSWORD, expiry)}; Path=/; Max-Age=${SESSION_DAYS * 86400}; HttpOnly; Secure; SameSite=Lax`;
  return new Response(null, { status: 303, headers: { location: '/', 'set-cookie': cookie } });
}

function loginPage(status, error = '') {
  const html = `<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>myViture</title><style>
body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0d1117;color:#e6edf3;font:15px system-ui,sans-serif}
form{display:grid;gap:12px;width:min(320px,90vw)}h1{font-size:20px;margin:0 0 4px}
input,button{font:inherit;padding:10px 12px;border-radius:8px;border:1px solid #30363d;background:#161b22;color:inherit}
button{background:#ff7900;border-color:#ff7900;color:#000;font-weight:600;cursor:pointer}p{margin:0;color:#ff7b72}
</style></head><body><form method="post" action="/login"><h1>myViture</h1>
<input type="password" name="password" placeholder="Mot de passe" autocomplete="current-password" autofocus required>
<button>Entrer</button>${error ? `<p>${error}</p>` : ''}</form></body></html>`;
  return new Response(html, { status, headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' } });
}

// ------------------------------------------------------------------ Orange gateway

async function orange(request, url, env) {
  if (!env.ORANGE_GATEWAY || !env.ORANGE_PROXY_SECRET) return json({ error: { message: 'passerelle Orange non configurée' } }, 503);
  // Same path as the dev server's proxy: /orange/v1/... on the gateway.
  const target = new URL(url.pathname + url.search, env.ORANGE_GATEWAY);
  const headers = new Headers();
  for (const h of ['content-type', 'accept']) if (request.headers.has(h)) headers.set(h, request.headers.get(h));
  headers.set('x-orange-auth', env.ORANGE_PROXY_SECRET);
  const res = await fetch(target, { method: request.method, headers, body: ['GET', 'HEAD'].includes(request.method) ? undefined : request.body });
  return new Response(res.body, { status: res.status, headers: res.headers });
}

const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
