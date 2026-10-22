// Orange LLM gateway, through the dev server's /orange proxy (which adds the gateway's
// secret): Gemini for vision and answers, Whisper for speech-to-text, Gemini TTS for
// speech. OpenAI-shaped API (LiteLLM).

/** vision is the one picked on the dashboard (any vision chat model of the gateway). */
export const MODELS = {
  vision: 'vertex_ai/gemini-3.5-flash-lite',
  transcribe: 'openai/whisper-1',
  speech: 'vertex_ai/gemini-2.5-flash-tts',
};
export const VOICE = 'Kore';

const BASE = '/orange/v1';

// ------------------------------------------------------------------ costs
//
// LiteLLM reports each call's cost (USD) in x-litellm-response-cost, and the key's
// total spend and budget in x-litellm-key-spend / x-litellm-key-max-budget. The key is
// the lab's, shared with the robot cell: its spend is everyone's.

export type CallKind = 'vision' | 'transcription' | 'voix';

export const costs = {
  session: 0,
  calls: 0,
  byKind: { vision: 0, transcription: 0, voix: 0 } as Record<CallKind, number>,
  last: 0,
  keySpend: NaN,
  keyBudget: NaN,
  /** measured per vision model: calls, total cost, total time (ms) */
  byModel: {} as Record<string, { n: number; total: number; ms: number }>,
};
const SESSION_KEY = 'orangeCost';
try { Object.assign(costs, JSON.parse(sessionStorage.getItem(SESSION_KEY) ?? '{}')); } catch { /* none */ }

function account(res: Response, kind: CallKind, model?: { id: string; ms: number }) {
  const cost = parseFloat(res.headers.get('x-litellm-response-cost') ?? '');
  if (Number.isFinite(cost)) {
    costs.session += cost;
    costs.byKind[kind] += cost;
    costs.last = cost;
  }
  if (model && res.ok) {
    const s = (costs.byModel[model.id] ??= { n: 0, total: 0, ms: 0 });
    s.n++;
    s.total += Number.isFinite(cost) ? cost : 0;
    s.ms += model.ms;
  }
  costs.calls++;
  const spend = parseFloat(res.headers.get('x-litellm-key-spend') ?? '');
  const budget = parseFloat(res.headers.get('x-litellm-key-max-budget') ?? '');
  if (Number.isFinite(spend)) costs.keySpend = spend;
  if (Number.isFinite(budget)) costs.keyBudget = budget;
  try { sessionStorage.setItem(SESSION_KEY, JSON.stringify(costs)); } catch { /* storage blocked */ }
}

async function failure(res: Response): Promise<Error> {
  const text = await res.text().catch(() => '');
  let msg = text.slice(0, 200);
  try { msg = JSON.parse(text).error?.message ?? msg; } catch { /* HTML or plain text */ }
  return new Error(`passerelle Orange ${res.status} : ${msg}`);
}

type Part = { type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } };

/** One chat turn, optionally with an image (data URL). Returns the text answer. */
export async function chat(system: string, prompt: string, image?: string, json = false): Promise<string> {
  const content: Part[] = [{ type: 'text', text: prompt }];
  if (image) content.push({ type: 'image_url', image_url: { url: image } });
  const model = MODELS.vision;
  const send = (temperature?: number) => fetch(`${BASE}/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model,
      messages: [{ role: 'system', content: system }, { role: 'user', content }],
      ...(temperature === undefined ? {} : { temperature }),
      ...(json ? { response_format: { type: 'json_object' } } : {}),
    }),
  });
  const t0 = performance.now();
  let res = await send(0.2);
  // Reasoning models (gpt-5, o-series) only take the default temperature.
  if (res.status === 400) {
    const text = await res.clone().text();
    if (/temperature/i.test(text)) res = await send();
  }
  account(res, 'vision', { id: model, ms: performance.now() - t0 });
  if (!res.ok) throw await failure(res);
  const j = await res.json();
  return String(j.choices?.[0]?.message?.content ?? '').trim();
}

/** A vision chat model of the gateway, prices in USD per million tokens. */
export interface ModelInfo { id: string; inCost: number; outCost: number }

/** The gateway's vision chat models, cheapest first (LiteLLM /model/info). */
export async function visionModels(): Promise<ModelInfo[]> {
  // The gateway's firewall answers GETs without this header with an HTML 403.
  const res = await fetch(`${BASE}/model/info`, { headers: { accept: 'application/json' } });
  if (!res.ok) throw await failure(res);
  const seen = new Map<string, ModelInfo>();
  for (const m of (await res.json()).data ?? []) {
    const i = m.model_info ?? {};
    if (i.mode !== 'chat' || !i.supports_vision || seen.has(m.model_name)) continue;
    seen.set(m.model_name, { id: m.model_name, inCost: (i.input_cost_per_token ?? 0) * 1e6, outCost: (i.output_cost_per_token ?? 0) * 1e6 });
  }
  // A call is mostly input (the image), a little output.
  return [...seen.values()].sort((a, b) => a.inCost + a.outCost / 10 - (b.inCost + b.outCost / 10));
}

/** Speech-to-text (Whisper), French by default. */
export async function transcribe(audio: Blob, language = 'fr'): Promise<string> {
  const form = new FormData();
  form.append('file', audio, `voix.${audio.type.includes('ogg') ? 'ogg' : 'webm'}`);
  form.append('model', MODELS.transcribe);
  form.append('language', language);
  const res = await fetch(`${BASE}/audio/transcriptions`, { method: 'POST', body: form });
  account(res, 'transcription');
  if (!res.ok) throw await failure(res);
  return String((await res.json()).text ?? '').trim();
}

/** Text-to-speech (Gemini TTS): WAV audio. */
export async function synthesize(text: string): Promise<Blob> {
  const res = await fetch(`${BASE}/audio/speech`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: MODELS.speech, input: text, voice: VOICE }),
  });
  account(res, 'voix');
  if (!res.ok) throw await failure(res);
  return res.blob();
}

/** Object found by the vision model; box in 0..1 image coordinates. */
export interface Detection {
  label: string;
  box: { x0: number; y0: number; x1: number; y1: number };
}

const DETECT_SYSTEM = 'Tu détectes les objets dans une image. Réponds uniquement en JSON.';
const DETECT_PROMPT =
  'Détecte au plus 12 objets distincts et utiles (pas les murs, le sol, le plafond ni les mains de l’utilisateur). ' +
  'Réponds avec {"objets": [{"label": "<nom court en français>", "box_2d": [ymin, xmin, ymax, xmax]}]}, ' +
  'coordonnées normalisées de 0 à 1000.';

/** Object detection with Gemini's normalised boxes. */
export async function detect(image: string): Promise<Detection[]> {
  const text = await chat(DETECT_SYSTEM, DETECT_PROMPT, image, true);
  const raw = parseJson(text);
  const list: unknown[] = Array.isArray(raw) ? raw : (raw as { objets?: unknown[] })?.objets ?? [];
  const out: Detection[] = [];
  for (const o of list as { label?: string; box_2d?: number[] }[]) {
    const b = o?.box_2d;
    if (!o?.label || !Array.isArray(b) || b.length !== 4) continue;
    const [y0, x0, y1, x1] = b.map(v => Math.min(1, Math.max(0, v / 1000)));
    if (x1 <= x0 || y1 <= y0) continue;
    out.push({ label: o.label, box: { x0, y0, x1, y1 } });
  }
  return out;
}

function parseJson(text: string): unknown {
  // Models sometimes wrap JSON in a ```json fence.
  const m = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  try { return JSON.parse(m ? m[1] : text); } catch { return null; }
}
