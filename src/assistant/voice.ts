// Microphone and speaker for the assistant.
//
// Input: push-to-talk, or hands-free listening where a simple voice-activity detector
// (level above the noise floor) cuts utterances. The glasses' own microphone and
// speakers are preferred when the browser lists them.
// Output: Gemini TTS through the gateway, sentence by sentence (synthesised in
// parallel, played in order), with the browser's own voice as a fallback.

import { synthesize } from './orange';

const GLASSES = /viture|luma/i;

export class VoiceInput {
  /** Called with each finished utterance; `pushed`: push-to-talk rather than detected speech. */
  onUtterance: (audio: Blob, pushed: boolean) => void = () => {};
  /** Hands-free detection paused (while the assistant speaks, so it does not hear itself). */
  muted = false;
  /** Input level 0..1, for a meter. */
  level = 0;
  device = '';
  private stream: MediaStream | null = null;
  private ctx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private rec: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private started = 0;
  private listening = false;
  private ptt = false;
  private pushed = false;
  private noise = 0.01;
  private loud = 0;
  private lastVoice = 0;
  private timer = 0;

  async open() {
    if (this.stream) return;
    const devices = await navigator.mediaDevices.enumerateDevices();
    const glasses = devices.find(d => d.kind === 'audioinput' && GLASSES.test(d.label));
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { deviceId: glasses ? { exact: glasses.deviceId } : undefined, echoCancellation: true, noiseSuppression: true },
    });
    this.device = this.stream.getAudioTracks()[0]?.label ?? '';
    this.ctx = new AudioContext();
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.ctx.createMediaStreamSource(this.stream).connect(this.analyser);
    const buf = new Float32Array(this.analyser.fftSize);
    this.timer = window.setInterval(() => this.tick(buf), 30);
  }

  close() {
    clearInterval(this.timer);
    this.stopRecording(false);
    this.stream?.getTracks().forEach(t => t.stop());
    this.ctx?.close();
    this.stream = null;
    this.ctx = null;
  }

  async setListening(on: boolean) {
    if (on) await this.open();
    this.listening = on;
    if (!on && !this.ptt) this.stopRecording(false);
  }

  async pushStart() {
    await this.open();
    this.ptt = true;
    this.startRecording();
  }

  pushEnd() {
    if (!this.ptt) return;
    this.ptt = false;
    this.stopRecording(true);
  }

  get recording() {
    return this.rec?.state === 'recording';
  }

  private tick(buf: Float32Array) {
    if (!this.analyser) return;
    this.analyser.getFloatTimeDomainData(buf as Float32Array<ArrayBuffer>);
    let s = 0;
    for (const v of buf) s += v * v;
    const rms = Math.sqrt(s / buf.length);
    this.level = Math.min(1, rms * 8);
    if (this.ptt || !this.listening || this.muted) return;
    const now = performance.now();
    const speaking = rms > Math.max(0.02, this.noise * 3);
    // The noise floor follows the quiet parts only.
    if (!speaking) this.noise += (rms - this.noise) * 0.02;
    if (speaking) { this.loud++; this.lastVoice = now; } else this.loud = 0;
    if (!this.recording && this.loud >= 3) this.startRecording();
    if (this.recording && now - this.lastVoice > 900) this.stopRecording(now - this.started > 700);
    if (this.recording && now - this.started > 15000) this.stopRecording(true);
  }

  private startRecording() {
    if (!this.stream || this.recording) return;
    this.chunks = [];
    this.rec = new MediaRecorder(this.stream, { mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : '' });
    this.rec.ondataavailable = e => { if (e.data.size) this.chunks.push(e.data); };
    this.rec.start();
    this.started = performance.now();
    this.pushed = this.ptt;
  }

  private stopRecording(keep: boolean) {
    const rec = this.rec;
    if (!rec || rec.state !== 'recording') return;
    const pushed = this.pushed;
    rec.onstop = () => {
      if (keep && this.chunks.length) this.onUtterance(new Blob(this.chunks, { type: rec.mimeType || 'audio/webm' }), pushed);
    };
    rec.stop();
  }
}

export type VoiceMode = 'orange' | 'browser' | 'off';

export class Speaker {
  mode: VoiceMode = 'orange';
  /** Called when speech starts and ends (to mute hands-free listening meanwhile). */
  onSpeaking: (on: boolean) => void = () => {};
  private audio = new Audio();
  private queue: Promise<void> = Promise.resolve();
  private generation = 0;

  constructor() {
    this.useGlassesOutput();
  }

  /** Plays on the glasses' speakers when the browser lists them. */
  private async useGlassesOutput() {
    try {
      const out = (await navigator.mediaDevices.enumerateDevices()).find(d => d.kind === 'audiooutput' && GLASSES.test(d.label));
      const a = this.audio as HTMLAudioElement & { setSinkId?: (id: string) => Promise<void> };
      if (out && a.setSinkId) await a.setSinkId(out.deviceId);
    } catch { /* default output */ }
  }

  stop() {
    this.generation++;
    this.audio.pause();
    speechSynthesis.cancel();
    this.onSpeaking(false);
  }

  say(text: string) {
    this.stop();
    if (this.mode === 'off' || !text) return;
    const gen = this.generation;
    const sentences = text.match(/[^.!?…]+[.!?…]*/g)?.map(s => s.trim()).filter(Boolean) ?? [text];
    if (this.mode === 'browser') return this.browserSay(text, gen);
    // All sentences are synthesised at once; each plays as soon as it and the previous are ready.
    const clips = sentences.map(s => synthesize(s).catch(() => null));
    this.queue = (async () => {
      this.onSpeaking(true);
      for (let i = 0; i < clips.length; i++) {
        const clip = await clips[i];
        if (gen !== this.generation) return;
        if (!clip) { await this.browserSay(sentences.slice(i).join(' '), gen); return; }
        await this.play(clip);
      }
      if (gen === this.generation) this.onSpeaking(false);
    })();
  }

  private play(clip: Blob): Promise<void> {
    return new Promise(resolve => {
      const url = URL.createObjectURL(clip);
      this.audio.src = url;
      const done = () => { URL.revokeObjectURL(url); resolve(); };
      this.audio.onended = done;
      this.audio.onerror = done;
      this.audio.onpause = done;
      this.audio.play().catch(done);
    });
  }

  private browserSay(text: string, gen: number): Promise<void> {
    return new Promise(resolve => {
      if (gen !== this.generation) return resolve();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = 'fr-FR';
      u.onend = u.onerror = () => { if (gen === this.generation) this.onSpeaking(false); resolve(); };
      this.onSpeaking(true);
      speechSynthesis.speak(u);
    });
  }
}
