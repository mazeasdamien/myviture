// Front RGB camera of the Luma Ultra: a standard UVC webcam (USB 0c45:636b, "USB Camera").

const LABEL_HINTS = [/0c45:636b/i, /usb camera/i, /viture/i];

export async function listCameras(): Promise<MediaDeviceInfo[]> {
  return (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
}

/** Picks the glasses camera among the video inputs, if labels are available. */
export function findGlassesCamera(cams: MediaDeviceInfo[]): MediaDeviceInfo | undefined {
  for (const re of LABEL_HINTS) {
    const hit = cams.find(c => re.test(c.label));
    if (hit) return hit;
  }
  return undefined;
}

export async function openCamera(deviceId?: string): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      frameRate: { ideal: 30 },
    },
  });
}

export function snapshot(video: HTMLVideoElement): Promise<Blob | null> {
  const c = document.createElement('canvas');
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext('2d')!.drawImage(video, 0, 0);
  return new Promise(r => c.toBlob(r, 'image/png'));
}

export class Recorder {
  private rec: MediaRecorder | null = null;
  private chunks: Blob[] = [];

  get recording() {
    return this.rec?.state === 'recording';
  }

  start(stream: MediaStream) {
    const type = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find(t => MediaRecorder.isTypeSupported(t));
    this.chunks = [];
    this.rec = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 12_000_000 });
    this.rec.ondataavailable = e => e.data.size && this.chunks.push(e.data);
    this.rec.start(1000);
  }

  stop(): Promise<Blob> {
    return new Promise(resolve => {
      const rec = this.rec!;
      rec.onstop = () => resolve(new Blob(this.chunks, { type: rec.mimeType }));
      rec.stop();
    });
  }
}

export function download(blob: Blob, name: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
