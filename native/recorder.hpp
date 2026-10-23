// Recording of the RGB view with the virtual scene over it: MP4 (H.264 video, AAC sound from
// the default microphone) through Media Foundation's sink writer. Video frames come from the
// render thread (BGRA, top-down) and are encoded on the recorder's own thread; the sound is
// captured with WASAPI on another. Both are stamped on the QPC clock, so they stay in sync.
// Also: a PNG of one frame (WIC).

#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <audioclient.h>
#include <mmreg.h>
#include <ksmedia.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mmdeviceapi.h>
#include <functiondiscoverykeys_devpkey.h>
#include <shlobj.h>
#include <wincodec.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "shell32.lib")

namespace rec {

struct Status {
    bool on = false;
    std::string path, error, mic;
    long frames = 0, dropped = 0;
    double seconds = 0, audio_seconds = 0;
    bool audio = false;
};

inline std::mutex g_mu;  // guards g_status and the writer
inline Status g_status;

namespace detail {
inline IMFSinkWriter* writer = nullptr;
inline DWORD video_stream = 0, audio_stream = 0;
inline int W = 0, H = 0;
inline int64_t t0_qpc = 0;
inline double qpc_freq = 0;
inline std::atomic<bool> stop_audio{false}, stop_video{false};
inline std::thread audio_thread, video_thread;
struct Frame { std::vector<uint8_t> px; int64_t t100; };
inline std::deque<Frame> queue;
inline std::mutex qmu;
inline std::condition_variable qcv;
inline int64_t last_video_t = -1;
inline std::vector<std::vector<uint8_t>> pool;  // reused frame buffers

inline int64_t qpc_to_100ns(int64_t qpc) { return (int64_t)((qpc - t0_qpc) * (10000000.0 / qpc_freq)); }

inline void fail(const char* what, HRESULT hr) {
    char b[160]; snprintf(b, sizeof b, "%s (0x%08lx)", what, (unsigned long)hr);
    fprintf(stderr, "rec: %s\n", b);
    std::lock_guard<std::mutex> l(g_mu); g_status.error = b;
}

inline void video_loop() {
    while (true) {
        Frame f;
        {
            std::unique_lock<std::mutex> l(qmu);
            qcv.wait(l, [] { return !queue.empty() || stop_video; });
            if (queue.empty()) return;
            f = std::move(queue.front());
            queue.pop_front();
        }
        IMFMediaBuffer* buf = nullptr;
        IMFSample* s = nullptr;
        DWORD len = (DWORD)(W * H * 4);
        if (SUCCEEDED(MFCreateMemoryBuffer(len, &buf))) {
            BYTE* p; buf->Lock(&p, nullptr, nullptr);
            memcpy(p, f.px.data(), len);
            buf->Unlock();
            buf->SetCurrentLength(len);
            MFCreateSample(&s);
            s->AddBuffer(buf);
            s->SetSampleTime(f.t100);
            s->SetSampleDuration(333333);
            std::lock_guard<std::mutex> l(g_mu);
            HRESULT hr = writer ? writer->WriteSample(video_stream, s) : E_FAIL;
            if (SUCCEEDED(hr)) { g_status.frames++; g_status.seconds = f.t100 / 1e7; }
            else if (g_status.error.empty()) { char b[96]; snprintf(b, sizeof b, "écriture vidéo (0x%08lx)", (unsigned long)hr); g_status.error = b; }
        }
        if (s) s->Release();
        if (buf) buf->Release();
        std::lock_guard<std::mutex> l(qmu);
        pool.push_back(std::move(f.px));
    }
}

/**
 * The default microphone, shared mode, as 16-bit PCM at `out_rate` for the AAC encoder
 * (linear resampling when the microphone runs at another rate, e.g. 16 kHz headsets).
 * Times: the first packet's QPC time, then the count of samples written.
 */
inline void audio_loop(int src_rate, int out_rate, int channels, IAudioClient* client, IAudioCaptureClient* cap, bool is_float, int src_bits, int src_channels) {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    client->Start();
    std::vector<float> in;        // source samples not yet used, interleaved
    std::vector<int16_t> pcm;
    double pos = 0;               // next output sample's position in `in` (source samples)
    const double step = (double)src_rate / out_rate;
    int64_t t_first = -1, written = 0;
    while (!stop_audio) {
        Sleep(10);
        UINT32 n = 0;
        while (SUCCEEDED(cap->GetNextPacketSize(&n)) && n) {
            BYTE* data; UINT32 frames; DWORD flags; UINT64 dev_pos, qpc_pos;
            if (FAILED(cap->GetBuffer(&data, &frames, &flags, &dev_pos, &qpc_pos))) break;
            for (UINT32 i = 0; i < frames; i++) {
                for (int c = 0; c < channels; c++) {
                    int sc = std::min(c, src_channels - 1);
                    float v = 0;
                    if (flags & AUDCLNT_BUFFERFLAGS_SILENT) v = 0;
                    else if (is_float) v = ((const float*)data)[(size_t)i * src_channels + sc];
                    else if (src_bits == 16) v = ((const int16_t*)data)[(size_t)i * src_channels + sc] / 32768.0f;
                    else if (src_bits == 32) v = ((const int32_t*)data)[(size_t)i * src_channels + sc] / 2147483648.0f;
                    in.push_back(v);
                }
            }
            // qpc_pos is in 100 ns units of the QPC clock.
            int64_t t100 = (int64_t)qpc_pos - (int64_t)(t0_qpc * (10000000.0 / qpc_freq));
            cap->ReleaseBuffer(frames);
            if (t100 < 0) { in.clear(); pos = 0; continue; }
            if (t_first < 0) t_first = t100;
            // Resample what is there.
            pcm.clear();
            size_t avail = in.size() / channels;
            while (pos + 1 < avail) {
                size_t i0 = (size_t)pos; double fr = pos - i0;
                for (int c = 0; c < channels; c++) {
                    float v = (float)(in[i0 * channels + c] * (1 - fr) + in[(i0 + 1) * channels + c] * fr);
                    v = v > 1 ? 1 : v < -1 ? -1 : v;
                    pcm.push_back((int16_t)(v * 32767));
                }
                pos += step;
            }
            size_t used = (size_t)pos;
            in.erase(in.begin(), in.begin() + std::min(in.size(), used * channels));
            pos -= used;
            if (pcm.empty()) continue;
            int64_t out_frames = (int64_t)(pcm.size() / channels);
            int64_t ts = t_first + written * 10000000 / out_rate;
            written += out_frames;
            IMFMediaBuffer* buf = nullptr; IMFSample* s = nullptr;
            DWORD len = (DWORD)(pcm.size() * 2);
            if (SUCCEEDED(MFCreateMemoryBuffer(len, &buf))) {
                BYTE* p; buf->Lock(&p, nullptr, nullptr); memcpy(p, pcm.data(), len); buf->Unlock(); buf->SetCurrentLength(len);
                MFCreateSample(&s); s->AddBuffer(buf);
                s->SetSampleTime(ts);
                s->SetSampleDuration(out_frames * 10000000 / out_rate);
                std::lock_guard<std::mutex> l(g_mu);
                if (writer && SUCCEEDED(writer->WriteSample(audio_stream, s))) g_status.audio_seconds = ts / 1e7;
            }
            if (s) s->Release();
            if (buf) buf->Release();
        }
    }
    client->Stop();
    cap->Release();
    client->Release();
    CoUninitialize();
}
}  // namespace detail

/** The glasses' own microphone ("VITURE" in its name), else the default one. */
inline IMMDevice* glasses_mic(IMMDeviceEnumerator* en, std::string& name) {
    IMMDevice* pick = nullptr;
    IMMDeviceCollection* all = nullptr;
    if (SUCCEEDED(en->EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE, &all))) {
        UINT n = 0; all->GetCount(&n);
        for (UINT i = 0; i < n && !pick; i++) {
            IMMDevice* d = nullptr; IPropertyStore* ps = nullptr;
            if (FAILED(all->Item(i, &d))) continue;
            PROPVARIANT v; PropVariantInit(&v);
            if (SUCCEEDED(d->OpenPropertyStore(STGM_READ, &ps)) && SUCCEEDED(ps->GetValue(PKEY_Device_FriendlyName, &v)) && v.vt == VT_LPWSTR) {
                std::wstring w = v.pwszVal;
                if (w.find(L"VITURE") != std::wstring::npos) {
                    pick = d; d = nullptr;
                    char b[256]; WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, b, sizeof b, nullptr, nullptr); name = b;
                }
            }
            PropVariantClear(&v);
            if (ps) ps->Release();
            if (d) d->Release();
        }
        all->Release();
    }
    if (!pick && SUCCEEDED(en->GetDefaultAudioEndpoint(eCapture, eConsole, &pick))) name = "micro par défaut de Windows";
    return pick;
}

/** Where recordings go: the user's Videos folder, myViture sub-folder. */
inline std::wstring folder() {
    PWSTR p = nullptr;
    std::wstring dir = L".";
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Videos, 0, nullptr, &p))) { dir = p; CoTaskMemFree(p); }
    dir += L"\\myViture";
    CreateDirectoryW(dir.c_str(), nullptr);
    return dir;
}

inline std::wstring stamped(const wchar_t* ext) {
    SYSTEMTIME st; GetLocalTime(&st);
    wchar_t name[96];
    swprintf(name, 96, L"\\myViture_%04d-%02d-%02d_%02d-%02d-%02d%s", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, ext);
    return folder() + name;
}

inline std::string utf8(const std::wstring& w) {
    char b[MAX_PATH * 3]; WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, b, sizeof b, nullptr, nullptr); return b;
}

/** Starts an MP4 of w x h BGRA frames; `audio`: also the default microphone. */
inline bool start(int w, int h, int mbps, bool audio) {
    using namespace detail;
    if (writer) return true;
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    MFStartup(MF_VERSION);
    LARGE_INTEGER f, q; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&q);
    qpc_freq = (double)f.QuadPart; t0_qpc = q.QuadPart;
    W = w; H = h; last_video_t = -1;
    std::wstring path = stamped(L".mp4");
    { std::lock_guard<std::mutex> l(g_mu); g_status = Status{}; g_status.path = utf8(path); }

    IMFAttributes* attr = nullptr;
    MFCreateAttributes(&attr, 2);
    attr->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
    attr->SetUINT32(MF_SINK_WRITER_DISABLE_THROTTLING, TRUE);
    HRESULT hr = MFCreateSinkWriterFromURL(path.c_str(), nullptr, attr, &writer);
    attr->Release();
    if (FAILED(hr)) { fail("création du MP4", hr); writer = nullptr; return false; }

    // Video: H.264 out, BGRA (RGB32, top-down) in; the writer adds the colour converter.
    IMFMediaType* out = nullptr; MFCreateMediaType(&out);
    out->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    out->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264);
    out->SetUINT32(MF_MT_AVG_BITRATE, (UINT32)mbps * 1000000);
    out->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
    MFSetAttributeSize(out, MF_MT_FRAME_SIZE, w, h);
    MFSetAttributeRatio(out, MF_MT_FRAME_RATE, 30, 1);
    MFSetAttributeRatio(out, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
    hr = writer->AddStream(out, &video_stream);
    out->Release();
    IMFMediaType* in = nullptr; MFCreateMediaType(&in);
    in->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    in->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
    in->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
    in->SetUINT32(MF_MT_DEFAULT_STRIDE, (UINT32)(w * 4));
    MFSetAttributeSize(in, MF_MT_FRAME_SIZE, w, h);
    MFSetAttributeRatio(in, MF_MT_FRAME_RATE, 30, 1);
    MFSetAttributeRatio(in, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
    if (SUCCEEDED(hr)) hr = writer->SetInputMediaType(video_stream, in, nullptr);
    in->Release();
    if (FAILED(hr)) { fail("format vidéo refusé", hr); writer->Release(); writer = nullptr; return false; }

    // Sound: the microphone's own rate and up to 2 channels, as 16-bit PCM into AAC.
    IAudioClient* client = nullptr; IAudioCaptureClient* cap = nullptr;
    int rate = 48000, src_rate = 48000, channels = 2, src_bits = 32, src_channels = 2; bool is_float = true;
    bool have_audio = false;
    if (audio) {
        IMMDeviceEnumerator* en = nullptr; IMMDevice* devi = nullptr; WAVEFORMATEX* mix = nullptr;
        std::string mic_name;
        if (SUCCEEDED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&en))) &&
            (devi = glasses_mic(en, mic_name)) != nullptr &&
            SUCCEEDED(devi->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&client)) &&
            SUCCEEDED(client->GetMixFormat(&mix)) &&
            SUCCEEDED(client->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 2000000, 0, mix, nullptr)) &&
            SUCCEEDED(client->GetService(IID_PPV_ARGS(&cap)))) {
            rate = (int)mix->nSamplesPerSec; src_channels = mix->nChannels; src_bits = mix->wBitsPerSample;
            is_float = mix->wFormatTag == WAVE_FORMAT_IEEE_FLOAT ||
                       (mix->wFormatTag == WAVE_FORMAT_EXTENSIBLE && ((WAVEFORMATEXTENSIBLE*)mix)->SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT);
            channels = src_channels >= 2 ? 2 : 1;
            have_audio = true;
        }
        // The AAC encoder takes 44.1 or 48 kHz: other rates are resampled to 48 kHz.
        src_rate = rate;
        if (rate != 44100 && rate != 48000) rate = 48000;
        if (mix) printf("rec: microphone '%s' %lu Hz, %u channels, %u bits\n", mic_name.c_str(), mix->nSamplesPerSec, mix->nChannels, mix->wBitsPerSample);
        else printf("rec: no microphone\n");
        { std::lock_guard<std::mutex> l(g_mu); g_status.mic = mic_name; }
        if (mix) CoTaskMemFree(mix);
        if (devi) devi->Release();
        if (en) en->Release();
        if (have_audio) {
            IMFMediaType* ao = nullptr; MFCreateMediaType(&ao);
            ao->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
            ao->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_AAC);
            ao->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16);
            ao->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, rate);
            ao->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, channels);
            ao->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, 16000);  // 128 kbit/s
            hr = writer->AddStream(ao, &audio_stream);
            ao->Release();
            IMFMediaType* ai = nullptr; MFCreateMediaType(&ai);
            ai->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
            ai->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_PCM);
            ai->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16);
            ai->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, rate);
            ai->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, channels);
            ai->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, channels * 2);
            ai->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, rate * channels * 2);
            if (SUCCEEDED(hr)) hr = writer->SetInputMediaType(audio_stream, ai, nullptr);
            ai->Release();
            if (FAILED(hr)) { have_audio = false; fail("son refusé par l'encodeur, vidéo sans son", hr); }
        }
        if (!have_audio) { if (cap) cap->Release(); if (client) client->Release(); cap = nullptr; client = nullptr; }
    }
    hr = writer->BeginWriting();
    if (FAILED(hr)) { fail("démarrage de l'écriture", hr); writer->Release(); writer = nullptr; return false; }
    stop_audio = false; stop_video = false;
    video_thread = std::thread(video_loop);
    if (have_audio) audio_thread = std::thread(audio_loop, src_rate, rate, channels, client, cap, is_float, src_bits, src_channels);
    std::lock_guard<std::mutex> l(g_mu);
    g_status.on = true;
    g_status.audio = have_audio;
    if (audio && !have_audio && g_status.error.empty()) g_status.error = "micro introuvable : vidéo sans son";
    return true;
}

/** One frame (BGRA, `pitch` bytes per row) captured at `t_qpc`. Dropped if the encoder lags. */
inline void push(const uint8_t* bgra, int pitch, int64_t t_qpc) {
    using namespace detail;
    if (!writer) return;
    int64_t t100 = qpc_to_100ns(t_qpc);
    if (t100 < 0 || t100 <= last_video_t) return;
    last_video_t = t100;
    std::vector<uint8_t> px;
    {
        std::lock_guard<std::mutex> l(qmu);
        if (queue.size() >= 8) { std::lock_guard<std::mutex> l2(g_mu); g_status.dropped++; return; }
        if (!pool.empty()) { px = std::move(pool.back()); pool.pop_back(); }
    }
    px.resize((size_t)W * H * 4);
    for (int y = 0; y < H; y++) memcpy(px.data() + (size_t)y * W * 4, bgra + (size_t)y * pitch, (size_t)W * 4);
    {
        std::lock_guard<std::mutex> l(qmu);
        queue.push_back({std::move(px), t100});
    }
    qcv.notify_one();
}

inline void stop() {
    using namespace detail;
    if (!writer) return;
    stop_audio = true;
    if (audio_thread.joinable()) audio_thread.join();
    stop_video = true;
    qcv.notify_all();
    if (video_thread.joinable()) video_thread.join();
    std::lock_guard<std::mutex> l(g_mu);
    writer->Finalize();
    writer->Release();
    writer = nullptr;
    queue.clear();
    g_status.on = false;
}

/** A PNG of one BGRA frame (alpha ignored). */
inline bool save_png(const std::wstring& path, const uint8_t* bgra, int w, int h, int pitch) {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    IWICImagingFactory* wic = nullptr; IWICStream* stream = nullptr; IWICBitmapEncoder* enc = nullptr; IWICBitmapFrameEncode* fr = nullptr;
    bool ok = SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic)));
    ok = ok && SUCCEEDED(wic->CreateStream(&stream)) && SUCCEEDED(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE));
    ok = ok && SUCCEEDED(wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc)) && SUCCEEDED(enc->Initialize(stream, WICBitmapEncoderNoCache));
    ok = ok && SUCCEEDED(enc->CreateNewFrame(&fr, nullptr)) && SUCCEEDED(fr->Initialize(nullptr)) && SUCCEEDED(fr->SetSize(w, h));
    // BGRA: PNG takes it as is (32bppBGR is turned into 24 bits, which shreds the rows).
    WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
    ok = ok && SUCCEEDED(fr->SetPixelFormat(&fmt)) && SUCCEEDED(fr->WritePixels(h, pitch, pitch * h, (BYTE*)bgra));
    ok = ok && SUCCEEDED(fr->Commit()) && SUCCEEDED(enc->Commit());
    if (fr) fr->Release(); if (enc) enc->Release(); if (stream) stream->Release(); if (wic) wic->Release();
    return ok;
}

}  // namespace rec
