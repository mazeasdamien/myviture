// The glasses' RGB camera (UVC "USB Camera", 0c45:636b) through Media Foundation, on its
// own thread: frames converted to BGRA by the source reader, newest kept for the dashboard.

#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "ole32.lib")

namespace rgbcam {

struct Frame {
    std::mutex mu;
    std::vector<uint32_t> bgra;
    int w = 0, h = 0;
    uint32_t seq = 0;
    long count = 0;
    std::string name, status = "recherche de la caméra…";
};
inline Frame g_frame;
inline std::atomic<bool> g_stop{false};
inline std::thread g_thread;

inline void set_status(const std::string& s) { std::lock_guard<std::mutex> l(g_frame.mu); g_frame.status = s; }

/** The glasses' camera: its symbolic link names vid_0c45&pid_636b; else a name with "USB Camera". */
inline IMFMediaSource* open_source(std::string& name) {
    IMFAttributes* attr = nullptr;
    MFCreateAttributes(&attr, 1);
    attr->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    IMFActivate** devs = nullptr;
    UINT32 n = 0;
    MFEnumDeviceSources(attr, &devs, &n);
    attr->Release();
    int pick = -1;
    for (UINT32 i = 0; i < n && pick < 0; i++) {
        WCHAR* link = nullptr; UINT32 len;
        if (SUCCEEDED(devs[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &link, &len))) {
            std::wstring l = link;
            for (auto& c : l) c = (wchar_t)towlower(c);
            if (l.find(L"vid_0c45&pid_636b") != std::wstring::npos) pick = (int)i;
            CoTaskMemFree(link);
        }
    }
    for (UINT32 i = 0; i < n && pick < 0; i++) {
        WCHAR* fn = nullptr; UINT32 len;
        if (SUCCEEDED(devs[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &fn, &len))) {
            if (std::wstring(fn).find(L"USB Camera") != std::wstring::npos) pick = (int)i;
            CoTaskMemFree(fn);
        }
    }
    IMFMediaSource* src = nullptr;
    if (pick >= 0) {
        WCHAR* fn = nullptr; UINT32 len;
        if (SUCCEEDED(devs[pick]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &fn, &len))) {
            char buf[256]; WideCharToMultiByte(CP_UTF8, 0, fn, -1, buf, sizeof buf, nullptr, nullptr); name = buf;
            CoTaskMemFree(fn);
        }
        devs[pick]->ActivateObject(IID_PPV_ARGS(&src));
    }
    for (UINT32 i = 0; i < n; i++) devs[i]->Release();
    CoTaskMemFree(devs);
    return src;
}

inline void run() {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    MFStartup(MF_VERSION);
    while (!g_stop) {
        std::string name;
        IMFMediaSource* src = open_source(name);
        if (!src) { set_status("caméra RGB des lunettes introuvable"); for (int i = 0; i < 30 && !g_stop; i++) Sleep(100); continue; }
        IMFAttributes* ra = nullptr;
        MFCreateAttributes(&ra, 2);
        // The reader converts the camera's MJPG / NV12 to RGB32 itself.
        ra->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
        IMFSourceReader* reader = nullptr;
        HRESULT hr = MFCreateSourceReaderFromMediaSource(src, ra, &reader);
        ra->Release();
        IMFMediaType* mt = nullptr;
        if (SUCCEEDED(hr)) {
            MFCreateMediaType(&mt);
            mt->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
            mt->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
            MFSetAttributeSize(mt, MF_MT_FRAME_SIZE, 1280, 720);
            hr = reader->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, mt);
            mt->Release();
        }
        UINT32 w = 0, h = 0;
        if (SUCCEEDED(hr)) {
            IMFMediaType* cur = nullptr;
            reader->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, &cur);
            MFGetAttributeSize(cur, MF_MT_FRAME_SIZE, &w, &h);
            cur->Release();
            { std::lock_guard<std::mutex> l(g_frame.mu); g_frame.name = name; g_frame.status = ""; }
        } else {
            char msg[96]; snprintf(msg, sizeof msg, "caméra RGB : format refusé (0x%08lx)", (unsigned long)hr);
            set_status(msg);
        }
        while (SUCCEEDED(hr) && !g_stop) {
            DWORD flags = 0; LONGLONG ts; IMFSample* sample = nullptr;
            hr = reader->ReadSample((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, nullptr, &flags, &ts, &sample);
            if (FAILED(hr) || (flags & MF_SOURCE_READERF_ERROR)) { set_status("caméra RGB perdue, nouvelle tentative…"); break; }
            if (!sample) continue;
            IMFMediaBuffer* buf = nullptr;
            if (SUCCEEDED(sample->ConvertToContiguousBuffer(&buf))) {
                BYTE* p; DWORD len;
                if (SUCCEEDED(buf->Lock(&p, nullptr, &len)) && len >= w * h * 4) {
                    std::lock_guard<std::mutex> l(g_frame.mu);
                    g_frame.bgra.resize((size_t)w * h);
                    // RGB32 from Media Foundation is bottom-up only for some sources; this
                    // one delivers top-down rows with a positive stride.
                    memcpy(g_frame.bgra.data(), p, (size_t)w * h * 4);
                    g_frame.w = (int)w; g_frame.h = (int)h;
                    g_frame.seq++; g_frame.count++;
                    buf->Unlock();
                }
                buf->Release();
            }
            sample->Release();
        }
        if (reader) reader->Release();
        src->Shutdown();
        src->Release();
        for (int i = 0; i < 20 && !g_stop; i++) Sleep(100);
    }
    MFShutdown();
    CoUninitialize();
}

inline void start() { g_thread = std::thread(run); }
inline void stop() { g_stop = true; if (g_thread.joinable()) g_thread.join(); }

}  // namespace rgbcam
