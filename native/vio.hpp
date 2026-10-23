// The VITURE engine (carina_vio.dll, installed with SpaceWalker), in-process: 6DoF pose,
// raw IMU, stereo tracking images, and the glasses' display mode through the engine's
// own USB handle. Same API as bridge/viture_bridge.cpp, without the WebSocket.

#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace vio {

typedef void (*PoseCb)(float*, double);
typedef void (*VsyncCb)(double);
typedef void (*ImuCb)(float*, double);
typedef void (*CamCb)(unsigned char*, unsigned char*, unsigned char*, unsigned char*, double, int, int);
typedef void (*PointsCb)(void*, double);
typedef void (*EventCb)(unsigned char);

struct Api {
    int (*init)(const char*, const char*, int, int, int) = nullptr;
    int (*start)(PoseCb, VsyncCb, ImuCb, CamCb, PointsCb, EventCb) = nullptr;
    int (*resume)() = nullptr;
    int (*stop)() = nullptr;
    int (*release)() = nullptr;
    int (*reset_pose)() = nullptr;
    /** column-major 4x4 head -> world, OpenGL axes (x right, y up, z back), metres; 0 = ok */
    int (*get_gl_pose)(float*, double) = nullptr;
    char* (*get_sn)() = nullptr;
};

inline Api api;

// ------------------------------------------------------------------ display MCU
//
// The engine's switch_display_mode writes to the camera board and returns -1 on the Luma
// Ultra; its display mode lives on the glasses MCU (512-byte frames, endpoint 0x06, see
// src/device/mcu.ts). The engine's USB handle is read from its jcx_stm_switch_disp_mode
// ("cmp qword ptr [rip+disp32], 0" at +0x25); libusb comes from the engine's own copy.

typedef int (*BulkTransfer)(void*, unsigned char, unsigned char*, int, int*, unsigned int);
inline void** usb_handle_slot = nullptr;
inline BulkTransfer usb_bulk = nullptr;

inline void bind_display_mcu(HMODULE m) {
    auto* f = reinterpret_cast<const unsigned char*>(GetProcAddress(m, "jcx_stm_switch_disp_mode"));
    if (HMODULE lu = GetModuleHandleA("libusb-1.0.dll")) usb_bulk = reinterpret_cast<BulkTransfer>(GetProcAddress(lu, "libusb_bulk_transfer"));
    if (!f || !usb_bulk) return;
    const unsigned char* ins = f + 0x25;
    if (ins[0] != 0x48 || ins[1] != 0x83 || ins[2] != 0x3D || ins[7] != 0x00) return;
    int32_t disp;
    memcpy(&disp, ins + 3, 4);
    usb_handle_slot = reinterpret_cast<void**>(const_cast<unsigned char*>(ins + 8 + disp));
}

inline uint16_t crc16_xmodem(const unsigned char* p, int n) {
    uint16_t crc = 0;
    for (int i = 0; i < n; i++) {
        crc ^= p[i] << 8;
        for (int b = 0; b < 8; b++) crc = crc & 0x8000 ? (crc << 1) ^ 0x1021 : crc << 1;
    }
    return crc;
}

// Display modes (the SDK's codes): 3840x1200 side by side at 60 or 90 Hz.
enum DisplayMode : unsigned char { MODE_2D = 0x41, MODE_3D = 0x42, MODE_3D_90 = 0x45 };
enum McuMsg : uint16_t { MSG_BRIGHTNESS = 6, MSG_DISPLAY_MODE = 8, MSG_FILM = 14 };

/** One MCU message with a payload of n bytes; 0 on success, libusb error otherwise. */
inline int mcu_send(uint16_t msg, const unsigned char* payload, int n) {
    void* h = usb_handle_slot ? *usb_handle_slot : nullptr;
    if (!h || !usb_bulk || n > 512 - 18) return -100;
    unsigned char f[512] = {};
    const uint16_t len = (uint16_t)(12 + n);
    f[0] = 0xff; f[1] = 0xfe;
    f[4] = len & 0xff; f[5] = len >> 8;
    f[14] = msg & 0xff; f[15] = msg >> 8;
    memcpy(f + 18, payload, n);
    uint16_t crc = crc16_xmodem(f + 4, len + 2);
    f[2] = crc & 0xff; f[3] = crc >> 8;
    int sent = 0;
    int rc = usb_bulk(h, 0x06, f, sizeof f, &sent, 1000);
    return rc ? rc : (sent == (int)sizeof f ? 0 : -101);
}

inline int set_display_mode(DisplayMode mode) { unsigned char m = mode; return mcu_send(MSG_DISPLAY_MODE, &m, 1); }
/** The electrochromic film: 1 dark, 0 clear (binary on the Luma series). */
inline int set_film(bool dark) { unsigned char v = dark ? 1 : 0; return mcu_send(MSG_FILM, &v, 1); }

// The engine's own exports, undocumented (signatures read from their code), behind SEH.
typedef int (*SetLevelFn)(unsigned char);
typedef int (*VoidFn)();
typedef void (*RenderParaFn)(void*, void*);
inline int seh_level(SetLevelFn f, unsigned char v) { __try { return f(v); } __except (EXCEPTION_EXECUTE_HANDLER) { return -1000; } }
inline int seh_void(VoidFn f) { __try { return f(); } __except (EXCEPTION_EXECUTE_HANDLER) { return -1000; } }
inline int seh_render(RenderParaFn f, void* a, void* b) { __try { f(a, b); return 0; } __except (EXCEPTION_EXECUTE_HANDLER) { return -1000; } }
inline FARPROC engine_fn(const char* name) { HMODULE m = GetModuleHandleA("carina_vio.dll"); return m ? GetProcAddress(m, name) : nullptr; }

/** The screen's brightness level (the SDK's 0..8). */
inline int set_brightness(int level) { auto f = (SetLevelFn)engine_fn("carina_a1088_set_display_level"); return f ? seh_level(f, (unsigned char)level) : -100; }
/** The tracking cameras' exposure: frozen where it is (fixed) or automatic again. */
inline int set_exposure_fixed(bool fixed) { auto f = (VoidFn)engine_fn(fixed ? "jcx_pause_auto_exposure" : "jcx_resume_auto_exposure"); return f ? seh_void(f) : -100; }
/** carina_get_render_para: 48 and 128 bytes of the engine's render parameters. */
inline int get_render_para(unsigned char a[48], unsigned char b[128]) { auto f = (RenderParaFn)engine_fn("carina_get_render_para"); return f ? seh_render(f, a, b) : -100; }

// ------------------------------------------------------------------ start / stop

template <typename F>
inline bool bind(HMODULE m, F& fn, const char* name) {
    fn = reinterpret_cast<F>(GetProcAddress(m, name));
    if (!fn) fprintf(stderr, "missing export %s\n", name);
    return fn != nullptr;
}

inline std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/** Loads and starts the engine; prints why on failure. */
inline bool start(const std::string& sdk, PoseCb pose, VsyncCb vsync, ImuCb imu, CamCb cam, PointsCb points, EventCb event) {
    SetDllDirectoryA(sdk.c_str());
    HMODULE m = LoadLibraryA((sdk + "\\carina_vio.dll").c_str());
    if (!m) {
        fprintf(stderr, "Cannot load carina_vio.dll from %s (error %lu). Is VITURE SpaceWalker installed?\n", sdk.c_str(), GetLastError());
        return false;
    }
    bool ok = bind(m, api.init, "carina_a1088_viture_init") & bind(m, api.start, "carina_a1088_viture_start") &
              bind(m, api.resume, "carina_a1088_viture_resume") & bind(m, api.stop, "carina_a1088_viture_stop") &
              bind(m, api.release, "carina_a1088_viture_release") & bind(m, api.reset_pose, "carina_a1088_viture_reset_pose") &
              bind(m, api.get_gl_pose, "carina_a1088_viture_get_gl_pose") & bind(m, api.get_sn, "carina_a1088_viture_get_sn");
    if (!ok) return false;
    bind_display_mcu(m);

    // The engine resolves relative paths (vocabulary, logs) against the working directory,
    // also after init: it stays there.
    SetCurrentDirectoryA((sdk + "\\Assets\\Configs").c_str());
    std::string cfg = read_file("custom_config.yaml");
    int rc = api.init(cfg.c_str(), (sdk + "\\Assets\\Configs\\database.bin").c_str(), -1, -1, -1);
    if (rc) {
        fprintf(stderr, "VIO init failed (%d): glasses unplugged, or already used by SpaceWalker / the bridge / the browser?\n", rc);
        return false;
    }
    rc = api.start(pose, vsync, imu, cam, points, event);
    if (!rc) rc = api.resume();
    if (rc) {
        fprintf(stderr, "VIO start failed (%d)\n", rc);
        api.release();
        return false;
    }
    return true;
}

/** Stops the engine; its release can hang, so after 3 s the process just ends (USB freed). */
inline void stop() {
    HANDLE t = CreateThread(nullptr, 0, [](LPVOID) -> DWORD {
        if (api.stop) api.stop();
        if (api.release) api.release();
        return 0;
    }, nullptr, 0, nullptr);
    if (t && WaitForSingleObject(t, 3000) == WAIT_TIMEOUT) {
        fflush(stdout);
        TerminateProcess(GetCurrentProcess(), 0);
    }
}

}  // namespace vio
