// VITURE's own hand tracking (gesture_recognition.dll, installed with SpaceWalker), in
// process: fed with the engine's stereo images and pose, it answers with both hands'
// joints. Same calls as SpaceWalker's GestureManager (VitureCommonLibrary.GestureSdk):
//   gesture_init(model dir) -> update_mode(1, cam0/cam1 calibration) -> register -> start,
//   then OnCameraData(left, right, w*h*2, t) per stereo frame and OnSlamPose(pose[16], t).
// The answer: 374 floats, 187 per hand (left then right): [0] valid, then 7 floats per
// joint (x, y, z, and 4 more), 21 joints used. SpaceWalker shows joint k at (-x, z, -y).

#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>

namespace vhands {

constexpr int N = 374, PER_HAND = 187, JOINTS = 21, STRIDE = 7;

typedef void (*SwrCb)(float*);
struct Api {
    int (*init)(const char*) = nullptr;
    int (*uninit)() = nullptr;
    int (*start)() = nullptr;
    int (*stop)() = nullptr;
    int (*slam_pose)(float*, double) = nullptr;
    int (*camera)(void*, void*, unsigned, double) = nullptr;
    int (*mode)(int, float*, float*, float*, float*, float*, float*, float*) = nullptr;
    int (*reg)(void*, SwrCb, void*, void*) = nullptr;
};
inline Api api;
inline std::atomic<bool> g_on{false};

struct Out {
    float data[N] = {};
    double t = 0;        // QPC seconds when it arrived
    double img_t = 0;    // the engine time of the last image fed before it
    uint32_t seq = 0;
    long count = 0;
    std::string status = "arrêté";
};
inline std::mutex g_mu;
inline Out g_out;
inline std::atomic<double> g_last_img_t{0};
inline FILE* g_raw = nullptr;  // first answers, in full, for the layout

/**
 * VITURE's joint order (SpaceWalker's HandSkeletonDriver bindings): 0-4 tips (thumb..little),
 * 5 wrist, 6-7 thumb proximal / distal, 8-10 index, 11-13 middle, 14-16 ring, 17-19 little
 * (proximal, intermediate, distal), 20 palm. MediaPipe joint k <- VITURE joint FROM[k];
 * -1: the thumb's base, halfway from the wrist to its proximal joint.
 */
inline const int FROM[21] = {5, -1, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4};

/**
 * One hand's joints in the engine's world frame (OpenGL axes, metres), MediaPipe order.
 * VITURE's axes against the engine's world (fitted on our stereo hands, 15 mm median):
 * x = -x_gl, y = z_gl, z = y_gl, same origin.
 */
inline bool hand_world(const float* data, int side, float out[21][3]) {
    const float* h = data + side * PER_HAND;
    if (h[0] == 0) return false;
    auto at = [&](int j, float* p) { const float* v = h + 1 + j * STRIDE; p[0] = -v[0]; p[1] = v[2]; p[2] = v[1]; };
    for (int k = 0; k < 21; k++) {
        if (FROM[k] >= 0) { at(FROM[k], out[k]); continue; }
        float a[3], b[3];
        at(5, a); at(6, b);
        for (int i = 0; i < 3; i++) out[k][i] = 0.5f * (a[i] + b[i]);
    }
    return true;
}

/** One Euro filter, per coordinate: smooth at rest, little lag when moving. */
struct OneEuro {
    double min_cutoff = 1.2, beta = 6, d_cutoff = 1;
    double x = 0, dx = 0, t = -1;
    static double alpha(double cutoff, double dt) { double tau = 1 / (2 * 3.14159265 * cutoff); return 1 / (1 + tau / dt); }
    double apply(double v, double now) {
        if (t < 0 || now - t > 0.3) { x = v; dx = 0; t = now; return v; }
        double dt = std::max(1e-4, now - t);
        double d = (v - x) / dt;
        dx += alpha(d_cutoff, dt) * (d - dx);
        x += alpha(min_cutoff + beta * std::fabs(dx), dt) * (v - x);
        t = now;
        return x;
    }
};

inline double qpc_s() { LARGE_INTEGER f, q; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&q); return (double)q.QuadPart / f.QuadPart; }

inline void on_hands(float* d) {
    if (!d) return;
    std::lock_guard<std::mutex> l(g_mu);
    memcpy(g_out.data, d, sizeof g_out.data);
    g_out.t = qpc_s();
    g_out.img_t = g_last_img_t;
    g_out.seq++;
    g_out.count++;
    if (g_raw && (d[0] != 0 || d[PER_HAND] != 0)) {
        static int n = 0;
        if (n++ < 40) {
            fprintf(g_raw, "%.4f", g_out.t);
            for (int i = 0; i < N; i++) fprintf(g_raw, ",%g", d[i]);
            fprintf(g_raw, "\n");
            fflush(g_raw);
        }
    }
}

/**
 * `n` numbers of `key:` inside the `block:` section of the engine's YAML (Kalibr layout:
 * "cam0:" then "  T_cam_imu:" and "  - [a, b, c, d]" rows, or "intrinsics: [fx, fy, cx, cy]").
 */
inline bool yaml_numbers(const std::string& y, const char* block, const char* key, int n, float* out) {
    size_t b = y.find(std::string(block) + ":");
    if (b == std::string::npos) return false;
    size_t k = y.find(std::string(key) + ":", b);
    if (k == std::string::npos) return false;
    const char* p = y.c_str() + k + strlen(key) + 1;
    int got = 0;
    while (*p && got < n) {
        bool num = (*p >= '0' && *p <= '9') || ((*p == '-' || *p == '+' || *p == '.') && ((p[1] >= '0' && p[1] <= '9') || p[1] == '.'));
        if (num) { char* e; out[got++] = strtof(p, &e); p = e; }
        else if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')) break;  // the next key: too few numbers
        else p++;
    }
    return got == n;
}

/** Up to `n` numbers, the rest zero (the "fov" model has one distortion coefficient). */
inline bool yaml_padded(const std::string& y, const char* block, const char* key, int n, float* out) {
    for (int i = 0; i < n; i++) out[i] = 0;
    for (int m = n; m >= 1; m--) if (yaml_numbers(y, block, key, m, out)) return true;
    return false;
}

/** Row-major (Kalibr) -> column-major, as SpaceWalker's Transpose4x4Matrix. */
inline void transpose(float* m) { for (int i = 0; i < 4; i++) for (int j = i + 1; j < 4; j++) std::swap(m[i * 4 + j], m[j * 4 + i]); }

/**
 * Loads VITURE's hand tracking next to the engine already running (`sdk`: SpaceWalker's
 * folder). `yaml`: the engine's configuration (carina_a1088_viture_get_config). The
 * camera block is the 33 lines after "Imu.gyro_bias:", as SpaceWalker reads it.
 */
inline bool start(const std::string& sdk, const std::string& yaml, const std::string& raw_path) {
    auto set = [](const char* s) { std::lock_guard<std::mutex> l(g_mu); g_out.status = s; };
    std::string plug = sdk + "\\SpaceWalker.Unity_Data\\Plugins\\x86_64";
    HMODULE m = LoadLibraryExA((plug + "\\gesture_recognition.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m) { fprintf(stderr, "vhands: cannot load gesture_recognition.dll (%lu)\n", GetLastError()); set("gesture_recognition.dll introuvable"); return false; }
    auto bind = [&](auto& fn, const char* name) { fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(GetProcAddress(m, name)); return fn != nullptr; };
    if (!(bind(api.init, "gesture_init") & bind(api.uninit, "gesture_uninit") & bind(api.start, "gesture_start") & bind(api.stop, "gesture_stop") &
          bind(api.slam_pose, "OnSlamPose") & bind(api.camera, "OnCameraData") & bind(api.mode, "update_mode") & bind(api.reg, "Register_Gesture_Callbacks"))) {
        set("gesture_recognition.dll : fonctions manquantes");
        return false;
    }
    std::string cams = yaml;
    size_t g = yaml.find("Imu.gyro_bias:");
    if (g != std::string::npos) {
        size_t s = yaml.find('\n', g);
        cams = s == std::string::npos ? "" : yaml.substr(s + 1);
    }
    float extL[16], extR[16], extLR[16], inL[4], inR[4], dL[4], dR[4];
    if (!yaml_numbers(cams, "cam0", "T_cam_imu", 16, extL) || !yaml_numbers(cams, "cam0", "intrinsics", 4, inL) || !yaml_padded(cams, "cam0", "distortion_coeffs", 4, dL) ||
        !yaml_numbers(cams, "cam1", "T_cam_imu", 16, extR) || !yaml_numbers(cams, "cam1", "intrinsics", 4, inR) || !yaml_padded(cams, "cam1", "distortion_coeffs", 4, dR) ||
        !yaml_numbers(cams, "cam1", "T_cn_cnm1", 16, extLR)) {
        fprintf(stderr, "vhands: camera calibration not found in the engine's YAML\n");
        set("calibration des caméras introuvable dans la config du moteur");
        return false;
    }
    transpose(extL); transpose(extR); transpose(extLR);
    std::string model = sdk + "\\SpaceWalker.Unity_Data\\StreamingAssets\\model/";
    for (auto& c : model) if (c == '\\') c = '/';
    int rc = api.init(model.c_str());
    if (rc) { fprintf(stderr, "vhands: gesture_init %d\n", rc); set("gesture_init a échoué"); return false; }
    rc = api.mode(1, extL, dL, inL, extR, dR, inR, extLR);
    if (rc) { fprintf(stderr, "vhands: update_mode %d\n", rc); set("update_mode a échoué"); return false; }
    rc = api.reg(nullptr, on_hands, nullptr, nullptr);
    if (rc) { fprintf(stderr, "vhands: register %d\n", rc); set("enregistrement du rappel a échoué"); return false; }
    rc = api.start();
    if (rc) { fprintf(stderr, "vhands: gesture_start %d\n", rc); set("gesture_start a échoué"); return false; }
    if (!raw_path.empty()) g_raw = fopen(raw_path.c_str(), "w");
    printf("vhands: VITURE hand tracking started (fx %.1f, baseline %.1f mm)\n", inL[0], 1000 * std::sqrt(extLR[12] * extLR[12] + extLR[13] * extLR[13] + extLR[14] * extLR[14]));
    set("");
    g_on = true;
    return true;
}

/** From the engine's camera callback: the two tracking images (8-bit, w x h each). */
inline void feed_images(unsigned char* l, unsigned char* r, int w, int h, double t) {
    if (!g_on || !l || !r) return;
    g_last_img_t = t;
    api.camera(l, r, (unsigned)(w * h * 2), t);
}

/** From the engine's pose callback: its 16 floats, as SpaceWalker forwards them. */
inline void feed_pose(float* pose, double t) {
    if (!g_on || !pose) return;
    api.slam_pose(pose, t);
}

inline void stop() {
    if (!g_on) return;
    g_on = false;
    api.reg(nullptr, nullptr, nullptr, nullptr);
    api.stop();
    api.uninit();
    if (g_raw) { fclose(g_raw); g_raw = nullptr; }
}

}  // namespace vhands
