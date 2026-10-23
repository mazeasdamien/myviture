// The hands, from VITURE's hand tracking (vhands.hpp), in the head frame, for the
// dashboard and the RGB camera's fit, on their own thread.
//
// The RGB camera is fitted to the hands: where VITURE's 3D joints of a still hand are
// found in the RGB image by MediaPipe's hand landmark model (Apache 2.0, ONNX from
// opencv_zoo, on the CPU), at most 4 times a second. The same joints are projected into
// the two tracking cameras for the dashboard's views.

#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "calib.hpp"
#include "state.hpp"
#include "rgbcam.hpp"
#include "rgbcal.hpp"
#include "vhands.hpp"
#include "third_party/onnxruntime/include/onnxruntime_cxx_api.h"

#pragma comment(lib, "third_party/onnxruntime/lib/onnxruntime.lib")

namespace hands {

constexpr int N_JOINTS = 21;
// Bones between MediaPipe's 21 landmarks (for drawing).
inline const int BONES[][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 5}, {5, 6}, {6, 7}, {7, 8}, {5, 9}, {9, 10}, {10, 11}, {11, 12},
                               {9, 13}, {13, 14}, {14, 15}, {15, 16}, {13, 17}, {17, 18}, {18, 19}, {19, 20}, {0, 17}};
constexpr int N_BONES = sizeof(BONES) / sizeof(BONES[0]);
inline const char* GESTURES[] = {"—", "pince", "poing (grab)", "pointe (poke)", "main ouverte"};

/**
 * Hand pose from the 3D joints (metres): each finger's extension is its tip's distance
 * from the palm centre over the hand's size (wrist to middle knuckle): ~1.5 straight,
 * ~0.5 folded. Thumb: tip to the index knuckle.
 */
inline int classify(const float j[21][3], bool pinching, float ext[5]) {
    auto d = [&](int a, const float* b) { return std::sqrt((j[a][0] - b[0]) * (j[a][0] - b[0]) + (j[a][1] - b[1]) * (j[a][1] - b[1]) + (j[a][2] - b[2]) * (j[a][2] - b[2])); };
    float palm[3] = {0, 0, 0};
    for (int k : {0, 5, 9, 13, 17}) for (int a = 0; a < 3; a++) palm[a] += j[k][a] / 5;
    float size = d(0, j[9]);
    if (size < 0.03f) return 0;
    ext[0] = d(4, j[5]) / size;
    const int tips[4] = {8, 12, 16, 20};
    for (int f = 0; f < 4; f++) ext[f + 1] = d(tips[f], palm) / size;
    auto straight = [&](int f) { return ext[f] > 1.15f; };
    auto folded = [&](int f) { return ext[f] < 0.8f; };
    if (pinching) return 1;
    if (folded(1) && folded(2) && folded(3)) return 2;
    if (straight(1) && folded(2) && folded(3)) return 3;
    if (straight(1) && straight(2) && straight(3) && straight(4)) return 4;
    return 0;
}

struct Hand {
    bool right = false;           // from the wearer's point of view
    float joints[N_JOINTS][3];    // head frame (OpenGL axes), metres, at `t`
    float image[2][N_JOINTS][2];  // projected into the left / right tracking image (pixels)
    bool in_image[2] = {false, false};
    float pinch_m = 1;
    int gesture = 0;              // GESTURES[]: none, pinch, fist (grab), point (poke), open
    float ext[5] = {};            // extension per finger (thumb..pinky): tip distance / hand size
    float speed = 0;              // wrist, m/s
    bool rgb_seen = false;        // the hand model found this hand in the RGB image too
    float rgb[N_JOINTS][2];       // where it is in the RGB image (pixels)
};

struct Result {
    std::vector<Hand> hands;
    double t = 0;                 // image time (QPC seconds)
    uint32_t seq = 0;
    // diagnostics
    float rate = 0;               // VITURE answers per second
    float rgb_ms = 0;             // the last RGB landmark run
    int rgb_runs = 0;             // RGB landmark runs in the last second
    long pose_fails = 0;          // answers dropped: no engine pose at their image's time
    std::string status = "chargement…";
};

inline std::mutex g_mu;
inline Result g_result;
inline FILE* g_log = nullptr;
inline std::wstring g_log_dir;  // native/, for the per-frame trace

/** Diagnostics to myviture.log (next to the exe) and the console. */
inline void logf(const char* fmt, ...) {
    char buf[512];
    va_list a; va_start(a, fmt); vsnprintf(buf, sizeof buf, fmt, a); va_end(a);
    printf("%s", buf);
    if (g_log) { fprintf(g_log, "%.3f %s", GetTickCount64() / 1000.0, buf); fflush(g_log); }
}
inline std::atomic<bool> g_stop{false}, g_enabled{true};
inline std::thread g_thread;
/** Engine pose at device time t: head -> world, row-major 4x4. */
typedef std::function<bool(double, float*)> PoseFn;

// ------------------------------------------------------------------ small helpers

inline double qpc_s() { LARGE_INTEGER f, q; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&q); return (double)q.QuadPart / f.QuadPart; }

/** Square, rotated region of interest in an image (pixels). */
struct Roi { double cx = 0, cy = 0, size = 0, angle = 0; };

/** Samples `roi` into an n x n RGB float image in [0, 1] (bilinear, black outside). `px(x, y, rgb)` reads one pixel. */
template <typename Px>
inline void crop_with(Px px, int w, int h, const Roi& r, int n, float* out) {
    double c = std::cos(r.angle), s = std::sin(r.angle);
    for (int v = 0; v < n; v++) for (int u = 0; u < n; u++) {
        double lx = ((u + 0.5) / n - 0.5) * r.size, ly = ((v + 0.5) / n - 0.5) * r.size;
        double x = r.cx + lx * c - ly * s - 0.5, y = r.cy + lx * s + ly * c - 0.5;
        float* o = out + (v * n + u) * 3;
        o[0] = o[1] = o[2] = 0;
        int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
        if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) continue;
        double fx = x - x0, fy = y - y0;
        float a[3], b[3], cc[3], d[3];
        px(x0, y0, a); px(x0 + 1, y0, b); px(x0, y0 + 1, cc); px(x0 + 1, y0 + 1, d);
        for (int k = 0; k < 3; k++) o[k] = (float)((a[k] * (1 - fx) + b[k] * fx) * (1 - fy) + (cc[k] * (1 - fx) + d[k] * fx) * fy);
    }
}

/** BGRA image (the RGB camera). */
inline void crop_bgra(const uint32_t* img, int w, int h, const Roi& r, int n, float* out) {
    crop_with([&](int x, int y, float* rgb) { uint32_t p = img[y * w + x]; rgb[0] = ((p >> 16) & 255) / 255.0f; rgb[1] = ((p >> 8) & 255) / 255.0f; rgb[2] = (p & 255) / 255.0f; }, w, h, r, n, out);
}

/** ROI-local normalised point (0..1) -> image pixels. */
inline void from_roi(const Roi& r, double nx, double ny, double& x, double& y) {
    double c = std::cos(r.angle), s = std::sin(r.angle);
    double lx = (nx - 0.5) * r.size, ly = (ny - 0.5) * r.size;
    x = r.cx + lx * c - ly * s;
    y = r.cy + lx * s + ly * c;
}

/** MediaPipe's rotation: the line from p0 to p1 points up. */
inline double hand_angle(double x0, double y0, double x1, double y1) {
    double a = 3.14159265 / 2 - std::atan2(-(y1 - y0), x1 - x0);
    return a - 2 * 3.14159265 * std::floor((a + 3.14159265) / (2 * 3.14159265));
}

// ------------------------------------------------------------------ model

struct Models {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "myviture"};
    Ort::Session lm{nullptr};
    std::vector<std::string> lm_in, lm_out;

    static std::vector<std::string> names(Ort::Session& s, bool input) {
        Ort::AllocatorWithDefaultOptions a;
        std::vector<std::string> v;
        size_t n = input ? s.GetInputCount() : s.GetOutputCount();
        for (size_t i = 0; i < n; i++) v.push_back((input ? s.GetInputNameAllocated(i, a) : s.GetOutputNameAllocated(i, a)).get());
        return v;
    }

    /** The landmark model, on the CPU (the GPU is the glasses' and the depth's). */
    bool load(const std::wstring& dir) {
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        so.SetIntraOpNumThreads(2);
        try {
            lm = Ort::Session(env, (dir + L"handpose_estimation_mediapipe_2023feb.onnx").c_str(), so);
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "hands: landmark model failed: %s\n", e.what());
            return false;
        }
        lm_in = names(lm, true); lm_out = names(lm, false);
        return true;
    }
};

/** Next region from the landmarks (MediaPipe's tracking): wrist->middle MCP up, x2, shift -0.1. */
inline Roi roi_from_landmarks(const float lm[N_JOINTS][2]) {
    double a = hand_angle(lm[0][0], lm[0][1], lm[9][0], lm[9][1]);
    double c = std::cos(-a), s = std::sin(-a);
    double minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
    for (int i : {0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18}) {
        double x = lm[i][0] * c - lm[i][1] * s, y = lm[i][0] * s + lm[i][1] * c;
        minx = std::min(minx, x); maxx = std::max(maxx, x); miny = std::min(miny, y); maxy = std::max(maxy, y);
    }
    double w = maxx - minx, h = maxy - miny, rx = (minx + maxx) / 2, ry = (miny + maxy) / 2;
    ry -= 0.1 * h;
    double cx = rx * std::cos(a) - ry * std::sin(a), cy = rx * std::sin(a) + ry * std::cos(a);
    return {cx, cy, std::max(w, h) * 2.0, a};
}

constexpr float PRESENCE_MIN = 0.3f;

/**
 * Landmarks of the hand in `roi`; false if the model says there is no hand. presence and
 * hand (handedness, 1 = the wearer's right) are returned for the diagnostics.
 */
inline bool landmarks_run(Models& M, std::vector<float>& in, const Roi& roi, float out[N_JOINTS][2], float& presence, float& hand) {
    const int N = 224;
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t shape[4] = {1, N, N, 3};
    Ort::Value t = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(), shape, 4);
    const char* in_names[] = {M.lm_in[0].c_str()};
    std::vector<const char*> out_names;
    for (auto& s : M.lm_out) out_names.push_back(s.c_str());
    auto o = M.lm.Run(Ort::RunOptions{nullptr}, in_names, &t, 1, out_names.data(), out_names.size());
    // Outputs: landmarks (63, in 224 px), presence, handedness, world landmarks (63); the
    // two scalars told apart by name when the export names them, else by order.
    const float* lmk = nullptr; int scalars = 0;
    presence = hand = 0;
    for (size_t i = 0; i < o.size(); i++) {
        size_t n = o[i].GetTensorTypeAndShapeInfo().GetElementCount();
        const std::string& name = M.lm_out[i];
        float v = n == 1 ? o[i].GetTensorData<float>()[0] : 0;
        if (n == 63 && !lmk) lmk = o[i].GetTensorData<float>();
        else if (n == 1 && name.find("hand") != std::string::npos) hand = v;
        else if (n == 1 && (name.find("conf") != std::string::npos || name.find("pres") != std::string::npos || name.find("flag") != std::string::npos)) presence = v;
        else if (n == 1) { (scalars++ == 0 ? presence : hand) = v; }
    }
    if (!lmk || presence < PRESENCE_MIN) return false;
    for (int k = 0; k < N_JOINTS; k++) {
        double x, y;
        from_roi(roi, lmk[k * 3] / N, lmk[k * 3 + 1] / N, x, y);
        out[k][0] = (float)x; out[k][1] = (float)y;
    }
    return true;
}

// ------------------------------------------------------------------ thread

inline void run(std::wstring model_dir, calib::Rig rig, PoseFn pose_at) {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    Models M;
    bool have_model = M.load(model_dir);
    { std::lock_guard<std::mutex> l(g_mu); g_result.status = have_model ? "" : "modèle de main introuvable : pas de calage RGB"; }
    uint32_t seen = 0;
    double last_rgb_t = 0, rate_t0 = qpc_s();
    int answers = 0, rgb_runs = 0;
    float rate = 0; int rgb_rate = 0;
    float prev_w[2][3] = {}; double prev_t[2] = {-1, -1};
    std::vector<float> crop_in(224 * 224 * 3);

    while (!g_stop) {
        Sleep(10);
        vhands::Out vo;
        { std::lock_guard<std::mutex> l(vhands::g_mu); if (vhands::g_out.seq == seen) continue; vo = vhands::g_out; }
        seen = vo.seq;
        answers++;
        if (!g_enabled || vo.img_t <= 0) continue;
        float pose[16];
        if (!pose_at(vo.img_t, pose)) { std::lock_guard<std::mutex> l(g_mu); g_result.pose_fails++; continue; }
        // World -> head: the pose's inverse (rigid).
        auto to_head = [&](const float* w, float* h) {
            float d[3] = {w[0] - pose[3], w[1] - pose[7], w[2] - pose[11]};
            for (int i = 0; i < 3; i++) h[i] = pose[0 + i] * d[0] + pose[4 + i] * d[1] + pose[8 + i] * d[2];
        };
        Result res;
        res.t = vo.img_t; res.seq = vo.seq;
        for (int side = 0; side < 2; side++) {
            float w[N_JOINTS][3];
            if (!vhands::hand_world(vo.data, side, w)) { prev_t[side] = -1; continue; }
            Hand h;
            h.right = side == 1;
            for (int k = 0; k < N_JOINTS; k++) to_head(w[k], h.joints[k]);
            for (int c = 0; c < 2 && rig.ok; c++) {
                h.in_image[c] = true;
                for (int k = 0; k < N_JOINTS; k++) {
                    double p[3] = {h.joints[k][0], h.joints[k][1], h.joints[k][2]}, u, v;
                    if (!rig.cam[c].project(p, u, v)) { h.in_image[c] = false; break; }
                    h.image[c][k][0] = (float)u; h.image[c][k][1] = (float)v;
                }
            }
            float dx = w[4][0] - w[8][0], dy = w[4][1] - w[8][1], dz = w[4][2] - w[8][2];
            h.pinch_m = std::sqrt(dx * dx + dy * dy + dz * dz);
            h.gesture = classify(h.joints, h.pinch_m < 0.025f, h.ext);
            double dt = vo.t - prev_t[side];
            if (prev_t[side] > 0 && dt > 0.005 && dt < 0.5) {
                float mx = h.joints[0][0] - prev_w[side][0], my = h.joints[0][1] - prev_w[side][1], mz = h.joints[0][2] - prev_w[side][2];
                h.speed = (float)(std::sqrt(mx * mx + my * my + mz * mz) / dt);
            } else h.speed = 1;
            for (int a = 0; a < 3; a++) prev_w[side][a] = h.joints[0][a];  // head frame: the RGB camera moves with the head
            prev_t[side] = vo.t;
            res.hands.push_back(h);
        }

        // RGB camera fit: still hands, at most 4 per second (2 once fitted).
        bool fitted = rgbcal::get().fitted;
        if (fitted && rgbcal::g_frozen) {
            double now = qpc_s();
            if (now - rate_t0 >= 1) { rate = (float)(answers / (now - rate_t0)); answers = 0; rgb_runs = 0; rate_t0 = now; }
            std::lock_guard<std::mutex> l(g_mu);
            res.status = g_result.status; res.rate = rate; res.pose_fails = g_result.pose_fails;
            g_result = std::move(res);
            continue;
        }
        bool any_still = false;
        for (auto& h : res.hands) if (h.speed < 0.15f) any_still = true;
        if (have_model && any_still && qpc_s() - last_rgb_t > (fitted ? 0.5 : 0.25)) {
            last_rgb_t = qpc_s();
            // The newest RGB frame and where the current camera model puts each still hand.
            static std::vector<uint32_t> frame;
            int fw = 0, fh = 0;
            rgbcal::Cam cam;
            {
                std::lock_guard<std::mutex> l(rgbcam::g_frame.mu);
                auto& f = rgbcam::g_frame;
                if (f.w && f.bgra.size() >= (size_t)f.w * f.h) { frame = f.bgra; fw = f.w; fh = f.h; cam = rgbcal::current(f.w, f.h); }
            }
            std::vector<rgbcal::Pair> pairs;
            double wrist[3] = {0, 0, 0};
            static double last_log = 0;
            float best_logged = 0;
            for (size_t i = 0; fw && i < res.hands.size() && i < 2; i++) {
                if (res.hands[i].speed >= 0.15f) continue;
                float proj[N_JOINTS][2];
                bool ok = true;
                for (int k = 0; k < N_JOINTS && ok; k++) {
                    double p[3] = {res.hands[i].joints[k][0], res.hands[i].joints[k][1], res.hands[i].joints[k][2]}, u, v;
                    ok = rgbcal::project(cam, p, u, v);
                    proj[k][0] = (float)u; proj[k][1] = (float)v;
                }
                if (!ok) continue;
                Roi base = roi_from_landmarks(proj);
                base.size *= cam.fitted ? 1.3 : 1.5;
                // Before the first fit the model can be far off: look around the projection
                // too (3 x 3 offsets of 60 % of the region), keep the most confident.
                std::vector<Roi> tries;
                if (cam.fitted) tries.push_back(base);
                else for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) { Roi r = base; r.cx += dx * 0.6 * base.size; r.cy += dy * 0.6 * base.size; tries.push_back(r); }
                float best = 0; float best_lm[N_JOINTS][2];
                double t0 = qpc_s();
                for (auto& r : tries) {
                    crop_bgra(frame.data(), fw, fh, r, 224, crop_in.data());
                    float pres, hand, lm[N_JOINTS][2];
                    rgb_runs++;
                    if (landmarks_run(M, crop_in, r, lm, pres, hand) && pres > best) { best = pres; memcpy(best_lm, lm, sizeof lm); }
                }
                // Refined once on the region the landmarks give (MediaPipe's own tracking).
                if (best >= 0.5f) {
                    Roi r = roi_from_landmarks(best_lm);
                    crop_bgra(frame.data(), fw, fh, r, 224, crop_in.data());
                    float pres, hand, lm[N_JOINTS][2];
                    rgb_runs++;
                    if (landmarks_run(M, crop_in, r, lm, pres, hand) && pres >= 0.5f) memcpy(best_lm, lm, sizeof lm);
                }
                res.rgb_ms = (float)((qpc_s() - t0) * 1000);
                best_logged = std::max(best_logged, best);
                if (best < 0.5f) continue;
                memcpy(res.hands[i].rgb, best_lm, sizeof best_lm);
                res.hands[i].rgb_seen = true;
                for (int a = 0; a < 3; a++) wrist[a] = res.hands[i].joints[0][a];
                for (int k = 0; k < N_JOINTS; k++)
                    pairs.push_back({{res.hands[i].joints[k][0], res.hands[i].joints[k][1], res.hands[i].joints[k][2]}, best_lm[k][0], best_lm[k][1]});
            }
            if (qpc_s() - last_log > 2) {
                last_log = qpc_s();
                int frames_now, runs; double spread; { std::lock_guard<std::mutex> l(rgbcal::g_mu); frames_now = rgbcal::g_frames; spread = rgbcal::g_spread; runs = rgbcal::g_fit_runs; }
                logf("rgb fit: still hand in RGB, best presence %.2f, %d poses so far, wrists spread %.0f cm, %d fits run%s\n", best_logged, frames_now, spread * 100, runs, cam.fitted ? " (fitted)" : "");
            }
            if (!pairs.empty()) {
                static int logged = 0;
                rgbcal::add(pairs, wrist);
                rgbcal::Fit fi; int runs;
                { std::lock_guard<std::mutex> l(rgbcal::g_mu); fi = rgbcal::g_fit; runs = rgbcal::g_fit_runs; }
                if (runs != logged && (logged = runs))
                    logf("rgb fit: %s rms %.1f px on %d points, rot %.1f deg, shift %.1f %.1f %.1f cm, f x%.2f\n", fi.ok ? "ok" : "rejected", fi.rms_px, fi.n, fi.rot_deg, fi.shift_cm[0], fi.shift_cm[1], fi.shift_cm[2], fi.f_rel);
            }
        }

        double now = qpc_s();
        if (now - rate_t0 >= 1) { rate = (float)(answers / (now - rate_t0)); rgb_rate = rgb_runs; answers = 0; rgb_runs = 0; rate_t0 = now; }
        res.rate = rate; res.rgb_runs = rgb_rate;
        {
            std::lock_guard<std::mutex> l(g_mu);
            res.status = g_result.status;
            if (!res.rgb_ms) res.rgb_ms = g_result.rgb_ms;
            res.pose_fails = g_result.pose_fails;
            g_result = std::move(res);
        }
    }
}

inline void start(const std::wstring& model_dir, const calib::Rig& rig, PoseFn pose_at) {
    std::wstring dir = model_dir.substr(0, model_dir.find_last_of(L'\\', model_dir.size() - 2) + 1);
    g_log = _wfopen((dir + L"myviture.log").c_str(), L"w");
    g_log_dir = dir;
    g_thread = std::thread(run, model_dir, rig, pose_at);
}
inline void stop() { g_stop = true; if (g_thread.joinable()) g_thread.join(); }

}  // namespace hands
