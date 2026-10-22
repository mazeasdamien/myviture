// Stereo hand tracking: MediaPipe's palm detector and hand landmark models (Apache 2.0,
// ONNX from opencv_zoo) on ONNX Runtime + DirectML, on its own thread.
//
// Per tracking camera: the palm detector runs only while no hand is followed; once a hand
// is found, the landmark model runs on a rotated square that follows it from frame to
// frame (MediaPipe's own scheme). Hands seen by both cameras are matched by side and
// their 21 joints triangulated with the factory calibration: metric, in the head frame
// at the image's time (QPC seconds, the engine's clock). Smoothed (One Euro), pinch
// with hysteresis.

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
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "calib.hpp"
#include "state.hpp"
#include "third_party/onnxruntime/include/onnxruntime_cxx_api.h"
#include "third_party/onnxruntime/include/dml_provider_factory.h"

#pragma comment(lib, "third_party/onnxruntime/lib/onnxruntime.lib")

namespace hands {

constexpr int N_JOINTS = 21;
// Bones between MediaPipe's 21 landmarks (for drawing).
inline const int BONES[][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 5}, {5, 6}, {6, 7}, {7, 8}, {5, 9}, {9, 10}, {10, 11}, {11, 12},
                               {9, 13}, {13, 14}, {14, 15}, {15, 16}, {13, 17}, {17, 18}, {18, 19}, {19, 20}, {0, 17}};
constexpr int N_BONES = sizeof(BONES) / sizeof(BONES[0]);

struct Hand {
    bool right = false;           // from the wearer's point of view
    float joints[N_JOINTS][3];    // head frame (OpenGL axes), metres, at `t`
    float image[2][N_JOINTS][2];  // pixel landmarks in the left / right image
    bool pinching = false;
    float pinch_m = 1;
    float gap_mm = 0;             // mean distance between the two cameras' rays
};

struct Result {
    std::vector<Hand> hands;
    double t = 0;                 // image time (QPC seconds)
    uint32_t seq = 0;
    // diagnostics
    float rate = 0, ms_palm = 0, ms_landmarks = 0, ms_total = 0, latency_ms = 0;
    int palm_runs = 0;            // palm detections in the last second
    std::string status = "chargement…";
    std::string device;
};

inline std::mutex g_mu;
inline Result g_result;
inline FILE* g_log = nullptr;

/** Diagnostics to myviture.log (next to the exe) and the console. */
inline void logf(const char* fmt, ...) {
    char buf[512];
    va_list a; va_start(a, fmt); vsnprintf(buf, sizeof buf, fmt, a); va_end(a);
    printf("%s", buf);
    if (g_log) { fprintf(g_log, "%.3f %s", GetTickCount64() / 1000.0, buf); fflush(g_log); }
}
inline std::atomic<bool> g_stop{false}, g_enabled{true};
inline std::thread g_thread;

// ------------------------------------------------------------------ small helpers

inline double qpc_s() { LARGE_INTEGER f, q; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&q); return (double)q.QuadPart / f.QuadPart; }

/** One Euro filter, per coordinate. */
struct OneEuro {
    double min_cutoff = 2.5, beta = 8, d_cutoff = 1;
    double x = 0, dx = 0, t = -1;
    static double alpha(double cutoff, double dt) { double tau = 1 / (2 * 3.14159265 * cutoff); return 1 / (1 + tau / dt); }
    double apply(double v, double now) {
        if (t < 0 || now - t > 0.3) { x = v; dx = 0; t = now; return v; }
        double dt = std::max(1e-4, now - t);
        double d = (v - x) / dt;
        dx += alpha(d_cutoff, dt) * (d - dx);
        double a = alpha(min_cutoff + beta * std::fabs(dx), dt);
        x += a * (v - x);
        t = now;
        return x;
    }
};

/** Square, rotated region of interest in an image (pixels). */
struct Roi { double cx = 0, cy = 0, size = 0, angle = 0; };

/** Samples `roi` into an n x n RGB float image in [0, 1] (gray replicated, bilinear, black outside). */
inline void crop(const uint8_t* img, int w, int h, const Roi& r, int n, float* out) {
    double c = std::cos(r.angle), s = std::sin(r.angle);
    for (int v = 0; v < n; v++) for (int u = 0; u < n; u++) {
        double lx = ((u + 0.5) / n - 0.5) * r.size, ly = ((v + 0.5) / n - 0.5) * r.size;
        double x = r.cx + lx * c - ly * s - 0.5, y = r.cy + lx * s + ly * c - 0.5;
        float g = 0;
        int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
        if (x0 >= 0 && y0 >= 0 && x0 + 1 < w && y0 + 1 < h) {
            double fx = x - x0, fy = y - y0;
            const uint8_t* p = img + y0 * w + x0;
            g = (float)(((p[0] * (1 - fx) + p[1] * fx) * (1 - fy) + (p[w] * (1 - fx) + p[w + 1] * fx) * fy) / 255.0);
        }
        float* o = out + (v * n + u) * 3;
        o[0] = o[1] = o[2] = g;
    }
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

// ------------------------------------------------------------------ models

struct Models {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "myviture"};
    Ort::Session palm{nullptr}, lm{nullptr};
    std::vector<std::string> palm_in, palm_out, lm_in, lm_out;
    std::vector<std::array<float, 2>> anchors;  // palm detector SSD anchors (normalised)
    std::string device;

    static std::vector<std::string> names(Ort::Session& s, bool input) {
        Ort::AllocatorWithDefaultOptions a;
        std::vector<std::string> v;
        size_t n = input ? s.GetInputCount() : s.GetOutputCount();
        for (size_t i = 0; i < n; i++) v.push_back((input ? s.GetInputNameAllocated(i, a) : s.GetOutputNameAllocated(i, a)).get());
        return v;
    }

    bool load(const std::wstring& dir) {
        auto open = [&](const std::wstring& file, bool dml) {
            Ort::SessionOptions so;
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            if (dml) {
                // DirectML wants sequential execution and no memory pattern.
                so.DisableMemPattern();
                so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(so, 0));
            } else so.SetIntraOpNumThreads(2);
            return Ort::Session(env, (dir + file).c_str(), so);
        };
        for (bool dml : {true, false}) {
            try {
                palm = open(L"palm_detection_mediapipe_2023feb.onnx", dml);
                lm = open(L"handpose_estimation_mediapipe_2023feb.onnx", dml);
                device = dml ? "GPU (DirectML)" : "CPU";
                break;
            } catch (const Ort::Exception& e) {
                fprintf(stderr, "hands: %s session failed: %s\n", dml ? "DirectML" : "CPU", e.what());
                palm = Ort::Session(nullptr); lm = Ort::Session(nullptr);
            }
        }
        if (!palm || !lm) return false;
        palm_in = names(palm, true); palm_out = names(palm, false);
        lm_in = names(lm, true); lm_out = names(lm, false);
        for (size_t i = 0; i < lm_out.size(); i++) {
            auto sh = lm.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            std::string dims;
            for (auto d : sh) dims += std::to_string(d) + " ";
            logf("hands: landmark output %zu '%s' [%s]\n", i, lm_out[i].c_str(), dims.c_str());
        }
        // MediaPipe palm SSD anchors, 192 x 192: stride 8 (2 per cell) then stride 16 (6 per cell).
        for (int y = 0; y < 24; y++) for (int x = 0; x < 24; x++) for (int k = 0; k < 2; k++) anchors.push_back({(x + 0.5f) / 24, (y + 0.5f) / 24});
        for (int y = 0; y < 12; y++) for (int x = 0; x < 12; x++) for (int k = 0; k < 6; k++) anchors.push_back({(x + 0.5f) / 12, (y + 0.5f) / 12});
        return true;
    }
};

struct Palm { double x0, y0, x1, y1, score; double kp[7][2]; };

/** Palm detections in a w x h gray image (letterboxed to 192 x 192). */
inline std::vector<Palm> detect_palms(Models& M, const uint8_t* img, int w, int h) {
    const int N = 192;
    double scale = std::max(w, h);                 // the letterbox square, in pixels
    double padx = (scale - w) / 2, pady = (scale - h) / 2;
    static std::vector<float> in(N * N * 3);
    Roi sq{w / 2.0, h / 2.0, scale, 0};
    crop(img, w, h, sq, N, in.data());
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t shape[4] = {1, N, N, 3};
    Ort::Value t = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(), shape, 4);
    const char* in_names[] = {M.palm_in[0].c_str()};
    std::vector<const char*> out_names;
    for (auto& s : M.palm_out) out_names.push_back(s.c_str());
    auto out = M.palm.Run(Ort::RunOptions{nullptr}, in_names, &t, 1, out_names.data(), out_names.size());
    // Outputs: regressors (1, 2016, 18) and scores (1, 2016, 1), in either order.
    int ri = out[0].GetTensorTypeAndShapeInfo().GetShape().back() == 18 ? 0 : 1;
    const float* reg = out[ri].GetTensorData<float>();
    const float* sc = out[1 - ri].GetTensorData<float>();
    std::vector<Palm> cand;
    for (size_t i = 0; i < M.anchors.size(); i++) {
        double s = 1 / (1 + std::exp(-sc[i]));
        if (s < 0.4) continue;
        const float* r = reg + i * 18;
        double ax = M.anchors[i][0], ay = M.anchors[i][1];
        double cx = r[0] / N + ax, cy = r[1] / N + ay, bw = r[2] / N, bh = r[3] / N;
        Palm p{(cx - bw / 2) * scale - padx, (cy - bh / 2) * scale - pady, (cx + bw / 2) * scale - padx, (cy + bh / 2) * scale - pady, s, {}};
        for (int k = 0; k < 7; k++) { p.kp[k][0] = (r[4 + 2 * k] / N + ax) * scale - padx; p.kp[k][1] = (r[5 + 2 * k] / N + ay) * scale - pady; }
        cand.push_back(p);
    }
    std::sort(cand.begin(), cand.end(), [](const Palm& a, const Palm& b) { return a.score > b.score; });
    std::vector<Palm> keep;
    for (auto& c : cand) {
        bool dup = false;
        for (auto& k : keep) {
            double ix = std::max(0.0, std::min(c.x1, k.x1) - std::max(c.x0, k.x0)), iy = std::max(0.0, std::min(c.y1, k.y1) - std::max(c.y0, k.y0));
            double inter = ix * iy, uni = (c.x1 - c.x0) * (c.y1 - c.y0) + (k.x1 - k.x0) * (k.y1 - k.y0) - inter;
            if (inter / uni > 0.3) { dup = true; break; }
        }
        if (!dup) keep.push_back(c);
        if (keep.size() == 2) break;
    }
    return keep;
}

/** MediaPipe's palm -> hand region: rotate wrist->middle finger up, shift up half a box, x2.6. */
inline Roi roi_from_palm(const Palm& p) {
    double w = p.x1 - p.x0, h = p.y1 - p.y0, a = hand_angle(p.kp[0][0], p.kp[0][1], p.kp[2][0], p.kp[2][1]);
    double cx = (p.x0 + p.x1) / 2 + 0.5 * h * std::sin(a), cy = (p.y0 + p.y1) / 2 - 0.5 * h * std::cos(a);
    return {cx, cy, std::max(w, h) * 2.6, a};
}

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
inline bool landmarks(Models& M, const uint8_t* img, int w, int h, const Roi& roi, float out[N_JOINTS][2], float& presence, float& hand, float world[N_JOINTS][3] = nullptr) {
    const int N = 224;
    static std::vector<float> in(N * N * 3);
    crop(img, w, h, roi, N, in.data());
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t shape[4] = {1, N, N, 3};
    Ort::Value t = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(), shape, 4);
    const char* in_names[] = {M.lm_in[0].c_str()};
    std::vector<const char*> out_names;
    for (auto& s : M.lm_out) out_names.push_back(s.c_str());
    auto o = M.lm.Run(Ort::RunOptions{nullptr}, in_names, &t, 1, out_names.data(), out_names.size());
    // Outputs: landmarks (63, in 224 px), presence, handedness, world landmarks (63); the
    // two scalars told apart by name when the export names them, else by order.
    const float* lmk = nullptr; const float* wld = nullptr; int scalars = 0;
    presence = hand = 0;
    for (size_t i = 0; i < o.size(); i++) {
        size_t n = o[i].GetTensorTypeAndShapeInfo().GetElementCount();
        const std::string& name = M.lm_out[i];
        float v = n == 1 ? o[i].GetTensorData<float>()[0] : 0;
        if (n == 63 && !lmk) lmk = o[i].GetTensorData<float>();
        else if (n == 63) wld = o[i].GetTensorData<float>();
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
    if (world && wld) {
        // Metres, hand-centred, in the crop's orientation: turn back like the image points.
        double c = std::cos(roi.angle), sn = std::sin(roi.angle);
        for (int k = 0; k < N_JOINTS; k++) {
            float wx = wld[k * 3], wy = wld[k * 3 + 1];
            world[k][0] = (float)(wx * c - wy * sn); world[k][1] = (float)(wx * sn + wy * c); world[k][2] = wld[k * 3 + 2];
        }
    }
    return true;
}

/**
 * Where a hand seen by camera `from` should be in the other camera: its depth from the
 * ratio of the metric hand (world landmarks) to the hand in the image, then the 3D
 * joints projected into the other camera. The region covers the projected joints.
 */
inline bool predict_roi(const calib::Rig& rig, int from, const float lm[N_JOINTS][2], const float world[N_JOINTS][3], Roi& out, double& depth) {
    const calib::Camera& A = rig.cam[from];
    const calib::Camera& B = rig.cam[1 - from];
    double n[N_JOINTS][2];
    for (int k = 0; k < N_JOINTS; k++) A.undistort(lm[k][0], lm[k][1], n[k][0], n[k][1]);
    // Scale: metric distances over normalised-image distances, summed over all joint pairs.
    double num = 0, den = 0;
    for (int i = 0; i < N_JOINTS; i++) for (int j = i + 1; j < N_JOINTS; j++) {
        double dn = std::hypot(n[i][0] - n[j][0], n[i][1] - n[j][1]);
        double dw = std::hypot(world[i][0] - world[j][0], world[i][1] - world[j][1]);
        num += dw; den += dn;
    }
    if (den < 1e-6) return false;
    depth = num / den;
    if (depth < 0.12 || depth > 1.5) return false;
    float proj[N_JOINTS][2];
    for (int k = 0; k < N_JOINTS; k++) {
        double c[3] = {n[k][0] * depth, n[k][1] * depth, depth}, p[3], u, v;
        A.to_head(c, p);
        if (!B.project(p, u, v)) return false;
        proj[k][0] = (float)u; proj[k][1] = (float)v;
    }
    out = roi_from_landmarks(proj);
    out.size *= 1.25;  // the depth is rough: a little margin
    return true;
}

// ------------------------------------------------------------------ tracking thread

/**
 * One hand followed in one camera. Its side is a running average of the model's
 * handedness: one frame's flip (fingers folded in a pinch) does not change it.
 */
struct Track {
    bool on = false;
    Roi roi;
    float side = 0.5f;  // 1 = right
    float lm[N_JOINTS][2];
    float world[N_JOINTS][3];
    int misses = 0;
    float presence = 0;
    bool fresh = false;  // landmarks from this frame
    bool assisted = false;  // started from the other camera's hand
    bool right() const { return side > 0.5f; }
};
constexpr int MAX_MISSES = 4;  // frames a hand may be missed before it is dropped

inline void run(std::wstring model_dir, calib::Rig rig) {
    Models M;
    if (!M.load(model_dir)) { std::lock_guard<std::mutex> l(g_mu); g_result.status = "modèles de mains introuvables (native\\models)"; return; }
    { std::lock_guard<std::mutex> l(g_mu); g_result.status = ""; g_result.device = M.device; }
    printf("hands: models loaded on %s\n", M.device.c_str());

    Track tracks[2][2];  // [camera][slot]
    OneEuro filters[2][N_JOINTS][3];  // [side][joint][axis]
    bool pinch_state[2] = {};
    uint32_t seen = 0;
    std::vector<uint8_t> img[2];
    int w = 0, h = 0;
    double rate_t0 = qpc_s(); int frames = 0, palms = 0;
    float rate = 0; int palm_rate = 0;

    while (!g_stop) {
        double t_img = 0;
        {
            std::lock_guard<std::mutex> l(g_cams.mu);
            if (g_cams.seq != seen && g_cams.w) {
                seen = g_cams.seq; img[0] = g_cams.px[0]; img[1] = g_cams.px[1]; w = g_cams.w; h = g_cams.h; t_img = g_cams.t;
            }
        }
        if (!t_img || !g_enabled) { Sleep(2); continue; }
        if (img[0].size() < (size_t)w * h || img[1].size() < (size_t)w * h) continue;

        double t0 = qpc_s(), t_palm = 0;
        // Per camera: follow the known hands, look for new ones when fewer than two.
        for (int c = 0; c < 2; c++) {
            for (int k = 0; k < 2; k++) {
                auto& tr = tracks[c][k];
                tr.fresh = false;
                if (!tr.on) continue;
                float pres, hand;
                float lm[N_JOINTS][2];
                if (landmarks(M, img[c].data(), w, h, tr.roi, lm, pres, hand, tr.world)) {
                    memcpy(tr.lm, lm, sizeof lm);
                    tr.side = 0.85f * tr.side + 0.15f * hand;
                    tr.roi = roi_from_landmarks(tr.lm);
                    tr.misses = 0; tr.presence = pres; tr.fresh = true;
                } else if (++tr.misses > MAX_MISSES) {
                    tr.on = false;
                    logf("hands: cam %d slot %d lost (presence %.2f, handedness %.2f)\n", c, k, pres, hand);
                }
            }
            int on = tracks[c][0].on + tracks[c][1].on;
            if (on < 2) {
                double tp = qpc_s();
                auto found = detect_palms(M, img[c].data(), w, h);
                t_palm += qpc_s() - tp;
                palms++;
                for (auto& p : found) {
                    Roi r = roi_from_palm(p);
                    // Skip palms inside a hand already followed.
                    bool known = false;
                    for (auto& tr : tracks[c]) if (tr.on && std::hypot(tr.roi.cx - r.cx, tr.roi.cy - r.cy) < tr.roi.size * 0.3) known = true;
                    if (known) continue;
                    for (int k = 0; k < 2; k++) {
                        auto& tr = tracks[c][k];
                        if (tr.on) continue;
                        float pres, hand;
                        if (landmarks(M, img[c].data(), w, h, r, tr.lm, pres, hand, tr.world)) {
                            tr.on = true; tr.side = hand; tr.roi = roi_from_landmarks(tr.lm); tr.misses = 0; tr.presence = pres; tr.fresh = true; tr.assisted = false;
                            logf("hands: cam %d slot %d found (presence %.2f, handedness %.2f, palm %.2f)\n", c, k, pres, hand, p.score);
                        }
                        break;
                    }
                }
            }
        }

        // Stereo assistance: a hand fresh in one camera and absent from the other is
        // looked for in the other at its predicted place, without the palm detector.
        for (int c = 0; c < 2; c++) {
            int o = 1 - c;
            for (auto& src : tracks[c]) {
                if (!src.on || !src.fresh) continue;
                // Already matched by a fresh track in the other camera?
                bool have = false;
                for (auto& tr : tracks[o]) if (tr.on && tr.fresh) {
                    double sum = 0; int n = 0;
                    for (int k : {0, 5, 9, 13, 17}) {
                        double d1[3], d2[3], p[3], gap;
                        rig.cam[c].ray(src.lm[k][0], src.lm[k][1], d1);
                        rig.cam[o].ray(tr.lm[k][0], tr.lm[k][1], d2);
                        if (calib::triangulate(rig.cam[c].origin, d1, rig.cam[o].origin, d2, p, gap)) { sum += gap; n++; }
                    }
                    if (n && sum / n < 0.04) have = true;
                }
                if (have) continue;
                Roi r; double depth;
                if (!predict_roi(rig, c, src.lm, src.world, r, depth)) continue;
                Track* slot = nullptr;
                for (auto& tr : tracks[o]) if (!tr.on) { slot = &tr; break; }
                if (!slot) for (auto& tr : tracks[o]) if (!tr.fresh) { slot = &tr; break; }  // a stale track: replace it
                if (!slot) continue;
                float pres, hand;
                float lm[N_JOINTS][2], wd[N_JOINTS][3];
                if (landmarks(M, img[o].data(), w, h, r, lm, pres, hand, wd)) {
                    memcpy(slot->lm, lm, sizeof lm); memcpy(slot->world, wd, sizeof wd);
                    slot->on = true; slot->side = src.side; slot->roi = roi_from_landmarks(lm); slot->misses = 0; slot->presence = pres; slot->fresh = true; slot->assisted = true;
                    logf("hands: cam %d found from cam %d (depth %.2f m, presence %.2f)\n", o, c, depth, pres);
                } else if (frames % 25 == 0) logf("hands: cam %d assist from cam %d failed (depth %.2f m, presence %.2f)\n", o, c, depth, pres);
            }
        }

        // Stereo: pair the two cameras' hands by geometry (the pairing whose rays meet
        // best on the palm joints), not by their handedness guesses.
        auto pair_gap = [&](const Track& a, const Track& b) {
            double sum = 0;
            for (int k : {0, 5, 9, 13, 17}) {
                double d1[3], d2[3], p[3], gap;
                rig.cam[0].ray(a.lm[k][0], a.lm[k][1], d1);
                rig.cam[1].ray(b.lm[k][0], b.lm[k][1], d2);
                if (!calib::triangulate(rig.cam[0].origin, d1, rig.cam[1].origin, d2, p, gap)) return 1e9;
                sum += gap;
            }
            return sum / 5;
        };
        struct Match { Track* L; Track* R; double gap; };
        std::vector<Match> matches;
        {
            Track* Ls[2]; Track* Rs[2]; int nl = 0, nr = 0;
            for (auto& tr : tracks[0]) if (tr.on && tr.fresh) Ls[nl++] = &tr;
            for (auto& tr : tracks[1]) if (tr.on && tr.fresh) Rs[nr++] = &tr;
            if (nl == 2 && nr == 2) {
                double a = pair_gap(*Ls[0], *Rs[0]) + pair_gap(*Ls[1], *Rs[1]), b = pair_gap(*Ls[0], *Rs[1]) + pair_gap(*Ls[1], *Rs[0]);
                if (a <= b) { matches.push_back({Ls[0], Rs[0], pair_gap(*Ls[0], *Rs[0])}); matches.push_back({Ls[1], Rs[1], pair_gap(*Ls[1], *Rs[1])}); }
                else { matches.push_back({Ls[0], Rs[1], pair_gap(*Ls[0], *Rs[1])}); matches.push_back({Ls[1], Rs[0], pair_gap(*Ls[1], *Rs[0])}); }
            } else if (nl && nr) {
                Match best{nullptr, nullptr, 1e9};
                for (int i = 0; i < nl; i++) for (int j = 0; j < nr; j++) { double g = pair_gap(*Ls[i], *Rs[j]); if (g < best.gap) best = {Ls[i], Rs[j], g}; }
                matches.push_back(best);
            }
        }
        std::vector<Hand> out;
        for (auto& m : matches) {
            if (m.gap > 0.04) { logf("hands: stereo pair rejected, palm rays %.0f mm apart\n", m.gap * 1000); continue; }
            Track* L = m.L; Track* R = m.R;
            // Side: both cameras' running handedness together.
            bool right = L->side + R->side > 1.0f;
            int side = right ? 1 : 0;
            if (std::any_of(out.begin(), out.end(), [&](const Hand& x) { return x.right == right; })) side = 1 - side, right = !right;
            Hand hd; hd.right = right;
            double gap_sum = 0; bool ok = true;
            for (int k = 0; k < N_JOINTS && ok; k++) {
                double d1[3], d2[3], p[3], gap;
                rig.cam[0].ray(L->lm[k][0], L->lm[k][1], d1);
                rig.cam[1].ray(R->lm[k][0], R->lm[k][1], d2);
                ok = calib::triangulate(rig.cam[0].origin, d1, rig.cam[1].origin, d2, p, gap);
                gap_sum += gap;
                for (int a = 0; a < 3; a++) hd.joints[k][a] = (float)filters[side][k][a].apply(p[a], t_img);
                for (int a = 0; a < 2; a++) { hd.image[0][k][a] = L->lm[k][a]; hd.image[1][k][a] = R->lm[k][a]; }
            }
            hd.gap_mm = (float)(gap_sum / N_JOINTS * 1000);
            if (!ok) continue;
            float* a = hd.joints[4]; float* b = hd.joints[8];
            hd.pinch_m = std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
            pinch_state[side] = pinch_state[side] ? hd.pinch_m < 0.04f : hd.pinch_m < 0.025f;
            hd.pinching = pinch_state[side];
            out.push_back(hd);
        }
        for (int side = 0; side < 2; side++) if (std::none_of(out.begin(), out.end(), [&](const Hand& x) { return x.right == (side == 1); })) pinch_state[side] = false;

        double t1 = qpc_s();
        frames++;
        if (t1 - rate_t0 > 1) {
            rate = (float)(frames / (t1 - rate_t0)); palm_rate = palms; frames = palms = 0; rate_t0 = t1;
            // Once a second: what each camera follows, and the stereo hands.
            char line[400]; int n = 0;
            for (int c = 0; c < 2; c++) for (int k = 0; k < 2; k++) {
                auto& tr = tracks[c][k];
                if (tr.on) n += snprintf(line + n, sizeof line - n, " cam%d.%d[%s p%.2f m%d]", c, k, tr.right() ? "R" : "L", tr.presence, tr.misses);
            }
            for (auto& hd : out) n += snprintf(line + n, sizeof line - n, " stereo[%s gap %.0fmm pinch %.1fcm%s]", hd.right ? "R" : "L", hd.gap_mm, hd.pinch_m * 100, hd.pinching ? " P" : "");
            logf("hands: %.0f/s%s\n", rate, n ? line : " (none)");
        }
        std::lock_guard<std::mutex> l(g_mu);
        g_result.hands = std::move(out);
        g_result.t = t_img;
        g_result.seq++;
        g_result.rate = rate;
        g_result.palm_runs = palm_rate;
        g_result.ms_palm = (float)(t_palm * 1000);
        g_result.ms_total = (float)((t1 - t0) * 1000);
        g_result.ms_landmarks = g_result.ms_total - g_result.ms_palm;
        g_result.latency_ms = (float)((t1 - t_img) * 1000);
    }
}

inline void start(const std::wstring& model_dir, const calib::Rig& rig) {
    std::wstring dir = model_dir.substr(0, model_dir.find_last_of(L'\\', model_dir.size() - 2) + 1);
    g_log = _wfopen((dir + L"myviture.log").c_str(), L"w");
    g_thread = std::thread(run, model_dir, rig);
}
inline void stop() { g_stop = true; if (g_thread.joinable()) g_thread.join(); }

}  // namespace hands
