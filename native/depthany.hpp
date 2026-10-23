// Dense depth from the RGB camera: Depth Anything V2 Small (Apache 2.0; ONNX int8 by
// onnx-community, models/depthanything/) on the CPU, about once a second at 392 x 224.
// It gives an affine-invariant inverse depth; the stereo's points of the same moment
// (room::g.dbg_live, projected into the RGB image) fix it in metres: 1/z = a d + b, least
// squares, twice, without the worst 20 %. The result, a grid of world points, goes to the
// room thread (dense.hpp), where it shows the floor and the walls the stereo cannot see.

#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <string>
#include <cmath>
#include <cstdio>
#include <functional>
#include <thread>
#include <vector>

#include "third_party/onnxruntime/include/onnxruntime_cxx_api.h"
#include "dense.hpp"
#include "rgbcal.hpp"
#include "rgbcam.hpp"
#include "roommap.hpp"

namespace dany {

constexpr int W = 392, H = 224;   // multiples of 14 (the ViT's patches), 16:9-ish
constexpr int STEP = 2;           // output grid: every 2nd pixel (196 x 112 points)

inline std::atomic<bool> g_stop{false};
inline std::atomic<float> g_latency_s{0.05f};
inline std::thread g_thread;
inline double now_s() { LARGE_INTEGER f, q; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&q); return (double)q.QuadPart / f.QuadPart; }
inline void set_status(const std::string& s) { std::lock_guard<std::mutex> l(dense::g_mu); dense::g_status = s; }

using PoseFn = std::function<bool(double, float*)>;

inline void run(std::wstring dir, PoseFn pose_at) {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "depthany"};
    Ort::Session sess{nullptr};
    try {
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        so.SetIntraOpNumThreads(3);
        sess = Ort::Session(env, (dir + L"models\\depthanything\\model.onnx").c_str(), so);
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "depthany: %s\n", e.what());
        set_status("modèle introuvable ou illisible");
        return;
    }
    printf("depthany: Depth Anything V2 Small on the CPU, %dx%d\n", W, H);
    set_status("CPU (3 cœurs)");
    std::vector<float> in((size_t)3 * W * H);
    std::vector<uint32_t> frame;
    uint32_t seen = 0, made = 0;
    double rate_t0 = now_s();
    int runs = 0;
    const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* in_names[] = {"pixel_values"};
    const char* out_names[] = {"predicted_depth"};
    while (!g_stop) {
        int fw = 0, fh = 0; double t_arr = 0;
        {
            std::lock_guard<std::mutex> l(rgbcam::g_frame.mu);
            auto& f = rgbcam::g_frame;
            if (f.seq != seen && f.w > 0 && f.bgra.size() == (size_t)f.w * f.h) { seen = f.seq; frame = f.bgra; fw = f.w; fh = f.h; t_arr = f.t; }
        }
        if (!fw) { Sleep(30); continue; }
        rgbcal::Cam cam = rgbcal::get();
        if (cam.w != fw || cam.h != fh) { cam.w = fw; cam.h = fh; if (cam.f <= 0) cam.f = rgbcal::default_focal(fw); }
        double t_cap = t_arr - g_latency_s;
        float head[16];
        if (!pose_at(t_cap, head)) { Sleep(30); continue; }
        // The stereo's newest points (world), to fix the scale: taken now, the scene holds still.
        std::vector<float> live;
        { std::lock_guard<std::mutex> l(room::g.mu); live = room::g.dbg_live; }
        double t0 = now_s();
        for (int y = 0; y < H; y++) {
            int y0 = y * fh / H, y1 = std::max(y0 + 1, (y + 1) * fh / H);
            for (int x = 0; x < W; x++) {
                int x0 = x * fw / W, x1 = std::max(x0 + 1, (x + 1) * fw / W);
                float s[3] = {0, 0, 0}; int n = 0;
                for (int yy = y0; yy < y1; yy += 2) for (int xx = x0; xx < x1; xx += 2) {
                    uint32_t p = frame[(size_t)yy * fw + xx];
                    s[0] += (p >> 16) & 255; s[1] += (p >> 8) & 255; s[2] += p & 255; n++;
                }
                for (int c = 0; c < 3; c++) in[(size_t)c * W * H + y * W + x] = (s[c] / n / 255.0f - mean[c]) / stdv[c];
            }
        }
        int64_t shape[4] = {1, 3, H, W};
        Ort::Value tin = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(), shape, 4);
        std::vector<Ort::Value> out;
        try { out = sess.Run(Ort::RunOptions{nullptr}, in_names, &tin, 1, out_names, 1); }
        catch (const Ort::Exception& e) { set_status(std::string("erreur : ") + e.what()); Sleep(1000); continue; }
        auto sh = out[0].GetTensorTypeAndShapeInfo().GetShape();
        if (sh.size() != 3 || sh[1] != H || sh[2] != W) { set_status("sortie inattendue"); Sleep(1000); continue; }
        const float* disp = out[0].GetTensorData<float>();
        double ms = (now_s() - t0) * 1000;
        // Head-frame point -> RGB pixel and camera depth (rgbcal::project's model).
        double R[9];
        rgbcal::rotation(cam.rv, R);
        auto to_cam = [&](const float* pw, double& u, double& v, double& z) {
            float d[3] = {pw[0] - head[3], pw[1] - head[7], pw[2] - head[11]};
            double hp[3] = {head[0] * d[0] + head[4] * d[1] + head[8] * d[2], head[1] * d[0] + head[5] * d[1] + head[9] * d[2], head[2] * d[0] + head[6] * d[1] + head[10] * d[2]};
            double q[3] = {hp[0] - cam.c[0], hp[1] - cam.c[1], hp[2] - cam.c[2]}, r[3];
            for (int i = 0; i < 3; i++) r[i] = R[i * 3] * q[0] + R[i * 3 + 1] * q[1] + R[i * 3 + 2] * q[2];
            z = -r[2];
            if (z < 0.2) return false;
            double xn = r[0] / z, yn = -r[1] / z, k = 1 + cam.k1 * (xn * xn + yn * yn);
            u = cam.f * xn * k + cam.w / 2.0; v = cam.f * yn * k + cam.h / 2.0;
            return u >= 0 && v >= 0 && u < cam.w && v < cam.h;
        };
        // The scale: pairs (disparity, 1/z), robust least squares.
        std::vector<std::array<double, 2>> pr;
        for (size_t i = 0; i + 5 < live.size(); i += 6) {
            double u, v, z;
            if (!to_cam(&live[i], u, v, z) || z > 5) continue;
            int x = (int)(u * W / cam.w), y = (int)(v * H / cam.h);
            if (x < 0 || y < 0 || x >= W || y >= H) continue;
            pr.push_back({disp[y * W + x], 1.0 / z});
        }
        double a = 0, b = 0, rms = 0;
        bool ok = false;
        std::vector<std::array<double, 2>> use = pr;
        for (int pass = 0; pass < 3 && use.size() >= 60; pass++) {
            double sx = 0, sy = 0, sxx = 0, sxy = 0, n = (double)use.size();
            for (auto& p : use) { sx += p[0]; sy += p[1]; sxx += p[0] * p[0]; sxy += p[0] * p[1]; }
            double den = n * sxx - sx * sx;
            if (std::fabs(den) < 1e-12) break;
            a = (n * sxy - sx * sy) / den; b = (sy - a * sx) / n;
            std::vector<std::pair<double, size_t>> res;
            for (size_t i = 0; i < use.size(); i++) res.push_back({std::fabs(a * use[i][0] + b - use[i][1]), i});
            std::sort(res.begin(), res.end());
            double e = 0; for (size_t i = 0; i < res.size() * 8 / 10; i++) e += res[i].first * res[i].first;
            rms = std::sqrt(e / std::max<size_t>(1, res.size() * 8 / 10));
            ok = a > 0;
            std::vector<std::array<double, 2>> keep;
            for (size_t i = 0; i < res.size() * 8 / 10; i++) keep.push_back(use[res[i].second]);
            use.swap(keep);
        }
        runs++;
        double now = now_s();
        {
            std::lock_guard<std::mutex> l(dense::g_mu);
            dense::g_ms = (float)ms; dense::g_pairs = (int)pr.size(); dense::g_scale_rms = (float)rms;
            if (now - rate_t0 > 3) { dense::g_hz = (float)(runs / (now - rate_t0)); runs = 0; rate_t0 = now; }
            dense::g_status = ok ? "CPU (3 cœurs)" : pr.size() < 60 ? "pas assez de points stéréo dans l'image pour l'échelle" : "échelle incohérente";
        }
        if (!ok) { Sleep(300); continue; }
        // Metric depth -> world points, every STEP pixels.
        dense::Frame df;
        df.w = W / STEP; df.h = H / STEP;
        df.pts.assign((size_t)df.w * df.h * 3, NAN);
        float cw[3];
        for (int i = 0; i < 3; i++) cw[i] = (float)(head[i * 4] * cam.c[0] + head[i * 4 + 1] * cam.c[1] + head[i * 4 + 2] * cam.c[2]) + head[i * 4 + 3];
        memcpy(df.eye, cw, sizeof cw);
        for (int gy = 0; gy < df.h; gy++) for (int gx = 0; gx < df.w; gx++) {
            int x = gx * STEP, y = gy * STEP;
            double inv = a * disp[y * W + x] + b;
            if (inv <= 1.0 / 6.0) continue;  // farther than 6 m (or behind): not trusted
            double z = 1.0 / inv;
            if (z < 0.3) continue;
            double xn = ((x + 0.5) * cam.w / W - cam.w / 2.0) / cam.f, yn = ((y + 0.5) * cam.h / H - cam.h / 2.0) / cam.f;
            double r[3] = {xn * z, -yn * z, -z};  // camera frame (r = R (p - c))
            double hp[3];
            for (int i = 0; i < 3; i++) hp[i] = R[i] * r[0] + R[3 + i] * r[1] + R[6 + i] * r[2] + cam.c[i];
            float* o = &df.pts[((size_t)gy * df.w + gx) * 3];
            for (int i = 0; i < 3; i++) o[i] = (float)(head[i * 4] * hp[0] + head[i * 4 + 1] * hp[1] + head[i * 4 + 2] * hp[2]) + head[i * 4 + 3];
        }
        df.t = t_cap; df.seq = ++made;
        { std::lock_guard<std::mutex> l(dense::g_mu); dense::g_frame = std::move(df); }
        double rest = 1.0 - (now_s() - t0);  // once a second: the CPU also runs the segmentation, the hands, the depth
        if (rest > 0.005) Sleep((DWORD)(rest * 1000));
    }
}

inline void start(const std::wstring& dir, PoseFn pose_at) { g_thread = std::thread(run, dir, pose_at); }
inline void stop() { g_stop = true; if (g_thread.joinable()) g_thread.join(); }

}  // namespace dany
