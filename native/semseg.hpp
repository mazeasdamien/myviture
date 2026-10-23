// What each pixel of the RGB camera shows: SegFormer-B0 trained on ADE20K (150 classes;
// NVIDIA's weights, research licence; ONNX export by Xenova, models/segformer/), run with
// ONNX Runtime on the GPU (DirectML) a few times a second, its classes folded into the
// room's categories: floor, wall, ceiling, table, door, window, furniture, person, screen.
//
// Each map keeps the head pose and the RGB camera model at its capture, so the room thread
// (roommap.hpp) can look up the category of any 3D point: project it into that image.

#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "third_party/onnxruntime/include/onnxruntime_cxx_api.h"
#include "third_party/onnxruntime/include/dml_provider_factory.h"
#include "rgbcam.hpp"
#include "rgbcal.hpp"

namespace semseg {

enum Cat : uint8_t { NONE = 0, FLOOR, WALL, CEILING, TABLE, DOOR, WINDOW, FURNITURE, PERSON, SCREEN, OTHER, NCAT };
inline const char* CAT_NAME[NCAT] = {"—", "sol", "mur", "plafond", "table", "porte", "fenêtre", "meuble", "personne", "écran", "autre"};
inline const float CAT_COLOR[NCAT][3] = {
    {0.5f, 0.5f, 0.5f}, {0.55f, 0.55f, 0.6f}, {0.25f, 0.55f, 1.0f}, {0.8f, 0.8f, 0.95f}, {1.0f, 0.6f, 0.15f}, {0.95f, 0.25f, 0.85f},
    {0.3f, 0.95f, 0.95f}, {0.6f, 0.4f, 0.2f}, {1.0f, 0.2f, 0.2f}, {0.95f, 0.95f, 0.2f}, {0.35f, 0.6f, 0.35f}};

/** ADE20K class name -> category. Things hung on walls count as wall (their surface is). */
inline Cat cat_of(const std::string& n) {
    auto any = [&](std::initializer_list<const char*> l) { for (auto s : l) if (n == s) return true; return false; };
    if (any({"floor", "rug"})) return FLOOR;
    if (any({"wall", "building", "column", "painting", "poster", "bulletin board", "mirror", "signboard", "clock", "sconce", "radiator"})) return WALL;
    if (any({"ceiling"})) return CEILING;
    if (any({"table", "desk", "coffee table", "counter", "countertop", "kitchen island", "pool table", "buffet", "bar"})) return TABLE;
    if (any({"door", "screen door"})) return DOOR;
    if (any({"windowpane", "curtain", "blind"})) return WINDOW;
    if (any({"cabinet", "bed", "chair", "sofa", "shelf", "armchair", "seat", "wardrobe", "chest of drawers", "bookcase", "bench", "stool",
             "swivel chair", "ottoman", "case", "box", "bathtub", "sink", "refrigerator", "stove", "oven", "washer", "fireplace", "toilet",
             "dishwasher", "microwave", "cushion", "pillow", "lamp", "plant", "basket", "ashcan", "fan", "stairs", "stairway", "step"})) return FURNITURE;
    if (any({"person"})) return PERSON;
    if (any({"computer", "monitor", "crt screen", "screen", "television receiver"})) return SCREEN;
    return OTHER;
}

constexpr int IN_PX = 512, OUT_PX = 128;  // model input, output (the logits are a quarter size)

/** A labelled image: OUT_PX x OUT_PX categories over the whole RGB frame, with the head and camera then. */
struct Map {
    std::vector<uint8_t> cat;
    double t = 0;                  // capture (QPC s)
    float head[16] = {};           // head -> tracking frame then, row-major
    rgbcal::Cam cam;
    uint32_t seq = 0;
    /** Category of a tracking-frame point, NONE if outside the image. */
    Cat at(const float p[3]) const {
        if (cat.empty() || cam.w <= 0) return NONE;
        // world -> head (the inverse of a rigid head pose)
        float d[3] = {p[0] - head[3], p[1] - head[7], p[2] - head[11]};
        double h[3] = {head[0] * d[0] + head[4] * d[1] + head[8] * d[2], head[1] * d[0] + head[5] * d[1] + head[9] * d[2], head[2] * d[0] + head[6] * d[1] + head[10] * d[2]};
        double u, v;
        if (!rgbcal::project(cam, h, u, v)) return NONE;
        int x = (int)(u * OUT_PX / cam.w), y = (int)(v * OUT_PX / cam.h);
        if (x < 0 || y < 0 || x >= OUT_PX || y >= OUT_PX) return NONE;
        return (Cat)cat[y * OUT_PX + x];
    }
};

using PoseFn = std::function<bool(double, float*)>;

inline std::mutex g_mu;
inline Map g_map;
inline std::string g_status = "démarrage…";
inline float g_ms = 0, g_hz = 0;
inline std::atomic<bool> g_stop{false}, g_on{true};
inline bool g_use_gpu = false;  // --seg-gpu: DirectML instead of the CPU
inline std::atomic<float> g_latency_s{0.05f};  // the RGB frame's age on arrival (the app's setting)
inline std::thread g_thread;

inline void set_status(const std::string& s) { std::lock_guard<std::mutex> l(g_mu); g_status = s; }
inline double now_s() { LARGE_INTEGER f, q; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&q); return (double)q.QuadPart / f.QuadPart; }

inline void run(std::wstring dir, PoseFn pose_at) {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    // Labels -> categories.
    std::vector<uint8_t> lut;
    {
        std::string d8(dir.begin(), dir.end());
        FILE* f = fopen((d8 + "models\\segformer\\labels.txt").c_str(), "r");
        if (!f) { set_status("pas de models/segformer/labels.txt"); return; }
        char line[128];
        while (fgets(line, sizeof line, f)) { std::string s = line; while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back(); lut.push_back(cat_of(s)); }
        fclose(f);
    }
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "semseg"};
    Ort::Session sess{nullptr};
    std::string device = "GPU (DirectML)";
    // On the CPU by default: on the GPU (DirectML, ~185 ms a map) it shares the integrated
    // GPU with the glasses and costs them 10 to 20 frames a second.
    for (int attempt = 0; attempt < 2 && !sess; attempt++) {
        bool gpu = g_use_gpu ? attempt == 0 : attempt == 1;
        try {
            Ort::SessionOptions so;
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            if (gpu) {
                so.DisableMemPattern();
                so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(so, 0));
                device = "GPU (DirectML)";
            } else { so.SetIntraOpNumThreads(4); device = "CPU (4 cœurs)"; }
            sess = Ort::Session(env, (dir + L"models\\segformer\\model.onnx").c_str(), so);
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "semseg: %s session failed: %s\n", attempt ? "CPU" : "DirectML", e.what());
        }
    }
    if (!sess) { set_status("modèle introuvable ou illisible"); return; }
    printf("semseg: SegFormer-B0 ADE20K on %s, %zu classes\n", device.c_str(), lut.size());
    set_status(device);
    std::vector<float> in((size_t)3 * IN_PX * IN_PX);
    std::vector<uint32_t> frame;
    uint32_t seen = 0, maps = 0;
    double rate_t0 = now_s();
    int runs = 0;
    const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* in_names[] = {"pixel_values"};
    const char* out_names[] = {"logits"};
    while (!g_stop) {
        if (!g_on) { Sleep(100); continue; }
        int w = 0, h = 0; double t_arr = 0;
        {
            std::lock_guard<std::mutex> l(rgbcam::g_frame.mu);
            auto& f = rgbcam::g_frame;
            if (f.seq != seen && f.w > 0 && f.bgra.size() == (size_t)f.w * f.h) { seen = f.seq; frame = f.bgra; w = f.w; h = f.h; t_arr = f.t; }
        }
        if (!w) { Sleep(20); continue; }
        rgbcal::Cam cam = rgbcal::get();
        if (cam.w != w || cam.h != h) { cam.w = w; cam.h = h; if (cam.f <= 0) cam.f = rgbcal::default_focal(w); }
        double t_cap = t_arr - g_latency_s;
        float head[16];
        if (!pose_at(t_cap, head)) { Sleep(20); continue; }
        double t0 = now_s();
        // Resize (box average over each target pixel's footprint), RGB, normalised, CHW.
        for (int y = 0; y < IN_PX; y++) {
            int y0 = y * h / IN_PX, y1 = std::max(y0 + 1, (y + 1) * h / IN_PX);
            for (int x = 0; x < IN_PX; x++) {
                int x0 = x * w / IN_PX, x1 = std::max(x0 + 1, (x + 1) * w / IN_PX);
                float s[3] = {0, 0, 0}; int n = 0;
                for (int yy = y0; yy < y1; yy += 2) for (int xx = x0; xx < x1; xx += 2) {
                    uint32_t p = frame[(size_t)yy * w + xx];
                    s[0] += (p >> 16) & 255; s[1] += (p >> 8) & 255; s[2] += p & 255; n++;
                }
                for (int c = 0; c < 3; c++) in[(size_t)c * IN_PX * IN_PX + y * IN_PX + x] = (s[c] / n / 255.0f - mean[c]) / stdv[c];
            }
        }
        int64_t shape[4] = {1, 3, IN_PX, IN_PX};
        Ort::Value t = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(), shape, 4);
        std::vector<Ort::Value> out;
        try { out = sess.Run(Ort::RunOptions{nullptr}, in_names, &t, 1, out_names, 1); }
        catch (const Ort::Exception& e) { set_status(std::string("erreur : ") + e.what()); Sleep(500); continue; }
        auto info = out[0].GetTensorTypeAndShapeInfo();
        auto sh = info.GetShape();
        if (sh.size() != 4 || sh[2] != OUT_PX || sh[3] != OUT_PX) { set_status("sortie inattendue"); Sleep(500); continue; }
        int nc = (int)sh[1];
        const float* lg = out[0].GetTensorData<float>();
        Map m;
        m.cat.resize(OUT_PX * OUT_PX);
        for (int i = 0; i < OUT_PX * OUT_PX; i++) {
            int best = 0; float bv = lg[i];
            for (int c = 1; c < nc; c++) { float v = lg[(size_t)c * OUT_PX * OUT_PX + i]; if (v > bv) { bv = v; best = c; } }
            m.cat[i] = best < (int)lut.size() ? lut[best] : OTHER;
        }
        m.t = t_cap; memcpy(m.head, head, sizeof head); m.cam = cam; m.seq = ++maps;
        double ms = (now_s() - t0) * 1000;
        runs++;
        double now = now_s();
        {
            std::lock_guard<std::mutex> l(g_mu);
            g_map = std::move(m);
            g_ms = (float)ms;
            if (now - rate_t0 > 2) { g_hz = (float)(runs / (now - rate_t0)); runs = 0; rate_t0 = now; }
        }
        // Two maps a second at most: enough for a room, and the CPU has the hands and the depth too.
        double rest = 0.5 - (now_s() - t0);
        if (rest > 0.005) Sleep((DWORD)(rest * 1000));
    }
}

inline void start(const std::wstring& dir, PoseFn pose_at) { g_thread = std::thread(run, dir, pose_at); }
inline void stop() { g_stop = true; if (g_thread.joinable()) g_thread.join(); }

}  // namespace semseg
