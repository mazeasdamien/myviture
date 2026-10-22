// The RGB camera's place on the glasses, found from the hands without any action from
// the wearer: the stereo cameras give each joint in 3D (head frame), the hand model run
// on the RGB image gives where that joint appears; a camera (rotation, position, focal)
// is fitted to the pairs, with priors keeping it near the factory-plausible default.
//
// Camera model: p_cam = B * R(rv) * (p_head - c), with B taking the head's OpenGL axes
// (x right, y up, z back) to the camera's (x right, y down, z forward), then a pinhole
// with focal f and the principal point at the image centre.

#pragma once
#include <cmath>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace rgbcal {

struct Cam {
    int w = 0, h = 0;
    double f = 0;
    double rv[3] = {0, 0, 0};  // rotation vector (radians)
    double c[3] = {0, 0, 0};   // camera centre in the head frame (m)
    bool fitted = false;
};

inline double default_focal(int w) { return (w / 2.0) / std::tan(78.0 / 2 * 3.14159265 / 180); }  // 78 deg horizontal field

/** Rodrigues: rotation vector -> row-major 3x3. */
inline void rotation(const double rv[3], double R[9]) {
    double a = std::sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
    double kx = 0, ky = 0, kz = 0;
    if (a > 1e-12) { kx = rv[0] / a; ky = rv[1] / a; kz = rv[2] / a; }
    double ca = std::cos(a), sa = std::sin(a), v = 1 - ca;
    double M[9] = {ca + kx * kx * v, kx * ky * v - kz * sa, kx * kz * v + ky * sa,
                   ky * kx * v + kz * sa, ca + ky * ky * v, ky * kz * v - kx * sa,
                   kz * kx * v - ky * sa, kz * ky * v + kx * sa, ca + kz * kz * v};
    for (int i = 0; i < 9; i++) R[i] = M[i];
}

/** Head-frame point -> pixel; false if behind the camera. */
inline bool project(const Cam& cam, const double p[3], double& u, double& v) {
    double R[9];
    rotation(cam.rv, R);
    double d[3] = {p[0] - cam.c[0], p[1] - cam.c[1], p[2] - cam.c[2]}, r[3];
    for (int i = 0; i < 3; i++) r[i] = R[i * 3] * d[0] + R[i * 3 + 1] * d[1] + R[i * 3 + 2] * d[2];
    double x = r[0], y = -r[1], z = -r[2];  // B
    if (z < 0.05) return false;
    u = cam.f * x / z + cam.w / 2.0;
    v = cam.f * y / z + cam.h / 2.0;
    return true;
}

struct Pair { double p[3]; double u, v; };

struct Fit {
    double rms_px = 0, rot_deg = 0, shift_cm[3] = {}, f_rel = 1;
    int n = 0;
    bool ok = false;
};

/** Levenberg-Marquardt over (rv, c, f) with priors; false if the result is implausible. */
inline bool fit(const std::vector<Pair>& P, Cam& cam, Fit& info) {
    if (P.size() < 40) return false;
    const double f0 = default_focal(cam.w);
    const double sig_px = 3, sig_rot = 6 * 3.14159265 / 180, sig_t = 0.04, sig_f = 0.15;
    const double k = 8 * std::sqrt(P.size() / 20.0);  // prior weight grows with the data, as the web fit
    double x[7] = {0, 0, 0, 0, 0, 0, std::log(f0)};  // start from the default, not the current fit
    std::vector<char> use(P.size(), 1);
    auto residuals = [&](const double* q, std::vector<double>& r) {
        Cam t = cam;
        for (int i = 0; i < 3; i++) { t.rv[i] = q[i]; t.c[i] = q[3 + i]; }
        t.f = std::exp(q[6]);
        r.clear();
        for (size_t i = 0; i < P.size(); i++) {
            if (!use[i]) continue;
            double u, v;
            if (!project(t, P[i].p, u, v)) { r.push_back(100); r.push_back(100); continue; }
            r.push_back((u - P[i].u) / sig_px);
            r.push_back((v - P[i].v) / sig_px);
        }
        for (int i = 0; i < 3; i++) r.push_back(k * q[i] / sig_rot);
        for (int i = 0; i < 3; i++) r.push_back(k * q[3 + i] / sig_t);
        r.push_back(k * (q[6] - std::log(f0)) / sig_f);
    };
    auto cost = [&](const std::vector<double>& r) { double s = 0; for (double v : r) s += v * v; return s; };
    for (int pass = 0; pass < 2; pass++) {
        std::vector<double> r, r2, J[7];
        residuals(x, r);
        double lambda = 1e-2, c0 = cost(r);
        for (int it = 0; it < 30; it++) {
            for (int j = 0; j < 7; j++) {
                double q[7]; for (int i = 0; i < 7; i++) q[i] = x[i];
                double h = j < 3 ? 1e-4 : j < 6 ? 1e-4 : 1e-3;
                q[j] += h;
                residuals(q, J[j]);
                for (size_t i = 0; i < r.size(); i++) J[j][i] = (J[j][i] - r[i]) / h;
            }
            double A[7][7] = {}, b[7] = {};
            for (int a = 0; a < 7; a++) {
                for (size_t i = 0; i < r.size(); i++) b[a] -= J[a][i] * r[i];
                for (int c = 0; c < 7; c++) for (size_t i = 0; i < r.size(); i++) A[a][c] += J[a][i] * J[c][i];
            }
            for (int a = 0; a < 7; a++) A[a][a] *= 1 + lambda;
            // Solve A dx = b (Gauss with pivoting).
            double M[7][8];
            for (int a = 0; a < 7; a++) { for (int c = 0; c < 7; c++) M[a][c] = A[a][c]; M[a][7] = b[a]; }
            bool singular = false;
            for (int c = 0; c < 7 && !singular; c++) {
                int piv = c;
                for (int a = c + 1; a < 7; a++) if (std::fabs(M[a][c]) > std::fabs(M[piv][c])) piv = a;
                for (int j = 0; j < 8; j++) std::swap(M[c][j], M[piv][j]);
                if (std::fabs(M[c][c]) < 1e-12) { singular = true; break; }
                for (int a = 0; a < 7; a++) if (a != c) { double f = M[a][c] / M[c][c]; for (int j = 0; j < 8; j++) M[a][j] -= f * M[c][j]; }
            }
            if (singular) break;
            double q[7];
            for (int a = 0; a < 7; a++) q[a] = x[a] + M[a][7] / M[a][a];
            residuals(q, r2);
            double c1 = cost(r2);
            if (c1 < c0) { for (int a = 0; a < 7; a++) x[a] = q[a]; r = r2; lambda = std::max(1e-6, lambda / 3); if (c0 - c1 < 1e-6 * c0) break; c0 = c1; }
            else lambda = std::min(1e6, lambda * 4);
        }
        if (pass == 0) {
            // Drop the pairs far off (a hand mismatched between the stereo and RGB), refit.
            Cam t = cam;
            for (int i = 0; i < 3; i++) { t.rv[i] = x[i]; t.c[i] = x[3 + i]; }
            t.f = std::exp(x[6]);
            std::vector<double> e;
            for (size_t i = 0; i < P.size(); i++) { double u, v; e.push_back(project(t, P[i].p, u, v) ? std::hypot(u - P[i].u, v - P[i].v) : 1e9); }
            std::vector<double> s = e;
            std::sort(s.begin(), s.end());
            double med = s[s.size() / 2];
            for (size_t i = 0; i < P.size(); i++) use[i] = e[i] < std::max(15.0, 3 * med);
        }
    }
    Cam t = cam;
    for (int i = 0; i < 3; i++) { t.rv[i] = x[i]; t.c[i] = x[3 + i]; }
    t.f = std::exp(x[6]);
    double e2 = 0; int n = 0;
    for (size_t i = 0; i < P.size(); i++) {
        if (!use[i]) continue;
        double u, v;
        if (project(t, P[i].p, u, v)) { e2 += (u - P[i].u) * (u - P[i].u) + (v - P[i].v) * (v - P[i].v); n++; }
    }
    info.n = n;
    info.rms_px = n ? std::sqrt(e2 / n) : 1e9;
    info.rot_deg = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) * 180 / 3.14159265;
    for (int i = 0; i < 3; i++) info.shift_cm[i] = x[3 + i] * 100;
    info.f_rel = t.f / f0;
    info.ok = n >= 100 && info.rms_px < 20 && info.rot_deg < 25 && std::fabs(x[3]) < 0.12 && std::fabs(x[4]) < 0.12 && std::fabs(x[5]) < 0.12 && info.f_rel > 0.6 && info.f_rel < 1.6;
    if (info.ok) { cam = t; cam.fitted = true; }
    return info.ok;
}

inline bool save(const std::string& path, const Cam& c) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;
    fprintf(f, "%d %d %.6f %.9f %.9f %.9f %.9f %.9f %.9f\n", c.w, c.h, c.f, c.rv[0], c.rv[1], c.rv[2], c.c[0], c.c[1], c.c[2]);
    fclose(f);
    return true;
}

inline bool load(const std::string& path, Cam& c) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    int n = fscanf(f, "%d %d %lf %lf %lf %lf %lf %lf %lf", &c.w, &c.h, &c.f, &c.rv[0], &c.rv[1], &c.rv[2], &c.c[0], &c.c[1], &c.c[2]);
    fclose(f);
    c.fitted = n == 9;
    return c.fitted;
}

// ------------------------------------------------------------------ shared state

inline std::mutex g_mu;
inline Cam g_cam;              // current model (default until fitted)
inline std::vector<Pair> g_pairs;
inline Fit g_fit;
inline int g_since_fit = 0;   // frames since the last fit
inline int g_frames = 0;      // frames of pairs collected
inline double g_lo[3] = {1e9, 1e9, 1e9}, g_hi[3] = {-1e9, -1e9, -1e9};  // wrist positions' extent (m)
inline std::string g_path;

/** The current model as it is (w = 0 before any RGB frame). */
inline Cam get() { std::lock_guard<std::mutex> l(g_mu); return g_cam; }

/** The current model for an image of this size (default when none yet). */
inline Cam current(int w, int h) {
    std::lock_guard<std::mutex> l(g_mu);
    if (g_cam.w != w || g_cam.h != h) { g_cam = Cam(); g_cam.w = w; g_cam.h = h; g_cam.f = default_focal(w); }
    return g_cam;
}

/**
 * One frame's stereo <-> RGB pairs (wrist: that hand's wrist, head frame). A fit needs
 * hands seen at several places (8 frames, 25 cm of spread) or focal and position trade
 * off; then every 4 frames. The last 600 points are kept.
 */
inline void add(const std::vector<Pair>& pairs, const double wrist[3]) {
    std::lock_guard<std::mutex> l(g_mu);
    g_pairs.insert(g_pairs.end(), pairs.begin(), pairs.end());
    if (g_pairs.size() > 600) g_pairs.erase(g_pairs.begin(), g_pairs.begin() + (g_pairs.size() - 600));
    g_frames++; g_since_fit++;
    for (int i = 0; i < 3; i++) { g_lo[i] = std::min(g_lo[i], wrist[i]); g_hi[i] = std::max(g_hi[i], wrist[i]); }
    double spread = std::max(g_hi[0] - g_lo[0], std::max(g_hi[1] - g_lo[1], g_hi[2] - g_lo[2]));
    // Enough spread to separate the focal from the position; with many frames, a little less.
    if (((g_frames >= 8 && spread >= 0.12) || (g_frames >= 30 && spread >= 0.06)) && g_since_fit >= 4) {
        g_since_fit = 0;
        Fit info;
        Cam trial = g_cam;
        if (fit(g_pairs, trial, info)) { g_cam = trial; if (!g_path.empty()) save(g_path, g_cam); }
        g_fit = info;
    }
}

inline void reset() {
    std::lock_guard<std::mutex> l(g_mu);
    g_pairs.clear(); g_since_fit = 0; g_frames = 0; g_fit = Fit();
    for (int i = 0; i < 3; i++) { g_lo[i] = 1e9; g_hi[i] = -1e9; }
    int w = g_cam.w, h = g_cam.h;
    g_cam = Cam(); g_cam.w = w; g_cam.h = h; g_cam.f = default_focal(w);
    if (!g_path.empty()) remove(g_path.c_str());
}

}  // namespace rgbcal
