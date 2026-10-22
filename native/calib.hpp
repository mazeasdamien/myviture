// The glasses' factory calibration (calibration.bin, a protobuf without published
// schema; field numbers as in src/device/calibration.ts) and the tracking cameras'
// geometry: FOV-model undistortion, camera -> head (OpenGL axes) transform, rays.

#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace calib {

struct Camera {
    double fx = 0, fy = 0, cx = 0, cy = 0, w = 0;  // pinhole + FOV distortion
    double R[9] = {};                               // camera -> head rotation, row-major
    double origin[3] = {};                          // camera centre in the head frame (m)

    /** Pixel -> undistorted normalised coordinates (z = 1). */
    void undistort(double u, double v, double& x, double& y) const {
        double xd = (u - cx) / fx, yd = (v - cy) / fy, rd = std::hypot(xd, yd);
        if (w < 1e-6 || rd < 1e-9) { x = xd; y = yd; return; }
        double ang = std::min(rd * w, 3.14159265 / 2 - 1e-3);
        double s = std::tan(ang) / (2 * std::tan(w / 2)) / rd;
        x = xd * s; y = yd * s;
    }
    /** Undistorted normalised coordinates -> pixel (FOV model). */
    void distort(double x, double y, double& u, double& v) const {
        double ru = std::hypot(x, y);
        double s = w < 1e-6 || ru < 1e-9 ? 1 : std::atan(ru * 2 * std::tan(w / 2)) / w / ru;
        u = fx * x * s + cx; v = fy * y * s + cy;
    }
    /** Head-frame point -> pixel; false if behind the camera. */
    bool project(const double p[3], double& u, double& v) const {
        double d[3] = {p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]}, c[3];
        for (int i = 0; i < 3; i++) c[i] = R[i] * d[0] + R[3 + i] * d[1] + R[6 + i] * d[2];  // R^T
        if (c[2] < 0.05) return false;
        distort(c[0] / c[2], c[1] / c[2], u, v);
        return true;
    }
    /** Camera-frame point (x right, y down, z forward) -> head frame. */
    void to_head(const double c[3], double p[3]) const {
        for (int i = 0; i < 3; i++) p[i] = R[i * 3] * c[0] + R[i * 3 + 1] * c[1] + R[i * 3 + 2] * c[2] + origin[i];
    }
    /** Unit ray in the head frame through pixel (u, v). */
    void ray(double u, double v, double d[3]) const {
        double x, y;
        undistort(u, v, x, y);
        for (int i = 0; i < 3; i++) d[i] = R[i * 3] * x + R[i * 3 + 1] * y + R[i * 3 + 2];
        double n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        for (int i = 0; i < 3; i++) d[i] /= n;
    }
};

struct Rig {
    Camera cam[2];
    bool ok = false;
};

namespace detail {
struct Reader {
    const uint8_t* p; size_t n, i = 0;
    uint64_t varint() { uint64_t r = 0; int s = 0; while (i < n) { uint8_t x = p[i++]; r |= (uint64_t)(x & 0x7f) << s; s += 7; if (x < 0x80) break; } return r; }
};
struct Field { int no, wire; uint64_t v; double d; const uint8_t* b; size_t len; };
inline std::vector<Field> fields(const uint8_t* p, size_t n) {
    std::vector<Field> out;
    Reader r{p, n};
    while (r.i < n) {
        uint64_t key = r.varint();
        Field f{(int)(key >> 3), (int)(key & 7), 0, 0, nullptr, 0};
        if (f.wire == 0) f.v = r.varint();
        else if (f.wire == 1) { memcpy(&f.d, p + r.i, 8); r.i += 8; }
        else if (f.wire == 5) { float x; memcpy(&x, p + r.i, 4); f.d = x; r.i += 4; }
        else if (f.wire == 2) { f.len = (size_t)r.varint(); f.b = p + r.i; r.i += f.len; }
        else break;
        out.push_back(f);
    }
    return out;
}
inline std::vector<double> doubles(const Field& f) { std::vector<double> v(f.len / 8); memcpy(v.data(), f.b, v.size() * 8); return v; }
inline std::vector<float> floats(const Field& f) { std::vector<float> v(f.len / 4); memcpy(v.data(), f.b, v.size() * 4); return v; }
}  // namespace detail

/** Reads calibration.bin; cameras 0 and 1 are the left and right tracking cameras. */
inline Rig load(const std::string& path) {
    using namespace detail;
    Rig rig;
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> blob((std::istreambuf_iterator<char>(f)), {});
    if (blob.empty()) return rig;
    double Rgl[9] = {-1, 0, 0, 0, 0, 1, 0, 1, 0};  // IMU -> OpenGL head, default Luma Ultra mounting
    struct Raw { std::vector<double> T, K, D; };
    std::vector<Raw> cams;
    for (auto& fl : fields(blob.data(), blob.size())) {
        if (fl.no == 80 && fl.wire == 2) {
            for (auto& g : fields(fl.b, fl.len)) if (g.no == 1 && g.wire == 2) {
                auto m = floats(g);
                if (m.size() == 16) { double r[9] = {m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]}; memcpy(Rgl, r, sizeof r); }
            }
        } else if (fl.no == 85 && fl.wire == 2) {
            Raw c;
            for (auto& g : fields(fl.b, fl.len)) {
                if (g.wire != 2) continue;
                if (g.no == 1) c.T = doubles(g);
                else if (g.no == 5) c.D = doubles(g);
                else if (g.no == 7) c.K = doubles(g);
            }
            cams.push_back(c);
        }
    }
    if (cams.size() < 2) return rig;
    for (int k = 0; k < 2; k++) {
        auto& c = cams[k];
        if (c.K.size() < 4 || c.T.size() < 16) return rig;
        Camera& cam = rig.cam[k];
        cam.fx = c.K[0]; cam.fy = c.K[1]; cam.cx = c.K[2]; cam.cy = c.K[3];
        cam.w = c.D.empty() ? 0 : c.D[0];
        // T_cam_imu: p_cam = Rci p_imu + t  =>  p_imu = Rci^T (p_cam - t); head = Rgl p_imu.
        double Rci[9] = {c.T[0], c.T[1], c.T[2], c.T[4], c.T[5], c.T[6], c.T[8], c.T[9], c.T[10]};
        double t[3] = {c.T[3], c.T[7], c.T[11]};
        double Ric[9];
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Ric[i * 3 + j] = Rci[j * 3 + i];
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            double s = 0;
            for (int m = 0; m < 3; m++) s += Rgl[i * 3 + m] * Ric[m * 3 + j];
            cam.R[i * 3 + j] = s;
        }
        double o[3];
        for (int i = 0; i < 3; i++) o[i] = -(Ric[i * 3] * t[0] + Ric[i * 3 + 1] * t[1] + Ric[i * 3 + 2] * t[2]);
        for (int i = 0; i < 3; i++) cam.origin[i] = Rgl[i * 3] * o[0] + Rgl[i * 3 + 1] * o[1] + Rgl[i * 3 + 2] * o[2];
    }
    rig.ok = true;
    return rig;
}

/** Midpoint of the closest points of two rays; gap = their distance. False if parallel. */
inline bool triangulate(const double o1[3], const double d1[3], const double o2[3], const double d2[3], double out[3], double& gap) {
    double w0[3] = {o1[0] - o2[0], o1[1] - o2[1], o1[2] - o2[2]};
    auto dot = [](const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    double a = dot(d1, d1), b = dot(d1, d2), c = dot(d2, d2), d = dot(d1, w0), e = dot(d2, w0);
    double den = a * c - b * b;
    if (std::fabs(den) < 1e-9) return false;
    double s = (b * e - c * d) / den, t = (a * e - b * d) / den;
    if (s <= 0 || t <= 0) return false;  // behind a camera
    double p1[3], p2[3];
    for (int i = 0; i < 3; i++) { p1[i] = o1[i] + d1[i] * s; p2[i] = o2[i] + d2[i] * t; out[i] = (p1[i] + p2[i]) / 2; }
    gap = std::sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[2] - p2[2]) * (p1[2] - p2[2]));
    return true;
}

}  // namespace calib
