// Display calibration with the hands: targets are drawn fixed to the head, the wearer
// pinches each one where they see it, and the cameras measure where the pinch really
// is. The real points r and the drawn points t (head frame) give, by Umeyama's method,
// the similarity D with r = s R t + c that the display applies: shown points then land
// where the real ones are. D covers the eyes' offset from the cameras, the optics' tilt
// and a field-of-view error (the scale).

#pragma once
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace align {

struct Pair { double t[3], r[3]; };

/** Eigenvector of the largest eigenvalue of a symmetric 4x4 matrix (Jacobi). */
inline void top_eigenvector(double A[4][4], double v[4]) {
    double V[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    for (int sweep = 0; sweep < 50; sweep++) {
        double off = 0;
        for (int p = 0; p < 4; p++) for (int q = p + 1; q < 4; q++) off += A[p][q] * A[p][q];
        if (off < 1e-18) break;
        for (int p = 0; p < 4; p++) for (int q = p + 1; q < 4; q++) {
            if (std::fabs(A[p][q]) < 1e-15) continue;
            double th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
            double t = (th >= 0 ? 1 : -1) / (std::fabs(th) + std::sqrt(th * th + 1));
            double c = 1 / std::sqrt(t * t + 1), s = t * c;
            for (int k = 0; k < 4; k++) { double a = A[k][p], b = A[k][q]; A[k][p] = c * a - s * b; A[k][q] = s * a + c * b; }
            for (int k = 0; k < 4; k++) { double a = A[p][k], b = A[q][k]; A[p][k] = c * a - s * b; A[q][k] = s * a + c * b; }
            for (int k = 0; k < 4; k++) { double a = V[k][p], b = V[k][q]; V[k][p] = c * a - s * b; V[k][q] = s * a + c * b; }
        }
    }
    int best = 0;
    for (int i = 1; i < 4; i++) if (A[i][i] > A[best][best]) best = i;
    for (int i = 0; i < 4; i++) v[i] = V[i][best];
}

struct Result { double M[16]; double scale, rms_mm, rot_deg, shift_mm[3]; bool ok; };

/** r = s R t + c over the pairs (Horn's quaternion rotation, then scale and shift). */
inline Result solve(const std::vector<Pair>& P) {
    Result out{};
    if (P.size() < 4) return out;
    double tc[3] = {}, rc[3] = {};
    for (auto& p : P) for (int i = 0; i < 3; i++) { tc[i] += p.t[i] / P.size(); rc[i] += p.r[i] / P.size(); }
    double S[3][3] = {};
    for (auto& p : P) for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) S[i][j] += (p.t[i] - tc[i]) * (p.r[j] - rc[j]);
    double N[4][4] = {
        {S[0][0] + S[1][1] + S[2][2], S[1][2] - S[2][1], S[2][0] - S[0][2], S[0][1] - S[1][0]},
        {S[1][2] - S[2][1], S[0][0] - S[1][1] - S[2][2], S[0][1] + S[1][0], S[2][0] + S[0][2]},
        {S[2][0] - S[0][2], S[0][1] + S[1][0], -S[0][0] + S[1][1] - S[2][2], S[1][2] + S[2][1]},
        {S[0][1] - S[1][0], S[2][0] + S[0][2], S[1][2] + S[2][1], -S[0][0] - S[1][1] + S[2][2]}};
    double q[4];
    top_eigenvector(N, q);  // (w, x, y, z)
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double R[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
                      {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
                      {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
    double num = 0, den = 0;
    for (auto& p : P) {
        double a[3] = {p.t[0] - tc[0], p.t[1] - tc[1], p.t[2] - tc[2]}, b[3] = {p.r[0] - rc[0], p.r[1] - rc[1], p.r[2] - rc[2]};
        for (int i = 0; i < 3; i++) num += (R[i][0] * a[0] + R[i][1] * a[1] + R[i][2] * a[2]) * b[i];
        den += a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
    }
    double s = num / den;
    double c[3];
    for (int i = 0; i < 3; i++) c[i] = rc[i] - s * (R[i][0] * tc[0] + R[i][1] * tc[1] + R[i][2] * tc[2]);
    for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) out.M[i * 4 + j] = s * R[i][j]; out.M[i * 4 + 3] = c[i]; }
    out.M[12] = out.M[13] = out.M[14] = 0; out.M[15] = 1;
    double e = 0;
    for (auto& p : P) {
        for (int i = 0; i < 3; i++) {
            double v = out.M[i * 4] * p.t[0] + out.M[i * 4 + 1] * p.t[1] + out.M[i * 4 + 2] * p.t[2] + out.M[i * 4 + 3] - p.r[i];
            e += v * v;
        }
    }
    out.rms_mm = std::sqrt(e / P.size()) * 1000;
    out.scale = s;
    out.rot_deg = 2 * std::acos(std::min(1.0, std::fabs(w))) * 57.2958;
    for (int i = 0; i < 3; i++) out.shift_mm[i] = c[i] * 1000;
    // A pinch far from its target, or an implausible correction: refuse.
    out.ok = out.rms_mm < 25 && s > 0.7 && s < 1.4 && out.rot_deg < 15;
    return out;
}

/** Targets in the head frame (OpenGL axes: x right, y up, -z forward), two depths. */
inline std::vector<std::array<double, 3>> targets() {
    std::vector<std::array<double, 3>> t;
    for (double d : {0.32, 0.48}) {
        double k = d / 0.4;
        double pts[5][2] = {{0, 0}, {-0.11, 0}, {0.11, 0}, {0, 0.06}, {0, -0.08}};
        for (auto& p : pts) t.push_back({p[0] * k, p[1] * k, -d});
    }
    return t;
}

inline bool save(const std::string& path, const double M[16]) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;
    for (int i = 0; i < 16; i++) fprintf(f, "%.9f%c", M[i], i % 4 == 3 ? '\n' : ' ');
    fclose(f);
    return true;
}

inline bool load(const std::string& path, double M[16]) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    int n = 0;
    for (int i = 0; i < 16; i++) n += fscanf(f, "%lf", &M[i]);
    fclose(f);
    return n == 16;
}

}  // namespace align
