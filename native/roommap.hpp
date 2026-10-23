// The room, scanned over time: every depth map (depth.hpp) is put into the VITURE
// tracking frame with the head pose at the image's time and accumulated in a voxel map
// (2.5 cm), which fills in as the wearer looks around. From it: the floor, the other
// horizontal surfaces (tables, shelves) and the walls, so virtual objects can rest on
// them; a surface mesh (surface nets over the confirmed voxels), drawn or used only to
// hide the virtual behind real things; and spatial anchors: a point on a real surface,
// kept in the tracking frame.
//
// The tracking frame is the engine's: it does not drift and does not move on recentre,
// but it starts anew at each launch. Anchors are written to myviture_anchors.txt for a
// later relocalisation; they are not loaded back yet.

#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "depth.hpp"
#include "hands.hpp"
#include "state.hpp"

namespace room {

constexpr float VOX = 0.025f;
constexpr int CONFIRM = 5;          // observations a voxel needs, from VIEWS different viewpoints
constexpr int VIEWS = 3;
constexpr size_t MAX_VOXELS = 400000;
constexpr size_t MAX_MESH_VERTS = 900000;

/** A voxel: mean of the points seen in it, how often, from how many viewpoints; misses when a ray passed through it. */
struct Voxel { float sum[3] = {0, 0, 0}; uint16_t n = 0; uint8_t misses = 0; uint32_t last_frame = 0; uint32_t views[4] = {}; uint8_t nviews = 0; };
inline bool confirmed(const Voxel& v) { return v.n >= CONFIRM && v.nviews >= VIEWS; }

struct Plane {
    int kind;                       // 0 floor, 1 horizontal surface, 2 wall
    float y;                        // height (tracking frame); walls: their bottom
    float minx, maxx, minz, maxz;   // extent (5th..95th percentiles of its voxels)
    int count;
    // Walls: n.(x, z) = d with n = (nx, nz) unit, from the room's side; along the wall
    // (t = (-nz, nx)) from t0 to t1, from y (bottom) to top.
    float nx = 0, nz = 0, d = 0, t0 = 0, t1 = 0, top = 0;
};

struct Anchor { int id; float p[3]; };

/** head -> tracking-frame pose at a time (QPC s), row-major 4x4; false if none. */
using PoseFn = std::function<bool(double, float*)>;

struct Shared {
    std::mutex mu;
    std::vector<float> points;      // confirmed voxels for drawing: x y z r g b, newest snapshot
    std::vector<Plane> planes;
    std::vector<Anchor> anchors;
    std::vector<float> depth;       // last depth map (W*H), for the dashboard
    std::vector<float> mesh;        // triangles, per vertex x y z nx ny nz (tracking frame)
    uint32_t mesh_version = 0;
    float valid_ratio = 0, rate = 0, ms = 0;
    size_t voxels = 0;
    std::string status = "démarrage…";
    int next_anchor = 1;
    uint32_t version = 0;           // bumped when `points` changes
};
inline Shared g;
inline std::atomic<bool> g_stop{false}, g_scan{true};
inline std::atomic<int> g_request{0};  // 1: anchor at the gaze, 2: clear
inline std::thread g_thread;
inline std::string g_anchor_path;

inline uint64_t key(int x, int y, int z) { return ((uint64_t)(x + 0x100000) << 42) | ((uint64_t)(y + 0x100000) << 21) | (uint64_t)(z + 0x100000); }

inline void save_anchors() {
    if (g_anchor_path.empty()) return;
    FILE* f = fopen(g_anchor_path.c_str(), "w");
    if (!f) return;
    for (auto& a : g.anchors) fprintf(f, "%d %.4f %.4f %.4f\n", a.id, a.p[0], a.p[1], a.p[2]);
    fclose(f);
}

/** Floor and horizontal surfaces from a height histogram of the confirmed voxels. */
inline std::vector<Plane> find_planes(const std::unordered_map<uint64_t, Voxel>& map, float head_min_y) {
    std::vector<Plane> out;
    std::vector<std::array<float, 3>> pts;
    for (auto& [k, v] : map) if (confirmed(v)) pts.push_back({v.sum[0] / v.n, v.sum[1] / v.n, v.sum[2] / v.n});
    if (pts.size() < 2000) return out;
    float ymin = 1e9, ymax = -1e9;
    for (auto& p : pts) { ymin = std::min(ymin, p[1]); ymax = std::max(ymax, p[1]); }
    const float BIN = 0.02f;
    int nb = (int)((ymax - ymin) / BIN) + 1;
    std::vector<int> hist(nb, 0);
    for (auto& p : pts) hist[(int)((p[1] - ymin) / BIN)]++;
    // Peaks: a bin above its neighbours and above 1.5 % of all voxels (floor: 0.8 %).
    auto extent = [&](int b, Plane& pl) {
        std::vector<float> xs, zs;
        float ys = 0; int n = 0;
        for (auto& p : pts) {
            int pb = (int)((p[1] - ymin) / BIN);
            if (std::abs(pb - b) > 1) continue;
            xs.push_back(p[0]); zs.push_back(p[2]); ys += p[1]; n++;
        }
        std::sort(xs.begin(), xs.end()); std::sort(zs.begin(), zs.end());
        pl.y = ys / n; pl.count = n;
        pl.minx = xs[xs.size() / 20]; pl.maxx = xs[xs.size() * 19 / 20];
        pl.minz = zs[zs.size() / 20]; pl.maxz = zs[zs.size() * 19 / 20];
    };
    int floor_bin = -1;
    for (int b = 0; b < nb; b++) {
        int c = hist[b], l = b ? hist[b - 1] : 0, r = b + 1 < nb ? hist[b + 1] : 0;
        if (c < l || c < r) continue;
        float frac = (float)(c + l + r) / pts.size();
        if (floor_bin < 0) {
            // The floor: wide, and well below the lowest the head has been (a bed or desk is not).
            if (frac < 0.01f || c < 300) continue;
            Plane pl{0, 0, 0, 0, 0, 0, 0};
            extent(b, pl);
            if (pl.y > head_min_y - 0.5f || pl.maxx - pl.minx < 1.2f || pl.maxz - pl.minz < 1.2f) continue;
            floor_bin = b;
            out.push_back(pl);
            continue;
        }
        if ((floor_bin >= 0 && b - floor_bin < 8) || frac < 0.03f || c < 600) continue;
        if (!out.empty() && out.back().kind == 1 && (b * BIN + ymin) - out.back().y < 0.08f) continue;  // same surface
        Plane pl{1, 0, 0, 0, 0, 0, 0};
        extent(b, pl);
        // A surface, not a wall's band: wide in both directions.
        if (pl.maxx - pl.minx > 0.4f && pl.maxz - pl.minz > 0.4f) out.push_back(pl);
    }
    return out;
}

/**
 * Walls: vertical planes in the confirmed voxels between 40 cm above the floor and 2 m,
 * found as lines in the top view (RANSAC, 3 cm), at least 80 cm long and 80 cm high.
 * Their normal faces the head (the room's side).
 */
inline std::vector<Plane> find_walls(const std::unordered_map<uint64_t, Voxel>& map, float floor_y, const float head_xz[2]) {
    std::vector<Plane> out;
    std::vector<std::array<float, 3>> pts;
    for (auto& [k, v] : map) {
        if (!confirmed(v)) continue;
        float y = v.sum[1] / v.n;
        if (y > floor_y + 0.4f && y < floor_y + 2.0f) pts.push_back({v.sum[0] / v.n, y, v.sum[2] / v.n});
    }
    std::mt19937 rng(7);
    if (pts.size() > 30000) { std::shuffle(pts.begin(), pts.end(), rng); pts.resize(30000); }
    for (int wall = 0; wall < 8 && pts.size() > 300; wall++) {
        std::uniform_int_distribution<size_t> pick(0, pts.size() - 1);
        size_t best_n = 0; float bnx = 0, bnz = 0, bd = 0;
        for (int it = 0; it < 400; it++) {
            auto& a = pts[pick(rng)]; auto& b = pts[pick(rng)];
            float dx = b[0] - a[0], dz = b[2] - a[2], len = std::sqrt(dx * dx + dz * dz);
            if (len < 0.4f) continue;
            float nx = -dz / len, nz = dx / len, d = nx * a[0] + nz * a[2];
            size_t n = 0;
            for (size_t i = 0; i < pts.size(); i += 4) if (std::fabs(nx * pts[i][0] + nz * pts[i][2] - d) < 0.03f) n++;
            if (n > best_n) { best_n = n; bnx = nx; bnz = nz; bd = d; }
        }
        if (best_n * 4 < 300) break;
        // Refit on the inliers (least squares in the top view), then the extent.
        std::vector<std::array<float, 3>> in, rest;
        for (auto& p : pts) (std::fabs(bnx * p[0] + bnz * p[2] - bd) < 0.03f ? in : rest).push_back(p);
        double mx = 0, mz = 0;
        for (auto& p : in) { mx += p[0]; mz += p[2]; }
        mx /= in.size(); mz /= in.size();
        double sxx = 0, sxz = 0, szz = 0;
        for (auto& p : in) { double x = p[0] - mx, z = p[2] - mz; sxx += x * x; sxz += x * z; szz += z * z; }
        double ang = 0.5 * std::atan2(2 * sxz, sxx - szz);  // direction of the line
        float tx = (float)std::cos(ang), tz = (float)std::sin(ang);
        bnx = -tz; bnz = tx; bd = (float)(bnx * mx + bnz * mz);
        std::vector<float> ts, ys;
        for (auto& p : in) { ts.push_back(tx * p[0] + tz * p[2]); ys.push_back(p[1]); }
        std::sort(ts.begin(), ts.end()); std::sort(ys.begin(), ys.end());
        Plane pl{2, ys[ys.size() / 20], 0, 0, 0, 0, (int)in.size()};
        pl.top = ys[ys.size() * 19 / 20];
        pl.t0 = ts[ts.size() / 20]; pl.t1 = ts[ts.size() * 19 / 20];
        // Normal towards the head; t = (-nz, nx) with that normal.
        if (bnx * head_xz[0] + bnz * head_xz[1] - bd < 0) { bnx = -bnx; bnz = -bnz; bd = -bd; }
        float ttx = -bnz, ttz = bnx;
        if (ttx * tx + ttz * tz < 0) { float a = -pl.t1, b = -pl.t0; pl.t0 = a; pl.t1 = b; }
        pl.nx = bnx; pl.nz = bnz; pl.d = bd;
        pl.minx = std::min(bd * bnx + ttx * pl.t0, bd * bnx + ttx * pl.t1); pl.maxx = std::max(bd * bnx + ttx * pl.t0, bd * bnx + ttx * pl.t1);
        pl.minz = std::min(bd * bnz + ttz * pl.t0, bd * bnz + ttz * pl.t1); pl.maxz = std::max(bd * bnz + ttz * pl.t0, bd * bnz + ttz * pl.t1);
        pts.swap(rest);
        if (pl.t1 - pl.t0 >= 0.8f && pl.top - pl.y >= 0.8f) out.push_back(pl);
    }
    return out;
}

/**
 * Surface mesh of the confirmed voxels (naive surface nets): one vertex per grid cell
 * that has both filled and empty corners, at the mean of its crossing edges' midpoints,
 * smoothed once; one quad per filled/empty voxel pair, across the 4 cells around it.
 * Lone voxels (fewer than 2 filled neighbours) are left out. Triangles, x y z nx ny nz.
 */
inline std::vector<float> build_mesh(const std::unordered_map<uint64_t, Voxel>& map, size_t max_verts) {
    auto unkey = [](uint64_t k, int& x, int& y, int& z) {
        x = (int)((k >> 42) & 0x1FFFFF) - 0x100000; y = (int)((k >> 21) & 0x1FFFFF) - 0x100000; z = (int)(k & 0x1FFFFF) - 0x100000;
    };
    std::unordered_set<uint64_t> conf;
    conf.reserve(map.size());
    for (auto& [k, v] : map) if (confirmed(v)) conf.insert(k);
    std::unordered_set<uint64_t> filled;
    filled.reserve(conf.size());
    for (uint64_t k : conf) {
        int x, y, z; unkey(k, x, y, z);
        int nb = 0;
        for (int dx = -1; dx <= 1 && nb < 2; dx++) for (int dy = -1; dy <= 1 && nb < 2; dy++) for (int dz = -1; dz <= 1 && nb < 2; dz++)
            if ((dx || dy || dz) && conf.count(key(x + dx, y + dy, z + dz))) nb++;
        if (nb >= 2) filled.insert(k);
    }
    auto at = [&](int x, int y, int z) { return filled.count(key(x, y, z)) != 0; };
    // Cell (x, y, z): corners (x..x+1, y..y+1, z..z+1). Its vertex, in voxel units.
    std::unordered_map<uint64_t, std::array<float, 3>> cell;
    cell.reserve(filled.size() * 3);
    static const int E[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    auto make_cell = [&](int x, int y, int z) {
        uint64_t ck = key(x, y, z);
        if (cell.count(ck)) return;
        bool c[8]; int n = 0;
        for (int i = 0; i < 8; i++) { c[i] = at(x + (i & 1), y + ((i >> 1) & 1), z + ((i >> 2) & 1)); n += c[i]; }
        if (n == 0 || n == 8) return;
        float s[3] = {0, 0, 0}; int m = 0;
        for (auto& e : E) {
            if (c[e[0]] == c[e[1]]) continue;
            int a = e[0], b = e[1];
            s[0] += x + ((a & 1) + (b & 1)) * 0.5f; s[1] += y + (((a >> 1) & 1) + ((b >> 1) & 1)) * 0.5f; s[2] += z + (((a >> 2) & 1) + ((b >> 2) & 1)) * 0.5f;
            m++;
        }
        cell[ck] = {s[0] / m, s[1] / m, s[2] / m};
    };
    for (uint64_t k : filled) {
        int x, y, z; unkey(k, x, y, z);
        for (int i = 0; i < 8; i++) make_cell(x - (i & 1), y - ((i >> 1) & 1), z - ((i >> 2) & 1));
    }
    // One smoothing pass: each vertex towards its face neighbours.
    std::unordered_map<uint64_t, std::array<float, 3>> smooth;
    smooth.reserve(cell.size());
    for (auto& [ck, p] : cell) {
        int x, y, z; unkey(ck, x, y, z);
        float s[3] = {p[0], p[1], p[2]}; int n = 1;
        const int D[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        for (auto& d : D) {
            auto it = cell.find(key(x + d[0], y + d[1], z + d[2]));
            if (it != cell.end()) { for (int a = 0; a < 3; a++) s[a] += it->second[a]; n++; }
        }
        smooth[ck] = {s[0] / n, s[1] / n, s[2] / n};
    }
    // Quads: for a filled voxel and an empty neighbour along +/- axis a, the 4 cells
    // around the edge between their centres. Facing the empty side.
    std::vector<float> out;
    auto vert = [&](const std::array<float, 3>& p, const float* n) {
        // Grid point i is the centre of voxel i: ((i + 0.5) * VOX).
        out.insert(out.end(), {(p[0] + 0.5f) * VOX, (p[1] + 0.5f) * VOX, (p[2] + 0.5f) * VOX, n[0], n[1], n[2]});
    };
    for (uint64_t k : filled) {
        if (out.size() / 6 + 6 > max_verts) break;
        int x, y, z; unkey(k, x, y, z);
        for (int a = 0; a < 3; a++) for (int sgn = -1; sgn <= 1; sgn += 2) {
            int q[3] = {x, y, z}; q[a] += sgn;
            if (at(q[0], q[1], q[2])) continue;
            int lo[3] = {x, y, z}; if (sgn < 0) lo[a] -= 1;  // the edge from lo to lo + e_a
            int b = (a + 1) % 3, c2 = (a + 2) % 3;
            std::array<float, 3> v4[4]; bool ok = true;
            const int off[4][2] = {{0, 0}, {-1, 0}, {-1, -1}, {0, -1}};
            for (int i = 0; i < 4 && ok; i++) {
                int cc[3] = {lo[0], lo[1], lo[2]}; cc[b] += off[i][0]; cc[c2] += off[i][1];
                auto it = smooth.find(key(cc[0], cc[1], cc[2]));
                if (it == smooth.end()) ok = false; else v4[i] = it->second;
            }
            if (!ok) continue;
            float n[3] = {0, 0, 0}; n[a] = (float)sgn;
            vert(v4[0], n); vert(v4[1], n); vert(v4[2], n);
            vert(v4[0], n); vert(v4[2], n); vert(v4[3], n);
        }
    }
    return out;
}

inline void run(calib::Rig rig, PoseFn pose_at) {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    depth::Stereo stereo(rig);
    if (!stereo.ok) { std::lock_guard<std::mutex> l(g.mu); g.status = "profondeur GPU : " + stereo.error; return; }
    { std::lock_guard<std::mutex> l(g.mu); g.status = ""; }
    std::unordered_map<uint64_t, Voxel> map;
    map.reserve(MAX_VOXELS);
    uint32_t seen = 0, frame_id = 0;
    std::vector<uint8_t> img[2];
    int w = 0, h = 0;
    double last_planes = 0, last_mesh = 0, rate_t0 = hands::qpc_s(); int frames = 0;
    float head_min_y = 1e9;
    depth::Frame df;

    while (!g_stop) {
        int req = g_request.exchange(0);
        if (req == 2) {
            map.clear();
            std::lock_guard<std::mutex> l(g.mu);
            g.points.clear(); g.planes.clear(); g.anchors.clear(); g.voxels = 0; g.version++;
            g.mesh.clear(); g.mesh_version++;
            save_anchors();
        }
        double t_img = 0;
        {
            std::lock_guard<std::mutex> l(g_cams.mu);
            // Every third camera frame (~8 Hz): the GPU is shared with the hands and the glasses.
            if (g_cams.w && g_cams.seq != seen && g_cams.seq % 3 == 0 && g_cams.px[0].size() >= (size_t)g_cams.w * g_cams.h && g_cams.px[1].size() >= (size_t)g_cams.w * g_cams.h) {
                seen = g_cams.seq; img[0] = g_cams.px[0]; img[1] = g_cams.px[1]; w = g_cams.w; h = g_cams.h; t_img = g_cams.t;
            }
        }
        if (!t_img || w != 640 || h != 480) { Sleep(3); continue; }
        float head[16];
        if (!pose_at(t_img, head)) continue;
        head_min_y = std::min(head_min_y, head[7]);

        double t0 = hands::qpc_s();
        if (!stereo.compute(img[0].data(), img[1].data(), df)) continue;
        df.t = t_img;
        double t1 = hands::qpc_s();

        // Hands at that time (head frame): their points must not enter the room.
        std::vector<std::array<float, 3>> hand_pts;
        {
            std::lock_guard<std::mutex> l(hands::g_mu);
            if (std::fabs(hands::g_result.t - t_img) < 0.15) for (auto& hd : hands::g_result.hands) for (int k = 0; k < hands::N_JOINTS; k++) hand_pts.push_back({hd.joints[k][0], hd.joints[k][1], hd.joints[k][2]});
        }
        auto near_hand = [&](const double p[3]) {
            for (auto& q : hand_pts) { double dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2]; if (dx * dx + dy * dy + dz * dz < 0.12 * 0.12) return true; }
            return false;
        };

        if (g_scan) {
            frame_id++;
            // Viewpoint: head position to 10 cm and heading to 15 deg; a voxel counts as
            // confirmed only when seen from several, so one viewpoint's noise does not stay.
            double hx = head[3], hy = head[7], hz = head[11], hyaw = std::atan2(-head[2], -head[10]);
            uint32_t view = (uint32_t)(((int)std::floor(hx / 0.1) & 1023) | (((int)std::floor(hy / 0.1) & 1023) << 10) | (((int)std::floor(hz / 0.1) & 1023) << 20)) ^ ((uint32_t)((int)std::floor(hyaw / 0.26) & 31) << 27);
            double o[3] = {hx, hy, hz};
            for (int v = 0; v < depth::H; v += 2) for (int u = 0; u < depth::W; u += 2) {
                float z = df.depth[v * depth::W + u];
                if (!(z > 0.3f && z < 3.0f)) continue;
                double ph[3], pw[3];
                df.point(u, v, z, ph);
                if (!hand_pts.empty() && near_hand(ph)) continue;
                for (int i = 0; i < 3; i++) pw[i] = head[i * 4] * ph[0] + head[i * 4 + 1] * ph[1] + head[i * 4 + 2] * ph[2] + head[i * 4 + 3];
                // Free space: voxels on the ray before the surface were seen through.
                double dx = pw[0] - o[0], dy = pw[1] - o[1], dz = pw[2] - o[2];
                double len = std::sqrt(dx * dx + dy * dy + dz * dz);
                for (double s = 0.3; s < len - 0.06; s += 0.05) {
                    auto it = map.find(key((int)std::floor((o[0] + dx * s / len) / VOX), (int)std::floor((o[1] + dy * s / len) / VOX), (int)std::floor((o[2] + dz * s / len) / VOX)));
                    if (it == map.end() || it->second.last_frame == frame_id) continue;
                    if (++it->second.misses >= 4 && it->second.misses > it->second.n) map.erase(it);
                }
                uint64_t k = key((int)std::floor(pw[0] / VOX), (int)std::floor(pw[1] / VOX), (int)std::floor(pw[2] / VOX));
                auto it = map.find(k);
                if (it == map.end()) {
                    if (map.size() >= MAX_VOXELS) continue;
                    it = map.emplace(k, Voxel{}).first;
                }
                Voxel& vx = it->second;
                if (vx.last_frame == frame_id) continue;  // once per frame
                vx.last_frame = frame_id;
                if (vx.n < 60000) { for (int i = 0; i < 3; i++) vx.sum[i] += (float)pw[i]; vx.n++; }
                if (vx.misses) vx.misses--;
                bool known = false;
                for (int i = 0; i < vx.nviews; i++) if (vx.views[i] == view) known = true;
                if (!known && vx.nviews < 4) vx.views[vx.nviews++] = view;
            }
        }

        // Anchor at the gaze: the head's -z ray against the confirmed voxels.
        if (req == 1) {
            double o[3] = {head[3], head[7], head[11]}, d[3] = {-head[2], -head[6], -head[10]};
            bool hit = false; double hp[3] = {};
            for (double s = 0.25; s < 5 && !hit; s += VOX * 0.5) {
                for (int i = 0; i < 3; i++) hp[i] = o[i] + d[i] * s;
                int cx = (int)std::floor(hp[0] / VOX), cy = (int)std::floor(hp[1] / VOX), cz = (int)std::floor(hp[2] / VOX);
                for (int dx = -1; dx <= 1 && !hit; dx++) for (int dy = -1; dy <= 1 && !hit; dy++) for (int dz = -1; dz <= 1 && !hit; dz++) {
                    auto it = map.find(key(cx + dx, cy + dy, cz + dz));
                    if (it != map.end() && confirmed(it->second)) { hit = true; for (int i = 0; i < 3; i++) hp[i] = it->second.sum[i] / it->second.n; }
                }
            }
            std::lock_guard<std::mutex> l(g.mu);
            if (hit) { g.anchors.push_back({g.next_anchor++, {(float)hp[0], (float)hp[1], (float)hp[2]}}); save_anchors(); g.status = ""; }
            else g.status = "ancre : rien de scanné dans cette direction";
        }

        frames++;
        double now = hands::qpc_s();
        bool refresh = now - last_planes > 2;
        bool remesh = refresh && now - last_mesh > 4;
        std::vector<Plane> planes;
        std::vector<float> pts, mesh;
        if (refresh) {
            last_planes = now;
            double tr0 = hands::qpc_s();
            planes = find_planes(map, head_min_y);
            if (!planes.empty() && planes[0].kind == 0) {
                float hxz[2] = {head[3], head[11]};
                for (auto& w : find_walls(map, planes[0].y, hxz)) planes.push_back(w);
            }
            if (remesh) { last_mesh = now; mesh = build_mesh(map, MAX_MESH_VERTS); }
            double tr1 = hands::qpc_s();
            // Drawing snapshot: confirmed voxels, coloured by height, at most 120k.
            size_t n_conf = 0;
            for (auto& [k, v] : map) if (confirmed(v)) n_conf++;
            int step = (int)std::max<size_t>(1, n_conf / 40000);
            int i = 0;
            float floor_y = planes.empty() ? -1.6f : planes[0].y;
            for (auto& [k, v] : map) {
                if (!confirmed(v) || (i++ % step)) continue;
                float x = v.sum[0] / v.n, y = v.sum[1] / v.n, z = v.sum[2] / v.n;
                float t = std::clamp((y - floor_y) / 2.0f, 0.0f, 1.0f);
                pts.insert(pts.end(), {x, y, z, 0.2f + 0.8f * t, 0.5f + 0.3f * (1 - t), 1.0f - 0.7f * t});
            }
            int walls = 0; for (auto& p : planes) walls += p.kind == 2;
            hands::logf("room: refresh %zu voxels, %d walls, planes+mesh %.0f ms (%zu mesh vertices), snapshot %.0f ms\n", map.size(), walls, (tr1 - tr0) * 1000, mesh.size() / 6, (hands::qpc_s() - tr1) * 1000);
        }
        std::lock_guard<std::mutex> l(g.mu);
        g.depth = df.depth;
        g.valid_ratio = df.valid_ratio;
        g.ms = (float)((t1 - t0) * 1000);
        g.voxels = map.size();
        if (now - rate_t0 > 1) { g.rate = (float)(frames / (now - rate_t0)); frames = 0; rate_t0 = now; }
        if (refresh) { g.planes = planes; g.points = std::move(pts); g.version++; }
        if (remesh) { g.mesh = std::move(mesh); g.mesh_version++; }
    }
}

inline void start(const calib::Rig& rig, PoseFn pose_at, const std::string& anchor_path) {
    g_anchor_path = anchor_path;
    g_thread = std::thread(run, rig, pose_at);
}
inline void stop() { g_stop = true; if (g_thread.joinable()) g_thread.join(); }

}  // namespace room
