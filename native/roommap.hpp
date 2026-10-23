// The room, scanned over time: every depth map (depth.hpp) is put into the VITURE
// tracking frame with the head pose at the image's time. Two models are built from it:
//   - planes first: each depth map's normals say which points lie on something level
//     (floor, table tops) or upright (walls); that evidence is kept in 5 cm cells and,
//     every 2 s, gives the floor's height (a height histogram), the table tops (peaks above
//     it, split into connected pieces) and the walls (a Hough vote on the walls' direction
//     and distance, snapped to the room's dominant axes). Textureless walls are where the
//     stereo has few points, and a plane needs few. From the floor, the obstacles and the
//     walls: the walkable grid (10 cm), for a virtual human to find its way.
//   - the clutter: a voxel map (2.5 cm) of what was seen from several viewpoints, and its
//     surface mesh (surface nets) near the head, without the floor and the walls, used only
//     to hide the virtual behind real things. Unconfirmed voxels are forgotten after 5 s.
// And spatial anchors: a point on a real surface, kept in the tracking frame.
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
#include "semseg.hpp"
#include "state.hpp"

namespace room {

constexpr float VOX = 0.025f;
constexpr int CONFIRM = 5;          // observations a voxel needs, from VIEWS different viewpoints
constexpr int VIEWS = 3;
constexpr size_t MAX_VOXELS = 400000;
constexpr size_t MAX_MESH_VERTS = 300000;   // nearest first, within MESH_RANGE of the head
constexpr float MESH_RANGE = 4.5f;
constexpr float EV = 0.05f;                 // plane evidence cells (m)
constexpr float WALK_CELL = 0.10f;          // walkable grid (m)
constexpr float HUMAN_RADIUS = 0.22f;       // kept away from obstacles (m)

/** A voxel: mean of the points seen in it, how often, from how many viewpoints; misses when a ray passed through it. */
struct Voxel { float sum[3] = {0, 0, 0}; uint16_t n = 0; uint8_t misses = 0; uint32_t last_frame = 0; uint32_t views[4] = {}; uint8_t nviews = 0; uint8_t cat = 0, cat_n = 0; };

/** A running majority (Boyer-Moore) of the categories seen at a place: cat wins while cat_n > 0. */
inline void vote(uint8_t& cat, uint8_t& n, uint8_t c) {
    if (!c) return;
    if (cat == c) { if (n < 250) n++; }
    else if (n == 0) { cat = c; n = 1; }
    else n--;
}
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

/** Walkable grid on the floor (tracking frame, 10 cm cells): 0 unknown, 1 walkable, 2 blocked. */
struct WalkGrid {
    float x0 = 0, z0 = 0, cell = WALK_CELL;
    int nx = 0, nz = 0;
    std::vector<uint8_t> c;
    uint8_t at(float x, float z) const {
        int i = (int)std::floor((x - x0) / cell), j = (int)std::floor((z - z0) / cell);
        return (i < 0 || j < 0 || i >= nx || j >= nz) ? 0 : c[j * nx + i];
    }
};

/** Plane evidence in a 5 cm cell: points on a level or an upright surface, and their summed normal. */
struct Evid { float sum[3] = {0, 0, 0}; float nx = 0, nz = 0; uint16_t n = 0; uint8_t cat = 0, cat_n = 0; };

/** A door or a window seen in a wall: along the wall (t) and in height (tracking frame). */
struct Opening { int kind; int wall; float t0, t1, y0, y1; };  // kind: semseg::DOOR or WINDOW; wall: index in planes

/** head -> tracking-frame pose at a time (QPC s), row-major 4x4; false if none. */
using PoseFn = std::function<bool(double, float*)>;

struct Shared {
    std::mutex mu;
    std::vector<float> points;      // confirmed voxels for drawing: x y z r g b, newest snapshot
    std::vector<Plane> planes;
    std::vector<Anchor> anchors;
    std::vector<float> depth;       // last depth map (W*H), for the dashboard
    std::vector<float> mesh;        // triangles, per vertex x y z nx ny nz (tracking frame); taken by the renderer
    uint32_t mesh_version = 0;
    size_t mesh_tris = 0;
    // The room model (tracking frame): floor height, walkable grid.
    bool has_floor = false;
    float floor_y = 0;
    WalkGrid walk;
    uint32_t walk_version = 0;
    std::vector<Opening> openings;
    size_t labelled = 0;            // plane evidence cells with a category (semantic segmentation)
    std::string floor_source;
    // For the 3D debug view, x y z r g b each (tracking frame): the plane evidence (level
    // green, upright orange, refreshed every 2 s) and the last depth map's points (white).
    std::vector<float> dbg_evidence, dbg_live;
    uint32_t dbg_version = 0, live_version = 0;
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

inline uint64_t ekey(float x, float y, float z) { return key((int)std::floor(x / EV), (int)std::floor(y / EV), (int)std::floor(z / EV)); }

/** Result of the room model. */
struct Model { std::vector<Plane> planes; bool has_floor = false; float floor_y = 0; WalkGrid walk; std::vector<Opening> openings; size_t labelled = 0; std::string floor_source; };

/**
 * The room from the plane evidence: floor (lowest strong level, well under the head),
 * table tops (level peaks 30 cm to 1.5 m above it, in connected pieces of 0.1 m2 or more),
 * walls (upright cells voting for a direction and a distance, snapped to the room's
 * dominant axes when within 12 degrees; at least 50 cm long and 40 cm high), then the
 * walkable grid: floor seen, not under a confirmed voxel 10 cm to 1.8 m above it, not on
 * a table or a wall, kept HUMAN_RADIUS away from them.
 */
inline Model room_model(const std::unordered_map<uint64_t, Evid>& level, const std::unordered_map<uint64_t, Evid>& upright,
                        const std::unordered_map<uint64_t, Voxel>& map, float head_min_y) {
    Model m;
    using namespace semseg;
    struct P { float x, y, z; int n; uint8_t cat; };
    std::vector<P> lv;
    for (auto& [k, e] : level) if (e.n >= 3) lv.push_back({e.sum[0] / e.n, e.sum[1] / e.n, e.sum[2] / e.n, e.n, e.cat_n ? e.cat : (uint8_t)NONE});
    for (auto& p : lv) m.labelled += p.cat != NONE;
    for (auto& [k, e] : upright) m.labelled += e.n >= 3 && e.cat_n;
    // Seen as floor by the segmentation: the floor is their median height (a bit of rug or
    // floor seen is enough, and a desk cannot pass for it).
    {
        std::vector<float> fy;
        for (auto& p : lv) if (p.cat == FLOOR && p.y < head_min_y - 0.5f) fy.push_back(p.y);
        if (fy.size() >= 15) {
            std::sort(fy.begin(), fy.end());
            float med = fy[fy.size() / 2];
            double sy = 0; int n = 0;
            for (float y : fy) if (std::fabs(y - med) < 0.04f) { sy += y; n++; }
            m.floor_y = (float)(sy / n); m.has_floor = true; m.floor_source = "segmentation";
        }
    }
    if (lv.size() < 150 && !m.has_floor) return m;
    // Level surfaces the segmentation calls furniture, a table or a screen are not the floor.
    std::vector<P> lv_floor;
    for (auto& p : lv) if (p.cat != TABLE && p.cat != FURNITURE && p.cat != SCREEN && p.cat != CEILING) lv_floor.push_back(p);
    float ymin = 1e9, ymax = -1e9;
    for (auto& p : lv) { ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y); }
    const float BIN = 0.02f;
    int nb = (int)((ymax - ymin) / BIN) + 3;
    std::vector<int> hist(nb, 0);
    for (auto& p : lv) hist[(int)((p.y - ymin) / BIN) + 1]++;
    auto around = [&](int b) { return hist[b] + (b ? hist[b - 1] : 0) + (b + 1 < nb ? hist[b + 1] : 0); };
    // Floor: the lowest bin, a local peak with 2 % of the level cells (at least 30), 95 cm
    // under the lowest head (seated, the eyes are 1.1 m or more above the floor; a desk is
    // about 0.5 m under them, a low table 0.75 m).
    std::vector<int> fhist(nb, 0);
    for (auto& p : lv_floor) fhist[(int)((p.y - ymin) / BIN) + 1]++;
    auto faround = [&](int b) { return fhist[b] + (b ? fhist[b - 1] : 0) + (b + 1 < nb ? fhist[b + 1] : 0); };
    for (int b = 1; b + 1 < nb && !m.has_floor; b++) {
        float y = ymin + (b - 1 + 0.5f) * BIN;
        if (y > head_min_y - 0.95f) break;
        if (fhist[b] < fhist[b - 1] || fhist[b] < fhist[b + 1] || faround(b) < std::max(30, (int)(0.02f * lv_floor.size()))) continue;
        double sy = 0, sw = 0;
        for (auto& p : lv_floor) if (std::fabs(p.y - y) < 0.04f) { sy += p.y * p.n; sw += p.n; }
        m.floor_y = (float)(sy / sw); m.has_floor = true; m.floor_source = "hauteurs";
        break;
    }
    if (!m.has_floor) {
        // Why not: the three strongest levels, for the log.
        std::vector<std::pair<int, float>> top;
        for (int b = 1; b + 1 < nb; b++) top.push_back({around(b), ymin + (b - 1 + 0.5f) * BIN});
        std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.first > b.first; });
        char buf[160]; int n = snprintf(buf, sizeof buf, "room: no floor yet (head lowest %.2f m); strongest levels:", head_min_y);
        for (size_t i = 0; i < std::min<size_t>(3, top.size()); i++) n += snprintf(buf + n, sizeof buf - n, " %.2f m (%d)", top[i].second, top[i].first);
        hands::logf("%s\n", buf);
        // Walls and table tops still, above a guessed floor (1.6 m under the lowest head);
        // no floor plane and no walkable grid until the floor is seen.
        m.floor_y = head_min_y - 1.6f;
    }
    if (m.has_floor) {
        Plane fl{0, m.floor_y, 1e9f, -1e9f, 1e9f, -1e9f, 0};
        for (auto& p : lv) if (std::fabs(p.y - m.floor_y) < 0.05f) { fl.minx = std::min(fl.minx, p.x); fl.maxx = std::max(fl.maxx, p.x); fl.minz = std::min(fl.minz, p.z); fl.maxz = std::max(fl.maxz, p.z); fl.count++; }
        m.planes.push_back(fl);
    }
    // Table tops: peaks 30 cm to 1.5 m above the floor, then connected pieces in 10 cm cells.
    for (int b = 1; b + 1 < nb; b++) {
        float y = ymin + (b - 1 + 0.5f) * BIN;
        if (y < m.floor_y + 0.3f || y > m.floor_y + 1.5f) continue;
        if (hist[b] < hist[b - 1] || hist[b] < hist[b + 1] || around(b) < 25) continue;
        std::unordered_map<uint64_t, std::array<float, 2>> cells;  // 10 cm cell -> sum y, count
        for (auto& p : lv) if (std::fabs(p.y - y) < 0.03f && (p.cat == NONE || p.cat == TABLE || p.cat == OTHER)) {
            auto& cc = cells[key((int)std::floor(p.x / 0.1f), 0, (int)std::floor(p.z / 0.1f))];
            cc[0] += p.y; cc[1] += 1;
        }
        std::unordered_set<uint64_t> seen;
        for (auto& [k0, v0] : cells) {
            if (seen.count(k0)) continue;
            std::vector<uint64_t> stack{k0}, comp;
            seen.insert(k0);
            while (!stack.empty()) {
                uint64_t k = stack.back(); stack.pop_back(); comp.push_back(k);
                int x = (int)((k >> 42) & 0x1FFFFF) - 0x100000, z = (int)(k & 0x1FFFFF) - 0x100000;
                for (int dx = -1; dx <= 1; dx++) for (int dz = -1; dz <= 1; dz++) {
                    uint64_t nk = key(x + dx, 0, z + dz);
                    if (cells.count(nk) && !seen.count(nk)) { seen.insert(nk); stack.push_back(nk); }
                }
            }
            if (comp.size() < 10) continue;
            Plane pl{1, 0, 1e9f, -1e9f, 1e9f, -1e9f, (int)comp.size()};
            double sy = 0, sn = 0;
            for (uint64_t k : comp) {
                int x = (int)((k >> 42) & 0x1FFFFF) - 0x100000, z = (int)(k & 0x1FFFFF) - 0x100000;
                pl.minx = std::min(pl.minx, x * 0.1f); pl.maxx = std::max(pl.maxx, (x + 1) * 0.1f);
                pl.minz = std::min(pl.minz, z * 0.1f); pl.maxz = std::max(pl.maxz, (z + 1) * 0.1f);
                sy += cells[k][0]; sn += cells[k][1];
            }
            pl.y = (float)(sy / sn);
            bool dup = false;  // the same top from a neighbouring bin
            for (auto& o : m.planes) if (o.kind == 1 && std::fabs(o.y - pl.y) < 0.06f && pl.minx < o.maxx && o.minx < pl.maxx && pl.minz < o.maxz && o.minz < pl.maxz) dup = true;
            if (!dup) m.planes.push_back(pl);
        }
    }
    // Walls.
    struct U { float x, y, z, th; int n; bool used; uint8_t cat; };
    std::vector<U> up;
    for (auto& [k, e] : upright) {
        if (e.n < 3) continue;
        uint8_t ec = e.cat_n ? e.cat : (uint8_t)NONE;
        if (ec != NONE && ec != WALL && ec != DOOR && ec != WINDOW) continue;  // furniture, screens, people: not walls
        float y = e.sum[1] / e.n;
        if (y < m.floor_y + 0.05f || y > m.floor_y + 2.6f) continue;
        if (e.nx * e.nx + e.nz * e.nz < 1e-6f) continue;
        up.push_back({e.sum[0] / e.n, y, e.sum[2] / e.n, std::atan2(e.nz, e.nx), e.n, false, ec});
    }
    const float PI = 3.14159265f;
    auto angdiff = [&](float a, float b) { float d = std::fmod(a - b, 2 * PI); if (d > PI) d -= 2 * PI; if (d < -PI) d += 2 * PI; return std::fabs(d); };
    if (up.size() >= 40) {
        // The room's axes: the strongest direction modulo 90 degrees.
        float h90[90] = {};
        for (auto& u : up) { float a = std::fmod(u.th + 4 * PI, PI / 2) * 180 / PI; h90[std::min(89, (int)a)] += (float)u.n; }
        int best = 0; float bv = -1;
        for (int i = 0; i < 90; i++) { float v = 0; for (int d = -2; d <= 2; d++) v += h90[(i + d + 90) % 90]; if (v > bv) { bv = v; best = i; } }
        float psi = (best + 0.5f) * PI / 180;
        for (auto& u : up) {
            float k = std::round((u.th - psi) / (PI / 2)), sn = psi + k * PI / 2;
            if (angdiff(u.th, sn) < 12 * PI / 180) u.th = sn;
        }
        for (int wall = 0; wall < 16; wall++) {
            std::unordered_map<uint64_t, float> acc;
            for (auto& u : up) if (!u.used) {
                int ti = (int)std::floor((u.th + PI) / (3 * PI / 180)), di = (int)std::floor((std::cos(u.th) * u.x + std::sin(u.th) * u.z) / 0.04f);
                acc[key(ti, 0, di)] += (float)u.n;
            }
            uint64_t bk = 0; float bw = 0;
            for (auto& [k, w] : acc) if (w > bw) { bw = w; bk = k; }
            if (bw < 60) break;
            int ti = (int)((bk >> 42) & 0x1FFFFF) - 0x100000, di = (int)(bk & 0x1FFFFF) - 0x100000;
            float th = (ti + 0.5f) * 3 * PI / 180 - PI, d0 = (di + 0.5f) * 0.04f;
            float nx = std::cos(th), nz = std::sin(th);
            std::vector<size_t> in;
            for (size_t i = 0; i < up.size(); i++) {
                auto& u = up[i];
                if (u.used || angdiff(u.th, th) > 10 * PI / 180) continue;
                if (std::fabs(nx * u.x + nz * u.z - d0) < 0.12f) in.push_back(i);
            }
            int wsum = 0; for (size_t i : in) wsum += up[i].n;
            for (size_t i : in) up[i].used = true;
            if (wsum < 60) continue;
            // Direction: the members' mean (snapped ones agree exactly); distance: the median.
            float sx = 0, sz = 0;
            for (size_t i : in) { sx += std::cos(up[i].th) * up[i].n; sz += std::sin(up[i].th) * up[i].n; }
            float nl = std::sqrt(sx * sx + sz * sz); nx = sx / nl; nz = sz / nl;
            std::vector<float> ds, ts, ys;
            for (size_t i : in) { ds.push_back(nx * up[i].x + nz * up[i].z); ts.push_back(-nz * up[i].x + nx * up[i].z); ys.push_back(up[i].y); }
            std::sort(ds.begin(), ds.end()); std::sort(ts.begin(), ts.end()); std::sort(ys.begin(), ys.end());
            Plane pl{2, ys[ys.size() * 3 / 100], 0, 0, 0, 0, wsum};
            pl.top = ys[ys.size() * 97 / 100];
            pl.t0 = ts[ts.size() * 3 / 100]; pl.t1 = ts[ts.size() * 97 / 100];
            pl.nx = nx; pl.nz = nz; pl.d = ds[ds.size() / 2];
            // A wall, not a screen or a desk's front: 80 cm long and 60 cm high at least.
            if (pl.t1 - pl.t0 < 0.8f || pl.top - pl.y < 0.6f) continue;
            float ttx = -nz, ttz = nx;
            pl.minx = std::min(pl.d * nx + ttx * pl.t0, pl.d * nx + ttx * pl.t1); pl.maxx = std::max(pl.d * nx + ttx * pl.t0, pl.d * nx + ttx * pl.t1);
            pl.minz = std::min(pl.d * nz + ttz * pl.t0, pl.d * nz + ttz * pl.t1); pl.maxz = std::max(pl.d * nz + ttz * pl.t0, pl.d * nz + ttz * pl.t1);
            // The same wall again (the depth's noise spreads it over 20 cm): merged into the
            // stronger one when parallel, facing the same way, within 30 cm and overlapping.
            bool merged = false;
            for (auto& o : m.planes) {
                if (o.kind != 2 || o.nx * nx + o.nz * nz < 0.985f || std::fabs(o.d - pl.d) > 0.3f) continue;
                if (pl.t0 > o.t1 + 0.3f || o.t0 > pl.t1 + 0.3f) continue;
                o.t0 = std::min(o.t0, pl.t0); o.t1 = std::max(o.t1, pl.t1); o.y = std::min(o.y, pl.y); o.top = std::max(o.top, pl.top); o.count += pl.count;
                float tx2 = -o.nz, tz2 = o.nx;
                o.minx = std::min(o.d * o.nx + tx2 * o.t0, o.d * o.nx + tx2 * o.t1); o.maxx = std::max(o.d * o.nx + tx2 * o.t0, o.d * o.nx + tx2 * o.t1);
                o.minz = std::min(o.d * o.nz + tz2 * o.t0, o.d * o.nz + tz2 * o.t1); o.maxz = std::max(o.d * o.nz + tz2 * o.t0, o.d * o.nz + tz2 * o.t1);
                merged = true;
                break;
            }
            if (!merged) m.planes.push_back(pl);
        }
        // Doors and windows: the cells labelled so on each wall, in pieces along it (a gap of
        // 25 cm starts a new one), at least 30 cm wide.
        for (size_t wi = 0; wi < m.planes.size(); wi++) {
            const Plane& w = m.planes[wi];
            if (w.kind != 2) continue;
            for (uint8_t kind : {(uint8_t)DOOR, (uint8_t)WINDOW}) {
                std::vector<std::array<float, 2>> ty;
                for (auto& u : up) {
                    if (u.cat != kind || std::cos(u.th) * w.nx + std::sin(u.th) * w.nz < 0.9f) continue;
                    if (std::fabs(w.nx * u.x + w.nz * u.z - w.d) > 0.15f) continue;
                    float t = -w.nz * u.x + w.nx * u.z;
                    if (t > w.t0 - 0.1f && t < w.t1 + 0.1f) ty.push_back({t, u.y});
                }
                std::sort(ty.begin(), ty.end());
                size_t a = 0;
                for (size_t i = 1; i <= ty.size(); i++) {
                    if (i < ty.size() && ty[i][0] - ty[i - 1][0] < 0.25f) continue;
                    if (i - a >= 4 && ty[i - 1][0] - ty[a][0] >= 0.3f) {
                        float y0 = 1e9f, y1 = -1e9f;
                        for (size_t k = a; k < i; k++) { y0 = std::min(y0, ty[k][1]); y1 = std::max(y1, ty[k][1]); }
                        if (kind == DOOR) y0 = std::min(y0, m.floor_y);
                        m.openings.push_back({kind, (int)wi, ty[a][0], ty[i - 1][0], y0, y1 + EV});
                    }
                    a = i;
                }
            }
        }
    }
    // Walkable grid over the floor seen.
    if (!m.has_floor) return m;
    const Plane fl = m.planes[0];
    WalkGrid& g = m.walk;
    float w = std::min(14.0f, fl.maxx - fl.minx + 0.4f), h = std::min(14.0f, fl.maxz - fl.minz + 0.4f);
    g.x0 = std::max(fl.minx - 0.2f, (fl.minx + fl.maxx) / 2 - 7); g.z0 = std::max(fl.minz - 0.2f, (fl.minz + fl.maxz) / 2 - 7);
    g.nx = std::max(1, (int)std::ceil(w / g.cell)); g.nz = std::max(1, (int)std::ceil(h / g.cell));
    g.c.assign((size_t)g.nx * g.nz, 0);
    auto cell_of = [&](float x, float z, int& i, int& j) { i = (int)std::floor((x - g.x0) / g.cell); j = (int)std::floor((z - g.z0) / g.cell); return i >= 0 && j >= 0 && i < g.nx && j < g.nz; };
    for (auto& p : lv) { int i, j; if (std::fabs(p.y - m.floor_y) < 0.05f && cell_of(p.x, p.z, i, j)) g.c[j * g.nx + i] = 1; }
    // Small holes in the floor seen (under-sampled): filled when most neighbours are floor.
    for (int pass = 0; pass < 2; pass++) {
        std::vector<uint8_t> c2 = g.c;
        for (int j = 1; j + 1 < g.nz; j++) for (int i = 1; i + 1 < g.nx; i++) {
            if (g.c[j * g.nx + i]) continue;
            int n = 0;
            for (int dj = -1; dj <= 1; dj++) for (int di = -1; di <= 1; di++) n += g.c[(j + dj) * g.nx + i + di] == 1;
            if (n >= 5) c2[j * g.nx + i] = 1;
        }
        g.c.swap(c2);
    }
    std::vector<uint8_t> obst(g.c.size(), 0), hits(g.c.size(), 0);
    for (auto& [k, v] : map) {
        if (!confirmed(v)) continue;
        float y = v.sum[1] / v.n;
        if (y < m.floor_y + 0.10f || y > m.floor_y + 1.8f) continue;
        int i, j;
        if (cell_of(v.sum[0] / v.n, v.sum[2] / v.n, i, j) && hits[j * g.nx + i] < 255) hits[j * g.nx + i]++;
    }
    for (size_t i = 0; i < hits.size(); i++) if (hits[i] >= 2) obst[i] = 1;
    for (auto& p : m.planes) {
        if (p.kind == 1) {
            for (float x = p.minx; x <= p.maxx; x += g.cell / 2) for (float z = p.minz; z <= p.maxz; z += g.cell / 2) { int i, j; if (cell_of(x, z, i, j)) obst[j * g.nx + i] = 1; }
        } else if (p.kind == 2) {
            for (float t = p.t0; t <= p.t1; t += g.cell / 2) {
                float x = p.d * p.nx - p.nz * t, z = p.d * p.nz + p.nx * t; int i, j;
                if (cell_of(x, z, i, j)) obst[j * g.nx + i] = 1;
                // and nothing behind the wall
                for (float b = g.cell / 2; b < 1.5f; b += g.cell / 2) if (cell_of(x - p.nx * b, z - p.nz * b, i, j)) g.c[j * g.nx + i] = 0;
            }
        }
    }
    int r = (int)std::ceil(HUMAN_RADIUS / g.cell);
    for (int j = 0; j < g.nz; j++) for (int i = 0; i < g.nx; i++) {
        if (!obst[j * g.nx + i]) continue;
        for (int dj = -r; dj <= r; dj++) for (int di = -r; di <= r; di++) {
            int a = i + di, b = j + dj;
            if (a < 0 || b < 0 || a >= g.nx || b >= g.nz || di * di + dj * dj > r * r) continue;
            if (g.c[b * g.nx + a] == 1) g.c[b * g.nx + a] = 2;
        }
    }
    return m;
}

/**
 * Surface mesh of the confirmed voxels (naive surface nets): one vertex per grid cell
 * that has both filled and empty corners, at the mean of its crossing edges' midpoints,
 * smoothed once; one quad per filled/empty voxel pair, across the 4 cells around it.
 * Lone voxels (fewer than 2 filled neighbours) are left out. Triangles, x y z nx ny nz.
 */
inline std::vector<float> build_mesh(const std::vector<uint64_t>& keys, size_t max_verts) {
    auto unkey = [](uint64_t k, int& x, int& y, int& z) {
        x = (int)((k >> 42) & 0x1FFFFF) - 0x100000; y = (int)((k >> 21) & 0x1FFFFF) - 0x100000; z = (int)(k & 0x1FFFFF) - 0x100000;
    };
    std::unordered_set<uint64_t> conf(keys.begin(), keys.end());
    std::unordered_set<uint64_t> filled;
    std::vector<uint64_t> order;  // the filled ones, in the given order (nearest first)
    filled.reserve(conf.size());
    for (uint64_t k : keys) {
        int x, y, z; unkey(k, x, y, z);
        int nb = 0;
        for (int dx = -1; dx <= 1 && nb < 2; dx++) for (int dy = -1; dy <= 1 && nb < 2; dy++) for (int dz = -1; dz <= 1 && nb < 2; dz++)
            if ((dx || dy || dz) && conf.count(key(x + dx, y + dy, z + dz))) nb++;
        if (nb >= 2 && filled.insert(k).second) order.push_back(k);
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
    for (uint64_t k : order) {
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
    std::unordered_map<uint64_t, Evid> level, upright;  // plane evidence
    Model model;
    // The mesh is built on its own thread (over a second): depth keeps coming meanwhile.
    std::thread mesher;
    std::atomic<bool> meshing{false};
    std::vector<float> wpts((size_t)(depth::W / 2) * (depth::H / 2) * 3);  // world points, every 2nd pixel
    std::vector<uint8_t> wcat((size_t)(depth::W / 2) * (depth::H / 2));      // their category
    semseg::Map sem; double sem_got = 0;  // the newest labelled RGB image, and when it came

    while (!g_stop) {
        int req = g_request.exchange(0);
        if (req == 2) {
            map.clear(); level.clear(); upright.clear(); model = Model{};
            std::lock_guard<std::mutex> l(g.mu);
            g.points.clear(); g.planes.clear(); g.anchors.clear(); g.voxels = 0; g.version++;
            g.mesh.clear(); g.mesh_version++; g.mesh_tris = 0;
            g.has_floor = false; g.walk = WalkGrid{}; g.walk_version++;
            save_anchors();
        }
        double t_img = 0;
        {
            std::lock_guard<std::mutex> l(g_cams.mu);
            // Every third camera frame (~8 Hz): the GPU is shared with the hands and the glasses.
            if (g_cams.w && g_cams.seq - seen >= 3 && g_cams.px[0].size() >= (size_t)g_cams.w * g_cams.h && g_cams.px[1].size() >= (size_t)g_cams.w * g_cams.h) {
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
            if (std::fabs(hands::g_result.t - t_img) < 0.15) for (auto& hd : hands::g_result.hands) {
                for (int k = 0; k < hands::N_JOINTS; k++) hand_pts.push_back({hd.joints[k][0], hd.joints[k][1], hd.joints[k][2]});
            }
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
            const int GW = depth::W / 2;
            std::fill(wpts.begin(), wpts.end(), NAN);
            std::fill(wcat.begin(), wcat.end(), (uint8_t)0);
            {
                std::lock_guard<std::mutex> l(semseg::g_mu);
                if (semseg::g_map.seq != sem.seq) { sem = semseg::g_map; sem_got = hands::qpc_s(); }
            }
            bool sem_ok = !sem.cat.empty() && hands::qpc_s() - sem_got < 1.5;
            for (int v = 0; v < depth::H; v += 2) for (int u = 0; u < depth::W; u += 2) {
                float z = df.depth[v * depth::W + u];
                if (!(z > 0.3f && z < 3.0f)) continue;
                double ph[3], pw[3];
                df.point(u, v, z, ph);
                if (!hand_pts.empty() && near_hand(ph)) continue;
                for (int i = 0; i < 3; i++) pw[i] = head[i * 4] * ph[0] + head[i * 4 + 1] * ph[1] + head[i * 4 + 2] * ph[2] + head[i * 4 + 3];
                float* wp = &wpts[((v / 2) * GW + u / 2) * 3];
                wp[0] = (float)pw[0]; wp[1] = (float)pw[1]; wp[2] = (float)pw[2];
                uint8_t pc = sem_ok ? (uint8_t)sem.at(wp) : 0;
                wcat[(v / 2) * GW + u / 2] = pc;
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
                vote(vx.cat, vx.cat_n, pc);
                if (vx.misses) vx.misses--;
                bool known = false;
                for (int i = 0; i < vx.nviews; i++) if (vx.views[i] == view) known = true;
                if (!known && vx.nviews < 4) vx.views[vx.nviews++] = view;
            }
            // Plane evidence: the normal at each point from its neighbours 4 px right and
            // down (same surface: depth within 4 cm + 3 %), facing the camera; level
            // (n.y > 0.96) or upright (|n.y| < 0.15). Under 2.8 m: farther normals are noise.
            const int GH = depth::H / 2, K = 2;
            float fx = (float)hx, fy = (float)hy, fz = (float)hz;
            for (int j = 0; j + K < GH; j++) for (int i = 0; i + K < GW; i++) {
                const float* a = &wpts[(j * GW + i) * 3];
                const float* b = &wpts[(j * GW + i + K) * 3];
                const float* c2 = &wpts[((j + K) * GW + i) * 3];
                if (std::isnan(a[0]) || std::isnan(b[0]) || std::isnan(c2[0])) continue;
                float da = std::sqrt((a[0] - fx) * (a[0] - fx) + (a[1] - fy) * (a[1] - fy) + (a[2] - fz) * (a[2] - fz));
                if (da > 2.8f) continue;
                float db = std::sqrt((b[0] - fx) * (b[0] - fx) + (b[1] - fy) * (b[1] - fy) + (b[2] - fz) * (b[2] - fz));
                float dc = std::sqrt((c2[0] - fx) * (c2[0] - fx) + (c2[1] - fy) * (c2[1] - fy) + (c2[2] - fz) * (c2[2] - fz));
                float tol = 0.04f + 0.03f * da;
                if (std::fabs(db - da) > tol || std::fabs(dc - da) > tol) continue;
                float e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, e2[3] = {c2[0] - a[0], c2[1] - a[1], c2[2] - a[2]};
                float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
                float nl = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                if (nl < 1e-8f) continue;
                for (float& x : n) x /= nl;
                if (n[0] * (fx - a[0]) + n[1] * (fy - a[1]) + n[2] * (fz - a[2]) < 0) for (float& x : n) x = -x;
                std::unordered_map<uint64_t, Evid>* dst = n[1] > 0.96f ? &level : std::fabs(n[1]) < 0.15f ? &upright : nullptr;
                if (!dst || dst->size() > 300000) continue;
                Evid& e = (*dst)[ekey(a[0], a[1], a[2])];
                if (e.n >= 2000) continue;
                for (int q = 0; q < 3; q++) e.sum[q] += a[q];
                e.nx += n[0]; e.nz += n[2]; e.n++;
                vote(e.cat, e.cat_n, wcat[j * GW + i]);
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
        bool remesh = refresh && now - last_mesh > 4 && !meshing;
        std::vector<Plane> planes;
        std::vector<float> pts;
        size_t purged = 0;
        if (refresh) {
            last_planes = now;
            double tr0 = hands::qpc_s();
            // Unconfirmed voxels not seen for 40 depth maps (5 s): noise, forgotten, so the
            // map does not fill up with it.
            for (auto it = map.begin(); it != map.end();) {
                if (!confirmed(it->second) && frame_id - it->second.last_frame > 40) { it = map.erase(it); purged++; }
                else ++it;
            }
            model = room_model(level, upright, map, head_min_y);
            planes = model.planes;
            if (remesh) {
                // Confirmed voxels within MESH_RANGE, nearest first, off the floor and the walls
                // (the planes hide what is behind them).
                last_mesh = now;
                std::vector<std::pair<float, uint64_t>> cand;
                for (auto& [k, v] : map) {
                    if (!confirmed(v)) continue;
                    float x = v.sum[0] / v.n, y = v.sum[1] / v.n, z = v.sum[2] / v.n;
                    float dd = (x - head[3]) * (x - head[3]) + (y - head[7]) * (y - head[7]) + (z - head[11]) * (z - head[11]);
                    if (dd > MESH_RANGE * MESH_RANGE) continue;
                    if (model.has_floor && y < model.floor_y + 0.06f) continue;
                    bool on_wall = false;
                    for (auto& p : model.planes) if (p.kind == 2 && y > p.y - 0.05f && y < p.top + 0.05f) {
                        float t = -p.nz * x + p.nx * z;
                        if (t > p.t0 - 0.05f && t < p.t1 + 0.05f && std::fabs(p.nx * x + p.nz * z - p.d) < 0.06f) { on_wall = true; break; }
                    }
                    if (!on_wall) cand.push_back({dd, k});
                }
                std::sort(cand.begin(), cand.end());
                std::vector<uint64_t> keys(cand.size());
                for (size_t i = 0; i < cand.size(); i++) keys[i] = cand[i].second;
                if (mesher.joinable()) mesher.join();
                meshing = true;
                mesher = std::thread([keys = std::move(keys), &meshing]() {
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                    double t0 = hands::qpc_s();
                    std::vector<float> mesh = build_mesh(keys, MAX_MESH_VERTS);
                    hands::logf("room: mesh %zu vertices from %zu voxels in %.0f ms\n", mesh.size() / 6, keys.size(), (hands::qpc_s() - t0) * 1000);
                    { std::lock_guard<std::mutex> l(g.mu); g.mesh_tris = mesh.size() / 18; g.mesh = std::move(mesh); g.mesh_version++; }
                    meshing = false;
                });
            }
            double tr1 = hands::qpc_s();
            // Drawing snapshot: confirmed voxels, coloured by height, at most 120k.
            size_t n_conf = 0;
            for (auto& [k, v] : map) if (confirmed(v)) n_conf++;
            int step = (int)std::max<size_t>(1, n_conf / 40000);
            int i = 0;
            float floor_y = model.floor_y;
            // Coloured by category where the segmentation saw it, else by height.
            for (auto& [k, v] : map) {
                if (!confirmed(v) || (i++ % step)) continue;
                float x = v.sum[0] / v.n, y = v.sum[1] / v.n, z = v.sum[2] / v.n;
                if (v.cat_n && v.cat) { auto& cc = semseg::CAT_COLOR[v.cat]; pts.insert(pts.end(), {x, y, z, cc[0], cc[1], cc[2]}); continue; }
                float t = std::clamp((y - floor_y) / 2.0f, 0.0f, 1.0f);
                pts.insert(pts.end(), {x, y, z, 0.2f + 0.8f * t, 0.5f + 0.3f * (1 - t), 1.0f - 0.7f * t});
            }
            int walls = 0, tables = 0; for (auto& p : planes) { walls += p.kind == 2; tables += p.kind == 1; }
            hands::logf("room: %zu voxels (%zu forgotten), evidence %zu level %zu upright, floor %s %.2f m %s(head %.2f), %d tables, %d walls, %zu openings, %zu labelled, walkable %d cells, %.0f ms, snapshot %.0f ms\n",
                        map.size(), purged, level.size(), upright.size(), model.has_floor ? "at" : "not seen, guessed at", model.floor_y, model.floor_source.c_str(), head[7], tables, walls,
                        model.openings.size(), model.labelled,
                        (int)std::count(model.walk.c.begin(), model.walk.c.end(), (uint8_t)1), (tr1 - tr0) * 1000, (hands::qpc_s() - tr1) * 1000);
        }
        std::vector<float> live;
        if (g_scan) {
            for (size_t i = 0; i + 2 < wpts.size(); i += 6) if (!std::isnan(wpts[i])) live.insert(live.end(), {wpts[i], wpts[i + 1], wpts[i + 2], 0.95f, 0.95f, 0.95f});
        }
        std::vector<float> ev;
        if (refresh) {
            auto col = [](const Evid& e, float r, float g2, float b) { if (e.cat_n && e.cat) { auto& cc = semseg::CAT_COLOR[e.cat]; return std::array<float, 3>{cc[0], cc[1], cc[2]}; } return std::array<float, 3>{r, g2, b}; };
            for (auto& [k, e] : level) if (e.n >= 3) { auto cc = col(e, 0.2f, 0.9f, 0.3f); ev.insert(ev.end(), {e.sum[0] / e.n, e.sum[1] / e.n, e.sum[2] / e.n, cc[0], cc[1], cc[2]}); }
            for (auto& [k, e] : upright) if (e.n >= 3) { auto cc = col(e, 1.0f, 0.55f, 0.15f); ev.insert(ev.end(), {e.sum[0] / e.n, e.sum[1] / e.n, e.sum[2] / e.n, cc[0], cc[1], cc[2]}); }
        }
        std::lock_guard<std::mutex> l(g.mu);
        if (!live.empty()) { g.dbg_live.swap(live); g.live_version++; }
        if (refresh) { g.dbg_evidence.swap(ev); g.dbg_version++; }
        g.depth = df.depth;
        g.valid_ratio = df.valid_ratio;
        g.ms = (float)((t1 - t0) * 1000);
        g.voxels = map.size();
        if (now - rate_t0 > 1) { g.rate = (float)(frames / (now - rate_t0)); frames = 0; rate_t0 = now; }
        if (refresh) {
            g.planes = planes; g.points = std::move(pts); g.version++;
            g.has_floor = model.has_floor; g.floor_y = model.floor_y; g.walk = model.walk; g.walk_version++;
            g.openings = model.openings; g.labelled = model.labelled; g.floor_source = model.floor_source;
        }
    }
    if (mesher.joinable()) mesher.join();
}

inline void start(const calib::Rig& rig, PoseFn pose_at, const std::string& anchor_path) {
    g_anchor_path = anchor_path;
    g_thread = std::thread(run, rig, pose_at);
}
inline void stop() { g_stop = true; if (g_thread.joinable()) g_thread.join(); }

}  // namespace room
