// The room, scanned over time: every depth map (depth.hpp) is put into the VITURE
// tracking frame with the head pose at the image's time and accumulated in a voxel map
// (2.5 cm), which fills in as the wearer looks around. From it: the floor and the other
// horizontal surfaces (tables, shelves), so virtual objects can rest on them, and
// spatial anchors: a point on a real surface, kept in the tracking frame.
//
// The tracking frame is the engine's: it does not drift and does not move on recentre,
// but it starts anew at each launch. Anchors are written to myviture_anchors.txt for a
// later relocalisation; they are not loaded back yet.

#pragma once
#include <atomic>
#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "depth.hpp"
#include "hands.hpp"
#include "state.hpp"

namespace room {

constexpr float VOX = 0.025f;
constexpr int CONFIRM = 5;          // observations a voxel needs, from VIEWS different viewpoints
constexpr int VIEWS = 3;
constexpr size_t MAX_VOXELS = 400000;

/** A voxel: mean of the points seen in it, how often, from how many viewpoints; misses when a ray passed through it. */
struct Voxel { float sum[3] = {0, 0, 0}; uint16_t n = 0; uint8_t misses = 0; uint32_t last_frame = 0; uint32_t views[4] = {}; uint8_t nviews = 0; };
inline bool confirmed(const Voxel& v) { return v.n >= CONFIRM && v.nviews >= VIEWS; }

struct Plane {
    int kind;                       // 0 floor, 1 horizontal surface
    float y;                        // height (tracking frame)
    float minx, maxx, minz, maxz;   // extent (5th..95th percentiles of its voxels)
    int count;
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
inline std::vector<Plane> find_planes(const std::unordered_map<uint64_t, Voxel>& map) {
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
            if (frac < 0.01f || c < 300) continue;
            floor_bin = b;
            Plane pl{0, 0, 0, 0, 0, 0, 0};
            extent(b, pl);
            out.push_back(pl);
            continue;
        }
        if (b - floor_bin < 8 || frac < 0.03f || c < 600) continue;
        if (!out.empty() && out.back().kind == 1 && (b * BIN + ymin) - out.back().y < 0.08f) continue;  // same surface
        Plane pl{1, 0, 0, 0, 0, 0, 0};
        extent(b, pl);
        // A surface, not a wall's band: wide in both directions.
        if (pl.maxx - pl.minx > 0.3f && pl.maxz - pl.minz > 0.3f) out.push_back(pl);
    }
    return out;
}

inline void run(calib::Rig rig, PoseFn pose_at) {
    depth::Stereo stereo(rig);
    if (!stereo.ok) { std::lock_guard<std::mutex> l(g.mu); g.status = "profondeur GPU : " + stereo.error; return; }
    { std::lock_guard<std::mutex> l(g.mu); g.status = ""; }
    std::unordered_map<uint64_t, Voxel> map;
    map.reserve(MAX_VOXELS);
    uint32_t seen = 0, frame_id = 0;
    std::vector<uint8_t> img[2];
    int w = 0, h = 0;
    double last_planes = 0, rate_t0 = hands::qpc_s(); int frames = 0;
    depth::Frame df;

    while (!g_stop) {
        int req = g_request.exchange(0);
        if (req == 2) {
            map.clear();
            std::lock_guard<std::mutex> l(g.mu);
            g.points.clear(); g.planes.clear(); g.anchors.clear(); g.voxels = 0; g.version++;
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
        std::vector<Plane> planes;
        std::vector<float> pts;
        if (refresh) {
            last_planes = now;
            planes = find_planes(map);
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
        }
        std::lock_guard<std::mutex> l(g.mu);
        g.depth = df.depth;
        g.valid_ratio = df.valid_ratio;
        g.ms = (float)((t1 - t0) * 1000);
        g.voxels = map.size();
        if (now - rate_t0 > 1) { g.rate = (float)(frames / (now - rate_t0)); frames = 0; rate_t0 = now; }
        if (refresh) { g.planes = planes; g.points = std::move(pts); g.version++; }
    }
}

inline void start(const calib::Rig& rig, PoseFn pose_at, const std::string& anchor_path) {
    g_anchor_path = anchor_path;
    g_thread = std::thread(run, rig, pose_at);
}
inline void stop() { g_stop = true; if (g_thread.joinable()) g_thread.join(); }

}  // namespace room
