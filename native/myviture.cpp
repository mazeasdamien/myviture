// myViture native: the VITURE Luma Ultra as a Windows application.
//
// The VITURE engine runs in this process (vio.hpp); a borderless window covers the
// glasses' screen in side-by-side 3D and is drawn with Direct3D 11 on a flip-model swap
// chain, so the compositor can scan it out directly. Each frame waits for the swap
// chain, reads the newest pose, extrapolates its rotation to the moment the frame is
// shown, and draws a world-locked test scene. DXGI frame statistics give, for every
// displayed frame, the time from the pose read to the vblank, and the missed vblanks.
//
//   myviture.exe [--sdk DIR] [--seconds N] [--probe]
//   A dashboard window (Dear ImGui, dashboard.cpp) opens on the PC's screen.
//   Keys (glasses window focused): Esc quit, R recenter, + / - prediction, P pause stats.

#include "vio.hpp"
#include "dashboard.hpp"
#include "orange.hpp"
#include "rgbcam.hpp"
#include "calib.hpp"
#include "hands.hpp"
#include "align.hpp"
#include "state.hpp"

#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <shellscalingapi.h>
#include <timeapi.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "shcore.lib")

// ------------------------------------------------------------------ time

static double qpc_freq() {
    static double f = [] { LARGE_INTEGER q; QueryPerformanceFrequency(&q); return (double)q.QuadPart; }();
    return f;
}
static int64_t qpc_now() { LARGE_INTEGER q; QueryPerformanceCounter(&q); return q.QuadPart; }
static double qpc_ms(int64_t ticks) { return ticks * 1000.0 / qpc_freq(); }

// ------------------------------------------------------------------ math (row-major, column vectors)

struct V3 { float x, y, z; };
struct Q { float x, y, z, w; };
struct M4 { float m[16]; float& operator()(int r, int c) { return m[r * 4 + c]; } float operator()(int r, int c) const { return m[r * 4 + c]; } };

static M4 identity() { M4 r{}; r(0, 0) = r(1, 1) = r(2, 2) = r(3, 3) = 1; return r; }
static M4 mul(const M4& a, const M4& b) {
    M4 r{};
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { float s = 0; for (int k = 0; k < 4; k++) s += a(i, k) * b(k, j); r(i, j) = s; }
    return r;
}
static M4 translate(V3 t) { M4 r = identity(); r(0, 3) = t.x; r(1, 3) = t.y; r(2, 3) = t.z; return r; }
static M4 scale(float s) { M4 r = identity(); r(0, 0) = r(1, 1) = r(2, 2) = s; return r; }
static M4 rot_y(float a) { M4 r = identity(); float c = cosf(a), s = sinf(a); r(0, 0) = c; r(0, 2) = s; r(2, 0) = -s; r(2, 2) = c; return r; }
static Q qmul(Q a, Q b) {
    return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}
static Q qconj(Q a) { return {-a.x, -a.y, -a.z, a.w}; }
static Q qnorm(Q a) { float n = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w); return {a.x / n, a.y / n, a.z / n, a.w / n}; }
static M4 qmat(Q q) {
    M4 r = identity();
    float x = q.x, y = q.y, z = q.z, w = q.w;
    r(0, 0) = 1 - 2 * (y * y + z * z); r(0, 1) = 2 * (x * y - z * w); r(0, 2) = 2 * (x * z + y * w);
    r(1, 0) = 2 * (x * y + z * w); r(1, 1) = 1 - 2 * (x * x + z * z); r(1, 2) = 2 * (y * z - x * w);
    r(2, 0) = 2 * (x * z - y * w); r(2, 1) = 2 * (y * z + x * w); r(2, 2) = 1 - 2 * (x * x + y * y);
    return r;
}
static Q mat_q(const M4& m) {
    float tr = m(0, 0) + m(1, 1) + m(2, 2);
    Q q;
    if (tr > 0) { float s = sqrtf(tr + 1) * 2; q = {(m(2, 1) - m(1, 2)) / s, (m(0, 2) - m(2, 0)) / s, (m(1, 0) - m(0, 1)) / s, s / 4}; }
    else if (m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2)) { float s = sqrtf(1 + m(0, 0) - m(1, 1) - m(2, 2)) * 2; q = {s / 4, (m(0, 1) + m(1, 0)) / s, (m(0, 2) + m(2, 0)) / s, (m(2, 1) - m(1, 2)) / s}; }
    else if (m(1, 1) > m(2, 2)) { float s = sqrtf(1 + m(1, 1) - m(0, 0) - m(2, 2)) * 2; q = {(m(0, 1) + m(1, 0)) / s, s / 4, (m(1, 2) + m(2, 1)) / s, (m(0, 2) - m(2, 0)) / s}; }
    else { float s = sqrtf(1 + m(2, 2) - m(0, 0) - m(1, 1)) * 2; q = {(m(0, 2) + m(2, 0)) / s, (m(1, 2) + m(2, 1)) / s, s / 4, (m(1, 0) - m(0, 1)) / s}; }
    return qnorm(q);
}
/** Rigid inverse (rotation + translation). */
static M4 rigid_inverse(const M4& a) {
    M4 r = identity();
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) r(i, j) = a(j, i);
    for (int i = 0; i < 3; i++) r(i, 3) = -(r(i, 0) * a(0, 3) + r(i, 1) * a(1, 3) + r(i, 2) * a(2, 3));
    return r;
}
/** General 4x4 inverse (Gauss-Jordan); the display correction can scale. */
static M4 invert(const M4& a) {
    double m[4][8];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) m[i][j] = j < 4 ? a(i, j) : (j - 4 == i);
    for (int c = 0; c < 4; c++) {
        int p = c;
        for (int r = c + 1; r < 4; r++) if (fabs(m[r][c]) > fabs(m[p][c])) p = r;
        for (int j = 0; j < 8; j++) std::swap(m[c][j], m[p][j]);
        double d = m[c][c];
        if (fabs(d) < 1e-12) return identity();
        for (int j = 0; j < 8; j++) m[c][j] /= d;
        for (int r = 0; r < 4; r++) if (r != c) { double f = m[r][c]; for (int j = 0; j < 8; j++) m[r][j] -= f * m[c][j]; }
    }
    M4 out;
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) out(i, j) = (float)m[i][j + 4];
    return out;
}

/** Right-handed perspective, depth 0..1 (Direct3D). */
static M4 perspective(float vfov_deg, float aspect, float n, float f) {
    M4 r{};
    float y = 1.0f / tanf(vfov_deg * 3.14159265f / 360.0f);
    r(0, 0) = y / aspect; r(1, 1) = y; r(2, 2) = f / (n - f); r(2, 3) = n * f / (n - f); r(3, 2) = -1;
    return r;
}

// ------------------------------------------------------------------ pose sampling

// A 1 kHz thread keeps the last second of engine poses, so the render thread can take
// the newest one and an angular velocity over the last ~15 ms.
struct PoseSample { int64_t qpc; Q q; V3 p; };
static std::mutex g_pose_mu;
static std::vector<PoseSample> g_poses;  // ring, newest last
static std::atomic<bool> g_quit{false};
static std::atomic<long> g_pose_updates{0};
/** Engine calls from the pose and render threads, one at a time. */
static std::mutex g_vio_mu;

// Device clock (seconds) from the IMU callbacks (1 kHz), mapped to QPC: the engine's
// get_gl_pose takes a device timestamp and predicts the pose at that time.
static std::atomic<double> g_imu_t{0};
static std::atomic<int64_t> g_imu_qpc{0};
static double device_now() {
    double t = g_imu_t;
    return t > 0 ? t + qpc_ms(qpc_now() - g_imu_qpc) / 1000.0 : 0;
}

/** get_gl_pose behind SEH: a bad timestamp must not take the app down while probing. */
static int safe_get_pose(float* m, double t) {
    __try { return vio::api.get_gl_pose(m, t); } __except (EXCEPTION_EXECUTE_HANDLER) { return -1000; }
}

// get_gl_pose writes past the 4x4 matrix it is documented to fill (a 16-float buffer
// ends the process with a stack buffer overrun): give it room. See probe_clocks.
constexpr int POSE_FLOATS = 64;

/** Engine pose at device time t (0: its latest visual estimate), row-major. */
static float g_last_status = 0;
static bool engine_pose(double t, M4& r) {
    float m[POSE_FLOATS];
    m[30] = -1;
    int rc;
    { std::lock_guard<std::mutex> l(g_vio_mu); rc = safe_get_pose(m, t); }
    if (rc) return false;
    g_last_status = m[30];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) r(i, j) = m[j * 4 + i];  // column-major -> row-major
    return true;
}

/** Which clock the engine's timestamp is: try each, see which answers and moves. */
static void probe_clocks() {
    Sleep(6000);  // let the engine initialise its tracking and predictor
    double imu = g_imu_t, qpc_s = qpc_now() / qpc_freq(), tick_s = GetTickCount64() / 1000.0;
    printf("clocks: imu %.6f  qpc %.6f  tick %.3f\n", imu, qpc_s, tick_s);
    struct Try { const char* name; double t; };
    Try tries[] = {{"0", 0}, {"imu", imu}, {"imu+20ms", imu + 0.02}, {"qpc", qpc_s}, {"qpc+20ms", qpc_s + 0.02}, {"imu ms", imu * 1000}, {"qpc ms", qpc_s * 1000}};
    // How far the engine writes: fill with a sentinel, find the last float changed.
    {
        float m[POSE_FLOATS];
        for (float& v : m) v = -12345.0f;
        safe_get_pose(m, 0);
        int last = -1;
        for (int i = 0; i < POSE_FLOATS; i++) if (m[i] != -12345.0f) last = i;
        printf("get_gl_pose wrote %d floats:", last + 1);
        for (int i = 16; i <= last; i++) printf(" %.4f", m[i]);
        printf("\n");
    }
    // Past, present and future of the newest IMU sample: which ones the predictor serves.
    for (double off : {-0.010, -0.002, 0.0, 0.002, 0.010, 0.020, 0.040}) {
        float m[POSE_FLOATS] = {};
        double t = (double)g_imu_t + off;
        int rc = safe_get_pose(m, t);
        printf("  imu%+5.0f ms -> rc %d  extra %.4f %.4f %.4f | %.4f %.4f %.4f  status %.0f %.0f  pos %.4f %.4f %.4f\n", off * 1000, rc,
               m[16], m[17], m[18], m[19], m[20], m[21], m[30], m[31], m[12], m[13], m[14]);
    }
    float ref[POSE_FLOATS] = {};
    safe_get_pose(ref, 0);
    for (auto& t : tries) {
        float m[POSE_FLOATS] = {};
        int rc = safe_get_pose(m, t.t);
        double diff = 0;
        for (int i = 0; i < 16; i++) diff += fabs(m[i] - ref[i]);
        printf("  t=%-9s %.6f -> rc %d, differs from latest by %.5f  (t %.3f %.3f %.3f, m15 %.2f, m0 %.3f)\n", t.name, t.t, rc, diff, m[12], m[13], m[14], m[15], m[0]);
    }
}

/** How often the engine's answer changes when asked every millisecond. */
static double unique_rate(bool predicted) {
    M4 last{}, r;
    int changes = 0;
    int64_t t0 = qpc_now();
    for (int i = 0; i < 300; i++) {
        if (engine_pose(predicted ? device_now() : 0.0, r) && memcmp(&r, &last, sizeof r)) { changes++; last = r; }
        Sleep(1);
    }
    return changes * 1000.0 / qpc_ms(qpc_now() - t0);
}

static std::atomic<bool> g_engine_predict{true};

static void pose_thread() {
    M4 r, last{};
    while (!g_quit) {
        // At the newest IMU sample's own time: the engine cannot integrate past it.
        if (engine_pose(g_engine_predict ? (double)g_imu_t : 0.0, r) && memcmp(&r, &last, sizeof r)) {
            last = r;
            PoseSample s{qpc_now(), mat_q(r), {r(0, 3), r(1, 3), r(2, 3)}};
            std::lock_guard<std::mutex> l(g_pose_mu);
            g_poses.push_back(s);
            if (g_poses.size() > 2000) g_poses.erase(g_poses.begin(), g_poses.begin() + 1000);
            g_pose_updates++;
        }
        Sleep(1);
    }
}

/** Newest pose with its rotation extrapolated by lead_ms; false before the first pose. */
static bool predicted_pose(double lead_ms, Q& q, V3& p, float& rate_dps) {
    std::lock_guard<std::mutex> l(g_pose_mu);
    if (g_poses.empty()) return false;
    const PoseSample& now = g_poses.back();
    q = now.q; p = now.p; rate_dps = 0;
    // Oldest sample at least 15 ms back: angular velocity from the rotation between them.
    const PoseSample* old = nullptr;
    for (auto it = g_poses.rbegin(); it != g_poses.rend(); ++it) if (qpc_ms(now.qpc - it->qpc) >= 15) { old = &*it; break; }
    if (!old) return true;
    double dt = qpc_ms(now.qpc - old->qpc) / 1000.0;
    if (dt > 0.1) return true;
    Q d = qmul(now.q, qconj(old->q));  // world-frame rotation over dt
    if (d.w < 0) d = {-d.x, -d.y, -d.z, -d.w};
    float s = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
    float angle = 2 * atan2f(s, d.w);
    if (s < 1e-7f) return true;
    float rate = (float)(angle / dt);  // rad/s
    rate_dps = rate * 57.2958f;
    // Lead from the newest sample's time, not from now: the sample is already a little old.
    double lead = (lead_ms + qpc_ms(qpc_now() - now.qpc)) / 1000.0;
    float a = rate * (float)lead;
    Q ext = {d.x / s * sinf(a / 2), d.y / s * sinf(a / 2), d.z / s * sinf(a / 2), cosf(a / 2)};
    q = qnorm(qmul(ext, now.q));
    return true;
}

// ------------------------------------------------------------------ the glasses' monitor

struct Monitor { HMONITOR h; RECT rect; std::wstring gdi; std::wstring name; bool primary; };

static std::vector<Monitor> monitors() {
    std::vector<Monitor> out;
    EnumDisplayMonitors(nullptr, nullptr, [](HMONITOR h, HDC, LPRECT, LPARAM lp) -> BOOL {
        MONITORINFOEXW mi{}; mi.cbSize = sizeof mi;
        GetMonitorInfoW(h, &mi);
        reinterpret_cast<std::vector<Monitor>*>(lp)->push_back({h, mi.rcMonitor, mi.szDevice, L"", (mi.dwFlags & MONITORINFOF_PRIMARY) != 0});
        return TRUE;
    }, reinterpret_cast<LPARAM>(&out));
    // Friendly names ("VITURE") come from the display configuration, keyed by GDI name.
    UINT32 np = 0, nm = 0;
    if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &np, &nm) == ERROR_SUCCESS) {
        std::vector<DISPLAYCONFIG_PATH_INFO> paths(np);
        std::vector<DISPLAYCONFIG_MODE_INFO> modes(nm);
        if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &np, paths.data(), &nm, modes.data(), nullptr) == ERROR_SUCCESS) {
            for (UINT32 i = 0; i < np; i++) {
                DISPLAYCONFIG_SOURCE_DEVICE_NAME src{}; src.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, sizeof src, paths[i].sourceInfo.adapterId, paths[i].sourceInfo.id};
                DISPLAYCONFIG_TARGET_DEVICE_NAME tgt{}; tgt.header = {DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME, sizeof tgt, paths[i].targetInfo.adapterId, paths[i].targetInfo.id};
                if (DisplayConfigGetDeviceInfo(&src.header) || DisplayConfigGetDeviceInfo(&tgt.header)) continue;
                for (auto& m : out) if (m.gdi == src.viewGdiDeviceName) m.name = tgt.monitorFriendlyDeviceName;
            }
        }
    }
    return out;
}

/** Named VITURE, else the first non-primary monitor. */
static bool glasses_monitor(Monitor& out) {
    auto all = monitors();
    for (auto& m : all) if (m.name.find(L"VITURE") != std::wstring::npos) { out = m; return true; }
    for (auto& m : all) if (!m.primary) { out = m; return true; }
    return false;
}

static int width(const RECT& r) { return r.right - r.left; }
static int height(const RECT& r) { return r.bottom - r.top; }

// ------------------------------------------------------------------ scene

struct Vertex { float x, y, z, nx, ny, nz; };
struct Constants { M4 mvp; M4 model; float color[4]; float lit[4]; };

static const char* SHADER = R"(
cbuffer C : register(b0) { row_major float4x4 mvp; row_major float4x4 model; float4 color; float4 lit; };
struct VSOut { float4 pos : SV_Position; float3 n : NORMAL; };
VSOut vs(float3 p : POSITION, float3 n : NORMAL) {
  VSOut o; o.pos = mul(mvp, float4(p, 1)); o.n = mul((float3x3)model, n); return o;
}
float4 ps(VSOut i) : SV_Target {
  float k = lit.x > 0.5 ? 0.45 + 0.55 * saturate(dot(normalize(i.n), normalize(float3(0.4, 1.0, 0.6)))) : 1.0;
  return float4(color.rgb * k, 1);
}
)";

static std::vector<Vertex> cube_vertices() {
    std::vector<Vertex> v;
    const float n[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (auto& f : n) {
        // Two unit axes spanning the face: u along the next axis, w = n x u.
        int ax = f[0] != 0 ? 0 : f[1] != 0 ? 1 : 2;
        float u[3] = {0, 0, 0};
        u[(ax + 1) % 3] = 1;
        float w[3] = {f[1] * u[2] - f[2] * u[1], f[2] * u[0] - f[0] * u[2], f[0] * u[1] - f[1] * u[0]};
        auto corner = [&](float a, float b) {
            return Vertex{0.5f * (f[0] + a * u[0] + b * w[0]), 0.5f * (f[1] + a * u[1] + b * w[1]), 0.5f * (f[2] + a * u[2] + b * w[2]), f[0], f[1], f[2]};
        };
        Vertex c0 = corner(-1, -1), c1 = corner(1, -1), c2 = corner(1, 1), c3 = corner(-1, 1);
        v.insert(v.end(), {c0, c1, c2, c0, c2, c3});
    }
    return v;
}

/** Floor grid (lines), 4 m square, 0.5 m spacing, at y = 0 in its own frame. */
static std::vector<Vertex> grid_vertices() {
    std::vector<Vertex> v;
    for (int i = -4; i <= 4; i++) {
        float a = i * 0.5f;
        v.push_back({a, 0, -2, 0, 1, 0}); v.push_back({a, 0, 2, 0, 1, 0});
        v.push_back({-2, 0, a, 0, 1, 0}); v.push_back({2, 0, a, 0, 1, 0});
    }
    return v;
}

// ------------------------------------------------------------------ window

static AppState g_app;
static std::string g_align_path;
static bool g_stats_paused = false;

static LRESULT CALLBACK wndproc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_KEYDOWN:
        if (wp == VK_ESCAPE) g_quit = true;
        else if (wp == 'R') g_app.recenter = true;
        else if (wp == VK_OEM_PLUS || wp == VK_ADD) g_app.predict_ms = std::min(60.0f, g_app.predict_ms + 2);
        else if (wp == VK_OEM_MINUS || wp == VK_SUBTRACT) g_app.predict_ms = std::max(0.0f, g_app.predict_ms - 2);
        else if (wp == 'P') g_stats_paused = !g_stats_paused;
        return 0;
    case WM_CLOSE: g_quit = true; return 0;
    case WM_SETCURSOR: SetCursor(nullptr); return TRUE;
    }
    return DefWindowProcW(h, msg, wp, lp);
}

static BOOL WINAPI on_ctrl(DWORD) { g_quit = true; return TRUE; }

// Engine callbacks: only the pose is used for now (polled), the rest comes later.
static void on_pose(float*, double) {}
static void on_vsync(double) {}
static void on_imu(float* d, double t) {
    g_imu_qpc = qpc_now();
    g_imu_t = t;
    std::lock_guard<std::mutex> l(g_imu.mu);
    for (int i = 0; i < 6; i++) {
        auto& a = g_imu.axis[i];
        a.push_back(d[i]);
        if (a.size() > 4000) a.erase(a.begin(), a.begin() + 2000);
    }
    g_imu.count++;
}
static void on_cam(unsigned char* l, unsigned char* r, unsigned char*, unsigned char*, double t, int w, int h) {
    std::lock_guard<std::mutex> lk(g_cams.mu);
    if (l) g_cams.px[0].assign(l, l + (size_t)w * h);
    if (r) g_cams.px[1].assign(r, r + (size_t)w * h);
    g_cams.w = w; g_cams.h = h; g_cams.t = t;
    g_cams.seq++;
    g_cams.count++;
}
static void on_points(void*, double) {}
static void on_event(unsigned char e) { printf("[vio] event %d\n", e); }

#define CHECK(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { fprintf(stderr, "%s failed: 0x%08lx\n", #x, (unsigned long)hr_); return 1; } } while (0)

template <typename T> struct Com {
    T* p = nullptr;
    ~Com() { if (p) p->Release(); }
    T** operator&() { return &p; }
    T* operator->() const { return p; }
    operator T*() const { return p; }
};

// ------------------------------------------------------------------ main

int main(int argc, char** argv) {
    std::string sdk = "C:\\Program Files\\VITURE\\SpaceWalker";
    double run_seconds = 0;  // 0: until Esc
    bool probe_only = false;
    double snapshot_s = 0;  // > 0: save the dashboard as dashboard.png after that many seconds
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--sdk") && i + 1 < argc) sdk = argv[++i];
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) run_seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--probe")) probe_only = true;
        else if (!strcmp(argv[i], "--snapshot") && i + 1 < argc) snapshot_s = atof(argv[++i]);
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    SetConsoleCtrlHandler(on_ctrl, TRUE);
    timeBeginPeriod(1);

    printf("myViture native: starting the VITURE engine...\n");
    if (!vio::start(sdk, on_pose, on_vsync, on_imu, on_cam, on_points, on_event)) return 2;
    g_app.serial = vio::api.get_sn ? vio::api.get_sn() : "?";
    printf("engine running, glasses %s\n", g_app.serial.c_str());
    for (int i = 0; i < 300 && device_now() == 0; i++) Sleep(10);
    if (probe_only) { probe_clocks(); vio::stop(); return 0; }
    // Asked for a timestamp before its tracking has started, the engine ends the process
    // (no exception to catch): wait for a real pose with t = 0 first, then a margin.
    printf("waiting for tracking...\n");
    // Measured: ~4 s after the first IMU sample is too early, 6 s is fine; 8 s for margin.
    int64_t wait0 = qpc_now();
    for (int i = 0; i < 300; i++) {
        M4 r;
        bool ok = engine_pose(0, r);
        if (i % 10 == 0) printf("  %.1f s: rc %s, t %.3f %.3f %.3f\n", qpc_ms(qpc_now() - wait0) / 1000, ok ? "ok" : "err", r(0, 3), r(1, 3), r(2, 3));
        if (ok && fabsf(r(0, 3)) + fabsf(r(1, 3)) + fabsf(r(2, 3)) > 0 && qpc_ms(qpc_now() - wait0) > 8000) break;
        Sleep(100);
    }
    printf("tracking ready\n");
    printf("engine pose changes when polled at 1 kHz: latest %.0f Hz, at device time %.0f Hz\n", unique_rate(false), unique_rate(true));
    std::thread poses(pose_thread);

    // The glasses' screen, switched to side-by-side 3D (3840 wide) through the MCU.
    Monitor mon;
    if (!glasses_monitor(mon)) { fprintf(stderr, "glasses screen not found: Win+P -> Extend\n"); g_quit = true; poses.join(); vio::stop(); return 3; }
    bool switched = false;
    if (width(mon.rect) < 3000) {
        int rc = vio::set_display_mode(vio::MODE_3D);
        printf("display 3D -> %d, waiting for the new resolution...\n", rc);
        switched = rc == 0;
        for (int i = 0; i < 40 && width(mon.rect) < 3000; i++) { Sleep(200); glasses_monitor(mon); }
    }
    printf("glasses screen %ls (%ls): %dx%d at %ld,%ld\n", mon.name.c_str(), mon.gdi.c_str(), width(mon.rect), height(mon.rect), mon.rect.left, mon.rect.top);
    if (width(mon.rect) < 3000) printf("warning: still not 3D, both eyes will not get their own image (press the glasses' 2D/3D button)\n");

    WNDCLASSW wc{}; wc.lpfnWndProc = wndproc; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"myviture"; wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    RegisterClassW(&wc);
    HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"myViture", WS_POPUP | WS_VISIBLE, mon.rect.left, mon.rect.top, width(mon.rect), height(mon.rect), nullptr, nullptr, wc.hInstance, nullptr);
    SetForegroundWindow(hwnd);
    UINT W = width(mon.rect), H = height(mon.rect);

    // Device and a flip-model swap chain, one frame of latency at most.
    Com<ID3D11Device> dev; Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
    CHECK(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &fl, 1, D3D11_SDK_VERSION, &dev, nullptr, &ctx));
    Com<IDXGIDevice> dxdev; CHECK(dev->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxdev));
    Com<IDXGIAdapter> adapter; CHECK(dxdev->GetAdapter(&adapter));
    Com<IDXGIFactory2> factory; CHECK(adapter->GetParent(__uuidof(IDXGIFactory2), (void**)&factory));
    DXGI_SWAP_CHAIN_DESC1 sd{};
    sd.Width = W; sd.Height = H; sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM; sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; sd.BufferCount = 2; sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    Com<IDXGISwapChain1> sc1; CHECK(factory->CreateSwapChainForHwnd(dev, hwnd, &sd, nullptr, nullptr, &sc1));
    factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    Com<IDXGISwapChain2> sc; CHECK(sc1->QueryInterface(__uuidof(IDXGISwapChain2), (void**)&sc));
    Com<IDXGISwapChainMedia> media;
    sc1->QueryInterface(__uuidof(IDXGISwapChainMedia), (void**)&media);
    long comp_counts[4] = {};
    sc->SetMaximumFrameLatency(1);
    HANDLE frame_wait = sc->GetFrameLatencyWaitableObject();

    // Back buffer and depth, rebuilt when the glasses switch between 2D and 3D.
    ID3D11RenderTargetView* rtv = nullptr;
    ID3D11DepthStencilView* dsv = nullptr;
    auto make_targets = [&]() -> HRESULT {
        if (rtv) { rtv->Release(); rtv = nullptr; }
        if (dsv) { dsv->Release(); dsv = nullptr; }
        HRESULT hr = sc->ResizeBuffers(0, W, H, DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);
        if (FAILED(hr)) return hr;
        Com<ID3D11Texture2D> back;
        if (FAILED(hr = sc->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&back))) return hr;
        if (FAILED(hr = dev->CreateRenderTargetView(back, nullptr, &rtv))) return hr;
        D3D11_TEXTURE2D_DESC dd{}; dd.Width = W; dd.Height = H; dd.MipLevels = 1; dd.ArraySize = 1; dd.Format = DXGI_FORMAT_D32_FLOAT; dd.SampleDesc.Count = 1; dd.BindFlags = D3D11_BIND_DEPTH_STENCIL;
        Com<ID3D11Texture2D> depth;
        if (FAILED(hr = dev->CreateTexture2D(&dd, nullptr, &depth))) return hr;
        return dev->CreateDepthStencilView(depth, nullptr, &dsv);
    };
    CHECK(make_targets());

    Com<ID3DBlob> vsb, psb, err;
    if (FAILED(D3DCompile(SHADER, strlen(SHADER), nullptr, nullptr, nullptr, "vs", "vs_5_0", 0, 0, &vsb, &err)) ||
        FAILED(D3DCompile(SHADER, strlen(SHADER), nullptr, nullptr, nullptr, "ps", "ps_5_0", 0, 0, &psb, &err))) {
        fprintf(stderr, "shader: %s\n", err ? (const char*)err->GetBufferPointer() : "?");
        return 1;
    }
    Com<ID3D11VertexShader> vs; CHECK(dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs));
    Com<ID3D11PixelShader> ps; CHECK(dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &ps));
    D3D11_INPUT_ELEMENT_DESC il[] = {{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
                                     {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0}};
    Com<ID3D11InputLayout> layout; CHECK(dev->CreateInputLayout(il, 2, vsb->GetBufferPointer(), vsb->GetBufferSize(), &layout));

    auto make_vb = [&](const std::vector<Vertex>& v, ID3D11Buffer** out) {
        D3D11_BUFFER_DESC bd{}; bd.ByteWidth = (UINT)(v.size() * sizeof(Vertex)); bd.BindFlags = D3D11_BIND_VERTEX_BUFFER; bd.Usage = D3D11_USAGE_IMMUTABLE;
        D3D11_SUBRESOURCE_DATA init{v.data()};
        return dev->CreateBuffer(&bd, &init, out);
    };
    auto cube = cube_vertices(), grid = grid_vertices();
    Com<ID3D11Buffer> cube_vb, grid_vb, cb;
    CHECK(make_vb(cube, &cube_vb));
    CHECK(make_vb(grid, &grid_vb));
    // Hand bones: rewritten every frame.
    const UINT BONE_VERTS = 2 * hands::N_BONES * 2;
    Com<ID3D11Buffer> bone_vb;
    {
        D3D11_BUFFER_DESC bd{}; bd.ByteWidth = BONE_VERTS * sizeof(Vertex); bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        bd.Usage = D3D11_USAGE_DYNAMIC; bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        CHECK(dev->CreateBuffer(&bd, nullptr, &bone_vb));
    }
    D3D11_BUFFER_DESC cbd{}; cbd.ByteWidth = sizeof(Constants); cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; cbd.Usage = D3D11_USAGE_DEFAULT;
    CHECK(dev->CreateBuffer(&cbd, nullptr, &cb));
    D3D11_RASTERIZER_DESC rd{}; rd.FillMode = D3D11_FILL_SOLID; rd.CullMode = D3D11_CULL_NONE; rd.FrontCounterClockwise = TRUE; rd.DepthClipEnable = TRUE;
    Com<ID3D11RasterizerState> rs; CHECK(dev->CreateRasterizerState(&rd, &rs));
    D3D11_DEPTH_STENCIL_DESC dsd{}; dsd.DepthEnable = TRUE; dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL; dsd.DepthFunc = D3D11_COMPARISON_LESS;
    Com<ID3D11DepthStencilState> dss; CHECK(dev->CreateDepthStencilState(&dsd, &dss));

    // Dashboard on the PC's screen (primary monitor, or any that is not the glasses').
    {
        HMONITOR pc = MonitorFromPoint({0, 0}, MONITOR_DEFAULTTOPRIMARY);
        for (auto& m : monitors()) if (m.h != mon.h) { pc = m.h; if (m.primary) break; }
        MONITORINFO mi{}; mi.cbSize = sizeof mi;
        GetMonitorInfoW(pc, &mi);
        char exe[MAX_PATH];
        GetModuleFileNameA(nullptr, exe, MAX_PATH);
        std::string ini = exe;
        ini = ini.substr(0, ini.find_last_of('\\') + 1) + "myviture_ui.ini";
        if (!dash::init(dev, ctx, mi.rcWork, ini)) fprintf(stderr, "dashboard window failed\n");
    }
    g_app.screen_w = W; g_app.screen_h = H;
    rgbcam::start();
    {
        // Hands: factory calibration (cached by the web app's dev server) and the models.
        wchar_t exe[MAX_PATH];
        GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring dir = exe;
        dir = dir.substr(0, dir.find_last_of(L'\\') + 1);
        calib::Rig rig = calib::load(std::string(dir.begin(), dir.end()) + "..\\calibration.bin");
        g_align_path = std::string(dir.begin(), dir.end()) + "myviture_display.txt";
        if (align::load(g_align_path, g_app.align)) printf("display correction loaded (%s)\n", g_align_path.c_str());
        if (rig.ok) hands::start(dir + L"models\\", rig);
        else { std::lock_guard<std::mutex> l(hands::g_mu); hands::g_result.status = "calibration.bin introuvable (dossier du projet)"; }
    }
    {
        char exe[MAX_PATH];
        GetModuleFileNameA(nullptr, exe, MAX_PATH);
        std::string dir = exe;
        dir = dir.substr(0, dir.find_last_of('\\') + 1);
        if (orange::load_config(dir + "..\\.env.local")) orange::refresh_spend();
        else { std::lock_guard<std::mutex> l(orange::g_costs.mu); orange::g_costs.status = ".env.local introuvable (à côté du dossier native)"; }
    }

    // Statistics: pose read time per present, matched to the vblank it was shown at.
    struct Pending { UINT present; int64_t pose_qpc; };
    std::vector<Pending> pending;
    std::vector<double> latencies, cpu_ms, jerk, gain;
    long status_counts[10] = {};
    UINT last_refresh = 0, last_present_seen = 0;
    long missed = 0, shown = 0, last_updates = 0;
    int64_t stats_t0 = qpc_now(), frame_t0 = qpc_now();
    float rate_dps = 0, max_rate = 0;
    M4 recenter = identity();
    bool dash_on = true;

    printf("rendering on the glasses: Esc quit, R recenter, +/- prediction (%.0f ms)\n", g_app.predict_ms);
    while (!g_quit && !g_app.quit) {
        g_engine_predict = g_app.engine_predict;
        const double g_predict_ms = g_app.predict_ms;
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
        WaitForSingleObjectEx(frame_wait, 100, TRUE);
        int64_t t_start = qpc_now();

        // Pose now, rotation extrapolated to when this frame reaches the screen.
        // The engine's prediction; the newest polled pose only if it fails.
        Q q; V3 p;
        bool have = predicted_pose(g_predict_ms, q, p, rate_dps);  // extrapolated, from the ~520 Hz polled poses
        {
            // The engine's own prediction (IMU-rate), for when this frame is on screen.
            M4 r;
            // From the newest IMU sample: its age plus the lead to the screen.
            double base = g_imu_t, target = device_now() + g_predict_ms / 1000.0;
            M4 b;
            bool got_base = engine_pose(base, b);
            if (engine_pose(target, r)) {
                q = mat_q(r); p = {r(0, 3), r(1, 3), r(2, 3)}; have = true;
                status_counts[(int)std::clamp(g_last_status, -1.0f, 8.0f) + 1]++;
                // How far the engine moved the pose ahead, against the head's speed.
                if (got_base && rate_dps > 30) {
                    Q d = qmul(q, qconj(mat_q(b)));
                    float ang = 2 * atan2f(sqrtf(d.x * d.x + d.y * d.y + d.z * d.z), fabsf(d.w)) * 57.2958f;
                    gain.push_back(ang / (rate_dps * (target - base)));
                }
            }
        }
        int64_t pose_qpc = qpc_now();
        // Shake seen by the eye: change of the per-frame rotation step (deg), RMS.
        if (have) {
            static Q prev_q{0, 0, 0, 1};
            static V3 prev_step{0, 0, 0};
            static bool primed = false;
            Q d = qmul(q, qconj(prev_q));
            if (d.w < 0) d = {-d.x, -d.y, -d.z, -d.w};
            V3 step = {2 * d.x * 57.2958f, 2 * d.y * 57.2958f, 2 * d.z * 57.2958f};
            if (primed) {
                float j = sqrtf((step.x - prev_step.x) * (step.x - prev_step.x) + (step.y - prev_step.y) * (step.y - prev_step.y) + (step.z - prev_step.z) * (step.z - prev_step.z));
                jerk.push_back(j);
                g_app.shake_history.push_back(j);
                if (g_app.shake_history.size() > 1200) g_app.shake_history.erase(g_app.shake_history.begin(), g_app.shake_history.begin() + 600);
            }
            prev_q = q; prev_step = step; primed = true;
        }
        max_rate = std::max(max_rate, rate_dps);
        M4 head = mul(translate(p), qmat(q));  // head -> world
        if (have && g_app.recenter) {
            // Scene frame: the head's position and heading at recenter, gravity kept.
            float fx = -head(0, 2), fz = -head(2, 2);
            float yaw = atan2f(-fx, -fz);
            recenter = mul(rot_y(-yaw), translate({-p.x, -p.y, -p.z}));
            g_app.recenter = false;
        }
        M4 head_scene = mul(recenter, head);
        {
            // For the dashboard: heading, elevation, roll and position in the scene frame.
            float fx = -head_scene(0, 2), fy = -head_scene(1, 2), fz = -head_scene(2, 2);
            g_app.yaw = atan2f(-fx, -fz) * 57.2958f;
            g_app.pitch = asinf(std::clamp(fy, -1.0f, 1.0f)) * 57.2958f;
            g_app.roll = atan2f(-head_scene(1, 0), head_scene(1, 1)) * 57.2958f;
            for (int i = 0; i < 3; i++) g_app.pos[i] = head_scene(i, 3);
            g_app.rate_dps = rate_dps;
        }

        // Hands from the tracking thread: head frame at the image's time -> scene, with the
        // engine's pose at that same time (no extrapolation yet).
        struct SceneHand { bool right, pinching; V3 j[hands::N_JOINTS]; };
        std::vector<SceneHand> scene_hands;
        hands::Result hr;
        { std::lock_guard<std::mutex> l(hands::g_mu); if (qpc_now() / qpc_freq() - hands::g_result.t < 0.25) hr.hands = hands::g_result.hands; hr.t = hands::g_result.t; hr.seq = hands::g_result.seq; }
        if (g_app.hands) {
            M4 at;
            if (!hr.hands.empty() && engine_pose(hr.t, at)) {
                M4 to_scene = mul(recenter, at);
                // Motion of each joint in the scene, from one measurement to the next, to
                // bring the ~55 ms old hands to the moment this frame is shown.
                static V3 prev[2][hands::N_JOINTS], vel[2][hands::N_JOINTS];
                static double prev_t[2] = {0, 0};
                static uint32_t prev_seq = 0;
                bool fresh = hr.seq != prev_seq;
                prev_seq = hr.seq;
                double lead = std::clamp(qpc_now() / qpc_freq() + g_predict_ms / 1000.0 - hr.t, 0.0, 0.12);
                for (auto& h : hr.hands) {
                    SceneHand sh{h.right, h.pinching, {}};
                    int s = h.right ? 1 : 0;
                    double dt = hr.t - prev_t[s];
                    for (int k = 0; k < hands::N_JOINTS; k++) {
                        const float* q = h.joints[k];
                        V3 p = {to_scene(0, 0) * q[0] + to_scene(0, 1) * q[1] + to_scene(0, 2) * q[2] + to_scene(0, 3),
                                to_scene(1, 0) * q[0] + to_scene(1, 1) * q[1] + to_scene(1, 2) * q[2] + to_scene(1, 3),
                                to_scene(2, 0) * q[0] + to_scene(2, 1) * q[1] + to_scene(2, 2) * q[2] + to_scene(2, 3)};
                        if (fresh) {
                            if (dt > 0.005 && dt < 0.15) {
                                V3 raw = {(float)((p.x - prev[s][k].x) / dt), (float)((p.y - prev[s][k].y) / dt), (float)((p.z - prev[s][k].z) / dt)};
                                vel[s][k] = {0.5f * vel[s][k].x + 0.5f * raw.x, 0.5f * vel[s][k].y + 0.5f * raw.y, 0.5f * vel[s][k].z + 0.5f * raw.z};
                            } else vel[s][k] = {0, 0, 0};
                            prev[s][k] = p;
                        }
                        if (g_app.hand_predict) p = {p.x + vel[s][k].x * (float)lead, p.y + vel[s][k].y * (float)lead, p.z + vel[s][k].z * (float)lead};
                        sh.j[k] = p;
                    }
                    if (fresh) prev_t[s] = hr.t;
                    scene_hands.push_back(sh);
                }
            }
        }

        // Display calibration: a target fixed to the head, pinched where it is seen.
        static std::vector<align::Pair> align_pairs;
        static double acc[3] = {0, 0, 0};
        static int acc_n = 0;
        static bool was_pinching = false;
        static uint32_t align_seq = 0;
        static const auto targets = align::targets();
        if (g_app.align_reset) {
            g_app.align_reset = false;
            M4 id = identity();
            for (int i = 0; i < 16; i++) g_app.align[i] = id.m[i];
            align::save(g_align_path, g_app.align);
            g_app.align_status = "correction remise à zéro";
        }
        if (g_app.align_start) {
            g_app.align_start = false;
            g_app.align_active = true; g_app.align_index = 0; g_app.align_total = (int)targets.size();
            align_pairs.clear(); acc_n = 0; was_pinching = false;
            g_app.align_status = "";
        }
        if (g_app.align_cancel) { g_app.align_cancel = false; g_app.align_active = false; g_app.align_status = "calibration annulée"; }
        if (g_app.align_active && hr.seq != align_seq) {
            align_seq = hr.seq;
            const hands::Hand* ph = nullptr;
            for (auto& h : hr.hands) if (h.pinching) ph = &h;
            if (ph) {
                for (int i = 0; i < 3; i++) acc[i] += (ph->joints[4][i] + ph->joints[8][i]) / 2;
                acc_n++;
                was_pinching = true;
            } else if (was_pinching) {
                was_pinching = false;
                if (acc_n >= 3) {
                    align::Pair pr;
                    for (int i = 0; i < 3; i++) { pr.t[i] = targets[g_app.align_index][i]; pr.r[i] = acc[i] / acc_n; }
                    align_pairs.push_back(pr);
                    printf("align target %d: drawn %.3f %.3f %.3f  pinched %.3f %.3f %.3f\n", g_app.align_index, pr.t[0], pr.t[1], pr.t[2], pr.r[0], pr.r[1], pr.r[2]);
                    if (++g_app.align_index == g_app.align_total) {
                        g_app.align_active = false;
                        align::Result res = align::solve(align_pairs);
                        char msg[256];
                        snprintf(msg, sizeof msg, "écart %.0f mm · décalage %.0f %.0f %.0f mm · rotation %.1f° · échelle %.3f%s", res.rms_mm,
                                 res.shift_mm[0], res.shift_mm[1], res.shift_mm[2], res.rot_deg, res.scale, res.ok ? "" : " → refusée, recommence");
                        g_app.align_status = msg;
                        printf("align: %s\n", msg);
                        if (res.ok) { for (int i = 0; i < 16; i++) g_app.align[i] = res.M[i]; align::save(g_align_path, g_app.align); }
                    }
                }
                acc[0] = acc[1] = acc[2] = 0; acc_n = 0;
            }
        }
        UINT bone_count = 0;
        if (!scene_hands.empty()) {
            D3D11_MAPPED_SUBRESOURCE m;
            if (SUCCEEDED(ctx->Map(bone_vb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
                Vertex* v = (Vertex*)m.pData;
                for (auto& h : scene_hands) for (auto& b : hands::BONES) {
                    if (bone_count + 2 > BONE_VERTS) break;
                    v[bone_count++] = {h.j[b[0]].x, h.j[b[0]].y, h.j[b[0]].z, 0, 1, 0};
                    v[bone_count++] = {h.j[b[1]].x, h.j[b[1]].y, h.j[b[1]].z, 0, 1, 0};
                }
                ctx->Unmap(bone_vb, 0);
            }
        }

        float clear[4] = {0, 0, 0, 1};
        ctx->ClearRenderTargetView(rtv, clear);
        ctx->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 1, 0);
        ID3D11RenderTargetView* rtvs[] = {rtv};
        ctx->OMSetRenderTargets(1, rtvs, dsv);
        ctx->OMSetDepthStencilState(dss, 0);
        ctx->RSSetState(rs);
        ctx->IASetInputLayout(layout);
        ctx->VSSetShader(vs, nullptr, 0);
        ctx->PSSetShader(ps, nullptr, 0);
        ID3D11Buffer* cbs[] = {cb};
        ctx->VSSetConstantBuffers(0, 1, cbs);
        ctx->PSSetConstantBuffers(0, 1, cbs);
        UINT stride = sizeof(Vertex), offset = 0;

        // Side by side in 3D; in 2D both eyes see the same image: one view, centred.
        const bool sbs = W >= 3000;
        const int views = sbs ? 2 : 1;
        const float ipd = g_app.ipd_mm / 1000.0f;
        if (have) for (int eye = 0; eye < views; eye++) {
            float vw = (float)W / views;
            D3D11_VIEWPORT vp{eye * vw, 0, vw, (float)H, 0, 1};
            ctx->RSSetViewports(1, &vp);
            float off = sbs ? (eye ? ipd / 2 : -ipd / 2) : 0;
            M4 D = identity();
            if (!g_app.align_active) for (int i = 0; i < 16; i++) D.m[i] = (float)g_app.align[i];
            M4 eye_world = mul(mul(head_scene, D), translate({off, 0, 0}));
            M4 vpm = mul(perspective(g_app.vfov, vw / H, 0.05f, 50), invert(eye_world));
            auto draw = [&](ID3D11Buffer* vb, UINT n, D3D11_PRIMITIVE_TOPOLOGY topo, const M4& model, float r, float g, float b, bool lit) {
                Constants c{mul(vpm, model), model, {r, g, b, 1}, {lit ? 1.0f : 0.0f, 0, 0, 0}};
                ctx->UpdateSubresource(cb, 0, nullptr, &c, 0, 0);
                ID3D11Buffer* vbs[] = {vb};
                ctx->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
                ctx->IASetPrimitiveTopology(topo);
                ctx->Draw(n, 0);
            };
            if (g_app.align_active) {
                auto& t = targets[g_app.align_index];
                M4 at = mul(head_scene, translate({(float)t[0], (float)t[1], (float)t[2]}));
                draw(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(at, scale(0.012f)), 0.2f, 1.0f, 0.4f, false);
            }
            // Hands: joints as small cubes, bones as lines (left blue, right orange; white when pinching).
            for (size_t hi = 0; hi < scene_hands.size(); hi++) {
                auto& h = scene_hands[hi];
                float r = h.pinching ? 1.0f : h.right ? 1.0f : 0.3f, g = h.pinching ? 1.0f : h.right ? 0.48f : 0.55f, b = h.pinching ? 1.0f : h.right ? 0.27f : 1.0f;
                for (int k = 0; k < hands::N_JOINTS; k++) draw(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(translate(h.j[k]), scale(k % 4 == 0 ? 0.012f : 0.009f)), r, g, b, true);
            }
            if (bone_count) {
                Constants c{vpm, identity(), {0.9f, 0.9f, 0.9f, 1}, {0, 0, 0, 0}};
                ctx->UpdateSubresource(cb, 0, nullptr, &c, 0, 0);
                ID3D11Buffer* vbs[] = {bone_vb};
                ctx->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
                ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
                ctx->Draw(bone_count, 0);
            }
            if (!g_app.scene) continue;
            // Eight cubes around the head at arm's length, one ahead at 1 m, floor 1.6 m down.
            for (int i = 0; i < 8; i++) {
                float a = i * 3.14159265f / 4;
                float hue = i / 8.0f;
                float r = 0.5f + 0.5f * cosf(6.2832f * hue), g = 0.5f + 0.5f * cosf(6.2832f * (hue - 0.333f)), b = 0.5f + 0.5f * cosf(6.2832f * (hue - 0.667f));
                draw(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(translate({sinf(a) * 0.9f, 0, -cosf(a) * 0.9f}), scale(0.12f)), r, g, b, true);
            }
            draw(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(translate({0, -0.1f, -1.0f}), scale(0.05f)), 1, 1, 1, true);
            if (g_app.floor_grid) draw(grid_vb, (UINT)grid.size(), D3D11_PRIMITIVE_TOPOLOGY_LINELIST, translate({0, -g_app.floor_m, 0}), 0.3f, 0.7f, 1.0f, false);
        }

        sc->Present(1, 0);
        UINT present_id = 0;
        sc->GetLastPresentCount(&present_id);
        pending.push_back({present_id, pose_qpc});
        cpu_ms.push_back(qpc_ms(qpc_now() - t_start));

        // Which present reached the screen, and at which vblank.
        DXGI_FRAME_STATISTICS st{};
        if (SUCCEEDED(sc->GetFrameStatistics(&st)) && st.PresentCount != last_present_seen) {
            // Vblanks that passed without a new frame (signed: the counters can restart).
            int64_t extra = ((int64_t)st.PresentRefreshCount - last_refresh) - ((int64_t)st.PresentCount - last_present_seen);
            if (last_present_seen && extra > 0 && extra < 600) missed += (long)extra;
            shown += st.PresentCount - last_present_seen;
            last_present_seen = st.PresentCount;
            last_refresh = st.PresentRefreshCount;
            for (auto& pd : pending) if (pd.present == st.PresentCount) {
                double l = qpc_ms(st.SyncQPCTime.QuadPart - pd.pose_qpc);
                latencies.push_back(l);
                g_app.latency_history.push_back((float)l);
                if (g_app.latency_history.size() > 1200) g_app.latency_history.erase(g_app.latency_history.begin(), g_app.latency_history.begin() + 600);
            }
            pending.erase(std::remove_if(pending.begin(), pending.end(), [&](const Pending& pd) { return pd.present <= st.PresentCount; }), pending.end());
        }
        if (pending.size() > 120) pending.erase(pending.begin(), pending.begin() + 60);
        // How Windows shows this window: straight to the screen (overlay / independent
        // flip, no added frame) or composed by DWM (one more frame).
        if (media) {
            DXGI_FRAME_STATISTICS_MEDIA sm{};
            if (SUCCEEDED(media->GetFrameStatisticsMedia(&sm))) comp_counts[std::min((int)sm.CompositionMode, 3)]++;
        }

        double since = qpc_ms(qpc_now() - stats_t0);
        if (since > 2000) {
            auto pct = [](std::vector<double> v, double p) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(p * v.size()))]; };
            long updates = g_pose_updates - last_updates;
            g_app.stats = {(float)(shown * 1000.0 / since), (float)pct(latencies, 0.5), (float)pct(latencies, 0.99), (float)pct(cpu_ms, 0.5), (float)(updates * 1000.0 / since),
                           (float)pct(jerk, 0.5), (float)pct(jerk, 0.99), (float)pct(gain, 0.5), missed};
            if (!g_stats_paused)
                printf("%5.1f fps  pose %4.0f Hz  pose->vblank p50 %4.1f p99 %4.1f ms  CPU %4.2f ms  missed %ld  head max %4.0f deg/s  shake p50 %.3f p99 %.3f deg  %s %.0f ms\n",
                       shown * 1000.0 / since, updates * 1000.0 / since, pct(latencies, 0.5), pct(latencies, 0.99), pct(cpu_ms, 0.5), missed, max_rate,
                       pct(jerk, 0.5), pct(jerk, 0.99), g_engine_predict ? "engine" : "ours", g_predict_ms);
            if (!g_stats_paused) printf("      presentation: composed %ld  overlay %ld  direct %ld  failed %ld\n", comp_counts[0], comp_counts[1], comp_counts[2], comp_counts[3]);
            if (!g_stats_paused) {
                std::lock_guard<std::mutex> l(hands::g_mu);
                auto& r = hands::g_result;
                printf("      hands: %zu  %.0f/s  %.0f ms/pair (palm %.0f, %d/s)  image->hand %.0f ms", r.hands.size(), r.rate, r.ms_total, r.ms_palm, r.palm_runs, r.latency_ms);
                for (auto& h : r.hands) printf("  [%s gap %.0f mm pinch %.1f cm wrist %.2f %.2f %.2f]", h.right ? "R" : "L", h.gap_mm, h.pinch_m * 100, h.joints[0][0], h.joints[0][1], h.joints[0][2]);
                printf("\n");
            }
            for (long& c : comp_counts) c = 0;
            if (g_engine_predict && !g_stats_paused) {
                printf("      engine lead used: %.2f of the head's motion (n %zu)  status:", pct(gain, 0.5), gain.size());
                for (int k = 0; k < 10; k++) if (status_counts[k]) printf(" [%d]=%ld", k - 1, status_counts[k]);
                printf("\n");
            }
            gain.clear();
            for (long& c : status_counts) c = 0;
            latencies.clear(); cpu_ms.clear(); jerk.clear(); shown = 0; missed = 0; max_rate = 0; last_updates = g_pose_updates;
            stats_t0 = qpc_now();
        }
        if (dash_on) dash::frame(g_app);
        if (snapshot_s > 0 && qpc_ms(qpc_now() - frame_t0) > snapshot_s * 1000) {
            snapshot_s = 0;
            wchar_t exe[MAX_PATH];
            GetModuleFileNameW(nullptr, exe, MAX_PATH);
            std::wstring png = exe;
            png = png.substr(0, png.find_last_of(L'\\') + 1) + L"dashboard.png";
            g_app.snapshot = true;
            (void)png;
            printf("dashboard snapshot requested\n");
        }

        if (g_app.display_request) {
            // The glasses change resolution: wait for Windows, then follow with the window.
            int want = g_app.display_request;
            g_app.display_request = 0;
            int rc = vio::set_display_mode(want == 3 ? vio::MODE_3D : vio::MODE_2D);
            printf("display %dD -> %d\n", want, rc);
            if (rc == 0) {
                for (int i = 0; i < 40; i++) { Sleep(150); glasses_monitor(mon); if ((width(mon.rect) >= 3000) == (want == 3)) break; }
                W = width(mon.rect); H = height(mon.rect);
                SetWindowPos(hwnd, nullptr, mon.rect.left, mon.rect.top, W, H, SWP_NOZORDER | SWP_NOACTIVATE);
                ctx->OMSetRenderTargets(0, nullptr, nullptr);
                if (FAILED(make_targets())) { fprintf(stderr, "resize failed\n"); break; }
                g_app.screen_w = W; g_app.screen_h = H;
                switched = want == 3;
            }
        }

        if (run_seconds > 0 && qpc_ms(qpc_now() - frame_t0) > run_seconds * 1000) g_quit = true;
    }

    printf("stopping...\n");
    // Whatever hangs on the way out (engine, camera, GPU), the process ends.
    CreateThread(nullptr, 0, [](LPVOID) -> DWORD { Sleep(6000); fflush(stdout); TerminateProcess(GetCurrentProcess(), 0); return 0; }, nullptr, 0, nullptr);
    hands::stop();
    rgbcam::stop();
    dash::shutdown();
    if (rtv) rtv->Release();
    if (dsv) dsv->Release();
    DestroyWindow(hwnd);
    if (switched) vio::set_display_mode(vio::MODE_2D);
    poses.join();
    vio::stop();
    return 0;
}
