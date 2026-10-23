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
#include "rgbcal.hpp"
#include "roommap.hpp"
#include "vhands.hpp"
#include "recorder.hpp"
#include "skinhands.hpp"
#include "textpanel.hpp"
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
static M4 rot_x(float a) { M4 r = identity(); float c = cosf(a), s = sinf(a); r(1, 1) = c; r(1, 2) = -s; r(2, 1) = s; r(2, 2) = c; return r; }
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
/** carina_vio_get_eyes_pose(float* left, float* right, double t) behind SEH (undocumented). */
typedef int (*EyesPoseFn)(float*, float*, double);
static int safe_eyes_pose(EyesPoseFn fn, float* l, float* r, double t) {
    __try { return fn(l, r, t); } __except (EXCEPTION_EXECUTE_HANDLER) { return -1000; }
}

static void probe_eyes() {
    auto fn = (EyesPoseFn)GetProcAddress(GetModuleHandleA("carina_vio.dll"), "carina_vio_get_eyes_pose");
    if (!fn) { printf("eyes: no export\n"); return; }
    float l[64], r[64], head[POSE_FLOATS];
    for (int i = 0; i < 64; i++) l[i] = r[i] = -12345.0f;
    double t = g_imu_t;
    int rc = safe_eyes_pose(fn, l, r, t);
    safe_get_pose(head, t);
    int nl = 0, nr = 0;
    for (int i = 0; i < 64; i++) { if (l[i] != -12345.0f) nl = i + 1; if (r[i] != -12345.0f) nr = i + 1; }
    printf("eyes: rc %d, wrote %d / %d floats\n", rc, nl, nr);
    auto dump = [](const char* name, const float* m, int n) {
        printf("  %s:", name);
        for (int i = 0; i < n; i++) printf("%s%.5f", i % 4 ? " " : (i ? " | " : " "), m[i]);
        printf("\n");
    };
    dump("left ", l, std::max(nl, 16));
    dump("right", r, std::max(nr, 16));
    dump("head ", head, 16);
}

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
// Textured quad (the uv rides in the normal slot).
Texture2D tex0 : register(t0);
SamplerState samp0 : register(s0);
struct VSOutT { float4 pos : SV_Position; float2 uv : TEXCOORD; };
VSOutT vs_tex(float3 p : POSITION, float3 n : NORMAL) { VSOutT o; o.pos = mul(mvp, float4(p, 1)); o.uv = n.xy; return o; }
float4 ps_tex(VSOutT i) : SV_Target { return float4(tex0.Sample(samp0, i.uv).rgb, 1); }
// Contact shadow: a soft dark disc (uv 0..1 across it), strength in color.x, premultiplied.
float4 ps_shadow(VSOutT i) : SV_Target { float r = length(i.uv * 2 - 1); float a = color.x * saturate(1 - r); return float4(0, 0, 0, a * a); }
// The RGB view: the camera frame (t0) under the virtual layer (t1, premultiplied), the layer
// softened like the camera (3x3 blur over lit.xy) and given its grain (color.y, seed color.z).
Texture2D tex1 : register(t1);
float grain_hash(float2 p) { return frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453); }
float4 ps_comp(VSOutT i) : SV_Target {
  float3 bg = tex0.Sample(samp0, i.uv).rgb;
  float2 px = lit.xy;
  float4 v = tex1.Sample(samp0, i.uv) * 0.25;
  v += (tex1.Sample(samp0, i.uv + float2(px.x, 0)) + tex1.Sample(samp0, i.uv - float2(px.x, 0)) + tex1.Sample(samp0, i.uv + float2(0, px.y)) + tex1.Sample(samp0, i.uv - float2(0, px.y))) * 0.125;
  v += (tex1.Sample(samp0, i.uv + px) + tex1.Sample(samp0, i.uv - px) + tex1.Sample(samp0, i.uv + float2(px.x, -px.y)) + tex1.Sample(samp0, i.uv + float2(-px.x, px.y))) * 0.0625;
  float g = (grain_hash(floor(i.uv * lit.zw) + color.z) - 0.5) * color.y;
  return float4(bg * (1 - v.a) + v.rgb + g * v.a, 1);
}
// Instanced: one small cube per scanned voxel (position and colour per instance, size in lit.y).
struct VSOutC { float4 pos : SV_Position; float3 n : NORMAL; float3 col : COLOR; };
VSOutC vs_inst(float3 p : POSITION, float3 n : NORMAL, float3 ip : INSTPOS, float3 ic : INSTCOL) {
  VSOutC o; o.pos = mul(mvp, float4(p * lit.y + ip, 1)); o.n = mul((float3x3)model, n); o.col = ic; return o;
}
float4 ps_inst(VSOutC i) : SV_Target {
  float k = 0.45 + 0.55 * saturate(dot(normalize(i.n), normalize(float3(0.4, 1.0, 0.6))));
  return float4(i.col * k, 1);
}
)";

/** Open cylinder, radius 1, from z = 0 to z = 1, `n` sides (the hands' bones). */
static std::vector<Vertex> cylinder_vertices(int n = 14) {
    std::vector<Vertex> v;
    for (int i = 0; i < n; i++) {
        float a0 = 6.2831853f * i / n, a1 = 6.2831853f * (i + 1) / n;
        float x0 = cosf(a0), y0 = sinf(a0), x1 = cosf(a1), y1 = sinf(a1);
        Vertex p00{x0, y0, 0, x0, y0, 0}, p10{x1, y1, 0, x1, y1, 0}, p01{x0, y0, 1, x0, y0, 0}, p11{x1, y1, 1, x1, y1, 0};
        v.insert(v.end(), {p00, p10, p11, p00, p11, p01});
    }
    return v;
}

/** Sphere, radius 1 (the hands' joints and tips). */
static std::vector<Vertex> sphere_vertices(int seg = 14, int rings = 8) {
    std::vector<Vertex> v;
    auto pt = [&](int i, int j) {
        float th = 3.14159265f * j / rings, ph = 6.2831853f * i / seg;
        float x = sinf(th) * cosf(ph), y = sinf(th) * sinf(ph), z = cosf(th);
        return Vertex{x, y, z, x, y, z};
    };
    for (int j = 0; j < rings; j++) for (int i = 0; i < seg; i++) {
        Vertex a = pt(i, j), b = pt(i + 1, j), c2 = pt(i + 1, j + 1), d = pt(i, j + 1);
        v.insert(v.end(), {a, b, c2, a, c2, d});
    }
    return v;
}

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
static bool g_no_room = false, g_no_rgb = false, g_no_vhands = false;  // test switches: --no-room, --no-rgb
static std::string g_settings_path;
static void save_settings() {
    if (g_settings_path.empty()) return;
    if (FILE* f = fopen(g_settings_path.c_str(), "w")) { fprintf(f, "%.3f %.2f %.3f %.2f %.2f %.2f %.2f\n", g_app.vfov, g_app.ipd_mm, g_app.floor_m, g_app.pitch_deg, g_app.eye_dy_cm, g_app.eye_dz_cm, g_app.yaw_deg); fclose(f); }
}
static bool g_stats_paused = false;

static LRESULT CALLBACK wndproc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_KEYDOWN:
        if (wp == VK_ESCAPE) g_quit = true;
        else if (wp == 'R') g_app.recenter = true;
        else if (wp == VK_OEM_PLUS || wp == VK_ADD) g_app.predict_ms = std::min(40.0f, g_app.predict_ms + 2);
        else if (wp == VK_OEM_MINUS || wp == VK_SUBTRACT) g_app.predict_ms = std::max(0.0f, g_app.predict_ms - 2);
        else if (wp == 'P') g_stats_paused = !g_stats_paused;
        else if (wp == 'A') g_app.room_request = 1;
        return 0;
    case WM_CLOSE: g_quit = true; return 0;
    case WM_SETCURSOR: SetCursor(nullptr); return TRUE;
    }
    return DefWindowProcW(h, msg, wp, lp);
}

static BOOL WINAPI on_ctrl(DWORD) { g_quit = true; return TRUE; }

// Engine callbacks: the pose is polled; its callback and the images also feed VITURE's
// hand tracking.
static void on_pose(float* d, double t) { vhands::feed_pose(d, t); }
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
    vhands::feed_images(l, r, w, h, t);
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
    double record_s = 0;     // > 0: record the RGB view for that long, from 5 s after start (tests)
    bool probe_only = false;
    double snapshot_s = 0;  // > 0: save the dashboard as dashboard.png after that many seconds
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--sdk") && i + 1 < argc) sdk = argv[++i];
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) run_seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--record") && i + 1 < argc) record_s = atof(argv[++i]);
        else if (!strcmp(argv[i], "--probe")) probe_only = true;
        else if (!strcmp(argv[i], "--no-room")) g_no_room = true;
        else if (!strcmp(argv[i], "--no-rgb")) g_no_rgb = true;
        else if (!strcmp(argv[i], "--no-vhands")) g_no_vhands = true;
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
    if (probe_only) { probe_clocks(); probe_eyes(); vio::stop(); return 0; }
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
    {
        // The engine's per-eye poses (its factory display calibration), relative to the head
        // pose at the same time: eye -> head, averaged over a few samples.
        auto fn = (EyesPoseFn)GetProcAddress(GetModuleHandleA("carina_vio.dll"), "carina_vio_get_eyes_pose");
        double acc[2][16] = {};
        int n = 0;
        for (int i = 0; fn && i < 20; i++, Sleep(5)) {
            float l[64], r[64], h[POSE_FLOATS];
            double t = g_imu_t;
            if (safe_eyes_pose(fn, l, r, t) != 0 || safe_get_pose(h, t) != 0) continue;
            M4 H, E[2];
            for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) { H(a, b) = h[b * 4 + a]; E[0](a, b) = l[b * 4 + a]; E[1](a, b) = r[b * 4 + a]; }
            M4 Hi = rigid_inverse(H);
            for (int e = 0; e < 2; e++) { M4 D = mul(Hi, E[e]); for (int k = 0; k < 16; k++) acc[e][k] += D.m[k]; }
            n++;
        }
        if (n >= 5) {
            M4 D[2];
            for (int e = 0; e < 2; e++) {
                for (int k = 0; k < 16; k++) D[e].m[k] = (float)(acc[e][k] / n);
                // Re-orthonormalise the averaged rotation.
                Q q = mat_q(D[e]); M4 R = qmat(q);
                for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) D[e](a, b) = R(a, b);
            }
            memcpy(g_app.eye_l, D[0].m, sizeof D[0].m); memcpy(g_app.eye_r, D[1].m, sizeof D[1].m);
            g_app.have_vendor_eyes = true;
            // The RGB camera's default orientation: the eyes' (same front, same tilt).
            double Rc[9];
            for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) Rc[a * 3 + b] = (D[0](b, a) + D[1](b, a)) / 2;  // head -> camera: transposed
            rgbcal::set_default(Rc);
            printf("RGB camera default tilt from the engine eyes: %.1f deg\n", std::atan2(Rc[5], Rc[4]) * 57.2958);
            printf("engine eyes: left %.1f %.1f %.1f cm, right %.1f %.1f %.1f cm (IPD %.1f mm)\n", D[0](0, 3) * 100, D[0](1, 3) * 100, D[0](2, 3) * 100,
                   D[1](0, 3) * 100, D[1](1, 3) * 100, D[1](2, 3) * 100, std::hypot(D[1](0, 3) - D[0](0, 3), D[1](1, 3) - D[0](1, 3)) * 1000);
        } else printf("engine eyes: not available, manual display model\n");
    }
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

    // The RGB view: the RGB frame, then the virtual scene as the RGB camera saw it at that
    // moment, in one picture (the dashboard's big view, recordings, photos).
    // Its size follows the RGB camera's (made again when that changes).
    UINT MR_W = 0, MR_H = 0;
    Com<ID3D11Texture2D> mr_tex, mr_depth;
    Com<ID3D11RenderTargetView> mr_rtv;
    Com<ID3D11ShaderResourceView> mr_srv;
    Com<ID3D11DepthStencilView> mr_dsv;
    Com<ID3D11Texture2D> rec_staging[3];     // RGB view -> CPU for the recorder, 2 frames behind
    Com<ID3D11Texture2D> virt_tex;           // the virtual alone (premultiplied), before the composite
    Com<ID3D11RenderTargetView> virt_rtv;
    Com<ID3D11ShaderResourceView> virt_srv;
    auto drop = [](auto& c) { if (c.p) { c.p->Release(); c.p = nullptr; } };
    auto make_mr = [&](UINT w, UINT h) {
        drop(mr_tex); drop(mr_depth); drop(mr_rtv); drop(mr_srv); drop(mr_dsv); drop(virt_tex); drop(virt_rtv); drop(virt_srv);
        for (auto& st : rec_staging) drop(st);
        D3D11_TEXTURE2D_DESC td{}; td.Width = w; td.Height = h; td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1; td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        bool ok = SUCCEEDED(dev->CreateTexture2D(&td, nullptr, &mr_tex)) && SUCCEEDED(dev->CreateRenderTargetView(mr_tex, nullptr, &mr_rtv)) &&
                  SUCCEEDED(dev->CreateShaderResourceView(mr_tex, nullptr, &mr_srv));
        ok = ok && SUCCEEDED(dev->CreateTexture2D(&td, nullptr, &virt_tex)) && SUCCEEDED(dev->CreateRenderTargetView(virt_tex, nullptr, &virt_rtv)) &&
             SUCCEEDED(dev->CreateShaderResourceView(virt_tex, nullptr, &virt_srv));
        D3D11_TEXTURE2D_DESC dd = td; dd.Format = DXGI_FORMAT_D32_FLOAT; dd.BindFlags = D3D11_BIND_DEPTH_STENCIL;
        ok = ok && SUCCEEDED(dev->CreateTexture2D(&dd, nullptr, &mr_depth)) && SUCCEEDED(dev->CreateDepthStencilView(mr_depth, nullptr, &mr_dsv));
        D3D11_TEXTURE2D_DESC sd = td; sd.BindFlags = 0; sd.Usage = D3D11_USAGE_STAGING; sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        for (auto& st : rec_staging) ok = ok && SUCCEEDED(dev->CreateTexture2D(&sd, nullptr, &st));
        MR_W = ok ? w : 0; MR_H = ok ? h : 0;
        g_app.mr_srv = ok ? mr_srv.p : nullptr; g_app.mr_w = MR_W; g_app.mr_h = MR_H;
        printf("RGB view %ux%u%s\n", w, h, ok ? "" : " FAILED");
        return ok;
    };
    Com<ID3D11Texture2D> rgb_in;             // the newest RGB frame, uploaded
    Com<ID3D11ShaderResourceView> rgb_in_srv;
    int rgb_in_w = 0, rgb_in_h = 0;
    int64_t rec_stamp[3] = {};
    long rec_k = 0;

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
    // Instanced pipeline for the scanned room.
    Com<ID3DBlob> vsib, psib;
    if (FAILED(D3DCompile(SHADER, strlen(SHADER), nullptr, nullptr, nullptr, "vs_inst", "vs_5_0", 0, 0, &vsib, &err)) ||
        FAILED(D3DCompile(SHADER, strlen(SHADER), nullptr, nullptr, nullptr, "ps_inst", "ps_5_0", 0, 0, &psib, &err))) {
        fprintf(stderr, "shader: %s\n", err ? (const char*)err->GetBufferPointer() : "?");
        return 1;
    }
    Com<ID3D11VertexShader> vs_inst; CHECK(dev->CreateVertexShader(vsib->GetBufferPointer(), vsib->GetBufferSize(), nullptr, &vs_inst));
    Com<ID3D11PixelShader> ps_inst; CHECK(dev->CreatePixelShader(psib->GetBufferPointer(), psib->GetBufferSize(), nullptr, &ps_inst));
    D3D11_INPUT_ELEMENT_DESC ili[] = {{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
                                      {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
                                      {"INSTPOS", 0, DXGI_FORMAT_R32G32B32_FLOAT, 1, 0, D3D11_INPUT_PER_INSTANCE_DATA, 1},
                                      {"INSTCOL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 1, 12, D3D11_INPUT_PER_INSTANCE_DATA, 1}};
    Com<ID3D11InputLayout> layout_inst; CHECK(dev->CreateInputLayout(ili, 4, vsib->GetBufferPointer(), vsib->GetBufferSize(), &layout_inst));
    // Textured pipeline and the test panel.
    Com<ID3DBlob> vstb, pstb;
    if (FAILED(D3DCompile(SHADER, strlen(SHADER), nullptr, nullptr, nullptr, "vs_tex", "vs_5_0", 0, 0, &vstb, &err)) ||
        FAILED(D3DCompile(SHADER, strlen(SHADER), nullptr, nullptr, nullptr, "ps_tex", "ps_5_0", 0, 0, &pstb, &err))) {
        fprintf(stderr, "shader: %s\n", err ? (const char*)err->GetBufferPointer() : "?");
        return 1;
    }
    Com<ID3D11VertexShader> vs_tex; CHECK(dev->CreateVertexShader(vstb->GetBufferPointer(), vstb->GetBufferSize(), nullptr, &vs_tex));
    Com<ID3D11PixelShader> ps_shadow, ps_comp;
    {
        Com<ID3DBlob> b1, b2;
        if (FAILED(D3DCompile(SHADER, strlen(SHADER), nullptr, nullptr, nullptr, "ps_shadow", "ps_5_0", 0, 0, &b1, &err)) ||
            FAILED(D3DCompile(SHADER, strlen(SHADER), nullptr, nullptr, nullptr, "ps_comp", "ps_5_0", 0, 0, &b2, &err))) {
            fprintf(stderr, "shader: %s\n", err ? (const char*)err->GetBufferPointer() : "?");
            return 1;
        }
        CHECK(dev->CreatePixelShader(b1->GetBufferPointer(), b1->GetBufferSize(), nullptr, &ps_shadow));
        CHECK(dev->CreatePixelShader(b2->GetBufferPointer(), b2->GetBufferSize(), nullptr, &ps_comp));
    }
    Com<ID3D11PixelShader> ps_tex; CHECK(dev->CreatePixelShader(pstb->GetBufferPointer(), pstb->GetBufferSize(), nullptr, &ps_tex));
    Com<ID3D11SamplerState> samp;
    { D3D11_SAMPLER_DESC sd{}; sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR; sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP; CHECK(dev->CreateSamplerState(&sd, &samp)); }
    Com<ID3D11Buffer> quad_vb;
    {
        using namespace panel;
        std::vector<Vertex> q = {{-PW / 2, -PH / 2, 0, 0, 1, 0}, {PW / 2, -PH / 2, 0, 1, 1, 0}, {PW / 2, PH / 2, 0, 1, 0, 0},
                                 {-PW / 2, -PH / 2, 0, 0, 1, 0}, {PW / 2, PH / 2, 0, 1, 0, 0}, {-PW / 2, PH / 2, 0, 0, 0, 0}};
        D3D11_BUFFER_DESC bd{}; bd.ByteWidth = (UINT)(q.size() * sizeof(Vertex)); bd.BindFlags = D3D11_BIND_VERTEX_BUFFER; bd.Usage = D3D11_USAGE_IMMUTABLE;
        D3D11_SUBRESOURCE_DATA init{q.data()};
        CHECK(dev->CreateBuffer(&bd, &init, &quad_vb));
    }
    Com<ID3D11Buffer> screen_vb;  // a quad over the whole target (identity transform), uv in the normal slot
    {
        std::vector<Vertex> q = {{-1, -1, 0, 0, 1, 0}, {1, -1, 0, 1, 1, 0}, {1, 1, 0, 1, 0, 0}, {-1, -1, 0, 0, 1, 0}, {1, 1, 0, 1, 0, 0}, {-1, 1, 0, 0, 0, 0}};
        D3D11_BUFFER_DESC bd{}; bd.ByteWidth = (UINT)(q.size() * sizeof(Vertex)); bd.BindFlags = D3D11_BIND_VERTEX_BUFFER; bd.Usage = D3D11_USAGE_IMMUTABLE;
        D3D11_SUBRESOURCE_DATA init{q.data()};
        CHECK(dev->CreateBuffer(&bd, &init, &screen_vb));
    }
    // Realistic hands (the Rocketbox human's, skinned on the tracked joints).
    skinhands::Renderer skin;
    {
        wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring d = exe; d = d.substr(0, d.find_last_of(L'\\') + 1);
        if (!skin.init(dev, d + L"models\\")) printf("realistic hands: %s\n", skin.error.c_str());
    }
    // The room's light (from the RGB camera): its level and colour tint the virtual.
    float room_tint[3] = {1, 1, 1};
    panel::Panel test_panel;
    bool panel_ok = test_panel.init(dev);
    const V3 PANEL_AT = {0, -0.08f, -0.6f};  // scene: in front of the recentred head, a little low
    int pokes = 0;
    bool poke_inside = false, lit = false;
    double lit_until = 0;
    const UINT MAX_INST = 120000;
    Com<ID3D11Buffer> inst_vb;
    {
        D3D11_BUFFER_DESC bd{}; bd.ByteWidth = MAX_INST * 24; bd.BindFlags = D3D11_BIND_VERTEX_BUFFER; bd.Usage = D3D11_USAGE_DYNAMIC; bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        CHECK(dev->CreateBuffer(&bd, nullptr, &inst_vb));
    }
    UINT inst_count = 0;
    uint32_t inst_version = 0;

    auto make_vb = [&](const std::vector<Vertex>& v, ID3D11Buffer** out) {
        D3D11_BUFFER_DESC bd{}; bd.ByteWidth = (UINT)(v.size() * sizeof(Vertex)); bd.BindFlags = D3D11_BIND_VERTEX_BUFFER; bd.Usage = D3D11_USAGE_IMMUTABLE;
        D3D11_SUBRESOURCE_DATA init{v.data()};
        return dev->CreateBuffer(&bd, &init, out);
    };
    auto cube = cube_vertices(), grid = grid_vertices();
    Com<ID3D11Buffer> cube_vb, grid_vb, cb;
    CHECK(make_vb(cube, &cube_vb));
    CHECK(make_vb(grid, &grid_vb));
    // The hands as hands: bones as rounded cylinders, joints as spheres, a palm.
    auto cyl = cylinder_vertices(), sph = sphere_vertices();
    Com<ID3D11Buffer> cyl_vb, sph_vb, palm_vb;
    CHECK(make_vb(cyl, &cyl_vb));
    CHECK(make_vb(sph, &sph_vb));
    const UINT PALM_VERTS = 2 * 2 * 6 * 3;  // 2 hands, 2 faces, 6 triangles
    {
        D3D11_BUFFER_DESC bd{}; bd.ByteWidth = PALM_VERTS * sizeof(Vertex); bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        bd.Usage = D3D11_USAGE_DYNAMIC; bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        CHECK(dev->CreateBuffer(&bd, nullptr, &palm_vb));
    }
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
    // Contact shadows (the RGB view only): premultiplied blending, no depth writes, a disc.
    Com<ID3D11BlendState> shadow_blend;
    {
        D3D11_BLEND_DESC bd{}; auto& r = bd.RenderTarget[0];
        r.BlendEnable = TRUE; r.SrcBlend = D3D11_BLEND_ONE; r.DestBlend = D3D11_BLEND_INV_SRC_ALPHA; r.BlendOp = D3D11_BLEND_OP_ADD;
        r.SrcBlendAlpha = D3D11_BLEND_ONE; r.DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA; r.BlendOpAlpha = D3D11_BLEND_OP_ADD; r.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        CHECK(dev->CreateBlendState(&bd, &shadow_blend));
    }
    Com<ID3D11DepthStencilState> dss_nowrite;
    { D3D11_DEPTH_STENCIL_DESC d2 = dsd; d2.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO; d2.DepthFunc = D3D11_COMPARISON_LESS_EQUAL; CHECK(dev->CreateDepthStencilState(&d2, &dss_nowrite)); }
    Com<ID3D11Buffer> disc_vb;  // a unit square on the floor plane (y = 0), uv 0..1 in the normal slot
    {
        std::vector<Vertex> q = {{-1, 0, -1, 0, 0, 0}, {1, 0, -1, 1, 0, 0}, {1, 0, 1, 1, 1, 0}, {-1, 0, -1, 0, 0, 0}, {1, 0, 1, 1, 1, 0}, {-1, 0, 1, 0, 1, 0}};
        D3D11_BUFFER_DESC bd{}; bd.ByteWidth = (UINT)(q.size() * sizeof(Vertex)); bd.BindFlags = D3D11_BIND_VERTEX_BUFFER; bd.Usage = D3D11_USAGE_IMMUTABLE;
        D3D11_SUBRESOURCE_DATA init{q.data()};
        CHECK(dev->CreateBuffer(&bd, &init, &disc_vb));
    }
    // Depth only (no colour): the room mesh, so real surfaces hide the virtual behind them.
    Com<ID3D11BlendState> no_color;
    { D3D11_BLEND_DESC bd{}; bd.RenderTarget[0].RenderTargetWriteMask = 0; CHECK(dev->CreateBlendState(&bd, &no_color)); }
    Com<ID3D11Buffer> mesh_vb;
    {
        D3D11_BUFFER_DESC bd{}; bd.ByteWidth = (UINT)(room::MAX_MESH_VERTS * sizeof(Vertex)); bd.BindFlags = D3D11_BIND_VERTEX_BUFFER; bd.Usage = D3D11_USAGE_DYNAMIC; bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        CHECK(dev->CreateBuffer(&bd, nullptr, &mesh_vb));
    }
    UINT mesh_count = 0;
    uint32_t mesh_version = 0;

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
    if (!g_no_rgb) rgbcam::start();
    {
        // Hands: factory calibration (cached by the web app's dev server) and the models.
        wchar_t exe[MAX_PATH];
        GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring dir = exe;
        dir = dir.substr(0, dir.find_last_of(L'\\') + 1);
        calib::Rig rig = calib::load(std::string(dir.begin(), dir.end()) + "..\\calibration.bin");
        rgbcal::g_path = std::string(dir.begin(), dir.end()) + "myviture_rgb.txt";
        g_settings_path = std::string(dir.begin(), dir.end()) + "myviture_settings.txt";
        if (FILE* f = fopen(g_settings_path.c_str(), "r")) {
            float v[7] = {0, 0, 0, 0, 0, 0, 0};
            int n = fscanf(f, "%f %f %f %f %f %f %f", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6]);
            if (n >= 5) { g_app.vfov = v[0]; g_app.ipd_mm = v[1]; g_app.floor_m = v[2]; g_app.pitch_deg = v[3]; g_app.eye_dy_cm = v[4]; }
            if (n >= 7) { g_app.eye_dz_cm = v[5]; g_app.yaw_deg = v[6]; }
            fclose(f);
        }
        { std::lock_guard<std::mutex> l(rgbcal::g_mu); if (rgbcal::load(rgbcal::g_path, rgbcal::g_cam)) printf("RGB camera fit loaded\n"); }
        // VITURE's hand tracking, with the engine's camera calibration (its YAML, kept next
        // to the exe for reference).
        {
            typedef const char* (*ConfigFn)();
            auto cfg = (ConfigFn)GetProcAddress(GetModuleHandleA("carina_vio.dll"), "carina_a1088_viture_get_config");
            std::string yaml = cfg && cfg() ? cfg() : "";
            std::string d8(dir.begin(), dir.end());
            if (FILE* f = fopen((d8 + "viture_config.yaml").c_str(), "w")) { fputs(yaml.c_str(), f); fclose(f); }
            // The engine's camera parameters (a YAML text), kept for reference.
            typedef const char* (*CamParamFn)();
            if (auto cp = (CamParamFn)GetProcAddress(GetModuleHandleA("carina_vio.dll"), "carina_a1088_viture_get_cam_param"))
                if (const char* txt = cp()) if (FILE* f = fopen((d8 + "viture_cam_param.yaml").c_str(), "w")) { fputs(txt, f); fclose(f); }
            if (!g_no_vhands) vhands::start(sdk, yaml, d8 + "vhands_raw.csv");
        }
        if (rig.ok) {
            hands::start(dir + L"models\\", rig, [](double t, float* out) { M4 r; if (!engine_pose(t, r)) return false; memcpy(out, r.m, sizeof r.m); return true; });
            if (!g_no_room) room::start(rig, [](double t, float* out) { M4 r; if (!engine_pose(t, r)) return false; memcpy(out, r.m, sizeof r.m); return true; },
                        std::string(dir.begin(), dir.end()) + "myviture_anchors.txt");
        }
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
    std::vector<double> pred_err_still, pred_err_slow, pred_err_fast;  // degrees, head < 15, 15-60, > 60 deg/s
    long status_counts[10] = {};
    int corr_n = 0; float corr_deg = 0, corr_mm = 0;
    UINT last_refresh = 0, last_present_seen = 0;
    long missed = 0, shown = 0, last_updates = 0;
    int64_t stats_t0 = qpc_now(), frame_t0 = qpc_now();
    float rate_dps = 0, max_rate = 0;
    M4 recenter = identity();
    bool dash_on = true;

    printf("rendering on the glasses: Esc quit, R recenter, +/- prediction (%.0f ms)\n", g_app.predict_ms);
    // vblank timing from the frame statistics, for late latching
    int64_t vblank_qpc = 0;
    double period_ms = 1000.0 / 60;
    int64_t prev_sync_qpc = 0; UINT prev_sync_count = 0;
    double margin_ms = g_app.latch_margin_ms;
    long missed_window = 0; int64_t quiet_since = qpc_now();
    std::vector<double> lead_hist;
    while (!g_quit && !g_app.quit) {
        g_engine_predict = g_app.engine_predict;
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
        WaitForSingleObjectEx(frame_wait, 100, TRUE);
        // Late latching: the waitable returns right after a vblank; rendering then would
        // show a pose ~17 ms old at the next one. Sleep until just before it instead.
        double to_vblank_ms = period_ms;
        if (vblank_qpc) {
            int64_t now = qpc_now();
            double next = qpc_ms(vblank_qpc - now);
            while (next < 0.5) next += period_ms;
            if (g_app.late_latch) {
                double wake = next - margin_ms;
                if (wake > 1.5) Sleep((DWORD)(wake - 1.0));
                while (qpc_ms(qpc_now() - now) < wake) {}
            }
            to_vblank_ms = qpc_ms(vblank_qpc - qpc_now());
            while (to_vblank_ms < 0.3) to_vblank_ms += period_ms;
        }
        // Predict to the vblank this frame should catch, plus the screen's own latency.
        double g_predict_ms = to_vblank_ms + g_app.predict_ms;
        lead_hist.push_back(g_predict_ms);
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
        // Prediction error (the swim): what this frame predicted for its display time against
        // the engine's estimate for that same time once it has passed (80 ms later, with the
        // camera updates in). Degrees, by head speed, for the stats line.
        {
            struct Pred { double t; Q q; float rate; };
            static std::deque<Pred> preds;
            double target = device_now() + g_predict_ms / 1000.0;
            if (have) preds.push_back({target, q, rate_dps});
            double now_dev = device_now();
            while (!preds.empty() && preds.front().t < now_dev - 0.08) {
                M4 truth;
                if (engine_pose(preds.front().t, truth)) {
                    Q d = qmul(mat_q(truth), qconj(preds.front().q));
                    float ang = 2 * atan2f(sqrtf(d.x * d.x + d.y * d.y + d.z * d.z), fabsf(d.w)) * 57.2958f;
                    (preds.front().rate > 60 ? pred_err_fast : preds.front().rate > 15 ? pred_err_slow : pred_err_still).push_back(ang);
                }
                preds.pop_front();
            }
            if (preds.size() > 600) preds.clear();
        }
        // The engine's visual update (25 Hz) revises the pose it predicted from the IMU:
        // a step every camera frame, seen as a twitch. Asking again for last frame's
        // target gives that revision; an offset absorbs it and fades out, so the scene
        // slides to the corrected place instead of jumping. Big steps (relocalisation,
        // recentre) are let through.
        {
            static double prev_target = 0;
            static Q prev_q_at{0, 0, 0, 1};
            static V3 prev_p_at{0, 0, 0};
            static Q oq{0, 0, 0, 1};
            static V3 op{0, 0, 0};
            static int64_t last_qpc = 0;
            double target = device_now() + g_predict_ms / 1000.0;
            M4 re;
            if (have && prev_target > 0 && engine_pose(prev_target, re)) {
                Q q_re = mat_q(re);
                V3 p_re = {re(0, 3), re(1, 3), re(2, 3)};
                Q dq = qmul(q_re, qconj(prev_q_at));
                if (dq.w < 0) dq = {-dq.x, -dq.y, -dq.z, -dq.w};
                V3 dp = {p_re.x - prev_p_at.x, p_re.y - prev_p_at.y, p_re.z - prev_p_at.z};
                float ang = 2 * atan2f(sqrtf(dq.x * dq.x + dq.y * dq.y + dq.z * dq.z), dq.w) * 57.2958f;
                float dist = sqrtf(dp.x * dp.x + dp.y * dp.y + dp.z * dp.z);
                if (ang > 0.003f || dist > 0.0002f) {
                    corr_n++; corr_deg += ang; corr_mm += dist * 1000;
                    if (g_app.smooth_corrections && ang < 3 && dist < 0.05) {
                        oq = qnorm(qmul(qconj(dq), oq));
                        op = {op.x - dp.x, op.y - dp.y, op.z - dp.z};
                    } else { oq = {0, 0, 0, 1}; op = {0, 0, 0}; }
                }
            }
            if (have) {
                M4 now_at;
                if (engine_pose(target, now_at)) { prev_target = target; prev_q_at = mat_q(now_at); prev_p_at = {now_at(0, 3), now_at(1, 3), now_at(2, 3)}; }
                // Fade: fast while the head turns (errors are hidden), slow when still.
                double dt = last_qpc ? qpc_ms(qpc_now() - last_qpc) / 1000.0 : 0.016;
                double tau = rate_dps > 20 ? 0.06 : 0.15;
                float k = (float)std::exp(-dt / tau);
                // oq^k by shrinking its angle
                float s = sqrtf(oq.x * oq.x + oq.y * oq.y + oq.z * oq.z);
                if (s > 1e-7f) { float a = 2 * atan2f(s, oq.w) * k; oq = {oq.x / s * sinf(a / 2), oq.y / s * sinf(a / 2), oq.z / s * sinf(a / 2), cosf(a / 2)}; }
                op = {op.x * k, op.y * k, op.z * k};
                if (g_app.smooth_corrections) { q = qnorm(qmul(oq, q)); p = {p.x + op.x, p.y + op.y, p.z + op.z}; }
            }
            last_qpc = qpc_now();
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
            for (int i = 0; i < 16; i++) g_app.head_scene[i] = head_scene.m[i];
            g_app.rate_dps = rate_dps;
        }

        // Hands from the tracking thread: head frame at the image's time -> scene, with the
        // engine's pose at that same time (no extrapolation yet).
        struct SceneHand { bool right, pinching; int gesture; V3 j[hands::N_JOINTS]; };
        std::vector<SceneHand> scene_hands;
        if (g_app.hands && vhands::g_on) {
            // VITURE's hands: already in the engine's world, so only the recentring remains.
            vhands::Out vo;
            { std::lock_guard<std::mutex> l(vhands::g_mu); vo = vhands::g_out; }
            static bool pinch_state[2] = {false, false};
            static int shown[2] = {0, 0}, cand[2] = {0, 0}, count[2] = {0, 0};
            static uint32_t seen_seq = 0;
            bool fresh = vo.seq != seen_seq;
            seen_seq = vo.seq;
            if (qpc_now() / qpc_freq() - vo.t < 0.25) {
                for (int side = 0; side < 2; side++) {
                    float w[21][3];
                    if (!vhands::hand_world(vo.data, side, w)) { pinch_state[side] = false; continue; }
                    // Smoothed per joint on the answers' own times (VITURE's joints come raw).
                    static vhands::OneEuro filt[2][hands::N_JOINTS][3];
                    static uint32_t filt_seq[2] = {0, 0};
                    static float smoothed[2][hands::N_JOINTS][3];
                    if (filt_seq[side] != vo.seq) {
                        filt_seq[side] = vo.seq;
                        for (int k = 0; k < hands::N_JOINTS; k++) for (int a = 0; a < 3; a++) smoothed[side][k][a] = (float)filt[side][k][a].apply(w[k][a], vo.t);
                    }
                    memcpy(w, smoothed[side], sizeof w);
                    SceneHand sh{side == 1, false, 0, {}};
                    for (int k = 0; k < hands::N_JOINTS; k++) {
                        const float* q = w[k];
                        sh.j[k] = {recenter(0, 0) * q[0] + recenter(0, 1) * q[1] + recenter(0, 2) * q[2] + recenter(0, 3),
                                   recenter(1, 0) * q[0] + recenter(1, 1) * q[1] + recenter(1, 2) * q[2] + recenter(1, 3),
                                   recenter(2, 0) * q[0] + recenter(2, 1) * q[1] + recenter(2, 2) * q[2] + recenter(2, 3)};
                    }
                    // Pinch with hysteresis, gesture after 3 agreeing answers (as for ours).
                    float dx = w[4][0] - w[8][0], dy = w[4][1] - w[8][1], dz = w[4][2] - w[8][2];
                    float pinch = std::sqrt(dx * dx + dy * dy + dz * dz);
                    pinch_state[side] = pinch_state[side] ? pinch < 0.04f : pinch < 0.025f;
                    sh.pinching = pinch_state[side];
                    if (fresh) {
                        float ext[5];
                        int g = hands::classify(w, sh.pinching, ext);
                        if (g == cand[side]) count[side]++; else { cand[side] = g; count[side] = 1; }
                        if (count[side] >= 3 || g == 1) shown[side] = g;
                    }
                    sh.gesture = shown[side];
                    scene_hands.push_back(sh);
                }
            }
        }

        // The scanned room: new voxel snapshot into the instance buffer, planes and anchors.
        room::g_scan = g_app.scan;
        if (g_app.room_request) { room::g_request = g_app.room_request; g_app.room_request = 0; }
        std::vector<room::Plane> planes;
        std::vector<room::Anchor> anchors;
        {
            std::lock_guard<std::mutex> l(room::g.mu);
            planes = room::g.planes;
            anchors = room::g.anchors;
            if (room::g.mesh_version != mesh_version) {
                mesh_version = room::g.mesh_version;
                D3D11_MAPPED_SUBRESOURCE m;
                if (SUCCEEDED(ctx->Map(mesh_vb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
                    mesh_count = (UINT)std::min(room::MAX_MESH_VERTS, room::g.mesh.size() / 6);
                    memcpy(m.pData, room::g.mesh.data(), (size_t)mesh_count * sizeof(Vertex));
                    ctx->Unmap(mesh_vb, 0);
                }
            }
            if (room::g.version != inst_version && !room::g.points.empty()) {
                inst_version = room::g.version;
                D3D11_MAPPED_SUBRESOURCE m;
                if (SUCCEEDED(ctx->Map(inst_vb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
                    inst_count = std::min<UINT>(MAX_INST, (UINT)(room::g.points.size() / 6));
                    memcpy(m.pData, room::g.points.data(), inst_count * 24);
                    ctx->Unmap(inst_vb, 0);
                }
            }
        }
        // Scene <-> tracking frame: recenter is a yaw and a translation, so heights shift by a constant.
        const M4 recenter_inv = rigid_inverse(recenter);
        const float y_shift = recenter(1, 3);
        // Highest surface under a scene point (floor or table), as a scene height; NAN if none.
        auto surface_under = [&](float sx, float sy, float sz) {
            float wx = recenter_inv(0, 0) * sx + recenter_inv(0, 1) * sy + recenter_inv(0, 2) * sz + recenter_inv(0, 3);
            float wy = recenter_inv(1, 0) * sx + recenter_inv(1, 1) * sy + recenter_inv(1, 2) * sz + recenter_inv(1, 3);
            float wz = recenter_inv(2, 0) * sx + recenter_inv(2, 1) * sy + recenter_inv(2, 2) * sz + recenter_inv(2, 3);
            float best = NAN;
            for (auto& p : planes) {
                if (p.y > wy + 0.05f) continue;
                if (p.kind == 1 && (wx < p.minx || wx > p.maxx || wz < p.minz || wz > p.maxz)) continue;
                if (std::isnan(best) || p.y > best) best = p.y;
            }
            return std::isnan(best) ? best : best + y_shift;
        };

        // Test panel: gestures per hand, poke on its button with the index tip.
        if (panel_ok && g_app.panel) {
            using namespace panel;
            // Hysteresis: pressed within 2.5 cm around the button and under 2 cm from the
            // surface (or through it), released only 4 cm outside or 5 cm in front, so the
            // fingertip's jitter does not toggle it. The tracked hand and the drawn panel
            // differ by the calibration's error: the margin absorbs it.
            bool pressed = false, released = true;
            float best_d = 1e9f, best[3] = {0, 0, 0};
            for (auto& h : scene_hands) {
                V3 t = h.j[8];
                float lx = t.x - PANEL_AT.x, ly = t.y - PANEL_AT.y, lz = t.z - PANEL_AT.z;
                const float m = 0.025f, r = 0.04f;
                if (lx > BX0 - m && lx < BX1 + m && ly > BY0 - m && ly < BY1 + m && lz < 0.02f && lz > -0.12f) pressed = true;
                if (lx > BX0 - r && lx < BX1 + r && ly > BY0 - r && ly < BY1 + r && lz < 0.05f && lz > -0.15f) released = false;
                // Index tip relative to the button's centre, for the panel's line.
                float dx = lx - (BX0 + BX1) / 2, dy = ly - (BY0 + BY1) / 2;
                float d = dx * dx + dy * dy + lz * lz;
                if (d < best_d) { best_d = d; best[0] = dx; best[1] = dy; best[2] = lz; }
            }
            double now_s = qpc_now() / qpc_freq();
            if (!poke_inside && pressed) {
                poke_inside = true; pokes++;
                hands::logf("panel: poke %d, index %.1f %.1f %.1f cm from the button\n", pokes, best[0] * 100, best[1] * 100, best[2] * 100);
            }
            else if (poke_inside && released) poke_inside = false;
            if (poke_inside) lit_until = now_s + 0.3;
            lit = now_s < lit_until;
            std::vector<std::string> lines;
            for (int side = 0; side < 2; side++) {
                const SceneHand* h = nullptr;
                for (auto& x : scene_hands) if (x.right == (side == 1)) h = &x;
                lines.push_back(std::string(side ? "Main droite : " : "Main gauche : ") + (h ? hands::GESTURES[h->gesture] : "pas vue"));
            }
            char buf[96];
            snprintf(buf, sizeof buf, "Bouton touché %d fois (index dans le carré)", pokes);
            lines.push_back(buf);
            // Where the tracked index tip is against the button: shows the overlay's error.
            if (best_d < 1e8f) {
                snprintf(buf, sizeof buf, "Index : %+.0f cm droite, %+.0f cm haut, %.0f cm %s", best[0] * 100, best[1] * 100, std::fabs(best[2]) * 100, best[2] > 0 ? "devant" : "derrière");
                lines.push_back(buf);
            }
            test_panel.draw(ctx, "myViture · test des mains", lines, lit ? "touché !" : "touche-moi", lit);
        }

        // Palms: the wrist and the knuckles as a fan, two faces 7 mm apart (some thickness).
        UINT palm_count = 0;
        {
            std::vector<Vertex> pv;
            for (auto& h : scene_hands) {
                const int ring[6] = {0, 1, 5, 9, 13, 17};
                V3 cen{0, 0, 0};
                for (int k : ring) { cen.x += h.j[k].x / 6; cen.y += h.j[k].y / 6; cen.z += h.j[k].z / 6; }
                V3 a{h.j[5].x - h.j[0].x, h.j[5].y - h.j[0].y, h.j[5].z - h.j[0].z}, b{h.j[17].x - h.j[0].x, h.j[17].y - h.j[0].y, h.j[17].z - h.j[0].z};
                V3 n{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
                float nl = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
                if (nl < 1e-6f) continue;
                n = {n.x / nl, n.y / nl, n.z / nl};
                for (int side = -1; side <= 1; side += 2) {
                    float o = 0.0035f * side;
                    auto P = [&](const V3& p) { return Vertex{p.x + n.x * o, p.y + n.y * o, p.z + n.z * o, n.x * side, n.y * side, n.z * side}; };
                    for (int i = 0; i < 6; i++) { pv.push_back(P(cen)); pv.push_back(P(h.j[ring[i]])); pv.push_back(P(h.j[ring[(i + 1) % 6]])); }
                }
            }
            D3D11_MAPPED_SUBRESOURCE m;
            if (!pv.empty() && pv.size() <= PALM_VERTS && SUCCEEDED(ctx->Map(palm_vb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
                memcpy(m.pData, pv.data(), pv.size() * sizeof(Vertex));
                ctx->Unmap(palm_vb, 0);
                palm_count = (UINT)pv.size();
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
        // One view of the scene: hands (joints and bones) and, if on, the test objects.
        bool rgb_pass = false;  // render_view for the RGB view: contact shadows too
        auto render_view = [&](const M4& vpm) {
            // A soft shadow on the surface under a point (RGB view only: the glasses cannot
            // darken the real world).
            auto shadow = [&](float x, float y_surface, float z, float radius, float strength) {
                if (!rgb_pass || !g_app.shadows || strength <= 0.01f) return;
                M4 model = mul(translate({x, y_surface + 0.003f, z}), scale(radius));
                Constants c{mul(vpm, model), model, {strength, 0, 0, 1}, {0, 0, 0, 0}};
                ctx->UpdateSubresource(cb, 0, nullptr, &c, 0, 0);
                ID3D11Buffer* vbs[] = {disc_vb};
                ctx->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
                ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                ctx->VSSetShader(vs_tex, nullptr, 0);
                ctx->PSSetShader(ps_shadow, nullptr, 0);
                ctx->OMSetBlendState(shadow_blend, nullptr, 0xffffffff);
                ctx->OMSetDepthStencilState(dss_nowrite, 0);
                ctx->Draw(6, 0);
                ctx->OMSetBlendState(nullptr, nullptr, 0xffffffff);
                ctx->OMSetDepthStencilState(dss, 0);
                ctx->VSSetShader(vs, nullptr, 0);
                ctx->PSSetShader(ps, nullptr, 0);
            };
            auto draw = [&](ID3D11Buffer* vb, UINT n, D3D11_PRIMITIVE_TOPOLOGY topo, const M4& model, float r, float g, float b, bool lit) {
                if (lit) { r *= room_tint[0]; g *= room_tint[1]; b *= room_tint[2]; }
                Constants c{mul(vpm, model), model, {r, g, b, 1}, {lit ? 1.0f : 0.0f, 0, 0, 0}};
                ctx->UpdateSubresource(cb, 0, nullptr, &c, 0, 0);
                ID3D11Buffer* vbs[] = {vb};
                ctx->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
                ctx->IASetPrimitiveTopology(topo);
                ctx->Draw(n, 0);
            };
            // The room mesh first: drawn (dim, so it stays see-through in the glasses) or depth
            // only, so the virtual behind a real surface is hidden.
            if (mesh_count && (g_app.show_map || g_app.occlusion)) {
                if (!g_app.show_map) ctx->OMSetBlendState(no_color, nullptr, 0xffffffff);
                draw(mesh_vb, mesh_count, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, recenter, 0.22f, 0.30f, 0.42f, true);
                if (!g_app.show_map) ctx->OMSetBlendState(nullptr, nullptr, 0xffffffff);
            }
            // Hands: skin-coloured bones (rounded cylinders) and a palm; lighter when pinching.
            {
                // Finger widths (radius, m): thumb to little, thinner towards the tips.
                auto radius = [](int k) {
                    static const float R[21] = {0.016f, 0.012f, 0.011f, 0.0095f, 0.0085f, 0.0105f, 0.0095f, 0.0085f, 0.0075f, 0.0105f, 0.0095f, 0.0085f, 0.0075f,
                                                0.0100f, 0.0090f, 0.0080f, 0.0072f, 0.0090f, 0.0080f, 0.0072f, 0.0065f};
                    return R[k];
                };
                auto bone_model = [](const V3& a, const V3& b, float r) {
                    float wx = b.x - a.x, wy = b.y - a.y, wz = b.z - a.z, L = sqrtf(wx * wx + wy * wy + wz * wz);
                    M4 m = identity();
                    if (L < 1e-5f) { m(0, 0) = m(1, 1) = m(2, 2) = 0; return m; }
                    wx /= L; wy /= L; wz /= L;
                    // u: any unit vector across the bone, v = w x u.
                    float ux = fabsf(wx) < 0.9f ? 0 : -wz, uy = fabsf(wx) < 0.9f ? -wz : 0, uz = fabsf(wx) < 0.9f ? wy : wx;
                    float un = sqrtf(ux * ux + uy * uy + uz * uz); ux /= un; uy /= un; uz /= un;
                    float vx = wy * uz - wz * uy, vy = wz * ux - wx * uz, vz = wx * uy - wy * ux;
                    m(0, 0) = ux * r; m(1, 0) = uy * r; m(2, 0) = uz * r;
                    m(0, 1) = vx * r; m(1, 1) = vy * r; m(2, 1) = vz * r;
                    m(0, 2) = wx * L; m(1, 2) = wy * L; m(2, 2) = wz * L;
                    m(0, 3) = a.x; m(1, 3) = a.y; m(2, 3) = a.z;
                    return m;
                };
                for (auto& h : scene_hands) {
                    float px = 0, py = 0, pz = 0;
                    for (int k : {0, 5, 9, 13, 17}) { px += h.j[k].x / 5; py += h.j[k].y / 5; pz += h.j[k].z / 5; }
                    float ys = surface_under(px, py, pz);
                    if (!std::isnan(ys) && py - ys < 0.5f) shadow(px, ys, pz, 0.06f + 0.1f * (py - ys), 0.5f * (1 - (py - ys) / 0.5f));
                }
                if (skin.ok && g_app.real_hands) {
                    V3 elbows[2], wrists[2]; float hand_size[2]; int n_arms = 0;
                    for (auto& h : scene_hands) {
                        float j[22][3];
                        for (int k = 0; k < 21; k++) { j[k][0] = h.j[k].x; j[k][1] = h.j[k].y; j[k][2] = h.j[k].z; }
                        skin.elbow(j, h.right, head_scene.m, j[21]);  // the forearm, by IK from an estimated shoulder
                        float t[3] = {room_tint[0] * (h.pinching ? 1.15f : 1.0f), room_tint[1] * (h.pinching ? 1.15f : 1.0f), room_tint[2] * (h.pinching ? 1.15f : 1.0f)};
                        skin.draw(ctx, vpm.m, j, h.right, t, g_app.sleeves ? g_app.sleeve_color : nullptr);
                        if (n_arms < 2) {
                            elbows[n_arms] = {j[21][0], j[21][1], j[21][2]}; wrists[n_arms] = h.j[0];
                            float dx = h.j[9].x - h.j[0].x, dy = h.j[9].y - h.j[0].y, dz = h.j[9].z - h.j[0].z;
                            hand_size[n_arms++] = sqrtf(dx * dx + dy * dy + dz * dz) / 0.096f;
                        }
                    }
                    // Back to the scene's pipeline.
                    ctx->IASetInputLayout(layout);
                    ctx->VSSetShader(vs, nullptr, 0);
                    ctx->PSSetShader(ps, nullptr, 0);
                    ID3D11Buffer* cbs0[] = {cb};
                    ctx->VSSetConstantBuffers(0, 1, cbs0);
                    ctx->PSSetConstantBuffers(0, 1, cbs0);
                }
                for (auto& h : scene_hands) {
                    if (skin.ok && g_app.real_hands) break;
                    float r = h.pinching ? 1.0f : 0.92f, g = h.pinching ? 0.86f : 0.68f, b = h.pinching ? 0.78f : 0.56f;
                    for (auto& bn : hands::BONES) {
                        if (bn[0] == 0 && bn[1] != 1) continue;  // wrist to the knuckles: the palm covers it
                        float rb = std::min(radius(bn[0]), radius(bn[1]));
                        draw(cyl_vb, (UINT)cyl.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, bone_model(h.j[bn[0]], h.j[bn[1]], rb), r, g, b, true);
                    }
                    for (int k = 1; k < hands::N_JOINTS; k++) draw(sph_vb, (UINT)sph.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(translate(h.j[k]), scale(radius(k))), r, g, b, true);
                    draw(sph_vb, (UINT)sph.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(translate(h.j[0]), scale(0.017f)), r, g, b, true);
                }
                if (palm_count && !(skin.ok && g_app.real_hands)) {
                    Constants c{vpm, identity(), {0.92f, 0.68f, 0.56f, 1}, {1, 0, 0, 0}};
                    ctx->UpdateSubresource(cb, 0, nullptr, &c, 0, 0);
                    ID3D11Buffer* vbs[] = {palm_vb};
                    ctx->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
                    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    ctx->Draw(palm_count, 0);
                }
            }
            if (panel_ok && g_app.panel) {
                Constants c{mul(vpm, translate(PANEL_AT)), translate(PANEL_AT), {1, 1, 1, 1}, {0, 0, 0, 0}};
                ctx->UpdateSubresource(cb, 0, nullptr, &c, 0, 0);
                ID3D11Buffer* vbs[] = {quad_vb};
                ctx->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
                ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                ctx->VSSetShader(vs_tex, nullptr, 0);
                ctx->PSSetShader(ps_tex, nullptr, 0);
                ID3D11ShaderResourceView* srvs[] = {test_panel.srv};
                ctx->PSSetShaderResources(0, 1, srvs);
                ID3D11SamplerState* ss[] = {samp};
                ctx->PSSetSamplers(0, 1, ss);
                ctx->Draw(6, 0);
                ID3D11ShaderResourceView* none[] = {nullptr};
                ctx->PSSetShaderResources(0, 1, none);
                ctx->VSSetShader(vs, nullptr, 0);
                ctx->PSSetShader(ps, nullptr, 0);
            }
            // Anchors: small cyan cubes on the real surfaces (tracking frame -> scene).
            for (auto& a : anchors) draw(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(mul(recenter, translate({a.p[0], a.p[1], a.p[2]})), scale(0.03f)), 0.3f, 1.0f, 1.0f, true);
            // The scanned room, one small cube per voxel.
            if ((g_app.calage || (g_app.show_map && !mesh_count)) && inst_count) {
                Constants c{mul(vpm, recenter), recenter, {1, 1, 1, 1}, {1, room::VOX * (g_app.calage ? 0.25f : 0.7f), 0, 0}};
                ctx->UpdateSubresource(cb, 0, nullptr, &c, 0, 0);
                ID3D11Buffer* vbs[] = {cube_vb, inst_vb};
                UINT strides[] = {sizeof(Vertex), 24}, offsets[] = {0, 0};
                ctx->IASetInputLayout(layout_inst);
                ctx->VSSetShader(vs_inst, nullptr, 0);
                ctx->PSSetShader(ps_inst, nullptr, 0);
                ctx->IASetVertexBuffers(0, 2, vbs, strides, offsets);
                ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                ctx->DrawInstanced((UINT)cube.size(), inst_count, 0, 0);
                ctx->IASetInputLayout(layout);
                ctx->VSSetShader(vs, nullptr, 0);
                ctx->PSSetShader(ps, nullptr, 0);
            }
            if (!g_app.scene || g_app.calage) return;
            // Eight cubes around the head at arm's length, one ahead at 1 m: resting on the
            // floor or the table found under them when the room is scanned.
            for (int i = 0; i < 8; i++) {
                float a = i * 3.14159265f / 4;
                float hue = i / 8.0f;
                float r = 0.5f + 0.5f * cosf(6.2832f * hue), g = 0.5f + 0.5f * cosf(6.2832f * (hue - 0.333f)), b = 0.5f + 0.5f * cosf(6.2832f * (hue - 0.667f));
                V3 at = {sinf(a) * 0.9f, 0, -cosf(a) * 0.9f};
                float ys = g_app.place_on_planes ? surface_under(at.x, at.y, at.z) : NAN;
                if (!std::isnan(ys)) at.y = ys + 0.06f;
                draw(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(translate(at), scale(0.12f)), r, g, b, true);
                if (!std::isnan(ys)) shadow(at.x, ys, at.z, 0.11f, 0.55f);
            }
            {
                V3 at = {0, -0.1f, -1.0f};
                float ys = g_app.place_on_planes ? surface_under(at.x, at.y, at.z) : NAN;
                if (!std::isnan(ys)) at.y = ys + 0.025f;
                draw(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(translate(at), scale(0.05f)), 1, 1, 1, true);
                if (!std::isnan(ys)) shadow(at.x, ys, at.z, 0.05f, 0.55f);
            }
            float floor_y = -g_app.floor_m;
            for (auto& p : planes) if (p.kind == 0) floor_y = p.y + y_shift;
            if (g_app.floor_grid) draw(grid_vb, (UINT)grid.size(), D3D11_PRIMITIVE_TOPOLOGY_LINELIST, translate({0, floor_y, 0}), 0.3f, 0.7f, 1.0f, false);
        };
        if (have) for (int eye = 0; eye < views; eye++) {
            float vw = (float)W / views;
            D3D11_VIEWPORT vp{eye * vw, 0, vw, (float)H, 0, 1};
            ctx->RSSetViewports(1, &vp);
            float off = sbs ? (eye ? ipd / 2 : -ipd / 2) : 0;
            // Eye frame: the RGB camera's fitted orientation when there is one (the optics are
            // on the same frame), then the manual tilt and eye height.
            // Fine adjustments, on top of the engine's eyes or of the manual model.
            M4 eye_rot = rot_x(g_app.pitch_deg * 3.14159265f / 180);
            if (g_app.vendor_eyes && g_app.have_vendor_eyes) {
                M4 D;
                if (sbs) memcpy(D.m, eye ? g_app.eye_r : g_app.eye_l, sizeof D.m);
                else for (int k = 0; k < 16; k++) D.m[k] = (g_app.eye_l[k] + g_app.eye_r[k]) / 2;  // 2D: one view, between the eyes
                M4 fine = mul(translate({0, g_app.eye_dy_cm / 100, g_app.eye_dz_cm / 100}), mul(rot_y(g_app.yaw_deg * 3.14159265f / 180), eye_rot));
                M4 eye_world = mul(mul(head_scene, fine), D);
                render_view(mul(perspective(g_app.vfov, vw / H, 0.05f, 50), rigid_inverse(eye_world)));
                continue;
            }
            if (g_app.display_from_rgb) {
                rgbcal::Cam rc = rgbcal::get();
                if (rc.fitted) {
                    double Rr[9];
                    rgbcal::rotation(rc.rv, Rr);
                    M4 cam_head = identity();
                    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) cam_head(i, j) = (float)Rr[j * 3 + i];
                    eye_rot = mul(cam_head, eye_rot);
                }
            }
            eye_rot = mul(rot_y(g_app.yaw_deg * 3.14159265f / 180), eye_rot);
            M4 eye_world = mul(mul(head_scene, translate({0, g_app.eye_dy_cm / 100, g_app.eye_dz_cm / 100})), mul(eye_rot, translate({off, 0, 0})));
            render_view(mul(perspective(g_app.vfov, vw / H, 0.05f, 50), rigid_inverse(eye_world)));
        }

        sc->Present(1, 0);

        // The RGB view, once per new RGB frame: the frame, then the virtual scene from the
        // fitted RGB camera with the head where it was when the frame was taken.
        {
            static uint32_t rgb_seen = 0;
            bool fresh = false;
            double t_rgb = 0;
            {
                std::lock_guard<std::mutex> l(rgbcam::g_frame.mu);
                auto& f = rgbcam::g_frame;
                if (f.seq != rgb_seen && f.w && f.bgra.size() >= (size_t)f.w * f.h) {
                    rgb_seen = f.seq; t_rgb = f.t;
                    if (!rgb_in || rgb_in_w != f.w || rgb_in_h != f.h) {
                        if (rgb_in.p) { rgb_in.p->Release(); rgb_in.p = nullptr; }
                        if (rgb_in_srv.p) { rgb_in_srv.p->Release(); rgb_in_srv.p = nullptr; }
                        D3D11_TEXTURE2D_DESC td{}; td.Width = f.w; td.Height = f.h; td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_B8G8R8X8_UNORM;
                        td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DYNAMIC; td.BindFlags = D3D11_BIND_SHADER_RESOURCE; td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
                        if (SUCCEEDED(dev->CreateTexture2D(&td, nullptr, &rgb_in)) && SUCCEEDED(dev->CreateShaderResourceView(rgb_in, nullptr, &rgb_in_srv))) { rgb_in_w = f.w; rgb_in_h = f.h; }
                    }
                    // The room's light: mean colour of the frame (every 16th pixel), as a level
                    // (dim rooms make the virtual dimmer, so it does not glow) and a tint.
                    {
                        double sr = 0, sg = 0, sb = 0; int n = 0;
                        for (int y = 8; y < f.h; y += 16) for (int x = 8; x < f.w; x += 16) {
                            uint32_t p = f.bgra[(size_t)y * f.w + x];
                            sr += (p >> 16) & 255; sg += (p >> 8) & 255; sb += p & 255; n++;
                        }
                        if (n) {
                            sr /= n * 255.0; sg /= n * 255.0; sb /= n * 255.0;
                            double lum = 0.2126 * sr + 0.7152 * sg + 0.0722 * sb, mx = std::max(sr, std::max(sg, sb));
                            double level = std::clamp(0.45 + 1.2 * lum, 0.45, 1.0);
                            double t[3] = {sr, sg, sb};
                            for (int i = 0; i < 3; i++) {
                                double tint = mx > 1e-3 ? 0.6 + 0.4 * t[i] / mx : 1;  // half towards the room's colour
                                double target = g_app.room_light ? level * tint : 1;
                                room_tint[i] = (float)(0.9 * room_tint[i] + 0.1 * target);  // slow: no flicker
                            }
                            g_app.room_level = (float)lum;
                            // Where the light comes from: the centroid of the brightest
                            // pixels (above 1.5 x the mean), as a direction from the camera.
                            double bx = 0, by = 0, bw = 0, thr = std::min(250.0, 1.5 * lum * 255);
                            for (int y = 8; y < f.h; y += 16) for (int x = 8; x < f.w; x += 16) {
                                uint32_t p = f.bgra[(size_t)y * f.w + x];
                                double l = 0.2126 * ((p >> 16) & 255) + 0.7152 * ((p >> 8) & 255) + 0.0722 * (p & 255);
                                if (l > thr) { double wgt = l - thr; bx += wgt * x; by += wgt * y; bw += wgt; }
                            }
                            rgbcal::Cam rc = rgbcal::get();
                            if (bw > 0 && rc.f > 0) {
                                double cx = (bx / bw - f.w / 2.0) / rc.f, cy = (by / bw - f.h / 2.0) / rc.f;
                                double R[9]; rgbcal::rotation(rc.rv, R);
                                // camera (x right, y down, z forward) -> head (GL) -> scene
                                double c3[3] = {cx, -cy, -1}, hd[3];
                                for (int i = 0; i < 3; i++) hd[i] = R[0 * 3 + i] * c3[0] + R[1 * 3 + i] * c3[1] + R[2 * 3 + i] * c3[2];
                                double sd[3];
                                for (int i = 0; i < 3; i++) sd[i] = head_scene(i, 0) * hd[0] + head_scene(i, 1) * hd[1] + head_scene(i, 2) * hd[2];
                                sd[1] += 0.6;  // lights are mostly above: keep some top light
                                double nl = std::sqrt(sd[0] * sd[0] + sd[1] * sd[1] + sd[2] * sd[2]);
                                for (int i = 0; i < 3; i++) skin.light[i] = (float)(0.9 * skin.light[i] + 0.1 * sd[i] / nl);
                            }
                        }
                    }
                    D3D11_MAPPED_SUBRESOURCE m;
                    if (rgb_in && SUCCEEDED(ctx->Map(rgb_in, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
                        for (int y = 0; y < f.h; y++) memcpy((uint8_t*)m.pData + y * m.RowPitch, f.bgra.data() + (size_t)y * f.w, (size_t)f.w * 4);
                        ctx->Unmap(rgb_in, 0);
                        fresh = true;
                    }
                }
            }
            if (fresh && ((UINT)rgb_in_w != MR_W || (UINT)rgb_in_h != MR_H)) {
                bool rec_on; { std::lock_guard<std::mutex> l(rec::g_mu); rec_on = rec::g_status.on; }
                if (rec_on) fresh = false; else make_mr(rgb_in_w, rgb_in_h);
            }
            if (fresh && MR_W) {
                D3D11_VIEWPORT vp{0, 0, (float)MR_W, (float)MR_H, 0, 1};
                ctx->RSSetViewports(1, &vp);
                // The virtual alone, premultiplied (empty: transparent).
                {
                    ID3D11RenderTargetView* t[] = {virt_rtv};
                    ctx->OMSetRenderTargets(1, t, mr_dsv);
                    float clear0[4] = {0, 0, 0, 0};
                    ctx->ClearRenderTargetView(virt_rtv, clear0);
                    ctx->ClearDepthStencilView(mr_dsv, D3D11_CLEAR_DEPTH, 1, 0);
                }
                rgbcal::Cam cam = rgbcal::get();
                if (g_app.rgb_overlay && cam.w > 0) {
                    // The head when the frame was taken: arrival minus the camera's delay.
                    double t_dev = device_now() - (qpc_now() / qpc_freq() - t_rgb) - g_app.rgb_latency_ms / 1000.0;
                    M4 at, head_then = head_scene;
                    if (t_dev > 0 && engine_pose(t_dev, at)) head_then = mul(recenter, at);
                    double R[9];
                    rgbcal::rotation(cam.rv, R);
                    // Camera -> head: R(rv)^T and the camera centre (the renderer's camera looks
                    // down -z with y up, which is what R(rv) maps the head to).
                    M4 cam_head = identity();
                    for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) cam_head(i, j) = (float)R[j * 3 + i]; cam_head(i, 3) = (float)cam.c[i]; }
                    M4 cam_world = mul(head_then, cam_head);
                    float vfov = 2 * atanf((float)(cam.h / 2.0 / cam.f)) * 57.2958f;
                    rgb_pass = true;
                    render_view(mul(perspective(vfov, (float)cam.w / cam.h, 0.05f, 50), rigid_inverse(cam_world)));
                    rgb_pass = false;
                }
                // The composite: the frame, the virtual softened like the camera (blur about
                // 1.3 px at 1080p) with its grain (stronger in a dim room, as the sensor's).
                {
                    ID3D11RenderTargetView* t[] = {mr_rtv};
                    ctx->OMSetRenderTargets(1, t, nullptr);
                    static float seed = 0; seed = fmodf(seed + 17.13f, 997.0f);
                    float blur = g_app.camera_look ? 1.3f * MR_H / 1080.0f : 0.0f;
                    float grain = g_app.camera_look ? 0.025f + 0.05f * (1 - std::min(1.0f, g_app.room_level * 2)) : 0.0f;
                    Constants c{identity(), identity(), {0, grain, seed, 1}, {blur / MR_W, blur / MR_H, (float)MR_W, (float)MR_H}};
                    ctx->UpdateSubresource(cb, 0, nullptr, &c, 0, 0);
                    ID3D11Buffer* vbs[] = {screen_vb};
                    ctx->IASetInputLayout(layout);
                    ctx->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
                    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    ctx->VSSetShader(vs_tex, nullptr, 0);
                    ctx->PSSetShader(ps_comp, nullptr, 0);
                    ID3D11ShaderResourceView* srvs[] = {rgb_in_srv, virt_srv};
                    ctx->PSSetShaderResources(0, 2, srvs);
                    ID3D11SamplerState* ss[] = {samp};
                    ctx->PSSetSamplers(0, 1, ss);
                    ctx->Draw(6, 0);
                    ID3D11ShaderResourceView* none[] = {nullptr, nullptr};
                    ctx->PSSetShaderResources(0, 2, none);
                    ctx->VSSetShader(vs, nullptr, 0);
                    ctx->PSSetShader(ps, nullptr, 0);
                }
                ctx->OMSetRenderTargets(0, nullptr, nullptr);
                // Recording: copy now, read the copy made two frames ago (no GPU stall).
                bool rec_on; { std::lock_guard<std::mutex> l(rec::g_mu); rec_on = rec::g_status.on; }
                if (rec_on) {
                    int k = rec_k % 3;
                    ctx->CopyResource(rec_staging[k], mr_tex);
                    rec_stamp[k] = (int64_t)(t_rgb * qpc_freq());
                    if (rec_k >= 2) {
                        int r = (rec_k - 2) % 3;
                        D3D11_MAPPED_SUBRESOURCE m;
                        if (SUCCEEDED(ctx->Map(rec_staging[r], 0, D3D11_MAP_READ, 0, &m))) {
                            rec::push((const uint8_t*)m.pData, (int)m.RowPitch, rec_stamp[r]);
                            ctx->Unmap(rec_staging[r], 0);
                        }
                    }
                    rec_k++;
                } else rec_k = 0;
                if (g_app.photo_request) {
                    g_app.photo_request = false;
                    ctx->CopyResource(rec_staging[0], mr_tex);
                    D3D11_MAPPED_SUBRESOURCE m;
                    std::wstring png = rec::stamped(L".png");
                    bool ok = false;
                    if (SUCCEEDED(ctx->Map(rec_staging[0], 0, D3D11_MAP_READ, 0, &m))) {
                        ok = rec::save_png(png, (const uint8_t*)m.pData, MR_W, MR_H, (int)m.RowPitch);
                        ctx->Unmap(rec_staging[0], 0);
                    }
                    g_app.notice = ok ? "photo : " + rec::utf8(png) : "photo : échec";
                    g_app.notice_until = -1;  // the dashboard starts the 6 s
                }
            }
        }
        if (g_app.rec_request == 1) { g_app.rec_request = 0; if (MR_W) rec::start(MR_W, MR_H, g_app.rec_mbps, g_app.rec_audio); }
        else if (g_app.rec_request == 2) { g_app.rec_request = 0; rec::stop(); }
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
            if (last_present_seen && extra > 0 && extra < 600) missed_window += (long)extra;
            vblank_qpc = st.SyncQPCTime.QuadPart;
            if (prev_sync_qpc && st.SyncRefreshCount > prev_sync_count) {
                double p = qpc_ms(st.SyncQPCTime.QuadPart - prev_sync_qpc) / (st.SyncRefreshCount - prev_sync_count);
                if (p > 5 && p < 40) period_ms = 0.95 * period_ms + 0.05 * p;
            }
            prev_sync_qpc = st.SyncQPCTime.QuadPart; prev_sync_count = st.SyncRefreshCount;
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
            // Margin: a missed vblank means the frame was not ready in time: give it more.
            // Shown frames below the refresh rate also mean missed vblanks (the counter can miss them).
            double expected = since / period_ms;
            if (missed_window > 0 || shown < 0.97 * expected) { margin_ms = std::min(14.0, margin_ms + 1.5); quiet_since = qpc_now(); }
            else if (qpc_ms(qpc_now() - quiet_since) > 10000 && margin_ms > 3) { margin_ms -= 0.5; quiet_since = qpc_now(); }
            missed_window = 0;
            g_app.latch_margin_ms = (float)margin_ms;
            {
                std::vector<double> lh = lead_hist; std::sort(lh.begin(), lh.end());
                g_app.stats.lead_ms = lh.empty() ? 0 : (float)lh[lh.size() / 2];
                lead_hist.clear();
            }
            static float saved[7] = {-1, -1, -1, -1, -1, -1, -1};
            float cur[7] = {g_app.vfov, g_app.ipd_mm, g_app.floor_m, g_app.pitch_deg, g_app.eye_dy_cm, g_app.eye_dz_cm, g_app.yaw_deg};
            if (memcmp(saved, cur, sizeof cur)) { memcpy(saved, cur, sizeof cur); save_settings(); }
            auto pct = [](std::vector<double> v, double p) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(p * v.size()))]; };
            long updates = g_pose_updates - last_updates;
            float keep_lead = g_app.stats.lead_ms;
            g_app.stats = {(float)(shown * 1000.0 / since), (float)pct(latencies, 0.5), (float)pct(latencies, 0.99), (float)pct(cpu_ms, 0.5), (float)(updates * 1000.0 / since),
                           (float)pct(jerk, 0.5), (float)pct(jerk, 0.99), (float)pct(gain, 0.5), missed,
                           corr_n, corr_n ? corr_deg / corr_n : 0, corr_n ? corr_mm / corr_n : 0, keep_lead};
            if (!g_stats_paused) printf("      engine corrections: %d (mean %.2f deg, %.1f mm) %s\n", corr_n, corr_n ? corr_deg / corr_n : 0, corr_n ? corr_mm / corr_n : 0, g_app.smooth_corrections ? "blended" : "raw");
            corr_n = 0; corr_deg = corr_mm = 0;
            if (!g_stats_paused)
                printf("%5.1f fps  pose %4.0f Hz  pose->vblank p50 %4.1f p99 %4.1f ms  CPU %4.2f ms  missed %ld  head max %4.0f deg/s  shake p50 %.3f p99 %.3f deg  %s %.0f ms\n",
                       shown * 1000.0 / since, updates * 1000.0 / since, pct(latencies, 0.5), pct(latencies, 0.99), pct(cpu_ms, 0.5), missed, max_rate,
                       pct(jerk, 0.5), pct(jerk, 0.99), g_engine_predict ? "engine" : "ours", g_predict_ms);
            if (!g_stats_paused) printf("      presentation: composed %ld  overlay %ld  direct %ld  failed %ld\n", comp_counts[0], comp_counts[1], comp_counts[2], comp_counts[3]);
            if (!g_stats_paused) {
                std::lock_guard<std::mutex> l(hands::g_mu);
                auto& r = hands::g_result;
                printf("      hands (VITURE): %zu  %.0f/s  RGB fit %d/s %.0f ms  no pose %ld", r.hands.size(), r.rate, r.rgb_runs, r.rgb_ms, r.pose_fails);
                for (auto& h : r.hands) printf("  [%s pinch %.1f cm %.2f m/s%s]", h.right ? "R" : "L", h.pinch_m * 100, h.speed, h.rgb_seen ? " rgb" : "");
                printf("\n");
            }
            for (long& c : comp_counts) c = 0;
            if (g_engine_predict && !g_stats_paused) {
                printf("      engine lead used: %.2f of the head's motion (n %zu)  status:", pct(gain, 0.5), gain.size());
                for (int k = 0; k < 10; k++) if (status_counts[k]) printf(" [%d]=%ld", k - 1, status_counts[k]);
                printf("\n");
            }
            if (!g_stats_paused && (pred_err_still.size() + pred_err_slow.size() + pred_err_fast.size())) {
                printf("      prediction error (deg, p50/p95): still %.3f/%.3f (n %zu)  slow %.3f/%.3f (n %zu)  fast %.3f/%.3f (n %zu)\n",
                       pct(pred_err_still, 0.5), pct(pred_err_still, 0.95), pred_err_still.size(), pct(pred_err_slow, 0.5), pct(pred_err_slow, 0.95), pred_err_slow.size(),
                       pct(pred_err_fast, 0.5), pct(pred_err_fast, 0.95), pred_err_fast.size());
                g_app.stats.pred_err_still = (float)pct(pred_err_still, 0.95); g_app.stats.pred_err_moving = (float)pct(pred_err_slow, 0.95); g_app.stats.pred_err_fast = (float)pct(pred_err_fast, 0.95);
            }
            pred_err_still.clear(); pred_err_slow.clear(); pred_err_fast.clear();
            gain.clear();
            for (long& c : status_counts) c = 0;
            latencies.clear(); cpu_ms.clear(); jerk.clear(); shown = 0; missed = 0; max_rate = 0; last_updates = g_pose_updates;
            stats_t0 = qpc_now();
        }
        if (dash_on) dash::frame(g_app);
        // Tests: --snapshot N saves both tabs, dashboard_rgb.png then dashboard_diag.png.
        {
            static int snap_phase = 0;
            double el = qpc_ms(qpc_now() - frame_t0) / 1000;
            wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
            std::wstring dir = exe; dir = dir.substr(0, dir.find_last_of(L'\\') + 1);
            if (snapshot_s > 0 && snap_phase == 0 && el > snapshot_s) { g_app.tab_request = 0; snap_phase = 1; }
            else if (snap_phase == 1 && el > snapshot_s + 1) { g_app.snapshot = true; snap_phase = 2; }
            else if (snap_phase == 2 && el > snapshot_s + 2) { MoveFileExW((dir + L"dashboard.png").c_str(), (dir + L"dashboard_rgb.png").c_str(), MOVEFILE_REPLACE_EXISTING); g_app.tab_request = 1; snap_phase = 3; }
            else if (snap_phase == 3 && el > snapshot_s + 3) { g_app.snapshot = true; snap_phase = 4; }
            else if (snap_phase == 4 && el > snapshot_s + 4) { MoveFileExW((dir + L"dashboard.png").c_str(), (dir + L"dashboard_diag.png").c_str(), MOVEFILE_REPLACE_EXISTING); g_app.tab_request = 0; snap_phase = 5; snapshot_s = 0; }
        }
        if (false) {
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
        if (record_s > 0) {
            static int rec_phase = 0;
            double el = qpc_ms(qpc_now() - frame_t0) / 1000;
            if (rec_phase == 0 && el > 5) { g_app.rec_request = 1; g_app.photo_request = true; rec_phase = 1; }
            else if (rec_phase == 1 && el > 5 + record_s) { g_app.rec_request = 2; rec_phase = 2; }
        }
    }

    printf("stopping...\n");
    // Whatever hangs on the way out (engine, camera, GPU), the process ends.
    CreateThread(nullptr, 0, [](LPVOID) -> DWORD { Sleep(6000); fflush(stdout); TerminateProcess(GetCurrentProcess(), 0); return 0; }, nullptr, 0, nullptr);
    rec::stop();
    room::stop();
    vhands::stop();
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
