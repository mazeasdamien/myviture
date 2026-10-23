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
//   No keyboard shortcuts: everything is a button on the dashboard.

// dbghelp before ONNX Runtime's DirectML header, which undefines OPTIONAL.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#include "vio.hpp"
#include "dashboard.hpp"
#include "orange.hpp"
#include "voice.hpp"
#include "rgbcam.hpp"
#include "calib.hpp"
#include "hands.hpp"
#include "rgbcal.hpp"
#include "roommap.hpp"
#include "vhands.hpp"
#include "recorder.hpp"
#include "textpanel.hpp"
#include "human.hpp"
#include "mcp.hpp"
#include "depthany.hpp"
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
static M4 scale3(float x, float y, float z) { M4 r = identity(); r(0, 0) = x; r(1, 1) = y; r(2, 2) = z; return r; }
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
// Device time minus QPC time: the largest seen (the sample that arrived the soonest after
// it was taken), sliding down 1 ms/s for the clocks' drift. Extrapolating from the last
// callback instead would carry the USB's delivery jitter into every prediction target.
static std::atomic<double> g_dev_off{0};
static std::vector<float> g_imu_gaps;  // ms between IMU callbacks (under g_imu.mu)
static double device_now() {
    double t = g_imu_t, off = g_dev_off;
    if (t <= 0) return 0;
    return off != 0 ? qpc_now() / qpc_freq() + off : t + qpc_ms(qpc_now() - g_imu_qpc) / 1000.0;
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
  return float4(color.rgb * k, color.a);
}
// Textured quad (the uv rides in the normal slot).
Texture2D tex0 : register(t0);
SamplerState samp0 : register(s0);
struct VSOutT { float4 pos : SV_Position; float2 uv : TEXCOORD; };
VSOutT vs_tex(float3 p : POSITION, float3 n : NORMAL) { VSOutT o; o.pos = mul(mvp, float4(p, 1)); o.uv = n.xy; return o; }
float4 ps_tex(VSOutT i) : SV_Target { return float4(tex0.Sample(samp0, i.uv).rgb, 1); }
// Timewarp: the eye image rendered with an older pose, turned by the head's rotation since.
// The quad covers one eye's viewport (uv 0..1 in it); mvp holds M (late eye -> rendered eye
// directions), color = (f, aspect, uv offset, uv width) of that eye in the rendered image.
VSOutT vs_screen(float3 p : POSITION, float3 n : NORMAL) { VSOutT o; o.pos = float4(p.xy, 0, 1); o.uv = n.xy; return o; }
float4 ps_warp(VSOutT i) : SV_Target {
  float f = color.x, aspect = color.y;
  float2 ndc = float2(i.uv.x * 2 - 1, 1 - i.uv.y * 2);
  float3 d = float3(ndc.x * aspect / f, ndc.y / f, -1);
  float3 r = mul((float3x3)mvp, d);
  if (r.z > -1e-3) return float4(0, 0, 0, 1);
  float2 s = float2(f / aspect * r.x / -r.z, f * r.y / -r.z);
  float2 uv = float2(s.x * 0.5 + 0.5, 0.5 - s.y * 0.5);
  if (any(uv < 0) || any(uv > 1)) return float4(0, 0, 0, 1);
  return float4(tex0.Sample(samp0, float2(color.z + uv.x * color.w, uv.y)).rgb, 1);
}
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
static int g_mcp_port = 8765;  // the scene's MCP server (--mcp-port N, 0: off)
static std::string g_settings_path;
static void save_settings() {
    if (g_settings_path.empty()) return;
    if (FILE* f = fopen(g_settings_path.c_str(), "w")) { fprintf(f, "%.3f %.2f %.3f %.2f %.2f %.2f %.2f %.1f %.1f\n", g_app.vfov, g_app.ipd_mm, g_app.floor_m, g_app.pitch_deg, g_app.eye_dy_cm, g_app.eye_dz_cm, g_app.yaw_deg, g_app.predict_ms, g_app.ipd_adj_mm); fclose(f); }
}
static bool g_stats_paused = false;

static LRESULT CALLBACK wndproc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CLOSE: g_quit = true; return 0;
    case WM_SETCURSOR: SetCursor(nullptr); return TRUE;
    }
    return DefWindowProcW(h, msg, wp, lp);
}

static BOOL WINAPI on_ctrl(DWORD) { g_quit = true; return TRUE; }

// Engine callbacks: the pose is polled; its callback and the images also feed VITURE's
// hand tracking.
static void on_pose(float* d, double t) { vhands::feed_pose(d, t); }
// The engine's vsync callback (device seconds): its rate, to know what it marks.
static std::atomic<double> g_vsync_last{0}, g_vsync_period{0};
static void on_vsync(double t) {
    double last = g_vsync_last.exchange(t);
    double d = t - last;
    if (last > 0 && d > 0.002 && d < 0.2) { double p = g_vsync_period; g_vsync_period = p > 0 ? p * 0.95 + d * 0.05 : d; }
}
static void on_imu(float* d, double t) {
    int64_t now = qpc_now(), prev = g_imu_qpc.exchange(now);
    g_imu_t = t;
    double off = t - now / qpc_freq(), e = g_dev_off;
    g_dev_off = (e == 0 || off > e - 1e-6 || off < e - 0.05) ? off : e - 1e-6;  // reset after a clock jump
    std::lock_guard<std::mutex> l(g_imu.mu);
    if (prev) { g_imu_gaps.push_back((float)qpc_ms(now - prev)); if (g_imu_gaps.size() > 8000) g_imu_gaps.erase(g_imu_gaps.begin(), g_imu_gaps.begin() + 4000); }
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

// ------------------------------------------------------------------ crashes

/**
 * A crash (any thread): the exception, then the stack with function names and source lines
 * (myviture.pdb next to the exe) into crash.txt and the console, and a minidump crash.dmp.
 */
static LONG WINAPI on_crash(EXCEPTION_POINTERS* ep) {
    static std::atomic<bool> once{false};
    if (once.exchange(true)) return EXCEPTION_CONTINUE_SEARCH;
    wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring dir = exe; dir = dir.substr(0, dir.find_last_of(L'\\') + 1);
    FILE* f = _wfopen((dir + L"crash.txt").c_str(), L"w");
    auto out = [&](const char* fmt, auto... a) { printf(fmt, a...); if (f) fprintf(f, fmt, a...); };
    out("\nCRASH: exception 0x%08lx at %p, thread %lu\n", ep->ExceptionRecord->ExceptionCode, ep->ExceptionRecord->ExceptionAddress, GetCurrentThreadId());
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    SymInitialize(proc, nullptr, TRUE);
    CONTEXT ctx = *ep->ContextRecord;
    STACKFRAME64 sf{};
    sf.AddrPC.Offset = ctx.Rip; sf.AddrPC.Mode = AddrModeFlat;
    sf.AddrFrame.Offset = ctx.Rbp; sf.AddrFrame.Mode = AddrModeFlat;
    sf.AddrStack.Offset = ctx.Rsp; sf.AddrStack.Mode = AddrModeFlat;
    for (int i = 0; i < 40 && StackWalk64(IMAGE_FILE_MACHINE_AMD64, proc, GetCurrentThread(), &sf, &ctx, nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr); i++) {
        DWORD64 a = sf.AddrPC.Offset;
        if (!a) break;
        alignas(SYMBOL_INFO) char buf[sizeof(SYMBOL_INFO) + 256] = {};
        SYMBOL_INFO* si = (SYMBOL_INFO*)buf; si->SizeOfStruct = sizeof(SYMBOL_INFO); si->MaxNameLen = 255;
        DWORD64 d = 0; DWORD dl = 0;
        IMAGEHLP_LINE64 ln{}; ln.SizeOfStruct = sizeof ln;
        IMAGEHLP_MODULE64 mi{}; mi.SizeOfStruct = sizeof mi;
        const char* mod = SymGetModuleInfo64(proc, a, &mi) ? mi.ModuleName : "?";
        bool named = SymFromAddr(proc, a, &d, si);
        if (SymGetLineFromAddr64(proc, a, &dl, &ln)) out("  %2d %s!%s  %s:%lu\n", i, mod, named ? si->Name : "?", ln.FileName, ln.LineNumber);
        else out("  %2d %s!%s +0x%llx\n", i, mod, named ? si->Name : "?", (unsigned long long)d);
    }
    HANDLE df = CreateFileW((dir + L"crash.dmp").c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (df != INVALID_HANDLE_VALUE) {
        MINIDUMP_EXCEPTION_INFORMATION mei{GetCurrentThreadId(), ep, FALSE};
        MiniDumpWriteDump(proc, GetCurrentProcessId(), df, MiniDumpNormal, &mei, nullptr, nullptr);
        CloseHandle(df);
    }
    if (f) fclose(f);
    fflush(stdout);
    return EXCEPTION_CONTINUE_SEARCH;
}

// ------------------------------------------------------------------ main

int main(int argc, char** argv) {
    SetUnhandledExceptionFilter(on_crash);
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
        else if (!strcmp(argv[i], "--mcp-port") && i + 1 < argc) g_mcp_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-rgb")) g_no_rgb = true;
        else if (!strcmp(argv[i], "--rgb-1080")) rgbcam::g_allow_1080 = true;
        else if (!strcmp(argv[i], "--no-vhands")) g_no_vhands = true;
        else if (!strcmp(argv[i], "--snapshot") && i + 1 < argc) snapshot_s = atof(argv[++i]);
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    SetConsoleCtrlHandler(on_ctrl, TRUE);
    timeBeginPeriod(1);

    // Without the glasses the engine ends the process when it starts: wait for their screen.
    {
        auto plugged = []() { for (auto& m : monitors()) if (m.name.find(L"VITURE") != std::wstring::npos) return true; return false; };
        bool waited = false;
        for (int i = 0; !plugged() && !g_quit; i++) {
            if (i % 15 == 0) printf("glasses not plugged in (no VITURE screen): waiting for them...\n");
            waited = true;
            Sleep(2000);
        }
        if (g_quit) return 0;
        // Just plugged in: their screen comes first; started then, the engine finds neither the
        // IMU nor the tracking cameras ready (3DoF, then spinning). 10 s for them to boot.
        if (waited) { printf("glasses plugged in: 10 s for them to start...\n"); Sleep(10000); }
    }
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
    {
        unsigned char ra[48] = {}, rb[128] = {};
        if (vio::get_render_para(ra, rb) == 0) {
            const float* fa = (const float*)ra; const float* fb = (const float*)rb;
            std::string o = "engine render para A (12 floats):";
            char b[32];
            for (int i = 0; i < 12; i++) { snprintf(b, sizeof b, " %.5g", fa[i]); o += b; }
            o += "\nengine render para B (32 floats):";
            for (int i = 0; i < 32; i++) { snprintf(b, sizeof b, " %.5g", fb[i]); o += b; }
            const double* da = (const double*)ra; const double* dbl = (const double*)rb;
            o += "\nengine render para A (6 doubles):";
            for (int i = 0; i < 6; i++) { snprintf(b, sizeof b, " %.6g", da[i]); o += b; }
            o += "\nengine render para B (16 doubles):";
            for (int i = 0; i < 16; i++) { snprintf(b, sizeof b, " %.6g", dbl[i]); o += b; }
            hands::logf("%s\n", o.c_str());
        } else hands::logf("engine render para: not available\n");
    }

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
    Com<ID3D11PixelShader> ps_shadow, ps_comp, ps_warp;
    Com<ID3D11VertexShader> vs_screen;
    {
        Com<ID3DBlob> b3, b4;
        if (FAILED(D3DCompile(SHADER, strlen(SHADER), nullptr, nullptr, nullptr, "ps_warp", "ps_5_0", 0, 0, &b3, &err)) ||
            FAILED(D3DCompile(SHADER, strlen(SHADER), nullptr, nullptr, nullptr, "vs_screen", "vs_5_0", 0, 0, &b4, &err))) {
            fprintf(stderr, "shader: %s\n", err ? (const char*)err->GetBufferPointer() : "?");
            return 1;
        }
        CHECK(dev->CreatePixelShader(b3->GetBufferPointer(), b3->GetBufferSize(), nullptr, &ps_warp));
        CHECK(dev->CreateVertexShader(b4->GetBufferPointer(), b4->GetBufferSize(), nullptr, &vs_screen));
    }
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
    D3D11_BUFFER_DESC cbd{}; cbd.ByteWidth = sizeof(Constants); cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; cbd.Usage = D3D11_USAGE_DEFAULT;
    CHECK(dev->CreateBuffer(&cbd, nullptr, &cb));
    D3D11_RASTERIZER_DESC rd{}; rd.FillMode = D3D11_FILL_SOLID; rd.CullMode = D3D11_CULL_NONE; rd.FrontCounterClockwise = TRUE; rd.DepthClipEnable = TRUE;
    Com<ID3D11RasterizerState> rs; CHECK(dev->CreateRasterizerState(&rd, &rs));
    D3D11_DEPTH_STENCIL_DESC dsd{}; dsd.DepthEnable = TRUE; dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL; dsd.DepthFunc = D3D11_COMPARISON_LESS;
    Com<ID3D11DepthStencilState> dss; CHECK(dev->CreateDepthStencilState(&dsd, &dss));
    // Timewarp: the eyes rendered here first, then turned into the back buffer.
    Com<ID3D11Texture2D> eye_tex;
    Com<ID3D11RenderTargetView> eye_rtv;
    Com<ID3D11ShaderResourceView> eye_srv;
    UINT eye_w = 0, eye_h = 0;
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
        if (g_mcp_port > 0) mcp::start(g_mcp_port, std::string(dir.begin(), dir.end()) + "myviture_robot_frame.txt");
        g_settings_path = std::string(dir.begin(), dir.end()) + "myviture_settings.txt";
        if (FILE* f = fopen(g_settings_path.c_str(), "r")) {
            float v[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            int n = fscanf(f, "%f %f %f %f %f %f %f %f %f", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8]);
            if (n >= 5) { g_app.vfov = v[0]; g_app.ipd_mm = v[1]; g_app.floor_m = v[2]; g_app.pitch_deg = v[3]; g_app.eye_dy_cm = v[4]; }
            if (n >= 7) { g_app.eye_dz_cm = v[5]; g_app.yaw_deg = v[6]; }
            if (n >= 8) g_app.predict_ms = v[7];
            if (n >= 9) g_app.ipd_adj_mm = v[8];
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
            // What the RGB camera sees, labelled (floor, walls, tables, doors, windows...): its
            // frames carry QPC times, the engine's poses device times.
            auto pose_qpc = [](double tq, float* out) {
                double td = device_now() - (qpc_now() / qpc_freq() - tq);
                M4 r; if (td <= 0 || !engine_pose(td, r)) return false; memcpy(out, r.m, sizeof r.m); return true;
            };
            // Dense depth from the RGB camera, in metres from the stereo's points.
            if (!g_no_room) dany::start(dir, pose_qpc);
        }
        else { std::lock_guard<std::mutex> l(hands::g_mu); hands::g_result.status = "calibration.bin introuvable (dossier du projet)"; }
    }
    {
        char exe[MAX_PATH];
        GetModuleFileNameA(nullptr, exe, MAX_PATH);
        std::string dir = exe;
        dir = dir.substr(0, dir.find_last_of('\\') + 1);
        if (orange::load_config(dir + "..\\.env.local")) { orange::refresh_spend(); voice::start(); }
        else { std::lock_guard<std::mutex> l(orange::g_costs.mu); orange::g_costs.status = ".env.local introuvable (à côté du dossier native)"; }
    }

    // Statistics: pose read time per present, matched to the vblank it was shown at.
    struct Pending { UINT present; int64_t pose_qpc; };
    std::vector<Pending> pending;
    std::vector<double> latencies, cpu_ms, jerk, gain;
    std::vector<double> pred_err_still, pred_err_slow, pred_err_fast;  // degrees, head < 15, 15-60, > 60 deg/s
    // The same after the timewarp: the late prediction each warped frame showed.
    struct LatePred { double t; Q q; float rate; };
    std::deque<LatePred> tw_late;
    std::vector<double> tw_err_slow, tw_err_fast;
    long status_counts[10] = {};
    int corr_n = 0; float corr_deg = 0, corr_mm = 0;
    UINT last_refresh = 0, last_present_seen = 0;
    long missed = 0, shown = 0, last_updates = 0;
    int64_t stats_t0 = qpc_now(), frame_t0 = qpc_now();
    float rate_dps = 0, max_rate = 0;
    M4 recenter = identity();
    bool dash_on = true;

    printf("rendering on the glasses, screen latency %.0f ms\n", g_app.predict_ms);
    // vblank timing from the frame statistics, for late latching
    int64_t vblank_qpc = 0;
    double period_ms = 1000.0 / 60;
    int64_t prev_sync_qpc = 0; UINT prev_sync_count = 0;
    double margin_ms = g_app.latch_margin_ms;
    long missed_window = 0; int64_t quiet_since = qpc_now();
    std::vector<double> lead_hist;
    // The smoother's reference: the pose last shown for a target time (the warp's when it ran).
    double sm_prev_target = 0; Q sm_prev_q{0, 0, 0, 1}; V3 sm_prev_p{0, 0, 0};
    room::WalkGrid walk; uint32_t walk_version = 0; float walk_floor = NAN, walk_floor_guess = 0;  // the room model, tracking frame
    std::string walk_src;  // where its floor comes from
    long fallbacks = 0, missed_fast = 0;
    // The stability test's cross, in the tracking frame (so a recentre does not move it).
    V3 stab_pos{0, 0, 0}; float stab_yaw = 0;
    panel::Panel stab_panel;
    bool stab_panel_ok = stab_panel.init(dev);
    int stab_applied = -1;
    // The virtual human and the room's planes as geometry (occluders, lines).
    human::Renderer human_r;
    human::Walker walker;
    {
        wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring dir = exe; dir = dir.substr(0, dir.find_last_of(L'\\') + 1);
        if (human_r.init(dev, ctx, dir)) {
            printf("human: %zu bones, %zu parts, idle %u frames, walk %u frames (%.2f m a cycle)\n", human_r.model.joint_node.size(), human_r.model.parts.size(),
                   human_r.model.frames, human_r.model.walk_frames, human_r.model.walk_stride);
            if (human_r.model.walk_stride > 0.3f) walker.stride = human_r.model.walk_stride;  // the feet do not slide
        }
        else printf("human: %s\n", human_r.error.c_str());
    }
    constexpr UINT ROOM_VERTS = 65536;
    Com<ID3D11Buffer> occ_vb, line_vb;
    {
        D3D11_BUFFER_DESC bd{}; bd.ByteWidth = ROOM_VERTS * sizeof(Vertex); bd.BindFlags = D3D11_BIND_VERTEX_BUFFER; bd.Usage = D3D11_USAGE_DYNAMIC; bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        CHECK(dev->CreateBuffer(&bd, nullptr, &occ_vb));
        CHECK(dev->CreateBuffer(&bd, nullptr, &line_vb));
    }
    UINT occ_count = 0, line_count = 0;
    uint32_t room_geo_version = ~0u;
    // The 3D debug view: its target, its point buffers (instanced cubes), a line buffer
    // rewritten every frame (head, trail, paths) and a wireframe state for the planes.
    Com<ID3D11Texture2D> v3_tex, v3_depth;
    Com<ID3D11RenderTargetView> v3_rtv;
    Com<ID3D11ShaderResourceView> v3_srv;
    Com<ID3D11DepthStencilView> v3_dsv;
    {
        D3D11_TEXTURE2D_DESC td{}; td.Width = g_app.v3_w; td.Height = g_app.v3_h; td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1; td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        CHECK(dev->CreateTexture2D(&td, nullptr, &v3_tex)); CHECK(dev->CreateRenderTargetView(v3_tex, nullptr, &v3_rtv)); CHECK(dev->CreateShaderResourceView(v3_tex, nullptr, &v3_srv));
        td.Format = DXGI_FORMAT_D32_FLOAT; td.BindFlags = D3D11_BIND_DEPTH_STENCIL;
        CHECK(dev->CreateTexture2D(&td, nullptr, &v3_depth)); CHECK(dev->CreateDepthStencilView(v3_depth, nullptr, &v3_dsv));
        g_app.v3_srv = v3_srv.p;
    }
    constexpr UINT V3_PTS = 120000, V3_LINES = 16384;
    Com<ID3D11Buffer> v3_ev_vb, v3_live_vb, v3_line_vb;
    {
        D3D11_BUFFER_DESC bd{}; bd.ByteWidth = V3_PTS * 24; bd.BindFlags = D3D11_BIND_VERTEX_BUFFER; bd.Usage = D3D11_USAGE_DYNAMIC; bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        CHECK(dev->CreateBuffer(&bd, nullptr, &v3_ev_vb)); CHECK(dev->CreateBuffer(&bd, nullptr, &v3_live_vb));
        bd.ByteWidth = V3_LINES * sizeof(Vertex);
        CHECK(dev->CreateBuffer(&bd, nullptr, &v3_line_vb));
    }
    UINT v3_ev_count = 0, v3_live_count = 0, v3_dense_count = 0;
    uint32_t v3_ev_version = 0, v3_live_version = 0, v3_dense_version = 0;
    Com<ID3D11Buffer> v3_dense_vb;
    { D3D11_BUFFER_DESC bd{}; bd.ByteWidth = V3_PTS * 24; bd.BindFlags = D3D11_BIND_VERTEX_BUFFER; bd.Usage = D3D11_USAGE_DYNAMIC; bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE; CHECK(dev->CreateBuffer(&bd, nullptr, &v3_dense_vb)); }
    Com<ID3D11RasterizerState> rs_wire;
    Com<ID3D11BlendState> alpha_blend;
    {
        D3D11_BLEND_DESC bd{}; auto& rt = bd.RenderTarget[0];
        rt.BlendEnable = TRUE; rt.SrcBlend = D3D11_BLEND_SRC_ALPHA; rt.DestBlend = D3D11_BLEND_INV_SRC_ALPHA; rt.BlendOp = D3D11_BLEND_OP_ADD;
        rt.SrcBlendAlpha = D3D11_BLEND_ONE; rt.DestBlendAlpha = D3D11_BLEND_ZERO; rt.BlendOpAlpha = D3D11_BLEND_OP_ADD; rt.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        CHECK(dev->CreateBlendState(&bd, &alpha_blend));
    }
    { D3D11_RASTERIZER_DESC rd{}; rd.FillMode = D3D11_FILL_WIREFRAME; rd.CullMode = D3D11_CULL_NONE; rd.DepthClipEnable = TRUE; CHECK(dev->CreateRasterizerState(&rd, &rs_wire)); }
    std::vector<V3> head_trail;  // tracking frame, every 0.1 s
    double trail_t = 0;
    double human_last_t = qpc_now() / qpc_freq(), human_t0 = human_last_t;
    // Showing an object: going to it, then pointing at it (tracking frame).
    struct Show { bool on = false; int id = 0; float a[3] = {0, 0, 0}; std::string label; double arrived = 0; } show;
    // Three cubes (tracking frame): dropped in front of the wearer once the floor is known,
    // falling on it (or on a table), taken with a pinch and thrown when let go.
    struct Cube { V3 p, v; int held = -1; V3 off; float col[3]; bool hover = false; float ground = NAN; };
    std::shared_ptr<const std::unordered_set<uint64_t>> solid;  // the room's confirmed voxels (tracking frame)
    std::vector<Cube> cubes;
    bool cubes_dropped = false;
    V3 pinch_prev[2] = {}; double pinch_prev_t = 0;
    constexpr float CUBE = 0.08f;
    while (!g_quit && !g_app.quit) {
        g_engine_predict = g_app.engine_predict;
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
        // The stabilisation setting compared in the stability test (dashboard buttons).
        {
            if (g_app.stab_mode != stab_applied) {
                stab_applied = g_app.stab_mode;
                g_app.smooth_corrections = stab_applied == 0 || stab_applied == 2;
                g_app.timewarp = stab_applied == 0 || stab_applied == 1;
            }
        }
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
            // With the timewarp the frame starts where the last one was warped, just before a
            // vblank: it aims at the next one (rendering, then the warp's margin, must fit).
            double need_ms = g_app.timewarp ? g_app.tw_margin_ms + 4.0 : 0.3;
            while (to_vblank_ms < need_ms) to_vblank_ms += period_ms;
        }
        // Predict to the vblank this frame should catch, plus the screen's own latency.
        double g_predict_ms = to_vblank_ms + g_app.predict_ms;
        lead_hist.push_back(g_predict_ms);
        int64_t t_start = qpc_now();

        // Pose now, rotation extrapolated to when this frame reaches the screen.
        // The engine's prediction; the newest polled pose only if it fails.
        Q q; V3 p;
        bool have = predicted_pose(g_predict_ms, q, p, rate_dps);  // extrapolated, from the ~520 Hz polled poses
        double tw_target = 0; Q tw_q{0, 0, 0, 1};  // the engine's prediction this frame was rendered with
        {
            // The engine's own prediction (IMU-rate), for when this frame is on screen.
            M4 r;
            // From the newest IMU sample: its age plus the lead to the screen.
            double base = g_imu_t, target = device_now() + g_predict_ms / 1000.0;
            M4 b;
            bool got_base = engine_pose(base, b);
            // It fails a few times a second ("imu preintegration"): again 2 ms earlier, then
            // our extrapolation; the warp still re-reads the engine for that moment.
            bool ok = engine_pose(target, r) || engine_pose(target - 0.002, r);
            if (!ok && have) { fallbacks++; tw_target = target; tw_q = q; }
            if (ok) {
                q = mat_q(r); p = {r(0, 3), r(1, 3), r(2, 3)}; have = true;
                tw_target = target; tw_q = q;
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
            while (!tw_late.empty() && tw_late.front().t < now_dev - 0.08) {
                M4 truth;
                if (engine_pose(tw_late.front().t, truth) && tw_late.front().rate > 15) {
                    Q d = qmul(mat_q(truth), qconj(tw_late.front().q));
                    float ang = 2 * atan2f(sqrtf(d.x * d.x + d.y * d.y + d.z * d.z), fabsf(d.w)) * 57.2958f;
                    (tw_late.front().rate > 60 ? tw_err_fast : tw_err_slow).push_back(ang);
                }
                tw_late.pop_front();
            }
            if (tw_late.size() > 600) tw_late.clear();
        }
        // The engine's visual update (25 Hz) revises the pose it predicted from the IMU:
        // a step every camera frame, seen as a twitch. Asking again for last frame's
        // target gives that revision; an offset absorbs it and fades out, so the scene
        // slides to the corrected place instead of jumping. Big steps (relocalisation,
        // recentre) are let through.
        {
            double& prev_target = sm_prev_target;
            Q& prev_q_at = sm_prev_q;
            V3& prev_p_at = sm_prev_p;
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
                    // Absorbed only while the head is nearly still, where a step shows; turning,
                    // the motion hides it and holding it back would be lag (on a see-through
                    // display, error kept against the real world).
                    if (g_app.smooth_corrections && ang < 3 && dist < 0.05) {
                        if (rate_dps < 30) {
                            oq = qnorm(qmul(qconj(dq), oq));
                            op = {op.x - dp.x, op.y - dp.y, op.z - dp.z};
                        }
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
        // Tracking guard: a head cannot turn more than ~15 degrees in one frame (900 deg/s)
        // nor move 25 cm. When the engine's pose does (it lost its tracking and spins), the
        // last good pose is held until it has been sane again for half a second, instead of
        // spinning the scene.
        {
            static Q good_q{0, 0, 0, 1};
            static V3 good_p{0, 0, 0};
            static Q last_q{0, 0, 0, 1};
            static V3 last_p{0, 0, 0};
            static bool primed = false;
            static int sane = 0;
            if (have) {
                Q d = qmul(q, qconj(last_q));
                float ang = 2 * atan2f(sqrtf(d.x * d.x + d.y * d.y + d.z * d.z), fabsf(d.w)) * 57.2958f;
                float dist = sqrtf((p.x - last_p.x) * (p.x - last_p.x) + (p.y - last_p.y) * (p.y - last_p.y) + (p.z - last_p.z) * (p.z - last_p.z));
                // Also when the IMU has gone quiet (its stream stopped): the engine would go on
                // extrapolating the last rotation, the scene spinning.
                bool imu_quiet = g_imu_qpc && qpc_ms(qpc_now() - g_imu_qpc) > 200;
                bool wild = primed && (ang > 15 || dist > 0.25f || rate_dps > 900 || imu_quiet);
                last_q = q; last_p = p; primed = true;
                if (wild) { sane = 0; if (!g_app.tracking_lost) { g_app.tracking_lost = true; g_app.tracking_losses++; } }
                else if (++sane >= 30) g_app.tracking_lost = false;
                if (g_app.tracking_lost) { q = good_q; p = good_p; }
                else { good_q = q; good_p = p; }
            }
        }
        M4 head = mul(translate(p), qmat(q));  // head -> world
        if (have && g_app.recenter) {
            // Scene frame: the head's position and heading at recenter, gravity kept.
            float fx = -head(0, 2), fz = -head(2, 2);
            float yaw = atan2f(-fx, -fz);
            recenter = mul(rot_y(-yaw), translate({-p.x, -p.y, -p.z}));
            g_app.recenter = false;
        }
        M4 head_scene = mul(recenter, head);
        if (g_app.stab_request) {
            // 2 m ahead along the gaze (level), facing the wearer, its crossing at eye height.
            if (g_app.stab_request == 1 && have) {
                float fx = -head(0, 2), fz = -head(2, 2), n = sqrtf(fx * fx + fz * fz);
                if (n > 1e-3f) {
                    fx /= n; fz /= n;
                    stab_pos = {p.x + 2 * fx, p.y, p.z + 2 * fz};
                    stab_yaw = atan2f(-fx, -fz);
                    g_app.stab_test = true;
                    hands::logf("stability test: cross placed, screen latency %.0f ms, setting %d\n", g_app.predict_ms, g_app.stab_mode);
                }
            } else if (g_app.stab_request == 2) g_app.stab_test = false;
            g_app.stab_request = 0;
        }
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
                    // Predicted to the display: the answer's age (plus ~30 ms of camera and
                    // inference before it arrived, and the screen's latency) at the filtered speed.
                    float ahead = (float)std::clamp(qpc_now() / qpc_freq() - vo.t + 0.03 + g_app.predict_ms / 1000.0, 0.0, 0.12);
                    for (int k = 0; k < hands::N_JOINTS; k++) for (int a = 0; a < 3; a++) w[k][a] += (float)filt[side][k][a].dx * ahead;
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
        room::g_eye_height = g_app.floor_m;
        dany::g_latency_s = g_app.rgb_latency_ms / 1000.0f;
        if (g_app.room_request) {
            if (g_app.room_request == 1) { std::lock_guard<std::mutex> l(room::g_label_mu); room::g_anchor_label = g_app.anchor_label; }
            room::g_request = g_app.room_request; g_app.room_request = 0;
        }
        std::vector<room::Plane> planes;
        std::vector<room::Anchor> anchors;
        std::vector<float> new_mesh;
        bool mesh_changed = false;
        {
            std::lock_guard<std::mutex> l(room::g.mu);
            planes = room::g.planes;
            anchors = room::g.anchors;
            if (room::g.walk_version != walk_version) { walk_version = room::g.walk_version; walk = room::g.walk; walk_floor = room::g.has_floor ? room::g.floor_y : NAN; walk_floor_guess = room::g.floor_y; walk_src = room::g.floor_source; }
            // The new mesh is taken (a swap), and copied to the GPU after the lock.
            if (room::g.mesh_version != mesh_version) { mesh_version = room::g.mesh_version; new_mesh.swap(room::g.mesh); mesh_changed = true; }
            if (room::g.version != inst_version && !room::g.points.empty()) {
                inst_version = room::g.version;
                solid = room::g.solid;
                D3D11_MAPPED_SUBRESOURCE m;
                if (SUCCEEDED(ctx->Map(inst_vb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
                    inst_count = std::min<UINT>(MAX_INST, (UINT)(room::g.points.size() / 6));
                    memcpy(m.pData, room::g.points.data(), inst_count * 24);
                    ctx->Unmap(inst_vb, 0);
                }
            }
        }
        if (walk_version != room_geo_version) {
            // Occluders (depth only): the floor, each wall from the floor to 2.5 m (or its top),
            // each table as a box down to the floor. Lines: the floor-wall seams and the walls'
            // edges, the tables' outlines, the walkable area's border.
            room_geo_version = walk_version;
            std::vector<Vertex> occ, lines;
            auto quad = [&](V3 a, V3 b, V3 c2, V3 d) { for (V3 v : {a, b, c2, a, c2, d}) occ.push_back({v.x, v.y, v.z, 0, 1, 0}); };
            auto seg = [&](V3 a, V3 b) { lines.push_back({a.x, a.y, a.z, 0, 1, 0}); lines.push_back({b.x, b.y, b.z, 0, 1, 0}); };
            if (!planes.empty()) {
                float fy = std::isnan(walk_floor) ? walk_floor_guess : walk_floor;  // walls and tables before the floor is seen: a guessed one
                if (walk.nx && !std::isnan(walk_floor)) quad({walk.x0, fy, walk.z0}, {walk.x0 + walk.nx * walk.cell, fy, walk.z0}, {walk.x0 + walk.nx * walk.cell, fy, walk.z0 + walk.nz * walk.cell}, {walk.x0, fy, walk.z0 + walk.nz * walk.cell});
                for (auto& pl : planes) {
                    if (pl.kind == 2) {
                        float top = std::max(pl.top, fy + 2.5f);
                        V3 a = {pl.d * pl.nx - pl.nz * pl.t0, fy, pl.d * pl.nz + pl.nx * pl.t0}, b = {pl.d * pl.nx - pl.nz * pl.t1, fy, pl.d * pl.nz + pl.nx * pl.t1};
                        quad(a, b, {b.x, top, b.z}, {a.x, top, a.z});
                        float lt = std::max(pl.top, fy + 0.5f);
                        seg({a.x, fy + 0.005f, a.z}, {b.x, fy + 0.005f, b.z});
                        seg({a.x, fy, a.z}, {a.x, lt, a.z}); seg({b.x, fy, b.z}, {b.x, lt, b.z});
                    } else if (pl.kind == 1) {
                        V3 c[4] = {{pl.minx, pl.y, pl.minz}, {pl.maxx, pl.y, pl.minz}, {pl.maxx, pl.y, pl.maxz}, {pl.minx, pl.y, pl.maxz}};
                        quad(c[0], c[1], c[2], c[3]);
                        for (int k = 0; k < 4; k++) {
                            V3 a = c[k], b = c[(k + 1) % 4];
                            quad({a.x, fy, a.z}, {b.x, fy, b.z}, b, a);
                            seg(a, b);
                        }
                    }
                }
                // The walkable area's border: cell sides between walkable and not.
                for (int j = 0; j < walk.nz && lines.size() + 8 < ROOM_VERTS; j++) for (int i = 0; i < walk.nx; i++) {
                    if (walk.c[j * walk.nx + i] != 1) continue;
                    float x0 = walk.x0 + i * walk.cell, z0 = walk.z0 + j * walk.cell, x1 = x0 + walk.cell, z1 = z0 + walk.cell, y = fy + 0.004f;
                    auto walkable = [&](int a, int b) { return a >= 0 && b >= 0 && a < walk.nx && b < walk.nz && walk.c[b * walk.nx + a] == 1; };
                    if (!walkable(i - 1, j)) seg({x0, y, z0}, {x0, y, z1});
                    if (!walkable(i + 1, j)) seg({x1, y, z0}, {x1, y, z1});
                    if (!walkable(i, j - 1)) seg({x0, y, z0}, {x1, y, z0});
                    if (!walkable(i, j + 1)) seg({x0, y, z1}, {x1, y, z1});
                }
            }
            occ_count = (UINT)std::min<size_t>(occ.size(), ROOM_VERTS);
            line_count = (UINT)std::min<size_t>(lines.size(), ROOM_VERTS) & ~1u;
            UINT all_lines = line_count;
            D3D11_MAPPED_SUBRESOURCE m;
            if (occ_count && SUCCEEDED(ctx->Map(occ_vb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) { memcpy(m.pData, occ.data(), occ_count * sizeof(Vertex)); ctx->Unmap(occ_vb, 0); }
            if (all_lines && SUCCEEDED(ctx->Map(line_vb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) { memcpy(m.pData, lines.data(), all_lines * sizeof(Vertex)); ctx->Unmap(line_vb, 0); }
        }
        // Voice commands: « scan X » names an anchor at the gaze, « où est X » shows it, « viens ».
        voice::g_on = g_app.voice;
        {
            std::deque<voice::Command> heard;
            { std::lock_guard<std::mutex> l(voice::g_mu); heard.swap(voice::g_cmds); }
            for (auto& vc : heard) {
                if (vc.op == "scan") {
                    g_app.anchor_label = vc.label.empty() ? "objet " + std::to_string(anchors.size() + 1) : vc.label;
                    g_app.room_request = 1;
                    voice::set_status("ancre « " + g_app.anchor_label + " » là où tu regardes");
                } else if (vc.op == "come") g_app.human_request = 1;
                else if (vc.op == "show") {
                    // The anchor whose name holds what was said (or the reverse), accents aside.
                    int id = 0;
                    for (auto& a : anchors) {
                        std::string n = voice::fold(a.label);
                        if (!n.empty() && (n.find(vc.label) != std::string::npos || vc.label.find(n) != std::string::npos)) { id = a.id; break; }
                    }
                    if (id) { g_app.show_request = id; voice::set_status("l'humain va montrer « " + vc.label + " »"); }
                    else voice::set_status("pas d'ancre « " + vc.label + " » : dis « scan " + vc.label + " » en la regardant");
                }
            }
        }
        // The human: placed in front of the wearer once the floor is known, then walking.
        {
            double now_s = qpc_now() / qpc_freq();
            float dt = (float)std::min(0.1, now_s - human_last_t);
            human_last_t = now_s;
            bool can = g_app.human && human_r.ok && !std::isnan(walk_floor) && walk.nx && have;
            // Once the floor is known: standing on it 1 m in front of the wearer, facing them (no
            // walkable area needed for that; walking needs one).
            if (g_app.human && human_r.ok && !std::isnan(walk_floor) && have && !walker.placed) {
                float fx = -head(0, 2), fz = -head(2, 2), n = sqrtf(fx * fx + fz * fz);
                if (n > 1e-3f) {
                    walker.x = p.x + fx / n; walker.z = p.z + fz / n; walker.heading = atan2f(p.x - walker.x, p.z - walker.z);
                    walker.placed = true; walker.state = "attend"; walker.next_wander = now_s + 8;
                }
            }
            if (can && g_app.human_request == 1) {
                float fx = -head(0, 2), fz = -head(2, 2), n = sqrtf(fx * fx + fz * fz);
                if (n > 1e-3f) { fx /= n; fz /= n; walker.go_to(walk, p.x + 1.6f * fx, p.z + 1.6f * fz); }
                g_app.human_request = 0;
            }
            // Show an anchor: go 0.6 m from it, on the wearer's side, face it and point (10 s).
            if (g_app.show_request && walker.placed) {
                for (auto& a : anchors) if (a.id == g_app.show_request) {
                    float dx = p.x - a.p[0], dz = p.z - a.p[2], n = sqrtf(dx * dx + dz * dz);
                    if (n < 0.05f) { dx = walker.x - a.p[0]; dz = walker.z - a.p[2]; n = std::max(0.05f, sqrtf(dx * dx + dz * dz)); }
                    float gx = a.p[0] + 0.6f * dx / n, gz = a.p[2] + 0.6f * dz / n;
                    // By the walkable area if it has a way, else straight there.
                    if (!walk.nx || !walker.go_to(walk, gx, gz)) { walker.path = {{walker.x, walker.z}, {gx, gz}}; walker.next = 1; walker.state = "marche"; }
                    show.on = true; show.id = a.id; memcpy(show.a, a.p, sizeof show.a); show.label = a.label; show.arrived = 0;
                    walker.look = true; walker.look_x = a.p[0]; walker.look_z = a.p[2];
                }
                g_app.show_request = 0;
            }
            if (show.on) {
                if (walker.path.empty() && show.arrived == 0) show.arrived = now_s;
                if (show.arrived > 0) walker.state = "montre « " + show.label + " »";
                if (show.arrived > 0 && now_s - show.arrived > 10) { show.on = false; walker.look = false; walker.state = "attend"; }
            }
            if (can && g_app.human_request == 2) { if (!walker.go_to(walk, g_app.human_target[0], g_app.human_target[1])) walker.state = "pas de chemin"; g_app.human_request = 0; }
            if (human_r.model.walk_stride > 0.3f) walker.stride = human_r.model.walk_stride * g_app.human_height / 1.72f;  // a smaller body, shorter steps
            if (walker.placed && !std::isnan(walk_floor) && have) walker.update(now_s, dt, walk, p.x, p.z, g_app.human_wander && !show.on);
            g_app.human_state = !g_app.human ? "caché" : !human_r.ok ? human_r.error : std::isnan(walk_floor) ? "attend le sol (regarde le sol)" : walker.placed ? walker.state : "pas de place pour lui";
            if (show.on && show.arrived == 0) g_app.human_state = "va montrer « " + show.label + " »";
            g_app.human_pos[0] = walker.x; g_app.human_pos[1] = std::isnan(walk_floor) ? 0 : walk_floor; g_app.human_pos[2] = walker.z;
        }
        if (mesh_changed) {
            D3D11_MAPPED_SUBRESOURCE m;
            if (SUCCEEDED(ctx->Map(mesh_vb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
                mesh_count = (UINT)std::min(room::MAX_MESH_VERTS, new_mesh.size() / 6);
                memcpy(m.pData, new_mesh.data(), (size_t)mesh_count * sizeof(Vertex));
                ctx->Unmap(mesh_vb, 0);
            }
        }
        // Scene <-> tracking frame: recenter is a yaw and a translation, so heights shift by a constant.
        const M4 recenter_inv = rigid_inverse(recenter);
        // The scene's MCP: its requests, then what it reads (tracking frame), a few times a second.
        // The cubes: dropped once the floor is found (measured or calibrated), 0.8 m ahead.
        {
            double now_s = qpc_now() / qpc_freq();
            float dt = (float)std::clamp(now_s - pinch_prev_t, 0.0, 0.05);
            bool floor_ok = !std::isnan(walk_floor) && walk_src.find("estim") == std::string::npos;
            if (have && floor_ok && (!cubes_dropped || g_app.cubes_request)) {
                float fx = -head(0, 2), fz = -head(2, 2), n = sqrtf(fx * fx + fz * fz);
                if (n > 1e-3f) {
                    fx /= n; fz /= n;
                    cubes.clear();
                    const float COL[3][3] = {{0.95f, 0.3f, 0.25f}, {0.3f, 0.85f, 0.35f}, {0.3f, 0.5f, 1.0f}};
                    for (int k = 0; k < 3; k++) {
                        float side = (k - 1) * 0.25f;
                        Cube cb2;
                        cb2.p = {p.x + 0.8f * fx - side * fz, p.y - 0.1f, p.z + 0.8f * fz + side * fx};
                        cb2.v = {0, 0, 0};
                        memcpy(cb2.col, COL[k], sizeof cb2.col);
                        cubes.push_back(cb2);
                    }
                    cubes_dropped = true;
                }
                g_app.cubes_request = 0;
            }
            // Pinch points (thumb and index tips' middle), tracking frame, and their speed.
            V3 pinch[2] = {{1e9f, 1e9f, 1e9f}, {1e9f, 1e9f, 1e9f}}; bool pinching[2] = {false, false};
            for (auto& h : scene_hands) {
                int k = h.right ? 1 : 0;
                V3 a = h.j[4], b = h.j[8], m2 = {(a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2};
                pinch[k] = {recenter_inv(0, 0) * m2.x + recenter_inv(0, 1) * m2.y + recenter_inv(0, 2) * m2.z + recenter_inv(0, 3),
                            recenter_inv(1, 0) * m2.x + recenter_inv(1, 1) * m2.y + recenter_inv(1, 2) * m2.z + recenter_inv(1, 3),
                            recenter_inv(2, 0) * m2.x + recenter_inv(2, 1) * m2.y + recenter_inv(2, 2) * m2.z + recenter_inv(2, 3)};
                pinching[k] = h.pinching;
            }
            std::vector<V3> tips;  // fingertips, tracking frame
            for (auto& h : scene_hands) for (int t : {4, 8, 12, 16, 20}) {
                V3 q = h.j[t];
                tips.push_back({recenter_inv(0, 0) * q.x + recenter_inv(0, 1) * q.y + recenter_inv(0, 2) * q.z + recenter_inv(0, 3),
                                recenter_inv(1, 0) * q.x + recenter_inv(1, 1) * q.y + recenter_inv(1, 2) * q.z + recenter_inv(1, 3),
                                recenter_inv(2, 0) * q.x + recenter_inv(2, 1) * q.y + recenter_inv(2, 2) * q.z + recenter_inv(2, 3)});
            }
            for (auto& cb2 : cubes) {
                // In contact: a fingertip within 1.5 cm of the cube, or a pinch able to take it.
                cb2.hover = false;
                for (auto& t : tips) cb2.hover |= std::fabs(t.x - cb2.p.x) < CUBE / 2 + 0.015f && std::fabs(t.y - cb2.p.y) < CUBE / 2 + 0.015f && std::fabs(t.z - cb2.p.z) < CUBE / 2 + 0.015f;
                for (int k = 0; k < 2; k++) {
                    float dx = pinch[k].x - cb2.p.x, dy = pinch[k].y - cb2.p.y, dz = pinch[k].z - cb2.p.z;
                    cb2.hover |= !scene_hands.empty() && dx * dx + dy * dy + dz * dz < 0.09f * 0.09f;
                }
                if (cb2.held >= 0) {
                    int k = cb2.held;
                    if (pinching[k]) {
                        V3 np = {pinch[k].x + cb2.off.x, pinch[k].y + cb2.off.y, pinch[k].z + cb2.off.z};
                        if (dt > 1e-4f) cb2.v = {(np.x - cb2.p.x) / dt * 0.5f + cb2.v.x * 0.5f, (np.y - cb2.p.y) / dt * 0.5f + cb2.v.y * 0.5f, (np.z - cb2.p.z) / dt * 0.5f + cb2.v.z * 0.5f};
                        cb2.p = np;
                        continue;
                    }
                    cb2.held = -1;  // let go: thrown with the hand's speed (3 m/s at most)
                    float sp = sqrtf(cb2.v.x * cb2.v.x + cb2.v.y * cb2.v.y + cb2.v.z * cb2.v.z);
                    if (sp > 3) { cb2.v.x *= 3 / sp; cb2.v.y *= 3 / sp; cb2.v.z *= 3 / sp; }
                }
                // A pinch starting at the cube (within 9 cm) takes it.
                for (int k = 0; k < 2; k++) {
                    if (!pinching[k] || cb2.held >= 0) continue;
                    float dx = pinch[k].x - cb2.p.x, dy = pinch[k].y - cb2.p.y, dz = pinch[k].z - cb2.p.z;
                    if (dx * dx + dy * dy + dz * dz < 0.09f * 0.09f) {
                        bool taken = false;
                        for (auto& o : cubes) taken |= o.held == k;
                        if (!taken) { cb2.held = k; cb2.off = {cb2.p.x - pinch[k].x, cb2.p.y - pinch[k].y, cb2.p.z - pinch[k].z}; }
                    }
                }
                if (cb2.held >= 0) continue;
                // Falling: gravity, then onto the scanned room's voxels (or the floor).
                auto occupied = [&](float x, float y, float z) {
                    return solid && solid->count(room::key((int)std::floor(x / room::VOX), (int)std::floor(y / room::VOX), (int)std::floor(z / room::VOX)));
                };
                cb2.v.y -= 9.81f * dt;
                V3 was = cb2.p;
                cb2.p = {cb2.p.x + cb2.v.x * dt, cb2.p.y + cb2.v.y * dt, cb2.p.z + cb2.v.z * dt};
                // Into something on the side (its middle in a voxel above the ground): back, bounced.
                if ((cb2.v.x != 0 || cb2.v.z != 0) && occupied(cb2.p.x, cb2.p.y, cb2.p.z) && !occupied(was.x, was.y, was.z)) {
                    cb2.p.x = was.x; cb2.p.z = was.z; cb2.v.x *= -0.3f; cb2.v.z *= -0.3f;
                }
                // The ground: the top of the highest voxel under the cube's footprint and below
                // its bottom (2 cm tolerance), searched down 3 m; the floor at most.
                float ground = std::isnan(walk_floor) ? -1e9f : walk_floor;
                if (solid) {
                    const float H = CUBE / 2 - 0.01f;
                    int top = (int)std::floor((cb2.p.y - CUBE / 2 + 0.02f) / room::VOX);
                    for (float ox : {-H, 0.0f, H}) for (float oz : {-H, 0.0f, H}) {
                        int cx = (int)std::floor((cb2.p.x + ox) / room::VOX), cz = (int)std::floor((cb2.p.z + oz) / room::VOX);
                        for (int cy = top; cy > top - 120; cy--) {
                            if ((cy + 1) * room::VOX <= ground) break;
                            if (solid->count(room::key(cx, cy, cz))) { ground = std::max(ground, (cy + 1) * room::VOX); break; }
                        }
                    }
                }
                cb2.ground = ground > -1e8f ? ground : NAN;
                if (cb2.p.y - CUBE / 2 < ground) {
                    cb2.p.y = ground + CUBE / 2;
                    cb2.v.y = cb2.v.y < -0.5f ? -cb2.v.y * 0.3f : 0.0f;
                    cb2.v.x *= 0.85f; cb2.v.z *= 0.85f;
                }
            }
            for (int k = 0; k < 2; k++) if (pinching[k]) pinch_prev[k] = pinch[k];
            pinch_prev_t = now_s;
        }
        std::vector<mcp::Marker> mcp_markers;
        {
            std::lock_guard<std::mutex> l(mcp::g_mu);
            for (auto& cm : mcp::g_cmds) {
                if (cm.op == "human_go") { g_app.human_target[0] = cm.v[0]; g_app.human_target[1] = cm.v[2]; g_app.human_request = 2; }
                else if (cm.op == "human_come") g_app.human_request = 1;
                else if (cm.op == "human_show") g_app.show_request = (int)cm.v[0];
                else if (cm.op == "anchor") { g_app.anchor_label = cm.text; g_app.room_request = 1; }
                else if (cm.op == "human_visible") g_app.human = cm.flag;
                else if (cm.op == "human_wander") g_app.human_wander = cm.flag;
            }
            mcp::g_cmds.clear();
            mcp_markers = mcp::g_markers;
            static int pub = 0;
            if (pub++ % 6 == 0) {
                mcp::Snapshot& sn = mcp::g_snap;
                sn.tracking = have && !g_app.tracking_lost;
                memcpy(sn.head, head.m, sizeof sn.head);
                sn.has_floor = !std::isnan(walk_floor); sn.floor_y = sn.has_floor ? walk_floor : 0;
                sn.planes = planes; sn.walk = walk; sn.anchors = anchors;
                sn.human_visible = g_app.human && walker.placed; sn.human_placed = walker.placed; sn.human_wander = g_app.human_wander;
                sn.human_pos[0] = walker.x; sn.human_pos[1] = sn.floor_y; sn.human_pos[2] = walker.z; sn.human_heading = walker.heading;
                sn.human_state = g_app.human_state;
                sn.hand[0] = sn.hand[1] = false;
                for (auto& h : scene_hands) {
                    int k = h.right ? 1 : 0;
                    V3 t = h.j[8];
                    sn.hand[k] = true;
                    for (int i = 0; i < 3; i++) sn.index_tip[k][i] = recenter_inv(i, 0) * t.x + recenter_inv(i, 1) * t.y + recenter_inv(i, 2) * t.z + recenter_inv(i, 3);
                }
            }
        }
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


        if (stab_panel_ok && g_app.stab_test) {
            static const char* MODES[4] = {"lissage + timewarp", "sans lissage", "sans timewarp", "ni lissage ni timewarp"};
            char a[96], b[96];
            snprintf(a, sizeof a, "Latence de l'écran : %.0f ms", g_app.predict_ms);
            snprintf(b, sizeof b, "Réglage : %s", MODES[g_app.stab_mode & 3]);
            char d[120];
            snprintf(d, sizeof d, "Profondeur %+.0f mm · hauteur %+.1f°", g_app.ipd_adj_mm, g_app.pitch_deg);
            stab_panel.draw(ctx, "Test de stabilité", {"Secoue la tête : la croix doit rester collée au bord réel.", a, b, d}, "réglages : onglet Vue RGB", false);
        }

        float clear[4] = {0, 0, 0, 1};
        bool warp = g_app.timewarp && have && tw_target > 0;
        if (warp && (eye_w != (UINT)W || eye_h != (UINT)H)) {
            if (eye_tex.p) { eye_tex.p->Release(); eye_tex.p = nullptr; }
            if (eye_rtv.p) { eye_rtv.p->Release(); eye_rtv.p = nullptr; }
            if (eye_srv.p) { eye_srv.p->Release(); eye_srv.p = nullptr; }
            D3D11_TEXTURE2D_DESC td{}; td.Width = W; td.Height = H; td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            td.SampleDesc.Count = 1; td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            if (SUCCEEDED(dev->CreateTexture2D(&td, nullptr, &eye_tex)) && SUCCEEDED(dev->CreateRenderTargetView(eye_tex, nullptr, &eye_rtv)) &&
                SUCCEEDED(dev->CreateShaderResourceView(eye_tex, nullptr, &eye_srv))) { eye_w = W; eye_h = H; }
        }
        warp = warp && eye_w == (UINT)W;
        ID3D11RenderTargetView* target_rtv = warp ? eye_rtv.p : rtv;
        ctx->ClearRenderTargetView(target_rtv, clear);
        ctx->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 1, 0);
        ID3D11RenderTargetView* rtvs[] = {target_rtv};
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
        bool v3_pass = false;   // render_view for the 3D debug view: no occluders, no room lines
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
            if (g_app.occlusion && !v3_pass) {
                ctx->OMSetBlendState(no_color, nullptr, 0xffffffff);
                if (occ_count) draw(occ_vb, occ_count, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, recenter, 0, 0, 0, false);
                if (mesh_count && !g_app.show_mesh) draw(mesh_vb, mesh_count, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, recenter, 0, 0, 0, false);
                ctx->OMSetBlendState(nullptr, nullptr, 0xffffffff);
            }
            if (mesh_count && g_app.show_mesh) draw(mesh_vb, mesh_count, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, recenter, 0.22f, 0.30f, 0.42f, true);
            if (line_count && g_app.show_map && !v3_pass) draw(line_vb, line_count, D3D11_PRIMITIVE_TOPOLOGY_LINELIST, recenter, 0.25f, 0.85f, 1.0f, false);
            // The human, behind the occluders.
            if (g_app.human && walker.placed && human_r.ok && !std::isnan(walk_floor)) {
                float hk = g_app.human_height / 1.72f;
                M4 place = mul(recenter, mul(translate({walker.x, walk_floor, walker.z}), mul(rot_y(walker.heading), scale(hk))));
                float tint[3] = {room_tint[0], room_tint[1], room_tint[2]};
                float walk_amt = std::min(1.0f, walker.speed / human::Walker::WALK_SPEED * 1.3f);
                float aim[3], aim_w = 0;
                if (show.on && show.arrived > 0) {
                    float dx = show.a[0] - walker.x, dy = show.a[1] - walk_floor, dz = show.a[2] - walker.z, ch = cosf(walker.heading), sh = sinf(walker.heading);
                    aim[0] = (ch * dx - sh * dz) / hk; aim[1] = dy / hk; aim[2] = (sh * dx + ch * dz) / hk;
                    aim_w = (float)std::clamp((qpc_now() / qpc_freq() - show.arrived) / 0.5, 0.0, 1.0);
                }
                human_r.draw(ctx, vpm.m, place.m, tint, qpc_now() / qpc_freq() - human_t0, walker.phase, walk_amt, aim, aim_w);
                ctx->IASetInputLayout(layout);
                ctx->VSSetShader(vs, nullptr, 0);
                ctx->PSSetShader(ps, nullptr, 0);
                ID3D11Buffer* cbs2[] = {cb};
                ctx->VSSetConstantBuffers(0, 1, cbs2);
                ctx->PSSetConstantBuffers(0, 1, cbs2);
                V3 hs = {place(0, 3), place(1, 3), place(2, 3)};
                shadow(hs.x, hs.y, hs.z, 0.35f, 0.5f);
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
            for (auto& a : anchors) {
                bool shown = show.on && show.id == a.id;
                draw(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(mul(recenter, translate({a.p[0], a.p[1], a.p[2]})), scale(shown ? 0.05f : 0.03f)), shown ? 1.0f : 0.3f, 1.0f, shown ? 0.2f : 1.0f, !shown);
            }
            if (g_app.aiming && have && !v3_pass) {
                V3 at = {p.x - head(0, 2) * 1.5f, p.y - head(1, 2) * 1.5f, p.z - head(2, 2) * 1.5f};
                draw(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(mul(recenter, translate(at)), scale(0.012f)), 0.2f, 1.0f, 0.3f, false);
            }
            // The scanned room, one small cube per voxel.
            if (g_app.calage && inst_count) {
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
            // The MCP's markers: a 5 cm cube and a thin line down to the floor.
            for (auto& mk : mcp_markers) {
                M4 at = mul(recenter, translate({mk.p[0], mk.p[1], mk.p[2]}));
                draw(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(at, scale(0.05f)), mk.color[0], mk.color[1], mk.color[2], false);
                float fy = std::isnan(walk_floor) ? mk.p[1] - 1 : walk_floor, hgt = mk.p[1] - fy;
                if (hgt > 0.03f) draw(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(mul(recenter, translate({mk.p[0], fy + hgt / 2, mk.p[2]})), scale3(0.004f, hgt, 0.004f)), mk.color[0], mk.color[1], mk.color[2], false);
            }
            if (g_app.stab_test) {
                // The cross: 3 mm wide, a 2.4 m vertical (head height -1.5 to +0.9 m) and a 1.2 m
                // horizontal at eye height; unlit green, bright on the glasses.
                M4 base = mul(recenter, mul(translate(stab_pos), rot_y(stab_yaw)));
                draw(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(base, mul(translate({0, -0.3f, 0}), scale3(0.003f, 2.4f, 0.003f))), 0.2f, 1.0f, 0.3f, false);
                draw(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(base, scale3(1.2f, 0.003f, 0.003f)), 0.2f, 1.0f, 0.3f, false);
                if (stab_panel_ok) {
                    M4 model = mul(base, mul(translate({0.75f, -0.45f, 0}), scale(2.2f)));
                    Constants c{mul(vpm, model), model, {1, 1, 1, 1}, {0, 0, 0, 0}};
                    ctx->UpdateSubresource(cb, 0, nullptr, &c, 0, 0);
                    ID3D11Buffer* vbs[] = {quad_vb};
                    ctx->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
                    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    ctx->VSSetShader(vs_tex, nullptr, 0);
                    ctx->PSSetShader(ps_tex, nullptr, 0);
                    ID3D11ShaderResourceView* srvs[] = {stab_panel.srv};
                    ctx->PSSetShaderResources(0, 1, srvs);
                    ID3D11SamplerState* ss[] = {samp};
                    ctx->PSSetSamplers(0, 1, ss);
                    ctx->Draw(6, 0);
                    ID3D11ShaderResourceView* none[] = {nullptr};
                    ctx->PSSetShaderResources(0, 1, none);
                    ctx->VSSetShader(vs, nullptr, 0);
                    ctx->PSSetShader(ps, nullptr, 0);
                }
            }
            for (auto& h : scene_hands) {
                float hr = h.right ? 1.0f : 0.3f, hg = h.right ? 0.48f : 0.55f, hb = h.right ? 0.27f : 1.0f;
                if (h.pinching) { hr = hr * 0.5f + 0.5f; hg = hg * 0.5f + 0.5f; hb = hb * 0.5f + 0.5f; }
                for (auto& bn : hands::BONES) {
                    V3 a = h.j[bn[0]], b = h.j[bn[1]], d = {b.x - a.x, b.y - a.y, b.z - a.z};
                    float len = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
                    if (len < 1e-4f) continue;
                    // A 4 mm box from a to b: its y axis along the bone.
                    V3 y = {d.x / len, d.y / len, d.z / len}, ref = fabsf(y.y) < 0.9f ? V3{0, 1, 0} : V3{1, 0, 0};
                    V3 x = {ref.y * y.z - ref.z * y.y, ref.z * y.x - ref.x * y.z, ref.x * y.y - ref.y * y.x};
                    float xn = sqrtf(x.x * x.x + x.y * x.y + x.z * x.z); x = {x.x / xn, x.y / xn, x.z / xn};
                    V3 z = {x.y * y.z - x.z * y.y, x.z * y.x - x.x * y.z, x.x * y.y - x.y * y.x};
                    const float T = 0.004f;
                    M4 mb = identity();
                    mb(0, 0) = x.x * T; mb(1, 0) = x.y * T; mb(2, 0) = x.z * T;
                    mb(0, 1) = d.x; mb(1, 1) = d.y; mb(2, 1) = d.z;
                    mb(0, 2) = z.x * T; mb(1, 2) = z.y * T; mb(2, 2) = z.z * T;
                    mb(0, 3) = (a.x + b.x) / 2; mb(1, 3) = (a.y + b.y) / 2; mb(2, 3) = (a.z + b.z) / 2;
                    draw(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mb, hr, hg, hb, false);
                }
                for (int k = 0; k < hands::N_JOINTS; k++) draw(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(translate(h.j[k]), scale(0.009f)), hr, hg, hb, false);
            }
            if (!g_app.scene || g_app.calage) return;
            // The three cubes (tracking frame -> scene), their contact shadow on what they rest on.
            for (auto& cb2 : cubes) {
                M4 at = mul(recenter, translate(cb2.p));
                // Touched: 12 % bigger and lit towards white; held: 20 % and brighter.
                float glow = cb2.held >= 0 ? 1.0f : cb2.hover ? 0.6f : 0.0f;
                float cr = cb2.col[0] + (1 - cb2.col[0]) * 0.5f * glow, cg = cb2.col[1] + (1 - cb2.col[1]) * 0.5f * glow, cbl = cb2.col[2] + (1 - cb2.col[2]) * 0.5f * glow;
                draw(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(at, scale(CUBE * (1 + 0.2f * glow))), cr, cg, cbl, glow == 0);
                if (!std::isnan(cb2.ground)) shadow(at(0, 3), cb2.ground + y_shift, at(2, 3), CUBE, 0.5f);
            }
            float floor_y = -g_app.floor_m;
            for (auto& p : planes) if (p.kind == 0) floor_y = p.y + y_shift;
            if (g_app.floor_grid) draw(grid_vb, (UINT)grid.size(), D3D11_PRIMITIVE_TOPOLOGY_LINELIST, translate({0, floor_y, 0}), 0.3f, 0.7f, 1.0f, false);
        };
        M4 eye_in_head[2] = {identity(), identity()};  // each eye's rotation in the head frame (for the warp)
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
                // The separation adjusted (the wearer's eyes, not the factory's): each eye half of it outwards.
                float adj = sbs ? (eye ? 0.5f : -0.5f) * g_app.ipd_adj_mm / 1000 : 0.0f;
                M4 eye_world = mul(mul(head_scene, mul(fine, translate({adj, 0, 0}))), D);
                eye_in_head[eye] = mul(rigid_inverse(head_scene), eye_world);
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
            eye_in_head[eye] = mul(rigid_inverse(head_scene), eye_world);
            render_view(mul(perspective(g_app.vfov, vw / H, 0.05f, 50), rigid_inverse(eye_world)));
        }

        // Timewarp: just before the vblank, the newest prediction for the same moment; the
        // eye images turned by the head's rotation between the two (the translation is
        // small over that time and left out).
        if (warp) {
            static double tw_margin_ms = 4.0;
            static long tw_missed_seen = 0;
            static double tw_quiet = 0;
            // Missed vblanks with the warp on: start the warp earlier (up to 9 ms).
            double now_s = qpc_now() / qpc_freq();
            if (missed > tw_missed_seen) { tw_margin_ms = std::min(9.0, tw_margin_ms + 0.5); tw_quiet = now_s; }
            else if (now_s - tw_quiet > 10 && tw_margin_ms > 3.0) { tw_margin_ms -= 0.25; tw_quiet = now_s; }
            tw_missed_seen = missed;
            g_app.tw_margin_ms = (float)tw_margin_ms;
            // Until the vblank this frame aims at, minus the margin.
            double wait_ms = qpc_ms(t_start - qpc_now()) + to_vblank_ms - tw_margin_ms;
            static int tw_dbg = 0;
            if (tw_dbg++ % 240 == 0) printf("      timewarp: to vblank %.1f ms at start, %.1f ms elapsed, wait %.1f ms, margin %.1f ms\n", to_vblank_ms, qpc_ms(qpc_now() - t_start), wait_ms, tw_margin_ms);
            if (wait_ms > 1.5) Sleep((DWORD)(wait_ms - 1.0));
            int64_t w0 = t_start + (int64_t)((to_vblank_ms - tw_margin_ms) * qpc_freq() / 1000.0);
            while (qpc_now() < w0) {}
            M4 late;
            Q dq{0, 0, 0, 1};
            if (engine_pose(tw_target, late)) {
                tw_late.push_back({tw_target, mat_q(late), rate_dps});
                // What was shown for that moment: the next frame's smoother measures the
                // engine's revisions from here, so it does not take back what the warp showed.
                sm_prev_target = tw_target; sm_prev_q = mat_q(late); sm_prev_p = {late(0, 3), late(1, 3), late(2, 3)};
                dq = qmul(qconj(tw_q), mat_q(late));  // rendered head -> late head, in the head frame
                if (dq.w < 0) dq = {-dq.x, -dq.y, -dq.z, -dq.w};
                float ang = 2 * atan2f(sqrtf(dq.x * dq.x + dq.y * dq.y + dq.z * dq.z), dq.w) * 57.2958f;
                g_app.tw_deg = 0.9f * g_app.tw_deg + 0.1f * ang;
            }
            M4 D = qmat(dq);
            ID3D11RenderTargetView* bb[] = {rtv};
            ctx->OMSetRenderTargets(1, bb, nullptr);
            ctx->ClearRenderTargetView(rtv, clear);
            ID3D11Buffer* vbs[] = {screen_vb};
            ctx->IASetInputLayout(layout);
            ctx->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
            ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx->VSSetShader(vs_screen, nullptr, 0);
            ctx->PSSetShader(ps_warp, nullptr, 0);
            ID3D11ShaderResourceView* srvs[] = {eye_srv};
            ctx->PSSetShaderResources(0, 1, srvs);
            ID3D11SamplerState* ss[] = {samp};
            ctx->PSSetSamplers(0, 1, ss);
            float vw = (float)W / views;
            float f = 1.0f / tanf(g_app.vfov * 3.14159265f / 360);
            for (int eye = 0; eye < views; eye++) {
                // Late eye -> rendered eye directions: E^T (head delta) E, E the eye in the head.
                M4 E = eye_in_head[eye]; E(0, 3) = E(1, 3) = E(2, 3) = 0;
                M4 Et = E; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Et(i, j) = E(j, i);
                M4 M = mul(Et, mul(D, E));
                Constants cw{M, identity(), {f, vw / H, (float)eye / views, 1.0f / views}, {0, 0, 0, 0}};
                ctx->UpdateSubresource(cb, 0, nullptr, &cw, 0, 0);
                D3D11_VIEWPORT vpw{eye * vw, 0, vw, (float)H, 0, 1};
                ctx->RSSetViewports(1, &vpw);
                ctx->Draw(6, 0);
            }
            ID3D11ShaderResourceView* none[] = {nullptr};
            ctx->PSSetShaderResources(0, 1, none);
            ctx->VSSetShader(vs, nullptr, 0);
            ctx->PSSetShader(ps, nullptr, 0);
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
            if (last_present_seen && extra > 0 && extra < 600) { missed += (long)extra; if (rate_dps > 60) missed_fast += (long)extra; }
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
            static float saved[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
            float cur[9] = {g_app.vfov, g_app.ipd_mm, g_app.floor_m, g_app.pitch_deg, g_app.eye_dy_cm, g_app.eye_dz_cm, g_app.yaw_deg, g_app.predict_ms, g_app.ipd_adj_mm};
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
            {
                std::vector<double> gaps;
                { std::lock_guard<std::mutex> l(g_imu.mu); gaps.assign(g_imu_gaps.begin(), g_imu_gaps.end()); g_imu_gaps.clear(); }
                g_app.stats.fallback_hz = (float)(fallbacks * 1000.0 / since);
                g_app.stats.imu_gap_p99 = (float)pct(gaps, 0.99);
                g_app.stats.missed_fast = missed_fast;
                if (!g_stats_paused) printf("      stability: engine prediction failed %ld frames (%.1f/s)  missed while fast %ld  IMU gaps p50 %.2f p99 %.2f max %.2f ms  screen latency %.0f ms  setting %d  tracking losses %ld  head y %.2f\n",
                                            fallbacks, fallbacks * 1000.0 / since, missed_fast, pct(gaps, 0.5), pct(gaps, 0.99), pct(gaps, 1.0), g_app.predict_ms, g_app.stab_mode, g_app.tracking_losses, g_app.pos[1]);
                fallbacks = 0; missed_fast = 0;
            }
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
            if (!g_stats_paused && (tw_err_slow.size() + tw_err_fast.size()))
                printf("      after timewarp (deg, p50/p95): slow %.3f/%.3f (n %zu)  fast %.3f/%.3f (n %zu)\n", pct(tw_err_slow, 0.5), pct(tw_err_slow, 0.95), tw_err_slow.size(), pct(tw_err_fast, 0.5), pct(tw_err_fast, 0.95), tw_err_fast.size());
            if (tw_err_slow.size() + tw_err_fast.size()) { g_app.stats.tw_err_moving = (float)pct(tw_err_slow, 0.95); g_app.stats.tw_err_fast = (float)pct(tw_err_fast, 0.95); }
            tw_err_slow.clear(); tw_err_fast.clear();
            pred_err_still.clear(); pred_err_slow.clear(); pred_err_fast.clear();
            gain.clear();
            for (long& c : status_counts) c = 0;
            latencies.clear(); cpu_ms.clear(); jerk.clear(); shown = 0; missed = 0; max_rate = 0; last_updates = g_pose_updates;
            stats_t0 = qpc_now();
        }
        // The 3D debug view (dashboard tab "Scène 3D"), tracking frame: an orbit camera around
        // the head (or a free target); the room's layers, the head, the human.
        {
            double now_s = qpc_now() / qpc_freq();
            if (have && now_s - trail_t > 0.1) { trail_t = now_s; head_trail.push_back(p); if (head_trail.size() > 1200) head_trail.erase(head_trail.begin(), head_trail.begin() + 200); }
        }
        static int v3_frame = 0;
        if (g_app.v3_open && v3_frame++ % 3 == 0) {  // 20 images a second: the GPU is the glasses' first
            g_app.v3_open = false;  // the dashboard sets it again while the tab shows
            {
                std::lock_guard<std::mutex> l(room::g.mu);
                auto upload = [&](const std::vector<float>& src, ID3D11Buffer* vb, UINT& count) {
                    D3D11_MAPPED_SUBRESOURCE m;
                    count = (UINT)std::min<size_t>(V3_PTS, src.size() / 6);
                    if (count && SUCCEEDED(ctx->Map(vb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) { memcpy(m.pData, src.data(), count * 24); ctx->Unmap(vb, 0); }
                };
                if (room::g.dbg_version != v3_ev_version) { v3_ev_version = room::g.dbg_version; upload(room::g.dbg_evidence, v3_ev_vb, v3_ev_count); }
                if (room::g.live_version != v3_live_version) { v3_live_version = room::g.live_version; upload(room::g.dbg_live, v3_live_vb, v3_live_count); }
                if (room::g.dense_version != v3_dense_version) { v3_dense_version = room::g.dense_version; upload(room::g.dbg_dense, v3_dense_vb, v3_dense_count); }
            }
            if (g_app.v3_follow) { g_app.v3_target[0] = p.x; g_app.v3_target[1] = p.y; g_app.v3_target[2] = p.z; }
            V3 tg = {g_app.v3_target[0], g_app.v3_target[1], g_app.v3_target[2]};
            float cp = cosf(g_app.v3_pitch), sp = sinf(g_app.v3_pitch);
            V3 eye = {tg.x + g_app.v3_dist * cp * sinf(g_app.v3_yaw), tg.y + g_app.v3_dist * sp, tg.z + g_app.v3_dist * cp * cosf(g_app.v3_yaw)};
            // Camera frame: -z towards the target, y up as much as it can.
            V3 zc = {eye.x - tg.x, eye.y - tg.y, eye.z - tg.z};
            float zl = sqrtf(zc.x * zc.x + zc.y * zc.y + zc.z * zc.z); zc = {zc.x / zl, zc.y / zl, zc.z / zl};
            V3 xc = {zc.z, 0, -zc.x}; float xl = sqrtf(xc.x * xc.x + xc.z * xc.z);
            if (xl < 1e-4f) xc = {1, 0, 0}; else xc = {xc.x / xl, 0, xc.z / xl};
            V3 yc = {zc.y * xc.z - zc.z * xc.y, zc.z * xc.x - zc.x * xc.z, zc.x * xc.y - zc.y * xc.x};
            M4 camw = identity();
            camw(0, 0) = xc.x; camw(1, 0) = xc.y; camw(2, 0) = xc.z;
            camw(0, 1) = yc.x; camw(1, 1) = yc.y; camw(2, 1) = yc.z;
            camw(0, 2) = zc.x; camw(1, 2) = zc.y; camw(2, 2) = zc.z;
            camw(0, 3) = eye.x; camw(1, 3) = eye.y; camw(2, 3) = eye.z;
            M4 vp3 = mul(perspective(55, (float)g_app.v3_w / g_app.v3_h, 0.05f, 100), rigid_inverse(camw));
            ID3D11RenderTargetView* t3[] = {v3_rtv};
            ctx->OMSetRenderTargets(1, t3, v3_dsv);
            float bg[4] = {0.07f, 0.075f, 0.09f, 1};
            ctx->ClearRenderTargetView(v3_rtv, bg);
            ctx->ClearDepthStencilView(v3_dsv, D3D11_CLEAR_DEPTH, 1, 0);
            D3D11_VIEWPORT vp{0, 0, (float)g_app.v3_w, (float)g_app.v3_h, 0, 1};
            ctx->RSSetViewports(1, &vp);
            ctx->OMSetDepthStencilState(dss, 0);
            ctx->OMSetBlendState(nullptr, nullptr, 0xffffffff);
            ctx->RSSetState(rs);
            ctx->IASetInputLayout(layout);
            ctx->VSSetShader(vs, nullptr, 0);
            ctx->PSSetShader(ps, nullptr, 0);
            ID3D11Buffer* cbs3[] = {cb};
            ctx->VSSetConstantBuffers(0, 1, cbs3);
            ctx->PSSetConstantBuffers(0, 1, cbs3);
            auto d3 = [&](ID3D11Buffer* vb, UINT n, D3D11_PRIMITIVE_TOPOLOGY topo, const M4& model, float r, float g, float b, bool lit, float a = 1) {
                Constants cc{mul(vp3, model), model, {r, g, b, a}, {lit ? 1.0f : 0.0f, 0, 0, 0}};
                ctx->UpdateSubresource(cb, 0, nullptr, &cc, 0, 0);
                ID3D11Buffer* vbs[] = {vb};
                ctx->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
                ctx->IASetPrimitiveTopology(topo);
                ctx->Draw(n, 0);
            };
            auto pts3 = [&](ID3D11Buffer* vb, UINT n, float size) {
                if (!n) return;
                Constants cc{vp3, identity(), {1, 1, 1, 1}, {1, size, 0, 0}};
                ctx->UpdateSubresource(cb, 0, nullptr, &cc, 0, 0);
                ID3D11Buffer* vbs[] = {cube_vb, vb};
                UINT strides[] = {sizeof(Vertex), 24}, offsets[] = {0, 0};
                ctx->IASetInputLayout(layout_inst);
                ctx->VSSetShader(vs_inst, nullptr, 0);
                ctx->PSSetShader(ps_inst, nullptr, 0);
                ctx->IASetVertexBuffers(0, 2, vbs, strides, offsets);
                ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                ctx->DrawInstanced((UINT)cube.size(), n, 0, 0);
                ctx->IASetInputLayout(layout);
                ctx->VSSetShader(vs, nullptr, 0);
                ctx->PSSetShader(ps, nullptr, 0);
            };
            float fy3 = std::isnan(walk_floor) ? p.y - g_app.floor_m : walk_floor;
            if (g_app.v3_voxels) pts3(inst_vb, inst_count, room::VOX * 0.8f);
            if (g_app.v3_evidence) pts3(v3_ev_vb, v3_ev_count, room::EV * 0.6f);
            if (g_app.v3_live) pts3(v3_live_vb, v3_live_count, 0.012f);
            if (g_app.v3_dense) pts3(v3_dense_vb, v3_dense_count, 0.018f);
            if (g_app.v3_mesh && mesh_count) d3(mesh_vb, mesh_count, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, identity(), 0.5f, 0.55f, 0.65f, true);
            if (g_app.v3_walk && line_count) d3(line_vb, line_count, D3D11_PRIMITIVE_TOPOLOGY_LINELIST, identity(), 0.25f, 0.95f, 1.0f, false);
            // Head (a small box, its view frustum to 0.6 m), the trail, the human's path, the gaze.
            std::vector<Vertex> L;
            auto seg = [&](V3 a, V3 b) { L.push_back({a.x, a.y, a.z, 0, 1, 0}); L.push_back({b.x, b.y, b.z, 0, 1, 0}); };
            auto hp = [&](float x, float y, float z) { return V3{head(0, 0) * x + head(0, 1) * y + head(0, 2) * z + head(0, 3), head(1, 0) * x + head(1, 1) * y + head(1, 2) * z + head(1, 3), head(2, 0) * x + head(2, 1) * y + head(2, 2) * z + head(2, 3)}; };
            if (have) {
                V3 o = hp(0, 0, 0), c4[4] = {hp(-0.3f, 0.19f, -0.6f), hp(0.3f, 0.19f, -0.6f), hp(0.3f, -0.19f, -0.6f), hp(-0.3f, -0.19f, -0.6f)};
                for (int k = 0; k < 4; k++) { seg(o, c4[k]); seg(c4[k], c4[(k + 1) % 4]); }
                seg(o, hp(0, 0, -3));      // the gaze, 3 m
                seg(o, {o.x, fy3, o.z});   // down to the floor: the eye height
            }
            if (g_app.v3_trail) for (size_t k = 1; k < head_trail.size(); k++) seg({head_trail[k - 1].x, fy3 + 0.01f, head_trail[k - 1].z}, {head_trail[k].x, fy3 + 0.01f, head_trail[k].z});
            UINT nl_head = (UINT)L.size();
            for (size_t k = walker.next; k < walker.path.size(); k++) {
                V3 a = k == walker.next ? V3{walker.x, fy3 + 0.02f, walker.z} : V3{walker.path[k - 1][0], fy3 + 0.02f, walker.path[k - 1][1]};
                seg(a, {walker.path[k][0], fy3 + 0.02f, walker.path[k][1]});
            }
            UINT n_lines = (UINT)std::min<size_t>(L.size(), V3_LINES) & ~1u;
            D3D11_MAPPED_SUBRESOURCE lm;
            if (n_lines && SUCCEEDED(ctx->Map(v3_line_vb, 0, D3D11_MAP_WRITE_DISCARD, 0, &lm))) {
                memcpy(lm.pData, L.data(), n_lines * sizeof(Vertex)); ctx->Unmap(v3_line_vb, 0);
                d3(v3_line_vb, std::min(nl_head, n_lines), D3D11_PRIMITIVE_TOPOLOGY_LINELIST, identity(), 1.0f, 0.9f, 0.2f, false);
                if (n_lines > nl_head) {
                    Constants cc{vp3, identity(), {1.0f, 0.3f, 0.8f, 1}, {0, 0, 0, 0}};
                    ctx->UpdateSubresource(cb, 0, nullptr, &cc, 0, 0);
                    ID3D11Buffer* vbs[] = {v3_line_vb};
                    ctx->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
                    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
                    ctx->Draw(n_lines - nl_head, nl_head);
                }
            }
            if (have) d3(cube_vb, (UINT)cube.size(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, mul(head, scale3(0.16f, 0.06f, 0.05f)), 1.0f, 0.9f, 0.2f, true);
            // The virtual content as the glasses show it (cubes, panel, cross, markers, anchors,
            // the human): the same drawing, from the scene frame into the tracking frame.
            v3_pass = true;
            render_view(mul(vp3, recenter_inv));
            v3_pass = false;
            ctx->IASetInputLayout(layout);
            ctx->VSSetShader(vs, nullptr, 0);
            ctx->PSSetShader(ps, nullptr, 0);
            ctx->VSSetConstantBuffers(0, 1, cbs3);
            ctx->PSSetConstantBuffers(0, 1, cbs3);
            ctx->RSSetState(rs);
            // The planes last, see-through (floor grey, walls blue, tables orange: one colour
            // here, the occluder buffer mixes them), then their edges.
            if (g_app.v3_planes && occ_count) {
                ctx->OMSetBlendState(alpha_blend, nullptr, 0xffffffff);
                ctx->OMSetDepthStencilState(dss_nowrite, 0);
                d3(occ_vb, occ_count, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, identity(), 0.35f, 0.55f, 1.0f, false, 0.18f);
                ctx->OMSetDepthStencilState(dss, 0);
                ctx->OMSetBlendState(nullptr, nullptr, 0xffffffff);
                ctx->RSSetState(rs_wire);
                d3(occ_vb, occ_count, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, identity(), 0.4f, 0.6f, 1.0f, false);
                ctx->RSSetState(rs);
            }
            ID3D11RenderTargetView* none_rt[] = {nullptr};
            ctx->OMSetRenderTargets(1, none_rt, nullptr);
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
            else if (snap_phase == 4 && el > snapshot_s + 4) { MoveFileExW((dir + L"dashboard.png").c_str(), (dir + L"dashboard_diag.png").c_str(), MOVEFILE_REPLACE_EXISTING); g_app.tab_request = 2; snap_phase = 5; }
            else if (snap_phase == 5 && el > snapshot_s + 5) { g_app.snapshot = true; snap_phase = 6; }
            else if (snap_phase == 6 && el > snapshot_s + 6) { MoveFileExW((dir + L"dashboard.png").c_str(), (dir + L"dashboard_3d.png").c_str(), MOVEFILE_REPLACE_EXISTING); g_app.tab_request = 2; snap_phase = 7; snapshot_s = 0; }
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
            int rc = vio::set_display_mode(want == 4 ? vio::MODE_3D_90 : want == 3 ? vio::MODE_3D : vio::MODE_2D);
            printf("display %s -> %d\n", want == 4 ? "3D 90 Hz" : want == 3 ? "3D" : "2D", rc);
            if (rc == 0) {
                auto hz = [&]() { DEVMODEW dm{}; dm.dmSize = sizeof dm; return EnumDisplaySettingsW(mon.gdi.c_str(), ENUM_CURRENT_SETTINGS, &dm) ? (int)dm.dmDisplayFrequency : 0; };
                for (int i = 0; i < 40; i++) {
                    Sleep(150); glasses_monitor(mon);
                    bool wide = width(mon.rect) >= 3000, ok = wide == (want >= 3);
                    if (want == 4) ok = ok && hz() >= 85;
                    if (want == 3) ok = ok && hz() < 75;
                    if (ok) break;
                }
                g_app.screen_hz = hz();
                hands::logf("display: %dx%d at %d Hz\n", width(mon.rect), height(mon.rect), g_app.screen_hz);
                W = width(mon.rect); H = height(mon.rect);
                SetWindowPos(hwnd, nullptr, mon.rect.left, mon.rect.top, W, H, SWP_NOZORDER | SWP_NOACTIVATE);
                ctx->OMSetRenderTargets(0, nullptr, nullptr);
                if (FAILED(make_targets())) { fprintf(stderr, "resize failed\n"); break; }
                g_app.screen_w = W; g_app.screen_h = H;
                switched = want >= 3;
            }
        }
        // The screen's brightness, the film, the tracking cameras' exposure (engine / MCU).
        if (g_app.brightness_request >= 0) { int rc = vio::set_brightness(g_app.brightness_request); hands::logf("brightness %d -> %d\n", g_app.brightness_request, rc); if (rc == 0) g_app.brightness = g_app.brightness_request; g_app.brightness_request = -1; }
        if (g_app.film_request >= 0) { int rc = vio::set_film(g_app.film_request == 1); hands::logf("film %d -> %d\n", g_app.film_request, rc); if (rc == 0) g_app.film_dark = g_app.film_request == 1; g_app.film_request = -1; }
        if (g_app.expo_request >= 0) { int rc = vio::set_exposure_fixed(g_app.expo_request == 1); hands::logf("tracking exposure %s -> %d\n", g_app.expo_request ? "fixed" : "auto", rc); if (rc >= 0) g_app.expo_fixed = g_app.expo_request == 1; g_app.expo_request = -1; }
        { double p = g_vsync_period; g_app.vsync_hz = p > 0 ? (float)(1.0 / p) : 0.0f; }
        if (!g_app.screen_hz) { DEVMODEW dm{}; dm.dmSize = sizeof dm; if (EnumDisplaySettingsW(mon.gdi.c_str(), ENUM_CURRENT_SETTINGS, &dm)) g_app.screen_hz = (int)dm.dmDisplayFrequency; }

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
    dany::stop();
    room::stop();
    mcp::stop();
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
