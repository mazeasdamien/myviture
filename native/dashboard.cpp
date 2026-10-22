// The dashboard: a normal window on the PC's screen, drawn with Dear ImGui on the same
// D3D11 device as the glasses. It is drawn after the glasses' frame is presented, and
// presented without waiting for vsync, so it never delays the glasses.

#include "dashboard.hpp"

#include <dxgi1_2.h>
#include <wincodec.h>
#pragma comment(lib, "windowscodecs.lib")

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "third_party/imgui/imgui.h"
#include "third_party/imgui/backends/imgui_impl_dx11.h"
#include "third_party/imgui/backends/imgui_impl_win32.h"

#include "orange.hpp"
#include "rgbcam.hpp"
#include "hands.hpp"
#include "rgbcal.hpp"

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

namespace dash {

static HWND hwnd = nullptr;
static ID3D11Device* dev = nullptr;
static ID3D11DeviceContext* ctx = nullptr;
static IDXGISwapChain1* sc = nullptr;
static ID3D11RenderTargetView* rtv = nullptr;
static UINT pending_w = 0, pending_h = 0;
static bool closed = false;
static std::string ini_path;

// Camera images as RGBA textures (ImGui draws a single-channel texture in red).
struct CamTex { ID3D11Texture2D* tex = nullptr; ID3D11ShaderResourceView* srv = nullptr; int w = 0, h = 0; };
static CamTex cams[2], rgb;
static uint32_t cam_seq = 0, rgb_seq = 0;
static std::vector<uint32_t> rgba;

static void make_rtv() {
    ID3D11Texture2D* back = nullptr;
    sc->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&back);
    dev->CreateRenderTargetView(back, nullptr, &rtv);
    back->Release();
}

static LRESULT CALLBACK wndproc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (ImGui_ImplWin32_WndProcHandler(h, msg, wp, lp)) return 1;
    switch (msg) {
    case WM_SIZE:
        if (wp != SIZE_MINIMIZED) { pending_w = LOWORD(lp); pending_h = HIWORD(lp); }
        return 0;
    case WM_CLOSE: closed = true; return 0;
    }
    return DefWindowProcW(h, msg, wp, lp);
}

bool init(ID3D11Device* d, ID3D11DeviceContext* c, const RECT& area, const std::string& ini) {
    dev = d; ctx = c; ini_path = ini;
    WNDCLASSW wc{}; wc.lpfnWndProc = wndproc; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"myviture-dashboard";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);
    int aw = area.right - area.left, ah = area.bottom - area.top;
    int w = std::min(aw, (int)(aw * 0.92)), h = std::min(ah, (int)(ah * 0.9));
    hwnd = CreateWindowExW(0, wc.lpszClassName, L"myViture", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                           area.left + (aw - w) / 2, area.top + (ah - h) / 2, w, h, nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd) return false;

    IDXGIDevice* dx = nullptr; IDXGIAdapter* ad = nullptr; IDXGIFactory2* f = nullptr;
    dev->QueryInterface(__uuidof(IDXGIDevice), (void**)&dx);
    dx->GetAdapter(&ad);
    ad->GetParent(__uuidof(IDXGIFactory2), (void**)&f);
    DXGI_SWAP_CHAIN_DESC1 sd{};
    sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM; sd.SampleDesc.Count = 1; sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 2; sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    HRESULT hr = f->CreateSwapChainForHwnd(dev, hwnd, &sd, nullptr, nullptr, &sc);
    f->Release(); ad->Release(); dx->Release();
    if (FAILED(hr)) return false;
    make_rtv();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = ini_path.c_str();
    float dpi = ImGui_ImplWin32_GetDpiScaleForHwnd(hwnd);
    ImGui::StyleColorsDark();
    ImGuiStyle& st = ImGui::GetStyle();
    st.WindowRounding = 6; st.FrameRounding = 4; st.GrabRounding = 4;
    st.ScaleAllSizes(dpi);
    st.FontScaleDpi = dpi;
    st.FontSizeBase = 15;
    if (!io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\segoeui.ttf", 15)) io.Fonts->AddFontDefault();
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(dev, ctx);
    return true;
}

static void update_cameras() {
    std::lock_guard<std::mutex> l(g_cams.mu);
    if (g_cams.seq == cam_seq || !g_cams.w) return;
    cam_seq = g_cams.seq;
    for (int i = 0; i < 2; i++) {
        CamTex& t = cams[i];
        if (g_cams.px[i].size() < (size_t)g_cams.w * g_cams.h) continue;
        if (t.w != g_cams.w || t.h != g_cams.h) {
            if (t.srv) t.srv->Release();
            if (t.tex) t.tex->Release();
            D3D11_TEXTURE2D_DESC td{};
            td.Width = g_cams.w; td.Height = g_cams.h; td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DYNAMIC; td.BindFlags = D3D11_BIND_SHADER_RESOURCE; td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            dev->CreateTexture2D(&td, nullptr, &t.tex);
            dev->CreateShaderResourceView(t.tex, nullptr, &t.srv);
            t.w = g_cams.w; t.h = g_cams.h;
        }
        D3D11_MAPPED_SUBRESOURCE m;
        if (FAILED(ctx->Map(t.tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) continue;
        const uint8_t* src = g_cams.px[i].data();
        for (int y = 0; y < t.h; y++) {
            uint32_t* row = (uint32_t*)((uint8_t*)m.pData + y * m.RowPitch);
            for (int x = 0; x < t.w; x++) { uint32_t g = src[y * t.w + x]; row[x] = 0xff000000u | g << 16 | g << 8 | g; }
        }
        ctx->Unmap(t.tex, 0);
    }
}

static void ensure_tex(CamTex& t, int w, int h) {
    if (t.w == w && t.h == h && t.tex) return;
    if (t.srv) t.srv->Release();
    if (t.tex) t.tex->Release();
    D3D11_TEXTURE2D_DESC td{};
    td.Width = w; td.Height = h; td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DYNAMIC; td.BindFlags = D3D11_BIND_SHADER_RESOURCE; td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    dev->CreateTexture2D(&td, nullptr, &t.tex);
    dev->CreateShaderResourceView(t.tex, nullptr, &t.srv);
    t.w = w; t.h = h;
}

static void update_rgb() {
    std::lock_guard<std::mutex> l(rgbcam::g_frame.mu);
    auto& f = rgbcam::g_frame;
    if (f.seq == rgb_seq || !f.w) return;
    rgb_seq = f.seq;
    ensure_tex(rgb, f.w, f.h);
    D3D11_MAPPED_SUBRESOURCE m;
    if (FAILED(ctx->Map(rgb.tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) return;
    for (int y = 0; y < f.h; y++) {
        uint32_t* row = (uint32_t*)((uint8_t*)m.pData + y * m.RowPitch);
        const uint32_t* src = f.bgra.data() + (size_t)y * f.w;
        for (int x = 0; x < f.w; x++) row[x] = src[x] | 0xff000000u;  // RGB32 leaves alpha undefined
    }
    ctx->Unmap(rgb.tex, 0);
}

/**
 * Three axes on one chart, as on the web dashboard: x red, y green, z blue, a zero line,
 * the latest values in the legend.
 */
static void axis_chart(const char* title, const char* unit, const std::vector<float>* v, float range, float height) {
    static const ImU32 COLORS[3] = {IM_COL32(255, 93, 108, 255), IM_COL32(47, 211, 154, 255), IM_COL32(76, 141, 255, 255)};
    ImGui::TextUnformatted(title);
    ImGui::SameLine();
    ImGui::TextDisabled("%s", unit);
    for (int a = 0; a < 3; a++) {
        ImGui::SameLine();
        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(COLORS[a]), "%c %.3f", "xyz"[a], v[a].empty() ? 0.f : v[a].back());
    }
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImVec2 size(ImGui::GetContentRegionAvail().x, height);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, ImVec2(p0.x + size.x, p0.y + size.y), IM_COL32(12, 15, 20, 255), 4);
    float mid = p0.y + size.y / 2;
    for (int g = -2; g <= 2; g++) dl->AddLine(ImVec2(p0.x, mid + g * size.y / 5), ImVec2(p0.x + size.x, mid + g * size.y / 5), g ? IM_COL32(40, 46, 56, 255) : IM_COL32(70, 78, 92, 255));
    char lbl[32]; snprintf(lbl, sizeof lbl, "%.3g", range);
    dl->AddText(ImVec2(p0.x + 4, p0.y + 2), IM_COL32(110, 118, 130, 255), lbl);
    for (int a = 0; a < 3; a++) {
        const auto& s = v[a];
        if (s.size() < 2) continue;
        std::vector<ImVec2> pts(s.size());
        for (size_t i = 0; i < s.size(); i++) {
            float y = std::clamp(s[i] / range, -1.0f, 1.0f);
            pts[i] = ImVec2(p0.x + size.x * i / (s.size() - 1), mid - y * size.y / 2);
        }
        dl->AddPolyline(pts.data(), (int)pts.size(), COLORS[a], 0, 1.5f);
    }
    ImGui::Dummy(size);
}

/** Last n values of a buffer, every `step`-th, for a plot. */
static std::vector<float> tail(const std::vector<float>& v, size_t n, size_t step = 1) {
    std::vector<float> out;
    size_t from = v.size() > n * step ? v.size() - n * step : 0;
    for (size_t i = from; i < v.size(); i += step) out.push_back(v[i]);
    return out;
}

static void plot(const char* label, const std::vector<float>& v, float lo, float hi, float height, const char* overlay = nullptr) {
    ImGui::PlotLines(label, v.empty() ? nullptr : v.data(), (int)v.size(), 0, overlay, lo, hi, ImVec2(-1, height));
}

static void build(AppState& s) {
    const float em = ImGui::GetFontSize();
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::Begin("myViture", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);

    // ---- top bar
    ImGui::TextUnformatted("myViture");
    ImGui::SameLine();
    ImGui::TextDisabled("VITURE Luma Ultra · %s · moteur VITURE · application Windows", s.serial.c_str());
    bool sbs = s.screen_w >= 3000;
    char mode[48];
    snprintf(mode, sizeof mode, sbs ? "● 3D %dx%d" : "● 2D %dx%d", s.screen_w, s.screen_h);
    const char* buttons[] = {sbs ? "Passer en 2D" : "Passer en 3D", "Recentrer", "Capture", "Quitter"};
    const ImGuiStyle& sty = ImGui::GetStyle();
    float right_w = ImGui::CalcTextSize(mode).x;
    for (auto b : buttons) right_w += sty.ItemSpacing.x + ImGui::CalcTextSize(b).x + 2 * sty.FramePadding.x;
    ImGui::SameLine(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - right_w));
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(sbs ? ImVec4(0.4f, 0.9f, 0.6f, 1) : ImVec4(1, 0.7f, 0.3f, 1), "%s", mode);
    ImGui::SameLine();
    if (ImGui::Button(buttons[0])) s.display_request = sbs ? 2 : 3;
    ImGui::SameLine();
    if (ImGui::Button(buttons[1])) s.recenter = true;
    ImGui::SameLine();
    if (ImGui::Button(buttons[2])) s.snapshot = true;
    ImGui::SameLine();
    if (ImGui::Button(buttons[3])) s.quit = true;
    ImGui::Separator();

    const float col = ImGui::GetContentRegionAvail().x / 3 - ImGui::GetStyle().ItemSpacing.x;
    const float panel_h = ImGui::GetContentRegionAvail().y / 2 - ImGui::GetStyle().ItemSpacing.y;

    // ---- rendering on the glasses
    ImGui::BeginChild("rendu", ImVec2(col, panel_h), ImGuiChildFlags_Borders);
    ImGui::SeparatorText("Rendu dans les lunettes");
    const Stats& t = s.stats;
    ImGui::Text("%.1f images/s   %ld vblank ratées", t.fps, t.missed);
    ImGui::Text("Pose → écran  %.1f ms (p99 %.1f)", t.latency_p50, t.latency_p99);
    ImGui::Text("Pose %.0f Hz   CPU %.2f ms/image", t.pose_hz, t.cpu_ms);
    ImGui::Text("Tremblement %.2f° (p99 %.2f°)", t.shake_p50, t.shake_p99);
    ImGui::TextDisabled("Prédiction utilisée : %.0f %% du mouvement", t.lead_used * 100);
    plot("##shake", tail(s.shake_history, 300), 0, 1.0f, 4 * em, "tremblement par image (0–1°)");
    plot("##lat", tail(s.latency_history, 300), 0, 40, 3 * em, "pose → écran (0–40 ms)");
    ImGui::SeparatorText("Prédiction");
    ImGui::SliderFloat("avance (ms)", &s.predict_ms, 0, 60, "%.0f");
    ImGui::TextDisabled("Les objets traînent : +   ·   ils partent en avance : −");
    ImGui::TextDisabled("Prédiction du moteur VITURE (IMU, ~520 Hz)");
    ImGui::EndChild();

    // ---- head
    ImGui::SameLine();
    ImGui::BeginChild("tete", ImVec2(col, panel_h), ImGuiChildFlags_Borders);
    ImGui::SeparatorText("Tête (VITURE 6DoF)");
    ImGui::Text("Lacet   %7.1f°", s.yaw);
    ImGui::Text("Tangage %7.1f°", s.pitch);
    ImGui::Text("Roulis  %7.1f°", s.roll);
    ImGui::Text("Position  %.3f  %.3f  %.3f m", s.pos[0], s.pos[1], s.pos[2]);
    ImGui::Text("Vitesse  %.0f°/s", s.rate_dps);
    ImGui::SeparatorText("Affichage");
    ImGui::SliderFloat("champ vertical (°)", &s.vfov, 20, 40, "%.2f");
    ImGui::SliderFloat("écart des yeux (mm)", &s.ipd_mm, 50, 75, "%.1f");
    ImGui::SliderFloat("sol sous les yeux (m)", &s.floor_m, 0.5f, 2.2f, "%.2f");
    ImGui::Checkbox("Scène de test (cubes)", &s.scene);
    ImGui::SameLine();
    ImGui::Checkbox("Grille au sol", &s.floor_grid);
    if (ImGui::Button("Valeurs d'usine")) { s.vfov = 28.147f; s.ipd_mm = 63; s.floor_m = 1.6f; }
    ImGui::Checkbox("Prédire le mouvement des mains", &s.hand_predict);
    ImGui::EndChild();

    // ---- IMU
    ImGui::SameLine();
    ImGui::BeginChild("imu", ImVec2(0, panel_h), ImGuiChildFlags_Borders);
    std::vector<float> axes[6];
    long imu_count;
    { std::lock_guard<std::mutex> l(g_imu.mu); for (int i = 0; i < 6; i++) axes[i] = tail(g_imu.axis[i], 500, 4); imu_count = g_imu.count; }
    static long last_count = 0; static double last_t = 0; static float imu_hz = 0;
    double now = ImGui::GetTime();
    if (now - last_t > 1) { imu_hz = (float)((imu_count - last_count) / (now - last_t)); last_count = imu_count; last_t = now; }
    char title[160];
    snprintf(title, sizeof title, "IMU · %.0f Hz · 2 s", imu_hz);
    ImGui::SeparatorText(title);
    float chart_h = (ImGui::GetContentRegionAvail().y - 3 * ImGui::GetTextLineHeightWithSpacing()) / 2;
    axis_chart("Gyro", "rad/s", axes + 3, 2.0f, chart_h);
    axis_chart("Accél.", "m/s²", axes, 12.0f, chart_h);
    ImGui::EndChild();

    // ---- cameras
    update_cameras();
    update_rgb();
    const float bottom_h = ImGui::GetContentRegionAvail().y;
    ImGui::BeginChild("cams", ImVec2(col * 1.35f, bottom_h), ImGuiChildFlags_Borders);
    static long last_frames = 0; static double last_ft = 0; static float cam_fps = 0;
    long frames; { std::lock_guard<std::mutex> l(g_cams.mu); frames = g_cams.count; }
    if (now - last_ft > 1) { cam_fps = (float)((frames - last_frames) / (now - last_ft)); last_frames = frames; last_ft = now; }
    snprintf(title, sizeof title, "Caméras de suivi · %.0f i/s", cam_fps);
    ImGui::SeparatorText(title);
    hands::Result hr;
    { std::lock_guard<std::mutex> l(hands::g_mu); hr = hands::g_result; }
    // Fixed height (three lines), so the images below never move.
    const float line = ImGui::GetTextLineHeightWithSpacing();
    ImGui::BeginChild("handinfo", ImVec2(0, 3 * line + sty.FramePadding.y * 2), ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);
    ImGui::Checkbox("Mains", &s.hands);
    hands::g_enabled = s.hands;
    ImGui::SameLine();
    if (!hr.status.empty()) ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "%s", hr.status.c_str());
    else ImGui::TextDisabled("%.0f/s · %s · %.0f ms/paire · image → main %.0f ms", hr.rate, hr.device.c_str(), hr.ms_total, hr.latency_ms);
    for (int side = 0; side < 2; side++) {
        const hands::Hand* h = nullptr;
        for (auto& x : hr.hands) if (x.right == (side == 1)) h = &x;
        if (h) ImGui::TextDisabled("%s : pince %.1f cm%s · stéréo %.1f mm · poignet %.2f %.2f %.2f m", side ? "droite" : "gauche", h->pinch_m * 100,
                                   h->pinching ? " ✓" : "", h->gap_mm, h->joints[0][0], h->joints[0][1], h->joints[0][2]);
        else ImGui::TextDisabled("%s : —", side ? "droite" : "gauche");
    }
    ImGui::EndChild();
    float avail_w = ImGui::GetContentRegionAvail().x / 2 - ImGui::GetStyle().ItemSpacing.x, avail_h = ImGui::GetContentRegionAvail().y;
    for (int i = 0; i < 2; i++) {
        if (i) ImGui::SameLine();
        if (!cams[i].srv) { ImGui::Dummy(ImVec2(avail_w, avail_h)); continue; }
        float k = std::min(avail_w / cams[i].w, (ImGui::GetContentRegionAvail().y) / cams[i].h);
        ImVec2 o = ImGui::GetCursorScreenPos();
        ImGui::Image((ImTextureID)(intptr_t)cams[i].srv, ImVec2(cams[i].w * k, cams[i].h * k));
        ImDrawList* dl = ImGui::GetWindowDrawList();
        for (auto& h : hr.hands) {
            ImU32 col = h.right ? IM_COL32(255, 122, 69, 255) : IM_COL32(76, 141, 255, 255);
            for (auto& b : hands::BONES)
                dl->AddLine(ImVec2(o.x + h.image[i][b[0]][0] * k, o.y + h.image[i][b[0]][1] * k), ImVec2(o.x + h.image[i][b[1]][0] * k, o.y + h.image[i][b[1]][1] * k), col, 2);
        }
    }
    ImGui::EndChild();

    // ---- RGB camera
    ImGui::SameLine();
    ImGui::BeginChild("rgb", ImVec2(col * 1.15f, bottom_h), ImGuiChildFlags_Borders);
    static long last_rgb = 0; static double last_rt = 0; static float rgb_fps = 0;
    std::string rgb_name, rgb_status;
    { std::lock_guard<std::mutex> l(rgbcam::g_frame.mu); rgb_name = rgbcam::g_frame.name; rgb_status = rgbcam::g_frame.status;
      if (now - last_rt > 1) { rgb_fps = (float)((rgbcam::g_frame.count - last_rgb) / (now - last_rt)); last_rgb = rgbcam::g_frame.count; last_rt = now; } }
    snprintf(title, sizeof title, "Caméra RGB · %dx%d · %.0f i/s", rgb.w, rgb.h, rgb_fps);
    ImGui::SeparatorText(title);
    // One line always, so the image never moves.
    if (!rgb_status.empty()) ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "%s", rgb_status.c_str());
    else ImGui::TextDisabled("%s", rgb_name.c_str());
    ImGui::SameLine();
    ImGui::Checkbox("Virtuel", &s.rgb_overlay);
    ImGui::SameLine();
    if (ImGui::SmallButton("Oublier le calage RGB")) rgbcal::reset();
    {
        rgbcal::Fit fi; bool fitted; size_t np;
        { std::lock_guard<std::mutex> l(rgbcal::g_mu); fi = rgbcal::g_fit; fitted = rgbcal::g_cam.fitted; np = rgbcal::g_pairs.size(); }
        if (fitted) ImGui::TextDisabled("calage RGB : écart %.1f px sur %d points · rotation %.1f° · décalage %.1f %.1f %.1f cm · focale ×%.2f", fi.rms_px, fi.n, fi.rot_deg, fi.shift_cm[0], fi.shift_cm[1], fi.shift_cm[2], fi.f_rel);
        else ImGui::TextDisabled("calage RGB : caméra par défaut · %zu points (montre tes mains, immobiles, à plusieurs endroits)%s", np, fi.n && !fi.ok ? " · dernier essai refusé" : "");
    }
    if (rgb.srv) {
        ImVec2 a = ImGui::GetContentRegionAvail();
        float k = std::min(a.x / rgb.w, a.y / rgb.h);
        ImVec2 o = ImGui::GetCursorScreenPos();
        ImGui::Image((ImTextureID)(intptr_t)rgb.srv, ImVec2(rgb.w * k, rgb.h * k));
        if (s.rgb_overlay) {
            // The virtual world as the RGB camera should see it: hands (stereo, through the
            // fitted camera) and the test cubes; in green, where the hand model sees the
            // hand in the RGB image itself. The gap between the two is the misalignment.
            rgbcal::Cam cam = rgbcal::current(rgb.w, rgb.h);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            auto px = [&](const double p[3], ImVec2& out) { double u, v; if (!rgbcal::project(cam, p, u, v)) return false; out = ImVec2(o.x + (float)u * k, o.y + (float)v * k); return true; };
            for (auto& h : hr.hands) {
                ImU32 col = h.right ? IM_COL32(255, 122, 69, 230) : IM_COL32(76, 141, 255, 230);
                for (auto& b : hands::BONES) {
                    double p0[3] = {h.joints[b[0]][0], h.joints[b[0]][1], h.joints[b[0]][2]}, p1[3] = {h.joints[b[1]][0], h.joints[b[1]][1], h.joints[b[1]][2]};
                    ImVec2 a0, a1;
                    if (px(p0, a0) && px(p1, a1)) dl->AddLine(a0, a1, col, 2.5f);
                }
                if (h.rgb_seen) for (auto& b : hands::BONES)
                    dl->AddLine(ImVec2(o.x + h.rgb[b[0]][0] * k, o.y + h.rgb[b[0]][1] * k), ImVec2(o.x + h.rgb[b[1]][0] * k, o.y + h.rgb[b[1]][1] * k), IM_COL32(80, 255, 120, 200), 1.5f);
            }
            if (s.scene) {
                // scene -> head: rigid inverse of head -> scene
                const float* m = s.head_scene;
                auto to_head = [&](float x, float y, float z, double out[3]) {
                    float dx = x - m[3], dy = y - m[7], dz = z - m[11];
                    out[0] = m[0] * dx + m[4] * dy + m[8] * dz; out[1] = m[1] * dx + m[5] * dy + m[9] * dz; out[2] = m[2] * dx + m[6] * dy + m[10] * dz;
                };
                auto cube_at = [&](float x, float y, float z, float half, ImU32 col) {
                    // the cube's 8 corners -> 12 edges
                    ImVec2 c[8]; bool ok = true;
                    for (int i = 0; i < 8 && ok; i++) { double p[3]; to_head(x + (i & 1 ? half : -half), y + (i & 2 ? half : -half), z + (i & 4 ? half : -half), p); ok = px(p, c[i]); }
                    if (!ok) return;
                    for (int i = 0; i < 8; i++) for (int b = 1; b < 8; b <<= 1) if (!(i & b)) dl->AddLine(c[i], c[i | b], col, 1.5f);
                };
                for (int i = 0; i < 8; i++) {
                    float a = i * 3.14159265f / 4, hue = i / 8.0f;
                    ImU32 col = IM_COL32((int)(255 * (0.5f + 0.5f * cosf(6.2832f * hue))), (int)(255 * (0.5f + 0.5f * cosf(6.2832f * (hue - 0.333f)))), (int)(255 * (0.5f + 0.5f * cosf(6.2832f * (hue - 0.667f)))), 220);
                    cube_at(sinf(a) * 0.9f, 0, -cosf(a) * 0.9f, 0.06f, col);
                }
                cube_at(0, -0.1f, -1.0f, 0.025f, IM_COL32(255, 255, 255, 220));
            }
        }
    }
    ImGui::EndChild();

    // ---- AI cost
    ImGui::SameLine();
    ImGui::BeginChild("ia", ImVec2(0, bottom_h), ImGuiChildFlags_Borders);
    ImGui::SeparatorText("Coût IA (passerelle Orange)");
    {
        std::lock_guard<std::mutex> l(orange::g_costs.mu);
        auto& c = orange::g_costs;
        auto usd = [](double v) { static char b[32]; if (v < 0.01) snprintf(b, sizeof b, "%.3f c$", v * 100); else snprintf(b, sizeof b, "%.3f $", v); return std::string(b); };
        ImGui::Text("Session  %s", usd(c.session).c_str());
        ImGui::TextDisabled("%ld appels · dernier %s", c.calls, usd(c.last).c_str());
        ImGui::Text("vision %s", usd(c.vision).c_str());
        ImGui::Text("transcription %s", usd(c.transcription).c_str());
        ImGui::Text("voix %s", usd(c.speech).c_str());
        ImGui::Separator();
        if (std::isfinite(c.key_spend)) {
            ImGui::Text("Clé du labo  %.2f / %.0f $", c.key_spend, c.key_budget);
            if (c.key_budget > 0) ImGui::ProgressBar((float)(c.key_spend / c.key_budget), ImVec2(-1, 0));
            ImGui::TextDisabled("partagée avec la cellule robot");
        } else ImGui::TextDisabled("Clé du labo : inconnue");
        if (!c.status.empty()) ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "%s", c.status.c_str());
    }
    if (ImGui::Button(orange::g_busy ? "…" : "Actualiser (≈0,0005 c$)")) orange::refresh_spend();
    ImGui::EndChild();

    ImGui::End();
}

void frame(AppState& s) {
    if (closed) { s.quit = true; return; }
    if (pending_w && pending_h) {
        if (rtv) { rtv->Release(); rtv = nullptr; }
        sc->ResizeBuffers(0, pending_w, pending_h, DXGI_FORMAT_UNKNOWN, 0);
        make_rtv();
        pending_w = pending_h = 0;
    }
    if (IsIconic(hwnd)) return;
    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    build(s);
    ImGui::Render();
    const float clear[4] = {0.06f, 0.07f, 0.09f, 1};
    ctx->OMSetRenderTargets(1, &rtv, nullptr);
    ctx->ClearRenderTargetView(rtv, clear);
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
    if (s.snapshot) {
        s.snapshot = false;
        wchar_t exe[MAX_PATH];
        GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring png = exe;
        png = png.substr(0, png.find_last_of(L'\\') + 1) + L"dashboard.png";
        save_png(png);
    }
    sc->Present(0, 0);
}

/** The dashboard's last frame as a PNG (WIC), for checks without capturing the screen. */
bool save_png(const std::wstring& path) {
    ID3D11Texture2D* back = nullptr;
    if (FAILED(sc->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&back))) return false;
    D3D11_TEXTURE2D_DESC td; back->GetDesc(&td);
    td.Usage = D3D11_USAGE_STAGING; td.BindFlags = 0; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ; td.MiscFlags = 0;
    ID3D11Texture2D* staging = nullptr;
    bool ok = SUCCEEDED(dev->CreateTexture2D(&td, nullptr, &staging));
    if (ok) ctx->CopyResource(staging, back);
    back->Release();
    D3D11_MAPPED_SUBRESOURCE m{};
    ok = ok && SUCCEEDED(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &m));
    IWICImagingFactory* wic = nullptr; IWICStream* stream = nullptr; IWICBitmapEncoder* enc = nullptr; IWICBitmapFrameEncode* fr = nullptr;
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    ok = ok && SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic)));
    ok = ok && SUCCEEDED(wic->CreateStream(&stream)) && SUCCEEDED(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE));
    ok = ok && SUCCEEDED(wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc)) && SUCCEEDED(enc->Initialize(stream, WICBitmapEncoderNoCache));
    ok = ok && SUCCEEDED(enc->CreateNewFrame(&fr, nullptr)) && SUCCEEDED(fr->Initialize(nullptr)) && SUCCEEDED(fr->SetSize(td.Width, td.Height));
    WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
    ok = ok && SUCCEEDED(fr->SetPixelFormat(&fmt)) && SUCCEEDED(fr->WritePixels(td.Height, m.RowPitch, m.RowPitch * td.Height, (BYTE*)m.pData));
    ok = ok && SUCCEEDED(fr->Commit()) && SUCCEEDED(enc->Commit());
    if (fr) fr->Release(); if (enc) enc->Release(); if (stream) stream->Release(); if (wic) wic->Release();
    if (m.pData) ctx->Unmap(staging, 0);
    if (staging) staging->Release();
    return ok;
}

void shutdown() {
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    for (auto& t : cams) { if (t.srv) t.srv->Release(); if (t.tex) t.tex->Release(); }
    if (rgb.srv) rgb.srv->Release();
    if (rgb.tex) rgb.tex->Release();
    if (rtv) rtv->Release();
    if (sc) sc->Release();
    if (hwnd) DestroyWindow(hwnd);
}

}  // namespace dash
