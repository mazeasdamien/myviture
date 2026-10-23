// The dashboard: a normal window on the PC's screen, drawn with Dear ImGui on the same
// D3D11 device as the glasses. It is drawn after the glasses' frame is presented, and
// presented without waiting for vsync, so it never delays the glasses.

#include "dashboard.hpp"

#include <dxgi1_2.h>
#include <wincodec.h>
#include <shellapi.h>
#pragma comment(lib, "shell32.lib")
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
#include "vhands.hpp"
#include "recorder.hpp"
#include "rgbcal.hpp"
#include "roommap.hpp"

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
static CamTex cams[2], rgb, depth_tex;
static void ensure_tex(CamTex& t, int w, int h, DXGI_FORMAT fmt);

/** The last depth map as a colour image: near warm, far cold, black where unknown. */
static void update_depth() {
    std::vector<float> d;
    { std::lock_guard<std::mutex> l(room::g.mu); d = room::g.depth; }
    if (d.size() != (size_t)depth::W * depth::H) return;
    ensure_tex(depth_tex, depth::W, depth::H, DXGI_FORMAT_B8G8R8A8_UNORM);
    D3D11_MAPPED_SUBRESOURCE m;
    if (FAILED(ctx->Map(depth_tex.tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) return;
    for (int y = 0; y < depth::H; y++) {
        uint32_t* row = (uint32_t*)((uint8_t*)m.pData + y * m.RowPitch);
        for (int x = 0; x < depth::W; x++) {
            float z = d[y * depth::W + x];
            if (!(z > 0)) { row[x] = 0xff000000u; continue; }
            float t = std::clamp((logf(z) - logf(0.2f)) / (logf(4.0f) - logf(0.2f)), 0.0f, 1.0f);
            int r = (int)(255 * powf(1 - t, 0.8f)), g = (int)(255 * sinf(3.14159f * std::min(1.0f, t * 1.1f)) * 0.85f), b = (int)(255 * t);
            row[x] = 0xff000000u | (r << 16) | (g << 8) | b;
        }
    }
    ctx->Unmap(depth_tex.tex, 0);
}
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

static void ensure_tex(CamTex& t, int w, int h, DXGI_FORMAT fmt = DXGI_FORMAT_B8G8R8A8_UNORM) {
    if (t.w == w && t.h == h && t.tex) return;
    if (t.srv) t.srv->Release();
    if (t.tex) t.tex->Release();
    D3D11_TEXTURE2D_DESC td{};
    td.Width = w; td.Height = h; td.MipLevels = 1; td.ArraySize = 1; td.Format = fmt;
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
    // X8 format: the camera's undefined alpha byte is ignored, rows copy as they are.
    ensure_tex(rgb, f.w, f.h, DXGI_FORMAT_B8G8R8X8_UNORM);
    D3D11_MAPPED_SUBRESOURCE m;
    if (FAILED(ctx->Map(rgb.tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) return;
    for (int y = 0; y < f.h; y++) memcpy((uint8_t*)m.pData + y * m.RowPitch, f.bgra.data() + (size_t)y * f.w, (size_t)f.w * 4);
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

/** A label above a full-width slider (narrow panels cut labels on the right). */
static bool slider_f(const char* label, float* v, float lo, float hi, const char* fmt) {
    ImGui::TextUnformatted(label);
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::PushID(label);
    bool r = ImGui::SliderFloat("##v", v, lo, hi, fmt);
    ImGui::PopID();
    return r;
}
static bool slider_i(const char* label, int* v, int lo, int hi) {
    ImGui::TextUnformatted(label);
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::PushID(label);
    bool r = ImGui::SliderInt("##v", v, lo, hi);
    ImGui::PopID();
    return r;
}

/** The loaded or last fit's numbers (a reloaded fit has not run this session). */
static void fit_numbers(double& rms, int& n) {
    std::lock_guard<std::mutex> l(rgbcal::g_mu);
    rms = rgbcal::g_fit.n ? rgbcal::g_fit.rms_px : rgbcal::g_loaded_rms;
    n = rgbcal::g_fit.n ? rgbcal::g_fit.n : rgbcal::g_loaded_n;
}

/**
 * The RGB view in large: the RGB camera's picture with the virtual scene drawn from that
 * camera, and what to record it with.
 */
static void rgb_view(AppState& s) {
    const float em = ImGui::GetFontSize();
    const ImGuiStyle& sty = ImGui::GetStyle();
    const float side = 24 * em;
    ImVec2 a = ImGui::GetContentRegionAvail();
    rec::Status rs; { std::lock_guard<std::mutex> l(rec::g_mu); rs = rec::g_status; }

    ImGui::BeginChild("big", ImVec2(a.x - side - sty.ItemSpacing.x, a.y), ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);
    {
        // What the view is made of: the camera, its mode, the frames that arrive, the fit.
        static long last_n = 0; static double last_t = 0; static float fps = 0;
        std::string name, mode, status; long n;
        { std::lock_guard<std::mutex> l(rgbcam::g_frame.mu); name = rgbcam::g_frame.name; mode = rgbcam::g_frame.mode; status = rgbcam::g_frame.status; n = rgbcam::g_frame.count; }
        double now = ImGui::GetTime();
        if (now - last_t > 1) { fps = (float)((n - last_n) / (now - last_t)); last_n = n; last_t = now; }
        rgbcal::Cam cam = rgbcal::get();
        rgbcal::Fit fi; { std::lock_guard<std::mutex> l(rgbcal::g_mu); fi = rgbcal::g_fit; }
        if (!status.empty()) ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "%s", status.c_str());
        else {
            ImGui::Text("%s · %s · reçues %.0f i/s · vue %dx%d · retard %.0f ms", name.c_str(), mode.c_str(), fps, s.mr_w, s.mr_h, s.rgb_latency_ms);
            ImGui::SameLine();
            double rms; int npts; fit_numbers(rms, npts);
            if (cam.fitted && npts) ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.6f, 1), "· calée (écart %.1f px, focale %.0f px)", rms, cam.f);
            else if (cam.fitted) ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.6f, 1), "· calée (calage enregistré, focale %.0f px)", cam.f);
            else ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "· pas encore calée (focale par défaut %.0f px)", cam.f);
        }
    }
    if (s.mr_srv && s.mr_w) {
        ImVec2 b = ImGui::GetContentRegionAvail();
        float k = std::min(b.x / s.mr_w, b.y / s.mr_h);
        ImVec2 size(s.mr_w * k, s.mr_h * k);
        ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX() + (b.x - size.x) / 2, ImGui::GetCursorPosY() + (b.y - size.y) / 2));
        ImVec2 o = ImGui::GetCursorScreenPos();
        ImGui::Image((ImTextureID)(intptr_t)s.mr_srv, size);
        if (rs.on) {
            // REC mark over the picture (not recorded: drawn by the dashboard only).
            ImDrawList* dl = ImGui::GetWindowDrawList();
            bool blink = fmod(ImGui::GetTime(), 1.0) < 0.6;
            if (blink) dl->AddCircleFilled(ImVec2(o.x + 1.2f * em, o.y + 1.2f * em), 0.5f * em, IM_COL32(235, 50, 50, 255));
            char t[48]; int sec = (int)rs.seconds;
            snprintf(t, sizeof t, "REC %02d:%02d", sec / 60, sec % 60);
            dl->AddText(ImVec2(o.x + 2.0f * em, o.y + 0.55f * em), IM_COL32(255, 255, 255, 230), t);
        }
    } else ImGui::TextDisabled("Pas encore d'image de la caméra RGB.");
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("side", ImVec2(0, a.y), ImGuiChildFlags_Borders);
    ImGui::SeparatorText("Enregistrement");
    ImGui::PushStyleColor(ImGuiCol_Button, rs.on ? ImVec4(0.75f, 0.15f, 0.15f, 1) : ImVec4(0.20f, 0.25f, 0.32f, 1));
    if (ImGui::Button(rs.on ? "■  Arrêter l'enregistrement" : "●  Enregistrer", ImVec2(-1, 2.2f * em))) s.rec_request = rs.on ? 2 : 1;
    ImGui::PopStyleColor();
    ImGui::BeginDisabled(rs.on);
    ImGui::Checkbox("Avec le son (micro)", &s.rec_audio);
    slider_i("qualité (Mbit/s)", &s.rec_mbps, 2, 30);
    ImGui::EndDisabled();
    ImGui::Checkbox("Virtuel par-dessus l'image", &s.rgb_overlay);
    if (ImGui::Button("Photo (PNG)")) s.photo_request = true;
    ImGui::SameLine();
    if (ImGui::Button("Ouvrir le dossier")) ShellExecuteW(nullptr, L"open", rec::folder().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    if (rs.on) {
        int sec = (int)rs.seconds;
        ImGui::Text("%02d:%02d · %ld images%s", sec / 60, sec % 60, rs.frames, rs.dropped ? "" : "");
        if (rs.dropped) ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "%ld images perdues (encodeur en retard)", rs.dropped);
        if (rs.audio) ImGui::TextDisabled("son : %.1f s · %s", rs.audio_seconds, rs.mic.c_str());
        else ImGui::TextDisabled("sans son");
    }
    if (!rs.path.empty()) ImGui::TextWrapped("%s %s", rs.on ? "vers" : "dernier :", rs.path.c_str());
    if (!rs.error.empty()) ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "%s", rs.error.c_str());
    ImGui::TextWrapped("MP4 H.264 %dx%d, son AAC · dossier Vidéos\\myViture", s.mr_w, s.mr_h);

    ImGui::SeparatorText("Superposition");
    slider_f("âge de l'image RGB à son arrivée (ms)", &s.rgb_latency_ms, 0, 200, "%.0f");
    ImGui::TextWrapped("Le virtuel est dessiné avec la tête où elle était quand l'image a été prise. Tourne la tête : si le virtuel part en avance sur l'image, augmente ; s'il traîne derrière, diminue.");
    {
        rgbcal::Fit fi; bool fitted; size_t np;
        { std::lock_guard<std::mutex> l(rgbcal::g_mu); fi = rgbcal::g_fit; fitted = rgbcal::g_cam.fitted; np = (size_t)rgbcal::g_frames; }
        double rms; int npts; fit_numbers(rms, npts);
        if (fitted && npts) ImGui::TextWrapped("Caméra RGB calée : écart %.1f px sur %d points", rms, npts);
        else if (fitted) ImGui::TextWrapped("Caméra RGB calée (calage enregistré)");
        else ImGui::TextWrapped("Caméra RGB pas encore calée (%zu/8 poses) : montre une main immobile, à plusieurs endroits.", np);
        if (ImGui::SmallButton("Oublier le calage RGB")) rgbcal::reset();
        bool refine = !rgbcal::g_frozen;
        if (fitted && ImGui::Checkbox("Continuer à affiner le calage", &refine)) rgbcal::g_frozen = !refine;
        if (fitted) ImGui::TextDisabled(rgbcal::g_frozen ? "calage figé (gardé d'une session à l'autre)" : "affinage en cours : montre une main immobile");
    }

    ImGui::SeparatorText("Mains (VITURE)");
    {
        bool l, r; std::string st;
        { std::lock_guard<std::mutex> lk(vhands::g_mu); st = vhands::g_out.status; l = vhands::g_out.data[0] != 0; r = vhands::g_out.data[vhands::PER_HAND] != 0; }
        if (!st.empty()) ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "VITURE : %s", st.c_str());
        else ImGui::TextDisabled("VITURE : gauche %s · droite %s", l ? "vue" : "—", r ? "vue" : "—");
    }
    ImGui::Checkbox("Mains réalistes", &s.real_hands);
    ImGui::SameLine();
    ImGui::Checkbox("Manches", &s.sleeves);
    if (s.sleeves) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(7 * ImGui::GetFontSize());
        ImGui::ColorEdit3("##manche", s.sleeve_color, ImGuiColorEditFlags_NoInputs);
    }
    ImGui::Checkbox("Ombres de contact", &s.shadows);
    ImGui::SameLine();
    ImGui::Checkbox("Grain et flou de la caméra", &s.camera_look);
    ImGui::Checkbox("Lumière de la pièce sur le virtuel", &s.room_light);
    ImGui::SameLine();
    ImGui::TextDisabled("(%.0f %%)", s.room_level * 100);
    ImGui::Checkbox("Panneau de test", &s.panel);
    ImGui::SameLine();
    ImGui::Checkbox("Maillage de la pièce", &s.show_map);
    ImGui::Checkbox("Occultation par la pièce (le réel cache le virtuel)", &s.occlusion);
    ImGui::EndChild();
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
    const char* buttons[] = {sbs ? "Passer en 2D" : "Passer en 3D", "Recentrer", "Quitter"};
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
    if (ImGui::Button(buttons[2])) s.quit = true;
    if (s.notice_until < 0) s.notice_until = ImGui::GetTime() + 6;
    {
        rec::Status rs; { std::lock_guard<std::mutex> l(rec::g_mu); rs = rec::g_status; }
        if (rs.on) { ImGui::SameLine(); ImGui::TextColored(ImVec4(1, 0.35f, 0.35f, 1), "● REC %02d:%02d", (int)rs.seconds / 60, (int)rs.seconds % 60); }
    }
    if (!s.notice.empty() && ImGui::GetTime() < s.notice_until) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.4f, 1, 0.6f, 1), "%s", s.notice.c_str());
    }
    ImGui::Separator();

    if (!ImGui::BeginTabBar("vues")) { ImGui::End(); return; }
    int req = s.tab_request; s.tab_request = -1;
    if (ImGui::BeginTabItem("Vue RGB + virtuel", nullptr, req == 0 ? ImGuiTabItemFlags_SetSelected : 0)) { rgb_view(s); ImGui::EndTabItem(); }
    if (!ImGui::BeginTabItem("Diagnostics", nullptr, req == 1 ? ImGuiTabItemFlags_SetSelected : 0)) { ImGui::EndTabBar(); ImGui::End(); return; }

    const float col = ImGui::GetContentRegionAvail().x / 3 - ImGui::GetStyle().ItemSpacing.x;
    const float panel_h = ImGui::GetContentRegionAvail().y * 0.58f - ImGui::GetStyle().ItemSpacing.y;  // the head panel's sliders fit

    // ---- rendering on the glasses
    ImGui::BeginChild("rendu", ImVec2(col, panel_h), ImGuiChildFlags_Borders);
    ImGui::SeparatorText("Rendu dans les lunettes");
    const Stats& t = s.stats;
    ImGui::Text("%.1f images/s   %ld vblank ratées", t.fps, t.missed);
    ImGui::Text("Pose → écran  %.1f ms (p99 %.1f)", t.latency_p50, t.latency_p99);
    ImGui::Text("Pose %.0f Hz   CPU %.2f ms/image", t.pose_hz, t.cpu_ms);
    ImGui::Text("Tremblement %.2f° (p99 %.2f°)", t.shake_p50, t.shake_p99);
    ImGui::Text("Flottement (erreur de prédiction p95) : %.2f° immobile · %.2f° lent · %.2f° rapide", t.pred_err_still, t.pred_err_moving, t.pred_err_fast);
    ImGui::TextDisabled("Prédiction utilisée : %.0f %% du mouvement", t.lead_used * 100);
    ImGui::Text("Corrections visuelles  %d en 2 s · %.2f° · %.1f mm", t.corrections, t.corr_deg, t.corr_mm);
    ImGui::Checkbox("Lisser les corrections (sinon elles sautent)", &s.smooth_corrections);
    plot("##shake", tail(s.shake_history, 300), 0, 1.0f, 2.5f * em, "tremblement par image (0–1°)");
    plot("##lat", tail(s.latency_history, 300), 0, 40, 2.5f * em, "pose → écran (0–40 ms)");
    ImGui::SeparatorText("Prédiction");
    ImGui::Checkbox("Rendu au dernier moment avant l'affichage", &s.late_latch);
    ImGui::TextDisabled("pose lue %.1f ms avant le vblank (marge auto) · prédiction totale %.1f ms", s.late_latch ? s.latch_margin_ms : 16.6f, s.stats.lead_ms);
    ImGui::SliderFloat("latence de l'écran (ms)", &s.predict_ms, 0, 40, "%.0f");
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
    if (s.have_vendor_eyes) {
        const float* l = s.eye_l; const float* r = s.eye_r;
        char lbl[200];
        snprintf(lbl, sizeof lbl, "écart %.1f mm, %.1f cm plus bas, %.1f cm en arrière, %.1f° d'inclinaison",
                 std::hypot(r[3] - l[3], r[7] - l[7]) * 1000, -(l[7] + r[7]) / 2 * 100, (l[11] + r[11]) / 2 * 100, std::atan2(l[9], l[5]) * 57.2958f);
        ImGui::Checkbox("Yeux du moteur VITURE (usine)", &s.vendor_eyes);
        ImGui::TextDisabled("%s", lbl);
        if (s.vendor_eyes) ImGui::TextDisabled("écart des yeux et yeux par défaut remplacés ; les curseurs ci-dessous ajustent par-dessus");
    }
    if (!(s.vendor_eyes && s.have_vendor_eyes)) {
        rgbcal::Cam rc = rgbcal::get();
        char lbl[96];
        snprintf(lbl, sizeof lbl, rc.fitted ? "Orientation de l'écran = caméra RGB calée (%.1f°)" : "Orientation de l'écran = caméra RGB calée (pas encore calée)", rc.fitted ? rc.rv[0] * 57.2958 : 0.0);
        ImGui::Checkbox(lbl, &s.display_from_rgb);
    }
    ImGui::SliderFloat("inclinaison (°)", &s.pitch_deg, -20, 20, "%.1f");
    ImGui::SliderFloat("rotation gauche/droite (°)", &s.yaw_deg, -10, 10, "%.1f");
    ImGui::SliderFloat("hauteur des yeux (cm)", &s.eye_dy_cm, -10, 10, "%.1f");
    ImGui::SliderFloat("recul des yeux (cm)", &s.eye_dz_cm, -5, 10, "%.1f");
    ImGui::Checkbox("Calage sur la pièce (points du scan)", &s.calage);
    if (s.calage) {
        ImGui::TextWrapped("Regarde des objets nets (porte, meuble, écran) : les points doivent tomber dessus. Tourne lentement la tête :");
        ImGui::BulletText("les points suivent la tête (glissent dans son sens) : champ vertical trop grand, diminue");
        ImGui::BulletText("ils glissent à l'inverse de la tête : champ trop petit, augmente");
        ImGui::BulletText("décalage fixe vers le haut/bas : inclinaison ; gauche/droite : rotation");
        ImGui::BulletText("en avançant/reculant la tête, ça bouge : recul des yeux");
        ImGui::TextDisabled("Puis, tête tournée vite : s'ils traînent, monte « latence de l'écran ».");
    }
    ImGui::Checkbox("Panneau de test", &s.panel);
    ImGui::SameLine();
    ImGui::Checkbox("Scène de test (cubes)", &s.scene);
    ImGui::SameLine();
    ImGui::Checkbox("Grille au sol", &s.floor_grid);
    if (ImGui::Button("Valeurs d'usine")) { s.vfov = 28.147f; s.ipd_mm = 63; s.floor_m = 1.6f; s.pitch_deg = 0; s.eye_dy_cm = 0; s.eye_dz_cm = 0; s.yaw_deg = 0; }
    ImGui::SeparatorText("Pièce (scan)");
    {
        std::lock_guard<std::mutex> l(room::g.mu);
        auto& r = room::g;
        if (!r.status.empty()) ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "%s", r.status.c_str());
        ImGui::Text("%zu voxels · profondeur %.0f/s · %.0f %% mesurés · %.0f ms", r.voxels, r.rate, r.valid_ratio * 100, r.ms);
        int tables = 0, walls = 0; float floor_y = NAN;
        for (auto& p : r.planes) { if (p.kind == 0) floor_y = p.y; else if (p.kind == 1) tables++; else walls++; }
        if (!std::isnan(floor_y)) ImGui::Text("Sol à %.2f m sous l'origine · %d surface(s) horizontale(s) · %d mur(s) · maillage %zu triangles", -floor_y, tables, walls, r.mesh.size() / 18);
        // Depth under the gaze (median of the centre 20x20), to check against a tape measure.
        if (r.depth.size() == (size_t)depth::W * depth::H) {
            std::vector<float> c;
            for (int y = depth::H / 2 - 10; y < depth::H / 2 + 10; y++) for (int x = depth::W / 2 - 10; x < depth::W / 2 + 10; x++) { float z = r.depth[y * depth::W + x]; if (z > 0) c.push_back(z); }
            if (c.size() > 20) { std::sort(c.begin(), c.end()); ImGui::Text("Distance au centre du regard : %.2f m (caméra gauche)", c[c.size() / 2]); }
            else ImGui::TextDisabled("Distance au centre du regard : pas mesurée");
        }
        else ImGui::TextDisabled("Sol pas encore trouvé : regarde le sol et autour de toi");
        ImGui::Text("%zu ancre(s)", r.anchors.size());
    }
    ImGui::Checkbox("Scanner", &s.scan);
    ImGui::SameLine();
    ImGui::Checkbox("Afficher le maillage", &s.show_map);
    ImGui::SameLine();
    ImGui::Checkbox("Occultation par la pièce", &s.occlusion);
    ImGui::SameLine();
    ImGui::Checkbox("Objets posés sur les surfaces", &s.place_on_planes);
    if (ImGui::Button("Ancrer là où je regarde (A)")) s.room_request = 1;
    ImGui::SameLine();
    if (ImGui::Button("Oublier la pièce")) s.room_request = 2;
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
    ImGui::BeginChild("handinfo", ImVec2(0, 4 * line + sty.FramePadding.y * 2), ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);
    ImGui::Checkbox("Mains (VITURE)", &s.hands);
    hands::g_enabled = s.hands;
    ImGui::SameLine();
    {
        std::string st;
        { std::lock_guard<std::mutex> lk(vhands::g_mu); st = vhands::g_out.status; }
        if (!st.empty()) ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "%s", st.c_str());
        else if (!hr.status.empty()) ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "%s", hr.status.c_str());
        else ImGui::TextDisabled("%.0f réponses/s · calage RGB %d/s (%.0f ms)", hr.rate, hr.rgb_runs, hr.rgb_ms);
    }
    for (int side = 0; side < 2; side++) {
        const hands::Hand* h = nullptr;
        for (auto& x : hr.hands) if (x.right == (side == 1)) h = &x;
        if (h) ImGui::TextDisabled("%s : %s · pince %.1f cm · doigts %.1f %.1f %.1f %.1f %.1f · %.2f m/s%s", side ? "droite" : "gauche",
                                   hands::GESTURES[h->gesture], h->pinch_m * 100, h->ext[0], h->ext[1], h->ext[2], h->ext[3], h->ext[4], h->speed, h->rgb_seen ? " · vue en RGB" : "");
        else ImGui::TextDisabled("%s : —", side ? "droite" : "gauche");
    }
    ImGui::EndChild();
    update_depth();
    float avail_w = ImGui::GetContentRegionAvail().x / 3 - ImGui::GetStyle().ItemSpacing.x, avail_h = ImGui::GetContentRegionAvail().y;
    for (int i = 0; i < 3; i++) {
        if (i) ImGui::SameLine();
        if (i == 2) {
            if (depth_tex.srv) { float kd = std::min(avail_w / depth_tex.w, avail_h / depth_tex.h); ImGui::Image((ImTextureID)(intptr_t)depth_tex.srv, ImVec2(depth_tex.w * kd, depth_tex.h * kd)); }
            continue;
        }
        if (!cams[i].srv) { ImGui::Dummy(ImVec2(avail_w, avail_h)); continue; }
        float k = std::min(avail_w / cams[i].w, (ImGui::GetContentRegionAvail().y) / cams[i].h);
        ImVec2 o = ImGui::GetCursorScreenPos();
        ImGui::Image((ImTextureID)(intptr_t)cams[i].srv, ImVec2(cams[i].w * k, cams[i].h * k));
        ImDrawList* dl = ImGui::GetWindowDrawList();
        for (auto& h : hr.hands) {
            if (!h.in_image[i]) continue;
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
        { std::lock_guard<std::mutex> l(rgbcal::g_mu); fi = rgbcal::g_fit; fitted = rgbcal::g_cam.fitted; np = (size_t)rgbcal::g_frames; }
        double rms; int npts; fit_numbers(rms, npts);
        if (fitted && npts) ImGui::TextDisabled("calage RGB : écart %.1f px sur %d points", rms, npts);
        else if (fitted) ImGui::TextDisabled("calage RGB : enregistré");
        else ImGui::TextDisabled("calage RGB : caméra par défaut · %zu/8 poses de main (immobile, à des endroits différents)%s", np, fi.n && !fi.ok ? " · dernier essai refusé" : "");
    }
    if (rgb.srv) {
        ImVec2 a = ImGui::GetContentRegionAvail();
        float k = std::min(a.x / rgb.w, a.y / rgb.h);
        ImVec2 o = ImGui::GetCursorScreenPos();
        // The composed view (video + virtual) when the render thread made one.
        ImGui::Image((ImTextureID)(intptr_t)(s.mr_srv ? s.mr_srv : (void*)rgb.srv), ImVec2(rgb.w * k, rgb.h * k));
        if (s.rgb_overlay && s.mr_srv) {
            // Until the camera is fitted: where the hand model sees the hand in the RGB image.
            bool fitted; { std::lock_guard<std::mutex> l(rgbcal::g_mu); fitted = rgbcal::g_cam.fitted; }
            if (!fitted) {
                ImDrawList* dl = ImGui::GetWindowDrawList();
                for (auto& h : hr.hands) if (h.rgb_seen) for (int j2 = 0; j2 < hands::N_JOINTS; j2++) dl->AddCircleFilled(ImVec2(o.x + h.rgb[j2][0] * k, o.y + h.rgb[j2][1] * k), 3, IM_COL32(80, 255, 120, 220));
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

    ImGui::EndTabItem();
    ImGui::EndTabBar();
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
        bool ok = save_png(png);
        char path[MAX_PATH]; WideCharToMultiByte(CP_UTF8, 0, png.c_str(), -1, path, sizeof path, nullptr, nullptr);
        s.notice = ok ? std::string("capture enregistrée : ") + path : "capture : échec";
        s.notice_until = ImGui::GetTime() + 6;
        if (ok) ShellExecuteA(nullptr, "open", "explorer.exe", ("/select,\"" + std::string(path) + "\"").c_str(), nullptr, SW_SHOWNORMAL);
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
