// The virtual human: the Rocketbox male (MIT, Microsoft) of the web app, baked for the
// native app by tools/avatar/bake_native.py (models/human.bin and its three textures).
// Skinned on the GPU (80 bones); its idle clip loops, and a procedural walk is laid over
// it in the model's frame (thighs, knees, feet, arms, hips: angles from the gait phase),
// with the stride matched to the speed so the feet do not slide.
//
// It walks on the room's walkable grid (roommap.hpp): A* over the 10 cm cells, the path
// shortened where the straight line stays walkable, turning towards where it goes; idle,
// it faces the wearer and, when wandering, picks a new place 1 to 3.5 m from them now and
// then. Positions are in the tracking frame, on the floor.

#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wincodec.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include "roommap.hpp"

#pragma comment(lib, "windowscodecs.lib")

namespace human {

// ---- small row-major 4x4 math (column vectors, like the app's M4)
struct Mat { float m[16]; float& operator()(int r, int c) { return m[r * 4 + c]; } float operator()(int r, int c) const { return m[r * 4 + c]; } };
inline Mat ident() { Mat a{}; a.m[0] = a.m[5] = a.m[10] = a.m[15] = 1; return a; }
inline Mat mul(const Mat& a, const Mat& b) {
    Mat r{};
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { float s = 0; for (int k = 0; k < 4; k++) s += a(i, k) * b(k, j); r(i, j) = s; }
    return r;
}
/** Inverse of an affine matrix (rotation x uniform scale + translation). */
inline Mat affine_inv(const Mat& a) {
    float s2 = a(0, 0) * a(0, 0) + a(1, 0) * a(1, 0) + a(2, 0) * a(2, 0);
    Mat r = ident();
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) r(i, j) = a(j, i) / s2;
    for (int i = 0; i < 3; i++) r(i, 3) = -(r(i, 0) * a(0, 3) + r(i, 1) * a(1, 3) + r(i, 2) * a(2, 3));
    return r;
}
struct TRS { float t[3], r[4], s[3]; };
inline Mat trs(const TRS& v) {
    float x = v.r[0], y = v.r[1], z = v.r[2], w = v.r[3];
    Mat M = ident();
    float R[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
                     {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
                     {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) M(i, j) = R[i][j] * v.s[j];
    for (int i = 0; i < 3; i++) M(i, 3) = v.t[i];
    return M;
}
inline Mat rot_axis(float ax, float ay, float az, float a) {
    float c = std::cos(a), s = std::sin(a), t = 1 - c;
    Mat M = ident();
    M(0, 0) = t * ax * ax + c;      M(0, 1) = t * ax * ay - s * az; M(0, 2) = t * ax * az + s * ay;
    M(1, 0) = t * ax * ay + s * az; M(1, 1) = t * ay * ay + c;      M(1, 2) = t * ay * az - s * ax;
    M(2, 0) = t * ax * az - s * ay; M(2, 1) = t * ay * az + s * ax; M(2, 2) = t * az * az + c;
    return M;
}
inline void slerp(const float* a, const float* b, float u, float* out) {
    float d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3], sg = 1;
    if (d < 0) { d = -d; sg = -1; }
    float wa, wb;
    if (d > 0.9995f) { wa = 1 - u; wb = u; }
    else { float th = std::acos(d), sn = std::sin(th); wa = std::sin((1 - u) * th) / sn; wb = std::sin(u * th) / sn; }
    float n = 0;
    for (int i = 0; i < 4; i++) { out[i] = wa * a[i] + wb * sg * b[i]; n += out[i] * out[i]; }
    n = std::sqrt(n);
    for (int i = 0; i < 4; i++) out[i] /= n;
}

struct SkinV { float p[3], n[3], uv[2]; uint8_t j[4]; float w[4]; };
struct Part { std::vector<SkinV> v; std::vector<uint32_t> idx; uint32_t alpha = 0; std::string tex; ID3D11Buffer* vb = nullptr; ID3D11Buffer* ib = nullptr; ID3D11ShaderResourceView* srv = nullptr; };
constexpr int MAX_BONES = 96;

struct Model {
    std::vector<int> parent;
    std::vector<TRS> rest;
    std::vector<std::string> names;
    std::vector<int> order;          // parents before children
    std::vector<uint32_t> joint_node;
    std::vector<Mat> ibm;
    Mat mesh_inv = ident();
    std::vector<Part> parts;
    float fps = 30;
    uint32_t frames = 0;
    std::vector<TRS> clip;           // frames x nodes
    float fwd[3] = {0, 0, 1}, foot_y = 0;
    uint32_t walk_frames = 0;        // the walk cycle (version 2): frames x nodes
    float walk_fps = 30, walk_stride = 0;  // metres covered per cycle
    std::vector<TRS> walk;
    int find(const char* n) const { for (size_t i = 0; i < names.size(); i++) if (names[i] == n) return (int)i; return -1; }
};

inline bool load_model(const std::string& path, Model& m, std::string& err) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { err = "pas de " + path; return false; }
    std::vector<uint8_t> b;
    { fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); b.resize(n); fread(b.data(), 1, n, f); fclose(f); }
    size_t o = 0;
    auto rd = [&](void* dst, size_t n) { if (o + n > b.size()) throw 1; memcpy(dst, &b[o], n); o += n; };
    auto u32 = [&]() { uint32_t v; rd(&v, 4); return v; };
    try {
        char magic[4]; rd(magic, 4);
        if (memcmp(magic, "MVHU", 4)) { err = "human.bin : format inconnu"; return false; }
        uint32_t version = u32();
        if (version < 1 || version > 2) { err = "human.bin : version inconnue"; return false; }
        uint32_t N = u32();
        m.parent.resize(N); m.rest.resize(N); m.names.resize(N);
        for (uint32_t i = 0; i < N; i++) {
            int32_t p; rd(&p, 4); m.parent[i] = p;
            rd(m.rest[i].t, 12); rd(m.rest[i].r, 16); rd(m.rest[i].s, 12);
            char nm[33] = {}; rd(nm, 32); m.names[i] = nm;
        }
        uint32_t J = u32();
        m.joint_node.resize(J); m.ibm.resize(J);
        for (uint32_t j = 0; j < J; j++) { m.joint_node[j] = u32(); rd(m.ibm[j].m, 64); }
        rd(m.mesh_inv.m, 64);
        uint32_t P = u32();
        m.parts.resize(P);
        for (auto& part : m.parts) {
            uint32_t nv = u32(), ni = u32(); part.alpha = u32();
            char nm[33] = {}; rd(nm, 32); part.tex = nm;
            part.v.resize(nv); part.idx.resize(ni);
            for (auto& v : part.v) { rd(v.p, 12); rd(v.n, 12); rd(v.uv, 8); rd(v.j, 4); rd(v.w, 16); }
            rd(part.idx.data(), ni * 4);
        }
        rd(&m.fps, 4); m.frames = u32();
        m.clip.resize((size_t)m.frames * N);
        for (auto& t : m.clip) { rd(t.t, 12); rd(t.r, 16); rd(t.s, 12); }
        rd(m.fwd, 12); rd(&m.foot_y, 4);
        if (version >= 2) {
            rd(&m.walk_fps, 4); m.walk_frames = u32(); rd(&m.walk_stride, 4);
            m.walk.resize((size_t)m.walk_frames * N);
            for (auto& t : m.walk) { rd(t.t, 12); rd(t.r, 16); rd(t.s, 12); }
        }
    } catch (...) { err = "human.bin tronqué"; return false; }
    if (m.joint_node.size() > MAX_BONES) { err = "trop d'os"; return false; }
    // Parents before children.
    std::vector<int> state(m.parent.size(), 0);
    std::function<void(int)> visit = [&](int i) { if (state[i]) return; if (m.parent[i] >= 0) visit(m.parent[i]); state[i] = 1; m.order.push_back(i); };
    for (size_t i = 0; i < m.parent.size(); i++) visit((int)i);
    return true;
}

/** Texture from a JPEG or PNG file, with mipmaps. */
inline ID3D11ShaderResourceView* load_texture(ID3D11Device* dev, ID3D11DeviceContext* ctx, const std::wstring& path) {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);  // WIC needs COM on this thread (harmless if already)
    IWICImagingFactory* wic = nullptr;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic)))) return nullptr;
    IWICBitmapDecoder* dec = nullptr; IWICBitmapFrameDecode* fr = nullptr; IWICFormatConverter* cv = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    if (SUCCEEDED(wic->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec)) && SUCCEEDED(dec->GetFrame(0, &fr)) &&
        SUCCEEDED(wic->CreateFormatConverter(&cv)) && SUCCEEDED(cv->Initialize(fr, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeCustom))) {
        UINT w, h; cv->GetSize(&w, &h);
        std::vector<uint8_t> px((size_t)w * h * 4);
        if (SUCCEEDED(cv->CopyPixels(nullptr, w * 4, (UINT)px.size(), px.data()))) {
            D3D11_TEXTURE2D_DESC td{}; td.Width = w; td.Height = h; td.MipLevels = 0; td.ArraySize = 1; td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            td.SampleDesc.Count = 1; td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET; td.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
            ID3D11Texture2D* tex = nullptr;
            if (SUCCEEDED(dev->CreateTexture2D(&td, nullptr, &tex))) {
                ctx->UpdateSubresource(tex, 0, nullptr, px.data(), w * 4, 0);
                if (SUCCEEDED(dev->CreateShaderResourceView(tex, nullptr, &srv))) ctx->GenerateMips(srv);
                tex->Release();
            }
        }
    }
    if (cv) cv->Release(); if (fr) fr->Release(); if (dec) dec->Release(); wic->Release();
    return srv;
}

static const char* HSHADER = R"(
cbuffer H : register(b0) { row_major float4x4 vp; row_major float4x4 model; float4 tint; float4 opt; row_major float4x4 bones[96]; };
Texture2D albedo : register(t0);
SamplerState samp0 : register(s0);
struct VIn { float3 p : POSITION; float3 n : NORMAL; float2 uv : TEXCOORD; uint4 j : BLENDINDICES; float4 w : BLENDWEIGHT; };
struct VOut { float4 pos : SV_Position; float3 n : NORMAL; float2 uv : TEXCOORD; };
VOut vs(VIn i) {
  float4x4 s = bones[i.j.x] * i.w.x + bones[i.j.y] * i.w.y + bones[i.j.z] * i.w.z + bones[i.j.w] * i.w.w;
  float4 wp = mul(model, mul(s, float4(i.p, 1)));
  VOut o; o.pos = mul(vp, wp); o.n = mul((float3x3)model, mul((float3x3)s, i.n)); o.uv = i.uv; return o;
}
float4 ps(VOut i) : SV_Target {
  float4 a = albedo.Sample(samp0, i.uv);
  if (opt.x > 0.5) clip(a.a - 0.5);
  float3 n = normalize(i.n);
  // Soft light from above and in front, a little fill from below: no black side (black is
  // see-through on the glasses).
  float k = 0.55 + 0.45 * saturate(dot(n, normalize(float3(0.3, 1.0, 0.5)))) + 0.1 * saturate(-n.y);
  return float4(a.rgb * k * tint.rgb, 1);
}
)";

struct Renderer {
    Model model;
    bool ok = false;
    std::string error;
    ID3D11VertexShader* vs = nullptr;
    ID3D11PixelShader* ps = nullptr;
    ID3D11InputLayout* layout = nullptr;
    ID3D11Buffer* cb = nullptr;
    ID3D11RasterizerState* rs = nullptr;
    ID3D11SamplerState* samp = nullptr;
    struct CB { Mat vp, model; float tint[4], opt[4]; Mat bones[MAX_BONES]; };
    CB cbuf{};
    // Indices of the bones the walk moves.
    int thigh[2], calf[2], foot[2], uarm[2], farm[2], hand[2], pelvis, spine;

    bool init(ID3D11Device* dev, ID3D11DeviceContext* ctx, const std::wstring& dir) {
        std::string d8(dir.begin(), dir.end());
        if (!load_model(d8 + "models\\human.bin", model, error)) return false;
        ID3DBlob *vb = nullptr, *pb = nullptr, *err = nullptr;
        if (FAILED(D3DCompile(HSHADER, strlen(HSHADER), nullptr, nullptr, nullptr, "vs", "vs_5_0", 0, 0, &vb, &err)) ||
            FAILED(D3DCompile(HSHADER, strlen(HSHADER), nullptr, nullptr, nullptr, "ps", "ps_5_0", 0, 0, &pb, &err))) {
            error = err ? (const char*)err->GetBufferPointer() : "shader"; return false;
        }
        dev->CreateVertexShader(vb->GetBufferPointer(), vb->GetBufferSize(), nullptr, &vs);
        dev->CreatePixelShader(pb->GetBufferPointer(), pb->GetBufferSize(), nullptr, &ps);
        D3D11_INPUT_ELEMENT_DESC ie[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"BLENDINDICES", 0, DXGI_FORMAT_R8G8B8A8_UINT, 0, 32, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"BLENDWEIGHT", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 36, D3D11_INPUT_PER_VERTEX_DATA, 0},
        };
        dev->CreateInputLayout(ie, 5, vb->GetBufferPointer(), vb->GetBufferSize(), &layout);
        vb->Release(); pb->Release();
        D3D11_BUFFER_DESC bd{}; bd.ByteWidth = sizeof(CB); bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; bd.Usage = D3D11_USAGE_DEFAULT;
        dev->CreateBuffer(&bd, nullptr, &cb);
        D3D11_RASTERIZER_DESC rd{}; rd.FillMode = D3D11_FILL_SOLID; rd.CullMode = D3D11_CULL_NONE; rd.DepthClipEnable = TRUE;
        dev->CreateRasterizerState(&rd, &rs);
        D3D11_SAMPLER_DESC sd{}; sd.Filter = D3D11_FILTER_ANISOTROPIC; sd.MaxAnisotropy = 8; sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_WRAP; sd.MaxLOD = D3D11_FLOAT32_MAX;
        dev->CreateSamplerState(&sd, &samp);
        for (auto& p : model.parts) {
            D3D11_BUFFER_DESC b{}; b.ByteWidth = (UINT)(p.v.size() * sizeof(SkinV)); b.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            D3D11_SUBRESOURCE_DATA init{p.v.data()};
            dev->CreateBuffer(&b, &init, &p.vb);
            b.ByteWidth = (UINT)(p.idx.size() * 4); b.BindFlags = D3D11_BIND_INDEX_BUFFER; init.pSysMem = p.idx.data();
            dev->CreateBuffer(&b, &init, &p.ib);
            p.srv = load_texture(dev, ctx, dir + L"models\\" + std::wstring(p.tex.begin(), p.tex.end()));
        }
        const char* side[2] = {"L", "R"};
        for (int s = 0; s < 2; s++) {
            char n[64];
            snprintf(n, sizeof n, "Bip01 %s Thigh", side[s]); thigh[s] = model.find(n);
            snprintf(n, sizeof n, "Bip01 %s Calf", side[s]); calf[s] = model.find(n);
            snprintf(n, sizeof n, "Bip01 %s Foot", side[s]); foot[s] = model.find(n);
            snprintf(n, sizeof n, "Bip01 %s UpperArm", side[s]); uarm[s] = model.find(n);
            snprintf(n, sizeof n, "Bip01 %s Forearm", side[s]); farm[s] = model.find(n);
            snprintf(n, sizeof n, "Bip01 %s Hand", side[s]); hand[s] = model.find(n);
        }
        pelvis = model.find("Bip01 Pelvis"); spine = model.find("Bip01 Spine");
        ok = vs && ps && layout && cb;
        return ok;
    }

    /**
     * Bone matrices for the idle clip at time t, with the walk laid over it: gait phase
     * (radians, one full cycle = two steps) and how much of the walk (0..1). aim: a point in
     * the model's frame (m) the right arm points at, by aim_w (0..1).
     */
    void pose(double t, double phase, float walk, Mat out_nodes_global[], const float* aim = nullptr, float aim_w = 0) {
        size_t N = model.parent.size();
        std::vector<TRS> local(N);
        double ft = std::fmod(t * model.fps, (double)std::max(1u, model.frames));
        uint32_t f0 = (uint32_t)ft, f1 = (f0 + 1) % std::max(1u, model.frames);
        float u = (float)(ft - f0);
        for (size_t i = 0; i < N; i++) {
            if (!model.frames) { local[i] = model.rest[i]; continue; }
            const TRS& a = model.clip[f0 * N + i];
            const TRS& b = model.clip[f1 * N + i];
            for (int k = 0; k < 3; k++) { local[i].t[k] = a.t[k] + u * (b.t[k] - a.t[k]); local[i].s[k] = a.s[k] + u * (b.s[k] - a.s[k]); }
            slerp(a.r, b.r, u, local[i].r);
        }
        // The Rocketbox walk cycle (in place), at the gait phase, blended in by how much it walks.
        if (walk > 0 && model.walk_frames) {
            double wf = std::fmod(phase / 6.2831853, 1.0); if (wf < 0) wf += 1;
            wf *= model.walk_frames;
            uint32_t w0 = (uint32_t)wf % model.walk_frames, w1 = (w0 + 1) % model.walk_frames;
            float wu = (float)(wf - std::floor(wf)), k = std::min(1.0f, walk);
            for (size_t i = 0; i < N; i++) {
                const TRS& a = model.walk[w0 * N + i];
                const TRS& b = model.walk[w1 * N + i];
                TRS w;
                for (int q = 0; q < 3; q++) { w.t[q] = a.t[q] + wu * (b.t[q] - a.t[q]); w.s[q] = a.s[q] + wu * (b.s[q] - a.s[q]); }
                slerp(a.r, b.r, wu, w.r);
                for (int q = 0; q < 3; q++) { local[i].t[q] += k * (w.t[q] - local[i].t[q]); local[i].s[q] += k * (w.s[q] - local[i].s[q]); }
                float r[4]; slerp(local[i].r, w.r, k, r); memcpy(local[i].r, r, sizeof r);
            }
        }
        // Without the clip: a procedural walk, as rotations in the model's frame (x: the
        // character's left, y up, z forward) about each joint; a leg swings forward with a
        // negative angle about x.
        std::vector<float> dx(N, 0.0f), dy(N, 0.0f);
        if (walk > 0 && !model.walk_frames) {
            const float D = 3.14159265f / 180;
            for (int s = 0; s < 2; s++) {
                double ph = phase + s * 3.14159265;
                float swing = (float)(20 * std::sin(ph)), knee = (float)(4 + 42 * std::pow(std::max(0.0, std::cos(ph)), 1.5));
                float toe = std::sin(ph) < -0.6 ? (float)(14 * (-std::sin(ph) - 0.6) / 0.4) : 0.0f;
                if (thigh[s] >= 0) dx[thigh[s]] = -swing * D * walk;
                if (calf[s] >= 0) dx[calf[s]] = knee * D * walk;
                if (foot[s] >= 0) dx[foot[s]] = (swing - knee + toe) * D * walk;
                // The arm swings with the other leg.
                if (uarm[s] >= 0) dx[uarm[s]] = (float)(14 * std::sin(ph)) * D * walk;
                if (farm[s] >= 0) dx[farm[s]] = -10 * D * walk;
            }
            if (pelvis >= 0) dy[pelvis] = (float)(5 * std::sin(phase)) * D * walk;
            if (spine >= 0) dy[spine] = (float)(-7 * std::sin(phase)) * D * walk;
        }
        for (int i : model.order) {
            Mat L = trs(local[i]);
            const Mat P = model.parent[i] >= 0 ? out_nodes_global[model.parent[i]] : ident();
            if (dx[i] != 0 || dy[i] != 0) {
                // L' = P^-1 * (T(p) R T(-p)) * P * L, p the joint's position in the model.
                Mat G = mul(P, L);
                float px = G(0, 3), py = G(1, 3), pz = G(2, 3);
                Mat R = mul(rot_axis(0, 1, 0, dy[i]), rot_axis(1, 0, 0, dx[i]));
                Mat T1 = ident(), T2 = ident();
                T1(0, 3) = px; T1(1, 3) = py; T1(2, 3) = pz; T2(0, 3) = -px; T2(1, 3) = -py; T2(2, 3) = -pz;
                L = mul(affine_inv(P), mul(mul(T1, mul(R, T2)), G));
            }
            out_nodes_global[i] = mul(P, L);
            // Pointing: the upper arm, then the forearm, turned so that each one's bone (to its
            // child joint) heads for the point: the whole arm straight towards it.
            if (aim && aim_w > 0 && (i == uarm[1] || i == farm[1])) {
                int child = i == uarm[1] ? farm[1] : hand[1];
                if (child < 0) continue;
                Mat& G = out_nodes_global[i];
                const float* ct = model.rest[child].t;
                float a[3], d[3] = {aim[0] - G(0, 3), aim[1] - G(1, 3), aim[2] - G(2, 3)};
                for (int r = 0; r < 3; r++) a[r] = G(r, 0) * ct[0] + G(r, 1) * ct[1] + G(r, 2) * ct[2];
                float na = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]), nd = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                if (na < 1e-6f || nd < 1e-6f) continue;
                for (int r = 0; r < 3; r++) { a[r] /= na; d[r] /= nd; }
                float ax[3] = {a[1] * d[2] - a[2] * d[1], a[2] * d[0] - a[0] * d[2], a[0] * d[1] - a[1] * d[0]};
                float sn = std::sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]), cs = a[0] * d[0] + a[1] * d[1] + a[2] * d[2];
                if (sn < 1e-5f) continue;
                Mat R = rot_axis(ax[0] / sn, ax[1] / sn, ax[2] / sn, std::atan2(sn, cs) * aim_w);
                Mat T1 = ident(), T2 = ident();
                T1(0, 3) = G(0, 3); T1(1, 3) = G(1, 3); T1(2, 3) = G(2, 3); T2(0, 3) = -G(0, 3); T2(1, 3) = -G(1, 3); T2(2, 3) = -G(2, 3);
                G = mul(T1, mul(R, mul(T2, G)));
            }
        }
    }

    /** Draws the human: vp the eye's view-projection, world the model -> scene placement. */
    void draw(ID3D11DeviceContext* ctx, const float vp[16], const float world[16], const float tint[3], double t, double phase, float walk, const float* aim = nullptr, float aim_w = 0) {
        if (!ok) return;
        std::vector<Mat> G(model.parent.size());
        pose(t, phase, walk, G.data(), aim, aim_w);
        memcpy(cbuf.vp.m, vp, 64); memcpy(cbuf.model.m, world, 64);
        cbuf.tint[0] = tint[0]; cbuf.tint[1] = tint[1]; cbuf.tint[2] = tint[2]; cbuf.tint[3] = 1;
        // glTF skinning: joint global x inverse bind. The mesh node's own transform is not
        // applied (here it sits under Bip01, scaled 0.01: undoing it made the body 100 times too big).
        for (size_t j = 0; j < model.joint_node.size(); j++) cbuf.bones[j] = mul(G[model.joint_node[j]], model.ibm[j]);
        ID3D11RasterizerState* old_rs = nullptr; ctx->RSGetState(&old_rs);
        ID3D11InputLayout* old_il = nullptr; ctx->IAGetInputLayout(&old_il);
        ctx->RSSetState(rs);
        ctx->IASetInputLayout(layout);
        ctx->VSSetShader(vs, nullptr, 0);
        ctx->PSSetShader(ps, nullptr, 0);
        ctx->VSSetConstantBuffers(0, 1, &cb);
        ctx->PSSetConstantBuffers(0, 1, &cb);
        ctx->PSSetSamplers(0, 1, &samp);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        for (auto& p : model.parts) {
            cbuf.opt[0] = p.alpha ? 1.0f : 0.0f;
            ctx->UpdateSubresource(cb, 0, nullptr, &cbuf, 0, 0);
            UINT stride = sizeof(SkinV), off = 0;
            ctx->IASetVertexBuffers(0, 1, &p.vb, &stride, &off);
            ctx->IASetIndexBuffer(p.ib, DXGI_FORMAT_R32_UINT, 0);
            ctx->PSSetShaderResources(0, 1, &p.srv);
            ctx->DrawIndexed((UINT)p.idx.size(), 0, 0);
        }
        ID3D11ShaderResourceView* none = nullptr;
        ctx->PSSetShaderResources(0, 1, &none);
        ctx->RSSetState(old_rs); if (old_rs) old_rs->Release();
        ctx->IASetInputLayout(old_il); if (old_il) old_il->Release();
    }
};

// ---- where it walks

/** A* over the walkable cells (8 neighbours), then shortened by straight lines. Empty if none. */
inline std::vector<std::array<float, 2>> find_path(const room::WalkGrid& g, float sx, float sz, float tx, float tz, int snap_cells = 15) {
    std::vector<std::array<float, 2>> out;
    if (!g.nx || !g.nz) return out;
    auto idx = [&](float x, float z, int& i, int& j) { i = (int)std::floor((x - g.x0) / g.cell); j = (int)std::floor((z - g.z0) / g.cell); return i >= 0 && j >= 0 && i < g.nx && j < g.nz; };
    auto free_at = [&](int i, int j) { return i >= 0 && j >= 0 && i < g.nx && j < g.nz && g.c[j * g.nx + i] == 1; };
    // The nearest walkable cell to a point (within snap_cells, 1.5 m by default).
    auto snap = [&](float x, float z, int& i, int& j) {
        idx(x, z, i, j);
        if (free_at(i, j)) return true;
        int ci = i, cj = j; float best = 1e9f;
        for (int r = 1; r <= snap_cells && best >= 1e9f; r++) for (int dj = -r; dj <= r; dj++) for (int di = -r; di <= r; di++) {
            if (std::max(std::abs(di), std::abs(dj)) != r || !free_at(ci + di, cj + dj)) continue;
            float d = (float)(di * di + dj * dj);
            if (d < best) { best = d; i = ci + di; j = cj + dj; }
        }
        return best < 1e9f;
    };
    int si, sj, ti, tj;
    if (!snap(sx, sz, si, sj) || !snap(tx, tz, ti, tj)) return out;
    size_t n = (size_t)g.nx * g.nz;
    std::vector<float> cost(n, 1e30f);
    std::vector<int> from(n, -1);
    using E = std::pair<float, int>;
    std::priority_queue<E, std::vector<E>, std::greater<E>> open;
    auto h = [&](int i, int j) { return std::sqrt((float)((i - ti) * (i - ti) + (j - tj) * (j - tj))); };
    int s = sj * g.nx + si, t = tj * g.nx + ti;
    cost[s] = 0; open.push({h(si, sj), s});
    while (!open.empty()) {
        auto [f, c] = open.top(); open.pop();
        if (c == t) break;
        int i = c % g.nx, j = c / g.nx;
        if (f - h(i, j) > cost[c] + 1e-4f) continue;
        for (int dj = -1; dj <= 1; dj++) for (int di = -1; di <= 1; di++) {
            if ((!di && !dj) || !free_at(i + di, j + dj)) continue;
            if (di && dj && (!free_at(i + di, j) || !free_at(i, j + dj))) continue;  // no corner cutting
            int nc = (j + dj) * g.nx + i + di;
            float nc_cost = cost[c] + ((di && dj) ? 1.4142f : 1.0f);
            if (nc_cost < cost[nc]) { cost[nc] = nc_cost; from[nc] = c; open.push({nc_cost + h(i + di, j + dj), nc}); }
        }
    }
    if (from[t] < 0 && t != s) return out;
    std::vector<int> cells;
    for (int c = t; c != -1; c = from[c]) { cells.push_back(c); if (c == s) break; }
    std::reverse(cells.begin(), cells.end());
    // Straight lines where every cell they cross is walkable.
    auto line_free = [&](int a, int b) {
        float ax = a % g.nx + 0.5f, az = a / g.nx + 0.5f, bx = b % g.nx + 0.5f, bz = b / g.nx + 0.5f;
        int steps = (int)(std::max(std::fabs(bx - ax), std::fabs(bz - az)) * 3) + 1;
        for (int k = 0; k <= steps; k++) {
            float u = (float)k / steps;
            if (!free_at((int)std::floor(ax + u * (bx - ax)), (int)std::floor(az + u * (bz - az)))) return false;
        }
        return true;
    };
    size_t a = 0;
    auto centre = [&](int c) { return std::array<float, 2>{g.x0 + (c % g.nx + 0.5f) * g.cell, g.z0 + (c / g.nx + 0.5f) * g.cell}; };
    out.push_back(centre(cells[0]));
    while (a + 1 < cells.size()) {
        size_t b = cells.size() - 1;
        while (b > a + 1 && !line_free(cells[a], cells[b])) b--;
        out.push_back(centre(cells[b]));
        a = b;
    }
    if (!out.empty() && t == tj * g.nx + ti) { int ci, cj; idx(tx, tz, ci, cj); if (free_at(ci, cj)) out.back() = {tx, tz}; }
    return out;
}

/** The human's place and what it is doing, in the tracking frame. */
struct Walker {
    bool placed = false;
    float x = 0, z = 0, heading = 0;     // heading: the model's forward (+z) turned by this about y
    std::vector<std::array<float, 2>> path;
    size_t next = 0;
    float speed = 0;                     // m/s, eased
    double phase = 0;
    double idle_since = 0, next_wander = 4;
    std::string state = "absent";
    std::mt19937 rng{1234};
    static constexpr float WALK_SPEED = 0.85f;  // about the walk clip's own pace
    float stride = 1.22f;  // one gait cycle (two steps), m: the walk clip's when there is one
    bool look = false; float look_x = 0, look_z = 0;  // idle: faces this place instead of the wearer

    bool go_to(const room::WalkGrid& g, float tx, float tz) {
        auto p = find_path(g, x, z, tx, tz);
        if (p.size() < 2) return false;
        path = p; next = 1; state = "marche";
        return true;
    }

    /** One frame: dt seconds; the wearer at (ux, uz); wander picks new places when idle. */
    void update(double now, float dt, const room::WalkGrid& g, float ux, float uz, bool wander) {
        if (!placed) return;
        float target_speed = 0;
        if (next < path.size()) {
            float dx = path[next][0] - x, dz = path[next][1] - z, d = std::sqrt(dx * dx + dz * dz);
            if (d < 0.08f) { next++; }
            else {
                float want = std::atan2(dx, dz);
                float diff = std::remainder(want - heading, 6.2831853f);
                float turn = std::clamp(diff, -3.5f * dt, 3.5f * dt);
                heading += turn;
                // Slow down to turn on the spot, and at the end.
                bool last = next + 1 >= path.size();
                target_speed = WALK_SPEED * std::max(0.0f, std::cos(std::min(1.5f, std::fabs(diff)))) * (last ? std::min(1.0f, d / 0.5f + 0.3f) : 1.0f);
            }
            if (next >= path.size()) { path.clear(); next = 0; idle_since = now; state = "attend"; std::uniform_real_distribution<double> w(6, 12); next_wander = now + w(rng); }
        } else {
            // Idle: face the wearer (or what it shows), slowly.
            float want = look ? std::atan2(look_x - x, look_z - z) : std::atan2(ux - x, uz - z);
            float diff = std::remainder(want - heading, 6.2831853f);
            if (std::fabs(diff) > 0.15f) heading += std::clamp(diff, -1.5f * dt, 1.5f * dt);
            if (wander && now > next_wander) {
                // A walkable cell 1 to 3.5 m from the wearer; else, with little floor known
                // there, any reachable one 1 to 4 m from where it stands.
                std::uniform_real_distribution<float> ang(0, 6.2831853f), rad(1.0f, 3.5f), rad2(1.0f, 4.0f);
                bool going = false;
                for (int k = 0; k < 30 && !going; k++) {
                    float a = ang(rng), r = rad(rng);
                    float tx = ux + std::sin(a) * r, tz = uz + std::cos(a) * r;
                    going = g.at(tx, tz) == 1 && go_to(g, tx, tz);
                }
                for (int k = 0; k < 40 && !going; k++) {
                    float a = ang(rng), r = rad2(rng);
                    float tx = x + std::sin(a) * r, tz = z + std::cos(a) * r;
                    going = g.at(tx, tz) == 1 && go_to(g, tx, tz);
                }
                if (!going) state = "attend (pas d'endroit où aller)";
                std::uniform_real_distribution<double> w(6, 12);
                next_wander = now + w(rng);
            }
        }
        speed += (target_speed - speed) * std::min(1.0f, dt * 4);
        x += std::sin(heading) * speed * dt;
        z += std::cos(heading) * speed * dt;
        phase += 6.2831853 * speed / stride * dt;
        if (speed < 0.02f && next >= path.size()) phase = 0;
    }
};

}  // namespace human
