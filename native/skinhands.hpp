// Realistic hands and forearms: the Rocketbox human's (tools/hands/extract_hands.py, MIT),
// skinned on the tracked joints and an elbow placed by the caller (two-bone IK from an
// estimated shoulder). Each of the 17 bones (Hand, 3 per finger, Forearm) is moved
// so that its bind-pose segment lands on the tracked one: the palm turns with the tracked
// palm frame (wrist, index and little knuckles) and is scaled to the tracked hand's size;
// each finger bone then turns the least it can onto its tracked direction and stretches
// to its tracked length. Textured with the body's skin, tinted by the room's light.

#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wincodec.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#pragma comment(lib, "windowscodecs.lib")

namespace skinhands {

struct SkinVertex { float p[3], n[3], uv[2]; uint8_t bone[4]; float w[4]; };

struct Mesh {
    std::vector<SkinVertex> v;
    std::vector<uint16_t> idx;
    std::vector<SkinVertex> sleeve;  // a tube around the forearm, on its bone (bind pose)
    ID3D11Buffer* sleeve_vb = nullptr;
    float bind[22][3];          // MediaPipe's landmarks, then the elbow, at bind (m)
    ID3D11Buffer* vb = nullptr;
    ID3D11Buffer* ib = nullptr;
};

// Bone b's segment: from point HEAD[b] to TAIL[b] (Hand: the palm frame instead); 21 is
// the elbow.
constexpr int BONES = 17;
inline const int HEAD[BONES] = {0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21};
inline const int TAIL[BONES] = {9, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 0};

struct V { double x, y, z; };
inline V sub(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V add(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V mulv(V a, double k) { return {a.x * k, a.y * k, a.z * k}; }
inline double dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V cross(V a, V b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double len(V a) { return std::sqrt(dot(a, a)); }
inline V unit(V a) { double l = len(a); return l > 1e-12 ? mulv(a, 1 / l) : V{0, 0, 0}; }
/** 3x3, row-major. */
struct M { double m[9]; };
inline M mm(const M& a, const M& b) { M r{}; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) r.m[i * 3 + j] += a.m[i * 3 + k] * b.m[k * 3 + j]; return r; }
inline V mv(const M& a, V v) { return {a.m[0] * v.x + a.m[1] * v.y + a.m[2] * v.z, a.m[3] * v.x + a.m[4] * v.y + a.m[5] * v.z, a.m[6] * v.x + a.m[7] * v.y + a.m[8] * v.z}; }
inline M tr(const M& a) { M r; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) r.m[i * 3 + j] = a.m[j * 3 + i]; return r; }
/** Columns a, b, c. */
inline M cols(V a, V b, V c) { return {{a.x, b.x, c.x, a.y, b.y, c.y, a.z, b.z, c.z}}; }
/** The smallest rotation taking unit a onto unit b. */
inline M between(V a, V b) {
    V v = cross(a, b); double c = dot(a, b);
    if (c < -0.9999) {  // opposite: half a turn about any perpendicular
        V p = unit(std::fabs(a.x) < 0.9 ? cross(a, V{1, 0, 0}) : cross(a, V{0, 1, 0}));
        return {{2 * p.x * p.x - 1, 2 * p.x * p.y, 2 * p.x * p.z, 2 * p.y * p.x, 2 * p.y * p.y - 1, 2 * p.y * p.z, 2 * p.z * p.x, 2 * p.z * p.y, 2 * p.z * p.z - 1}};
    }
    double k = 1 / (1 + c);
    return {{v.x * v.x * k + c, v.x * v.y * k - v.z, v.x * v.z * k + v.y,
             v.y * v.x * k + v.z, v.y * v.y * k + c, v.y * v.z * k - v.x,
             v.z * v.x * k - v.y, v.z * v.y * k + v.x, v.z * v.z * k + c}};
}
/** The palm's frame: towards the middle knuckle, across the knuckles, out of the palm. */
inline M palm(const V p[21]) {
    V a = unit(sub(p[9], p[0]));
    V n = unit(cross(a, sub(p[5], p[17])));
    V b = cross(n, a);
    return cols(a, b, n);
}

inline bool load(const std::string& path, Mesh& m) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4]; uint32_t ver, nv, ni;
    bool ok = fread(magic, 1, 4, f) == 4 && !memcmp(magic, "MVHD", 4) && fread(&ver, 4, 1, f) && ver == 2 && fread(&nv, 4, 1, f) && fread(&ni, 4, 1, f);
    ok = ok && fread(m.bind, sizeof m.bind, 1, f) == 1;
    if (ok) {
        m.v.resize(nv);
        for (auto& x : m.v) ok = ok && fread(x.p, 4, 8, f) == 8 && fread(x.bone, 1, 4, f) == 4 && fread(x.w, 4, 4, f) == 4;
        m.idx.resize(ni);
        ok = ok && fread(m.idx.data(), 2, ni, f) == ni;
    }
    fclose(f);
    return ok;
}

inline const char* SHADER = R"(
cbuffer H : register(b0) { row_major float4x4 vp; float4 light_dir; float4 tint; };
cbuffer B : register(b1) { row_major float4x4 bones[17]; };
Texture2D skin : register(t0);
SamplerState samp : register(s0);
struct VI { float3 p : POSITION; float3 n : NORMAL; float2 uv : TEXCOORD; uint4 bi : BLENDINDICES; float4 bw : BLENDWEIGHT; };
struct VO { float4 pos : SV_Position; float3 n : NORMAL; float2 uv : TEXCOORD; };
VO vs(VI i) {
  float4x4 m = bones[i.bi.x] * i.bw.x + bones[i.bi.y] * i.bw.y + bones[i.bi.z] * i.bw.z + bones[i.bi.w] * i.bw.w;
  VO o; o.pos = mul(vp, mul(m, float4(i.p, 1))); o.n = mul((float3x3)m, i.n); o.uv = i.uv; return o;
}
float4 ps(VO i, bool front : SV_IsFrontFace) : SV_Target {
  float3 n = normalize(i.n);
  float d = abs(dot(n, normalize(light_dir.xyz)));
  float wrap = saturate((d + 0.35) / 1.35);  // soft, skin-like
  float3 albedo = skin.Sample(samp, i.uv).rgb;
  // Light through the skin: warm (reddish) where the surface turns from the light.
  float3 sss = float3(0.9, 0.35, 0.25) * (1 - d) * 0.12;
  float3 c = albedo * (0.42 + 0.58 * wrap) + albedo * sss;
  // The room's light, only partly: skin keeps its own colour.
  return float4(c * lerp(float3(1, 1, 1), tint.rgb, 0.7), 1);
}
// Sleeves: cloth in the tint's colour (tint = fabric x room), soft folds along the arm.
float4 ps_cloth(VO i) : SV_Target {
  float3 n = normalize(i.n);
  float d = abs(dot(n, normalize(light_dir.xyz)));
  float folds = 0.88 + 0.12 * sin(i.uv.x * 55 + sin(i.uv.y * 3) * 2.5) * sin(i.uv.y * 2 + i.uv.x * 9);
  float k = (0.45 + 0.55 * saturate((d + 0.3) / 1.3)) * folds;
  return float4(tint.rgb * k, 1);
}
)";

struct Renderer {
    Mesh hand[2];
    ID3D11VertexShader* vs = nullptr;
    ID3D11PixelShader* ps = nullptr;
    ID3D11PixelShader* ps_cloth = nullptr;
    ID3D11InputLayout* layout = nullptr;
    ID3D11Buffer* cb = nullptr;      // vp, light, tint
    ID3D11Buffer* bones = nullptr;   // 17 matrices
    ID3D11ShaderResourceView* tex = nullptr;
    ID3D11SamplerState* samp = nullptr;
    bool ok = false;
    std::string error;

    bool init(ID3D11Device* dev, const std::wstring& dir) {
        std::string d8(dir.begin(), dir.end());
        if (!load(d8 + "hand_l.bin", hand[0]) || !load(d8 + "hand_r.bin", hand[1])) { error = "hand_l.bin / hand_r.bin introuvables (tools/hands/extract_hands.py)"; return false; }
        ID3DBlob *vsb = nullptr, *psb = nullptr, *err = nullptr;
        if (FAILED(D3DCompile(SHADER, strlen(SHADER), nullptr, nullptr, nullptr, "vs", "vs_5_0", 0, 0, &vsb, &err)) ||
            FAILED(D3DCompile(SHADER, strlen(SHADER), nullptr, nullptr, nullptr, "ps", "ps_5_0", 0, 0, &psb, &err))) {
            error = err ? (const char*)err->GetBufferPointer() : "shader";
            return false;
        }
        dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs);
        dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &ps);
        ID3DBlob* pcb = nullptr;
        if (SUCCEEDED(D3DCompile(SHADER, strlen(SHADER), nullptr, nullptr, nullptr, "ps_cloth", "ps_5_0", 0, 0, &pcb, &err))) {
            dev->CreatePixelShader(pcb->GetBufferPointer(), pcb->GetBufferSize(), nullptr, &ps_cloth);
            pcb->Release();
        }
        D3D11_INPUT_ELEMENT_DESC il[] = {{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
                                         {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
                                         {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0},
                                         {"BLENDINDICES", 0, DXGI_FORMAT_R8G8B8A8_UINT, 0, 32, D3D11_INPUT_PER_VERTEX_DATA, 0},
                                         {"BLENDWEIGHT", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 36, D3D11_INPUT_PER_VERTEX_DATA, 0}};
        dev->CreateInputLayout(il, 5, vsb->GetBufferPointer(), vsb->GetBufferSize(), &layout);
        vsb->Release(); psb->Release();
        for (auto& h : hand) {
            make_sleeve(h);
            D3D11_BUFFER_DESC sb{}; sb.ByteWidth = (UINT)(h.sleeve.size() * sizeof(SkinVertex)); sb.BindFlags = D3D11_BIND_VERTEX_BUFFER; sb.Usage = D3D11_USAGE_IMMUTABLE;
            D3D11_SUBRESOURCE_DATA si{h.sleeve.data()};
            dev->CreateBuffer(&sb, &si, &h.sleeve_vb);
            D3D11_BUFFER_DESC bd{}; bd.ByteWidth = (UINT)(h.v.size() * sizeof(SkinVertex)); bd.BindFlags = D3D11_BIND_VERTEX_BUFFER; bd.Usage = D3D11_USAGE_IMMUTABLE;
            D3D11_SUBRESOURCE_DATA init{h.v.data()};
            dev->CreateBuffer(&bd, &init, &h.vb);
            D3D11_BUFFER_DESC ib{}; ib.ByteWidth = (UINT)(h.idx.size() * 2); ib.BindFlags = D3D11_BIND_INDEX_BUFFER; ib.Usage = D3D11_USAGE_IMMUTABLE;
            D3D11_SUBRESOURCE_DATA ii{h.idx.data()};
            dev->CreateBuffer(&ib, &ii, &h.ib);
        }
        D3D11_BUFFER_DESC cbd{}; cbd.ByteWidth = 96; cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; cbd.Usage = D3D11_USAGE_DEFAULT;
        dev->CreateBuffer(&cbd, nullptr, &cb);
        cbd.ByteWidth = BONES * 64;
        dev->CreateBuffer(&cbd, nullptr, &bones);
        D3D11_SAMPLER_DESC sd{}; sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR; sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_WRAP; sd.MaxLOD = D3D11_FLOAT32_MAX;
        dev->CreateSamplerState(&sd, &samp);
        tex = load_texture(dev, dir + L"hand_skin.jpg");
        if (!tex) { error = "hand_skin.jpg illisible"; return false; }
        ok = vs && ps && layout && cb && bones;
        return ok;
    }

    /**
     * The sleeve: a tube around the forearm (elbow to wrist, bind pose), its radius the
     * forearm's own (largest vertex distance from the axis, per slice) plus 6 mm, a cuff
     * at the wrist end. All on the forearm bone, so it moves with the arm, rigid.
     * uv: x along (0 elbow .. 1 wrist), y around (radians).
     */
    static void make_sleeve(Mesh& h) {
        V E = {h.bind[21][0], h.bind[21][1], h.bind[21][2]}, W = {h.bind[0][0], h.bind[0][1], h.bind[0][2]};
        V ax = sub(W, E); double L = len(ax); V u = unit(ax);
        V p1 = unit(cross(u, std::fabs(u.y) < 0.9 ? V{0, 1, 0} : V{1, 0, 0})), p2 = cross(u, p1);
        const int SL = 14, SIDES = 24;
        double rad[SL + 1] = {};
        for (auto& v : h.v) {
            if (v.bone[0] != 16 || v.w[0] < 0.6f) continue;
            V p = sub(V{v.p[0], v.p[1], v.p[2]}, E);
            double t = dot(p, u) / L;
            if (t < 0 || t > 1) continue;
            double r = len(sub(p, mulv(u, dot(p, u))));
            int k = (int)(t * SL);
            rad[k] = std::max(rad[k], r);
        }
        double last = 0.04;
        for (int k = 0; k <= SL; k++) { if (rad[k] <= 0) rad[k] = last; last = rad[k]; }
        auto radius = [&](double t) {
            double x = std::max(0.0, std::min(1.0, t)) * SL; int k = std::min(SL - 1, (int)x); double f = x - k;
            double r = rad[k] * (1 - f) + rad[k + 1] * f + 0.006;
            if (t > 0.93) r += 0.004;  // the cuff
            return r;
        };
        std::vector<SkinVertex> ring_prev, ring;
        const int RINGS = 40;
        for (int i = 0; i <= RINGS; i++) {
            double t = -0.08 + 1.08 * i / RINGS;  // a little past the elbow (the cut is hidden)
            double r = radius(t);
            ring.clear();
            for (int a = 0; a <= SIDES; a++) {
                double ang = 6.2831853 * a / SIDES;
                V n = add(mulv(p1, std::cos(ang)), mulv(p2, std::sin(ang)));
                V p = add(add(E, mulv(u, t * L)), mulv(n, r));
                SkinVertex sv{};
                sv.p[0] = (float)p.x; sv.p[1] = (float)p.y; sv.p[2] = (float)p.z;
                sv.n[0] = (float)n.x; sv.n[1] = (float)n.y; sv.n[2] = (float)n.z;
                sv.uv[0] = (float)t; sv.uv[1] = (float)ang;
                sv.bone[0] = 16; sv.w[0] = 1;
                ring.push_back(sv);
            }
            if (i > 0) for (int a = 0; a < SIDES; a++) {
                h.sleeve.insert(h.sleeve.end(), {ring_prev[a], ring_prev[a + 1], ring[a + 1], ring_prev[a], ring[a + 1], ring[a]});
            }
            ring_prev = ring;
        }
        // The cuff's rim: a ring facing the hand, so the opening has a thickness.
        double tr_ = 1.0, r_out = radius(tr_), r_in = r_out - 0.005;
        V cpos = add(E, mulv(u, tr_ * L));
        for (int a = 0; a < SIDES; a++) {
            double a0 = 6.2831853 * a / SIDES, a1 = 6.2831853 * (a + 1) / SIDES;
            auto vtx = [&](double ang, double r) {
                V n = add(mulv(p1, std::cos(ang)), mulv(p2, std::sin(ang)));
                V p = add(cpos, mulv(n, r));
                SkinVertex sv{}; sv.p[0] = (float)p.x; sv.p[1] = (float)p.y; sv.p[2] = (float)p.z;
                sv.n[0] = (float)u.x; sv.n[1] = (float)u.y; sv.n[2] = (float)u.z; sv.uv[0] = 1; sv.uv[1] = (float)ang; sv.bone[0] = 16; sv.w[0] = 1;
                return sv;
            };
            SkinVertex o0 = vtx(a0, r_out), o1 = vtx(a1, r_out), i0 = vtx(a0, r_in), i1 = vtx(a1, r_in);
            h.sleeve.insert(h.sleeve.end(), {o0, o1, i1, o0, i1, i0});
        }
    }

    static ID3D11ShaderResourceView* load_texture(ID3D11Device* dev, const std::wstring& path) {
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        IWICImagingFactory* wic = nullptr; IWICBitmapDecoder* dec = nullptr; IWICBitmapFrameDecode* fr = nullptr; IWICFormatConverter* cv = nullptr;
        ID3D11ShaderResourceView* srv = nullptr;
        if (SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic))) &&
            SUCCEEDED(wic->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, &dec)) &&
            SUCCEEDED(dec->GetFrame(0, &fr)) && SUCCEEDED(wic->CreateFormatConverter(&cv)) &&
            SUCCEEDED(cv->Initialize(fr, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeCustom))) {
            UINT w, h; cv->GetSize(&w, &h);
            std::vector<uint8_t> px((size_t)w * h * 4);
            cv->CopyPixels(nullptr, w * 4, (UINT)px.size(), px.data());
            D3D11_TEXTURE2D_DESC td{}; td.Width = w; td.Height = h; td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;  // as the glasses' target: no sRGB conversion
            td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_IMMUTABLE; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            D3D11_SUBRESOURCE_DATA init{px.data(), w * 4, 0};
            ID3D11Texture2D* t = nullptr;
            if (SUCCEEDED(dev->CreateTexture2D(&td, &init, &t))) { dev->CreateShaderResourceView(t, nullptr, &srv); t->Release(); }
        }
        if (cv) cv->Release(); if (fr) fr->Release(); if (dec) dec->Release(); if (wic) wic->Release();
        return srv;
    }

    /** Bone matrices (row-major 4x4, bind -> scene) for tracked joints `j` and the elbow (scene, m). */
    static void pose(const Mesh& m, const float j[22][3], float out[BONES][16]) {
        V B[22], T[22];
        for (int k = 0; k < 22; k++) { B[k] = {m.bind[k][0], m.bind[k][1], m.bind[k][2]}; T[k] = {j[k][0], j[k][1], j[k][2]}; }
        M Rh = mm(palm(T), tr(palm(B)));
        double s = len(sub(T[9], T[0])) / std::max(1e-6, len(sub(B[9], B[0])));
        // The thumb's base is not tracked (VITURE gives none; ours is a midpoint): it moves
        // rigidly with the palm, so the palm does not twist under the thumb.
        T[1] = add(T[0], mv(Rh, mulv(sub(B[1], B[0]), s)));
        for (int b = 0; b < BONES; b++) {
            V h0 = B[HEAD[b]], h1 = T[HEAD[b]];
            M R = Rh, S{{s, 0, 0, 0, s, 0, 0, 0, s}};
            if (b > 0) {
                V d0 = sub(B[TAIL[b]], h0), d1 = sub(T[TAIL[b]], h1);
                double l0 = len(d0), l1 = len(d1);
                V u0 = unit(d0);
                R = mm(between(unit(mv(Rh, u0)), unit(d1)), Rh);
                // Stretch along the bone to its tracked length, the hand's scale across it.
                double k = l0 > 1e-6 ? l1 / l0 : s;
                k = std::max(0.6 * s, std::min(1.6 * s, k));  // a noisy joint must not crush or stretch the mesh
                double e[3] = {u0.x, u0.y, u0.z};
                for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) S.m[r * 3 + c] = (r == c ? s : 0) + (k - s) * e[r] * e[c];
            }
            M A = mm(R, S);
            V t = sub(h1, mv(A, h0));
            float* o = out[b];
            for (int r = 0; r < 3; r++) { for (int c = 0; c < 3; c++) o[r * 4 + c] = (float)A.m[r * 3 + c]; }
            o[3] = (float)t.x; o[7] = (float)t.y; o[11] = (float)t.z;
            o[12] = o[13] = o[14] = 0; o[15] = 1;
        }
    }

    /**
     * The elbow for a hand (scene, m): two-bone IK from a shoulder estimated from the head
     * (`head`: head -> scene, row-major 4x4; 19 cm to the side, 24 cm down, 10 cm back,
     * with the head's heading only), the elbow bending down, out and back.
     */
    void elbow(const float j[21][3], bool right, const float head[16], float out[3]) const {
        const Mesh& m = hand[right ? 1 : 0];
        V B0 = {m.bind[0][0], m.bind[0][1], m.bind[0][2]}, B9 = {m.bind[9][0], m.bind[9][1], m.bind[9][2]}, BE = {m.bind[21][0], m.bind[21][1], m.bind[21][2]};
        V W = {j[0][0], j[0][1], j[0][2]}, M9 = {j[9][0], j[9][1], j[9][2]};
        double s = len(sub(M9, W)) / std::max(1e-6, len(sub(B9, B0)));
        double fore = len(sub(B0, BE)) * s, upper = fore * 1.12;
        V hp = {head[3], head[7], head[11]};
        V fwd = unit(V{-head[2], 0, -head[10]}), up = {0, 1, 0}, side = cross(fwd, up);  // side: the wearer's right
        double sg = right ? 1 : -1;
        V S = add(add(hp, mulv(side, 0.19 * sg)), add(mulv(up, -0.24), mulv(fwd, -0.10)));
        V d = sub(W, S);
        double L = std::min(len(d), (fore + upper) * 0.999), Lmin = std::fabs(upper - fore) + 1e-3;
        V hand_axis = unit(sub(M9, W));
        L = std::max(L, Lmin);
        V u = unit(d);
        double cosA = (upper * upper + L * L - fore * fore) / (2 * upper * L);
        cosA = std::max(-1.0, std::min(1.0, cosA));
        double sinA = std::sqrt(1 - cosA * cosA);
        V pole = add(add(mulv(up, -1), mulv(side, 0.5 * sg)), mulv(fwd, -0.3));
        V v = unit(sub(pole, mulv(u, dot(pole, u))));
        V E_ik = add(S, add(mulv(u, upper * cosA), mulv(v, upper * sinA)));
        // The forearm half continues the hand's axis, half follows the shoulder's IK: the
        // wrist bends, but not far.
        V dir = unit(add(mulv(hand_axis, -0.5), mulv(unit(sub(E_ik, W)), 0.5)));
        V E = add(W, mulv(dir, fore));
        out[0] = (float)E.x; out[1] = (float)E.y; out[2] = (float)E.z;
    }

    /** One hand; `vp`: view-projection (row-major), `tint`: the room light's colour and level. */
    float light[3] = {0.3f, 1.0f, 0.5f};  // towards the light (scene), set from the room

    void draw(ID3D11DeviceContext* ctx, const float vp[16], const float j[22][3], bool right, const float tint[3], const float* sleeve = nullptr) {
        if (!ok) return;
        const Mesh& m = hand[right ? 1 : 0];
        float bones_m[BONES][16];
        pose(m, j, bones_m);
        struct { float vp[16]; float light[4]; float tint[4]; } c;
        memcpy(c.vp, vp, sizeof c.vp);
        c.light[0] = light[0]; c.light[1] = light[1]; c.light[2] = light[2]; c.light[3] = 0;
        c.tint[0] = tint[0]; c.tint[1] = tint[1]; c.tint[2] = tint[2]; c.tint[3] = 1;
        ctx->UpdateSubresource(cb, 0, nullptr, &c, 0, 0);
        ctx->UpdateSubresource(bones, 0, nullptr, bones_m, 0, 0);
        UINT stride = sizeof(SkinVertex), offset = 0;
        ctx->IASetInputLayout(layout);
        ctx->IASetVertexBuffers(0, 1, &m.vb, &stride, &offset);
        ctx->IASetIndexBuffer(m.ib, DXGI_FORMAT_R16_UINT, 0);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx->VSSetShader(vs, nullptr, 0);
        ID3D11Buffer* cbs[] = {cb, bones};
        ctx->VSSetConstantBuffers(0, 2, cbs);
        ctx->PSSetShader(ps, nullptr, 0);
        ctx->PSSetConstantBuffers(0, 1, cbs);
        ctx->PSSetShaderResources(0, 1, &tex);
        ctx->PSSetSamplers(0, 1, &samp);
        ctx->DrawIndexed((UINT)m.idx.size(), 0, 0);
        // The sleeve: fabric colour x the room's light, same bones.
        if (sleeve && ps_cloth && m.sleeve_vb) {
            auto c2 = c;
            for (int i = 0; i < 3; i++) c2.tint[i] = sleeve[i] * (0.4f + 0.6f * tint[i]);
            ctx->UpdateSubresource(cb, 0, nullptr, &c2, 0, 0);
            ctx->IASetVertexBuffers(0, 1, &m.sleeve_vb, &stride, &offset);
            ctx->PSSetShader(ps_cloth, nullptr, 0);
            ctx->Draw((UINT)m.sleeve.size(), 0);
        }
        ID3D11ShaderResourceView* none[] = {nullptr};
        ctx->PSSetShaderResources(0, 1, none);
    }
};

}  // namespace skinhands
