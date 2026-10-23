// Stereo depth from the two tracking cameras, on the GPU with Direct3D 11 compute shaders,
// by semi-global matching (SGM, Hirschmüller 2008; costs and penalties as in libSGM):
//   1. rectification of both fisheye images into a common pinhole frame whose x axis is
//      the baseline (lookup tables from the factory calibration), 320 x 240 (at 640 x 480
//      these cameras' noise leaves SGM grainy, 7x slower for little: measured on 20 pairs);
//   2. census transform 9x7 (62 bits): matching on intensity order, so the cameras'
//      different exposure and vignetting do not matter;
//   3. cost volume: Hamming distance for 64 disparities (one byte each);
//   4. semi-global aggregation along 8 directions: along each, a disparity step of one
//      costs P1, a larger jump P2, so a plain wall or floor takes the disparity of its
//      textured edges instead of none;
//   5. winner-takes-all with uniqueness and sub-pixel refinement, left and right, then a
//      left-right check;
//   6. CPU: speckle removal, 3x3 median, depth = f * baseline / disparity.
// Runs on its own D3D11 device, on the room-mapping thread (roommap.hpp).

#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "calib.hpp"

namespace depth {

constexpr int W = 320, H = 240, MAX_DISP = 64;
constexpr float RECT_F = 150;     // px, rectified focal (~93 deg horizontal field)
constexpr int SRC_W = 640, SRC_H = 480;  // the tracking cameras' images

/** Rectified frame in the head coordinates (x along the baseline, y down, z forward). */
struct Axes { double ex[3], ey[3], ez[3], origin[3], baseline, f; };

struct Frame {
    Axes axes;
    std::vector<float> depth;    // W*H metres along ez, 0 = none
    std::vector<uint8_t> left;   // rectified left image
    float valid_ratio = 0;
    double t = 0;                // image time (QPC s)
    /** head-frame point of rectified pixel (u, v) at depth z */
    void point(int u, int v, float z, double out[3]) const {
        double x = ((u + 0.5 - W / 2.0) / axes.f) * z, y = ((v + 0.5 - H / 2.0) / axes.f) * z;
        for (int i = 0; i < 3; i++) out[i] = axes.origin[i] + axes.ex[i] * x + axes.ey[i] * y + axes.ez[i] * z;
    }
};

// W, H, MAXD, SRC_W, SRC_H: defined by the compiler call from the constants above.
static const char* CS = R"(
cbuffer P : register(b0) { int dirx; int diry; int first; int pad; };
Texture2D<float> img : register(t0);
Texture2D<float2> lut : register(t1);
Texture2D<float> rectIn : register(t2);
Texture2D<uint2> L : register(t3);
Texture2D<uint2> R : register(t4);
Texture2D<float> dispLIn : register(t6);
Texture2D<float> dispRIn : register(t7);
SamplerState lin : register(s0);
RWTexture2D<float> rectOut : register(u0);
RWTexture2D<uint2> censusOut : register(u1);
RWTexture2D<float> dispOut : register(u3);
RWByteAddressBuffer costBuf : register(u4);  // W*H*MAXD bytes: pixel-major, disparity inside
RWByteAddressBuffer sumBuf : register(u5);   // W*H*MAXD/2 uints: two 16-bit sums of the paths
static const uint P1 = 10, P2 = 120, BAD = 62;
static const uint INVALID = 0x80000000u;

[numthreads(16, 16, 1)]
void rectify(uint3 id : SV_DispatchThreadID) {
  if (id.x >= W || id.y >= H) return;
  float2 p = lut[id.xy];
  if (p.x < 0) { rectOut[id.xy] = 0; return; }
  // Real pixels are at least 1/255: 0 means outside the rectified field.
  rectOut[id.xy] = max(img.SampleLevel(lin, (p + 0.5) / float2(SRC_W, SRC_H), 0), 1.0 / 255.0);
}

[numthreads(16, 16, 1)]
void census(uint3 id : SV_DispatchThreadID) {
  if (id.x >= W || id.y >= H) return;
  int2 c = int2(id.xy);
  float ctr = rectIn[c];
  uint2 bits = 0;
  uint k = 0;
  [unroll] for (int y = -3; y <= 3; y++) [unroll] for (int x = -4; x <= 4; x++) {
    if (x == 0 && y == 0) continue;
    int2 q = clamp(c + int2(x, y), int2(0, 0), int2(W - 1, H - 1));
    if (rectIn[q] < ctr) { if (k < 31) bits.x |= 1u << k; else bits.y |= 1u << (k - 31); }
    k++;
  }
  if (ctr == 0) bits.x |= INVALID;
  censusOut[c] = bits;
}

[numthreads(16, 16, 1)]
void cost(uint3 id : SV_DispatchThreadID) {
  if (id.x >= W || id.y >= H) return;
  int2 c = int2(id.xy);
  uint2 a = L[c];
  bool bad = (a.x & INVALID) != 0;
  uint base = (c.y * W + c.x) * MAXD;
  for (int d = 0; d < MAXD; d += 4) {
    uint w = 0;
    [unroll] for (int j = 0; j < 4; j++) {
      int xr = c.x - d - j;
      uint v = BAD;
      if (!bad && xr >= 0) {
        uint2 b = R[int2(xr, c.y)];
        if ((b.x & INVALID) == 0) v = countbits((a.x ^ b.x) & ~INVALID) + countbits(a.y ^ b.y);
      }
      w |= v << (8 * j);
    }
    costBuf.Store(base + d, w);
  }
}

// One path direction: a thread group per image line along (dirx, diry), a thread per two
// disparities; L(p, d) = C(p, d) + min(L(p-r, d), L(p-r, d+-1) + P1, min L(p-r) + P2) - min L(p-r).
groupshared uint gsL[MAXD + 2];  // the previous pixel's L at d + 1; both ends: none
groupshared uint gsMin[3];       // min over d of L, rotating: written, read, cleared

[numthreads(MAXD / 2, 1, 1)]
void sgm(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
  int2 dir = int2(dirx, diry);
  int n = gid.x;
  int2 p;
  if (dir.y == 0) p = int2(dir.x > 0 ? 0 : W - 1, n);
  else if (n < W) p = int2(n, dir.y > 0 ? 0 : H - 1);
  else { int m = n - W; p = int2(dir.x > 0 ? 0 : W - 1, dir.y > 0 ? m + 1 : H - 2 - m); }
  uint t = tid.x, d0 = 2 * t;
  if (t == 0) { gsL[0] = 0xFFFF; gsL[MAXD + 1] = 0xFFFF; gsMin[0] = 0xFFFF; gsMin[1] = 0xFFFF; gsMin[2] = 0xFFFF; }
  GroupMemoryBarrierWithGroupSync();
  uint k = 0;
  [loop] while (p.x >= 0 && p.x < W && p.y >= 0 && p.y < H) {
    uint pix = p.y * W + p.x;
    uint cw = costBuf.Load(pix * MAXD + (d0 & ~3u));
    uint sh = (d0 & 3u) * 8;
    uint c0 = (cw >> sh) & 0xFF, c1 = (cw >> (sh + 8)) & 0xFF;
    uint l0 = c0, l1 = c1;
    if (k > 0) {
      uint m = gsMin[(k + 2) % 3];
      uint am = gsL[d0], a = gsL[d0 + 1], b = gsL[d0 + 2], bp = gsL[d0 + 3];
      l0 = c0 + min(min(a, m + P2), min(am, b) + P1) - m;
      l1 = c1 + min(min(b, m + P2), min(a, bp) + P1) - m;
    }
    GroupMemoryBarrierWithGroupSync();
    gsL[d0 + 1] = l0; gsL[d0 + 2] = l1;
    if (t == 0) gsMin[(k + 1) % 3] = 0xFFFF;
    InterlockedMin(gsMin[k % 3], min(l0, l1));
    uint addr = (pix * (MAXD / 2) + t) * 4, v = l0 | (l1 << 16);
    sumBuf.Store(addr, first ? v : sumBuf.Load(addr) + v);
    GroupMemoryBarrierWithGroupSync();
    p += dir; k++;
  }
}

uint S(uint pix, int d) { uint v = sumBuf.Load((pix * (MAXD / 2) + (d >> 1)) * 4); return (d & 1) ? v >> 16 : v & 0xFFFF; }

[numthreads(16, 16, 1)]
void wtaLeft(uint3 id : SV_DispatchThreadID) {
  if (id.x >= W || id.y >= H) return;
  int2 c = int2(id.xy);
  int dmax = min(MAXD - 1, c.x - 4);
  if ((L[c].x & INVALID) != 0 || dmax < 3) { dispOut[c] = 0; return; }
  uint pix = c.y * W + c.x;
  uint best = 0xFFFFFFFF; int bd = -1;
  for (int d = 0; d <= dmax; d++) { uint v = S(pix, d); if (v < best) { best = v; bd = d; } }
  if (bd < 1 || bd >= dmax) { dispOut[c] = 0; return; }
  uint second = 0xFFFFFFFF;
  for (int d2 = 0; d2 <= dmax; d2++) if (abs(d2 - bd) > 1) second = min(second, S(pix, d2));
  if (best * 100 > second * 95) { dispOut[c] = 0; return; }  // not unique by 5 %
  float c0 = S(pix, bd - 1), c2 = S(pix, bd + 1);
  float den = c0 - 2.0 * best + c2;
  float sub = den > 1e-6 ? clamp(0.5 * (c0 - c2) / den, -0.5, 0.5) : 0.0;
  dispOut[c] = bd + sub;
}

[numthreads(16, 16, 1)]
void wtaRight(uint3 id : SV_DispatchThreadID) {
  if (id.x >= W || id.y >= H) return;
  int2 c = int2(id.xy);
  uint best = 0xFFFFFFFF; int bd = -1;
  for (int d = 0; d < MAXD; d++) {
    int xl = c.x + d;
    if (xl >= W) break;
    uint v = S(c.y * W + xl, d);
    if (v < best) { best = v; bd = d; }
  }
  dispOut[c] = bd;
}

[numthreads(16, 16, 1)]
void lrCheck(uint3 id : SV_DispatchThreadID) {
  if (id.x >= W || id.y >= H) return;
  int2 c = int2(id.xy);
  float dl = dispLIn[c];
  if (dl <= 0) { dispOut[c] = 0; return; }
  int xr = c.x - (int)floor(dl + 0.5);
  float dr = xr >= 0 ? dispRIn[int2(xr, c.y)] : -10.0;
  dispOut[c] = abs(dl - dr) <= 1.0 ? dl : 0;
}
)";

template <typename T> struct Com {
    T* p = nullptr;
    ~Com() { if (p) p->Release(); }
    T** operator&() { return &p; }
    T* operator->() const { return p; }
    operator T*() const { return p; }
};

class Stereo {
public:
    bool ok = false;
    std::string error;
    Axes axes{};
    int paths = 8;  // SGM directions: 8 (with the diagonals) or 4

    Stereo(const calib::Rig& rig) {
        axes = rectified_axes(rig);
        D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
        if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &fl, 1, D3D11_SDK_VERSION, &dev, nullptr, &ctx))) { error = "D3D11 device"; return; }
        std::string mw = std::to_string(W), mh = std::to_string(H), md = std::to_string(MAX_DISP), msw = std::to_string(SRC_W), msh = std::to_string(SRC_H);
        D3D_SHADER_MACRO defs[] = {{"W", mw.c_str()}, {"H", mh.c_str()}, {"MAXD", md.c_str()}, {"SRC_W", msw.c_str()}, {"SRC_H", msh.c_str()}, {nullptr, nullptr}};
        const char* entries[] = {"rectify", "census", "cost", "sgm", "wtaLeft", "wtaRight", "lrCheck"};
        for (int i = 0; i < 7; i++) {
            Com<ID3DBlob> blob, err;
            if (FAILED(D3DCompile(CS, strlen(CS), nullptr, defs, nullptr, entries[i], "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &err))) {
                error = std::string("shader ") + entries[i] + ": " + (err ? (const char*)err->GetBufferPointer() : "?");
                return;
            }
            if (FAILED(dev->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &cs[i]))) { error = "compute shader"; return; }
        }
        auto tex2d = [&](DXGI_FORMAT f, int w, int h, UINT bind, ID3D11Texture2D** t, D3D11_USAGE usage = D3D11_USAGE_DEFAULT, UINT cpu = 0) {
            D3D11_TEXTURE2D_DESC d{}; d.Width = w; d.Height = h; d.MipLevels = 1; d.ArraySize = 1; d.Format = f; d.SampleDesc.Count = 1; d.Usage = usage; d.BindFlags = bind; d.CPUAccessFlags = cpu;
            return dev->CreateTexture2D(&d, nullptr, t);
        };
        for (int i = 0; i < 2; i++) {
            tex2d(DXGI_FORMAT_R8_UNORM, SRC_W, SRC_H, D3D11_BIND_SHADER_RESOURCE, &img[i], D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE);
            dev->CreateShaderResourceView(img[i], nullptr, &img_srv[i]);
            tex2d(DXGI_FORMAT_R32G32_FLOAT, W, H, D3D11_BIND_SHADER_RESOURCE, &lut[i]);
            dev->CreateShaderResourceView(lut[i], nullptr, &lut_srv[i]);
            std::vector<float> table = build_lut(rig.cam[i]);
            ctx->UpdateSubresource(lut[i], 0, nullptr, table.data(), W * 8, 0);
            tex2d(DXGI_FORMAT_R32_FLOAT, W, H, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, &rect[i]);
            dev->CreateShaderResourceView(rect[i], nullptr, &rect_srv[i]);
            dev->CreateUnorderedAccessView(rect[i], nullptr, &rect_uav[i]);
            tex2d(DXGI_FORMAT_R32G32_UINT, W, H, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, &cen[i]);
            dev->CreateShaderResourceView(cen[i], nullptr, &cen_srv[i]);
            dev->CreateUnorderedAccessView(cen[i], nullptr, &cen_uav[i]);
        }
        auto rawbuf = [&](UINT bytes, ID3D11Buffer** b, ID3D11UnorderedAccessView** u) {
            D3D11_BUFFER_DESC d{}; d.ByteWidth = bytes; d.BindFlags = D3D11_BIND_UNORDERED_ACCESS; d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
            if (FAILED(dev->CreateBuffer(&d, nullptr, b))) return false;
            D3D11_UNORDERED_ACCESS_VIEW_DESC ud{}; ud.Format = DXGI_FORMAT_R32_TYPELESS; ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
            ud.Buffer.NumElements = bytes / 4; ud.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
            return SUCCEEDED(dev->CreateUnorderedAccessView(*b, &ud, u));
        };
        if (!rawbuf(W * H * MAX_DISP, &cost_buf, &cost_uav) || !rawbuf(W * H * MAX_DISP * 2, &sum_buf, &sum_uav)) { error = "GPU memory (cost volume)"; return; }
        for (int i = 0; i < 3; i++) {
            tex2d(DXGI_FORMAT_R32_FLOAT, W, H, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, &disp[i]);
            dev->CreateShaderResourceView(disp[i], nullptr, &disp_srv[i]);
            dev->CreateUnorderedAccessView(disp[i], nullptr, &disp_uav[i]);
        }
        tex2d(DXGI_FORMAT_R32_FLOAT, W, H, 0, &stage_disp, D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ);
        tex2d(DXGI_FORMAT_R32_FLOAT, W, H, 0, &stage_rect, D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ);
        D3D11_BUFFER_DESC bd{}; bd.ByteWidth = 16; bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; bd.Usage = D3D11_USAGE_DEFAULT;
        dev->CreateBuffer(&bd, nullptr, &cb);
        D3D11_SAMPLER_DESC sd{}; sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR; sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        dev->CreateSamplerState(&sd, &sampler);
        ok = true;
    }

    /** One depth map from two 640x480 gray images. */
    bool compute(const uint8_t* left, const uint8_t* right, Frame& out) {
        if (!ok) return false;
        const uint8_t* src[2] = {left, right};
        for (int i = 0; i < 2; i++) {
            D3D11_MAPPED_SUBRESOURCE m;
            if (FAILED(ctx->Map(img[i], 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) return false;
            for (int y = 0; y < SRC_H; y++) memcpy((uint8_t*)m.pData + y * m.RowPitch, src[i] + y * SRC_W, SRC_W);
            ctx->Unmap(img[i], 0);
        }
        ID3D11Buffer* cbs[] = {cb};
        ctx->CSSetConstantBuffers(0, 1, cbs);
        ID3D11SamplerState* ss[] = {sampler};
        ctx->CSSetSamplers(0, 1, ss);
        auto run = [&](int shader, std::initializer_list<ID3D11ShaderResourceView*> srvs, std::initializer_list<ID3D11UnorderedAccessView*> uavs, UINT groups = 0) {
            ID3D11ShaderResourceView* s[8] = {}; int i = 0;
            for (auto v : srvs) s[i++] = v;
            ID3D11UnorderedAccessView* u[6] = {}; i = 0;
            for (auto v : uavs) u[i++] = v;
            ctx->CSSetShader(cs[shader], nullptr, 0);
            ctx->CSSetShaderResources(0, 8, s);
            ctx->CSSetUnorderedAccessViews(0, 6, u, nullptr);
            if (groups) ctx->Dispatch(groups, 1, 1); else ctx->Dispatch((W + 15) / 16, (H + 15) / 16, 1);
            ID3D11ShaderResourceView* none_s[8] = {};
            ID3D11UnorderedAccessView* none_u[6] = {};
            ctx->CSSetShaderResources(0, 8, none_s);
            ctx->CSSetUnorderedAccessViews(0, 6, none_u, nullptr);
        };
        auto params = [&](int dx, int dy, int first) { int p[4] = {dx, dy, first, 0}; ctx->UpdateSubresource(cb, 0, nullptr, p, 0, 0); };
        ID3D11ShaderResourceView* _ = nullptr;
        for (int i = 0; i < 2; i++) run(0, {img_srv[i], lut_srv[i]}, {rect_uav[i]});
        for (int i = 0; i < 2; i++) run(1, {_, _, rect_srv[i]}, {nullptr, cen_uav[i]});
        run(2, {_, _, _, cen_srv[0], cen_srv[1]}, {nullptr, nullptr, nullptr, nullptr, cost_uav});
        const int DIRS[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, 1}, {1, -1}, {-1, -1}};
        for (int k = 0; k < std::clamp(paths, 1, 8); k++) {
            int dx = DIRS[k][0], dy = DIRS[k][1];
            params(dx, dy, k == 0);
            run(3, {}, {nullptr, nullptr, nullptr, nullptr, cost_uav, sum_uav}, dy == 0 ? H : dx == 0 ? W : W + H - 1);
        }
        run(4, {_, _, _, cen_srv[0]}, {nullptr, nullptr, nullptr, disp_uav[0], nullptr, sum_uav});
        run(5, {}, {nullptr, nullptr, nullptr, disp_uav[1], nullptr, sum_uav});
        run(6, {_, _, _, _, _, _, disp_srv[0], disp_srv[1]}, {nullptr, nullptr, nullptr, disp_uav[2]});
        ctx->CopyResource(stage_disp, disp[2]);
        ctx->CopyResource(stage_rect, rect[0]);

        std::vector<float> d(W * H);
        D3D11_MAPPED_SUBRESOURCE m;
        if (FAILED(ctx->Map(stage_disp, 0, D3D11_MAP_READ, 0, &m))) return false;
        for (int y = 0; y < H; y++) memcpy(d.data() + y * W, (uint8_t*)m.pData + y * m.RowPitch, W * 4);
        ctx->Unmap(stage_disp, 0);
        out.left.resize(W * H);
        if (SUCCEEDED(ctx->Map(stage_rect, 0, D3D11_MAP_READ, 0, &m))) {
            for (int y = 0; y < H; y++) { const float* row = (const float*)((uint8_t*)m.pData + y * m.RowPitch); for (int x = 0; x < W; x++) out.left[y * W + x] = (uint8_t)std::clamp(row[x] * 255.0f, 0.0f, 255.0f); }
            ctx->Unmap(stage_rect, 0);
        }
        remove_speckles(d, 1.0f, 25);
        std::vector<float> f = median3(d);
        out.depth.assign(W * H, 0);
        out.axes = axes;
        const double fb = RECT_F * axes.baseline;
        int valid = 0;
        for (int i = 0; i < W * H; i++) if (f[i] > 1.5f) { out.depth[i] = (float)(fb / f[i]); valid++; }
        out.valid_ratio = (float)valid / (W * H);
        return true;
    }

private:
    Com<ID3D11Device> dev; Com<ID3D11DeviceContext> ctx;
    Com<ID3D11ComputeShader> cs[7];
    Com<ID3D11Texture2D> img[2], lut[2], rect[2], cen[2], disp[3], stage_disp, stage_rect;
    Com<ID3D11Buffer> cost_buf, sum_buf;
    Com<ID3D11ShaderResourceView> img_srv[2], lut_srv[2], rect_srv[2], cen_srv[2], disp_srv[3];
    Com<ID3D11UnorderedAccessView> rect_uav[2], cen_uav[2], disp_uav[3], cost_uav, sum_uav;
    Com<ID3D11Buffer> cb;
    Com<ID3D11SamplerState> sampler;

    static Axes rectified_axes(const calib::Rig& rig) {
        const calib::Camera& l = rig.cam[0];
        const calib::Camera& r = rig.cam[1];
        Axes a{};
        double b[3] = {r.origin[0] - l.origin[0], r.origin[1] - l.origin[1], r.origin[2] - l.origin[2]};
        a.baseline = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
        for (int i = 0; i < 3; i++) a.ex[i] = b[i] / a.baseline;
        double fl[3], fr[3];
        l.ray(l.cx, l.cy, fl); r.ray(r.cx, r.cy, fr);
        double ez[3] = {fl[0] + fr[0], fl[1] + fr[1], fl[2] + fr[2]};
        double k = ez[0] * a.ex[0] + ez[1] * a.ex[1] + ez[2] * a.ex[2];
        for (int i = 0; i < 3; i++) ez[i] -= k * a.ex[i];
        double n = std::sqrt(ez[0] * ez[0] + ez[1] * ez[1] + ez[2] * ez[2]);
        for (int i = 0; i < 3; i++) a.ez[i] = ez[i] / n;
        a.ey[0] = a.ez[1] * a.ex[2] - a.ez[2] * a.ex[1]; a.ey[1] = a.ez[2] * a.ex[0] - a.ez[0] * a.ex[2]; a.ey[2] = a.ez[0] * a.ex[1] - a.ez[1] * a.ex[0];
        for (int i = 0; i < 3; i++) a.origin[i] = l.origin[i];
        a.f = RECT_F;
        return a;
    }

    /** For rectified pixel (u, v): source pixel in this camera, or -1 when outside. */
    std::vector<float> build_lut(const calib::Camera& cam) const {
        std::vector<float> t(W * H * 2);
        for (int v = 0; v < H; v++) for (int u = 0; u < W; u++) {
            double x = (u + 0.5 - W / 2.0) / RECT_F, y = (v + 0.5 - H / 2.0) / RECT_F;
            double d[3], pc[3];
            for (int i = 0; i < 3; i++) d[i] = axes.ex[i] * x + axes.ey[i] * y + axes.ez[i];
            // head -> camera: R^T
            for (int i = 0; i < 3; i++) pc[i] = cam.R[i] * d[0] + cam.R[3 + i] * d[1] + cam.R[6 + i] * d[2];
            float* o = t.data() + (v * W + u) * 2;
            if (pc[2] <= 0.05) { o[0] = o[1] = -1; continue; }
            double pu, pv;
            cam.distort(pc[0] / pc[2], pc[1] / pc[2], pu, pv);
            bool inside = pu >= 0 && pu < SRC_W && pv >= 0 && pv < SRC_H;
            o[0] = inside ? (float)pu : -1; o[1] = inside ? (float)pv : -1;
        }
        return t;
    }

    static void remove_speckles(std::vector<float>& d, float max_diff, int min_size) {
        std::vector<int> label(d.size(), 0), stack(d.size()), region;
        int next = 1;
        for (size_t s = 0; s < d.size(); s++) {
            if (!(d[s] > 0) || label[s]) continue;
            region.clear();
            int sp = 0;
            stack[sp++] = (int)s; label[s] = next;
            while (sp) {
                int i = stack[--sp];
                region.push_back(i);
                int x = i % W, y = i / W;
                int nb[4] = {x > 0 ? i - 1 : -1, x < W - 1 ? i + 1 : -1, y > 0 ? i - W : -1, y < H - 1 ? i + W : -1};
                for (int j : nb) {
                    if (j < 0 || label[j] || !(d[j] > 0) || std::fabs(d[j] - d[i]) > max_diff) continue;
                    label[j] = next; stack[sp++] = j;
                }
            }
            if ((int)region.size() < min_size) for (int i : region) d[i] = 0;
            next++;
        }
    }

    static std::vector<float> median3(const std::vector<float>& d) {
        std::vector<float> out(d.size(), 0);
        float v[9];
        for (int y = 1; y < H - 1; y++) for (int x = 1; x < W - 1; x++) {
            int i = y * W + x;
            if (!(d[i] > 0)) continue;
            int n = 0;
            for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) { float z = d[i + dy * W + dx]; if (z > 0) v[n++] = z; }
            std::sort(v, v + n);
            out[i] = v[n / 2];
        }
        return out;
    }
};

}  // namespace depth
