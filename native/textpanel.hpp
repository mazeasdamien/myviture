// A text panel in the room, for tests: lines of text drawn with GDI into a texture,
// shown on a quad; one button that the index finger can poke.

#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#pragma comment(lib, "gdi32.lib")

namespace panel {

constexpr int TW = 1024, TH = 512;        // texture pixels
constexpr float PW = 0.36f, PH = 0.18f;   // metres
// Button, in panel-local metres (x right, y up, origin at the centre).
constexpr float BX0 = 0.06f, BX1 = 0.16f, BY0 = -0.075f, BY1 = -0.03f;

struct Panel {
    ID3D11Texture2D* tex = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    HDC dc = nullptr;
    HBITMAP bmp = nullptr;
    void* bits = nullptr;
    HFONT font_big = nullptr, font_small = nullptr;
    std::string last;

    bool init(ID3D11Device* dev) {
        D3D11_TEXTURE2D_DESC td{}; td.Width = TW; td.Height = TH; td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_B8G8R8X8_UNORM;
        td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DYNAMIC; td.BindFlags = D3D11_BIND_SHADER_RESOURCE; td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (FAILED(dev->CreateTexture2D(&td, nullptr, &tex)) || FAILED(dev->CreateShaderResourceView(tex, nullptr, &srv))) return false;
        dc = CreateCompatibleDC(nullptr);
        BITMAPINFO bi{}; bi.bmiHeader.biSize = sizeof bi.bmiHeader; bi.bmiHeader.biWidth = TW; bi.bmiHeader.biHeight = -TH;  // top-down
        bi.bmiHeader.biPlanes = 1; bi.bmiHeader.biBitCount = 32; bi.bmiHeader.biCompression = BI_RGB;
        bmp = CreateDIBSection(dc, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
        SelectObject(dc, bmp);
        font_big = CreateFontW(-56, 0, 0, 0, FW_SEMIBOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
        font_small = CreateFontW(-40, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
        SetBkMode(dc, TRANSPARENT);
        return bits != nullptr;
    }

    /** Redraws when the content changed: title, lines, the button lit or not. */
    void draw(ID3D11DeviceContext* ctx, const std::string& title, const std::vector<std::string>& lines, const std::string& button, bool lit) {
        std::string key = title + "\n" + button + (lit ? "1" : "0");
        for (auto& l : lines) key += "\n" + l;
        if (key == last) return;
        last = key;
        // Black is see-through on the glasses: the panel is a frame, its text and its button.
        memset(bits, 0, (size_t)TW * TH * 4);
        HBRUSH frame = CreateSolidBrush(RGB(70, 140, 255));
        RECT r{0, 0, TW, TH};
        FrameRect(dc, &r, frame);
        RECT r2{3, 3, TW - 3, TH - 3};
        FrameRect(dc, &r2, frame);
        DeleteObject(frame);
        auto text = [&](HFONT f, COLORREF c, int x, int y, const std::string& s) {
            int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
            std::wstring w(n, 0);
            MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
            SelectObject(dc, f);
            SetTextColor(dc, c);
            TextOutW(dc, x, y, w.c_str(), (int)w.size() - 1);
        };
        text(font_big, RGB(255, 255, 255), 36, 24, title);
        int y = 110;
        for (auto& l : lines) { text(font_small, RGB(200, 225, 255), 36, y, l); y += 52; }
        // Button: panel-local metres -> pixels.
        auto px = [](float x) { return (int)((x / PW + 0.5f) * TW); };
        auto py = [](float y) { return (int)((0.5f - y / PH) * TH); };
        RECT b{px(BX0), py(BY1), px(BX1), py(BY0)};
        HBRUSH fill = CreateSolidBrush(lit ? RGB(60, 230, 120) : RGB(40, 60, 90));
        FillRect(dc, &b, fill);
        DeleteObject(fill);
        SIZE sz;
        SelectObject(dc, font_small);
        int n = MultiByteToWideChar(CP_UTF8, 0, button.c_str(), -1, nullptr, 0);
        std::wstring w(n, 0);
        MultiByteToWideChar(CP_UTF8, 0, button.c_str(), -1, w.data(), n);
        GetTextExtentPoint32W(dc, w.c_str(), (int)w.size() - 1, &sz);
        SetTextColor(dc, RGB(255, 255, 255));
        TextOutW(dc, (b.left + b.right - sz.cx) / 2, (b.top + b.bottom - sz.cy) / 2, w.c_str(), (int)w.size() - 1);
        GdiFlush();
        D3D11_MAPPED_SUBRESOURCE m;
        if (SUCCEEDED(ctx->Map(tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
            for (int yy = 0; yy < TH; yy++) memcpy((uint8_t*)m.pData + yy * m.RowPitch, (uint8_t*)bits + (size_t)yy * TW * 4, TW * 4);
            ctx->Unmap(tex, 0);
        }
    }
};

}  // namespace panel
