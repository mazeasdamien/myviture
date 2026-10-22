// The dashboard window (Dear ImGui) on the PC's screen. See dashboard.cpp.

#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>

#include <string>

#include "state.hpp"

namespace dash {

/** Opens the window centred in `area` (the PC screen's work area); ini: ImGui layout file. */
bool init(ID3D11Device* dev, ID3D11DeviceContext* ctx, const RECT& area, const std::string& ini);
/** Builds, draws and presents one dashboard frame (no vsync wait). */
void frame(AppState& s);
void shutdown();
/** Saves the dashboard's last frame as a PNG. */
bool save_png(const std::wstring& path);

}  // namespace dash
