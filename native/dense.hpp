// A dense depth frame from the RGB camera (depthany.hpp), already in metres and in the
// tracking frame: a grid of world points, NaN where there is none. Written by the Depth
// Anything thread, read by the room thread (roommap.hpp).

#pragma once
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace dense {

struct Frame {
    int w = 0, h = 0;                // grid size
    std::vector<float> pts;          // w*h*3, world (tracking frame), NaN: none
    float eye[3] = {0, 0, 0};        // the camera's centre then (normals face it)
    double t = 0;                    // capture (QPC s)
    uint32_t seq = 0;
};

inline std::mutex g_mu;
inline Frame g_frame;
// What the dashboard shows.
inline std::string g_status = "démarrage…";
inline float g_ms = 0, g_hz = 0, g_scale_rms = 0;  // inference, rate, fit residual (1/m)
inline int g_pairs = 0;                             // stereo points the scale was fitted on

}  // namespace dense
