// What the glasses renderer and the dashboard share: settings and requests the
// dashboard writes, measurements and sensor data it shows. Touched by the render thread
// only (the dashboard is drawn there too), except the sensor buffers (own mutexes).

#pragma once
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct Stats {
    float fps = 0, latency_p50 = 0, latency_p99 = 0, cpu_ms = 0, pose_hz = 0;
    float shake_p50 = 0, shake_p99 = 0, lead_used = 0;
    long missed = 0;
    int corrections = 0;          // visual corrections of the engine in the last 2 s
    float corr_deg = 0, corr_mm = 0;  // their mean size
};

struct AppState {
    // ---- written by the dashboard (or the glasses window's keys)
    float predict_ms = 20;
    bool engine_predict = true;
    float vfov = 28.147f;   // per-eye vertical field, degrees
    float ipd_mm = 63.0f;
    float floor_m = 1.6f;   // floor below the eyes
    bool scene = true;
    bool floor_grid = false;
    bool hands = true;
    bool hand_predict = false;  // extrapolate the hands' motion to the display time (felt worse: off)
    bool rgb_overlay = true;   // virtual hands and cubes drawn over the RGB view
    bool rgb_reset = false;    // forget the RGB camera fit
    bool smooth_corrections = true;  // blend the engine's visual corrections instead of jumping
    bool recenter = true;
    bool quit = false;
    bool snapshot = false;  // save the dashboard as dashboard.png
    int display_request = 0;  // 2 or 3: switch the glasses to that mode

    // ---- written by the renderer
    std::string serial;
    int screen_w = 0, screen_h = 0;
    Stats stats;
    float yaw = 0, pitch = 0, roll = 0, pos[3] = {}, rate_dps = 0;
    float head_scene[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};  // row-major
    std::vector<float> shake_history;    // deg per frame, newest last
    std::vector<float> latency_history;  // pose -> vblank, ms
};

/** IMU samples (1 kHz) from the engine callback, newest last. */
struct ImuBuffer {
    std::mutex mu;
    std::vector<float> axis[6];  // ax ay az (m/s^2), gx gy gz (rad/s)
    long count = 0;
};

/** Newest stereo tracking frames from the engine callback. */
struct CameraBuffer {
    std::mutex mu;
    std::vector<uint8_t> px[2];
    int w = 0, h = 0;
    double t = 0;  // image time, QPC seconds (the engine's clock)
    uint32_t seq = 0;
    long count = 0;
};

inline ImuBuffer g_imu;
inline CameraBuffer g_cams;
