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
    float lead_ms = 0;          // prediction used: pose read -> vblank -> screen
    float pred_err_still = 0, pred_err_moving = 0, pred_err_fast = 0;  // prediction error p95 (deg): head still, slow, fast
    float tw_err_moving = 0, tw_err_fast = 0;  // the same after the timewarp
    float fallback_hz = 0;      // frames without the engine's prediction (it failed), per second
    float imu_gap_p99 = 0;      // IMU callbacks: gap between arrivals, p99 (ms; 1 = steady 1 kHz)
    long missed_fast = 0;       // missed vblanks while the head turned fast (> 60 deg/s)
};

struct AppState {
    // ---- written by the dashboard (or the glasses window's keys)
    float predict_ms = 12;      // the screen's own latency after the vblank (prediction goes to vblank + this)
    // Stability test: a thin cross, world-locked, laid on a real edge (a door frame); the
    // screen latency is tuned until it neither trails nor leads when the head shakes.
    bool stab_test = false;
    int stab_request = 0;       // 1: place the cross at the gaze, 2: remove it
    int stab_mode = 0;          // compared settings: 0 smoothing + timewarp, 1 no smoothing, 2 no timewarp, 3 neither
    bool late_latch = false;    // read the pose and draw just before the vblank (off: GPU sharing made it miss vblanks)
    bool tracking_lost = false; // (written) the engine pose jumps impossibly: the scene holds the last good pose
    long tracking_losses = 0;
    bool timewarp = true;       // turn the rendered eyes by the head's rotation read just before the vblank
    float tw_margin_ms = 4;     // (written) how long before the vblank the warp reads the pose
    float tw_deg = 0;           // (written) mean rotation the warp corrected per frame
    float latch_margin_ms = 5;  // how long before the vblank (adapts: more when a vblank is missed)
    bool engine_predict = true;
    float vfov = 28.147f;   // per-eye vertical field, degrees
    float ipd_mm = 63.0f;
    float ipd_adj_mm = 0;      // added to the engine's eyes' separation: the virtual's depth (more: nearer and smaller)
    float floor_m = 1.6f;   // floor below the eyes
    float pitch_deg = 0;    // display tilt about x: objects too high -> lower it
    bool display_from_rgb = false; // the display takes the RGB camera's fitted orientation (superseded by the engine's eyes)
    bool vendor_eyes = true;       // per-eye poses from the VITURE engine (factory calibration)
    bool have_vendor_eyes = false;
    float eye_l[16] = {}, eye_r[16] = {};  // eye -> head, row-major (from the engine)
    float eye_dy_cm = 0;    // eyes above (+) / below (-) the tracking origin
    float eye_dz_cm = 0;    // eyes behind (+) the tracking origin
    float yaw_deg = 0;      // display turn about the vertical axis
    bool calage = false;    // calibration view: the scanned room as small points over the real one
    bool scene = true;
    bool floor_grid = false;
    bool hands = true;
    bool room_light = true;    // the virtual takes the room's light level and colour (RGB camera)
    bool shadows = true;       // contact shadows under virtual objects and hands (RGB view)
    bool camera_look = true;   // the RGB view's virtual softened and grained like the camera
    float room_level = 0;      // the room's mean luminance (0..1), for the dashboard
    bool panel = false;         // the test text panel in front of the wearer
    bool rgb_overlay = true;   // virtual hands and cubes drawn over the RGB view
    float rgb_latency_ms = 50; // an RGB frame's age when it arrives: the virtual is drawn for that moment
    int rec_request = 0;       // 1: start recording the RGB view, 2: stop
    bool rec_audio = true;     // with the microphone
    int rec_mbps = 8;
    bool photo_request = false;
    bool rgb_reset = false;    // forget the RGB camera fit
    bool smooth_corrections = true;  // blend the engine's visual corrections instead of jumping
    bool scan = true;            // accumulate the room from the depth maps
    bool voice = true;           // voice commands (Orange's Whisper)
    bool show_map = false;       // draw the room as thin lines: floor-wall seams, wall edges, tables, walkable area
    bool show_mesh = false;      // draw the clutter mesh (debug: it fills the view)
    // The virtual human: shown, wandering on its own; requests from the dashboard or the MCP.
    bool human = true;
    bool human_wander = true;
    float human_height = 1.72f;  // the human's height (m; the model's own: 1.72)
    int cubes_request = 0;     // 1: three new cubes in front of the wearer
    int human_request = 0;       // 1: come in front of me, 2: go to human_target (tracking frame)
    float human_target[2] = {0, 0};
    std::string human_state;     // (written) what it is doing
    int show_request = 0;        // an anchor's id: the human goes and points at it
    std::string anchor_label;    // the name for the next anchor (room_request 1)
    bool aiming = false;         // a name is being typed: the aiming dot in the glasses
    float human_pos[3] = {0, 0, 0};  // (written) tracking frame, on the floor
    bool occlusion = true;       // real surfaces (the room mesh) hide the virtual behind them
    bool place_on_planes = true; // rest the test objects on the floor / tables found
    int room_request = 0;        // 1: anchor at the gaze, 2: forget the room
    bool recenter = true;
    bool quit = false;
    bool snapshot = false;  // save the dashboard as dashboard.png
    int tab_request = -1;   // tests: 0 shows the RGB view tab, 1 the diagnostics
    std::string notice;     // shown in the top bar for a few seconds
    double notice_until = 0;
    int display_request = 0;  // 2 or 3: switch the glasses to that mode; 4: 3D at 90 Hz
    int screen_hz = 0;        // (written) the glasses' refresh rate, from Windows
    float vsync_hz = 0;       // (written) the engine's vsync callback rate
    int brightness = 4, brightness_request = -1;  // the screen's level, 0..8
    bool film_dark = false; int film_request = -1;
    bool expo_fixed = false; int expo_request = -1;  // the tracking cameras' exposure

    // ---- written by the renderer
    std::string serial;
    int screen_w = 0, screen_h = 0;
    Stats stats;
    float yaw = 0, pitch = 0, roll = 0, pos[3] = {}, rate_dps = 0;
    float head_scene[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};  // row-major
    /** the RGB view: the newest RGB frame with the virtual scene over it (BGRA), for the dashboard */
    void* mr_srv = nullptr;
    // The 3D debug view of the scene (dashboard tab): drawn only while the tab is open.
    bool v3_open = false;              // set by the dashboard every frame the tab shows
    void* v3_srv = nullptr; int v3_w = 1280, v3_h = 800;
    float v3_yaw = 0.6f, v3_pitch = 0.6f, v3_dist = 5.0f;   // orbit around the target (rad, m)
    float v3_target[3] = {0, 0, 0};    // tracking frame
    bool v3_follow = true;             // target = the head
    bool v3_voxels = true, v3_evidence = false, v3_live = false, v3_planes = true, v3_walk = true, v3_mesh = false, v3_trail = true, v3_dense = true;
    int mr_w = 0, mr_h = 0;
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
