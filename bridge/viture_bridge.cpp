// viture_bridge: the VITURE VIO engine (carina_vio.dll, installed with SpaceWalker) for
// the browser app. The engine owns the Luma Ultra's USB link and computes the 6DoF pose;
// this program polls that pose at 500 Hz and streams it, with the raw IMU, the vsync and
// the stereo tracking images, to WebSocket clients on ws://127.0.0.1:8765.
//
//   viture_bridge.exe [--sdk "C:\Program Files\VITURE\SpaceWalker"] [--port 8765]
//
// Binary messages, little-endian (parsed by src/device/vitureBridge.ts):
//   1 pose   u8 type, u8 status, u16 0, i32 rc, f64 hostMs, f32[16] head->world (column-major, OpenGL axes)
//   2 imu    u8 type, u8[3] 0, u32 n, n x { f64 deviceT, f64 hostMs, f32[6] ax ay az gx gy gz }
//   3 frame  u8 type, u8 camera, u16 w, u16 h, u16 0, f64 deviceT, f64 hostMs, u8[w*h]
//   4 vsync  u8 type, u8[7] 0, f64 deviceT, f64 hostMs
// Text: {"type":"info",...} on connect, {"type":"display","mode":..,"rc":..} after a switch.
// Commands (text): "reset" (full VIO re-initialisation), "display <mode>" (0x41 2D, 0x42 3D).
//
// hostMs is the Windows system time in milliseconds since 1970, the clock the page's
// performance.timeOrigin + performance.now() follows.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <bcrypt.h>
#include <wincrypt.h>
#include <timeapi.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "bcrypt.lib")
#pragma comment(lib, "crypt32.lib")
#pragma comment(lib, "winmm.lib")

// ------------------------------------------------------------------ carina_vio API

typedef void (*PoseCb)(float*, double);
typedef void (*VsyncCb)(double);
typedef void (*ImuCb)(float*, double);
typedef void (*CamCb)(unsigned char*, unsigned char*, unsigned char*, unsigned char*, double, int, int);
typedef void (*PointsCb)(void*, double);
typedef void (*EventCb)(unsigned char);

struct Carina {
    int (*init)(const char*, const char*, int, int, int);
    int (*start)(PoseCb, VsyncCb, ImuCb, CamCb, PointsCb, EventCb);
    int (*resume)();
    int (*pause)();
    int (*stop)();
    int (*release)();
    int (*reset_pose)();
    int (*get_gl_pose)(float*, double);
    char* (*get_sn)();
    char* (*get_firmware)();
    int (*switch_display_mode)(unsigned char);
};

static Carina vio;

// ------------------------------------------------------------------ display MCU
//
// The engine's switch_display_mode writes "fa 55 06 .." to the camera board (endpoint
// 0x04), which the Luma Ultra refuses (-1): its display mode lives on the glasses MCU,
// 512-byte frames on endpoint 0x06 (see src/device/mcu.ts). The engine holds the only
// USB handle, so the frame goes through that handle: its address is read from the
// engine's own jcx_stm_switch_disp_mode ("cmp qword ptr [rip+disp32], 0" at +0x25),
// and libusb_bulk_transfer comes from the libusb-1.0.dll the engine loaded.

typedef int (*BulkTransfer)(void*, unsigned char, unsigned char*, int, int*, unsigned int);
static void** usb_handle_slot = nullptr;
static BulkTransfer usb_bulk = nullptr;

static void bind_display_mcu(HMODULE m) {
    auto* f = reinterpret_cast<const unsigned char*>(GetProcAddress(m, "jcx_stm_switch_disp_mode"));
    if (HMODULE lu = GetModuleHandleA("libusb-1.0.dll")) usb_bulk = reinterpret_cast<BulkTransfer>(GetProcAddress(lu, "libusb_bulk_transfer"));
    if (!f || !usb_bulk) return;
    const unsigned char* ins = f + 0x25;
    if (ins[0] != 0x48 || ins[1] != 0x83 || ins[2] != 0x3D || ins[7] != 0x00) {
        printf("[bridge] display MCU: unknown engine build, using the engine's switch\n");
        return;
    }
    int32_t disp;
    memcpy(&disp, ins + 3, 4);
    usb_handle_slot = reinterpret_cast<void**>(const_cast<unsigned char*>(ins + 8 + disp));
}

static uint16_t crc16_xmodem(const unsigned char* p, int n) {
    uint16_t crc = 0;
    for (int i = 0; i < n; i++) {
        crc ^= p[i] << 8;
        for (int b = 0; b < 8; b++) crc = crc & 0x8000 ? (crc << 1) ^ 0x1021 : crc << 1;
    }
    return crc;
}

/** Display mode through the glasses MCU (msg 8): 0 on success, libusb error otherwise. */
static int mcu_display_mode(unsigned char mode) {
    void* h = usb_handle_slot ? *usb_handle_slot : nullptr;
    if (!h || !usb_bulk) return -100;
    unsigned char f[512] = {};
    const uint16_t len = 12 + 1;
    f[0] = 0xff; f[1] = 0xfe;
    f[4] = len & 0xff; f[5] = len >> 8;
    f[14] = 8;
    f[18] = mode;
    uint16_t crc = crc16_xmodem(f + 4, len + 2);
    f[2] = crc & 0xff; f[3] = crc >> 8;
    int sent = 0;
    int rc = usb_bulk(h, 0x06, f, sizeof f, &sent, 1000);
    return rc ? rc : (sent == (int)sizeof f ? 0 : -101);
}

template <typename F>
static bool bind_fn(HMODULE m, F& fn, const char* name, bool required = true) {
    fn = reinterpret_cast<F>(GetProcAddress(m, name));
    if (!fn && required) fprintf(stderr, "missing export %s\n", name);
    return fn != nullptr || !required;
}

static double host_ms() {
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    uint64_t t = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    return (double)(t - 116444736000000000ULL) / 10000.0;
}

// ------------------------------------------------------------------ data from the engine's threads

static std::mutex g_mu;
static std::vector<unsigned char> g_imu;    // packed imu records (40 bytes each)
static std::vector<unsigned char> g_vsync;  // packed vsync messages
struct Frame { std::vector<unsigned char> px; int w = 0, h = 0; double t = 0, host = 0; bool fresh = false; };
static Frame g_frames[2];
static std::atomic<long> n_imu{0}, n_cam{0};
static std::atomic<bool> g_quit{false};

template <typename T>
static void put(std::vector<unsigned char>& b, T v) {
    size_t n = b.size();
    b.resize(n + sizeof(T));
    memcpy(&b[n], &v, sizeof(T));
}

static void on_pose(float*, double) {}
static void on_vsync(double t) {
    double h = host_ms();
    std::lock_guard<std::mutex> l(g_mu);
    if (g_vsync.size() > 16 * 1000) return;
    put<uint8_t>(g_vsync, 4); for (int i = 0; i < 7; i++) put<uint8_t>(g_vsync, 0);
    put(g_vsync, t); put(g_vsync, h);
}
static void on_imu(float* d, double t) {
    double h = host_ms();
    n_imu++;
    std::lock_guard<std::mutex> l(g_mu);
    if (g_imu.size() > 40 * 4000) return;  // no client draining
    put(g_imu, t); put(g_imu, h);
    for (int i = 0; i < 6; i++) put(g_imu, d[i]);
}
static void on_cam(unsigned char* l, unsigned char* r, unsigned char*, unsigned char*, double t, int w, int h) {
    double hm = host_ms();
    n_cam++;
    std::lock_guard<std::mutex> lk(g_mu);
    unsigned char* src[2] = {l, r};
    for (int i = 0; i < 2; i++) {
        if (!src[i]) continue;
        g_frames[i].px.assign(src[i], src[i] + (size_t)w * h);
        g_frames[i].w = w; g_frames[i].h = h; g_frames[i].t = t; g_frames[i].host = hm; g_frames[i].fresh = true;
    }
}
static void on_points(void*, double) {}
static void on_event(unsigned char e) { printf("[vio] event %d\n", e); }

// ------------------------------------------------------------------ WebSocket server

static std::string ws_accept_key(const std::string& key) {
    std::string s = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    unsigned char hash[20];
    BCryptHash(BCRYPT_SHA1_ALG_HANDLE, nullptr, 0, (PUCHAR)s.data(), (ULONG)s.size(), hash, 20);
    char out[64]; DWORD n = sizeof out;
    CryptBinaryToStringA(hash, 20, CRYPT_STRING_BASE64 | CRYPT_STRING_NOCRLF, out, &n);
    return std::string(out, n);
}

struct Client { SOCKET s; std::string inbuf; };

static bool send_all(SOCKET s, const char* p, size_t n) {
    while (n) {
        int k = send(s, p, (int)std::min<size_t>(n, 1 << 20), 0);
        if (k == SOCKET_ERROR && WSAGetLastError() == WSAEWOULDBLOCK) {
            fd_set w; FD_ZERO(&w); FD_SET(s, &w);
            timeval tv{1, 0};
            if (select(0, nullptr, &w, nullptr, &tv) <= 0) return false;
            continue;
        }
        if (k <= 0) return false;
        p += k; n -= k;
    }
    return true;
}

static bool ws_send(SOCKET s, const void* data, size_t n, bool binary) {
    unsigned char h[10]; size_t hl = 2;
    h[0] = 0x80 | (binary ? 2 : 1);
    if (n < 126) h[1] = (unsigned char)n;
    else if (n < 65536) { h[1] = 126; h[2] = (unsigned char)(n >> 8); h[3] = (unsigned char)n; hl = 4; }
    else { h[1] = 127; for (int i = 0; i < 8; i++) h[2 + i] = (unsigned char)((uint64_t)n >> (56 - 8 * i)); hl = 10; }
    return send_all(s, (const char*)h, hl) && send_all(s, (const char*)data, n);
}

static bool handshake(SOCKET s) {
    std::string req; char buf[2048];
    while (req.find("\r\n\r\n") == std::string::npos) {
        int k = recv(s, buf, sizeof buf, 0);
        if (k <= 0 || req.size() > 16384) return false;
        req.append(buf, k);
    }
    std::string low = req;
    for (auto& c : low) c = (char)tolower((unsigned char)c);
    auto pos = low.find("sec-websocket-key:");
    if (pos == std::string::npos) return false;
    auto end = req.find("\r\n", pos);
    std::string key = req.substr(pos + 18, end - pos - 18);
    key.erase(0, key.find_first_not_of(' '));
    key.erase(key.find_last_not_of(' ') + 1);
    std::string resp = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                       "Sec-WebSocket-Accept: " + ws_accept_key(key) + "\r\n\r\n";
    return send_all(s, resp.data(), resp.size());
}

static void command(Client& c, const std::string& cmd) {
    if (cmd == "reset") {
        printf("[bridge] VIO reset\n");
        vio.reset_pose();
    } else if (cmd.rfind("display ", 0) == 0) {
        int mode = atoi(cmd.c_str() + 8);
        int rc = mcu_display_mode((unsigned char)mode);
        printf("[bridge] display mode 0x%02X via MCU -> %d\n", mode, rc);
        if (rc && vio.switch_display_mode) {
            rc = vio.switch_display_mode((unsigned char)mode);
            printf("[bridge] display mode 0x%02X via engine -> %d\n", mode, rc);
        }
        char msg[96];
        int n = snprintf(msg, sizeof msg, "{\"type\":\"display\",\"mode\":%d,\"rc\":%d}", mode, rc);
        ws_send(c.s, msg, n, false);
    }
}

// Reads what the client sent; false when it closed.
static bool ws_poll(Client& c) {
    char buf[4096];
    for (;;) {
        int k = recv(c.s, buf, sizeof buf, 0);
        if (k == 0) return false;
        if (k < 0) { if (WSAGetLastError() == WSAEWOULDBLOCK) break; return false; }
        c.inbuf.append(buf, k);
    }
    for (;;) {
        auto& b = c.inbuf;
        if (b.size() < 2) break;
        unsigned char op = b[0] & 0x0f; uint64_t len = b[1] & 0x7f; size_t off = 2;
        if (len == 126) { if (b.size() < 4) break; len = ((unsigned char)b[2] << 8) | (unsigned char)b[3]; off = 4; }
        else if (len == 127) { if (b.size() < 10) break; len = 0; for (int i = 0; i < 8; i++) len = (len << 8) | (unsigned char)b[2 + i]; off = 10; }
        bool masked = b[1] & 0x80;
        size_t need = off + (masked ? 4 : 0) + (size_t)len;
        if (b.size() < need) break;
        std::string payload = b.substr(off + (masked ? 4 : 0), (size_t)len);
        if (masked) for (size_t i = 0; i < payload.size(); i++) payload[i] ^= b[off + (i & 3)];
        b.erase(0, need);
        if (op == 8) return false;
        if (op == 1) command(c, payload);
    }
    return true;
}

// ------------------------------------------------------------------ main

static BOOL WINAPI on_ctrl(DWORD) { g_quit = true; return TRUE; }

static std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

static std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if ((unsigned char)c < 0x20) { char u[8]; snprintf(u, sizeof u, "\\u%04x", c); o += u; }
        else o += c;
    }
    return o;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string sdk = "C:\\Program Files\\VITURE\\SpaceWalker";
    int port = 8765;
    for (int i = 1; i + 1 < argc; i++) {
        if (!strcmp(argv[i], "--sdk")) sdk = argv[++i];
        else if (!strcmp(argv[i], "--port")) port = atoi(argv[++i]);
    }

    SetDllDirectoryA(sdk.c_str());
    HMODULE m = LoadLibraryA((sdk + "\\carina_vio.dll").c_str());
    if (!m) {
        fprintf(stderr, "Cannot load carina_vio.dll from %s (error %lu). Is VITURE SpaceWalker installed?\n", sdk.c_str(), GetLastError());
        return 1;
    }
    bool ok = bind_fn(m, vio.init, "carina_a1088_viture_init") & bind_fn(m, vio.start, "carina_a1088_viture_start") &
              bind_fn(m, vio.resume, "carina_a1088_viture_resume") & bind_fn(m, vio.pause, "carina_a1088_viture_pause") &
              bind_fn(m, vio.stop, "carina_a1088_viture_stop") & bind_fn(m, vio.release, "carina_a1088_viture_release") &
              bind_fn(m, vio.reset_pose, "carina_a1088_viture_reset_pose") & bind_fn(m, vio.get_gl_pose, "carina_a1088_viture_get_gl_pose") &
              bind_fn(m, vio.get_sn, "carina_a1088_viture_get_sn") &
              bind_fn(m, vio.get_firmware, "carina_a1088_viture_get_firmware_version", false) &
              bind_fn(m, vio.switch_display_mode, "carina_a1088_viture_switch_display_mode", false);
    if (!ok) return 1;
    bind_display_mcu(m);

    // The engine resolves relative paths (vocabulary, logs) against the working directory.
    SetCurrentDirectoryA((sdk + "\\Assets\\Configs").c_str());
    std::string cfg = read_file("custom_config.yaml");
    int rc = vio.init(cfg.c_str(), (sdk + "\\Assets\\Configs\\database.bin").c_str(), -1, -1, -1);
    if (rc) {
        fprintf(stderr, "VIO init failed (%d): glasses unplugged, or already used by SpaceWalker / the browser?\n", rc);
        return 2;
    }
    rc = vio.start(on_pose, on_vsync, on_imu, on_cam, on_points, on_event);
    if (!rc) rc = vio.resume();
    if (rc) { fprintf(stderr, "VIO start failed (%d)\n", rc); vio.release(); return 3; }
    std::string serial, firmware;
    if (char* sn = vio.get_sn()) serial = sn;
    if (vio.get_firmware) if (char* fw = vio.get_firmware()) firmware = fw;
    printf("[bridge] VIO running, glasses %s, firmware %s\n", serial.c_str(), firmware.c_str());
    SetConsoleCtrlHandler(on_ctrl, TRUE);

    WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa);
    SOCKET ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons((u_short)port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (bind(ls, (sockaddr*)&addr, sizeof addr) || listen(ls, 4)) {
        fprintf(stderr, "Port %d busy\n", port);
        g_quit = true;
    } else {
        printf("[bridge] ws://127.0.0.1:%d  (Ctrl+C to quit)\n", port);
    }
    u_long nb = 1; ioctlsocket(ls, FIONBIO, &nb);
    timeBeginPeriod(1);

    std::vector<Client> clients;
    auto last_stats = std::chrono::steady_clock::now();
    std::vector<unsigned char> imu, vs, posemsg, bin;
    Frame frames[2];
    long n_pose = 0;
    float pose[16] = {0};
    int pose_rc = 0;

    while (!g_quit) {
        SOCKET cs = accept(ls, nullptr, nullptr);
        if (cs != INVALID_SOCKET) {
            u_long block = 0; ioctlsocket(cs, FIONBIO, &block);
            int one = 1; setsockopt(cs, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof one);
            if (handshake(cs)) {
                u_long nb2 = 1; ioctlsocket(cs, FIONBIO, &nb2);
                std::string info = "{\"type\":\"info\",\"serial\":\"" + json_escape(serial) + "\",\"firmware\":\"" + json_escape(firmware) + "\"}";
                ws_send(cs, info.data(), info.size(), false);
                clients.push_back({cs, {}});
                printf("[bridge] client connected (%zu)\n", clients.size());
            } else closesocket(cs);
        }

        // get_gl_pose fills a column-major 4x4 head->world matrix (OpenGL axes, metres).
        pose_rc = vio.get_gl_pose(pose, 0.0);
        double now = host_ms();
        n_pose++;
        posemsg.clear();
        put<uint8_t>(posemsg, 1); put<uint8_t>(posemsg, (uint8_t)(pose_rc == 0 ? 0 : 1)); put<uint16_t>(posemsg, 0);
        put<int32_t>(posemsg, pose_rc); put(posemsg, now);
        for (int i = 0; i < 16; i++) put(posemsg, pose[i]);
        {
            std::lock_guard<std::mutex> l(g_mu);
            imu.swap(g_imu); g_imu.clear();
            vs.swap(g_vsync); g_vsync.clear();
            for (int i = 0; i < 2; i++) if (g_frames[i].fresh) {
                frames[i].px.swap(g_frames[i].px);
                frames[i].w = g_frames[i].w; frames[i].h = g_frames[i].h; frames[i].t = g_frames[i].t; frames[i].host = g_frames[i].host;
                frames[i].fresh = true; g_frames[i].fresh = false;
            }
        }
        std::vector<unsigned char> imumsg;
        if (!imu.empty()) {
            put<uint8_t>(imumsg, 2); put<uint8_t>(imumsg, 0); put<uint16_t>(imumsg, 0);
            put<uint32_t>(imumsg, (uint32_t)(imu.size() / 40));
            imumsg.insert(imumsg.end(), imu.begin(), imu.end());
        }

        for (size_t c = 0; c < clients.size();) {
            bool alive = ws_poll(clients[c]) && ws_send(clients[c].s, posemsg.data(), posemsg.size(), true);
            if (alive && !imumsg.empty()) alive = ws_send(clients[c].s, imumsg.data(), imumsg.size(), true);
            for (size_t k = 0; alive && k + 24 <= vs.size(); k += 24) alive = ws_send(clients[c].s, &vs[k], 24, true);
            for (int i = 0; i < 2 && alive; i++) {
                if (!frames[i].fresh) continue;
                bin.clear();
                put<uint8_t>(bin, 3); put<uint8_t>(bin, (uint8_t)i);
                put<uint16_t>(bin, (uint16_t)frames[i].w); put<uint16_t>(bin, (uint16_t)frames[i].h); put<uint16_t>(bin, 0);
                put(bin, frames[i].t); put(bin, frames[i].host);
                bin.insert(bin.end(), frames[i].px.begin(), frames[i].px.end());
                alive = ws_send(clients[c].s, bin.data(), bin.size(), true);
            }
            if (!alive) {
                closesocket(clients[c].s);
                clients.erase(clients.begin() + c);
                printf("[bridge] client left (%zu)\n", clients.size());
            } else c++;
        }
        frames[0].fresh = frames[1].fresh = false;
        imu.clear(); vs.clear();

        auto t = std::chrono::steady_clock::now();
        if (t - last_stats > std::chrono::seconds(5)) {
            last_stats = t;
            printf("[bridge] 5 s: pose %ld, imu %ld, frames %ld | rc %d position=[%.3f %.3f %.3f]\n",
                   n_pose, n_imu.exchange(0), n_cam.exchange(0), pose_rc, pose[12], pose[13], pose[14]);
            n_pose = 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    printf("[bridge] stopping\n");
    timeEndPeriod(1);
    for (auto& c : clients) closesocket(c.s);
    closesocket(ls);
    vio.pause();
    vio.stop();
    vio.release();
    WSACleanup();
    return 0;
}
