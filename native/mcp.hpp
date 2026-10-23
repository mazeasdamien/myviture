// The scene as an MCP server (Model Context Protocol, streamable HTTP, JSON-RPC 2.0) on
// http://127.0.0.1:8765/mcp, local only. An assistant orchestrating this app and the robot
// cell's MCP reads the room (floor, walls, tables, walkable area), the wearer and the
// virtual human, moves the human, marks points (the index fingertip or the gaze on the
// room) and lays virtual markers for the technician.
//
// The robot cell's frame: three or more points marked here (a fingertip on the gripper,
// say) with their coordinates in the robot's frame give the rigid transform between the
// two (Horn's method), saved to myviture_robot_frame.txt; every tool can then speak in
// either frame.
//
// Coordinates: "world" is the glasses' tracking frame (metres, y up, fixed for the session),
// "robot" the calibrated cell frame.

#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "roommap.hpp"

#pragma comment(lib, "ws2_32.lib")

namespace mcp {

// ---------------------------------------------------------------- JSON (just enough)
struct J;
using JP = std::shared_ptr<J>;
struct J {
    enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
    bool b = false; double n = 0; std::string s;
    std::vector<JP> a;
    std::vector<std::pair<std::string, JP>> o;
    const J* get(const char* k) const { if (t != OBJ) return nullptr; for (auto& [kk, v] : o) if (kk == k) return v.get(); return nullptr; }
    double num(const char* k, double d) const { auto v = get(k); return v && v->t == NUM ? v->n : d; }
    std::string str(const char* k, const std::string& d = "") const { auto v = get(k); return v && v->t == STR ? v->s : d; }
    bool flag(const char* k, bool d) const { auto v = get(k); return v && v->t == BOOL ? v->b : d; }
};
struct Parser {
    const char* p; const char* e; bool bad = false;
    void ws() { while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++; }
    static void utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
        else { out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
    }
    bool string(std::string& out) {
        if (p >= e || *p != '"') return false;
        p++;
        while (p < e && *p != '"') {
            if (*p == '\\' && p + 1 < e) {
                p++;
                char c = *p++;
                if (c == 'n') out += '\n'; else if (c == 't') out += '\t'; else if (c == 'r') out += '\r'; else if (c == 'b') out += '\b'; else if (c == 'f') out += '\f';
                else if (c == 'u' && p + 4 <= e) {
                    unsigned cp = (unsigned)strtoul(std::string(p, 4).c_str(), nullptr, 16); p += 4;
                    if (cp >= 0xD800 && cp < 0xDC00 && p + 6 <= e && p[0] == '\\' && p[1] == 'u') { unsigned lo = (unsigned)strtoul(std::string(p + 2, 4).c_str(), nullptr, 16); p += 6; cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); }
                    utf8(out, cp);
                } else out += c;
            } else out += *p++;
        }
        if (p >= e) return false;
        p++;
        return true;
    }
    JP value(int depth = 0) {
        auto v = std::make_shared<J>();
        ws();
        if (p >= e || depth > 64) { bad = true; return v; }
        if (*p == '{') {
            v->t = J::OBJ; p++; ws();
            if (p < e && *p == '}') { p++; return v; }
            while (p < e) {
                ws(); std::string k;
                if (!string(k)) { bad = true; return v; }
                ws(); if (p >= e || *p != ':') { bad = true; return v; }
                p++;
                v->o.push_back({k, value(depth + 1)});
                ws();
                if (p < e && *p == ',') { p++; continue; }
                if (p < e && *p == '}') { p++; return v; }
                bad = true; return v;
            }
        } else if (*p == '[') {
            v->t = J::ARR; p++; ws();
            if (p < e && *p == ']') { p++; return v; }
            while (p < e) {
                v->a.push_back(value(depth + 1));
                ws();
                if (p < e && *p == ',') { p++; continue; }
                if (p < e && *p == ']') { p++; return v; }
                bad = true; return v;
            }
        } else if (*p == '"') { v->t = J::STR; if (!string(v->s)) bad = true; }
        else if (!strncmp(p, "true", std::min<size_t>(4, e - p))) { v->t = J::BOOL; v->b = true; p += 4; }
        else if (!strncmp(p, "false", std::min<size_t>(5, e - p))) { v->t = J::BOOL; p += 5; }
        else if (!strncmp(p, "null", std::min<size_t>(4, e - p))) { p += 4; }
        else { std::string tmp(p, std::min<size_t>(64, e - p)); char* end; v->t = J::NUM; v->n = strtod(tmp.c_str(), &end); size_t used = end - tmp.c_str(); if (!used) bad = true; p += used; }
        return v;
    }
};
inline std::string esc(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"') o += "\\\""; else if (c == '\\') o += "\\\\"; else if (c == '\n') o += "\\n"; else if (c == '\r') o += "\\r"; else if (c == '\t') o += "\\t";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o + "\"";
}
inline std::string dump(const J& v) {
    switch (v.t) {
    case J::NUL: return "null";
    case J::BOOL: return v.b ? "true" : "false";
    case J::NUM: { char b[32]; snprintf(b, sizeof b, "%.10g", v.n); return b; }
    case J::STR: return esc(v.s);
    case J::ARR: { std::string o = "["; for (size_t i = 0; i < v.a.size(); i++) { if (i) o += ","; o += dump(*v.a[i]); } return o + "]"; }
    case J::OBJ: { std::string o = "{"; for (size_t i = 0; i < v.o.size(); i++) { if (i) o += ","; o += esc(v.o[i].first) + ":" + dump(*v.o[i].second); } return o + "}"; }
    }
    return "null";
}
inline std::string num3(const float* p) { char b[96]; snprintf(b, sizeof b, "[%.3f, %.3f, %.3f]", p[0], p[1], p[2]); return b; }

// ---------------------------------------------------------------- state shared with the app
/** What the render thread publishes (tracking frame), a few times a second. */
struct Snapshot {
    bool tracking = false;
    float head[16] = {};              // head -> world, row-major
    bool has_floor = false; float floor_y = 0;
    std::vector<room::Plane> planes;
    room::WalkGrid walk;
    std::vector<room::Anchor> anchors;
    std::vector<room::Opening> openings;
    bool human_visible = false, human_placed = false, human_wander = false;
    float human_pos[3] = {}; float human_heading = 0; std::string human_state;
    bool hand[2] = {false, false}; float index_tip[2][3] = {};  // [0] left, [1] right
};
/** Requests for the render thread. */
struct Cmd { std::string op; float v[3] = {0, 0, 0}; bool flag = false; };
struct Marker { std::string name; float p[3]; float color[3]; };

inline std::mutex g_mu;
inline Snapshot g_snap;
inline std::vector<Cmd> g_cmds;
inline std::vector<Marker> g_markers;
inline std::map<std::string, std::array<float, 3>> g_marks;  // named points, world
inline bool g_robot_ok = false;
inline float g_robot[16];      // world -> robot, row-major (rigid)
inline float g_robot_rms = 0;
inline std::string g_robot_path;
inline std::atomic<bool> g_stop{false};
inline std::thread g_thread;
inline std::atomic<int> g_calls{0};
inline std::string g_status = "arrêté";

inline void apply(const float* M, const float* p, float* out) { for (int i = 0; i < 3; i++) out[i] = M[i * 4] * p[0] + M[i * 4 + 1] * p[1] + M[i * 4 + 2] * p[2] + M[i * 4 + 3]; }
inline void rigid_inv(const float* M, float* out) {
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) out[i * 4 + j] = M[j * 4 + i];
    for (int i = 0; i < 3; i++) out[i * 4 + 3] = -(out[i * 4] * M[3] + out[i * 4 + 1] * M[7] + out[i * 4 + 2] * M[11]);
    out[12] = out[13] = out[14] = 0; out[15] = 1;
}
inline void save_robot() {
    if (g_robot_path.empty()) return;
    if (FILE* f = fopen(g_robot_path.c_str(), "w")) { for (int i = 0; i < 16; i++) fprintf(f, "%.6f ", g_robot[i]); fprintf(f, "%.5f\n", g_robot_rms); fclose(f); }
}
inline void load_robot() {
    if (FILE* f = fopen(g_robot_path.c_str(), "r")) {
        float m[16], r = 0; int n = 0;
        for (int i = 0; i < 16; i++) n += fscanf(f, "%f", &m[i]);
        fscanf(f, "%f", &r);
        fclose(f);
        if (n == 16) { memcpy(g_robot, m, sizeof m); g_robot_rms = r; g_robot_ok = true; }
    }
}

/**
 * Rigid transform a -> b from point pairs (Horn 1987): the rotation is the quaternion of
 * the largest eigenvalue of the 4x4 matrix built from the cross-covariance (Jacobi).
 */
inline bool horn(const std::vector<std::array<double, 3>>& a, const std::vector<std::array<double, 3>>& b, float* M, float& rms) {
    size_t n = a.size();
    if (n < 3) return false;
    double ca[3] = {0, 0, 0}, cb[3] = {0, 0, 0};
    for (size_t i = 0; i < n; i++) for (int k = 0; k < 3; k++) { ca[k] += a[i][k] / n; cb[k] += b[i][k] / n; }
    double S[3][3] = {};
    for (size_t i = 0; i < n; i++) for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) S[r][c] += (a[i][r] - ca[r]) * (b[i][c] - cb[c]);
    double N[4][4] = {
        {S[0][0] + S[1][1] + S[2][2], S[1][2] - S[2][1], S[2][0] - S[0][2], S[0][1] - S[1][0]},
        {S[1][2] - S[2][1], S[0][0] - S[1][1] - S[2][2], S[0][1] + S[1][0], S[2][0] + S[0][2]},
        {S[2][0] - S[0][2], S[0][1] + S[1][0], -S[0][0] + S[1][1] - S[2][2], S[1][2] + S[2][1]},
        {S[0][1] - S[1][0], S[2][0] + S[0][2], S[1][2] + S[2][1], -S[0][0] - S[1][1] + S[2][2]}};
    double V[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    for (int sweep = 0; sweep < 50; sweep++) {
        double off = 0;
        for (int p = 0; p < 4; p++) for (int q = p + 1; q < 4; q++) off += N[p][q] * N[p][q];
        if (off < 1e-18) break;
        for (int p = 0; p < 4; p++) for (int q = p + 1; q < 4; q++) {
            if (std::fabs(N[p][q]) < 1e-15) continue;
            double th = 0.5 * std::atan2(2 * N[p][q], N[q][q] - N[p][p]);
            double c = std::cos(th), s = std::sin(th);
            for (int k = 0; k < 4; k++) { double kp = N[k][p], kq = N[k][q]; N[k][p] = c * kp - s * kq; N[k][q] = s * kp + c * kq; }
            for (int k = 0; k < 4; k++) { double pk = N[p][k], qk = N[q][k]; N[p][k] = c * pk - s * qk; N[q][k] = s * pk + c * qk; }
            for (int k = 0; k < 4; k++) { double kp = V[k][p], kq = V[k][q]; V[k][p] = c * kp - s * kq; V[k][q] = s * kp + c * kq; }
        }
    }
    int best = 0;
    for (int i = 1; i < 4; i++) if (N[i][i] > N[best][best]) best = i;
    double w = V[0][best], x = V[1][best], y = V[2][best], z = V[3][best];
    double R[3][3] = {{w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)},
                      {2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)},
                      {2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z}};
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) M[r * 4 + c] = (float)R[r][c];
        M[r * 4 + 3] = (float)(cb[r] - (R[r][0] * ca[0] + R[r][1] * ca[1] + R[r][2] * ca[2]));
    }
    M[12] = M[13] = M[14] = 0; M[15] = 1;
    double e = 0;
    for (size_t i = 0; i < n; i++) {
        float pa[3] = {(float)a[i][0], (float)a[i][1], (float)a[i][2]}, pb[3];
        apply(M, pa, pb);
        for (int k = 0; k < 3; k++) e += (pb[k] - b[i][k]) * (pb[k] - b[i][k]);
    }
    rms = (float)std::sqrt(e / n);
    return true;
}

/** The gaze (head -z) against the room's planes: the nearest hit, world. */
inline bool gaze_hit(const Snapshot& s, float out[3], std::string& what) {
    float o[3] = {s.head[3], s.head[7], s.head[11]}, d[3] = {-s.head[2], -s.head[6], -s.head[10]};
    float best = 1e9f;
    for (auto& p : s.planes) {
        float t = -1;
        if (p.kind == 0 || p.kind == 1) {
            if (std::fabs(d[1]) < 1e-4f) continue;
            t = (p.y - o[1]) / d[1];
            if (t <= 0) continue;
            float x = o[0] + d[0] * t, z = o[2] + d[2] * t;
            if (p.kind == 1 && (x < p.minx || x > p.maxx || z < p.minz || z > p.maxz)) continue;
        } else {
            float den = p.nx * d[0] + p.nz * d[2];
            if (std::fabs(den) < 1e-4f) continue;
            t = (p.d - p.nx * o[0] - p.nz * o[2]) / den;
            if (t <= 0) continue;
            float x = o[0] + d[0] * t, y = o[1] + d[1] * t, z = o[2] + d[2] * t, tt = -p.nz * x + p.nx * z;
            if (tt < p.t0 || tt > p.t1 || y < s.floor_y || y > std::max(p.top, s.floor_y + 2.5f)) continue;
        }
        if (t < best) { best = t; what = p.kind == 0 ? "sol" : p.kind == 1 ? "table" : "mur"; }
    }
    if (best > 20) return false;
    for (int k = 0; k < 3; k++) out[k] = o[k] + d[k] * best;
    return true;
}

// ---------------------------------------------------------------- tools
struct Tool { const char* name; const char* description; const char* schema; };
static const Tool TOOLS[] = {
    {"get_room", "La pièce vue par les lunettes : hauteur du sol, murs (extrémités au sol, hauteur), portes et fenêtres (reconnues par segmentation), tables (rectangle, hauteur), zone marchable (surface, bornes), ancres. frame: world (repère des lunettes, m, y vers le haut) ou robot (repère de la cellule, si calé).",
     R"({"type":"object","properties":{"frame":{"type":"string","enum":["world","robot"]}}})"},
    {"get_user", "Le porteur des lunettes : position de la tête, direction du regard, point regardé sur la pièce (sol, mur, table), mains visibles et bout des index.",
     R"({"type":"object","properties":{"frame":{"type":"string","enum":["world","robot"]}}})"},
    {"get_human", "L'humain virtuel : visible, position au sol, cap, ce qu'il fait (attend, marche, pas de chemin).",
     R"({"type":"object","properties":{"frame":{"type":"string","enum":["world","robot"]}}})"},
    {"move_human", "Fait marcher l'humain virtuel jusqu'à un point du sol (x, z), par un chemin de la zone marchable ; to='user' : devant le porteur.",
     R"({"type":"object","properties":{"x":{"type":"number"},"z":{"type":"number"},"y":{"type":"number"},"frame":{"type":"string","enum":["world","robot"]},"to":{"type":"string","enum":["point","user"]}}})"},
    {"set_human", "Affiche ou cache l'humain virtuel, et le laisse se promener ou non.",
     R"({"type":"object","properties":{"visible":{"type":"boolean"},"wander":{"type":"boolean"}}})"},
    {"mark_point", "Mémorise un point nommé : le bout de l'index (source index, main droite sinon gauche) ou le point regardé sur la pièce (source gaze). Sert au calage du repère robot (pointer un même point vu par le robot).",
     R"({"type":"object","properties":{"name":{"type":"string"},"source":{"type":"string","enum":["index","gaze"]}},"required":["name"]})"},
    {"calibrate_robot_frame", "Cale le repère de la cellule robot sur celui des lunettes : au moins 3 points déjà marqués (mark_point) avec leurs coordonnées dans le repère robot (m). Rend l'erreur RMS.",
     R"({"type":"object","properties":{"points":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"robot":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3}},"required":["name","robot"]}}},"required":["points"]})"},
    {"convert_point", "Convertit un point entre le repère des lunettes (world) et celui de la cellule robot (robot).",
     R"({"type":"object","properties":{"point":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},"from":{"type":"string","enum":["world","robot"]}},"required":["point","from"]})"},
    {"place_marker", "Pose un repère virtuel (petit cube et fil jusqu'au sol) visible dans les lunettes, pour montrer un endroit au technicien. Même nom : déplacé.",
     R"({"type":"object","properties":{"name":{"type":"string"},"point":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},"frame":{"type":"string","enum":["world","robot"]},"color":{"type":"string","enum":["vert","rouge","orange","bleu","blanc"]}},"required":["name","point"]})"},
    {"clear_markers", "Enlève un repère virtuel (name) ou tous.",
     R"({"type":"object","properties":{"name":{"type":"string"}}})"},
};

inline std::string text_result(const std::string& text, bool error = false) {
    return std::string("{\"content\":[{\"type\":\"text\",\"text\":") + esc(text) + "}],\"isError\":" + (error ? "true" : "false") + "}";
}

/** A point in the asked frame (world -> robot when asked and calibrated). */
inline std::string pt(const float* w, bool robot) {
    if (!robot) return num3(w);
    float r[3]; apply(g_robot, w, r); return num3(r);
}

inline std::string call_tool(const std::string& name, const J& args) {
    std::lock_guard<std::mutex> l(g_mu);
    const Snapshot& s = g_snap;
    bool robot = args.str("frame") == "robot";
    if (robot && !g_robot_ok) return text_result("Repère robot pas encore calé : marque 3 points (mark_point) puis calibrate_robot_frame.", true);
    const char* fr = robot ? "robot" : "world";
    char b[512];
    if (name == "get_room") {
        std::string o = std::string("Repère ") + fr + " (m).\n";
        if (!s.has_floor) o += "Sol pas encore trouvé (il faut regarder le sol).\n";
        else { float f[3] = {s.head[3], s.floor_y, s.head[11]}; snprintf(b, sizeof b, "Sol : y = %.3f (world)%s\n", s.floor_y, robot ? (" ; point du sol sous la tête en robot " + pt(f, true)).c_str() : ""); o += b; }
        int nw = 0, nt = 0;
        for (auto& p : s.planes) {
            if (p.kind == 2) {
                float a[3] = {p.d * p.nx - p.nz * p.t0, s.floor_y, p.d * p.nz + p.nx * p.t0}, c[3] = {p.d * p.nx - p.nz * p.t1, s.floor_y, p.d * p.nz + p.nx * p.t1};
                snprintf(b, sizeof b, "Mur %d : de %s à %s, %.2f m de long, vu jusqu'à %.2f m de haut, face vers (%.2f, %.2f)\n", ++nw, pt(a, robot).c_str(), pt(c, robot).c_str(), p.t1 - p.t0, p.top - s.floor_y, p.nx, p.nz);
                o += b;
            } else if (p.kind == 1) {
                float c0[3] = {p.minx, p.y, p.minz}, c1[3] = {p.maxx, p.y, p.maxz};
                snprintf(b, sizeof b, "Table %d : coins %s et %s, plateau à %.2f m du sol, %.2f x %.2f m\n", ++nt, pt(c0, robot).c_str(), pt(c1, robot).c_str(), p.y - s.floor_y, p.maxx - p.minx, p.maxz - p.minz);
                o += b;
            }
        }
        int walk = 0; for (uint8_t c : s.walk.c) walk += c == 1;
        snprintf(b, sizeof b, "Zone marchable : %.1f m² dans x %.2f..%.2f, z %.2f..%.2f (world, cases de %.0f cm)\n", walk * s.walk.cell * s.walk.cell, s.walk.x0, s.walk.x0 + s.walk.nx * s.walk.cell, s.walk.z0, s.walk.z0 + s.walk.nz * s.walk.cell, s.walk.cell * 100);
        o += b;
        for (auto& op : s.openings) {
            if (op.wall < 0 || op.wall >= (int)s.planes.size()) continue;
            const room::Plane& w = s.planes[op.wall];
            int wn = 0; for (int i = 0; i <= op.wall; i++) wn += s.planes[i].kind == 2;
            float a[3] = {w.d * w.nx - w.nz * op.t0, op.y0, w.d * w.nz + w.nx * op.t0}, c2[3] = {w.d * w.nx - w.nz * op.t1, op.y1, w.d * w.nz + w.nx * op.t1};
            snprintf(b, sizeof b, "%s sur le mur %d : coins %s et %s, %.2f m de large, de %.2f à %.2f m du sol\n", op.kind == semseg::DOOR ? "Porte" : "Fenêtre", wn,
                     pt(a, robot).c_str(), pt(c2, robot).c_str(), op.t1 - op.t0, op.y0 - s.floor_y, op.y1 - s.floor_y);
            o += b;
        }
        for (auto& a : s.anchors) { snprintf(b, sizeof b, "Ancre %d : %s\n", a.id, pt(a.p, robot).c_str()); o += b; }
        for (auto& m : g_markers) { o += "Repère virtuel « " + m.name + " » : " + pt(m.p, robot) + "\n"; }
        return text_result(o);
    }
    if (name == "get_user") {
        if (!s.tracking) return text_result("Pas de suivi 6DoF des lunettes pour l'instant.", true);
        float h[3] = {s.head[3], s.head[7], s.head[11]}, d[3] = {-s.head[2], -s.head[6], -s.head[10]};
        if (robot) { float r[16]; memcpy(r, g_robot, sizeof r); r[3] = r[7] = r[11] = 0; float dd[3]; apply(r, d, dd); memcpy(d, dd, sizeof d); }
        snprintf(b, sizeof b, "Tête %s (%s), regard vers %s", pt(h, robot).c_str(), fr, num3(d).c_str());
        std::string o = b;
        if (s.has_floor) { snprintf(b, sizeof b, ", yeux à %.2f m du sol", s.head[7] - s.floor_y); o += b; }
        float g[3]; std::string what;
        if (gaze_hit(s, g, what)) o += "\nRegarde : " + what + " en " + pt(g, robot);
        for (int k = 0; k < 2; k++) if (s.hand[k]) o += std::string("\nIndex ") + (k ? "droit" : "gauche") + " : " + pt(s.index_tip[k], robot);
        return text_result(o);
    }
    if (name == "get_human") {
        float pp[3] = {s.human_pos[0], s.human_pos[1], s.human_pos[2]};
        snprintf(b, sizeof b, "Humain : %s, %s, en %s (%s), cap %.0f°, promenade %s", s.human_visible ? "visible" : "caché", s.human_state.c_str(), pt(pp, robot).c_str(), fr, s.human_heading * 57.2958f, s.human_wander ? "oui" : "non");
        return text_result(b);
    }
    if (name == "move_human") {
        Cmd c; c.op = "human_go";
        if (args.str("to") == "user") c.op = "human_come";
        else {
            float p[3] = {(float)args.num("x", NAN), (float)args.num("y", s.floor_y), (float)args.num("z", NAN)};
            if (std::isnan(p[0]) || std::isnan(p[2])) return text_result("x et z sont nécessaires (ou to='user').", true);
            if (robot) { float inv[16], w[3]; rigid_inv(g_robot, inv); apply(inv, p, w); memcpy(p, w, sizeof p); }
            c.v[0] = p[0]; c.v[1] = p[1]; c.v[2] = p[2];
            if (s.walk.at(p[0], p[2]) != 1) { snprintf(b, sizeof b, "Le point (%.2f, %.2f) world n'est pas dans la zone marchable : l'humain ira au plus près.", p[0], p[2]); g_cmds.push_back(c); return text_result(b); }
        }
        g_cmds.push_back(c);
        return text_result(c.op == "human_come" ? "L'humain vient devant le porteur." : "L'humain y va.");
    }
    if (name == "set_human") {
        if (auto v = args.get("visible")) { Cmd c; c.op = "human_visible"; c.flag = v->b; g_cmds.push_back(c); }
        if (auto v = args.get("wander")) { Cmd c; c.op = "human_wander"; c.flag = v->b; g_cmds.push_back(c); }
        return text_result("C'est fait.");
    }
    if (name == "mark_point") {
        std::string nm = args.str("name"), src = args.str("source", "index");
        if (nm.empty()) return text_result("name est nécessaire.", true);
        float p[3];
        if (src == "gaze") {
            std::string what;
            if (!gaze_hit(s, p, what)) return text_result("Le regard ne touche aucune surface connue de la pièce.", true);
        } else {
            int k = s.hand[1] ? 1 : s.hand[0] ? 0 : -1;
            if (k < 0) return text_result("Aucune main visible : tends l'index devant les lunettes.", true);
            memcpy(p, s.index_tip[k], sizeof p);
        }
        g_marks[nm] = {p[0], p[1], p[2]};
        return text_result("Point « " + nm + " » : " + num3(p) + " (world)" + (g_robot_ok ? ", " + pt(p, true) + " (robot)" : ""));
    }
    if (name == "calibrate_robot_frame") {
        auto pts = args.get("points");
        if (!pts || pts->t != J::ARR) return text_result("points est nécessaire.", true);
        std::vector<std::array<double, 3>> a, bb;
        std::string missing;
        for (auto& e : pts->a) {
            std::string nm = e->str("name");
            auto r = e->get("robot");
            if (!r || r->t != J::ARR || r->a.size() != 3) continue;
            auto it = g_marks.find(nm);
            if (it == g_marks.end()) { missing += " " + nm; continue; }
            a.push_back({it->second[0], it->second[1], it->second[2]});
            bb.push_back({r->a[0]->n, r->a[1]->n, r->a[2]->n});
        }
        if (!missing.empty()) return text_result("Points pas encore marqués :" + missing, true);
        float M[16], rms;
        if (!horn(a, bb, M, rms)) return text_result("Il faut au moins 3 points.", true);
        memcpy(g_robot, M, sizeof M); g_robot_rms = rms; g_robot_ok = true;
        save_robot();
        snprintf(b, sizeof b, "Repère robot calé sur %zu points, erreur RMS %.1f cm%s.", a.size(), rms * 100, rms > 0.03f ? " (élevée : refais les points en visant mieux)" : "");
        return text_result(b);
    }
    if (name == "convert_point") {
        auto p = args.get("point");
        if (!p || p->t != J::ARR || p->a.size() != 3) return text_result("point [x, y, z] est nécessaire.", true);
        if (!g_robot_ok) return text_result("Repère robot pas encore calé.", true);
        float in[3] = {(float)p->a[0]->n, (float)p->a[1]->n, (float)p->a[2]->n}, out[3];
        if (args.str("from") == "robot") { float inv[16]; rigid_inv(g_robot, inv); apply(inv, in, out); return text_result(num3(out) + " (world)"); }
        apply(g_robot, in, out);
        return text_result(num3(out) + " (robot)");
    }
    if (name == "place_marker") {
        auto p = args.get("point");
        std::string nm = args.str("name");
        if (!p || p->t != J::ARR || p->a.size() != 3 || nm.empty()) return text_result("name et point [x, y, z] sont nécessaires.", true);
        float w[3] = {(float)p->a[0]->n, (float)p->a[1]->n, (float)p->a[2]->n};
        if (robot) { float inv[16], o[3]; rigid_inv(g_robot, inv); apply(inv, w, o); memcpy(w, o, sizeof w); }
        std::string col = args.str("color", "vert");
        float c[3] = {0.2f, 1, 0.3f};
        if (col == "rouge") { c[0] = 1; c[1] = 0.2f; c[2] = 0.2f; } else if (col == "orange") { c[0] = 1; c[1] = 0.6f; c[2] = 0.1f; }
        else if (col == "bleu") { c[0] = 0.3f; c[1] = 0.6f; c[2] = 1; } else if (col == "blanc") { c[0] = c[1] = c[2] = 1; }
        bool found = false;
        for (auto& m : g_markers) if (m.name == nm) { memcpy(m.p, w, sizeof w); memcpy(m.color, c, sizeof c); found = true; }
        if (!found) { Marker m; m.name = nm; memcpy(m.p, w, sizeof w); memcpy(m.color, c, sizeof c); g_markers.push_back(m); }
        return text_result("Repère « " + nm + " » en " + num3(w) + " (world).");
    }
    if (name == "clear_markers") {
        std::string nm = args.str("name");
        if (nm.empty()) g_markers.clear();
        else g_markers.erase(std::remove_if(g_markers.begin(), g_markers.end(), [&](const Marker& m) { return m.name == nm; }), g_markers.end());
        return text_result("C'est fait.");
    }
    return text_result("Outil inconnu : " + name, true);
}

/** One JSON-RPC message -> its response ("" for a notification). */
inline std::string handle(const J& req) {
    const J* id = req.get("id");
    std::string method = req.str("method");
    if (!id) return "";  // notifications (initialized, cancelled)
    std::string ids = dump(*id), result;
    if (method == "initialize") {
        std::string ver = "2025-06-18";
        if (auto p = req.get("params")) ver = p->str("protocolVersion", ver);
        result = "{\"protocolVersion\":" + esc(ver) + ",\"capabilities\":{\"tools\":{\"listChanged\":false}},\"serverInfo\":{\"name\":\"myviture-scene\",\"version\":\"1.0\"},"
                 "\"instructions\":" + esc("Scène des lunettes VITURE : pièce (sol, murs, tables, zone marchable), porteur, humain virtuel, repères virtuels, calage du repère de la cellule robot. Coordonnées en mètres ; world = repère des lunettes (y vers le haut), robot = repère de la cellule une fois calé.") + "}";
    } else if (method == "ping") result = "{}";
    else if (method == "tools/list") {
        result = "{\"tools\":[";
        for (size_t i = 0; i < sizeof TOOLS / sizeof TOOLS[0]; i++) {
            if (i) result += ",";
            result += std::string("{\"name\":") + esc(TOOLS[i].name) + ",\"description\":" + esc(TOOLS[i].description) + ",\"inputSchema\":" + TOOLS[i].schema + "}";
        }
        result += "]}";
    } else if (method == "tools/call") {
        const J* p = req.get("params");
        J empty; empty.t = J::OBJ;
        const J* a = p ? p->get("arguments") : nullptr;
        g_calls++;
        result = call_tool(p ? p->str("name") : "", a && a->t == J::OBJ ? *a : empty);
    } else return "{\"jsonrpc\":\"2.0\",\"id\":" + ids + ",\"error\":{\"code\":-32601,\"message\":\"method not found\"}}";
    return "{\"jsonrpc\":\"2.0\",\"id\":" + ids + ",\"result\":" + result + "}";
}

inline void serve(int port) {
    WSADATA wd; WSAStartup(MAKEWORD(2, 2), &wd);
    SOCKET ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons((u_short)port); inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    BOOL one = TRUE; setsockopt(ls, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char*)&one, sizeof one);
    if (bind(ls, (sockaddr*)&a, sizeof a) || listen(ls, 8)) { std::lock_guard<std::mutex> l(g_mu); g_status = "port " + std::to_string(port) + " pris"; closesocket(ls); return; }
    { std::lock_guard<std::mutex> l(g_mu); g_status = "http://127.0.0.1:" + std::to_string(port) + "/mcp"; }
    printf("MCP: http://127.0.0.1:%d/mcp\n", port);
    while (!g_stop) {
        fd_set fs; FD_ZERO(&fs); FD_SET(ls, &fs);
        timeval tv{0, 200000};
        if (select(0, &fs, nullptr, nullptr, &tv) <= 0) continue;
        SOCKET cs = accept(ls, nullptr, nullptr);
        if (cs == INVALID_SOCKET) continue;
        DWORD to = 5000; setsockopt(cs, SOL_SOCKET, SO_RCVTIMEO, (const char*)&to, sizeof to);
        // Keep-alive: requests on the same connection until it closes.
        std::string buf;
        while (!g_stop) {
            size_t hend;
            while ((hend = buf.find("\r\n\r\n")) == std::string::npos) {
                char tmp[8192]; int n = recv(cs, tmp, sizeof tmp, 0);
                if (n <= 0) { hend = std::string::npos; break; }
                buf.append(tmp, n);
                if (buf.size() > (1 << 20)) break;
            }
            if (hend == std::string::npos) break;
            std::string head = buf.substr(0, hend), lower = head;
            for (auto& ch : lower) ch = (char)tolower((unsigned char)ch);
            size_t cl = 0, k = lower.find("content-length:");
            if (k != std::string::npos) cl = strtoul(lower.c_str() + k + 15, nullptr, 10);
            if (cl > (1 << 20)) break;
            while (buf.size() < hend + 4 + cl) { char tmp[8192]; int n = recv(cs, tmp, sizeof tmp, 0); if (n <= 0) break; buf.append(tmp, n); }
            if (buf.size() < hend + 4 + cl) break;
            std::string body = buf.substr(hend + 4, cl);
            buf.erase(0, hend + 4 + cl);
            // Browsers cannot reach it from a page (no CORS headers; Origin, if any, must be local).
            bool origin_ok = true;
            size_t ok_ = lower.find("\r\norigin:");
            if (ok_ != std::string::npos) { std::string o = lower.substr(ok_ + 9, lower.find("\r\n", ok_ + 2) - ok_ - 9); origin_ok = o.find("://127.0.0.1") != std::string::npos || o.find("://localhost") != std::string::npos; }
            std::string status = "200 OK", out;
            if (!origin_ok) status = "403 Forbidden";
            else if (head.rfind("POST", 0) != 0) status = "405 Method Not Allowed";
            else {
                Parser ps{body.data(), body.data() + body.size()};
                JP req = ps.value();
                if (ps.bad) { out = "{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{\"code\":-32700,\"message\":\"parse error\"}}"; }
                else if (req->t == J::ARR) {
                    std::string all;
                    for (auto& r : req->a) { std::string one = handle(*r); if (!one.empty()) all += (all.empty() ? "" : ",") + one; }
                    out = all.empty() ? "" : "[" + all + "]";
                } else out = handle(*req);
                if (out.empty()) status = "202 Accepted";
            }
            std::string resp = "HTTP/1.1 " + status + "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(out.size()) + "\r\nConnection: keep-alive\r\n\r\n" + out;
            send(cs, resp.data(), (int)resp.size(), 0);
        }
        closesocket(cs);
    }
    closesocket(ls);
}

inline void start(int port, const std::string& robot_path) {
    g_robot_path = robot_path;
    load_robot();
    g_thread = std::thread(serve, port);
}
inline void stop() { g_stop = true; if (g_thread.joinable()) g_thread.join(); }

}  // namespace mcp
