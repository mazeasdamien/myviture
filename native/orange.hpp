// The lab's Orange gateway (LiteLLM behind the orange-gateway worker), from the native
// app: called directly with the worker's secret, read from ..\.env.local and never shown.
// Costs come from the response headers (x-litellm-response-cost, -key-spend,
// -key-max-budget); the key only reports its spend on LLM calls, so "refresh" makes a
// one-token call (~0.0005 c$).

#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

#pragma comment(lib, "winhttp.lib")

namespace orange {

struct Costs {
    std::mutex mu;
    double session = 0, last = 0, vision = 0, transcription = 0, speech = 0;
    long calls = 0;
    double key_spend = NAN, key_budget = NAN;
    std::string status = "non interrogée";
};
inline Costs g_costs;
inline std::string g_gateway, g_secret;
inline std::atomic<bool> g_busy{false};

/** ORANGE_GATEWAY and ORANGE_PROXY_SECRET from the repository's .env.local. */
inline bool load_config(const std::string& env_path) {
    std::ifstream f(env_path);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        if (k == "ORANGE_GATEWAY") g_gateway = v;
        else if (k == "ORANGE_PROXY_SECRET") g_secret = v;
    }
    return !g_gateway.empty() && !g_secret.empty();
}

inline std::wstring widen(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n ? n - 1 : 0, L'\0');
    if (n) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    return w;
}

inline double header_number(HINTERNET req, const wchar_t* name) {
    wchar_t buf[64]; DWORD len = sizeof buf;
    if (!WinHttpQueryHeaders(req, WINHTTP_QUERY_CUSTOM, name, buf, &len, WINHTTP_NO_HEADER_INDEX)) return NAN;
    return _wtof(buf);
}

enum class Kind { vision, transcription, speech };

/** POST to /orange/v1/<route> (JSON unless told); accounts the cost. Returns the HTTP status (0: no connection). */
inline int post(const std::string& route, const std::string& body, Kind kind, std::string* response = nullptr, const std::string& content_type = "application/json") {
    URL_COMPONENTS uc{}; uc.dwStructSize = sizeof uc;
    wchar_t host[256], path[1024];
    uc.lpszHostName = host; uc.dwHostNameLength = 256; uc.lpszUrlPath = path; uc.dwUrlPathLength = 1024;
    std::wstring url = widen(g_gateway + "/orange/v1/" + route);
    if (!WinHttpCrackUrl(url.c_str(), 0, 0, &uc)) return 0;
    HINTERNET s = WinHttpOpen(L"myViture/1.0", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    HINTERNET c = s ? WinHttpConnect(s, host, uc.nPort, 0) : nullptr;
    HINTERNET r = c ? WinHttpOpenRequest(c, L"POST", path, nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0) : nullptr;
    int status = 0;
    if (r) {
        std::wstring headers = L"content-type: " + widen(content_type) + L"\r\naccept: application/json\r\nx-orange-auth: " + widen(g_secret) + L"\r\n";
        if (WinHttpSendRequest(r, headers.c_str(), (DWORD)-1, (LPVOID)body.data(), (DWORD)body.size(), (DWORD)body.size(), 0) && WinHttpReceiveResponse(r, nullptr)) {
            DWORD code = 0, len = sizeof code;
            WinHttpQueryHeaders(r, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &code, &len, WINHTTP_NO_HEADER_INDEX);
            status = (int)code;
            double cost = header_number(r, L"x-litellm-response-cost");
            double spend = header_number(r, L"x-litellm-key-spend"), budget = header_number(r, L"x-litellm-key-max-budget");
            {
                std::lock_guard<std::mutex> l(g_costs.mu);
                g_costs.calls++;
                if (std::isfinite(cost)) {
                    g_costs.session += cost; g_costs.last = cost;
                    (kind == Kind::vision ? g_costs.vision : kind == Kind::transcription ? g_costs.transcription : g_costs.speech) += cost;
                }
                if (std::isfinite(spend)) g_costs.key_spend = spend;
                if (std::isfinite(budget)) g_costs.key_budget = budget;
            }
            if (response) {
                DWORD avail = 0;
                while (WinHttpQueryDataAvailable(r, &avail) && avail) {
                    std::string chunk(avail, '\0'); DWORD got = 0;
                    WinHttpReadData(r, chunk.data(), avail, &got);
                    response->append(chunk, 0, got);
                }
            }
        }
    }
    if (r) WinHttpCloseHandle(r);
    if (c) WinHttpCloseHandle(c);
    if (s) WinHttpCloseHandle(s);
    return status;
}

/** One-token call, in the background: refreshes the key's spend and budget. */
inline void refresh_spend() {
    if (g_busy.exchange(true)) return;
    std::thread([] {
        int st = g_gateway.empty() ? -1 : post("chat/completions", R"({"model":"vertex_ai/gemini-2.5-flash-lite","messages":[{"role":"user","content":"ok"}],"max_tokens":1})", Kind::vision);
        {
            std::lock_guard<std::mutex> l(g_costs.mu);
            g_costs.status = st == -1 ? ".env.local introuvable (ORANGE_GATEWAY / ORANGE_PROXY_SECRET)" : st == 200 ? "" : st == 0 ? "passerelle injoignable" : "passerelle : HTTP " + std::to_string(st);
        }
        g_busy = false;
    }).detach();
}

}  // namespace orange
