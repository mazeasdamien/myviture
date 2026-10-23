// Voice commands: the glasses' microphone, cut into utterances by their loudness, each one
// transcribed by Orange's Whisper (French), then read as a command:
//   « scan perceuse »          an anchor named "perceuse" on what the wearer looks at
//   « où est la perceuse ? »   the human goes to it and points at it
//   « viens »                  the human comes in front of the wearer
// Only what is loud enough for a while is sent (about 0.006 $ a minute of speech).

#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <mmsystem.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "orange.hpp"

#pragma comment(lib, "winmm.lib")

namespace voice {

struct Command { std::string op, label, heard; };  // op: "scan", "show", "come"

inline std::atomic<bool> g_on{true}, g_stop{false};
inline std::mutex g_mu;
inline std::deque<Command> g_cmds;
inline std::string g_status = "micro pas encore ouvert", g_heard, g_mic;
inline std::atomic<float> g_level{0};  // the microphone's loudness (0..1), for the meter
inline std::atomic<int> g_sending{0};   // transcriptions on their way

inline void set_status(const std::string& s) { std::lock_guard<std::mutex> l(g_mu); g_status = s; }

/** Lowercase ASCII, French accents and punctuation folded away, single spaces. */
inline std::string fold(const std::string& in) {
    static const struct { const char* from; char to; } MAP[] = {
        {"\xC3\xA0", 'a'}, {"\xC3\xA2", 'a'}, {"\xC3\xA4", 'a'}, {"\xC3\xA7", 'c'}, {"\xC3\xA8", 'e'}, {"\xC3\xA9", 'e'}, {"\xC3\xAA", 'e'}, {"\xC3\xAB", 'e'},
        {"\xC3\xAE", 'i'}, {"\xC3\xAF", 'i'}, {"\xC3\xB4", 'o'}, {"\xC3\xB6", 'o'}, {"\xC3\xB9", 'u'}, {"\xC3\xBB", 'u'}, {"\xC3\xBC", 'u'},
        {"\xC3\x80", 'a'}, {"\xC3\x89", 'e'}, {"\xC3\x88", 'e'}, {"\xC3\x8A", 'e'}, {"\xC3\x87", 'c'}, {"\xC3\x94", 'o'}};
    std::string out;
    for (size_t i = 0; i < in.size();) {
        bool done = false;
        for (auto& m : MAP) if (in.compare(i, 2, m.from) == 0) { out += m.to; i += 2; done = true; break; }
        if (done) continue;
        unsigned char ch = (unsigned char)in[i++];
        if (std::isalnum(ch)) out += (char)std::tolower(ch);
        else if (!out.empty() && out.back() != ' ') out += ' ';
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

/** "la perceuse rouge" -> "perceuse rouge": the leading articles off. */
inline std::string strip_articles(std::string s) {
    static const char* ART[] = {"le ", "la ", "les ", "l ", "un ", "une ", "des ", "du ", "de la ", "de ", "ce ", "cette ", "cet ", "mon ", "ma ", "mes "};
    for (bool again = true; again;) {
        again = false;
        for (const char* a : ART) if (s.rfind(a, 0) == 0) { s = s.substr(strlen(a)); again = true; }
    }
    return s;
}

/** The command in a transcription, if any. */
inline bool parse(const std::string& heard, Command& c) {
    std::string t = " " + fold(heard) + " ";
    c.heard = heard;
    // « scan X » (Whisper writes it scan, scanne, scanner, skan...)
    // Any word starting so: scan, scans, scanne, scanner, skan, ancre, ancrer...
    for (size_t p = 0; p != std::string::npos && p + 1 < t.size(); p = t.find(' ', p + 1)) {
        size_t w = p + 1, e = t.find(' ', w);
        std::string word = t.substr(w, e - w);
        if (word.rfind("scan", 0) != 0 && word.rfind("skan", 0) != 0 && word.rfind("ancr", 0) != 0) continue;
        c.op = "scan"; c.label = strip_articles(e == std::string::npos ? "" : t.substr(e + 1));
        while (!c.label.empty() && c.label.back() == ' ') c.label.pop_back();
        return true;
    }
    // « où est X », « montre-moi X », « il est où X », « trouve X »
    static const char* WHERE[] = {" ou est ce que ", " ou se trouve ", " ou sont ", " ou est ", " montre moi ", " montre ", " trouve ", " cherche "};
    for (const char* k : WHERE) {
        size_t p = t.find(k);
        if (p == std::string::npos) continue;
        c.op = "show"; c.label = strip_articles(t.substr(p + strlen(k)));
        while (!c.label.empty() && c.label.back() == ' ') c.label.pop_back();
        if (!c.label.empty()) return true;
    }
    // « la perceuse, elle est où ? »: the name before it
    for (const char* k : {" il est ou ", " elle est ou ", " est ou "}) {
        size_t p = t.find(k);
        if (p == std::string::npos) continue;
        c.op = "show"; c.label = strip_articles(fold(t.substr(0, p)));
        if (!c.label.empty()) return true;
    }
    if (t.find(" viens ") != std::string::npos || t.find(" venez ") != std::string::npos) { c.op = "come"; return true; }
    return false;
}

/** The "text" of an OpenAI-shaped transcription answer (\uXXXX escapes to UTF-8). */
inline std::string json_text(const std::string& r) {
    size_t p = r.find("\"text\"");
    if (p == std::string::npos) return "";
    p = r.find('"', r.find(':', p) + 1);
    if (p == std::string::npos) return "";
    std::string out;
    for (size_t i = p + 1; i < r.size() && r[i] != '"'; i++) {
        if (r[i] != '\\' || i + 1 >= r.size()) { out += r[i]; continue; }
        char e = r[++i];
        if (e == 'u' && i + 4 < r.size()) {
            unsigned cp = (unsigned)strtoul(r.substr(i + 1, 4).c_str(), nullptr, 16); i += 4;
            if (cp < 0x80) out += (char)cp;
            else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
            else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
        } else out += e == 'n' ? ' ' : e;
    }
    return out;
}

/** 16 kHz mono 16-bit samples as a WAV file. */
inline std::string wav(const std::vector<int16_t>& pcm) {
    auto u32 = [](std::string& s, uint32_t v) { s.append((const char*)&v, 4); };
    auto u16 = [](std::string& s, uint16_t v) { s.append((const char*)&v, 2); };
    std::string w = "RIFF";
    u32(w, 36 + (uint32_t)pcm.size() * 2); w += "WAVEfmt ";
    u32(w, 16); u16(w, 1); u16(w, 1); u32(w, 16000); u32(w, 32000); u16(w, 2); u16(w, 16);
    w += "data"; u32(w, (uint32_t)pcm.size() * 2);
    w.append((const char*)pcm.data(), pcm.size() * 2);
    return w;
}

/** One utterance to Whisper; its text, or "" (status says why). */
inline std::string transcribe(std::vector<int16_t> pcm) {
    const std::string B = "----myviture7f3a91";
    // The glasses' microphone is quiet: the utterance brought to 70 % of full scale (20x at most).
    int peak = 1;
    for (int16_t v : pcm) peak = std::max(peak, std::abs((int)v));
    float gain = std::min(20.0f, 0.7f * 32767 / peak);
    for (auto& v : pcm) v = (int16_t)std::clamp((int)std::lround(v * gain), -32767, 32767);
    // The last one kept next to the exe (voice_last.wav), to hear what Whisper got.
    {
        char exe[MAX_PATH]; GetModuleFileNameA(nullptr, exe, MAX_PATH);
        std::string path = exe; path = path.substr(0, path.find_last_of('\\') + 1) + "voice_last.wav";
        std::string w = wav(pcm);
        if (FILE* f = fopen(path.c_str(), "wb")) { fwrite(w.data(), 1, w.size(), f); fclose(f); }
    }
    const char* PROMPT = "Scan perceuse. Où est la perceuse ? Viens.";
    // Orange's firewall reads the body: now and then audio bytes look like an attack (403).
    // Then sent again with every sample's lowest bit changed (inaudible), 3 times at most.
    for (int attempt = 0; attempt < 4; attempt++) {
        if (attempt) { uint32_t r = 2463534242u * attempt; for (auto& v : pcm) { r ^= r << 13; r ^= r >> 17; r ^= r << 5; v ^= (int16_t)(r & 1); } }
        std::string body;
        body += "--" + B + "\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nopenai/whisper-1\r\n";
        body += "--" + B + "\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nfr\r\n";
        // The words to expect: Whisper then writes "scan", "où est" (echoed back alone: dropped below).
        body += "--" + B + "\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\n" + PROMPT + "\r\n";
        body += "--" + B + "\r\nContent-Disposition: form-data; name=\"file\"; filename=\"voix.wav\"\r\nContent-Type: audio/wav\r\n\r\n";
        body += wav(pcm) + "\r\n--" + B + "--\r\n";
        std::string resp;
        int st = orange::post("audio/transcriptions", body, orange::Kind::transcription, &resp, "multipart/form-data; boundary=" + B);
        if (st == 403) continue;
        if (st != 200) { set_status(st == 0 ? "Orange injoignable" : "Orange : HTTP " + std::to_string(st)); return ""; }
        set_status("j'écoute");
        std::string text = json_text(resp), f = fold(text);
        // What Whisper says on noise: subtitles credits, a lone "merci".
        if (f.find("amara") != std::string::npos || f.find("sous titres") != std::string::npos || f == "merci" || f == "allez" || f.empty() || f == fold(PROMPT)) return "";
        return text;
    }
    set_status("Orange : refusé par son pare-feu (403), 4 fois");
    return "";
}

/** The capture thread: the VITURE microphone (else the default one), 16 kHz mono. */
inline void run() {
    if (orange::g_gateway.empty() || orange::g_secret.empty()) { set_status(".env.local introuvable : pas de commandes vocales"); return; }
    UINT dev = WAVE_MAPPER;
    for (UINT i = 0; i < waveInGetNumDevs(); i++) {
        WAVEINCAPSW caps{};
        if (waveInGetDevCapsW(i, &caps, sizeof caps) == MMSYSERR_NOERROR && wcsstr(caps.szPname, L"VITURE")) {
            dev = i;
            std::lock_guard<std::mutex> l(g_mu);
            char nm[64] = {};
            WideCharToMultiByte(CP_UTF8, 0, caps.szPname, -1, nm, sizeof nm - 1, nullptr, nullptr);
            g_mic = nm;
            break;
        }
    }
    WAVEFORMATEX fmt{}; fmt.wFormatTag = WAVE_FORMAT_PCM; fmt.nChannels = 1; fmt.nSamplesPerSec = 16000; fmt.wBitsPerSample = 16;
    fmt.nBlockAlign = 2; fmt.nAvgBytesPerSec = 32000;
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    HWAVEIN in = nullptr;
    if (waveInOpen(&in, dev, &fmt, (DWORD_PTR)ev, 0, CALLBACK_EVENT) != MMSYSERR_NOERROR) { set_status("micro introuvable"); return; }
    constexpr int BLOCK = 800;  // 50 ms
    std::vector<int16_t> bufs[6];
    WAVEHDR hdr[6] = {};
    for (int i = 0; i < 6; i++) {
        bufs[i].resize(BLOCK);
        hdr[i].lpData = (LPSTR)bufs[i].data(); hdr[i].dwBufferLength = BLOCK * 2;
        waveInPrepareHeader(in, &hdr[i], sizeof hdr[i]); waveInAddBuffer(in, &hdr[i], sizeof hdr[i]);
    }
    waveInStart(in);
    {
        std::lock_guard<std::mutex> l(g_mu);
        g_status = g_mic.empty() ? "j'écoute (micro par défaut : celui des lunettes est absent)" : "j'écoute";
    }
    // Speech: louder than 3 times the room's noise (followed slowly) for 150 ms; the
    // utterance ends after 700 ms quiet, 8 s at most; 300 ms kept before it starts.
    float noise = 300;
    std::deque<std::vector<int16_t>> pre;
    std::vector<int16_t> utt;
    int loud = 0, quiet = 0;
    bool speaking = false;
    while (!g_stop) {
        if (WaitForSingleObject(ev, 200) != WAIT_OBJECT_0) continue;
        for (int i = 0; i < 6; i++) {
            if (!(hdr[i].dwFlags & WHDR_DONE)) continue;
            std::vector<int16_t> blk(bufs[i].begin(), bufs[i].begin() + hdr[i].dwBytesRecorded / 2);
            waveInAddBuffer(in, &hdr[i], sizeof hdr[i]);
            if (blk.empty()) continue;
            double e = 0;
            for (int16_t v : blk) e += (double)v * v;
            float rms = (float)std::sqrt(e / blk.size());
            g_level = std::min(1.0f, rms / 3000.0f);
            if (!g_on) { speaking = false; utt.clear(); pre.clear(); continue; }
            bool is_loud = rms > std::max(3 * noise, 400.0f);
            if (!speaking) noise += (rms - noise) * (rms < noise ? 0.2f : 0.01f);
            if (!speaking) {
                pre.push_back(blk);
                if (pre.size() > 10) pre.pop_front();  // 500 ms before: the "s" of "scan" is quiet
                loud = is_loud ? loud + 1 : 0;
                if (loud >= 3) { speaking = true; quiet = 0; utt.clear(); for (auto& b : pre) utt.insert(utt.end(), b.begin(), b.end()); pre.clear(); }
                continue;
            }
            utt.insert(utt.end(), blk.begin(), blk.end());
            quiet = is_loud ? 0 : quiet + 1;
            if (quiet < 14 && utt.size() < 16000 * 8) continue;
            speaking = false; loud = 0;
            if (utt.size() < 16000 * 6 / 10 || g_sending > 1) { utt.clear(); continue; }  // under 0.6 s: a noise
            g_sending++;
            set_status("transcription…");
            std::thread([pcm = std::move(utt)] {
                std::string text = transcribe(pcm);
                if (!text.empty()) {
                    Command c;
                    std::lock_guard<std::mutex> l(g_mu);
                    g_heard = text;
                    if (parse(text, c)) g_cmds.push_back(c);
                }
                g_sending--;
            }).detach();
            utt.clear();
        }
    }
    waveInStop(in); waveInReset(in);
    for (int i = 0; i < 6; i++) waveInUnprepareHeader(in, &hdr[i], sizeof hdr[i]);
    waveInClose(in);
    while (g_sending > 0) Sleep(20);
}

inline void start() { std::thread(run).detach(); }

}  // namespace voice
