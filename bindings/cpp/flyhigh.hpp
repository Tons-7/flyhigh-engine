// FlyHigh Engine for C++: RAII over the C ABI (include/flyhigh.h), header only, C++17.
//
//   fh::Engine engine;                                     // the device
//   fh::Model model("models");                             // graph.json, flyhighfg.json, flyhighfg.bin
//   fh::Session s(engine, model, fh::Session::options(1920, 1080));
//   auto a = s.addInput(), b = s.addInput(), out = s.addOutput();
//   s.write(a, frame0); s.write(b, frame1);
//   s.pair(a, b); s.mid(0.5f, out);
//   std::vector<uint8_t> rgba = s.read(out);
//
// Every failed call throws fh::Error with the engine's message. Handles are movable, not copyable; destroy order is
// the C++ scope order (sessions and streams before the engine they were created on).
#pragma once
#include "flyhigh.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fh {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace detail {
inline Error last(const char* what) { return Error(std::string(what) + ": " + fh_last_error()); }
inline int check(int rc, const char* what) {
    if (rc < 0) throw last(what);
    return rc;
}
template <typename F>
inline std::string json(F fn, const char* what) {
    std::string buf(1 << 16, '\0');
    for (;;) {
        int n = fn(buf.data(), buf.size());
        if (n >= 0) {
            buf.resize(static_cast<size_t>(n));
            return buf;
        }
        std::string msg = fh_last_error();
        if (msg.find("buffer too small") != std::string::npos && buf.size() < (1u << 26)) {
            buf.resize(buf.size() * 4);
            continue;
        }
        throw Error(std::string(what) + ": " + msg);
    }
}
}  // namespace detail

template <typename T, void (*Destroy)(T*)>
class Handle {
public:
    Handle() = default;
    explicit Handle(T* h) : h_(h) {}
    Handle(Handle&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
    Handle& operator=(Handle&& o) noexcept {
        if (this != &o) {
            reset();
            h_ = o.h_;
            o.h_ = nullptr;
        }
        return *this;
    }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    ~Handle() { reset(); }
    T* get() const { return h_; }
    explicit operator bool() const { return h_ != nullptr; }
    void reset() {
        if (h_) Destroy(h_);
        h_ = nullptr;
    }

private:
    T* h_ = nullptr;
};

/// The device; one per GPU.
class Engine {
public:
    static FhEngineOptions options() { return fh_engine_options_default(); }
    Engine() : Engine(options()) {}
    explicit Engine(const FhEngineOptions& o) : h_(fh_engine_create(&o)) {
        if (!h_) throw detail::last("fh_engine_create");
    }
    FhEngine* get() const { return h_.get(); }
    /// {"name","class","tensor_cores","timestamps","bgra_storage","shader_compiler"}
    std::string info() const {
        return detail::json([&](char* b, size_t n) { return fh_engine_info(h_.get(), b, n); }, "fh_engine_info");
    }
    std::string report() const {
        return detail::json([&](char* b, size_t n) { return fh_engine_report_json(h_.get(), b, n); }, "fh_engine_report_json");
    }

private:
    Handle<FhEngine, fh_engine_destroy> h_;
};

/// Weights and graph.
class Model {
public:
    explicit Model(const std::string& dir) : h_(fh_model_load(dir.c_str())) {
        if (!h_) throw detail::last("fh_model_load");
    }
    static Model fromBytes(const std::vector<uint8_t>& graph, const std::vector<uint8_t>& manifest,
                           const std::vector<uint8_t>& weights) {
        FhModel* m = fh_model_from_bytes(graph.data(), graph.size(), manifest.data(), manifest.size(), weights.data(),
                                         weights.size());
        if (!m) throw detail::last("fh_model_from_bytes");
        return Model(m);
    }
    FhModel* get() const { return h_.get(); }
    std::string info() const {
        return detail::json([&](char* b, size_t n) { return fh_model_info(h_.get(), b, n); }, "fh_model_info");
    }

private:
    explicit Model(FhModel* m) : h_(m) {}
    Handle<FhModel, fh_model_destroy> h_;
};

/// The plans for one frame size: two frames in, one frame at phase t out.
class Session {
public:
    static FhSessionOptions options(uint32_t width, uint32_t height) { return fh_session_options_default(width, height); }
    Session(const Engine& e, const Model& m, const FhSessionOptions& o)
        : h_(fh_session_create(e.get(), m.get(), &o)), w_(o.width), hgt_(o.height) {
        if (!h_) throw detail::last("fh_session_create");
    }
    FhSession* get() const { return h_.get(); }
    uint32_t addInput() { return static_cast<uint32_t>(detail::check(fh_session_add_input(h_.get()), "fh_session_add_input")); }
    uint32_t addOutput() { return static_cast<uint32_t>(detail::check(fh_session_add_output(h_.get()), "fh_session_add_output")); }
    void write(uint32_t slot, const uint8_t* rgba, size_t len) {
        detail::check(fh_session_write_rgba(h_.get(), slot, rgba, len), "fh_session_write_rgba");
    }
    void write(uint32_t slot, const std::vector<uint8_t>& rgba) { write(slot, rgba.data(), rgba.size()); }
    /// The protection mask (protection_mask): width * height bytes, 0 interpolates; from the next pair.
    void writeMask(const uint8_t* mask, size_t len) { detail::check(fh_session_write_mask(h_.get(), mask, len), "fh_session_write_mask"); }
    void writeMask(const std::vector<uint8_t>& mask) { writeMask(mask.data(), mask.size()); }
    std::vector<uint8_t> read(uint32_t slot) const {
        std::vector<uint8_t> v(static_cast<size_t>(w_) * hgt_ * 4);
        detail::check(fh_session_read_rgba(h_.get(), slot, v.data(), v.size()), "fh_session_read_rgba");
        return v;
    }
    void pair(uint32_t a, uint32_t b) { detail::check(fh_session_pair(h_.get(), a, b), "fh_session_pair"); }
    void mid(float t, uint32_t out) { detail::check(fh_session_mid(h_.get(), t, out), "fh_session_mid"); }
    void generate(uint32_t a, uint32_t b, float t, uint32_t out) {
        detail::check(fh_session_generate(h_.get(), a, b, t, out), "fh_session_generate");
    }
    /// GPU milliseconds, or a negative number without timestamp queries.
    double pairTimed(uint32_t a, uint32_t b) {
        double ms = -1.0;
        return detail::check(fh_session_pair_timed(h_.get(), a, b, &ms), "fh_session_pair_timed") == 1 ? ms : -1.0;
    }
    double midTimed(float t, uint32_t out) {
        double ms = -1.0;
        return detail::check(fh_session_mid_timed(h_.get(), t, out, &ms), "fh_session_mid_timed") == 1 ? ms : -1.0;
    }
    void warmUp() { detail::check(fh_session_warm_up(h_.get()), "fh_session_warm_up"); }
    std::string report() const {
        return detail::json([&](char* b, size_t n) { return fh_session_report_json(h_.get(), b, n); }, "fh_session_report_json");
    }

private:
    Handle<FhSession, fh_session_destroy> h_;
    uint32_t w_, hgt_;
};

/// Frames with timestamps in, intervals out (ARCHITECTURE.md).
class Stream {
public:
    static FhStreamOptions options() { return fh_stream_options_default(); }
    Stream(const Engine& e, const Model& m, const FhSessionOptions& so, const FhStreamOptions& o)
        : e_(e.get()), h_(fh_stream_create(e.get(), m.get(), &so, &o)), w_(so.width), hgt_(so.height) {
        if (!h_) throw detail::last("fh_stream_create");
    }
    FhStream* get() const { return h_.get(); }
    uint32_t addInput() { return static_cast<uint32_t>(detail::check(fh_stream_add_input(h_.get(), e_), "fh_stream_add_input")); }
    uint32_t addOutput() { return static_cast<uint32_t>(detail::check(fh_stream_add_output(h_.get(), e_), "fh_stream_add_output")); }
    /// The slot to write the next frame into, or -1 when the ring is exhausted.
    int nextInput(const std::vector<uint32_t>& inUse = {}) {
        return detail::check(fh_stream_next_input(h_.get(), inUse.data(), inUse.size()), "fh_stream_next_input");
    }
    void write(uint32_t slot, const uint8_t* rgba, size_t len) {
        detail::check(fh_stream_write_rgba(h_.get(), e_, slot, rgba, len), "fh_stream_write_rgba");
    }
    void write(uint32_t slot, const std::vector<uint8_t>& rgba) { write(slot, rgba.data(), rgba.size()); }
    void writeMask(const uint8_t* mask, size_t len) { detail::check(fh_stream_write_mask(h_.get(), e_, mask, len), "fh_stream_write_mask"); }
    void writeMask(const std::vector<uint8_t>& mask) { writeMask(mask.data(), mask.size()); }
    /// true and `out` filled; false = take the pair as motion (no classifier, or every readback in flight).
    bool classify(uint32_t previous, uint32_t slot, FhClassification& out) {
        return detail::check(fh_stream_classify(h_.get(), e_, previous, slot, &out), "fh_stream_classify") == 1;
    }
    FhPush push(uint32_t slot, double pts) {
        FhPush p{};
        detail::check(fh_stream_push(h_.get(), e_, slot, pts, &p), "fh_stream_push");
        return p;
    }
    void resolveStats() { detail::check(fh_stream_resolve_stats(h_.get(), e_), "fh_stream_resolve_stats"); }
    bool realReplaced() const { return detail::check(fh_stream_real_replaced(h_.get()), "fh_stream_real_replaced") != 0; }
    /// true when the output slot holds the frame; false when the source frame is enough (t 0, no re-timing).
    bool generate(float t, uint32_t out) { return detail::check(fh_stream_generate(h_.get(), e_, t, out), "fh_stream_generate") == 1; }
    std::vector<uint8_t> read(uint32_t out) const {
        std::vector<uint8_t> v(static_cast<size_t>(w_) * hgt_ * 4);
        detail::check(fh_stream_read_rgba(h_.get(), e_, out, v.data(), v.size()), "fh_stream_read_rgba");
        return v;
    }
    void warmUp() { detail::check(fh_stream_warm_up(h_.get(), e_), "fh_stream_warm_up"); }
    void reset() { detail::check(fh_stream_reset(h_.get()), "fh_stream_reset"); }
    std::string report() const {
        return detail::json([&](char* b, size_t n) { return fh_stream_report_json(h_.get(), b, n); }, "fh_stream_report_json");
    }

private:
    const FhEngine* e_;
    Handle<FhStream, fh_stream_destroy> h_;
    uint32_t w_, hgt_;
};

/// What to present on this display tick and what to generate (ARCHITECTURE.md); milliseconds of the host's
/// monotonic clock.
class Pacer {
public:
    static FhPacerOptions options() { return fh_pacer_options_default(); }
    Pacer() : Pacer(options()) {}
    explicit Pacer(const FhPacerOptions& o) : h_(fh_pacer_create(&o)) {
        if (!h_) throw detail::last("fh_pacer_create");
    }
    FhPacer* get() const { return h_.get(); }
    void setOptions(double now, const FhPacerOptions& o) { detail::check(fh_pacer_set_options(h_.get(), now, &o), "fh_pacer_set_options"); }
    /// The tick, with the requests of this tick in `requests`.
    FhTick tick(double now, std::vector<FhRequest>& requests) {
        FhTick t{};
        detail::check(fh_pacer_tick(h_.get(), now, &t), "fh_pacer_tick");
        fetch(t.requests, requests);
        return t;
    }
    /// A source frame with its kind; the requests to generate before the pair changes come back in `requests`.
    void sourceFrame(double now, const FhSourceFrame& f, std::vector<FhRequest>& requests) {
        int n = detail::check(fh_pacer_source_frame(h_.get(), now, &f), "fh_pacer_source_frame");
        fetch(static_cast<uint32_t>(n), requests);
    }
    void generated(double now, uint64_t request, uint32_t output) {
        detail::check(fh_pacer_generated(h_.get(), now, request, output), "fh_pacer_generated");
    }
    void skipped(uint64_t request) { detail::check(fh_pacer_skipped(h_.get(), request), "fh_pacer_skipped"); }
    /// stage 0 pair, 1 mid, 2 present; false when ignored inside the settle window.
    bool costSample(double now, uint32_t stage, double ms) {
        return detail::check(fh_pacer_cost_sample(h_.get(), now, stage, ms), "fh_pacer_cost_sample") == 1;
    }
    void hostStall(double now, double ms) { detail::check(fh_pacer_host_stall(h_.get(), now, ms), "fh_pacer_host_stall"); }
    void modeChange(double now) { detail::check(fh_pacer_mode_change(h_.get(), now), "fh_pacer_mode_change"); }
    void reset(double now) { detail::check(fh_pacer_reset(h_.get(), now), "fh_pacer_reset"); }
    /// Records every call from now on as a trace v1; trace() reads it.
    void record(const std::string& host) { detail::check(fh_pacer_record(h_.get(), host.c_str()), "fh_pacer_record"); }
    std::string trace() const {
        return detail::json([&](char* b, size_t n) { return fh_pacer_trace_json(h_.get(), b, n); }, "fh_pacer_trace_json");
    }
    std::string report() const {
        return detail::json([&](char* b, size_t n) { return fh_pacer_report_json(h_.get(), b, n); }, "fh_pacer_report_json");
    }

private:
    void fetch(uint32_t n, std::vector<FhRequest>& out) {
        out.resize(n);
        if (n) detail::check(fh_pacer_requests(h_.get(), out.data(), out.size()), "fh_pacer_requests");
    }
    Handle<FhPacer, fh_pacer_destroy> h_;
};

}  // namespace fh
