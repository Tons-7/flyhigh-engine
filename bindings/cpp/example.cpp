// The C++ wrapper in use: an in-between through a Session, then a Pacer driven by synthetic ticks. Build next to the
// C example (see examples/c/CMakeLists.txt) or by hand:
//   g++ -std=c++17 -I crates/flyhigh-capi/include bindings/cpp/example.cpp -L target/release -lflyhigh -o flyhigh_cpp_example
#include "flyhigh.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

static const uint32_t W = 256, H = 192;

static std::vector<uint8_t> frame(float dx) {
    std::vector<uint8_t> v(static_cast<size_t>(W) * H * 4);
    for (uint32_t y = 0; y < H; y++) {
        float w = y * 0.05f;
        for (uint32_t x = 0; x < W; x++) {
            float u = (x - dx) * 0.07f;
            uint8_t* p = &v[(static_cast<size_t>(y) * W + x) * 4];
            p[0] = static_cast<uint8_t>(128.0f + 100.0f * std::sin(u));
            p[1] = static_cast<uint8_t>(128.0f + 100.0f * std::cos(u * 0.7f + w));
            p[2] = static_cast<uint8_t>(128.0f + 90.0f * std::sin(u + w));
            p[3] = 255;
        }
    }
    return v;
}

static double mae(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i++) s += std::fabs(double(a[i]) - double(b[i]));
    return s / double(a.size());
}

int main(int argc, char** argv) try {
    fh::Engine engine;
    fh::Model model(argc > 1 ? argv[1] : "models");
    std::printf("engine: %s\nmodel: %s\n", engine.info().c_str(), model.info().c_str());

    fh::Session s(engine, model, fh::Session::options(W, H));
    auto a = s.addInput(), b = s.addInput(), out = s.addOutput();
    s.write(a, frame(0.0f));
    s.write(b, frame(8.0f));
    s.pair(a, b);
    s.mid(0.5f, out);
    auto got = s.read(out);
    double toMid = mae(got, frame(4.0f)), toF0 = mae(got, frame(0.0f)), toF1 = mae(got, frame(8.0f));
    std::printf("in-between: MAE to the true middle %.2f, to f0 %.2f, to f1 %.2f\n", toMid, toF0, toF1);
    std::printf("pair %.3f ms, generated frame %.3f ms\n", s.pairTimed(a, b), s.midTimed(0.5f, out));

    // the Pacer alone: 120 Hz ticks and a 24 fps source, every request generated at once
    fh::Pacer pacer;
    std::vector<FhRequest> reqs;
    double t = 0.0, nextSrc = 5.0;
    uint32_t n = 0, presents = 0, drops = 0;
    while (t < 6000.0) {
        if (nextSrc <= t) {
            FhSourceFrame f{n % 8, n * 1000.0 / 24.0, 0, 1, 10.0, 0};
            pacer.sourceFrame(t, f, reqs);
            for (auto& r : reqs) pacer.generated(t, r.id, 0);
            n++;
            nextSrc += 1000.0 / 24.0;
        }
        FhTick tick = pacer.tick(t, reqs);
        presents += tick.has_present;
        drops += tick.dropped;
        for (auto& r : reqs) pacer.generated(t, r.id, static_cast<uint32_t>(r.id % 8));
        t += 1000.0 / 120.0;
    }
    std::printf("pacer: %u presents in 6 s, %u drops\n%s\n", presents, drops, pacer.report().substr(0, 200).c_str());
    bool ok = toMid < toF0 && toMid < toF1 && toMid < 6.0 && presents > 400 && drops == 0;
    std::puts(ok ? "ok" : "not ok");
    return ok ? 0 : 2;
} catch (const fh::Error& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
