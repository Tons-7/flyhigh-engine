/* A C host of the engine: the device, the model, a session for one frame size, two synthetic frames in, the in-between
 * out, and the reports as JSON. Build with the CMakeLists.txt next to this file against the built library
 * (cargo build --release -p flyhigh-capi); run from the repository root so "models" resolves, or pass the model
 * directory as the first argument. */
#include "flyhigh.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 256
#define H 192

/* a soft pattern shifted dx pixels to the right */
static void frame(float dx, uint8_t* v) {
    for (int y = 0; y < H; y++) {
        float w = y * 0.05f;
        for (int x = 0; x < W; x++) {
            float u = (x - dx) * 0.07f;
            uint8_t* p = v + (y * W + x) * 4;
            p[0] = (uint8_t)(128.0f + 100.0f * sinf(u));
            p[1] = (uint8_t)(128.0f + 100.0f * cosf(u * 0.7f + w));
            p[2] = (uint8_t)(128.0f + 90.0f * sinf(u + w));
            p[3] = 255;
        }
    }
}

static double mae(const uint8_t* a, const uint8_t* b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += fabs((double)a[i] - (double)b[i]);
    return s / (double)n;
}

static int fail(const char* what) {
    fprintf(stderr, "%s: %s\n", what, fh_last_error());
    return 1;
}

int main(int argc, char** argv) {
    const char* models = argc > 1 ? argv[1] : "models";
    char json[4096];

    FhEngineOptions eo = fh_engine_options_default();
    FhEngine* engine = fh_engine_create(&eo);
    if (!engine) return fail("fh_engine_create");
    if (fh_engine_info(engine, json, sizeof json) > 0) printf("engine: %s\n", json);

    FhModel* model = fh_model_load(models);
    if (!model) return fail("fh_model_load");
    if (fh_model_info(model, json, sizeof json) > 0) printf("model: %s\n", json);

    FhSessionOptions so = fh_session_options_default(W, H);
    FhSession* s = fh_session_create(engine, model, &so);
    if (!s) return fail("fh_session_create");

    int a = fh_session_add_input(s), b = fh_session_add_input(s), out = fh_session_add_output(s);
    size_t n = (size_t)W * H * 4;
    uint8_t* f0 = malloc(n); uint8_t* f1 = malloc(n); uint8_t* mid = malloc(n); uint8_t* got = malloc(n);
    frame(0.0f, f0); frame(8.0f, f1); frame(4.0f, mid);
    if (fh_session_write_rgba(s, a, f0, n) < 0 || fh_session_write_rgba(s, b, f1, n) < 0) return fail("write");
    if (fh_session_pair(s, a, b) < 0) return fail("fh_session_pair");
    if (fh_session_mid(s, 0.5f, out) < 0) return fail("fh_session_mid");
    if (fh_session_read_rgba(s, out, got, n) < 0) return fail("fh_session_read_rgba");

    double to_mid = mae(got, mid, n), to_f0 = mae(got, f0, n), to_f1 = mae(got, f1, n);
    printf("in-between: MAE to the true middle %.2f, to f0 %.2f, to f1 %.2f\n", to_mid, to_f0, to_f1);
    double ms;
    if (fh_session_pair_timed(s, a, b, &ms) == 1) printf("pair %.3f ms", ms);
    if (fh_session_mid_timed(s, 0.5f, out, &ms) == 1) printf(", generated frame %.3f ms", ms);
    printf("\n");
    if (fh_session_report_json(s, json, sizeof json) > 0) printf("session: %s\n", json);

    int ok = to_mid < to_f0 && to_mid < to_f1 && to_mid < 6.0;
    fh_session_destroy(s);
    fh_model_destroy(model);
    fh_engine_destroy(engine);
    free(f0); free(f1); free(mid); free(got);
    puts(ok ? "ok" : "the in-between is not between its neighbours");
    return ok ? 0 : 2;
}
