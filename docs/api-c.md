# C API

`crates/flyhigh-capi/` builds `flyhigh.dll` / `libflyhigh.so` / `libflyhigh.dylib` and a static library; the header
is `include/flyhigh.h`. C++, C#, Python and other FFI hosts call the same functions (P/Invoke, ctypes). The engine's
layers are opaque handles: `FhEngine` (the device), `FhModel`, `FhSession` (two frames in, one frame at phase `t`
out), `FhStream` (frames with timestamps in, intervals out), `FhPacer` (what to present on this tick, what to
generate) and `FhGenerator` (the CPU convenience: source frames in, the frames to display out, in order). Frames cross
the ABI as RGBA8 bytes (`width * height * 4`); GPU textures come in through the interop calls (`fh_session_import_d3d11`, `fh_stream_import_d3d11`). Single-threaded use per
handle; a failed call returns `NULL` or `-1` and `fh_last_error()` has the message. Options always start from the
`fh_*_options_default()` struct. All options and defaults: [options.md](options.md).

## Core

```c
#include "flyhigh.h"

FhEngineOptions eo = fh_engine_options_default();
FhEngine* engine = fh_engine_create(&eo);                       // the device, once; NULL = defaults
FhModel* model = fh_model_load("models");                       // graph.json, flyhighfg.json, flyhighfg.bin
FhSessionOptions so = fh_session_options_default(1920, 1080);   // internal auto, k 1, repair 1, the model's sampler
FhSession* s = fh_session_create(engine, model, &so);           // the plans for one size

int a = fh_session_add_input(s), b = fh_session_add_input(s), out = fh_session_add_output(s);
fh_session_write_rgba(s, a, frame0, 1920 * 1080 * 4);
fh_session_write_rgba(s, b, frame1, 1920 * 1080 * 4);
fh_session_pair(s, a, b);                                       // the motion of the interval: the expensive half
fh_session_mid(s, 0.5f, out);                                   // phase 0.5 into the output slot: the cheap half
fh_session_read_rgba(s, out, rgba, 1920 * 1080 * 4);           // waits for the GPU
```

- `fh_engine_create(&FhEngineOptions)`: backend, tensor cores, autotune, adapter selector, precision, tune cache,
  compact memory, DXC path, as `EngineOptions`. `fh_engine_info` writes the device as JSON (name, kernel class
  `tc-f16` / `portable-f16` / `portable-f32`, timestamp queries, and on DX12 `shader_compiler`: `dxc`, or `fxc`
  when `dxcompiler.dll` was not found); `fh_engine_report_json` adds kernels compiled, autotune entries and tensor
  bytes. Destroy sessions and streams before the engine.
- `fh_model_load(dir)` / `fh_model_from_bytes(graph, len, manifest, len, weights, len)` load and check a checkpoint
  (format version, tensor table); `fh_model_info` is the model as JSON. Several models may live beside one engine.
- `FhSessionOptions`: `width`, `height`, `internal_w` / `internal_h` (0, 0 = automatic: the 480-line rung above
  960 px), `k`, `repair`, `render` (0 model, 1 bilinear, 2 bicubic), `keep_static` with `static_frames` /
  `static_tolerance`, `fast_render`, `linear_t`, `keep_maps`, `protection_mask` (the mask comes through
  `fh_session_write_mask` / `fh_stream_write_mask`, one byte per pixel), `tune` (the autotune table text). An option out of range
  fails `fh_session_create` with the field named in `fh_last_error`.
- `fh_session_pair_timed` / `fh_session_mid_timed` give GPU milliseconds from timestamp queries (1 measured, 0 none
  on this device). `fh_session_warm_up` compiles every kernel before the first real frame. `fh_session_generate` is
  `pair` and `mid` in one call. `fh_session_report_json`: sizes, plan steps, slots.

## Pacer recording

`fh_pacer_record(pacer, "my host")` makes the pacer record every call from then on; `fh_pacer_trace_json` writes the
`trace v1` so far, the file `flyhigh-dev replay` takes and the one to attach to a bug report.

## Stream

```c
FhStreamOptions sto = fh_stream_options_default();              // realtime, scene cuts at 0.08, 4 + 4 slots
FhStream* st = fh_stream_create(engine, model, &so, &sto);
// per source frame, with its timestamp (media or capture time, milliseconds):
int slot = fh_stream_next_input(st, still_shown, n_still_shown);
fh_stream_write_rgba(st, engine, slot, frame, len);
FhClassification c; int kind = fh_stream_classify(st, engine, previous, slot, &c) == 1 ? c.kind : 0;   // motion / cut / hold
FhPush p; fh_stream_push(st, engine, slot, pts_ms, &p);          // runs the pair; p.ready = an interval can be generated
if (p.stats_pending) fh_stream_resolve_stats(st, engine);      // video + cadence
if (fh_stream_generate(st, engine, 0.5f, out) == 1) present_output(out); else present_source(p.base_slot);
```

`FhStreamOptions`: `mode` (0 realtime, 1 video with one frame of lookahead), `cadence`, `scene_cut` (the threshold;
0 = never), `held_frames` with `held_mean` / `held_max`, `inputs` / `outputs` created up front. `fh_stream_generate`
returns 1 written, 0 the source frame is enough (t = 0 without re-timing). `fh_stream_read_rgba`, `fh_stream_warm_up`,
`fh_stream_reset` (seek, decoder gap), `fh_stream_report_json`. Details: [ARCHITECTURE.md](../ARCHITECTURE.md).

Camera warp (games): `fh_stream_mouse(st, engine, t_ms, dx, dy)` per mouse event, on the clock of `pts` (arrival
times in host ms), `fh_stream_set_mouse_captured(st, 0 / 1)`, and at present time
`fh_stream_warp(st, engine, output, slot, out, now_ms)`: the frame you would present (input slot, or output slot
with `output` 1) turned by the mouse motion since, into output slot `out`; 1 = present `out`, 0 = present the frame
as it is ([integration guide](integration.md#input-latency-the-camera-warp)).

## Pacer (`fh_pacer_*`)

The engine's scheduler for a host with its own display loop ([ARCHITECTURE.md](../ARCHITECTURE.md)): pure logic, no clock,
every call takes `now` in the host's monotonic milliseconds.

```c
FhPacerOptions o = fh_pacer_options_default();          // mode 0 auto; 1 factor, 2 display rate, 3 target_fps
o.min_present_interval_ms = 0;                           // the host's present spacing limit, if it has one
FhPacer* pacer = fh_pacer_create(&o);

// per display tick (vsync):
FhTick tick; fh_pacer_tick(pacer, now_ms, &tick);
if (tick.has_present) present(tick.frame_kind == 0 ? source_slot(tick.frame_id) : output_slot(tick.frame_id));
FhRequest req[64]; int n = fh_pacer_requests(pacer, req, 64);
for (int i = 0; i < n; i++) {
    int out = next_output();
    if (fh_stream_generate(st, engine, req[i].t, out) == 1) fh_pacer_generated(pacer, now_ms, req[i].id, out);
    else fh_pacer_skipped(pacer, req[i].id);
}

// per source frame, once its kind (motion / cut / hold) is known:
FhSourceFrame f = { slot, pts_ms, kind, 1, motion, 0 };
n = fh_pacer_source_frame(pacer, now_ms, &f);           // requests of the previous pair: generate them first
fh_pacer_requests(pacer, req, 64); /* generate, report */ fh_stream_push(st, engine, slot, pts_ms, &p);
```

`fh_pacer_cost_sample` feeds measured GPU times (stage 0 pair, 1 mid, 2 present; 1 taken, 0 ignored inside the settle
window), `fh_pacer_mode_change` opens the settle window after a fullscreen change, `fh_pacer_reset` forgets the
history on a seek, `fh_pacer_report_json` writes the report (what was decided, measured and dropped, and why).
`fh_pacer_set_options` changes options live. The C surface decides exactly what the Rust one does:
`crates/flyhigh-capi/tests/pacer_parity.rs` replays a recorded host trace through both and compares the reports.

## Generator: the CPU convenience

```c
FhOptions go = fh_options_default(1920, 1080);
FhGenerator* g = fh_generator_create(engine, model, &go);
for (each source frame) {
    int n = fh_generator_push(g, rgba, 1920 * 1080 * 4, 2);    // factor 2 = one in-between per source frame
    float phase; int real; uint8_t out[1920 * 1080 * 4];
    while (fh_generator_pop(g, out, sizeof out, &phase, &real) == 1) present(out, phase, real);
}
fh_generator_destroy(g);
```

`FhOptions` mirrors the session options plus `mode` (0 realtime, 1 video) and `cadence`; `push` queues the frames to
display in order (realtime: the in-betweens of (previous, this), then this frame; video: the interval two frames back
after the third push, `fh_generator_warmup` says how many). It uploads and reads back every frame; GPU hosts use the
Core and the Stream instead.

## Zero-filled structs

A zero-filled `FhSessionOptions` or `FhOptions` is valid but turns repair off (its default is 1) and asks for `k` 0
(`FhOptions`: the default 1; `FhSessionOptions`: an error, `k` must be 1 to 3). Start from the defaults.

## Bindings

`bindings/python/flyhigh.py` (ctypes, standard library only) and `bindings/cpp/flyhigh.hpp` (header only, RAII) wrap
this ABI one to one; `examples/c/` is a C host with a `CMakeLists.txt`. See [bindings/README.md](../bindings/README.md).

## Tests

`crates/flyhigh-capi/tests/`: `defaults.rs` and `core.rs` check that the C defaults match the Rust ones and that every
option struct maps both ways without loss (no GPU); `core.rs` also generates an in-between through the Core and the
Stream from C on a GPU (`cargo test -p flyhigh-capi -- --ignored`); `pacer_parity.rs` replays a recorded host trace
through the C Pacer and the Rust one.
