# Architecture

How the engine is put together: the layers, the module map, the network, the static plans, the kernels and the two
content-aware stages. The public API of each layer is designed in [docs/design](docs/design/README.md); this document
is about the code behind it.

## The layers

The engine is the whole frame-generation product; a host is a thin shell around it ([design 00](docs/design/00-overview.md)).

| Layer | Modules | Takes | Gives |
|---|---|---|---|
| Core | `engine.rs`, `model/`, `session.rs` | a device, a model, two frames and a phase `t` | the frame at `t`: `pair` once per interval, `mid` once per generated frame |
| Stream | `stream/` | source frames with timestamps | generatable intervals, their kind (motion, scene cut, held frame), the cadence statistics |
| Pacer | `pacer/` | display ticks, source frames, measured GPU costs | what to present on this tick and what to generate for the coming ones; no clock of its own, replayable from a trace |
| Host adapters | `examples/web`, `examples/player`, the desktop overlay | the platform's frames and ticks | calls into the three layers above |

Each layer uses only the public API of the one below it. The CPU convenience `api.rs` (frames as byte buffers, the
display order resolved for the host) sits on the Core and the Stream.

## One engine crate, every target

The engine is one Rust library crate (`crates/flyhigh-engine`); the C ABI crate, the wasm module crate and the
developer CLI wrap it. Natively it runs on wgpu's Vulkan, DX12 or Metal backend; compiled to `wasm32-unknown-unknown`
the same code runs on the browser's WebGPU through wgpu's web backend. The GPU work is in the WGSL files of `shaders/`,
included at compile time, so there is one copy of every kernel for all targets.

| Module (`crates/flyhigh-engine/src/`) | Role |
|---|---|
| `engine.rs` | `Engine`: the device (own, or the host's through `with_device`), the compiled kernels, the autotune table, `DeviceInfo` and the report |
| `model/` | `Model`: the weight blob and manifest (`blob`), the loader checks, helpers over the operation graph (`graph`) |
| `session.rs` | `Session` and `SessionOptions`: the plans for one frame size, own slots or the host's views, `pair` / `mid`, the protection mask, `Readback` results |
| `stream/` | `Stream` and `StreamOptions` over `gen_tex`: timestamps and intervals, the pair classifier (`classify`, scene cuts and held frames on a 48 x 27 grid), the report |
| `pacer/` | `Pacer` and `PacerOptions`: display and source estimators (`display`, `source`), the output rate policy (`rate`), the exact-grid planner (`plan`), the presentation queue (`queue`), the AIMD controller (`control`), the cost model (`costs`), `trace v1` recording and replay (`trace`) |
| `options.rs` | `EngineOptions`, `NetOpts`, the defaults; `OptionInfo` and `groups()` behind the `Options` derive (the reference is generated from them) |
| `gpu/` | `Gpu`: device setup and adapter choice (`device`), the DX12 shader compiler discovery (`dxc`), shader compilation and the f32 rewrite (`pipeline`), buffers and the per-plan arena (`alloc`), recording and submitting a plan (`exec`), GPU timing (`timing`), readback (`readback`) |
| `ops/` | the kernels as plan steps: `step`, `tensor`, `params`, `elementwise`, `interp`, `network`, `render`, `texture` |
| `ops/conv/` | convolutions: the kernels (`kernels`), the candidates (`candidates`), per-shape autotune (`autotune`), the persisted table (`tune_cache`) |
| `net/` | graph to plan (`build`), the plan sessions underneath the Core (`session` on byte frames, `session_ext` on textures with the fast render, `session_tex` ping-pong) |
| `gen_tex.rs` | the texture-path generator with modes (realtime, video with lookahead, cadence) that the Stream wraps |
| `gmhs.rs` | the repair stage and the cadence statistics and composite |
| `interop.rs` | Windows: a D3D11 shared texture into the Vulkan device without a copy |
| `api.rs` | the CPU convenience: `api::Engine`, `api::Generator`, frames as bytes |

| Other crates and directories | Role |
|---|---|
| `crates/flyhigh-options` | the `Options` derive (proc macro): doc comment, default and `#[option]` attributes become `OptionInfo` |
| `crates/flyhigh-capi` | the C ABI: `fh_engine_*`, `fh_model_*`, `fh_session_*`, `fh_stream_*`, `fh_pacer_*`, `fh_generator_*`; the header `include/flyhigh.h` |
| `crates/flyhigh-web` | the wasm-bindgen module: the classes `FhEngine`, `FhModel`, `FhSession`, `FhStream`, `FhPacer` and `fh_pacer_replay` |
| `crates/flyhigh-cli` | `flyhigh-dev`: self-tests, parity, quality regression, timings, trace replay, the options reference; one module per subcommand |
| `examples/player`, `examples/web`, `examples/c` | the reference native host, the reference web host, a C host |
| `bindings/` | Python (ctypes) and C++ (header only) over the C ABI |
| `shaders/` | the WGSL kernels |
| `bench/` | per-GPU baselines and the regression rule |
| `fixtures/` | kernel fixtures and the full-graph golden dumps the self-tests compare against |

## The network

FlyHighFG (776 k parameters, f16) takes two frames A and B and a phase t and predicts the frame at t. The graph is
data (`models/graph.json`, see [model-format.md](docs/model-format.md)); the engine walks it. Its stages:

1. **Encoders** on the frames downscaled 4x: features at 1/8 and 1/16 of the input.
2. **Coarse flow** at 1/16 from the features and a luma cost volume (radius 3): bidirectional flow and visibility.
3. **Fine iterations** at 1/8, `k` times with shared weights: warp the features and images with the current flow,
   a cost volume (radius 2), refine flow, visibility and a hidden state.
4. *(pair / mid boundary: everything above depends only on A and B, everything below on t)*
5. **FiLM(t)**: a tiny MLP turns t into a modulation of the hidden state; flow and visibility are moved to time t.
6. **Convex upsample** to 1/4, a **refine** head at 1/4 that predicts a residual.
7. **Render** at full resolution: warp A and B with the upsampled flow (bicubic or bilinear sampling), blend by
   visibility, add the residual.

Frames enter in BGR order (the training order), padded to multiples of 32 by edge replication.

**Internal size.** Steps 1 to 6 run on the frames downscaled to the internal size; only the render runs at the frame
size, with the flow rescaled. The cost of the network is then nearly independent of the output resolution.

## Static plans

When a session is built, `net::build` walks the graph once and records a plan: a list of compute dispatches with every
buffer, uniform and bind group created up front. A frame replays the plan into one command buffer, in one compute
pass, with one submit. Building also simplifies the graph:

- slice / scale / neg are views (a channel offset and a scale carried by the reader), with no dispatch;
- conv + add becomes a conv with a fused residual; blend weights + 2 multiplies + add is one `blendmix`; concat is
  one `chcopy` dispatch;
- outputs nobody reads (oracle-only heads) are not computed;
- tensors share buffers through an arena: a buffer is reused once the last reader of its tensor is recorded (about
  half the memory of a plan without reuse). Kernels therefore never assume zeroed memory.

**Pair and mid plans.** The plan splits at the pair / mid boundary. The pair plan runs once per source frame; the mid
plan runs per generated frame. t lives in a few uniform buffers (the `mul_t` scales and the FiLM vector) that are
rewritten before a mid, so any t works without a rebuild. Pipelines compile on first use; `warm_up` compiles a
session's kernels before the first real frame.

**Golden plans.** `tests/golden_plans.rs` dumps the plans of a fixed set of sessions (steps, bindings, parameters) and
compares them with the files in `tests/golden/`; a deliberate change re-records them. The plan structure is therefore
a regression guard of its own, beside the pixel parity.

## Conv kernels

Convs are 3x3 implicit GEMMs, f16 in and out, f32 accumulation, bias + PReLU (+ residual) fused. Candidates:

- tensor-core kernels (WGSL cooperative matrix, native Vulkan on NVIDIA): `direct` (input pre-shifted by `xshift3`,
  no gather) and `tc` (gathering);
- portable FMA kernels, everywhere including the browser: `tiled` (4x4 register block, several tile shapes),
  `tiled8` (8x4), `tiled88` (8x8), each optionally split along K (`splitk_reduce`) when a layer has too few output
  tiles to fill the GPU.

With autotune, each conv shape of the plan is timed with every applicable candidate (GPU timestamps, min of 15) and
the fastest is kept; the choices persist through the tune cache. Without it, a fixed routing applies. A plan at k = 1
is about 80 dispatches, about 100 with repair. GPUs without `shader-f16` run the same kernels rewritten to f32
(`gpu::pipeline`), the `portable-f32` class.

## Repair (GMHS)

The network matches at 1/8 to 1/16 of its input, where a fence or a window grid under a pan is ambiguous: the true
motion and the motion plus or minus one period look the same. At the end of the pair plan the repair stage fits a
robust affine motion to the flow, scores the network's flow and the affine hypothesis photometrically on a coarse
grid, and replaces the flow where the hypothesis explains the frames clearly better, jumps by a real distance and
the texture is not flat. Level 2 drops the jump requirement (better on drawn content, worse on natural video).

## Static protection and the mask

Pixels unchanged for `frames` source pairs (a HUD, chat, burnt-in subtitles) are taken from the source frames instead
of being warped with the motion behind them: an age pass in the pair stage counts unchanged pairs per pixel, the
render avoids warp sources on protected pixels, and an apply pass after the render mixes them toward the sources'
value with a two-pair ramp. The host's protection mask (`protection_mask`, `write_mask`) enters the same age pass as
age 255, so it works with or without the automatic detection (`shaders/keep_static*.wgsl`).

## Cadence (video mode)

Anime often holds each drawing for two or more frames while the camera pans on every frame. Interpolating such a
pair moves the background smoothly but makes the character jump. In video mode the generator keeps three pair
contexts (previous, current, next) and per-pair statistics from the repair maps: how much of the frame moves against
the global motion (an element jump), how much changed at all, and whether the pair is a scene cut. From the recent
hold lengths it places each drawing on a uniform path and renders the element from the pair that contains its
transition, at a re-timed phase, composited over the background by a mask (or the whole frame on a still shot). The
real slot of an interval may then be re-synthesised too.

## Browser specifics

- WebGPU exposes no tensor cores and no build-time autotune (timing is asynchronous): the autotune runs explicitly
  and its table is persisted by the host.
- Per-frame objects are cached per slot: in the browser every uniform buffer and bind group is a garbage-collected JS
  object, and churning them every frame costs frame-time stability.
- The engine creates the `GPUDevice` (or adopts the page's); hosts present, sample and copy on the same device and use
  the slot textures directly.
- Firefox needs three host rules, all in the engine or its adapter: presents at least 7 ms apart, kernels warmed up
  before playback, a submit every millisecond while a readback is pending ([platforms](docs/platforms.md)).
