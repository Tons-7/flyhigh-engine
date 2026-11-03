# Rust API

The crate is `flyhigh-engine` (`crates/flyhigh-engine/`). The public API has four layers: the Core
(`Engine`, `Model`, `Session`: two frames in, one frame at phase `t` out), the Stream (frames with timestamps in,
intervals out), the Pacer (what to present on this display tick and what to generate for the coming ones) and the
hosts' own glue. Enter at the level you need. All options and their defaults: [options.md](options.md).

## Core: `Engine`, `Model`, `Session`

```rust
use flyhigh_engine::{Engine, EngineOptions, Model, Session, SessionOptions};

let engine = Engine::new(&EngineOptions::default())?;          // the device, once
let model = Model::from_dir(std::path::Path::new("models"))?;  // weights and graph, CPU data
let mut s = Session::new(&engine, &model, &SessionOptions::new(1920, 1080))?;   // the plans for one size

let (a, b) = (s.add_input(), s.add_input());                  // rgba8 textures the session owns
let out = s.add_output();
s.write_rgba(a, &frame0)?;                                    // or write into s.input_texture(a) yourself
s.write_rgba(b, &frame1)?;
s.pair(a, b)?;                                                // the motion of the interval: the expensive half
s.mid(0.5, out)?;                                             // phase 0.5 into the output slot: the cheap half
let rgba = s.read_rgba(out)?.block(engine.gpu())?;            // or present s.output_texture(out)
```

- `Engine::new(&EngineOptions)` creates a device (adapter, features, limits per the options);
  `Engine::with_device(instance, adapter, device, queue, &EngineOptions)` runs the engine on a device the application
  already has and never reconfigures it (shader-f16 or the f32 path, cooperative matrices and timestamp queries when
  the device was created with them). `engine.info()` names the device, its kernel class (`tc-f16`,
  `portable-f16`, `portable-f32`) and, on DX12, the HLSL compiler wgpu loaded (`shader_compiler`: `dxc`, or `fxc`
  without `dxcompiler.dll`, see `EngineOptions::dxc_path`); `engine.report()` counts compiled kernels, autotune
  entries and tensor bytes.
  `engine.gpu()` is the wgpu context for the host's own surfaces and copies.
- `Model::from_dir(dir)` / `Model::from_bytes(graph, manifest, weights)` load and validate a checkpoint
  (`model.info()`: name, training step, graph size, render sampler, format version). Several models may live beside
  one engine.
- `SessionOptions::new(w, h)` is the defaults; fields: `internal` (`Auto`: the 480-line rung above 960 px; `Frame`;
  `Size(w, h)`), `k` (1 to 3), `repair` (0 to 2), `render` (`Model`, `Bilinear`, `Bicubic`), `static_protection`
  (`Some(StaticProtection { frames, tolerance })`), `fast_render`, `linear_t`, `keep_maps`, `tune`.
  `validate()` names the field and the reason; `Session::new` validates.
- `Session`: `add_input` / `add_output`, `input_texture` / `output_texture` (and views), `write_rgba`, `write_mask`
  (the protection mask of `protection_mask`, one byte per pixel), `read_rgba`
  (a `Readback<Vec<u8>>`: native `block(gpu)`, browser `finish().await`), `pair(a, b)`, `mid(t, out)`,
  `generate(a, b, t, out)`, `pair_timed` / `mid_timed` (GPU ms from timestamp queries), `warm_up` (compile every
  kernel before the first real frame), `report()`. For a host that keeps its frames in its own ring, `pair_views` /
  `mid_views` take `(id, &TextureView)` pairs; the view behind an id must never change. `add_input_texture(texture)`
  makes a texture the host owns an input slot (the frame size, rgba8 or bgra8, sampleable); on Windows
  `interop::d3d11_shared_texture(&engine, handle, &SharedTexture::rgba8(w, h))` imports a D3D11 shared texture into
  the Vulkan device without a copy for exactly that.

## Stream

```rust
use flyhigh_engine::stream::{Stream, StreamOptions, Mode, Generated};

let mut stream = Stream::open(&engine, &model, &SessionOptions::new(w, h), &StreamOptions::default())?;
// per source frame, with its timestamp (media or capture time, milliseconds):
let slot = stream.next_input(|s| still_shown(s)).unwrap();      // the ring slot to write the frame into
write(stream.input_texture(slot));
let kind = stream.classify(engine.gpu(), previous, slot).map(|r| r.block(engine.gpu()).unwrap().kind);  // motion / cut / hold
let p = stream.push(engine.gpu(), slot, pts_ms)?;               // runs the pair; p.interval = what can be generated now
if p.stats_pending { stream.resolve_stats(engine.gpu())?; }    // video + cadence
match stream.generate(engine.gpu(), 0.5, out)? { Generated::Written => present(out), Generated::SourceIsEnough => present(p.interval.unwrap().base) }
```

`StreamOptions`: `mode` (`Realtime`, `Video` with one frame of lookahead), `cadence` (anime on twos, video mode),
`scene_cut` (`Some(SceneCut { threshold })`, default 0.08), `held_frames` (`Some(HeldFrames { mean, max })`, default
off), `inputs` / `outputs` created up front. `push_timed` and `generate_timed` do the same work under timestamp
queries and return the `TimedRun` to await (`block(gpu)` natively) for the GPU milliseconds the Pacer's
`cost_sample` takes. `stream.report()` counts pushes, pairs, mids, classifications. Details:
[ARCHITECTURE.md](../ARCHITECTURE.md).

Camera warp for games: `stream.mouse(gpu, t_ms, dx, dy)` per mouse event (on the clock of `push`'s timestamps:
arrival times in host ms), `stream.set_mouse_captured(bool)`, and at present time
`stream.warp(gpu, Shown::Input(slot) | Shown::Output(slot), out, now_ms)?`: true when output slot `out` holds the
turned frame. The pieces are public for other hosts: `warp::CameraWarp` (GPU measurement and render on any textures)
and `warp::InputCalibrator` (pure logic) ([integration guide](integration.md#input-latency-the-camera-warp)).

## Pacer

```rust
use flyhigh_engine::pacer::{Pacer, PacerOptions, SourceFrame, Frame};

let mut pacer = Pacer::new(&PacerOptions::default())?;          // Auto
// per display tick (vsync, rAF), in the host's milliseconds:
let tick = pacer.display_tick(now);
if let Some(p) = tick.present { show(p.frame); }               // Frame::Source(slot) | Frame::Output(slot)
for r in tick.generate {                                       // mids wanted for the coming ticks
    match stream.generate(engine.gpu(), r.t, out)? { Generated::Written => pacer.generated(now, r.id, Frame::Output(out)), _ => pacer.skipped(r.id) }
}
// per source frame, once its kind is known: the requests it returns go out before the pair changes
let flush = pacer.source_frame(now, SourceFrame { id: slot, pts_ms, kind, motion, skipped: 0 });
```

`pacer.cost_sample(now, Stage::Mid, ms)` feeds measured GPU times, `mode_change` opens the settle window after a
fullscreen change, `reset` forgets the history on a seek, `report()` says what was decided, measured and dropped and
why; `record(host)` makes the pacer record every call from then on as a `trace v1` (`trace()`, `stop_recording()`)
that `pacer::trace::replay` reproduces bit for bit. Every heuristic is an option of `PacerOptions` with the default in [ARCHITECTURE.md](../ARCHITECTURE.md). `pacer::trace`
records and replays sessions (`flyhigh-dev replay`). `examples/player` is the whole loop in one file.

## CPU convenience: `api::Engine` and `api::Generator`

For hosts that move frames through the CPU:

```rust
use flyhigh_engine::api::{Engine, Generator, Mode, Options};

let engine = Engine::load(std::path::Path::new("models"))?;        // GPU and model in one
let mut gen = Generator::new(&engine, Options::new(1920, 1080))?;
for frame in source_frames {                                       // RGBA8, width * height * 4 bytes, display order
    for out in gen.push(&engine, &frame, 2)? { present(&out.rgba, out.phase, out.real); }
}
```

`Options` mirrors `SessionOptions` plus `mode` and `cadence`; `push` returns the frames to display in order (realtime:
the in-betweens of (previous, this), then this frame; video: the interval two frames back after the third push). It
uploads and reads back every frame; GPU hosts use the Core and the Stream instead.

## Lower level

`Gpu`, `gen_tex::GenTex` (the generator under the Stream), `PlanSession` (byte frames, one fixed t), `SessionTex`
(ping-pong textures), `SessionExt` (the plans under `Session`) and `build_network_ex` are public for tools and
experiments. Their network options travel in `NetOpts`. They are not a stable interface.
