# Design 01: Core (Engine, Model, Session)

Status: agreed 2026-11-01; implemented in `src/engine.rs`, `src/model/mod.rs` and `src/session.rs` (tests in
`tests/core.rs`) and on every surface. Open: typed errors (design 06; `anyhow` until then), `Submission` handles,
`Send + Sync` (decision 0005), the Stream over `Session` instead of `GenTex`. `api::Engine` is the CPU convenience
over the Core; the byte-path plan session is `PlanSession`.

## Engine: the device

```rust
pub struct Engine { /* Gpu */ }

impl Engine {
    /// Create a device the engine owns (adapter selection, features, limits per `EngineOptions`).
    pub fn new(opts: &EngineOptions) -> Result<Engine, Error>;
    /// Run on a device the host already has: a game's or a player's wgpu device, a WebGPU device of the page.
    /// The engine checks the features it needs (shader-f16 or falls back to f32; timestamps and cooperative
    /// matrices are optional) and never reconfigures the device.
    pub fn with_device(device: wgpu::Device, queue: wgpu::Queue, info: wgpu::AdapterInfo, opts: &EngineOptions)
        -> Result<Engine, Error>;
    pub fn info(&self) -> EngineInfo;           // adapter, backend, class (tc-f16 / portable-f16 / portable-f32), features
    pub fn report(&self) -> EngineReport;       // compiled kernels, memory, autotune table size (design 05)
    pub fn warm_up(&self, kernels: Kernels);    // compile a kernel set now (design 04: Firefox compiles 0.1-2 s each)
}
```

`EngineOptions`: backend, tensor cores, autotune, adapter, precision, tune cache, compact memory and the DXC path
(design 04); the reference is generated from the declarations (design 05). The device holds its caches in `Rc`
today; `Send + Sync` is decision 0005.

## Model: weights and graph

```rust
pub struct Model { /* Blob + ops, weights uploaded lazily per session */ }

impl Model {
    pub fn from_dir(path: &Path) -> Result<Model, Error>;                      // graph.json, manifest, weights
    pub fn from_bytes(graph: &[u8], manifest: &[u8], weights: Vec<u8>) -> Result<Model, Error>; // browser, embedded
    pub fn info(&self) -> ModelInfo;   // name, train step, task (interpolation / extrapolation), k range, render sampling, params
}
```

A `Model` is CPU data; sessions upload what they use. Several models may live beside one engine (v6p for interpolation
and v7x for extrapolation in the desktop insert mode), and a model may be reloaded without touching the device.
Capabilities come from the manifest `meta` block (design 08) and gate the options a session may ask for.

## Session: the plans for one frame size

```rust
pub struct Session { /* plan_pair, plan_mid, slots, fast render path, autotune choices */ }

pub struct SessionOptions {
    pub width: u32, pub height: u32,
    pub internal: Internal,              // Auto (480-line rung above 960 px) | Frame | Size(w, h)
    pub k: u32,                          // fine iterations 1..3
    pub repair: u32,                     // 0, 1, 2
    pub render: RenderSampling,          // Model | Bilinear | Bicubic
    pub static_protection: Option<StaticProtection>,   // { frames: 8, tolerance: 4 }
    pub fast_render: bool, pub linear_t: bool,
    pub keep_maps: bool,                 // keep the repair stage's maps (the Stream's cadence needs them)
    pub tune: Option<TuneTable>,         // persisted autotune choices for this size
}

impl Session {
    pub fn new(engine: &Engine, model: &Model, opts: &SessionOptions) -> Result<Session, Error>;
    pub fn options(&self) -> &SessionOptions;
    pub fn report(&self) -> SessionReport;  // plan sizes, kernels chosen, measured pair / mid ms when available

    // slots the engine owns (rgba8unorm textures the host writes / reads)
    pub fn add_input(&mut self) -> InputSlot;
    pub fn add_output(&mut self) -> OutputSlot;
    pub fn input_texture(&self, s: InputSlot) -> &wgpu::Texture;
    pub fn output_texture(&self, s: OutputSlot) -> &wgpu::Texture;
    pub fn write_rgba(&self, s: InputSlot, rgba: &[u8]);               // CPU hosts
    pub fn write_mask(&self, mask: &[u8]);                             // protection_mask: pixels never interpolated (phase 6)
    pub fn read_rgba(&self, s: OutputSlot) -> Readback<Vec<u8>>;       // async readback (design 02: Readback)

    // the two halves of the network
    pub fn pair(&self, a: InputSlot, b: InputSlot) -> Submission;      // motion for interval (a, b)
    pub fn mid(&self, t: f32, out: OutputSlot) -> Submission;          // phase t of the last pair into `out`
    pub fn generate(&self, a: InputSlot, b: InputSlot, t: f32, out: OutputSlot) -> Submission; // both, for one-shot use

    pub fn pair_timed(&self, a: InputSlot, b: InputSlot) -> Option<Timed>;  // GPU ms via timestamp queries, if the device has them
    pub fn mid_timed(&self, t: f32, out: OutputSlot) -> Option<Timed>;
}
```

- `Submission` is a lightweight handle (submission index) so a host can `await engine.done(sub)` or pass it to the
  Pacer's cost samples; it is not required to be awaited.
- `linear_t` with `fast_render`: `pair` also runs the t-dependent stage at t = 0.5; each `mid` is one render step.
  `mid` with another `t` scales the motion linearly (today's behaviour, measured neutral on 60 fps video).
- `Internal::Auto` is the current `default_internal`. The option documents the quality cost of other choices.
- User textures: the first version keeps engine-owned slots (what every host uses today). `pair_with(&TextureView,
  &TextureView)` on host textures is a later addition behind the same plan machinery; the design leaves room (the
  pair / mid steps are already cached per slot view).
- Resizing: a new `Session`; `Engine` keeps the compiled kernels and the autotune table, so a rebuild is tens of ms.

## Precision, classes, tuning

Unchanged from today and surfaced in `EngineInfo.class`: `tc-f16` (cooperative matrices), `portable-f16`,
`portable-f32` (devices without shader-f16). The autotune table is per class and persists through
`EngineOptions::tune_cache` (native) or `TuneTable` text the host stores (browser: `engine.autotune(model, options)`
resolves with it, the `tune` session option takes it back).

## What the Core does not do

No timestamps, no clocks, no decision about which interval or phase to generate (Stream), no display (Pacer), no
scene cuts or held frames (Stream). The Core is deterministic: the same slots and the same `t` give the same pixels.

## Decisions

1. `Engine` and `Session` are `Send + Sync` (`Arc` plus interior locks around the arena and the caches; wasm stays
   single threaded).
2. `Model` keeps a per-engine weight cache, so two sessions of one size share uploaded weights.
3. User-provided textures in `pair` / `mid` are open; the first version keeps engine slots, which is what every host
   uses today.
