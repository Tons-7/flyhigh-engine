# Design 02: Stream

Status: agreed 2026-11-01; implemented in `crates/flyhigh-engine/src/stream/` as a layer over `GenTex`, with the
hosts' pair classification moved into the engine. Open: the Core `Session` replaces `GenTex` underneath without
changing this API.

## What a Stream is

A Stream turns a sequence of source frames with timestamps into intervals that can be generated. It owns the input
ring and the output slots, remembers which slots hold which frames, runs the pair stage at the right time (at once in
realtime mode, one frame behind in video mode), classifies each pair, keeps the per-pair cadence statistics and
answers one question: for the interval I can show now, what does phase `t` look like.

```rust
pub struct StreamOptions {
    pub mode: Mode,                       // Realtime (no lookahead) | Video (one frame of lookahead)
    pub cadence: bool,                    // Video: re-time elements animated on 2s (anime); needs repair >= 1
    pub scene_cut: Option<SceneCut>,      // Some { threshold: 0.08 } by default: the interval is not interpolated
    pub held_frames: Option<HeldFrames>,  // None by default; Some { mean: 0.0033, max: 0.06 } for anime
    pub inputs: u32,                      // input slots created up front, 4
    pub outputs: u32,                     // output slots created up front, 4
}

impl Stream {
    pub fn new(gpu: &Gpu, blob: &Blob, ops: &[Value], tex: TexOptions, opts: &StreamOptions, tune: Option<&TuneTable>) -> Result<Stream>;
    pub fn next_input(&mut self, in_use: impl Fn(usize) -> bool) -> Option<usize>;  // the slot to write the next frame into
    pub fn classify(&mut self, gpu: &Gpu, previous: usize, slot: usize) -> Option<Readback<Classification>>;
    pub fn push(&mut self, gpu: &Gpu, slot: usize, pts: f64) -> Result<Push>;
    pub fn generate(&mut self, gpu: &Gpu, t: f32, out: usize) -> Result<Generated>;
    pub fn take_stats(&mut self) -> Option<StatsRun>;   // video + cadence: await, then set_stats (native: resolve_stats)
    pub fn real_replaced(&self) -> bool;                 // the cadence re-timed the interval's real slot: generate(0) instead of presenting base
    pub fn reset(&mut self);                             // seek, decoder gap: forget the frames, keep the plans
    pub fn report(&self) -> StreamReport;
    // slots: add_input / add_output, input_texture / output_texture (and views), inputs / outputs, warmup
}

pub struct Push { pub interval: Option<Interval>, pub stats_pending: bool }
pub struct Interval { pub start_pts: f64, pub end_pts: f64, pub base: usize }
pub enum Generated { Written, SourceIsEnough }
pub struct Classification { pub kind: IntervalKind /* Motion | Cut | Hold */, pub mean: f32, pub max: f32, pub black: bool }
```

`IntervalKind` is defined here and used by the Pacer (the layer above). Timestamps are the host's media or capture
time; the Stream never interprets wall time.

## Behaviour

- **Realtime.** `push` runs the pair (previous, new) and returns the interval at once. `generate(t)` renders phase t.
- **Video.** `push` runs the pair in the "next" context; the interval returned is the one that started two frames
  back (one frame of lookahead). With `cadence`, the statistics of the newest pair are read back asynchronously
  (`stats_pending`); the browser awaits `take_stats().finish()`, native hosts call `resolve_stats`. Three rotating
  pair contexts, as before.
- **Cadence** (video): the pure `cadence_retime` decision over the statistics window, unchanged.
- **Classification** (new in the engine, from the extension's dedup pass and the CLI's `skipcuts`): the two frames are
  sampled on a 48 x 27 grid (`classify.wgsl`); the mean per-channel difference above `scene_cut.threshold` is a cut,
  a mean below `held_frames.mean` with every cell below `held_frames.max` is a hold, anything else motion. The
  readback is asynchronous (three may be in flight; beyond that the pair counts as motion) and lands before the
  interval's first mid is due. `Classification::motion_units()` is the statistic the Pacer's motion-adaptive ceiling
  uses.
- **Order with the Pacer.** Classify the new frame against the previous one, then `pacer.source_frame` with the kind,
  generate the requests it returns, then `push`. The pair must not change before the previous interval's mids are out.
- **Held frames.** The Stream reports them; the Pacer shows the frame and generates nothing for the interval, so
  nothing is dropped silently and the report counts them. Stretching the interval was not needed.

## Readbacks

`gpu::readback::Readback<T>`: a one-shot future over a mapped staging buffer with a decode closure and a release
hook (the classifier's staging ring). Native hosts `block(gpu)`, the browser awaits `finish()`. The Firefox nudge
(`queue.submit([])` while a readback is pending) stays in the web host adapter (design 04).

## Costs the Stream measures

`push_timed` and `generate_timed` run the real pair and mid under timestamp queries and hand back a `TimedRun` the
host awaits (`block` natively, a promise in the browser): no second pass, the numbers are the frames' own, and they
feed the Pacer's cost samples (its probe requests exist for this). `pair_cost_ms` and `frame_cost_ms` re-run the
stages for a measurement, native only.

## Report

`StreamReport`: mode, cadence, frame and internal size, k, repair, slots, frames held, and counters (pushes, pairs,
mids, retimed, source-enough answers, classifications, classifier busy skips, resets).

## Decisions

1. Scene-cut threshold default 0.08 (20 / 255), what the CLI uses; the extension's 0.118 is a host preset.
2. A held frame is reported, never dropped silently; the Pacer counts it.
3. Extrapolation (v7x) is a later Stream mode: intervals look ahead (`t` > 1), no lookahead latency.
4. Session options stay `TexOptions` until the Core `Session` lands; `Stream::new` takes them and overrides the mode
   fields from `StreamOptions`.
