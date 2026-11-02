# Design 03: Pacer

Status: agreed 2026-11-01; implemented in `crates/flyhigh-engine/src/pacer/` with the measured behaviours of the
extension's scheduler and Auto controller. Open: the game mode and the insert mode the desktop's vblank pacer needs.

## What the Pacer is

A pure scheduler. The host feeds events with its own clock; the Pacer answers what to present on this tick and what
to generate for the coming ones. It never calls the GPU and never reads a clock, so a recorded trace replays exactly.

```rust
pub struct Pacer { /* estimators, planner, queue, controller */ }

impl Pacer {
    pub fn new(opts: &PacerOptions) -> Result<Pacer, OptionsError>;
    pub fn set_options(&mut self, now: f64, opts: &PacerOptions) -> Result<(), OptionsError>; // live; a new mode restarts the grid

    // events (all times in milliseconds of the host's monotonic clock)
    pub fn display_tick(&mut self, now: f64) -> Tick;                       // a vsync / rAF callback: the heartbeat
    pub fn source_frame(&mut self, now: f64, frame: SourceFrame) -> Vec<Request>; // a source frame, its interval kind known
    pub fn generated(&mut self, now: f64, request: u64, frame: Frame);      // the host submitted a requested frame
    pub fn skipped(&mut self, request: u64);                                // the Stream said the source frame is enough
    pub fn cost_sample(&mut self, now: f64, stage: Stage, ms: f64);         // measured pair / mid / present GPU time
    pub fn host_stall(&mut self, now: f64, ms: f64);                        // the host's loop stood still (also inferred from tick gaps)
    pub fn mode_change(&mut self, now: f64);                                // fullscreen, resize: a settle window starts
    pub fn reset(&mut self, now: f64);                                      // seek, new source
    pub fn report(&self) -> PacerReport;
}

pub struct SourceFrame { pub id: u32, pub pts_ms: f64, pub kind: IntervalKind /* Motion | Cut | Hold */, pub motion: Option<f64>, pub skipped: u32 }
pub struct Request { pub id: u64, pub previous: u32, pub current: u32, pub t: f32, pub due_ms: f64, pub retime: bool, pub probe: bool }
pub struct Tick { pub present: Option<Present>, pub generate: Vec<Request>, pub factor: u32, pub dropped: u32 }
pub struct Present { pub frame: Frame /* Source(id) | Output(id) */, pub due_ms: f64, pub late_ms: f64, pub mid: bool, pub fallback: bool }
```

Frames are the host's own ids (slot indices): the Pacer never touches a texture.

## Inside, in the order a tick runs

1. **Display estimator** (`display.rs`). From tick intervals: the refresh floor (service interval, not the fastest
   sample), the nominal rate (snapped to 360/240/180/165/144/120/100/90/75/72/60/50 within 3 %), stability
   (coefficient of variation <= 0.12 over a 32-sample window), transitions in either direction (10 agreeing samples,
   a faster candidate survives 3 misses). Capacity = min(raw, nominal, host present limit). A tick gap above
   max(25 ms, 4 x floor) is a host stall.
2. **Source estimator** (`source.rs`). From source pts: the interval (trimmed mean over 32 samples, snapped to common
   video rates within 1.5 %, held through 3 % noise once nominal, a transition after 4 agreeing samples outside 12 %).
   The arrival-interval EMA, the unique-frame interval (holds and cuts excluded) and the PLL schedule clock
   (follows arrivals at 8 %, resyncs hard beyond 80 ms) that places presentations.
3. **Output rate policy** (`rate.rs`). Modes: `Factor(n)`, `Auto` (as many as the display and GPU allow, capped by
   motion), `DisplayHz`, `Target(fps)`; Auto with `fps_limit` is a strict-ceiling target. The 2x floor, display
   headroom 0.97, compute capacity from pair / mid / present costs within `compute_budget` of the source interval,
   the queue capacity, the clamp reason (display / gpu / runtime / minimum) and a warning text. Exact modes hold their
   plan until a different one has been proposed 16 frames in a row (`RateStabilizer`). When the GPU looks too slow
   for 2x, one real pair with tiny assumed costs every 2.5 s measures it (`probe`).
4. **Presentation planner** (`plan.rs`). Exact modes: target ticks on the exact output grid in media phase, carried
   across intervals; each tick becomes one presentation, an anchor (`previous` / `current`) or an `interpolate` at
   phase t; cuts and holds disable interpolation but keep the grid. Factor modes: mids at k / n of the interval and
   the source frame at its arrival plus the delay. The delay target (pair lookahead in exact modes + pair cost + 2 x
   mid cost + 25 ms + burst pad, floor 60 ms, max 180 ms) sets how far ahead frames are due and moves 2 ms per
   frame.
5. **Presentation queue** (`queue.rs`). Bounded (24), ordered by due time; one presentation per tick; the oldest
   entry inside the lateness contract (3 output intervals) is presented when the display can service the target,
   else the newest due. Entries before it are dropped and counted; entries due before the last presented one are
   stale. A requested mid that is not there in time shows its anchor instead (`fallback`).
6. **Auto controller** (`control.rs`, AIMD). Drop pressure is a leaky integrator (tau 300 ms); several frames due in
   one tick count as one event; drops within 20 ms after a host stall are not charged while stalls are rarer than 4
   per 2 s; pressure > 1.2 lowers the factor by one (by two above 3), not more often than every 500 ms; after 6 s
   with pressure < 0.15 the factor recovers one step. Tick strain (ticks 1.7 x slower than the floor) adds a
   provisional one-step penalty that lifts 500 ms after the strain ends. A settle window after a mode change exempts
   drops. The motion-adaptive ceiling (> 45 motion units: 2, > 28: 3, > 16: 4, else 6) is an option. Auto that
   cannot run by the budget probes one 2x pair at 2.5 / 5 / 10 / 20 s.
7. **Just-in-time requests.** A mid is requested `pair + 2 x (mid + present) + 8` ms before its slot, so present
   blits interleave with computes instead of the first mid waiting for the whole batch.
8. **Host present limit.** No two presents closer than `min_present_interval_ms` (Firefox 7 ms: its compositor
   stalls 40-90 ms below that, measured 2026-11-01); the display capacity is capped accordingly and a frame that
   would violate it waits for the next tick.

## Options (all with these defaults)

| Option | Default | Why you would change it |
|---|---|---|
| `mode` | `auto` | `{"factor": n}` for benchmarks and players that want exactly nx; `display_hz`; `{"target": fps}` |
| `max_factor` | 6 | cap Auto |
| `fps_limit` | none | Auto capped at a rate (strict, never raised to the 2x floor) |
| `lookahead` | false | the Stream's video mode: the generated interval is one frame back, its real slot may be re-timed |
| `limit_to_display` | true | false lets the GPU generate above the refresh rate (superseded, not dropped) |
| `fill_display` | false | Display Hz fills the whole panel rate instead of leaving the headroom |
| `display_headroom` | 0.97 | how much of the refresh rate the planner may use |
| `compute_budget` | 0.9 | fraction of the source interval the GPU may spend per interval |
| `delay_floor_ms`, `delay_max_ms` | 60, 180 | the presentation delay window (exact modes may grow to 2500) |
| `lateness_intervals` | 3 | how late a frame may still be shown |
| `queue_capacity` | 24 | pending presentations |
| `pressure_tau_ms` | 300 | drop pressure memory |
| `penalty_up`, `penalty_burst`, `penalty_min_gap_ms`, `penalty_max` | 1.2, 3, 500, 3 | AIMD thresholds |
| `recovery_after_ms` | 6000 | clean time before Auto recovers a step |
| `stall_window_ms`, `stall_rate_limit` | 20, 4 per 2 s | what counts as a host stall |
| `settle_ms` | 1200 | drops exempt after a mode change (the extension's measured fullscreen reflow) |
| `motion_adaptive` | true | fewer mids on fast scenes |
| `min_present_interval_ms` | 0 (the Firefox adapter sets 7) | the host's present limit |
| `playback_rate` | 1 | media seconds per wall second |
| `mid_cost_seed_ms`, `pair_cost_seed_ms` | 10, 4 | assumed GPU costs until the first measurement (scale the mid seed with resolution) |
| `gpu_timestamps` | true | false on hosts that cannot measure GPU time: no probes |

`PacerOptions::validate()` names the field and the reason; options serialise to JSON with every field optional.

## Traces and tests

`trace v1` (JSON, `pacer/trace.rs`): `{ "version": 1, "host": "...", "events": [...] }` with events `tick {t}`,
`source {t, id, pts, kind_of, motion, skipped}`, `cost {t, stage, ms}`, `stall {t, ms}`, `mode_change {t}`,
`reset {t}`. `trace::replay` runs a trace through a Pacer standing in for the host (every request generated at once,
every present on its tick) and returns the report plus a per-second table; `flyhigh-dev replay <trace.json>` prints
it. Floats round-trip exactly (`serde_json` with `float_roundtrip`), so a replay is bit for bit.

`tests/pacer.rs` builds the first regression set synthetically, shaped like the hosts' recordings:

- steady Chrome, 120 Hz and 24 fps: factor 4 (24 x 5 exceeds the 97 % headroom), 96 presents/s, zero drops, no
  late mids;
- Firefox, 60 Hz with a 70 ms compositor stall every 1.5 s and the 7 ms present limit: the drops after each stall
  exist and are stall-exempt, the penalty never moves, 44+ presents/s;
- a saturated loop (ticks at 30 ms for three seconds): the first step lands inside the first overloaded second,
  sustained overload keeps stepping, six clean seconds per step recover it fully;
- a fullscreen change with a 300 ms reflow gap: no penalty;
- Display Hz from 24 fps on 60 Hz: 58.2 Hz output (the headroom), zero drops, mostly mids;
- a fixed factor through a cut: the grid holds, the cut intervals show source frames only;
- the estimators and the planner on their own (snapping, trimming, one anchor per interval, recovery inside the
  lateness contract, cost attack and release).

Recorded traces live in `tests/traces/` (its README lists them). The first two, 30 s of the extension on the engine
runtime in Chrome and in Firefox (240 Hz display, 24 fps anime, Auto), replay to what the hosts measured: about
100 and 90 presents/s, no penalties, Firefox's few drops all stall-exempt and its factor capped at 5 by the 7 ms
present limit. The extension records them through its bench bridge (`trace` command); each new host finding becomes
a trace and an assertion.

## Host contract

Per display tick: `display_tick(now)`; present `tick.present` (or nothing); for every request in `tick.generate`,
generate phase `t` of the pair through the Stream and call `generated(now, id, Frame::Output(slot))`, or `skipped(id)`
when the Stream answers that the source frame is enough. Per source frame, once its interval kind is known:
`source_frame(now, frame)`, generate the requests it returns (mids of the previous pair that must go out before the
pair changes), then push the frame into the Stream. Costs come from the Stream's measurements through
`cost_sample`. The reference web host (design 04) is the executable version of this paragraph.

## What moved and what changed against the extension

- Everything measured in `cadence.js` and the controller of `content.js` is here with its reason as a comment.
- The settle window is 1200 ms, the extension's measured fullscreen reflow (an earlier draft of this document said
  4 s).
- The host does not report presents: a present happens on the tick that hands it out, and the Pacer counts it there.
- Leftover mids of the previous pair are returned from `source_frame` on every new pair, because the engine Stream
  runs the pair on push whatever the factor decision; the extension could keep driving the old job when the new pair
  did not run.
- In lookahead mode the generated interval's own kind decides interpolation; the extension skipped the previous
  interval's mids when the current pair was a cut.

## Decisions

1. The planner's output grid keeps anchors on exact source frames when the output rate is a multiple of the source
   rate (the nearest tick to a source frame becomes the anchor), as today.
2. The desktop's insert mode (real frames shown natively, predicted in-betweens on other refreshes, vblank-phase
   locked) becomes a Pacer `mode` of its own (`Insert`) with extrapolation intervals from the Stream, designed after
   the interpolation modes land.
