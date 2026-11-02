# Design 05: options and the report

Status: agreed 2026-11-01; the declarations and the generated reference are implemented (`crates/flyhigh-options`,
`flyhigh-dev options-doc`, checked in CI). Open: `validate()` from the declarations, presets, the report tree, the C
header and TypeScript types from `OptionInfo` (design 07).

## One source of truth for every option

Each option is declared once, with everything a user needs, and the documentation table is generated from it:

```rust
#[derive(Options)]            // derive: Default, validate(), describe(), and the docs table
pub struct PacerOptions {
    /// How many frames to generate per source frame: a fixed factor, or Auto (as many as the display and the GPU
    /// allow, capped by motion), the display's rate, or a target rate.
    #[option(default = "Auto", cost = "none", when = "benchmarks and players that want exactly 2x")]
    pub mode: RateMode,
    /// Fraction of the refresh rate the planner may use; the rest absorbs hitches.
    #[option(default = 0.97, range = "0.9..=1.0", cost = "lower = fewer frames", when = "a display that drops ticks")]
    pub display_headroom: f64,
    // ...
}
```

`describe()` returns the list of `OptionInfo { name, default, range, cost, when, doc }`; `tools/gen_options_doc`
writes `docs/options.md` from it and CI fails when the file is stale. The same `OptionInfo` feeds the C header
comments and the TypeScript types (design 07).

Options are grouped by layer: `EngineOptions`, `SessionOptions`, `StreamOptions`, `PacerOptions`. `validate()` runs
on construction and returns `Error::Options { field, reason }` (design 06), never a panic. Dependencies between options
(cadence needs repair >= 1; linear_t needs fast_render; tensor cores need f16) are validated there, and the resolution
(raising repair to 1) is reported, not silent.

## Presets

A preset is a named set of values across the groups; a host can print it and a user can start from it:

| Preset | Session | Stream | Pacer |
|---|---|---|---|
| `quality` | k 2, repair 1, exact render, internal auto | scene cut on | Auto, motion adaptive, headroom 0.97 |
| `balanced` (default) | k 1, repair 1, fast render + linear t, internal auto | scene cut on | Auto |
| `performance` | k 1, repair 0, fast render + linear t, internal auto | scene cut on | Auto, max factor 4 |
| `low-end` | k 1, repair 0, fast render + linear t, internal one rung lower | scene cut on | Factor(2) |
| `game` | balanced + static protection (8 frames, tolerance 4) | realtime, held frames on | Auto |
| `anime` | balanced | video mode, cadence on | Auto |

Presets are data (`presets.rs`), listed by `Preset::all()`, and the documentation table is generated too.

## The report

Everything the engine decided and measured, in one tree, available at any time, serialisable to JSON for bug reports
and for the host's UI:

```rust
pub struct Report {
    pub engine: EngineReport,   // adapter, backend, class, features, kernels compiled (count, total compile ms),
                                // memory (tensors MB, allocator reserved MB), autotune entries
    pub session: SessionReport, // frame / internal size, k, repair, render path, plan steps (pair, mid), kernels chosen,
                                // measured pair / mid ms (last, p50, p95)
    pub stream: StreamReport,   // pairs, mids, intervals: motion / cut / hold, readback latency ms, resets
    pub pacer: PacerReport,     // display Hz (raw, nominal, capacity), source Hz, output mode and rate, clamp reason,
                                // factor (current, ceiling), penalties (current, total, last reason), pressure,
                                // drops (total, classified: late / stall-exempt / superseded / held), presents,
                                // delay target ms, late p50 / p99, settle window active
}
```

Counters are monotonic since creation; the host diffs them for rates. `Report::to_json()` is the one blob a user
attaches to an issue. The reports of the layers today (`engine.report()`, `session.report()`, `stream.report()`,
`pacer.report()`) become its branches.

## Environment variables

The `FH_*` developer variables stay for the CLI and the kernel builders only; library hosts set options through the
API. Each is listed in `docs/development.md` with the option it mirrors.

## Decisions

1. Options are declared through a derive macro in a small proc-macro crate `flyhigh-options`; the documentation table
   and the surface types are generated from its output.
2. Presets apply at creation. Pacer options may change live; Session options rebuild the session.
