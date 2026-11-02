# Design 00: the layers

Status: agreed 2026-11-01; implemented.

## What the engine is

A host gives the engine frames and the time; the engine gives back the frames to show and when. Everything between
those two lines (motion estimation, synthesis, scheduling, adapting to the GPU and the display) lives in the engine, as
options with defaults. A host is a thin shell: capture or decode, hand over, present.

Four layers, each usable on its own, each built only on the public API of the layer below:

| Layer | What it owns | Enter here when |
|---|---|---|
| **Core** (`Engine`, `Model`, `Session`) | the GPU device, a model, the compiled plans for one frame size; `pair(a, b)`, `mid(t, out)` | you schedule frames yourself and only want the synthesis |
| **Stream** | frames with timestamps: ring of inputs, pair / mid scheduling, lookahead, cadence detection, held frames, scene cuts, per-pair statistics | you have your own display loop but want the engine to decide what each interval contains |
| **Pacer** | the display: tick-driven scheduler, output rate policy, drop classification, adaptive factor, presentation queue, host limits | you want frame generation that just works on a display |
| **Host adapters** | optional glue: canvas presenter, external texture import, display tick sources, warm-up | you are a browser page, a native app, a capture plugin |

The dependency rule: a layer uses only the public API of the layer below. The Pacer knows nothing about `Gpu`
internals; the Stream knows nothing about displays; the Core knows nothing about time. The engine's own hosts (the
desktop overlay, the reference web page, later the extension) use only these public layers. A gap in the API shows up
on us first.

## Principles that every design document follows

1. **The engine never reads a clock.** Hosts pass `now` (milliseconds, their monotonic clock) and source timestamps
   into every call that needs time. All scheduling logic is a pure function of its inputs, so a recorded trace replays
   bit for bit in a test (design 03).
2. **Every heuristic is an option.** Default, range, cost, when to change it, in one place in the code; the options
   table in the documentation is generated from it (design 05). Auto is a preset, not hidden magic.
3. **The engine reports what it did.** One `Report` struct, the same in all three APIs, serialisable to JSON: what was
   chosen, measured, dropped and why (design 05).
4. **No panics on input.** Typed errors in Rust, codes in C, exceptions with codes in JS (design 06). A panic is an
   engine bug and is reported as such.
5. **One semantics, three surfaces.** Rust is the source of truth; the C header and the JS types are generated; a
   parity test feeds the same options through all three and compares the plan hash (design 07).
6. **Behaviour stays measurable.** Golden plans, parity, quality baselines and trace tests gate every change.

## Vocabulary

| Word | Meaning here |
|---|---|
| source frame | a frame the host decoded or captured, with its presentation timestamp (pts, seconds of media or capture time) |
| interval | the time between two consecutive source frames, [pts_a, pts_b) |
| phase `t` | position inside an interval, 0 = the first source frame's time, 1 = the second's |
| pair | the motion estimate for one interval (the expensive, t-independent half of the network) |
| mid | one synthesised frame at phase `t` of a pair (the cheap, t-dependent half plus the render) |
| factor | output frames per source frame (2 = one mid per interval) |
| slot | an engine-owned input or output texture a host writes into or presents from |
| plan | the fixed list of GPU dispatches for a session (static; built once per size) |
| display tick | the host's vsync / `requestAnimationFrame` callback, the Pacer's heartbeat |

Documents: [01 Core](01-core.md), [02 Stream](02-stream.md), [03 Pacer](03-pacer.md),
[04 Host adapters](04-host-adapters.md), [05 Options and report](05-options-and-report.md), [06 Errors](06-errors.md),
[07 Surfaces](07-surfaces.md), [08 Model format](08-model-format.md).
