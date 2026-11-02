# Design documents

The public API of the engine, designed before it was written and agreed on 2026-11-01. Each document described the
target first and is updated to describe what exists.

| Document | Covers |
|---|---|
| [00-overview.md](00-overview.md) | the four layers, the dependency rule, the principles, the vocabulary |
| [01-core.md](01-core.md) | `Engine`, `Model`, `Session`: devices, models, plans, slots, pair / mid |
| [02-stream.md](02-stream.md) | `Stream`: frames with timestamps, lookahead, cadence, scene cuts, held frames, readbacks |
| [03-pacer.md](03-pacer.md) | `Pacer`: the tick-driven scheduler, output rate policy, drop classification, Auto, traces |
| [04-host-adapters.md](04-host-adapters.md) | web and native glue, the reference hosts, the host quirks table |
| [05-options-and-report.md](05-options-and-report.md) | options declared once, presets, the `Report` |
| [06-errors.md](06-errors.md) | the error model in Rust, C and JavaScript |
| [07-surfaces.md](07-surfaces.md) | the mapping between the three APIs, generation, the parity test, versioning |
| [08-model-format.md](08-model-format.md) | format versioning, capabilities, the catalogue, the loader contract |
