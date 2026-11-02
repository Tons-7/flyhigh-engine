# 0001: the engine is the product

Date: 2026-11-01. Status: accepted.

## Context

The runtime began as the kernels and plans behind a browser extension and a desktop overlay. Each host carried its own
pacing, drop handling, scene cut logic, static protection and auto-configuration, and the two drifted apart: a fix in
one did not reach the other, and a third host would have started from zero. An audit found a good foundation (one
crate, documentation, CI) under a monolithic library file, global state and four JavaScript API families.

## Decision

The engine is the complete frame-generation product and every host is a thin shell around it. The logic moves into
the engine as layers with options and defaults: the Core (two frames in, one frame at a phase out), the Stream
(timestamps, intervals, cadence, classification), the Pacer (pure scheduling logic driven by the host's ticks, testable
from recorded traces) and optional host adapters. Each layer uses only the public API of the one below. The
restructuring comes before any new layer; every heuristic becomes an option with a default and a field in the report;
public API changes start as design documents.

## Consequences

In this order: a safety net of golden plans and baselines, the split of the library into modules and
crates, nine design documents, then the Pacer and the Stream inside the engine with the reference hosts on them. The
desktop overlay and the extension become consumers; the extension is rewritten on the engine (done as Framegen 2.0 on 2026-11-02).
