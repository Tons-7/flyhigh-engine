# 0005: `Engine` and `Session` become `Send + Sync`

Date: 2026-11-01. Status: accepted, not implemented yet.

## Context

The device context holds its caches, the buffer arena and the compiled pipelines in `Rc` and `RefCell`, so an engine
cannot cross threads. Native applications with a render thread and a capture thread want to build sessions on one
and generate on the other; the browser is single threaded either way.

## Decision

`Engine` and `Session` are `Send + Sync`: `Arc` where there is `Rc`, interior `Mutex` where the arena and the caches
live. The wasm build keeps working unchanged (the locks are uncontended there).

## Consequences

A refactor of `Gpu` and the plan sessions before 1.0; until then native hosts keep the engine
on one thread.
