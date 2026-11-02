# Design 07: the three surfaces

Status: agreed 2026-11-01. All three surfaces carry every layer: Rust, the C ABI (`fh_engine_*`, `fh_model_*`,
`fh_session_*`, `fh_stream_*`, `fh_pacer_*`, `fh_generator_*`; the header hand-written) and the JavaScript classes
(`FhEngine`, `FhModel`, `FhSession`, `FhStream`, `FhPacer`, plus `fh_pacer_replay`). The Pacer's parity is tested on
the recorded traces: the C test gets the Rust report byte for byte, the browser the same numbers as `flyhigh-dev
replay --json`. The earlier `fh_*` function families of the browser module are gone. Open: cbindgen for the header,
the TypeScript option types from `OptionInfo` (design 05), the parity test of the plan hash, `Submission` handles.

## Mapping

| Concept | Rust | C | JavaScript (`@flyhigh/engine`) |
|---|---|---|---|
| device | `Engine::new(&EngineOptions)` / `with_device` | `fh_engine_create(&FhEngineOptions)` | `await FhEngine.create(opts)` |
| model | `Model::from_dir` / `from_bytes` | `fh_model_load(dir)` / `fh_model_from_bytes(...)` | `engine.loadModel(graph, manifest, weights)` |
| session | `Session::new(&engine, &model, &SessionOptions)` | `fh_session_create(engine, model, &FhSessionOptions)` | `engine.session(model, opts)` |
| slots | `add_input` / `add_output`, `input_texture`, `add_input_texture` | `fh_session_add_input` ..., `fh_session_import_d3d11` | `session.addInput()` returns the index, `inputTexture(slot)` the `GPUTexture` |
| core calls | `pair`, `mid`, `generate`, `write_rgba`, `write_mask`, `read_rgba` | same names with `fh_session_` prefix; `read_rgba` fills a caller buffer | same names in camelCase; `readRgba()` returns a promise |
| stream | `Stream::open(&engine, &model, &SessionOptions, &StreamOptions)`, `push`, `generate`, `reset` | `fh_stream_*` | `engine.stream(model, sessionOpts, streamOpts)` |
| pacer | `Pacer::new(&PacerOptions)`, `display_tick`, ... | `fh_pacer_*`, `Tick` as a struct out-parameter | `new FhPacer(opts)`, `pacer.tick(now)` returns `{ present, generate, factor }` |
| report | `report()` on each object, `Report::to_json()` | `fh_*_report_json(buf, cap)` | `.report()` returns the object |
| options | structs with `Default` | `#[repr(C)]` structs with `fh_*_options_default()` | plain objects, every field optional, unknown keys rejected with the valid list |
| errors | `Error` | status codes + `fh_last_error` | `FlyHighError { code }` |

## Generation

- **C header**: `cbindgen` over `flyhigh-capi` produces `include/flyhigh.h`; doc comments carry the option
  descriptions from design 05. The hand-written header of today is replaced; CI diffs the generated header against the
  committed one.
- **TypeScript types**: `wasm-bindgen` generates the raw bindings; the classes are a thin TypeScript layer in
  `crates/flyhigh-web/js/` compiled into the npm package. Option types are generated from `OptionInfo` (design 05) so
  the three surfaces cannot disagree on a field name.
- **One function family in wasm**: the module exports the classes above and `fh_pacer_replay`, nothing else.

## Parity test

The same options through all three surfaces must produce the same plan: the Rust test builds a session and hashes
`plan_dump`; the C test calls through the cdylib and reads the hash from `fh_session_report_json`; the browser test
(headless, lavapipe where available) reads it from `session.report()`. The three hashes are compared in CI.

## Versioning

Semantic versions from 1.0. Options may gain fields (defaults keep behaviour); a field never changes meaning. Open:
the C structs carry a `size` field first so an older caller's struct is read correctly. The wasm module and the
TypeScript layer ship together under one version.

## Decisions

1. Names: crate `flyhigh-engine`, npm `@flyhigh/engine`, C library `libflyhigh` with header `flyhigh.h`.
2. The C API exposes all three layers from the first version; the Pacer is the piece native hosts lack most.
