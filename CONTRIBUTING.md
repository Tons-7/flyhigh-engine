# Contributing

Thank you for looking into the engine. This page says how the project works so that a change lands with as little
friction as possible: how to build and check it, the rules every change follows, what a good pull request looks like
and where to start.

## Before you start

- [ARCHITECTURE.md](ARCHITECTURE.md) is the map: the layers, the modules, the network, the plans, the kernels.
- [docs/design/](docs/design/README.md) holds the public API as designed; a change to the public API starts there.
- [docs/decisions/](docs/decisions/README.md) records the decisions already taken, with their reasons.
- An issue before a large change saves both sides time. Small fixes and documentation can go straight to a pull
  request.

## Building

Stable Rust (the version in CI), and for the browser module the `wasm32-unknown-unknown` target with the
`wasm-bindgen` CLI at the version in `Cargo.lock` (0.2.129 today). A GPU is not needed to build or to run the CPU
tests; the GPU tests run on any wgpu backend, in CI on lavapipe (software Vulkan) and on the macOS runner's Metal.

```
cargo build --release                 # engine, C library, CLI, example player
sh tools/build_web.sh                 # browser module into web/pkg
```

Details, the CLI commands and the developer environment variables: [docs/development.md](docs/development.md).

## Checks

CI runs exactly these; run them before opening a pull request.

| Check | Command |
|---|---|
| CPU tests (logic, options, the Pacer on recorded traces, the C defaults) | `cargo test --release` |
| GPU tests (Core, Stream, interop, the golden plans, the kernel self-tests) | `cargo test --release -- --ignored` |
| formatting | `cargo fmt --all -- --check` |
| lints, warnings are errors | `cargo clippy --release --all-targets -- -D warnings` |
| the options reference is current | `cargo run --release -p flyhigh-cli -- options-doc --check` |
| the browser module builds and stays under 1.3 MB | `sh tools/build_web.sh` |
| the bench suite runs end to end | `python tools/baseline.py --check --quick` |
| licences and advisories | `cargo deny check` |

Two checks need an eye rather than a command:

- **Golden plans.** `tests/golden_plans.rs` compares the plans of a fixed set of sessions with `tests/golden/`. A
  change that alters a plan on purpose re-records them with `FH_UPDATE_GOLDENS=1 cargo test --release --test
  golden_plans -- --ignored` and says in the pull request what changed in the dump and why.
- **Performance.** `python tools/baseline.py --check` compares a run on your GPU with `bench/baselines/<gpu>.json`:
  a pair or mid time more than 3% slower, a quality mean more than 0.05 dB lower or the parity more than 0.1 dB lower
  is a regression. Run it before a change to kernels or plans; a regression needs a documented reason
  ([bench/README.md](bench/README.md)). Without a baseline for your GPU it checks the structure only.

## Working rules

These are the rules the engine is built under; a pull request that follows them is easy to review.

1. **Behaviour changes only with tests.** A kernel or plan change comes with the golden plans and the parity; a Pacer
   change with a trace test (`tests/traces/`); a Stream change with a GPU test; an option with its validation test.
2. **Every heuristic is an option.** A threshold, a window, a default factor: it is a field of the layer's options
   struct with a default, a doc comment (the reference is generated from it, so a field without one does not compile)
   and a field in the report that says what the engine decided. Nothing silent inside a layer.
3. **The public API starts as a design document.** A new call, option or report field is written into the relevant
   document under `docs/design/` first, as the target; the code follows and the document is updated to say what exists.
4. **Each layer uses only the public API of the layer below.** The Pacer knows no textures, the Stream no clocks.
5. **Numbers come with their setup.** A measurement in a doc or a commit message names the GPU, the backend, the
   resolution and the date; the platforms matrix and the baselines are the reference.
6. **Comments say why, not what.** The code says what it does; a comment carries the reason, the measurement or the
   invariant a reader cannot see. Doc comments on public items are the documentation.
7. **Errors, not panics, in the library.** A bad option names its field and the reason; a bad state names the state.
   `unwrap` and `expect` belong to tests and tools.
8. **Everything in English**, in code, comments, documentation and commit messages.
9. **No dependency without a reason.** The engine depends on `wgpu` and little else; a new crate needs a sentence in
   the pull request on why it is worth its weight (the wasm size budget is 1.3 MB).

## Pull requests

- One topic per pull request, small enough to review in one sitting. A refactor and a behaviour change are two pull
  requests.
- The description says what changed, why, and what was measured (with the setup). Link the issue or the design
  document when there is one.
- Update what the change touches: the design document, `docs/platforms.md` for a platform finding, the goldens, the
  baseline, `CHANGELOG.md` under Unreleased.
- CI must be green. A failure in a GPU job is usually real: lavapipe has no tensor cores and the macOS runner no
  Vulkan, so a change that only works on one kernel class or backend shows up there.
- Commit messages: an imperative subject line, then a body with the reasons and the numbers, as the history does.

## Good first issues

Work that is well bounded, documented and verifiable without special hardware or with hardware you may have:

- Numbers for the platforms matrix: Metal on a real Mac, Safari, Edge, a GTX 10xx class GPU on the f32 path
  (`FH_BACKEND`, `flyhigh-dev texbench`, `docs/platforms.md`).
- A GPU-side protection mask: a texture the host owns instead of the byte mask (`Session::write_mask`,
  `shaders/keep_static_age*.wgsl`).
- Buffer pool reuse across sessions on one engine (`gpu/alloc.rs`).
- Doxygen (or any) reference generated from `crates/flyhigh-capi/include/flyhigh.h`.
- TypeScript types for the JavaScript classes generated from `OptionInfo` (design 07).
- A cookbook recipe: a short, runnable host for one case (offline video processing, a capture plugin skeleton).
- A Pacer trace from a host or display not yet in `tests/traces/` (a 120 Hz or 144 Hz panel, a laptop iGPU),
  recorded with the reference web host and added as a replay test.

## Releasing

Versions follow semantic versioning. Before 1.0 a minor version (0.x) may change the Rust API; the C ABI and the
JavaScript classes change additively within a minor, and a replaced call stays one release beside its
replacement. From 1.0 the three surfaces are stable within a major version; design 07 holds the
surface mapping and the parity test between them.

Checklist for a release:

1. `cargo test --release` and `-- --ignored` green locally and in CI; `python tools/baseline.py --check` on the rig
   with no regression, or the regression explained in the changelog.
2. `CHANGELOG.md`: move Unreleased to the version with the date; `docs/platforms.md` dated for anything re-measured.
3. The version in the workspace `Cargo.toml` (every crate inherits it); `cargo build --release` once more.
4. Tag `vX.Y.Z` and push the tag: `.github/workflows/release.yml` builds the C library with the header, the bindings
   and the model for Windows (with the DXC compiler), Linux and macOS, the browser module, and publishes the GitHub
   release. crates.io and npm are published by hand until the accounts exist.

## Conduct and security

[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) says how we treat each other; [SECURITY.md](SECURITY.md) says how to report
a vulnerability privately.
