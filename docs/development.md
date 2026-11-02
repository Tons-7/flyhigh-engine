# Development

## Build

```
cargo build --release          # engine (crates/flyhigh-engine), C library (flyhigh-capi), CLI (flyhigh-cli), player
sh tools/build_web.sh          # browser module (crates/flyhigh-web) into web/pkg, wasm-bindgen 0.2.129 (Cargo.lock)
```

`web/pkg/` is not in git: build it after any change to the engine or `shaders/` (CI publishes it as the `web-pkg`
artifact). Serve the repository root (`python -m http.server 8765 --bind 127.0.0.1`) and open `web/index.html`
(the byte path in the browser, parity against a native `ref.png`) or `examples/web/index.html` (the reference host:
a video element through the Stream and the Pacer).

## CLI (`target/release/flyhigh-dev`)

One subcommand per task (`crates/flyhigh-cli/src/cli/<command>.rs`); `flyhigh-dev --help` lists them, `<command> --help`
the options.

| Command | |
|---|---|
| `selftest` (the default) | tensor-core probes, kernel self-tests and full-graph parity against a PyTorch golden dump |
| `microbench` | conv kernel microbenchmarks for this device's kernel class (tensor-core or portable) |
| `frames A.png B.png out.png [--t 0.5] [--k 2] [--internal WxH]` | one generated frame between two images of any size |
| `play <input> <output> [--factor 2] [--mode realtime\|video] [--cadence] [--repair 1] [--k 1] [--internal WxH] [--fps N] [--max N] [--stats] [--skip-cuts]` | the `api::Generator` end to end through ffmpeg pipes (any ffmpeg input / output) |
| `bench W H [--k K] [--internal WxH]` | GPU time per generated frame on procedural frames (`--k 0`, the default, sweeps 1, 2, 3) |
| `quality <sets_dir> [--baseline file.json] [--save out.json] [--k 1] [--static] [--fast]` | quality regression on ground-truth triplets through `api::Generator`; exits 1 on a regression |
| `texbench W H [--internal WxH] [--fast] [--linear] [--static] [--json]` | GPU time of the pair stage and of one generated frame on the texture path |
| `phases <video> [--factor 4] [--max 240] [--fast] [--linear]` | PSNR per generated phase against the dropped frames of a high-frame-rate video |
| `replay <trace.json>` | a recorded host trace through the Pacer: what it would present and generate, per second |
| `options-doc [path] [--check]` | writes `docs/options.md` from the option declarations (`#[derive(Options)]`, design 05); `--check` exits 1 when the file is stale, which CI runs |

`flyhigh-player <input> [factor=2] [realtime|video] [cadence] [repair=1] [k=1] [IWxIH] [fps=auto] [seconds=N]` plays
any ffmpeg input in a window through the texture-path generator.

The model directory is `FH_MODEL_DIR`, else `models/` of this repo.

## Developer environment variables

Read only by the CLI tools and the kernel builders; library hosts set options through the APIs instead
([options.md](options.md)).

| Variable | Effect |
|---|---|
| `FH_MODEL_DIR=<dir>` | model to load (`graph.json`, `flyhighfg.json`, `flyhighfg.bin`) |
| `FH_FIXTURES=<dir>` | parity golden for the loaded model (default: the shipped model's, `parity_fixture` in `models/models.json`, today `fixtures/v6p`) |
| `FH_BACKEND=vulkan\|dx12\|metal\|gl` | `EngineOptions::backend` |
| `FH_PRECISION=f16\|f32` | `EngineOptions::precision` |
| `FH_ADAPTER=<selector>` | `EngineOptions::adapter` |
| `FH_TUNE_CACHE=<dir>` | `EngineOptions::tune_cache` |
| `FH_DXC_PATH=<dir>` | `EngineOptions::dxc_path`: where `dxcompiler.dll` and `dxil.dll` are (DX12; default beside the binary, `sh tools/fetch_dxc.sh` puts them in `target/release`) |
| `WGPU_DX12_COMPILER=Fxc\|Dxc` | wgpu's own override of the DX12 shader compiler; wins over the option |
| `FH_NO_COOP=1` | `EngineOptions::tensor_cores = false` (the browser's kernels on native) |
| `FH_NO_AUTOTUNE=1` | `EngineOptions::autotune = false` |
| `FH_GMHS=0\|1\|2` | repair level of `frames` / `bench` sessions (default 1) |
| `FH_CUBIC=0\|1` | render sampler of CLI sessions (default: the manifest's) |
| `FH_TUNE_VERBOSE=1` | print every autotune decision |
| `FH_TRY_ALL=1` | also time the kernels that never won on the RTX 5090 (direct64, tiledm32, tiled8p) |
| `FH_NO_DIRECT=1`, `FH_DIRECT16=1`, `FH_NAIVE=1`, `FH_KSPLIT=<n>` | force kernel choices (A/B experiments) |
| `FH_NO_ARENA=1` | no buffer reuse inside a plan (memory comparison) |
| `FH_ARENA_TRACE=1` | every arena decision while a plan is built (allocation, reuse, release and why not) on stderr |
| `FH_ALL_OUTPUTS=1` | also compute the oracle-only graph outputs |

## Parity fixtures

The kernel fixtures (`fixtures/kernels/`) and the full-graph golden dumps (`fixtures/v5/`, `fixtures/v6p/`) are produced by
the export scripts of the research repo and are not in git; without them the self-tests print SKIPPED. Each golden
belongs to one checkpoint: run it with the matching model, e.g. with a local copy of the v5 model in `models_v5/`,
`FH_MODEL_DIR=models_v5 FH_FIXTURES=fixtures/v5 flyhigh-dev` (73 dB render PSNR on both kernel paths).
The shipped model checks with `FH_FIXTURES=fixtures/v6p flyhigh-dev` (72 dB). A golden records its render
sampler in `parity.json` (`render_sampling`, absent = bilinear) and the parity test renders with that sampler; dump one with
`python runtime/gen_parity_fixture.py <fixture> <K> <checkpoint> <bilinear|bicubic>` in the research repo.

## Quality regression

`quality` scores the generated middle frame of every triplet in a set directory (`manifest.json` with `{set, name}`
entries, files `<set>/<name>_1.png`, `_2.png`, `_3.png`) against the real middle frame. The compare sets of
`web/compare/sets/` (not in git) hold 84 triplets: 36 CS2, 28 natural video, 20 anime (ATD-12K).
`bench/quality_baseline.json` is the shipped model on those sets, k = 1, default internal size:

| Set | PSNR |
|---|---|
| anime (atd) | 31.78 dB |
| CS2 | 29.39 dB |
| natural | 38.16 dB |

```
target/release/flyhigh-dev quality web/compare/sets --baseline bench/quality_baseline.json
```

It fails when a set mean drops by more than 0.05 dB or one triplet by more than 0.5 dB. Results are identical on the
tensor-core and portable kernel paths. After a deliberate change (a new model), write a new baseline with `save=`.

## Performance baseline

`python tools/baseline.py` records this GPU's numbers into `bench/baselines/<gpu>.json` (texbench on every kernel class
and backend the GPU offers, the quality means, the parity, the wasm size); `--check` compares a fresh run with the
committed file and exits 1 when a pair or mid time is more than 3% slower, a set mean more than 0.05 dB lower or the
parity more than 0.1 dB lower. Run it before an engine commit that touches kernels or plans; CI runs `--check --quick`
on lavapipe for the structure. Details in [bench/README.md](../bench/README.md).

## Golden plans

`crates/flyhigh-engine/tests/golden_plans.rs` builds the plans of a fixed set of sessions (realtime at 720p and 1080p, fast render, static
protection, repair 0 / 1 / 2, k = 2, video mode with cadence, the CPU path) with autotune off and compares
`plan_dump` (pipeline, workgroups, what every binding is, the parameter words of each step) with
`tests/golden/<config>.<class>.txt`; the class is the kernel set: `tc-f16`, `portable-f16`, `portable-f32`. Any
accidental change to a plan's structure (a dropped step, another kernel, a new buffer size) fails the test on any
GPU, lavapipe included. After a deliberate change regenerate with
`FH_UPDATE_GOLDENS=1 cargo test --release --test golden_plans -- --ignored`. A missing golden for a class is skipped,
except `portable-f32`, which every device can build.

## Conventions worth knowing

- Tensors are f16 `[C][H][W]`; images enter the network in BGR order (the training order), see `shaders/rgba_in.wgsl`.
- Plans are static: buffers, uniforms and bind groups are built once; a frame records the plan into one command
  buffer. Anything built per frame (a uniform buffer, a bind group) is a garbage-collected object in the browser:
  cache it per slot instead (`SessionExt::tex_in_cached`, `CadenceScratch::cached`).
- Buffers come from a per-plan arena and hold a dead tensor's data: a kernel must write every element it later reads
  (no zero-initialisation assumptions).
