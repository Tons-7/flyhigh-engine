# Decision records

Decisions that shape the engine, numbered in the order they were taken. A record says what the situation was, what
was decided and what follows from it, and is not edited afterwards; a reversal is a new record that supersedes it.

| Record | Decision |
|---|---|
| [0001](#0001-the-engine-is-the-product) | the engine is the whole product, in layers; hosts are thin shells |
| [0002](#0002-dual-licence-for-the-weights) | the weights are dual licensed, the code stays Apache-2.0 |
| [0003](#0003-names) | `flyhigh-engine`, `@flyhigh/engine`, `libflyhigh` with `flyhigh.h` |
| [0004](#0004-vulkan-is-the-primary-native-backend-dx12-runs-through-dxc-and-ships-it) | Vulkan is the primary native backend; DX12 runs through the DXC compiler and ships it |
| [0005](#0005-engine-and-session-become-send--sync) | `Engine` and `Session` become `Send + Sync` |
| [0006](#0006-python-is-a-tier-2-binding-over-the-c-abi) | Python is a tier-2 binding over the C ABI, no native extension |
| [0007](#0007-every-option-is-declared-once) | every option is declared once; the reference is generated and checked |
| [0008](#0008-the-performance-regression-rule) | a 3% slower pair or mid time against the GPU's baseline is a regression |

## 0001: the engine is the product

Date: 2026-11-01. Status: accepted.

**Context.** The runtime began as the kernels and plans behind a browser extension and a desktop overlay. Each host carried its own
pacing, drop handling, scene cut logic, static protection and auto-configuration, and the two drifted apart: a fix in
one did not reach the other, and a third host would have started from zero. An audit found a good foundation (one
crate, documentation, CI) under a monolithic library file, global state and four JavaScript API families.

**Decision.** The engine is the complete frame-generation product and every host is a thin shell around it. The logic moves into
the engine as layers with options and defaults: the Core (two frames in, one frame at a phase out), the Stream
(timestamps, intervals, cadence, classification), the Pacer (pure scheduling logic driven by the host's ticks, testable
from recorded traces) and optional host adapters. Each layer uses only the public API of the one below. The
restructuring comes before any new layer; every heuristic becomes an option with a default and a field in the report;
public API changes start as design documents.

**Consequences.** In this order: a safety net of golden plans and baselines, the split of the library into modules and
crates, nine design documents, then the Pacer and the Stream inside the engine with the reference hosts on them. The
desktop overlay and the extension become consumers; the extension is rewritten on the engine (done as Framegen 2.0 on 2026-11-02).

## 0002: dual licence for the weights

Date: 2026-11-01. Status: accepted.

**Context.** The code is Apache-2.0. The shipped weights were CC BY-NC 4.0 only, which blocks commercial integrators outright and
leaves them no path to use the engine with the shipped model in a product.

**Decision.** The weights are dual licensed: CC BY-NC 4.0 for everyone at no cost, and a commercial licence by agreement with the
maintainers for products that ship or run the weights commercially. The code stays Apache-2.0. A model someone trains
themselves is theirs. `models/WEIGHTS_LICENSE.md`, `NOTICE` and the README say so in one paragraph each.

**Consequences.** Commercial use has a path without changing the terms for everyone else. The engine loads other checkpoints in the
same format, so a user who cannot accept either licence can bring a model of their own.

## 0003: names

Date: 2026-11-01. Status: accepted.

**Context.** The crate was called a runtime, the repository the same, and the C library and npm package had no settled names.
Hosts and documentation referred to the same thing by several names.

**Decision.** The Rust crate is `flyhigh-engine` (library `flyhigh_engine`), the npm package `@flyhigh/engine`, the C library
`libflyhigh` with the header `flyhigh.h` (C symbols `fh_*`). The repository is `getflyhigh/flyhigh-engine`.

**Consequences.** The workspace crates are `flyhigh-engine`, `flyhigh-options`, `flyhigh-capi`, `flyhigh-web`, `flyhigh-cli`; the
developer binary is `flyhigh-dev`. crates.io and npm publishing wait for the accounts.

## 0004: Vulkan is the primary native backend; DX12 runs through DXC and ships it

Date: 2026-11-01, amended 2026-11-02. Status: accepted.

**Context.** The engine runs on every wgpu backend, but one of them has to be the one measured first and kept fastest. Vulkan
exposes the cooperative-matrix (tensor core) kernels and the D3D11 texture import works through its external memory.
DX12 matters for interop with games and overlays on Windows. wgpu compiles HLSL for DX12 through FXC unless it finds
the DXC compiler: with FXC the kernel set took 139 s to compile, the first autotuned plan 460 s, and there is no
shader-f16, so only the f32 kernels ran. The statically linked DXC needs the Visual Studio ATL component to build and
a 114 MB static library.

**Decision.** Vulkan stays the primary native backend. DX12 is supported through the dynamically loaded DXC (`dxcompiler.dll` with
`dxil.dll`): the engine looks for it beside its own module, takes `EngineOptions::dxc_path`, and reports which
compiler loaded. The Windows release archive carries the two DLLs with their licence files (dxcompiler under the LLVM
licence, dxil under Microsoft's terms for the DirectX Shader Compiler, Windows only).

**Consequences.** DX12 compiles the 24 kernels in 4 s and a plan with the autotune in 9.6 s, runs the f16 kernels (parity 72.2 dB, pair
1.76 / mid 0.74 ms at 720p on an RTX 4060 Ti) and is listed as experimental until the game pacing lands. Without the
DLLs the backend still works on the slow FXC path, and the report says so.

## 0005: `Engine` and `Session` become `Send + Sync`

Date: 2026-11-01. Status: accepted, not implemented yet.

**Context.** The device context holds its caches, the buffer arena and the compiled pipelines in `Rc` and `RefCell`, so an engine
cannot cross threads. Native applications with a render thread and a capture thread want to build sessions on one
and generate on the other; the browser is single threaded either way.

**Decision.** `Engine` and `Session` are `Send + Sync`: `Arc` where there is `Rc`, interior `Mutex` where the arena and the caches
live. The wasm build keeps working unchanged (the locks are uncontended there).

**Consequences.** A refactor of `Gpu` and the plan sessions before 1.0; until then native hosts keep the engine
on one thread.

## 0006: Python is a tier-2 binding over the C ABI

Date: 2026-11-01. Status: accepted.

**Context.** Research and evaluation want the engine from Python. A native extension (PyO3) means a second build system, wheels per
Python version and platform, and a second place where the API is mirrored by hand.

**Decision.** Python is binding tier 2: `bindings/python/flyhigh.py` over the C ABI with `ctypes`, no native extension. Tier 1 is
Rust, C and JavaScript; tier 2 also holds the C++ header-only wrapper; tier 3 is community bindings with a guide.

**Consequences.** One file, no build step, tested in CI on lavapipe against the C library. The C ABI stays the one contract every
binding follows; a new C function gets its Python and C++ line in the same change.

## 0007: every option is declared once

Date: 2026-11-02 (designed 2026-11-01). Status: accepted.

**Context.** The options reference was written by hand and drifted: it described the pre-layer API and lacked the Stream and the
Pacer entirely, while the Pacer alone has thirty heuristics. A reader could not trust it, and a new option needed the
same text in three places.

**Decision.** An option is declared once, as a field of its layer's options struct: the doc comment is its description, the default
instance gives its default, and `#[option(default, values, c)]` adds what the type cannot say. The `Options` derive
(`crates/flyhigh-options`) turns that into `OptionInfo`; a field without a doc comment does not compile.
`flyhigh-dev options-doc` writes `docs/options.md` from it and CI fails when the file is stale.

**Consequences.** Fifty-three options across the four layers in one generated reference; rustdoc shows the same text. The same
`OptionInfo` is the source for the C header comments and the TypeScript types when those are generated.

## 0008: the performance regression rule

Date: 2026-11-02. Status: accepted.

**Context.** The speed goal is "not below today's numbers, a regression over 3% only with a documented reason", but
nothing enforced it. A single texbench process swings up to 10% on the f32 path because the autotune picks between
candidates a few percent apart; a bare comparison would raise false alarms.

**Decision.** `tools/baseline.py --check` runs the fixed texbench set, the quality regression and the parity, each configuration the
best of three processes, and compares them with the committed file of the GPU: a pair or mid time more than 3%
slower, a set mean more than 0.05 dB lower or the parity more than 0.1 dB lower is a regression and exits 1. It runs
by hand on the rig before a change to kernels or plans; CI runs a quick form on lavapipe for the structure, not the
numbers. The baseline is re-recorded after a deliberate change, with the reason in the commit.

**Consequences.** The numbers in the README and the platforms matrix have a guard. The f32 path still moves about 2% between
best-of-three runs, so a borderline verdict deserves a second run before it is called a regression.
