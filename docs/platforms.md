# Platforms

What the engine runs on, verified, with the date and the machine. "Supported" here means: the kernels pass the
self-tests and the full-graph parity against the PyTorch golden of the shipped model on that backend, and the numbers
below were measured there. Anything else is listed as what it is.

Rig for the hand measurements: Windows 11, NVIDIA GeForce RTX 4060 Ti, 240 Hz display; Vulkan driver 610.88, D3D12
driver 32.0.16.1088; the shipped model (FlyHighFG, step 110000), 1280 x 720 frames at the 856 x 480 rung, repair 1.
Parity: the v6p golden (`fixtures/v6p`, K 2, t 0.5), flow L1 in network pixels and render PSNR against torch.

## Native backends (wgpu)

| Backend | Date | Kernel class | Parity | GPU time at 720p | Notes |
|---|---|---|---|---|---|
| Vulkan | 2026-11-02 | `tc-f16` (cooperative matrices, shader-f16, subgroups, timestamps) | flow 0.0037 px, 72.1 dB | pair 1.47 ms, generated frame 0.60 ms (texture path); 2.08 ms per frame on the plan-session path at K 1, 2.40 at K 2, 2.70 at K 3 | the reference backend; a plan builds in 0.86 s with the autotune, 10 ms after |
| DX12 | 2026-11-02 | `portable-f16` with DXC (shader-f16, subgroups, timestamps; no cooperative matrices exposed) | flow 0.0036 px, 72.2 dB (f32 path: 0.0033 px, 84.1 dB) | pair 1.76 ms, generated frame 0.74 ms (autotuned, texture path; the f32 path 1.71 / 0.74); 2.54 ms per frame on the plan-session path at K 1 | wgpu compiles HLSL through `dxcompiler.dll` (with `dxil.dll` beside it) when it finds one: beside the engine's own module or the executable, at `EngineOptions::dxc_path`, or on the DLL search path. The Windows release archive ships both DLLs beside `flyhigh.dll` and `flyhigh-dev`; for a checkout, `sh tools/fetch_dxc.sh` puts the pinned Microsoft release (v1.9.2609) in `target/release`, and the Windows SDK ships one too. The 24 kernels then compile in 4 s and a plan with the autotune builds in 9.6 s (Vulkan: 16 ms and 0.86 s). Without the DLL wgpu falls back to FXC, `portable-f32` only: 139 s for the kernel set (`tconv32` alone 78 s), 460 s for the first autotuned plan, pair 2.30 / mid 0.95 ms with fixed routing. `DeviceInfo::shader_compiler` reports which one loaded |
| Metal | 2026-11-01 | `portable-f16` (the cooperative-matrix feature is exposed with 8 x 8 tiles only, which the kernels do not use) | CI: the GPU tests and the golden plans pass on the GitHub macOS runner | not measured on a real Mac | numbers wanted from an Apple GPU |
| Vulkan on lavapipe (software) | 2026-11-02 | `portable-f32` | CI: the golden plans and the GPU smoke tests pass on every push | not meaningful | the structure check of every change |
| OpenGL | | | | | not a target: the kernels need compute shaders with large storage buffers, rgba8 storage textures and 16 KB of workgroup storage, which wgpu's GL backend offers only in part (GLES 3.1 / GL 4.3 with gaps), and no platform that lacks Vulkan, D3D12 or Metal is one the engine aims at |

## Browsers (WebGPU)

| Browser | Date | Kernel class | Result | Notes |
|---|---|---|---|---|
| Chrome 156 (Windows, D3D12 underneath) | 2026-11-02 | `portable-f16` (shader-f16, timestamp-query, bgra8unorm-storage, subgroups) | the reference web host: 95 presents/s at Auto from 24 fps anime, no drops, no penalties; the extension's recorded session replays in the tests (`crates/flyhigh-engine/tests/traces/chrome-anime-auto.json`) | all kernels compile in 1.8 s in parallel |
| Edge | | the same engine as Chrome | not measured separately | |
| Firefox 159 (Windows) | 2026-11-02 | `portable-f16` (shader-f16, timestamp-query, bgra8unorm-storage; no subgroups) | the reference web host: 97 presents/s at Auto, a minute without a compositor stall (`crates/flyhigh-engine/tests/traces/firefox-refhost-auto.json`) | three host rules, all in the engine or its adapter: presents at least 7 ms apart (the compositor stalls 40 to 90 ms below that; the Pacer option `min_present_interval_ms`), kernels warmed up before playback (pipelines compile serially in the GPU process, 0.1 to 2 s each), an empty submit every millisecond while a readback is pending (mapAsync resolves only on GPU polls, 70 to 100 ms otherwise). `copyExternalImageToTexture` rejects `<video>`: the host blits through `importExternalTexture` |
| Safari 26 | | | not measured (needs a Mac) | |

## Interop

| Path | Date | Result |
|---|---|---|
| D3D11 shared texture into the Vulkan engine (Windows) | 2026-11-02 | `interop::d3d11_shared_texture`: two frames written by a D3D11 device, imported without a copy, their in-between at MAE 0.26 to the true middle (the same as uploaded frames); the 4060 Ti exposes `VK_KHR_external_memory_win32` |
| D3D12 shared handles, Vulkan external memory (Linux, Android), Metal IOSurface | | open |

## Mobile

Android (Vulkan) and iOS (Metal) builds and wasm in mobile browsers: not built yet, nothing measured.

## Reproducing

```sh
flyhigh-dev selftest                                 # kernels and the full-graph parity (the shipped model's golden)
FH_BACKEND=dx12 flyhigh-dev selftest                 # the same on another backend; FH_PRECISION=f32 forces the f32 path
flyhigh-dev bench 1280 720 --internal 856x480        # GPU time per generated frame, plan-session path, K 1 / 2 / 3
flyhigh-dev texbench 1280 720                        # the texture path: pair and generated frame apart
sh tools/fetch_dxc.sh target/release                 # Windows, DX12: the DXC DLLs beside flyhigh-dev (else FXC)
```

The browser numbers come from the reference web host (`examples/web/`) driven by the harness described in
[development.md](development.md); the recorded sessions in `crates/flyhigh-engine/tests/traces/` replay in `cargo test`.
