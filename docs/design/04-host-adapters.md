# Design 04: host adapters

Status: agreed 2026-11-01; the web pieces live in the reference web host, the native ones in the player and the
`interop` module. Optional, thin, replaceable. Each adapter
turns a platform's way of delivering frames and ticks into the calls of designs 01 to 03, and the engine's outputs back
into something the platform shows.

## Web (`adapters.js`, shipped next to the wasm module)

| Piece | Does |
|---|---|
| `FhEngine.create(options)` | creates the `GPUDevice`; `engine.device()` is the one the host presents on (adopting a page's own device is open: `Engine::with_device` exists in Rust) |
| `VideoSource` | writes a `<video>` frame into a stream slot: `pushVideo` (`copyExternalImageToTexture`) where the browser takes it, an `importExternalTexture` render pass into the slot where it does not (Firefox) |
| `CanvasPresenter` | configures the canvas and blits the slot the Pacer names on each tick |
| `TickSource` | `requestAnimationFrame` as `pacer.tick(now)`; a tick later than 50 ms after the previous one is a host stall |
| `warmUp` | the session's or stream's kernels compiled and awaited through `onSubmittedWorkDone` before playback |
| `ReadbackNudge` | while a readback is pending in Firefox, `queue.submit([])` every millisecond so its promise resolves in 5 ms instead of 70 to 100 |
| `CostSampler` | `push` and `generate` with a timed run every thirtieth call and on every probe, the GPU milliseconds fed to `costSample` |
| `pacerDefaults()` | the host quirks below as Pacer options for this browser |

The reference web host (`examples/web/`) is these pieces plus a page's own UI over `FhStream` and `FhPacer`; it is
the template for a browser host and the bench for browser findings. 24 fps anime at 240 Hz on the 4060 Ti: Chrome
and Firefox both above 85 presents/s at Auto, no drops, no penalties; Firefox, a minute with the 7 ms present
limit, no compositor stall at all (`tests/traces/firefox-refhost-auto.json`). The Pacer records the session itself
(`pacer.record`, `pacer.trace()`), so any host hands over a replayable trace.

## Native (`flyhigh-engine` features, `examples/player`)

| Piece | Does |
|---|---|
| `VblankTicks` | Windows: DWM vblank timing (today's `dwm_vblank`), the PLL that keeps the phase (the desktop's `sync_vblank`); Linux / macOS: the window system's frame callbacks |
| `interop::d3d11_shared_texture(engine, handle, &desc)` | a D3D11 shared texture (screen capture) into the Vulkan device without a copy, then `add_input_texture` on a session or a stream; landed 2026-11-02 from the desktop's `gpucap.rs` with a D3D11-to-Vulkan test (`tests/interop_d3d11.rs`); C: `fh_session_import_d3d11` / `fh_stream_import_d3d11`. DX12 import is open |
| `import::vulkan_external_memory`, `import::metal_iosurface` | the same for Linux / Android and macOS |
| presentation | the host's swapchain; the engine hands back the output texture, the host blits or composites it |

The desktop overlay stays its own repository and a consumer of these pieces; its capture ring, hotkeys, settings
window and insert mode are not engine concerns. Its pacer becomes `Pacer` with `mode: Insert` once that mode exists
(design 03, open point 2).

## Capture and media plugins (later)

A plugin for OBS or a media framework is a native host with frames arriving as textures or buffers: the same import
functions, `Stream` and `Pacer`, with the plugin's own clock. Nothing engine-specific beyond design 01 to 03 is
needed; the binding tiers (design 07) cover the languages.

## Host quirks table

The engine keeps a table of host-specific facts with the measurement behind each, applied as option defaults by the
adapter (never silently inside the layers):

| Host | Quirk | Default set by the adapter |
|---|---|---|
| Firefox (WebGPU) | the compositor stalls 40-90 ms when two canvas presents are closer than ~7 ms (5 ms still stalls, 7 ms is clean) | `min_present_interval_ms = 7` |
| Firefox | pipelines compile serially in the GPU process, 0.1-2 s each | `warm_up()` before playback |
| Firefox | `mapAsync` resolves only on GPU polls (70-100 ms) unless work is submitted | readback nudge |
| Firefox | `copyExternalImageToTexture` rejects `<video>` and `VideoFrame` | `importExternalTexture` blit |
| Firefox content scripts | DOM prototypes are Xray wrappers (no patching, no spread of `device.features`) | `engine.device()` and the texture getters |
| Chrome MV3 extension | wasm needs `'wasm-unsafe-eval'` in the extension CSP; pages whose CSP forbids wasm cannot run it from a content script | documented; the extension falls back |
| Windows Optimus laptops | capture lands on the iGPU, the engine on the dGPU | capture adapter option (desktop) |
| Windows DX12 (wgpu) | wgpu compiles HLSL through FXC unless it finds `dxcompiler.dll`: 139 s for the kernel set, no shader-f16 | the engine looks for the DLL beside its own module; `EngineOptions::dxc_path`; `DeviceInfo::shader_compiler` says what loaded |

## Decisions

1. One npm package `@flyhigh/engine` ships the wasm module and the web classes together.
2. The desktop's D3D11 to Vulkan import became `interop::d3d11_shared_texture`; the Windows Graphics Capture
   session itself stays in the desktop.
