# Integration guide

## Pick an API

| Your frames are | Use |
|---|---|
| CPU bytes (decoded video, screenshots, any language) | the Core through [C](api-c.md) (`fh_session_*`) or [Rust](api-rust.md) (`Session::write_rgba` / `read_rgba`), or the CPU convenience `api::Generator` / `fh_generator_*`: RGBA8 in, RGBA8 out, one upload and one readback per frame |
| GPU textures in a Rust / wgpu app | [`Session`](api-rust.md) on your own views, or the [`Stream`](api-rust.md#stream) when frames carry timestamps: no CPU copies; `examples/player/src/main.rs` |
| A `<video>`, canvas or `ImageBitmap` in a page or extension | the [JavaScript classes](api-js.md): `FhStream` and `FhPacer` over textures on the engine's `GPUDevice` (`engine.device()`); `examples/web/main.js` |
| frames of a game or a capture, on their own display | the [Pacer](api-rust.md#pacer) in any of the three APIs: it says what to present on each tick and what to generate |

All of them run the same network and the same WGSL kernels with the same [defaults](options.md).

## Realtime or video mode

- **Realtime** (default): each new source frame produces the in-betweens of (previous, new). No lookahead. Use it for
  games, screen capture and anything interactive.
- **Video**: one source frame of lookahead. Needed for `cadence` (anime drawn on twos), and gives the engine the pair
  statistics before it renders an interval. Use it for video players.

Frame generation by interpolation always needs the newer frame first, so the output trails the source:

| Mode | Extra latency (factor 2) |
|---|---|
| realtime | half a source interval (the in-between is shown first, the new frame one output interval later) |
| video | one more source interval (lookahead) |

At 24 fps that is about 21 ms (realtime) and about 63 ms (video). Players hide it by delaying audio by the same amount.

## Presenting and pacing

The Pacer does this for you: feed it the display ticks, the source frames with their timestamps and kinds, and the
measured GPU costs; present what it says and generate what it asks for ([design 03](design/03-pacer.md)). A host that
paces itself follows the same rules:

- Ask for the phases you want per source interval: factor 2 is t = 0.5; factor 3 is t = 1/3, 2/3; any t in (0, 1)
  works (`mid(t, out)` takes any t without a rebuild).
- Present the outputs in the order returned, at **source interval / factor** spacing, from a clock tied to the
  display (vsync / `requestAnimationFrame`), not from when the GPU finished. Uneven spacing looks worse than no
  frame generation.
- Measure the source interval from the source timestamps (media time, capture time), not from arrival times.
- If a generated frame is not ready when it is due, skip it and show the next real frame. Never stall the source.

## Events the host must handle

| Event | Do |
|---|---|
| seek, decoder gap, dropped source frame | `stream.reset()` and `pacer.reset(now)`; with the Core alone, start a new pair. Otherwise the next in-between interpolates across the gap |
| scene cut | the Stream classifies it (`scene_cut`, on by default) and the Pacer shows the source frames across it. With the Core alone, compare consecutive frames yourself and present the new frame directly |
| frame size change | build a new session (or stream) for the new size: tens of milliseconds, plus the autotune once per new size |
| GPU device lost | drop the engine and create a new one, then the model, sessions and streams |
| paused video | stop pushing; nothing to generate |
| repeated source frames (duplicates in a stream) | the Stream's `held_frames` option detects them (on for anime); otherwise push only distinct frames with their real timestamps |
| a fullscreen change or a resize of the page | `pacer.modeChange(now)`: the reflow stall is not a GPU limit |

## Performance

GPU time per generated frame with the defaults: about 1 ms at 720p on an RTX 5090, 2.1 ms at 720p and 2.4 ms at 1080p
on an RTX 4060 Ti (native). Levers, in order of effect:

1. **Internal size.** The automatic size keeps the network at about 480 lines; a smaller one saves time on weak GPUs.
2. **Autotune.** Native: on by default, persisted through `tune_cache`. Browser: `await engine.autotune(model,
   sessionOptions)` once per GPU and size while idle, store the text it resolves with (for example in `localStorage`,
   keyed by GPU, size and `k`) and pass it as the `tune` session option.
3. **k.** 1 is the default; 2 costs about 20% more.
4. **Higher factors are cheaper per frame**: the pair stage runs once per source interval and only the mid stage per
   generated frame (3x costs about two thirds of 2x per generated frame).
5. **Fast render with linear t** (`fastRender`, `linearT`): the same picture, 30 to 40% less per generated frame, and
   each further phase of an interval costs only the render.
6. **repair** costs 0.2 to 0.6 ms per source pair depending on GPU and network size; set 0 only if you have measured
   that you need the time.

Keep the engine on the same `GPUDevice` / `wgpu::Device` as your presentation so nothing is copied between devices.

## Platform notes

- DX12: wgpu compiles the kernels through DXC when `dxcompiler.dll` and `dxil.dll` are beside the engine library or
  your executable (the Windows release archive ships them next to `flyhigh.dll`; `sh tools/fetch_dxc.sh` fetches
  them for a checkout), or where `EngineOptions::dxc_path` points. Without them it falls back to FXC: the kernels take
  minutes to compile and the device has no `shader-f16`, so the f32 kernels run; `DeviceInfo::shader_compiler` tells
  you which happened. Vulkan stays the primary native backend on Windows ([platforms.md](platforms.md)).
- macOS (Metal) and Safari: nothing platform-specific in the code; Metal runs in CI, Safari is untested.
- Tensor cores: the cooperative-matrix kernels run where the driver exposes them (NVIDIA on Vulkan today); elsewhere
  the portable kernels run, within a few percent of the same speed on this network.

## Browser notes

- WebGPU in Chrome / Edge 120+ and Firefox 141+ on Windows. Firefox needs three host rules, all in the engine or the
  reference host: presents at least 7 ms apart (`min_present_interval_ms`), kernels warmed up before playback, a
  submit every millisecond while a readback is pending ([platforms.md](platforms.md)).
- Chrome MV3 extensions: add `'wasm-unsafe-eval'` to `content_security_policy.extension_pages`. Pages whose CSP
  forbids wasm cannot run the engine from a content script.
- Firefox cannot `copyExternalImageToTexture` from a `<video>`: sample the video with `importExternalTexture` in your
  own render pass and draw it into `inputTexture(slot)`, or push an `ImageBitmap`.
- Firefox content scripts cannot patch DOM prototypes; use `engine.device()` and the texture getters instead of
  intercepting `requestDevice` / `createTexture`.
- A GPU to CPU readback (`readRgba`) costs about 100 ms in Firefox; the presentation path never reads back.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `precision F16 requested but the device lacks shader-f16` | the GPU or browser has no f16 shaders; leave `precision` automatic and the f32 kernels run |
| `unknown field` in an options object | a typo in a JavaScript options key; the message lists the valid keys |
| `option k: 4 is not 1..=3` and the like | an option out of range; the field is named |
| in-betweens smear across a cut or after a seek | the host pushed across a cut with scene cuts off, or did not reset after a seek (see Events) |
| the first build takes seconds | the autotune at a new size: persist it (`tune_cache` natively, the `tune` option in the browser) or turn `autotune` off |
| fences or window grids wobble during camera pans | `repair` is 0; use 1 (default) or 2 for drawn content |
| anime characters stutter while the background pans smoothly | animation on twos: use video mode with `cadence` |
| a HUD or subtitles are dragged along with the motion | turn on `static_protection`, or paint the host's `protection_mask` |
