# Changelog

Notable changes to the engine, in the format of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions
follow semantic versioning as described in [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

## [0.3.0] - 2026-11-03

### Added

- Camera warp (`Stream::mouse`, `set_mouse_captured`, `warp`; C `fh_stream_mouse`, `fh_stream_set_mouse_captured`,
  `fh_stream_warp`; JavaScript `mouse`, `setMouseCaptured`, `warp`; Python `mouse`, `set_mouse_captured`, `warp`):
  for games, the frame about to be presented is turned by the camera motion the mouse made since the moment it shows,
  so the camera answers the mouse at once however late the picture is. The host passes the mouse counts and whether
  the mouse is captured; the Stream measures each source frame's turn and learns how many pixels a count turns the
  picture and how late the picture shows it (on synthetic play: the lag to the millisecond, the gain within 0.1 %).
  It holds the warp off until the fit is strong, in menus and for games whose mouse drives a cursor. A weapon in
  hand and the HUD are detected and keep their pixels. 0.07 ms per source frame and 0.06 ms per presented frame at
  1080p; a host that never passes the mouse runs none of it. `warp::CameraWarp` and `warp::InputCalibrator` are
  public for hosts with their own textures.
- `Readback::try_finish`: a readback checked once per frame instead of awaited.

### Changed

- Output textures are also `COPY_DST` (the warp writes a turned frame back into its slot).
- `StreamReport` has `camera_warp` (the calibration, once started) and `counters.warped`.

## [0.2.0] - 2026-11-02

### Added

- `low_latency` (Pacer option, C `FhPacerOptions.low_latency`, JavaScript `lowLatency`): for games, the presentation
  delay is what the next in-between needs, (1 - 1/N) of a source interval plus the measured pair, frame and present
  cost and the arrival jitter, instead of the video default with its 60 ms floor. A 60 fps stream to a 240 Hz display:
  a source frame reaches the screen 18 ms after its arrival instead of 61, without drops in Auto (heavy network jitter
  22 ms, a slow GPU 23 ms, 30 fps sources 31 ms). Off by default; video keeps its behaviour.
- Trace replays report how long source frames take from arrival to the screen, and can charge GPU time per pair and
  per generated frame (`flyhigh-dev replay --pair-ms --mid-ms`), so a shorter delay that the work does not fit shows
  as late frames.

### Changed

- `FhPacerOptions` grew a field at its end (`low_latency`): C hosts rebuild against the new header.

## [0.1.4] - 2026-11-02

### Fixed

- Static protection no longer holds parts of a moving scene as overlays. After the camera stood still every pixel
  had a high age, and in the first pairs of a fast turn the pixels that matched by coincidence (more of them with a
  looser tolerance) were rendered as the unwarped average of the two frames: walls, palms and the edges of the weapon
  appeared twice. A pixel is now protected only after it stayed unchanged while the picture around it moved, in two
  pairs: the HUD and the crosshair qualify from the second in-between of a turn on. On a still-then-turn sequence
  the first in-between of the turn rises from 32.6 to 34.9 dB (30.1 to 34.9 with frames 4 and tolerance 8), the same
  as without protection, and the later ones keep the protected HUD (36.6 to 36.8 dB, against 35.8 without). The age
  pass costs 0.34 ms more per source frame at 1080p.

## [0.1.3] - 2026-11-02

### Added

- `compiled_kernels` and `precompile` (Rust `Engine`, the browser `FhEngine` as `compiledKernels` / `precompile`): a
  host records the kernels its configurations compile and compiles them a few per frame before the next stream opens.
  Chrome compiles in the process that composites the page and caches per site, so opening a stream on a new site froze
  a playing video for about a second (Framegen: three to four stalls of up to 1.1 s; one of 0.1 to 0.3 s with the
  kernels precompiled while the pointer rests on the video).

## [0.1.2] - 2026-11-02

### Changed

- Cadence re-timing keeps the real drawing where the transition pair's model could not match the frames (an arm
  appearing from behind the body): the re-timed render there was a guess and showed as a translucent or doubled limb.
  Still shots go through the same mask. On a clip with such a disocclusion the re-timed real slots are about half as far
  from the source frames, with the same 2 % of stopped steps; ripples and ghost limbs on crowd and walk shots are gone.
  No measurable cost: the same frame times natively, and in Chrome the same presents, delay and zero drops.
- The cadence option describes its price: presented frames are generated in-betweens, so big drawing steps show the
  generator's softness more often than plain interpolation does. A choice for the viewer, not a default.

## [0.1.1] - 2026-11-02

### Fixed

- The model loader checks a model completely before anything reads it: integer dimensions of at least 1, offsets and
  sizes computed without overflow, every tensor inside the blob, every weight the graph names present (the FiLM
  weights included), operands that are strings, repeat bodies that are lists. A malformed model directory is an
  error with the tensor's name; it used to pass the loader and panic in the builder.
- The browser classes refuse an option number that does not fit (`internal`, `staticProtection.frames` and
  `tolerance`) instead of turning it into a nearby one.
- Every C ABI function turns a panic into `fh_last_error` and its error return (null or -1); a panic that reached the
  boundary aborted the host's process.

## [0.1.0] - 2026-11-02

The first release: the engine as a library with a Rust API, a C ABI and a browser module, the reference hosts, and the
Framegen 2.0 extension built on it.

### Added

- The engine as the whole product in layers: the Core (`Engine`, `Model`, `Session`: two frames in, the frame at any
  phase out), the Stream (timestamped frames in, intervals out, with scene cut and held frame classification and the
  anime cadence), the Pacer (what to present on a display tick and what to generate, replayable from `trace v1`
  recordings). Reference hosts: a web page (`examples/web`) and a native player (`examples/player`).
- Surfaces: the Rust API, the C ABI (`flyhigh.h`, `libflyhigh`) and the JavaScript classes (`FhEngine`, `FhModel`,
  `FhSession`, `FhStream`, `FhPacer`); Python (ctypes) and C++ (header only) bindings over the C ABI; a C example.
- Options declared once with the `Options` derive; `docs/options.md` generated by `flyhigh-dev options-doc` and checked
  in CI.
- Static protection for HUDs and subtitles, and the host's protection mask (`protection_mask`, `write_mask`).
- D3D11 shared textures into the Vulkan device without a copy (`interop::d3d11_shared_texture`).
- DX12 through the DXC compiler: the DLLs beside the library or `EngineOptions::dxc_path`, `DeviceInfo::shader_compiler`
  in the report; the Windows release archive carries them.
- The bench suite: per-GPU baselines and `tools/baseline.py --check` with a 3% rule on times.
- GPU costs from the frames themselves: `Stream::push_timed` / `generate_timed` (`pushTimed` / `generateTimed` in
  the browser) run the real work under timestamp queries; the reference web host feeds them to the Pacer.
- The Pacer records its own `trace v1` (`record`, `trace`, `stop_recording`; `fh_pacer_record` / `fh_pacer_trace_json`
  in C, `record` / `trace` / `stopRecording` in the browser), with live option changes as events a replay applies.
- `adapters.js` next to the browser module: `VideoSource`, `CanvasPresenter`, `TickSource`, `CostSampler`,
  `ReadbackNudge`, `warmUp`, `pacerDefaults`; the reference web host is built from them.
- Stream and Pacer options and inputs in the browser accept camelCase keys beside snake_case.
- A release workflow on `v*` tags: the C library, header, bindings and model per platform, the browser module.

### Changed

- The browser module is built with a size-first profile: 1.02 MB, with a 1.3 MB budget in CI.
- `generate_timed` measures the mid in video mode too (a cadence composite stays unmeasured), so a host in video mode
  gets its mid cost samples instead of the seed.
- `VideoSource` (adapters.js) draws a frame whose size differs from the slot's through `importExternalTexture`, so a
  host can run the stream at the presentation size without a copy of its own; the 1:1 copy stays for matching sizes.
- Pipelines compile on first use; `warm_up` compiles a session's kernels before the first frame.
- Firefox hosts present at least 7 ms apart (the Pacer's `min_present_interval_ms`), the adapter's default.

### Removed

- The `fh_*` function families of the browser module (`fh_create`, `fh_vid_*`, `fh_ext_*`, `fh_prepare*`,
  `fh_autotune`, `fh_tune_*`, `fh_set_*`): the classes replace them; `engine.autotune` and the session's
  `pairTimed` / `midTimed` carry over what only they had.
