# Performance

GPU time of the engine itself, measured with GPU timestamps on an idle GPU (best of 60 runs). Presentation, the
host's own work and the page around it are not included.

**Setup:** RTX 4060 Ti (desktop), Windows 11, Vulkan, tensor-core kernels, default options: network at 856x480
(the 480-line rung of a 16:9 frame), repair 1, autotune on. Measured 2026-10-29 with
`FH_SPLIT=1 flyhigh-dev bench <W> <H> --internal 856x480`.

## One generated frame (2x)

Time to generate one frame between two source frames.

| Output | k = 1 (default) | k = 2 | k = 3 |
|---|---|---|---|
| 1280x720 | 2.08 ms | 2.40 ms | 2.71 ms |
| 1920x1080 | 2.46 ms | 2.77 ms | 3.07 ms |
| 2560x1440 | 2.99 ms | 3.31 ms | 3.61 ms |
| 3840x2160 | 4.66 ms | 4.95 ms | 5.28 ms |

`k` is the number of refinement passes of the motion estimate (option `k`). Each extra pass adds about 0.3 ms,
independent of the output size, because it runs at the network size.

## Higher factors

The work splits into two parts: estimating the motion between two source frames (once per source frame) and
synthesising a frame at a given phase (once per generated frame). With the host-pool and generator APIs the first part
is shared by all the frames of an interval, so 3x and 4x cost less per generated frame than 2x.

| Output (k = 1) | Motion, per source frame | Synthesis, per generated frame | 2x | 3x | 4x |
|---|---|---|---|---|---|
| 1280x720 | 1.49 ms | 0.59 ms | 2.08 ms | 1.33 ms | 1.08 ms |
| 1920x1080 | 1.61 ms | 0.81 ms | 2.41 ms | 1.61 ms | 1.34 ms |
| 2560x1440 | 1.81 ms | 1.09 ms | 2.90 ms | 1.99 ms | 1.69 ms |
| 3840x2160 | 2.43 ms | 1.91 ms | 4.34 ms | 3.13 ms | 2.72 ms |

The 2x / 3x / 4x columns are per generated frame. The motion part runs at the network size and grows little with the
output; the synthesis renders at the output size and carries most of the growth from 1080p to 4K.

## Other measurements

- RTX 5090, Linux, Vulkan, 1280x720 with the network at full resolution, repair off (2026-10-25 build): 0.98 / 1.21 /
  1.40 ms at k = 1 / 2 / 3; motion 0.68 ms plus synthesis 0.33 ms.
- Browser, Chrome 155 on the RTX 4060 Ti (portable kernels, no tensor cores in WebGPU), 1280x720 with the network at
  full resolution, k = 2: 4.3 ms. Firefox runs the same work with more GPU time; see the
  [integration guide](integration.md#browser-notes).

## Latency

Interpolation shows a moment between two source frames, so the newer one has to exist first: the added latency is at
least one source interval, whatever the GPU. The engine's own share is small (the times above); the rest comes from
the host.

Measured layer by layer on the desktop overlay (a separate host app: monitor capture, engine, overlay window) with a
perfectly paced 48 fps synthetic source on a 240 Hz monitor at 1920x1080, 2026-10-29:

| Layer | Time | Notes |
|---|---|---|
| Compositor timestamp to capture callback | 0.4 ms | |
| Capture readback to the CPU | 2.5 ms (p95 3.8) | copying the captured frame out of the GPU; a GPU-side capture removes it |
| Waiting for the next source frame | 20.8 ms | one source interval, inherent to interpolation |
| Jitter margin | 2 to 4 ms | covers late arrivals; `margin=auto` raises it to 5 to 7 ms |
| Engine: motion for the new pair | 0.33 ms | once per source frame |
| Engine: one generated frame | 0.2 ms | |
| Presenting before the vblank, then scanout | 5.7 ms | 1.5 ms ahead of the vblank plus one refresh |
| **Added latency, total** | **29 ms** | 32 ms with `margin=auto` |

The capture readback does not add to the total directly: frames are timed by the compositor's timestamps. It decides
how often a frame is not there yet when it is needed. Then the host briefly extrapolates past the newest frame or
holds the last one. With the default margin that happens to 8 of the 240 output frames per second, holds to 0.2. With
`margin=auto` it is 1.3 and none.

What the host can change:

- Embedded in the application (a game or player calling the engine), there is no capture and no second compositor
  pass: the added latency drops to about half a source interval plus the generation time.
- A higher source frame rate shortens the dominant layer: 20.8 ms at 48 fps, 16.7 ms at 60, 8.3 ms at 120.
- Extrapolation (predicting past the newest frame instead of waiting for the next one) removes the wait, at a cost in
  quality on sudden changes.

## Making it faster

In order of effect ([options](options.md)):

1. A smaller internal size (fewer network pixels). The automatic size is already the 480-line rung.
2. k = 1 (the default).
3. Higher factors on the host-pool / generator APIs: the motion is shared, only the synthesis repeats.
4. repair 0 saves 0.2 to 0.6 ms per source frame, at the cost of wobbling fences and window grids under camera pans.
5. Native autotune is on by default. In the browser, run `engine.autotune(model, options)` once per GPU and size and
   keep the table it resolves with (the `tune` session option).

## Measuring it yourself

```
cd native && cargo build --release
target/release/flyhigh-dev bench 1920 1080 --internal 856x480            # k = 1..3, one generated frame
FH_SPLIT=1 target/release/flyhigh-dev bench 1920 1080 --k 1 --internal 856x480  # plus the motion / synthesis split
```

`bench` runs on procedural frames (a smooth pattern with noise, the second frame shifted by a few pixels), so the
flow and the warps do real work. In the browser use `session.pairTimed` / `session.midTimed` on an
idle GPU; wall-clock numbers there include the page and the readback.
