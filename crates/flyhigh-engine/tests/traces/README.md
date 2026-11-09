# Recorded host traces (`trace v1`)

Sessions recorded from real hosts and replayed through the Pacer in `crates/flyhigh-engine/tests/pacer.rs`. Each
file is the host's events only (ticks, source frames with their interval kind, GPU cost samples, mode changes), so a
replay is a pure function of the file: the same decisions on every machine, with no GPU.

| File | Host | Session |
|---|---|---|
| `chrome-anime-auto.json` | Chrome 156, 240 Hz display, RTX 4060 Ti | the Framegen extension on the engine runtime, 1080p anime fixture (`anime_1080_12m.webm`, 24 fps), Auto, 720p network, 30 s after a 1 s warm-up. The extension measured 101.7 presents/s, 0 drops, factor up to 6 |
| `firefox-anime-auto.json` | Firefox 159, the same display and GPU | the same session; the 7 ms present limit caps the capacity at 142.86 Hz, so the factor peaks at 5. The extension measured 91.8 presents/s, 0.1 drops/s |
| `player-anime-auto.json` + `.report.json` | the reference native host (`examples/player`, a window with timer ticks at the monitor's 240 Hz), Vulkan on the 4060 Ti | 12 s of the same video, Auto, GPU costs from timestamp queries (pair 1.37 ms, mid 0.25 ms): factor up to 5, about 106 presents/s, no drops. The player wrote its own report next to the trace and the replay reproduces it byte for byte |
| `firefox-refhost-auto.json` | Firefox 159, the same display and GPU | the reference web host (`examples/web`, nothing on the page but the Stream and the Pacer), 60 s of the same video, Auto: no compositor stalls in a minute, one drop, about 84 presents/s. The extension's remaining Firefox stalls are therefore its own page work, not the engine or WebGPU |

Recorded 2026-11-01 (the player's session again 2026-11-09, for the even pacing) through the browser harness: the extension's product-bench bridge (`trace` command), the
reference web host's `window.fhHost.trace()` and the player's `trace=` option. Times
are rounded to 0.01 ms. A new host finding becomes a trace here and an assertion in the tests.
