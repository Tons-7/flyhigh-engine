# Reference web host

A video element to a canvas through the engine's classes `FhEngine`, `FhModel`, `FhStream` and `FhPacer` and the
browser adapters of `adapters.js` (shipped next to the module), in one file (`main.js`). It is what a browser host has to do and nothing else: write source frames into the Stream's slots,
classify each pair, hand frames with their timestamps to the Pacer, feed it the display ticks, present what it says,
generate what it asks for. It is the template for a player or an extension and the bench where browser findings are
measured (the Firefox present spacing and readback nudge live here as host code, see
[ARCHITECTURE.md](../../ARCHITECTURE.md)).

## Run

```sh
bash tools/build_web.sh            # examples/web/pkg/ (not in git)
python -m http.server 8000         # from the repository root
# open http://localhost:8000/examples/web/?autoplay=1
```

`?src=<url>` picks the video (default: the anime fixture in `web/fixtures/`), the rate menu the Pacer mode, the
stream menu realtime or video + cadence. The status line shows presents per second, the factor, penalties, the
display and source rates and the Pacer's delay. `window.fhHost.report()`, `.stream()` and `.trace()` give the Pacer
report, the Stream report and the session as a `trace v1` object for `flyhigh-dev replay`.

Requirements: WebGPU with `shader-f16` (Chrome / Edge 120+, Firefox 141+ on Windows).
