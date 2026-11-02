# JavaScript API (browser, WebGPU)

`examples/web/pkg/flyhigh_web.js` is a wasm-bindgen ES module built by `tools/build_web.sh` (not in git; rebuild:
[development.md](development.md)). The engine's layers are classes: `FhEngine` (the device), `FhModel`, `FhSession`
(two frames in, one frame out), `FhStream` (frames with timestamps in, intervals out) and `FhPacer` (what to present
on each tick, what to generate). Options, infos and reports are plain objects with the JSON shapes of the Rust types;
option and input objects take their keys in camelCase or in snake_case, reports and ticks come back in snake_case;
unknown option keys are rejected with the valid list. One `FhEngine` per page. Calls are synchronous unless they return
a promise. Options and defaults: [options.md](options.md). The browser pieces around the classes (how a `<video>`
gets into a slot in each browser, the canvas, the ticks, the readbacks Firefox sits on, the cost samples) are
`adapters.js`, shipped next to the module (below).

Requirements: WebGPU (Chrome / Edge 120+, Firefox 141+ on Windows). With the `shader-f16` feature the engine runs its
f16 kernels, without it the f32 kernels. Chrome MV3 extensions need `'wasm-unsafe-eval'` in their CSP.

```js
import init, * as fh from './pkg/flyhigh_web.js';

await init();
const engine = await fh.FhEngine.create({});                       // the WebGPU device
const device = engine.device();                                      // present, sample and copy on the same device
const [bin, man, graph] = await Promise.all([
  fetch('models/flyhighfg.bin').then(r => r.arrayBuffer()),
  fetch('models/flyhighfg.json').then(r => r.text()),
  fetch('models/graph.json').then(r => r.text())]);
const model = engine.loadModel(graph, man, new Uint8Array(bin));
console.log(engine.info(), model.info());                            // { name, class: 'tc-f16' | 'portable-f16' | 'portable-f32', ... }
```

## `FhEngine`

| Call | |
|---|---|
| `await FhEngine.create(options)` | the device; `{ precision: 'auto' \| 'f16' \| 'f32', tensorCores, autotune }` |
| `engine.device()` | the `GPUDevice` |
| `engine.info()`, `engine.report()` | the device, its kernel class, timestamp queries; kernels compiled so far, autotune entries, tensor bytes |
| `engine.loadModel(graphText, manifestText, weightBytes)` | an `FhModel`; the loader checks the format version and the tensor table. `model.info()`: name, training step, graph size, render sampler |
| `engine.session(model, sessionOptions)` | an `FhSession` |
| `engine.stream(model, sessionOptions, streamOptions)` | an `FhStream` |
| `await engine.autotune(model, sessionOptions)` | times the conv kernels of that configuration (5 to 15 s of GPU time; run it idle, once per GPU and size) and resolves with the table as text: store it (for example in `localStorage`, keyed by GPU, size and `k`) and pass it as the `tune` session option. Resolves with an empty string on a device without timestamp queries |

Session options: `{ width, height, internal: 'auto' | 'frame' | [w, h], k, repair, render: 'model' | 'bilinear' |
'bicubic', staticProtection: false | true | { frames, tolerance }, fastRender, linearT, keepMaps, protectionMask,
tune }`. Only `width` and `height` are required; the defaults are those of [options.md](options.md) (`internal` auto =
the 480-line rung above 960 px, `k` 1, `repair` 1).

## `FhSession`: two frames in, one frame out

```js
const s = engine.session(model, { width: 1920, height: 1080 });
const [a, b] = [s.addInput(), s.addInput()], out = s.addOutput();   // rgba8 textures the session owns
s.pushVideo(a, videoA); s.pushVideo(b, videoB);                     // or write into s.inputTexture(a) yourself
s.pair(a, b);                                                       // the motion of the interval: the expensive half
s.mid(0.5, out);                                                    // phase 0.5 into the output slot: the cheap half
present(s.outputTexture(out));                                      // or: const rgba = await s.readRgba(out);
```

`addInput` / `addOutput`, `inputTexture(slot)` / `outputTexture(slot)`, `pushVideo` / `pushCanvas` / `pushBitmap(slot,
source)` (Firefox rejects `<video>` sources: blit through `importExternalTexture` into `inputTexture(slot)` there),
`writeRgba(slot, bytes)`, `writeMask(bytes)` (the protection mask, one byte per pixel), `pair(a, b)`, `mid(t, out)`,
`generate(a, b, t, out)`, `await readRgba(out)`, `warmUp()` (compile every kernel before the first real frame; await
`device.queue.onSubmittedWorkDone()` after it), `report()`. `await pairTimed(a, b)` and `await midTimed(t, out)` run
the stage and resolve with its GPU milliseconds from timestamp queries (`null` on a device without them): the numbers
a host feeds the Pacer's `costSample`, and the way to measure the engine in a page.

## `FhStream`: frames with timestamps in, intervals out

```js
const stream = engine.stream(model, { width, height, fastRender: true, linearT: true },
                             { mode: 'realtime', inputs: 12, outputs: 8 });
stream.warmUp(); await device.queue.onSubmittedWorkDone();
// per video frame (requestVideoFrameCallback):
const slot = stream.nextInput(slotsStillShown);                     // the ring slot to write into
stream.pushVideo(slot, video);
const c = await stream.classify(previousSlot, slot);                // { kind: 'motion' | 'cut' | 'hold', mean, max, black, motion }
const flush = pacer.sourceFrame(now, { id: slot, pts_ms: metadata.mediaTime * 1000, kind: c.kind, motion: c.motion });
// generate `flush` first (mids of the previous pair), then:
const p = stream.push(slot, metadata.mediaTime * 1000);            // { ready, baseSlot, statsPending, startPts, endPts }
if (p.statsPending) await stream.stats();                           // video + cadence
```

Stream options: `{ mode: 'realtime' | 'video', cadence, scene_cut: { threshold } | null, held_frames: { mean, max }
| null, inputs, outputs }` (defaults: realtime, scene cuts on at 0.08, held frames off, four slots each).
`generate(t, out)` returns `false` when nothing was written (t = 0 without re-timing: the source frame is enough);
`realReplaced()`, `reset()` (seek, decoder gap), `writeMask(bytes)`, `await readRgba(out)`, `report()`, `options()`.
`pushTimed(slot, pts)` and `generateTimed(t, out)` do the same work under timestamp queries and add `cost`, a promise
of the GPU milliseconds (`null` without timestamps, when nothing ran, or for a cadence composite): what the Pacer's `costSample` wants, for
its probe requests and now and then otherwise (`generateTimed` returns `{ written, cost }`). In Firefox a pending
readback (`classify`, `stats`, a `cost`) resolves only on GPU polls (70 to 100 ms); submit an empty command buffer
every millisecond while one is pending (`device.queue.submit([])`) and it resolves in 5 ms.

## `FhPacer`: what to present, what to generate

The scheduler between a video's frames and the display's ticks ([ARCHITECTURE.md](../ARCHITECTURE.md)). Pure logic with
`performance.now()` as its clock.

```js
const pacer = new fh.FhPacer({ mode: 'auto', min_present_interval_ms: isFirefox ? 7 : 0 });

function onTick(now) {                                     // requestAnimationFrame
  const tick = pacer.tick(now);
  if (tick.present) present(tick.present.frame);           // { source: slot } or { output: slot }
  for (const r of tick.generate) {                         // { id, previous, current, t, due_ms, retime, probe }
    const out = nextOutput();
    if (stream.generate(r.t, out)) pacer.generated(now, r.id, out); else pacer.skipped(r.id);
  }
}
```

`costSample(now, 'pair' | 'mid' | 'present', ms)` feeds measured GPU times (returns whether the sample was taken: not
inside the settle window), `modeChange(now)` opens the settle window after a fullscreen change, `reset(now)` forgets the
history on a seek, `report()` returns what was decided, measured and dropped, and why; `setOptions(now, options)`
changes options live. `record(host)` makes the pacer record every call from then on as a `trace v1` object with the
options in force; `trace()` reads it, `stopRecording()` ends and returns it: the one thing to attach to a bug report.
`fh_pacer_replay(trace, options)` replays such an object (the per-second table and the report), the same numbers as
`flyhigh-dev replay --json`. The whole loop in one file: `examples/web/main.js`.

## Adapters (`adapters.js`)

A plain ES module next to `flyhigh_web.js`, no build step. It holds what every WebGPU host needs around the classes
and nothing the engine should know about ([ARCHITECTURE.md](../ARCHITECTURE.md)):

| Export | Does |
|---|---|
| `pacerDefaults()` | the Pacer options this browser needs: Firefox gets `min_present_interval_ms: 7` (its compositor stalls below that) |
| `new ReadbackNudge(device).wrap(promise)` | Firefox resolves `mapAsync` only on GPU polls; empty submits while a readback is pending bring it from 70 to 100 ms down to 5. Wrap `classify`, `stats`, `cost`, `readRgba` |
| `await warmUp(device, sessionOrStream)` | compiles every kernel and waits for the GPU before the first frame |
| `new VideoSource(device, stream).write(slot, video)` | a `<video>` frame into an input slot: `pushVideo` where the browser takes it, an `importExternalTexture` render pass into the slot where it does not (Firefox) |
| `new CanvasPresenter(device, canvas, stream).present(frame)` | the frame a tick names (`{ source }` or `{ output }`) onto the canvas |
| `new TickSource(pacer, onTick).start()` | `requestAnimationFrame` as the Pacer's ticks; a tick later than 50 ms after the previous one is reported as a host stall |
| `new CostSampler(pacer, nudge).push(stream, slot, pts)` / `.generate(stream, request, out)` | `push` and `generate` with a timed run every thirtieth call and on every probe, the GPU milliseconds fed to `costSample` |

`examples/web/main.js` is these pieces plus a page's own UI; a player or an extension starts from it.
