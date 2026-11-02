// The reference web host: a video element to a canvas through the engine's classes: FhEngine (the device), FhModel,
// FhStream (frames with timestamps in, intervals out) and FhPacer (what to present on each tick, what to generate),
// with the browser adapters (adapters.js) for what each browser does differently. Everything a host has to do is in
// this file: source frames in, ticks in, frames out. It is the template for a player or an extension and the bench
// for browser findings (ARCHITECTURE.md).
import init, * as fh from './pkg/flyhigh_web.js';
import { pacerDefaults, ReadbackNudge, warmUp, VideoSource, CanvasPresenter, TickSource, CostSampler } from './pkg/adapters.js';

const $ = (id) => document.getElementById(id);
const video = $('video'), canvas = $('canvas'), status = $('status');
const params = new URL(location.href).searchParams;
const INPUTS = 12, OUTPUTS = 8, HISTORY = 8;

let engine, model, device, nudge, pacer, running = false, stream = null;
const stats = { presents: 0, mids: 0, drops: 0, ticks: 0, sources: 0, lastAt: 0, cuts: 0, holds: 0 };

function pacerOptions() {
  const m = $('mode').value;
  const mode = m === 'auto' ? 'auto' : m === 'display_hz' ? 'display_hz' : m.startsWith('target:') ? { target: +m.slice(7) } : { factor: +m };
  return { ...pacerDefaults(), mode, lookahead: $('smode').value === 'video' };
}

async function start() {
  if (running) return;
  const src = params.get('src') || $('src').value;
  if (video.src !== new URL(src, location.href).href) { video.src = src; }
  await video.play();
  if (!video.videoWidth) await new Promise(r => video.addEventListener('loadedmetadata', r, { once: true }));
  const w = video.videoWidth, h = video.videoHeight;
  canvas.width = w; canvas.height = h;
  const smode = $('smode').value;
  const realtime = smode === 'realtime';
  const obj = engine.stream(model,
    { width: w, height: h, fastRender: realtime, linearT: realtime },
    { mode: smode, cadence: !realtime, inputs: INPUTS, outputs: OUTPUTS });
  pacer = new fh.FhPacer(pacerOptions());
  pacer.record(`reference web host, ${navigator.userAgent}`);
  stream = {
    obj, w, h, prev: -1, recent: [], nextOut: 0,
    source: new VideoSource(device, obj),
    presenter: new CanvasPresenter(device, canvas, obj),
    costs: new CostSampler(pacer, nudge),
    ticks: new TickSource(pacer, onTick),
  };
  Object.assign(stats, { presents: 0, mids: 0, drops: 0, ticks: 0, sources: 0, lastAt: performance.now(), cuts: 0, holds: 0 });
  running = true;
  $('start').disabled = true; $('stop').disabled = false;
  $('info').textContent = `${w}x${h}, ${smode}, ${JSON.stringify(obj.report())}`;
  const t0 = performance.now();
  await warmUp(device, obj);
  console.log(`warm-up ${(performance.now() - t0).toFixed(0)} ms`);
  video.requestVideoFrameCallback(onVideoFrame);
  stream.ticks.start();
}

function stop() {
  running = false;
  $('start').disabled = false; $('stop').disabled = true;
  if (stream) { stream.ticks.stop(); try { stream.obj.free(); } catch { /* already gone */ } }
  stream = null;
}

function generate(now, requests) {
  for (const r of requests) {
    const out = stream.nextOut; stream.nextOut = (out + 1) % OUTPUTS;
    let written = false;
    try { written = stream.costs.generate(stream.obj, r, out); } catch (e) { console.warn('generate', e); }
    if (written) pacer.generated(now, r.id, out); else pacer.skipped(r.id);
  }
}

let busy = false;
async function onVideoFrame(_now, metadata) {
  if (!running) return;
  video.requestVideoFrameCallback(onVideoFrame);
  if (busy) return;
  busy = true;
  try {
    const now = performance.now();
    const pts = metadata.mediaTime * 1000;
    const slot = stream.obj.nextInput(stream.recent);
    if (slot < 0) { console.warn('input ring exhausted'); return; }
    stream.source.write(slot, video);
    let kind = 'motion', motion = undefined;
    if (stream.prev >= 0) {
      const c = await nudge.wrap(stream.obj.classify(stream.prev, slot));
      if (!running) return;
      kind = c.kind; motion = c.motion;
      if (kind === 'cut') stats.cuts++; else if (kind === 'hold') stats.holds++;
    }
    const flush = pacer.sourceFrame(now, { id: slot, ptsMs: pts, kind, motion });   // mids of the previous pair go out before the pair changes
    generate(now, flush);
    const p = stream.costs.push(stream.obj, slot, pts);
    if (p.statsPending) await nudge.wrap(stream.obj.stats());
    stream.prev = slot;
    stream.recent.push(slot); while (stream.recent.length > HISTORY) stream.recent.shift();
    stats.sources++;
  } finally { busy = false; }
}

function onTick(now, tick) {
  if (!running) return;
  stats.ticks++;
  if (tick.present) { stream.presenter.present(tick.present.frame); stats.presents++; if (tick.present.mid) stats.mids++; }
  stats.drops += tick.dropped;
  generate(now, tick.generate);
  if (now - stats.lastAt > 500) {
    const dt = (now - stats.lastAt) / 1000;
    const r = pacer.report();
    status.textContent = [
      `present ${(stats.presents / dt).toFixed(0)}/s (mids ${(stats.mids / dt).toFixed(0)}) drops ${(stats.drops / dt).toFixed(1)}/s ticks ${(stats.ticks / dt).toFixed(0)}/s source ${(stats.sources / dt).toFixed(1)}/s`,
      `factor ${r.factor} penalty ${r.penalty}+${r.strain_penalty} pressure ${r.pressure.toFixed(2)} delay ${r.delay_ms.toFixed(0)} ms late ${r.late_avg_ms.toFixed(1)} ms`,
      `display ${r.display.display_hz} Hz (capacity ${r.display.capacity_hz}, floor ${r.display_floor_ms.toFixed(2)} ms) source ${r.source_hz ? r.source_hz.toFixed(2) : '?'} fps`,
      `costs pair ${r.pair_cost_ms.toFixed(2)} ms mid ${r.mid_cost_ms.toFixed(2)} ms | cuts ${stats.cuts} holds ${stats.holds} motion ${r.motion_avg.toFixed(0)} queued ${r.queued} history ${r.history_frames}` + (r.plan && r.plan.warning ? ` | ${r.plan.warning}` : ''),
    ].join('\n');
    Object.assign(stats, { presents: 0, mids: 0, drops: 0, ticks: 0, sources: 0, lastAt: now });
  }
}

document.addEventListener('fullscreenchange', () => { if (pacer) pacer.modeChange(performance.now()); });
video.addEventListener('seeking', () => { if (pacer && running) { pacer.reset(performance.now()); stream.obj.reset(); stream.prev = -1; } });
$('start').onclick = () => start().catch(e => { status.textContent = 'error: ' + (e.stack || e); console.error(e); });
$('stop').onclick = stop;

// the harness reads these: the reports and the recorded session (trace v1, recorded by the Pacer itself)
window.fhHost = {
  report: () => pacer ? pacer.report() : null,
  stream: () => stream ? stream.obj.report() : null,
  engine: () => engine ? engine.report() : null,
  trace: () => pacer ? pacer.trace() : null,
  running: () => running,
};

(async () => {
  await init();
  const [bin, man, graph] = await Promise.all([
    fetch('../../models/flyhighfg.bin').then(r => r.arrayBuffer()),
    fetch('../../models/flyhighfg.json').then(r => r.text()),
    fetch('../../models/graph.json').then(r => r.text())]);
  engine = await fh.FhEngine.create({});
  device = engine.device();
  nudge = new ReadbackNudge(device);
  model = engine.loadModel(graph, man, new Uint8Array(bin));
  status.textContent = `${JSON.stringify(engine.info())}\n${JSON.stringify(model.info())}\nReady.`;
  if (params.get('autoplay') === '1') start().catch(e => { status.textContent = 'error: ' + (e.stack || e); console.error(e); });
})().catch(e => { status.textContent = 'engine failed: ' + (e.stack || e); console.error(e); });
