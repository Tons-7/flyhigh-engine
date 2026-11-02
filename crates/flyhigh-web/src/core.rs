//! The engine's layers as JavaScript classes (docs/design/07-surfaces.md): `FhEngine` (the device), `FhModel`, `FhSession` (two frames
//! in, one frame out), `FhStream` (frames with timestamps in, intervals out); `FhPacer` lives in pacer.rs. Options,
//! infos and reports are plain objects with the JSON shapes of the Rust types; unknown option keys are rejected.
//! The `fh_*` function families are the previous surface and stay for one release; do not mix them with the classes
//! on one page (each `FhEngine.create` makes a device).
use crate::pacer::{from_js, from_js_snake, to_js};
use flyhigh_engine::gpu::readback::Readback;
use flyhigh_engine::stream::{Classification, Generated, Push, Stream, StreamOptions};
use flyhigh_engine::{autotune, format_tune, TimedRun, TuneRequest};
use flyhigh_engine::{
    Engine, EngineOptions, Gpu, Internal, Model, Precision, RenderSampling, Session, SessionOptions, StaticProtection,
};
use serde::Deserialize;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

fn err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn webgpu_device(gpu: &Gpu) -> Result<JsValue, JsValue> {
    gpu.device
        .as_webgpu()
        .map(|d| JsValue::from(d.clone()))
        .ok_or_else(|| JsValue::from_str("not on the WebGPU backend"))
}

fn webgpu_texture(t: &wgpu::Texture) -> Result<JsValue, JsValue> {
    t.as_webgpu().map(|t| JsValue::from(t.clone())).ok_or_else(|| JsValue::from_str("not on the WebGPU backend"))
}

/// A video, canvas or ImageBitmap frame into an rgba8 texture of `w` x `h`.
fn copy_external(gpu: &Gpu, texture: &wgpu::Texture, w: u32, h: u32, src: wgpu::ExternalImageSource) {
    gpu.queue.copy_external_image_to_texture(
        &wgpu::CopyExternalImageSourceInfo { source: src, origin: wgpu::Origin2d::ZERO, flip_y: false },
        wgpu::CopyExternalImageDestInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
            color_space: wgpu::PredefinedColorSpace::Srgb,
            premultiplied_alpha: false,
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
}

fn rgba_promise(rb: Readback<Vec<u8>>) -> js_sys::Promise {
    future_to_promise(async move {
        let v = rb.finish().await.map_err(err)?;
        Ok(js_sys::Uint8Array::from(&v[..]).into())
    })
}

fn push_to_js(p: &Push) -> Result<JsValue, JsValue> {
    let text = match p.interval {
        Some(iv) => format!(
            r#"{{"ready":true,"baseSlot":{},"statsPending":{},"startPts":{},"endPts":{}}}"#,
            iv.base, p.stats_pending, iv.start_pts, iv.end_pts
        ),
        None => format!(
            r#"{{"ready":false,"baseSlot":-1,"statsPending":{},"startPts":null,"endPts":null}}"#,
            p.stats_pending
        ),
    };
    js_sys::JSON::parse(&text)
}

/// An option value that must be an unsigned 32-bit integer; anything else is an error, never a nearby number.
fn u32_value(v: &serde_json::Value, what: &str) -> Result<u32, JsValue> {
    v.as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| JsValue::from_str(&format!("{what} must be an unsigned 32-bit integer, got {v}")))
}

fn timed_promise(run: Option<TimedRun>) -> js_sys::Promise {
    match run {
        Some(r) => future_to_promise(async move { Ok(JsValue::from_f64(r.finish().await)) }),
        None => js_sys::Promise::resolve(&JsValue::NULL),
    }
}

/// `{ precision: 'auto' | 'f16' | 'f32', tensorCores, autotune }`.
#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields, rename_all = "camelCase")]
struct EngineOptionsJs {
    precision: Option<String>,
    tensor_cores: Option<bool>,
    autotune: Option<bool>,
}

impl EngineOptionsJs {
    fn into_rust(self) -> Result<EngineOptions, JsValue> {
        let precision = match self.precision.as_deref() {
            None | Some("auto") => Precision::Auto,
            Some("f16") => Precision::F16,
            Some("f32") => Precision::F32,
            Some(p) => return Err(JsValue::from_str(&format!("precision '{p}' is not auto, f16 or f32"))),
        };
        let d = EngineOptions::default();
        Ok(EngineOptions {
            precision,
            tensor_cores: self.tensor_cores.unwrap_or(d.tensor_cores),
            autotune: self.autotune.unwrap_or(d.autotune),
            ..d
        })
    }
}

/// `{ width, height, internal: 'auto' | 'frame' | [w, h], k, repair, render: 'model' | 'bilinear' | 'bicubic',
///    staticProtection: false | true | { frames, tolerance }, fastRender, linearT, keepMaps, protectionMask, tune }`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct SessionOptionsJs {
    width: u32,
    height: u32,
    #[serde(default)]
    internal: Option<serde_json::Value>,
    #[serde(default)]
    k: Option<u32>,
    #[serde(default)]
    repair: Option<u32>,
    #[serde(default)]
    render: Option<String>,
    #[serde(default)]
    static_protection: Option<serde_json::Value>,
    #[serde(default)]
    fast_render: Option<bool>,
    #[serde(default)]
    linear_t: Option<bool>,
    #[serde(default)]
    keep_maps: Option<bool>,
    #[serde(default)]
    protection_mask: Option<bool>,
    #[serde(default)]
    tune: Option<String>,
}

impl SessionOptionsJs {
    fn into_rust(self) -> Result<SessionOptions, JsValue> {
        let mut o = SessionOptions::new(self.width, self.height);
        o.internal = match self.internal {
            None | Some(serde_json::Value::Null) => Internal::Auto,
            Some(serde_json::Value::String(s)) if s == "auto" => Internal::Auto,
            Some(serde_json::Value::String(s)) if s == "frame" => Internal::Frame,
            Some(serde_json::Value::Array(a)) if a.len() == 2 => {
                Internal::Size(u32_value(&a[0], "internal")?, u32_value(&a[1], "internal")?)
            }
            Some(v) => return Err(JsValue::from_str(&format!("internal must be 'auto', 'frame' or [w, h], got {v}"))),
        };
        if let Some(k) = self.k {
            o.k = k;
        }
        if let Some(r) = self.repair {
            o.repair = r;
        }
        o.render = match self.render.as_deref() {
            None | Some("model") => RenderSampling::Model,
            Some("bilinear") => RenderSampling::Bilinear,
            Some("bicubic") => RenderSampling::Bicubic,
            Some(r) => return Err(JsValue::from_str(&format!("render '{r}' is not model, bilinear or bicubic"))),
        };
        o.static_protection = match self.static_protection {
            None | Some(serde_json::Value::Null) | Some(serde_json::Value::Bool(false)) => None,
            Some(serde_json::Value::Bool(true)) => Some(StaticProtection::default()),
            Some(serde_json::Value::Object(m)) => {
                let d = StaticProtection::default();
                Some(StaticProtection {
                    frames: m.get("frames").map_or(Ok(d.frames), |v| u32_value(v, "staticProtection.frames"))?,
                    tolerance: m
                        .get("tolerance")
                        .map_or(Ok(d.tolerance), |v| u32_value(v, "staticProtection.tolerance"))?,
                })
            }
            Some(v) => {
                return Err(JsValue::from_str(&format!(
                    "staticProtection must be a boolean or {{ frames, tolerance }}, got {v}"
                )))
            }
        };
        o.fast_render = self.fast_render.unwrap_or(false);
        o.linear_t = self.linear_t.unwrap_or(false);
        o.keep_maps = self.keep_maps.unwrap_or(false);
        o.protection_mask = self.protection_mask.unwrap_or(false);
        o.tune = self.tune.as_deref().filter(|t| !t.is_empty()).and_then(flyhigh_engine::tune_table_from);
        o.validate().map_err(err)?;
        Ok(o)
    }
}

/// The device the engine runs on; one per page. `await FhEngine.create(options)`.
#[wasm_bindgen]
pub struct FhEngine {
    inner: Rc<Engine>,
}

#[wasm_bindgen]
impl FhEngine {
    /// Creates the WebGPU device (`{ precision: 'auto' | 'f16' | 'f32' }`; the device needs `shader-f16` for the f16
    /// path and falls back to f32 without it).
    pub async fn create(options: JsValue) -> Result<FhEngine, JsValue> {
        console_error_panic_hook::set_once();
        let o: EngineOptionsJs = from_js(&options, "engine options")?;
        let engine = Engine::new_async(&o.into_rust()?).await.map_err(err)?;
        Ok(FhEngine { inner: Rc::new(engine) })
    }

    /// The `GPUDevice`, to present, sample and copy on the same device.
    pub fn device(&self) -> Result<JsValue, JsValue> {
        webgpu_device(self.inner.gpu())
    }

    /// `{ name, class: 'tc-f16' | 'portable-f16' | 'portable-f32', tensor_cores, timestamps, bgra_storage }`
    pub fn info(&self) -> Result<JsValue, JsValue> {
        to_js(&self.inner.info())
    }

    pub fn report(&self) -> Result<JsValue, JsValue> {
        to_js(&self.inner.report())
    }

    /// A model from its three files: `graph.json` text, the manifest text and the weight bytes.
    #[wasm_bindgen(js_name = loadModel)]
    pub fn load_model(&self, graph: String, manifest: String, weights: Vec<u8>) -> Result<FhModel, JsValue> {
        let m = Model::from_bytes(graph.as_bytes(), manifest.as_bytes(), weights).map_err(err)?;
        Ok(FhModel { inner: Rc::new(m) })
    }

    /// The plans of `model` for one frame size.
    pub fn session(&self, model: &FhModel, options: JsValue) -> Result<FhSession, JsValue> {
        let o: SessionOptionsJs = from_js(&options, "session options")?;
        let s = Session::new(&self.inner, &model.inner, &o.into_rust()?).map_err(err)?;
        Ok(FhSession { gpu: self.inner.shared(), inner: s })
    }

    /// Times the conv kernels of one configuration (5 to 15 s of GPU time; run it idle) and resolves with the table as
    /// text for the `tune` session option; an empty string on a device without timestamp queries.
    pub fn autotune(&self, model: &FhModel, options: JsValue) -> Result<js_sys::Promise, JsValue> {
        let o: SessionOptionsJs = from_js(&options, "session options")?;
        let o = o.into_rust()?;
        let (iw, ih) = o.internal_size();
        let req = TuneRequest { w: o.width, h: o.height, k: o.k, t: 0.5, iw, ih, reps: 8 };
        let (engine, model) = (self.inner.clone(), model.inner.clone());
        Ok(future_to_promise(async move {
            let table = autotune(engine.gpu(), model.blob(), model.ops(), &req).await;
            Ok(JsValue::from_str(&format_tune(&table)))
        }))
    }

    /// A Stream of `model`: `session` are the session options, `stream` the stream options
    /// (`{ mode: 'realtime' | 'video', cadence, sceneCut, heldFrames, inputs, outputs }`, camelCase or snake_case).
    pub fn stream(&self, model: &FhModel, session: JsValue, stream: JsValue) -> Result<FhStream, JsValue> {
        let so: SessionOptionsJs = from_js(&session, "session options")?;
        let sto: StreamOptions = from_js_snake(&stream, "stream options")?;
        let s = Stream::open(&self.inner, &model.inner, &so.into_rust()?, &sto).map_err(err)?;
        Ok(FhStream { gpu: self.inner.shared(), inner: Rc::new(RefCell::new(s)) })
    }
}

#[wasm_bindgen]
pub struct FhModel {
    inner: Rc<Model>,
}

#[wasm_bindgen]
impl FhModel {
    /// `{ name, train_step, graph_ops, tensors, render_sampling, format_version }`
    pub fn info(&self) -> Result<JsValue, JsValue> {
        to_js(self.inner.info())
    }
}

/// Two frames in, one frame at phase `t` out, on textures the session owns.
#[wasm_bindgen]
pub struct FhSession {
    gpu: Rc<Gpu>,
    inner: Session,
}

#[wasm_bindgen]
impl FhSession {
    #[wasm_bindgen(js_name = addInput)]
    pub fn add_input(&mut self) -> u32 {
        self.inner.add_input() as u32
    }

    #[wasm_bindgen(js_name = addOutput)]
    pub fn add_output(&mut self) -> u32 {
        self.inner.add_output() as u32
    }

    /// The slot's `GPUTexture` (rgba8unorm).
    #[wasm_bindgen(js_name = inputTexture)]
    pub fn input_texture(&self, slot: u32) -> Result<JsValue, JsValue> {
        if slot as usize >= self.inner.inputs() {
            return Err(JsValue::from_str("bad input slot"));
        }
        webgpu_texture(self.inner.input_texture(slot as usize))
    }

    #[wasm_bindgen(js_name = outputTexture)]
    pub fn output_texture(&self, slot: u32) -> Result<JsValue, JsValue> {
        if slot as usize >= self.inner.outputs() {
            return Err(JsValue::from_str("bad output slot"));
        }
        webgpu_texture(self.inner.output_texture(slot as usize))
    }

    fn push(&self, slot: u32, src: wgpu::ExternalImageSource) -> Result<(), JsValue> {
        if slot as usize >= self.inner.inputs() {
            return Err(JsValue::from_str("bad input slot"));
        }
        let o = self.inner.options();
        copy_external(&self.gpu, self.inner.input_texture(slot as usize), o.width, o.height, src);
        Ok(())
    }

    /// Copies a frame into an input slot (sizes must match). Firefox rejects `<video>` sources: blit through
    /// `importExternalTexture` into `inputTexture(slot)` there.
    #[wasm_bindgen(js_name = pushVideo)]
    pub fn push_video(&self, slot: u32, v: web_sys::HtmlVideoElement) -> Result<(), JsValue> {
        self.push(slot, wgpu::ExternalImageSource::HTMLVideoElement(v))
    }

    #[wasm_bindgen(js_name = pushCanvas)]
    pub fn push_canvas(&self, slot: u32, c: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
        self.push(slot, wgpu::ExternalImageSource::HTMLCanvasElement(c))
    }

    #[wasm_bindgen(js_name = pushBitmap)]
    pub fn push_bitmap(&self, slot: u32, b: web_sys::ImageBitmap) -> Result<(), JsValue> {
        self.push(slot, wgpu::ExternalImageSource::ImageBitmap(b))
    }

    /// RGBA8 bytes into an input slot.
    #[wasm_bindgen(js_name = writeRgba)]
    pub fn write_rgba(&self, slot: u32, rgba: &[u8]) -> Result<(), JsValue> {
        self.inner.write_rgba(slot as usize, rgba).map_err(err)
    }

    /// The protection mask (`protectionMask`): width * height bytes, 0 interpolates, anything else is taken from the
    /// source frames; applies from the next pair.
    #[wasm_bindgen(js_name = writeMask)]
    pub fn write_mask(&self, mask: &[u8]) -> Result<(), JsValue> {
        self.inner.write_mask(mask).map_err(err)
    }

    /// The motion of the interval (`a`, `b`).
    pub fn pair(&self, a: u32, b: u32) -> Result<(), JsValue> {
        self.inner.pair(a as usize, b as usize).map_err(err)
    }

    /// Phase `t` of the last pair into output slot `out`.
    pub fn mid(&self, t: f32, out: u32) -> Result<(), JsValue> {
        self.inner.mid(t, out as usize).map_err(err)
    }

    pub fn generate(&self, a: u32, b: u32, t: f32, out: u32) -> Result<(), JsValue> {
        self.inner.generate(a as usize, b as usize, t, out as usize).map_err(err)
    }

    /// Resolves with the output slot's RGBA8 bytes (a `Uint8Array`).
    #[wasm_bindgen(js_name = readRgba)]
    pub fn read_rgba(&self, out: u32) -> Result<js_sys::Promise, JsValue> {
        Ok(rgba_promise(self.inner.read_rgba(out as usize).map_err(err)?))
    }

    /// Runs the pair stage of (`a`, `b`) and resolves with its GPU milliseconds (timestamp queries; `null` without
    /// them).
    #[wasm_bindgen(js_name = pairTimed)]
    pub fn pair_timed(&self, a: u32, b: u32) -> Result<js_sys::Promise, JsValue> {
        Ok(timed_promise(self.inner.pair_timed_begin(a as usize, b as usize).map_err(err)?))
    }

    /// Runs one more generated frame of the last pair and resolves with its GPU milliseconds (`null` without timestamp
    /// queries).
    #[wasm_bindgen(js_name = midTimed)]
    pub fn mid_timed(&self, t: f32, out: u32) -> Result<js_sys::Promise, JsValue> {
        Ok(timed_promise(self.inner.mid_timed_begin(t, out as usize).map_err(err)?))
    }

    /// Compiles every kernel the session runs (one pair and one mid); await `device.queue.onSubmittedWorkDone()` after
    /// it, before the first real frame.
    #[wasm_bindgen(js_name = warmUp)]
    pub fn warm_up(&mut self) -> Result<(), JsValue> {
        self.inner.warm_up().map_err(err)
    }

    pub fn report(&self) -> Result<JsValue, JsValue> {
        to_js(&self.inner.report())
    }
}

/// Frames with timestamps in, intervals out (docs/design/02-stream.md); the Pacer decides the phases and their times.
#[wasm_bindgen]
pub struct FhStream {
    gpu: Rc<Gpu>,
    inner: Rc<RefCell<Stream>>,
}

#[wasm_bindgen]
impl FhStream {
    #[wasm_bindgen(js_name = addInput)]
    pub fn add_input(&self) -> u32 {
        self.inner.borrow_mut().add_input(&self.gpu) as u32
    }

    #[wasm_bindgen(js_name = addOutput)]
    pub fn add_output(&self) -> u32 {
        self.inner.borrow_mut().add_output(&self.gpu) as u32
    }

    /// The protection mask (`protectionMask` of the session options): width * height bytes, 0 interpolates, anything
    /// else is taken from the source frames; applies from the next pair.
    #[wasm_bindgen(js_name = writeMask)]
    pub fn write_mask(&self, mask: &[u8]) -> Result<(), JsValue> {
        self.inner.borrow().write_mask(&self.gpu, mask).map_err(err)
    }

    #[wasm_bindgen(js_name = inputTexture)]
    pub fn input_texture(&self, slot: u32) -> Result<JsValue, JsValue> {
        let s = self.inner.borrow();
        if slot as usize >= s.inputs() {
            return Err(JsValue::from_str("bad input slot"));
        }
        webgpu_texture(s.input_texture(slot as usize))
    }

    #[wasm_bindgen(js_name = outputTexture)]
    pub fn output_texture(&self, slot: u32) -> Result<JsValue, JsValue> {
        let s = self.inner.borrow();
        if slot as usize >= s.outputs() {
            return Err(JsValue::from_str("bad output slot"));
        }
        webgpu_texture(s.output_texture(slot as usize))
    }

    /// The slot to write the next source frame into: never one the stream still reads nor one in `inUse` (an array
    /// of slots the host still shows); -1 when the ring is exhausted (add inputs).
    #[wasm_bindgen(js_name = nextInput)]
    pub fn next_input(&self, in_use: JsValue) -> i32 {
        let used: Vec<u32> = if in_use.is_undefined() || in_use.is_null() {
            Vec::new()
        } else {
            js_sys::Array::from(&in_use).iter().filter_map(|v| v.as_f64().map(|x| x as u32)).collect()
        };
        self.inner.borrow_mut().next_input(|s| used.contains(&(s as u32))).map_or(-1, |s| s as i32)
    }

    fn push_frame(&self, slot: u32, src: wgpu::ExternalImageSource) -> Result<(), JsValue> {
        let s = self.inner.borrow();
        if slot as usize >= s.inputs() {
            return Err(JsValue::from_str("bad input slot"));
        }
        let o = s.tex_options();
        copy_external(&self.gpu, s.input_texture(slot as usize), o.w, o.h, src);
        Ok(())
    }

    /// Copies a frame into an input slot. Firefox rejects `<video>` sources: blit through `importExternalTexture`
    /// into `inputTexture(slot)` there.
    #[wasm_bindgen(js_name = pushVideo)]
    pub fn push_video(&self, slot: u32, v: web_sys::HtmlVideoElement) -> Result<(), JsValue> {
        self.push_frame(slot, wgpu::ExternalImageSource::HTMLVideoElement(v))
    }

    #[wasm_bindgen(js_name = pushCanvas)]
    pub fn push_canvas(&self, slot: u32, c: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
        self.push_frame(slot, wgpu::ExternalImageSource::HTMLCanvasElement(c))
    }

    #[wasm_bindgen(js_name = pushBitmap)]
    pub fn push_bitmap(&self, slot: u32, b: web_sys::ImageBitmap) -> Result<(), JsValue> {
        self.push_frame(slot, wgpu::ExternalImageSource::ImageBitmap(b))
    }

    /// Classifies the pair (`previous`, `slot`) before pushing it: resolves with `{ kind: 'motion' | 'cut' | 'hold',
    /// mean, max, black, motion }`, or `{ kind: 'motion', busy: true }` when every readback is still in flight. In
    /// Firefox submit an empty command buffer every millisecond while it is pending (`device.queue.submit([])`).
    pub fn classify(&self, previous: u32, slot: u32) -> Result<js_sys::Promise, JsValue> {
        let rb = {
            let mut s = self.inner.borrow_mut();
            if previous as usize >= s.inputs() || slot as usize >= s.inputs() {
                return Err(JsValue::from_str("bad input slot"));
            }
            s.classify(&self.gpu, previous as usize, slot as usize)
        };
        Ok(future_to_promise(async move {
            match rb {
                Some(r) => {
                    let c: Classification = r.finish().await.map_err(err)?;
                    to_js(&c)
                }
                None => js_sys::JSON::parse(r#"{"kind":"motion","busy":true}"#),
            }
        }))
    }

    /// The newest source frame is in `slot` with media timestamp `pts` (ms): runs its pair. Returns
    /// `{ ready, baseSlot, statsPending, startPts, endPts }`.
    pub fn push(&self, slot: u32, pts: f64) -> Result<JsValue, JsValue> {
        let p = self.inner.borrow_mut().push(&self.gpu, slot as usize, pts).map_err(err)?;
        push_to_js(&p)
    }

    /// `push` with the pair's GPU time: the same object plus `cost`, a promise of milliseconds (`null` without
    /// timestamp queries, or when no pair ran). The Pacer's `costSample(now, 'pair', ms)` takes it.
    #[wasm_bindgen(js_name = pushTimed)]
    pub fn push_timed(&self, slot: u32, pts: f64) -> Result<JsValue, JsValue> {
        let (p, run) = self.inner.borrow_mut().push_timed(&self.gpu, slot as usize, pts).map_err(err)?;
        let obj = push_to_js(&p)?;
        js_sys::Reflect::set(&obj, &JsValue::from_str("cost"), &timed_promise(run))?;
        Ok(obj)
    }

    /// `generate` with its GPU time: `{ written, cost }`, `cost` a promise of milliseconds (`null` without timestamp
    /// queries, when nothing was written, or for a cadence composite). The Pacer's `costSample(now, 'mid', ms)` takes it; its
    /// probe requests are meant for this call.
    #[wasm_bindgen(js_name = generateTimed)]
    pub fn generate_timed(&self, t: f32, out: u32) -> Result<JsValue, JsValue> {
        let (g, run) = self.inner.borrow_mut().generate_timed(&self.gpu, t, out as usize).map_err(err)?;
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &JsValue::from_str("written"), &JsValue::from_bool(g == Generated::Written))?;
        js_sys::Reflect::set(&obj, &JsValue::from_str("cost"), &timed_promise(run))?;
        Ok(obj.into())
    }

    /// Video + cadence: resolves when the pending pair statistics are in (await it after a push with `statsPending`,
    /// before the interval's first `generate`).
    pub fn stats(&self) -> js_sys::Promise {
        let run = self.inner.borrow_mut().take_stats();
        let inner = self.inner.clone();
        future_to_promise(async move {
            if let Some(run) = run {
                let s = run.finish().await.map_err(err)?;
                inner.borrow_mut().set_stats(s);
            }
            Ok(JsValue::UNDEFINED)
        })
    }

    /// Whether the current interval's real slot is re-timed (present `generate(0, out)` instead of the source).
    #[wasm_bindgen(js_name = realReplaced)]
    pub fn real_replaced(&self) -> bool {
        self.inner.borrow().real_replaced()
    }

    /// Phase `t` of the current interval into output slot `out`; false = nothing written (t = 0 without re-timing:
    /// the source frame is enough).
    pub fn generate(&self, t: f32, out: u32) -> Result<bool, JsValue> {
        let mut s = self.inner.borrow_mut();
        if out as usize >= s.outputs() {
            return Err(JsValue::from_str("bad output slot"));
        }
        Ok(s.generate(&self.gpu, t, out as usize).map_err(err)? == flyhigh_engine::stream::Generated::Written)
    }

    /// Resolves with an output slot's RGBA8 bytes (tests; a player presents the texture).
    #[wasm_bindgen(js_name = readRgba)]
    pub fn read_rgba(&self, out: u32) -> Result<js_sys::Promise, JsValue> {
        let s = self.inner.borrow();
        if out as usize >= s.outputs() {
            return Err(JsValue::from_str("bad output slot"));
        }
        let (w, h) = (s.tex_options().w, s.tex_options().h);
        let bpr = (w * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let stage = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fh_stream_read"),
            size: (bpr * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: s.output_texture(out as usize),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &stage,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.gpu.queue.submit(Some(enc.finish()));
        let rb = Readback::new(
            stage,
            Box::new(move |d: &[u8]| {
                let row = (w * 4) as usize;
                let mut v = Vec::with_capacity(row * h as usize);
                for y in 0..h as usize {
                    let o = y * bpr as usize;
                    v.extend_from_slice(&d[o..o + row]);
                }
                v
            }),
            None,
        );
        Ok(rgba_promise(rb))
    }

    /// Compiles every kernel the stream runs (the warm-up pushes and one mid on its own slots, then `reset`); await
    /// `device.queue.onSubmittedWorkDone()` after it, before the first real frame.
    #[wasm_bindgen(js_name = warmUp)]
    pub fn warm_up(&self) -> Result<(), JsValue> {
        self.inner.borrow_mut().warm_up(&self.gpu).map_err(err)
    }

    /// Forget the pushed frames after a seek or a decoder gap.
    pub fn reset(&self) {
        self.inner.borrow_mut().reset();
    }

    pub fn report(&self) -> Result<JsValue, JsValue> {
        to_js(&self.inner.borrow().report())
    }

    pub fn options(&self) -> Result<JsValue, JsValue> {
        to_js(self.inner.borrow().options())
    }
}
