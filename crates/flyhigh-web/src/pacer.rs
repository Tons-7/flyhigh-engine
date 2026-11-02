//! The Pacer in the browser: `FhPacer` over `flyhigh_engine::pacer`, with options, ticks, frames and reports as
//! plain objects (the JSON shapes of the Rust types). Times are `performance.now()` milliseconds.
use flyhigh_engine::pacer::trace::{replay, Trace};
use flyhigh_engine::pacer::{Frame, Pacer, PacerOptions, SourceFrame, Stage};
use wasm_bindgen::prelude::*;

pub(crate) fn to_js<T: serde::Serialize>(v: &T) -> Result<JsValue, JsValue> {
    let s = serde_json::to_string(v).map_err(|e| JsValue::from_str(&e.to_string()))?;
    js_sys::JSON::parse(&s)
}

pub(crate) fn from_js<T: serde::de::DeserializeOwned>(v: &JsValue, what: &str) -> Result<T, JsValue> {
    let text = if v.is_undefined() || v.is_null() {
        "{}".to_string()
    } else {
        js_sys::JSON::stringify(v)
            .map_err(|_| JsValue::from_str(&format!("{what}: not serializable")))?
            .as_string()
            .unwrap_or_default()
    };
    serde_json::from_str(&text).map_err(|e| JsValue::from_str(&format!("{what}: {e}")))
}

/// Options and inputs of the Stream and the Pacer come in snake_case (the JSON of the Rust types) or in camelCase
/// (the JavaScript habit): the top-level keys are folded to snake_case before parsing. Reports and ticks come back
/// in snake_case.
pub(crate) fn from_js_snake<T: serde::de::DeserializeOwned>(v: &JsValue, what: &str) -> Result<T, JsValue> {
    if !v.is_object() || js_sys::Array::is_array(v) {
        return from_js(v, what);
    }
    let src = js_sys::Object::from(v.clone());
    let dst = js_sys::Object::new();
    for key in js_sys::Object::keys(&src).iter() {
        let snake = snake_case(&key.as_string().unwrap_or_default());
        js_sys::Reflect::set(&dst, &JsValue::from_str(&snake), &js_sys::Reflect::get(&src, &key)?)?;
    }
    from_js(&dst.into(), what)
}

fn snake_case(key: &str) -> String {
    let mut s = String::with_capacity(key.len() + 4);
    for c in key.chars() {
        if c.is_ascii_uppercase() {
            s.push('_');
            s.push(c.to_ascii_lowercase());
        } else {
            s.push(c);
        }
    }
    s
}

fn err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// The scheduler between a video's frames and the display's ticks (ARCHITECTURE.md). Per
/// `requestAnimationFrame`: `tick(now)`, present what it says, generate its requests through the generator and
/// `generated(now, id, outputSlot)` each. Per video frame, once its kind is known: `sourceFrame(now, frame)`,
/// generate the requests it returns, then push the frame.
#[wasm_bindgen]
pub struct FhPacer {
    pacer: Pacer,
}

#[wasm_bindgen]
impl FhPacer {
    /// `options`: any subset of the Pacer options (`{ mode: 'auto' | 'display_hz' | { factor: 3 } | { target: 120 },
    /// fpsLimit, minPresentIntervalMs, ... }`, camelCase or snake_case); unknown keys are rejected with the valid list.
    #[wasm_bindgen(constructor)]
    pub fn new(options: JsValue) -> Result<FhPacer, JsValue> {
        let o: PacerOptions = from_js_snake(&options, "pacer options")?;
        Pacer::new(&o).map(|pacer| FhPacer { pacer }).map_err(err)
    }

    /// Changes options live; a new mode restarts the controller and the output grid.
    #[wasm_bindgen(js_name = setOptions)]
    pub fn set_options(&mut self, now: f64, options: JsValue) -> Result<(), JsValue> {
        let o: PacerOptions = from_js_snake(&options, "pacer options")?;
        self.pacer.set_options(now, &o).map_err(err)
    }

    pub fn options(&self) -> Result<JsValue, JsValue> {
        to_js(self.pacer.options())
    }

    /// `{ present: { frame: { source: id } | { output: id }, due_ms, late_ms, mid, fallback } | null,
    ///    generate: [{ id, previous, current, t, due_ms, retime, probe }], factor, dropped }`
    pub fn tick(&mut self, now: f64) -> Result<JsValue, JsValue> {
        to_js(&self.pacer.display_tick(now))
    }

    /// `frame`: `{ id, ptsMs, kind: 'motion' | 'cut' | 'hold', motion, skipped }`. Returns the requests to generate
    /// before the pair changes.
    #[wasm_bindgen(js_name = sourceFrame)]
    pub fn source_frame(&mut self, now: f64, frame: JsValue) -> Result<JsValue, JsValue> {
        let f: SourceFrame = from_js_snake(&frame, "source frame")?;
        to_js(&self.pacer.source_frame(now, f))
    }

    /// The host generated (submitted) the frame of request `request` into output slot `output`.
    pub fn generated(&mut self, now: f64, request: f64, output: u32) {
        self.pacer.generated(now, request as u64, Frame::Output(output));
    }

    /// The generator answered that the source frame is enough for the request.
    pub fn skipped(&mut self, request: f64) {
        self.pacer.skipped(request as u64);
    }

    /// A measured GPU cost of `stage` ('pair' | 'mid' | 'present'). Returns whether it was taken (not inside the
    /// settle window after a mode change).
    #[wasm_bindgen(js_name = costSample)]
    pub fn cost_sample(&mut self, now: f64, stage: &str, ms: f64) -> Result<bool, JsValue> {
        let s = match stage {
            "pair" => Stage::Pair,
            "mid" => Stage::Mid,
            "present" => Stage::Present,
            other => return Err(JsValue::from_str(&format!("unknown stage '{other}' (pair, mid, present)"))),
        };
        Ok(self.pacer.cost_sample(now, s, ms))
    }

    #[wasm_bindgen(js_name = hostStall)]
    pub fn host_stall(&mut self, now: f64, ms: f64) {
        self.pacer.host_stall(now, ms);
    }

    /// Fullscreen, resize: drops are exempt for the settle window.
    #[wasm_bindgen(js_name = modeChange)]
    pub fn mode_change(&mut self, now: f64) {
        self.pacer.mode_change(now);
    }

    /// Seek, new source.
    pub fn reset(&mut self, now: f64) {
        self.pacer.reset(now);
    }

    pub fn report(&self) -> Result<JsValue, JsValue> {
        to_js(&self.pacer.report())
    }

    /// Records every call from now on into a `trace v1` object named after `host` (a user agent, a page), with the
    /// options in force: what a bug report attaches, and what `fh_pacer_replay` and `flyhigh-dev replay` reproduce.
    pub fn record(&mut self, host: &str) {
        self.pacer.record(host);
    }

    /// The recording so far as a plain object, or `null` when not recording.
    pub fn trace(&self) -> Result<JsValue, JsValue> {
        match self.pacer.trace() {
            Some(t) => to_js(t),
            None => Ok(JsValue::NULL),
        }
    }

    /// Ends the recording and returns it (`null` when there was none).
    #[wasm_bindgen(js_name = stopRecording)]
    pub fn stop_recording(&mut self) -> Result<JsValue, JsValue> {
        match self.pacer.stop_recording() {
            Some(t) => to_js(&t),
            None => Ok(JsValue::NULL),
        }
    }

    pub fn factor(&self) -> u32 {
        self.pacer.factor()
    }

    pub fn penalty(&self) -> u32 {
        self.pacer.penalty()
    }

    #[wasm_bindgen(js_name = delayMs)]
    pub fn delay_ms(&self) -> f64 {
        self.pacer.delay_ms()
    }
}

/// Replays a `trace v1` object through a fresh Pacer: the report and the per-second table, as `flyhigh-dev replay
/// --json` prints them. Options: the trace's own when `options` is null.
#[wasm_bindgen(js_name = fh_pacer_replay)]
pub fn fh_pacer_replay(trace: JsValue, options: JsValue) -> Result<JsValue, JsValue> {
    let t: Trace = from_js(&trace, "trace")?;
    let o: PacerOptions = if options.is_undefined() || options.is_null() {
        t.options.clone().unwrap_or_default()
    } else {
        from_js_snake(&options, "pacer options")?
    };
    to_js(&replay(&t, &o).map_err(err)?)
}
