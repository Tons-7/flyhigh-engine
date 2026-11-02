//! The Pacer over C: the scheduler of `flyhigh_engine::pacer` with options in a struct, a tick as an out-parameter
//! and the requests of the last call copied into the caller's array. Times are the host's milliseconds. Single-threaded
//! use per handle; a failed call returns -1 or null and `fh_last_error` has the message.
use crate::{guard, set_err};
use flyhigh_engine::pacer::{Frame, IntervalKind, Pacer, PacerOptions, RateMode, Request, SourceFrame, Stage};
use std::ffi::c_char;

/// Options for `fh_pacer_create`; start from `fh_pacer_options_default()`. `mode` 0 auto, 1 a fixed `factor` (1..6),
/// 2 the display's rate, 3 `target_fps`; `fps_limit` 0 = none. Every other field is the Rust option of the same name
/// (ARCHITECTURE.md), flags as 0 / 1.
#[repr(C)]
pub struct FhPacerOptions {
    pub mode: u32,
    pub factor: u32,
    pub target_fps: f64,
    pub max_factor: u32,
    pub fps_limit: f64,
    pub lookahead: u32,
    pub limit_to_display: u32,
    pub fill_display: u32,
    pub display_headroom: f64,
    pub compute_budget: f64,
    pub delay_floor_ms: f64,
    pub delay_max_ms: f64,
    pub lateness_intervals: f64,
    pub queue_capacity: u32,
    pub pressure_tau_ms: f64,
    pub penalty_up: f64,
    pub penalty_burst: f64,
    pub penalty_min_gap_ms: f64,
    pub penalty_max: u32,
    pub recovery_after_ms: f64,
    pub stall_window_ms: f64,
    pub stall_rate_limit: f64,
    pub settle_ms: f64,
    pub motion_adaptive: u32,
    pub min_present_interval_ms: f64,
    pub playback_rate: f64,
    pub mid_cost_seed_ms: f64,
    pub pair_cost_seed_ms: f64,
    pub gpu_timestamps: u32,
    pub low_latency: u32,
}

impl FhPacerOptions {
    pub fn from_rust(o: &PacerOptions) -> FhPacerOptions {
        let (mode, factor, target_fps) = match o.mode {
            RateMode::Auto => (0, 0, 0.0),
            RateMode::Factor(f) => (1, f, 0.0),
            RateMode::DisplayHz => (2, 0, 0.0),
            RateMode::Target(fps) => (3, 0, fps),
        };
        FhPacerOptions {
            mode,
            factor,
            target_fps,
            max_factor: o.max_factor,
            fps_limit: o.fps_limit.unwrap_or(0.0),
            lookahead: o.lookahead as u32,
            limit_to_display: o.limit_to_display as u32,
            fill_display: o.fill_display as u32,
            display_headroom: o.display_headroom,
            compute_budget: o.compute_budget,
            delay_floor_ms: o.delay_floor_ms,
            delay_max_ms: o.delay_max_ms,
            lateness_intervals: o.lateness_intervals,
            queue_capacity: o.queue_capacity as u32,
            pressure_tau_ms: o.pressure_tau_ms,
            penalty_up: o.penalty_up,
            penalty_burst: o.penalty_burst,
            penalty_min_gap_ms: o.penalty_min_gap_ms,
            penalty_max: o.penalty_max,
            recovery_after_ms: o.recovery_after_ms,
            stall_window_ms: o.stall_window_ms,
            stall_rate_limit: o.stall_rate_limit,
            settle_ms: o.settle_ms,
            motion_adaptive: o.motion_adaptive as u32,
            min_present_interval_ms: o.min_present_interval_ms,
            playback_rate: o.playback_rate,
            mid_cost_seed_ms: o.mid_cost_seed_ms,
            pair_cost_seed_ms: o.pair_cost_seed_ms,
            gpu_timestamps: o.gpu_timestamps as u32,
            low_latency: o.low_latency as u32,
        }
    }

    pub fn to_rust(&self) -> Result<PacerOptions, String> {
        let mode = match self.mode {
            0 => RateMode::Auto,
            1 => RateMode::Factor(self.factor),
            2 => RateMode::DisplayHz,
            3 => RateMode::Target(self.target_fps),
            m => return Err(format!("unknown mode {m} (0 auto, 1 factor, 2 display_hz, 3 target)")),
        };
        Ok(PacerOptions {
            mode,
            max_factor: self.max_factor,
            fps_limit: (self.fps_limit > 0.0).then_some(self.fps_limit),
            lookahead: self.lookahead != 0,
            limit_to_display: self.limit_to_display != 0,
            fill_display: self.fill_display != 0,
            display_headroom: self.display_headroom,
            compute_budget: self.compute_budget,
            delay_floor_ms: self.delay_floor_ms,
            delay_max_ms: self.delay_max_ms,
            lateness_intervals: self.lateness_intervals,
            queue_capacity: self.queue_capacity as usize,
            pressure_tau_ms: self.pressure_tau_ms,
            penalty_up: self.penalty_up,
            penalty_burst: self.penalty_burst,
            penalty_min_gap_ms: self.penalty_min_gap_ms,
            penalty_max: self.penalty_max,
            recovery_after_ms: self.recovery_after_ms,
            stall_window_ms: self.stall_window_ms,
            stall_rate_limit: self.stall_rate_limit,
            settle_ms: self.settle_ms,
            motion_adaptive: self.motion_adaptive != 0,
            min_present_interval_ms: self.min_present_interval_ms,
            playback_rate: self.playback_rate,
            mid_cost_seed_ms: self.mid_cost_seed_ms,
            pair_cost_seed_ms: self.pair_cost_seed_ms,
            gpu_timestamps: self.gpu_timestamps != 0,
            low_latency: self.low_latency != 0,
        })
    }
}

/// A source frame for `fh_pacer_source_frame`: `kind` 0 motion, 1 cut, 2 hold; `motion` is used when `has_motion`.
#[repr(C)]
pub struct FhSourceFrame {
    pub id: u32,
    pub pts_ms: f64,
    pub kind: u32,
    pub has_motion: u32,
    pub motion: f64,
    pub skipped: u32,
}

/// A frame to generate: phase `t` of the pair (`previous`, `current`), due at `due_ms`. `retime` 1: the interval's
/// real slot, the Stream may answer that the source is enough (then `fh_pacer_skipped`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FhRequest {
    pub id: u64,
    pub previous: u32,
    pub current: u32,
    pub t: f32,
    pub due_ms: f64,
    pub retime: u32,
    pub probe: u32,
}

/// What a tick decided. `frame_kind` 0 = one of the host's source frames, 1 = an output slot. `requests` frames to
/// generate now: fetch them with `fh_pacer_requests`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FhTick {
    pub has_present: u32,
    pub frame_kind: u32,
    pub frame_id: u32,
    pub due_ms: f64,
    pub late_ms: f64,
    pub mid: u32,
    pub fallback: u32,
    pub factor: u32,
    pub dropped: u32,
    pub requests: u32,
}

pub struct FhPacer {
    pacer: Pacer,
    last: Vec<Request>,
}

fn req(r: &Request) -> FhRequest {
    FhRequest {
        id: r.id,
        previous: r.previous,
        current: r.current,
        t: r.t,
        due_ms: r.due_ms,
        retime: r.retime as u32,
        probe: r.probe as u32,
    }
}

#[no_mangle]
pub extern "C" fn fh_pacer_options_default() -> FhPacerOptions {
    FhPacerOptions::from_rust(&PacerOptions::default())
}

/// # Safety
/// `o` is null (the defaults) or points to a valid `FhPacerOptions`.
#[no_mangle]
pub unsafe extern "C" fn fh_pacer_create(o: *const FhPacerOptions) -> *mut FhPacer {
    guard(std::ptr::null_mut(), || {
        let opts = if o.is_null() { Ok(PacerOptions::default()) } else { unsafe { &*o }.to_rust() };
        let opts = match opts {
            Ok(o) => o,
            Err(e) => {
                set_err(e);
                return std::ptr::null_mut();
            }
        };
        match Pacer::new(&opts) {
            Ok(pacer) => Box::into_raw(Box::new(FhPacer { pacer, last: Vec::new() })),
            Err(e) => {
                set_err(e);
                std::ptr::null_mut()
            }
        }
    })
}

/// # Safety
/// `p` is null or a handle from `fh_pacer_create` that is not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn fh_pacer_destroy(p: *mut FhPacer) {
    guard((), || {
        if !p.is_null() {
            unsafe {
                drop(Box::from_raw(p));
            }
        }
    })
}

/// Changes options live (a new mode restarts the controller and the output grid).
/// # Safety
/// `p` is a live handle; `o` points to a valid `FhPacerOptions`.
#[no_mangle]
pub unsafe extern "C" fn fh_pacer_set_options(p: *mut FhPacer, now: f64, o: *const FhPacerOptions) -> i32 {
    guard(-1, || {
        if p.is_null() || o.is_null() {
            set_err("null argument");
            return -1;
        }
        let (p, o) = unsafe { (&mut *p, &*o) };
        match o.to_rust().and_then(|o| p.pacer.set_options(now, &o).map_err(|e| e.to_string())) {
            Ok(()) => 0,
            Err(e) => {
                set_err(e);
                -1
            }
        }
    })
}

/// The display tick: fills `out` with what to present and how many frames to generate (`fh_pacer_requests`).
/// # Safety
/// `p` is a live handle; `out` points to a writable `FhTick`.
#[no_mangle]
pub unsafe extern "C" fn fh_pacer_tick(p: *mut FhPacer, now: f64, out: *mut FhTick) -> i32 {
    guard(-1, || {
        if p.is_null() || out.is_null() {
            set_err("null argument");
            return -1;
        }
        let p = unsafe { &mut *p };
        let tick = p.pacer.display_tick(now);
        let (frame_kind, frame_id) = match tick.present.map(|pr| pr.frame) {
            Some(Frame::Source(id)) => (0, id),
            Some(Frame::Output(id)) => (1, id),
            None => (0, 0),
        };
        let pr = tick.present;
        p.last = tick.generate;
        unsafe {
            *out = FhTick {
                has_present: pr.is_some() as u32,
                frame_kind,
                frame_id,
                due_ms: pr.map_or(0.0, |x| x.due_ms),
                late_ms: pr.map_or(0.0, |x| x.late_ms),
                mid: pr.is_some_and(|x| x.mid) as u32,
                fallback: pr.is_some_and(|x| x.fallback) as u32,
                factor: tick.factor,
                dropped: tick.dropped,
                requests: p.last.len() as u32,
            };
        }
        0
    })
}

/// A source frame arrived, its kind known. Returns how many requests must be generated before the pair changes
/// (fetch them with `fh_pacer_requests`), or -1.
/// # Safety
/// `p` is a live handle; `f` points to a valid `FhSourceFrame`.
#[no_mangle]
pub unsafe extern "C" fn fh_pacer_source_frame(p: *mut FhPacer, now: f64, f: *const FhSourceFrame) -> i32 {
    guard(-1, || {
        if p.is_null() || f.is_null() {
            set_err("null argument");
            return -1;
        }
        let (p, f) = unsafe { (&mut *p, &*f) };
        let kind = match f.kind {
            0 => IntervalKind::Motion,
            1 => IntervalKind::Cut,
            2 => IntervalKind::Hold,
            k => {
                set_err(format!("unknown interval kind {k} (0 motion, 1 cut, 2 hold)"));
                return -1;
            }
        };
        let frame = SourceFrame {
            id: f.id,
            pts_ms: f.pts_ms,
            kind,
            motion: (f.has_motion != 0).then_some(f.motion),
            skipped: f.skipped,
        };
        p.last = p.pacer.source_frame(now, frame);
        p.last.len() as i32
    })
}

/// Copies the requests of the last `fh_pacer_tick` / `fh_pacer_source_frame` into `buf` (at most `cap`); returns
/// how many there are in all.
/// # Safety
/// `p` is a live handle; `buf` points to `cap` writable `FhRequest`s (may be null when `cap` is 0).
#[no_mangle]
pub unsafe extern "C" fn fh_pacer_requests(p: *const FhPacer, buf: *mut FhRequest, cap: usize) -> i32 {
    guard(-1, || {
        if p.is_null() || (buf.is_null() && cap > 0) {
            set_err("null argument");
            return -1;
        }
        let p = unsafe { &*p };
        for (i, r) in p.last.iter().take(cap).enumerate() {
            unsafe {
                *buf.add(i) = req(r);
            }
        }
        p.last.len() as i32
    })
}

/// The host generated (submitted) the frame of a request into output slot `output`.
/// # Safety
/// `p` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_pacer_generated(p: *mut FhPacer, now: f64, request: u64, output: u32) -> i32 {
    guard(-1, || {
        if p.is_null() {
            set_err("null argument");
            return -1;
        }
        unsafe { &mut *p }.pacer.generated(now, request, Frame::Output(output));
        0
    })
}

/// The Stream answered that the source frame is enough for a request.
/// # Safety
/// `p` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_pacer_skipped(p: *mut FhPacer, request: u64) -> i32 {
    guard(-1, || {
        if p.is_null() {
            set_err("null argument");
            return -1;
        }
        unsafe { &mut *p }.pacer.skipped(request);
        0
    })
}

/// A measured GPU cost: `stage` 0 pair, 1 mid, 2 present. Returns 1 when taken, 0 when ignored (inside the settle
/// window after a mode change: measure again), -1 on error.
/// # Safety
/// `p` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_pacer_cost_sample(p: *mut FhPacer, now: f64, stage: u32, ms: f64) -> i32 {
    guard(-1, || {
        if p.is_null() {
            set_err("null argument");
            return -1;
        }
        let stage = match stage {
            0 => Stage::Pair,
            1 => Stage::Mid,
            2 => Stage::Present,
            s => {
                set_err(format!("unknown stage {s} (0 pair, 1 mid, 2 present)"));
                return -1;
            }
        };
        unsafe { &mut *p }.pacer.cost_sample(now, stage, ms) as i32
    })
}

/// # Safety
/// `p` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_pacer_host_stall(p: *mut FhPacer, now: f64, ms: f64) -> i32 {
    guard(-1, || {
        if p.is_null() {
            set_err("null argument");
            return -1;
        }
        unsafe { &mut *p }.pacer.host_stall(now, ms);
        0
    })
}

/// # Safety
/// `p` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_pacer_mode_change(p: *mut FhPacer, now: f64) -> i32 {
    guard(-1, || {
        if p.is_null() {
            set_err("null argument");
            return -1;
        }
        unsafe { &mut *p }.pacer.mode_change(now);
        0
    })
}

/// # Safety
/// `p` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_pacer_reset(p: *mut FhPacer, now: f64) -> i32 {
    guard(-1, || {
        if p.is_null() {
            set_err("null argument");
            return -1;
        }
        unsafe { &mut *p }.pacer.reset(now);
        0
    })
}

/// Records every call from now on into a `trace v1` named after `host` (NULL = unnamed); `fh_pacer_trace_json` reads it.
/// # Safety
/// `p` is a live handle; `host` is null or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn fh_pacer_record(p: *mut FhPacer, host: *const c_char) -> i32 {
    guard(-1, || {
        if p.is_null() {
            set_err("null argument");
            return -1;
        }
        let host = if host.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(host) }.to_string_lossy().to_string()
        };
        unsafe { &mut *p }.pacer.record(&host);
        0
    })
}

/// The recording so far as JSON (`trace v1`); bytes written, or -1 when not recording or the buffer is too small.
/// # Safety
/// `p` is a live handle; `buf` points to `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn fh_pacer_trace_json(p: *const FhPacer, buf: *mut c_char, cap: usize) -> i32 {
    guard(-1, || {
        if p.is_null() || buf.is_null() {
            set_err("null argument");
            return -1;
        }
        let Some(t) = unsafe { &*p }.pacer.trace() else {
            set_err("not recording (fh_pacer_record first)");
            return -1;
        };
        let s = t.to_json();
        let b = s.as_bytes();
        if b.len() + 1 > cap {
            set_err(format!("buffer too small ({} needed)", b.len() + 1));
            return -1;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(b.as_ptr(), buf as *mut u8, b.len());
            *buf.add(b.len()) = 0;
        }
        b.len() as i32
    })
}

/// The report as JSON (`PacerReport`); bytes written or -1 (`fh_last_error` names the size needed).
/// # Safety
/// `p` is a live handle; `buf` points to `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn fh_pacer_report_json(p: *const FhPacer, buf: *mut c_char, cap: usize) -> i32 {
    guard(-1, || {
        if p.is_null() || buf.is_null() {
            set_err("null argument");
            return -1;
        }
        let s = unsafe { &*p }.pacer.report().to_json();
        let b = s.as_bytes();
        if b.len() + 1 > cap {
            set_err(format!("buffer too small ({} needed)", b.len() + 1));
            return -1;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(b.as_ptr(), buf as *mut u8, b.len());
            *buf.add(b.len()) = 0;
        }
        b.len() as i32
    })
}
