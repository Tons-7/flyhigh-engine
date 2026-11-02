//! C ABI of the engine for hosts that are not Rust (C / C++ / C# / Python): the layers (docs/design/00-overview.md) behind opaque
//! handles. Built as `flyhigh.dll` / `libflyhigh.so` / `libflyhigh.dylib` (and a static library); the header is
//! `include/flyhigh.h`, the guide `docs/api-c.md`. Single-threaded use per handle. Errors: a call returns null or -1
//! and `fh_last_error` holds the message.
//!
//! - core.rs: `fh_engine_*` (the device), `fh_model_*`, `fh_session_*` (two frames in, one frame out)
//! - stream.rs: `fh_stream_*` (frames with timestamps in, intervals out)
//! - pacer.rs: `fh_pacer_*` (what to present on this tick, what to generate)
//! - here: `fh_generator_*`, the CPU convenience over `api::Generator`, and `fh_last_error`
mod core;
mod pacer;
mod stream;

pub use core::*;
pub use pacer::*;
pub use stream::*;

use flyhigh_engine::api::{Generator, Mode, Options, OutFrame};
use flyhigh_engine::{Engine, Model};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::{c_char, CString};

thread_local! { static LAST_ERR: RefCell<CString> = RefCell::new(CString::new("").unwrap()); }
pub(crate) fn set_err(e: impl std::fmt::Display) {
    LAST_ERR.with(|l| *l.borrow_mut() = CString::new(e.to_string().replace('\0', " ")).unwrap());
}

/// A string into a caller buffer, NUL-terminated; bytes written (without the NUL) or -1 when `cap` is too small.
/// # Safety
/// `buf` points to `cap` writable bytes.
pub(crate) unsafe fn write_json(s: &str, buf: *mut c_char, cap: usize) -> i32 {
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
}

/// Options for `fh_generator_create`; start from `fh_options_default(w, h)`. `k` 1..3 (0 = default 1); `internal_w/h`
/// 0 = automatic (the 480-line rung for frames above 960 px, else the frame size); `mode` 0 = realtime, 1 = video (one
/// frame of lookahead); `repair` 0/1/2 (default 1: a zero-filled struct turns repair OFF); `cadence` 0/1 (video only);
/// `render_sampling` 0 = the model's, 1 = bilinear, 2 = bicubic; `keep_static` 0/1 static protection (HUD, logos), `static_frames` unchanged frames before protecting (0 = default 8),
/// `static_tolerance` levels still counted as unchanged (default 4); `fast_render` 1 = texture-sampled render (same
/// quality, cheaper per generated frame), `linear_t` 1 = the t-dependent network once per pair (with fast_render).
#[repr(C)]
pub struct FhOptions {
    pub width: u32,
    pub height: u32,
    pub k: u32,
    pub internal_w: u32,
    pub internal_h: u32,
    pub mode: u32,
    pub repair: u32,
    pub cadence: u32,
    pub render_sampling: u32,
    pub keep_static: u32,
    pub static_frames: u32,
    pub static_tolerance: u32,
    pub fast_render: u32,
    pub linear_t: u32,
}

pub struct FhGenerator {
    engine: flyhigh_engine::api::Engine,
    gen: Generator,
    pending: VecDeque<OutFrame>,
}

#[no_mangle]
pub extern "C" fn fh_last_error() -> *const c_char {
    LAST_ERR.with(|l| l.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn fh_options_default(width: u32, height: u32) -> FhOptions {
    let o = Options::new(width, height);
    let (iw, ih) = o.internal.unwrap_or((0, 0));
    FhOptions {
        width,
        height,
        k: o.k,
        internal_w: iw,
        internal_h: ih,
        mode: 0,
        repair: o.repair,
        cadence: 0,
        render_sampling: 0,
        keep_static: 0,
        static_frames: o.static_frames,
        static_tolerance: o.static_tolerance,
        fast_render: 0,
        linear_t: 0,
    }
}

/// The CPU generator of `m` on `e`: frames in and out as RGBA8 bytes. NULL on error.
/// # Safety
/// `e` and `m` are live handles; `o` points to a valid `FhOptions`.
#[no_mangle]
pub unsafe extern "C" fn fh_generator_create(
    e: *const Engine,
    m: *const Model,
    o: *const FhOptions,
) -> *mut FhGenerator {
    if e.is_null() || m.is_null() || o.is_null() {
        set_err("null argument");
        return std::ptr::null_mut();
    }
    let (e, m, o) = unsafe { (&*e, &*m, &*o) };
    let mut opts = Options::new(o.width, o.height);
    if o.k > 0 {
        opts.k = o.k;
    }
    if o.internal_w > 0 && o.internal_h > 0 {
        opts.internal = Some((o.internal_w, o.internal_h));
    }
    opts.mode = if o.mode == 1 { Mode::Video } else { Mode::Realtime };
    opts.repair = o.repair.min(2);
    opts.cadence = o.cadence != 0;
    opts.cubic = match o.render_sampling {
        1 => Some(false),
        2 => Some(true),
        _ => None,
    };
    opts.keep_static = o.keep_static != 0;
    if o.static_frames > 0 {
        opts.static_frames = o.static_frames;
    }
    opts.static_tolerance = o.static_tolerance;
    opts.fast_render = o.fast_render != 0;
    opts.linear_t = o.linear_t != 0;
    let engine = flyhigh_engine::api::Engine::from_core(e, m);
    match Generator::new(&engine, opts) {
        Ok(gen) => Box::into_raw(Box::new(FhGenerator { engine, gen, pending: VecDeque::new() })),
        Err(err) => {
            set_err(err);
            std::ptr::null_mut()
        }
    }
}
/// # Safety
/// `g` is null or a handle from `fh_generator_create` that is not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn fh_generator_destroy(g: *mut FhGenerator) {
    if !g.is_null() {
        unsafe {
            drop(Box::from_raw(g));
        }
    }
}

/// Push one source frame (RGBA8, width*height*4 bytes). Returns the number of output frames now queued (pop them
/// in order before the next push), or -1.
/// # Safety
/// `g` is a live handle; `rgba` points to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn fh_generator_push(g: *mut FhGenerator, rgba: *const u8, len: usize, factor: u32) -> i32 {
    if g.is_null() || rgba.is_null() {
        set_err("null argument");
        return -1;
    }
    let g = unsafe { &mut *g };
    let frame = unsafe { std::slice::from_raw_parts(rgba, len) };
    match g.gen.push(&g.engine, frame, factor) {
        Ok(outs) => {
            g.pending.extend(outs);
            g.pending.len() as i32
        }
        Err(err) => {
            set_err(err);
            -1
        }
    }
}

/// Pop the next output frame into `rgba` (cap >= width*height*4). Returns 1 with `phase` / `real` filled, 0 when the
/// queue is empty, -1 on error.
/// # Safety
/// `g` is a live handle; `rgba` points to `cap` writable bytes; `phase` and `real` are null or writable.
#[no_mangle]
pub unsafe extern "C" fn fh_generator_pop(
    g: *mut FhGenerator,
    rgba: *mut u8,
    cap: usize,
    phase: *mut f32,
    real: *mut i32,
) -> i32 {
    if g.is_null() || rgba.is_null() {
        set_err("null argument");
        return -1;
    }
    let g = unsafe { &mut *g };
    let Some(f) = g.pending.pop_front() else {
        return 0;
    };
    if f.rgba.len() > cap {
        set_err(format!("output buffer too small ({} needed)", f.rgba.len()));
        g.pending.push_front(f);
        return -1;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(f.rgba.as_ptr(), rgba, f.rgba.len());
        if !phase.is_null() {
            *phase = f.phase;
        }
        if !real.is_null() {
            *real = f.real as i32;
        }
    }
    1
}

/// Pushes until the first push that queues frames: 1 (realtime) or 3 (video).
/// # Safety
/// `g` is null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_generator_warmup(g: *const FhGenerator) -> i32 {
    if g.is_null() {
        return -1;
    }
    unsafe { &*g }.gen.warmup() as i32
}
