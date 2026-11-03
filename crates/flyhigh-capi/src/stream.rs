//! The Stream over C: frames with timestamps in, intervals out (CPU RGBA8 frames until the texture interop of phase
//! 4). Single-threaded use per handle; a failed call returns -1 or null and `fh_last_error` has the message.
use crate::core::FhSessionOptions;
use crate::{guard, set_err, write_json};
use flyhigh_engine::stream::{Generated, HeldFrames, IntervalKind, Mode, SceneCut, Shown, Stream, StreamOptions};
use flyhigh_engine::{Engine, Model};
use std::ffi::c_char;

/// Options for `fh_stream_create`; start from `fh_stream_options_default()`. `mode` 0 realtime, 1 video (one frame
/// of lookahead); `cadence` 1 = re-time elements animated on 2s (video); `scene_cut` the mean-difference threshold
/// (0..1) above which a pair is a cut, 0 = never (default 0.08); `held_frames` 1 = detect repeated frames with
/// `held_mean` / `held_max` (0 = the defaults 0.0033 / 0.06); `inputs` / `outputs` slots created up front.
#[repr(C)]
pub struct FhStreamOptions {
    pub mode: u32,
    pub cadence: u32,
    pub scene_cut: f64,
    pub held_frames: u32,
    pub held_mean: f64,
    pub held_max: f64,
    pub inputs: u32,
    pub outputs: u32,
}

impl FhStreamOptions {
    pub fn from_rust(o: &StreamOptions) -> FhStreamOptions {
        let h = o.held_frames.unwrap_or_default();
        FhStreamOptions {
            mode: match o.mode {
                Mode::Realtime => 0,
                Mode::Video => 1,
            },
            cadence: o.cadence as u32,
            scene_cut: o.scene_cut.map_or(0.0, |c| c.threshold as f64),
            held_frames: o.held_frames.is_some() as u32,
            held_mean: h.mean as f64,
            held_max: h.max as f64,
            inputs: o.inputs,
            outputs: o.outputs,
        }
    }

    pub fn to_rust(&self) -> Result<StreamOptions, String> {
        let d = HeldFrames::default();
        Ok(StreamOptions {
            mode: match self.mode {
                0 => Mode::Realtime,
                1 => Mode::Video,
                m => return Err(format!("unknown mode {m} (0 realtime, 1 video)")),
            },
            cadence: self.cadence != 0,
            scene_cut: (self.scene_cut > 0.0).then_some(SceneCut { threshold: self.scene_cut as f32 }),
            held_frames: (self.held_frames != 0).then_some(HeldFrames {
                mean: if self.held_mean > 0.0 { self.held_mean as f32 } else { d.mean },
                max: if self.held_max > 0.0 { self.held_max as f32 } else { d.max },
            }),
            inputs: self.inputs,
            outputs: self.outputs,
        })
    }
}

/// What `fh_stream_classify` found: `kind` 0 motion, 1 cut, 2 hold; `mean` / `max` in 0..1; `motion` in the Pacer's
/// units (0..765).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FhClassification {
    pub kind: u32,
    pub mean: f32,
    pub max: f32,
    pub black: u32,
    pub motion: f64,
}

/// What `fh_stream_push` opened: `ready` 1 = an interval can be generated now, from the real frame in `base_slot`
/// (`start_pts` to `end_pts`); `stats_pending` 1 = call `fh_stream_resolve_stats` before generating (video + cadence).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FhPush {
    pub ready: u32,
    pub base_slot: i32,
    pub stats_pending: u32,
    pub start_pts: f64,
    pub end_pts: f64,
}

#[no_mangle]
pub extern "C" fn fh_stream_options_default() -> FhStreamOptions {
    FhStreamOptions::from_rust(&StreamOptions::default())
}

macro_rules! live {
    ($p:expr) => {
        if $p.is_null() {
            set_err("null argument");
            return -1;
        }
    };
}

/// A Stream of `m` on `e`: `so` are the session options, `o` the stream options. NULL on error.
/// # Safety
/// `e` and `m` are live handles; `so` and `o` point to valid option structs (`so->tune` null or NUL-terminated).
#[no_mangle]
pub unsafe extern "C" fn fh_stream_create(
    e: *const Engine,
    m: *const Model,
    so: *const FhSessionOptions,
    o: *const FhStreamOptions,
) -> *mut Stream {
    guard(std::ptr::null_mut(), || {
        if e.is_null() || m.is_null() || so.is_null() || o.is_null() {
            set_err("null argument");
            return std::ptr::null_mut();
        }
        let opts = unsafe { (*so).to_rust() }.and_then(|s| unsafe { &*o }.to_rust().map(|st| (s, st)));
        let (session, stream) = match opts {
            Ok(v) => v,
            Err(err) => {
                set_err(err);
                return std::ptr::null_mut();
            }
        };
        match Stream::open(unsafe { &*e }, unsafe { &*m }, &session, &stream) {
            Ok(s) => Box::into_raw(Box::new(s)),
            Err(err) => {
                set_err(err);
                std::ptr::null_mut()
            }
        }
    })
}

/// # Safety
/// `s` is null or a handle from `fh_stream_create` that is not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_destroy(s: *mut Stream) {
    guard((), || {
        if !s.is_null() {
            unsafe {
                drop(Box::from_raw(s));
            }
        }
    })
}

/// # Safety
/// `s` is a live handle created on `e`.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_add_input(s: *mut Stream, e: *const Engine) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        unsafe { &mut *s }.add_input(unsafe { &*e }.gpu()) as i32
    })
}

/// # Safety
/// `s` is a live handle created on `e`.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_add_output(s: *mut Stream, e: *const Engine) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        unsafe { &mut *s }.add_output(unsafe { &*e }.gpu()) as i32
    })
}

/// The slot to write the next source frame into: never one the stream still reads nor one of the `n` slots in
/// `in_use` (the host still shows them); -1 when the ring is exhausted.
/// # Safety
/// `s` is a live handle; `in_use` points to `n` readable slot numbers (may be null when `n` is 0).
#[no_mangle]
pub unsafe extern "C" fn fh_stream_next_input(s: *mut Stream, in_use: *const u32, n: usize) -> i32 {
    guard(-1, || {
        live!(s);
        let used: &[u32] =
            if in_use.is_null() || n == 0 { &[] } else { unsafe { std::slice::from_raw_parts(in_use, n) } };
        unsafe { &mut *s }.next_input(|slot| used.contains(&(slot as u32))).map_or(-1, |slot| slot as i32)
    })
}

/// A frame (RGBA8, width * height * 4 bytes) into an input slot; 0 or -1.
/// # Safety
/// `s` is a live handle created on `e`; `rgba` points to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_write_rgba(
    s: *const Stream,
    e: *const Engine,
    slot: u32,
    rgba: *const u8,
    len: usize,
) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        if rgba.is_null() {
            set_err("null argument");
            return -1;
        }
        let (s, gpu) = (unsafe { &*s }, unsafe { &*e }.gpu());
        if slot as usize >= s.inputs() {
            set_err(format!("bad input slot {slot} (the stream has {})", s.inputs()));
            return -1;
        }
        let (w, h) = (s.tex_options().w, s.tex_options().h);
        if len != (w * h * 4) as usize {
            set_err(format!("frame is {len} bytes, {w}x{h}x4 = {} expected", w * h * 4));
            return -1;
        }
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: s.input_texture(slot as usize),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            unsafe { std::slice::from_raw_parts(rgba, len) },
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        0
    })
}

/// The protection mask (`protection_mask` of the session options): width * height bytes, 0 interpolates, anything
/// else is taken from the source frames; applies from the next pair. 0 or -1.
/// # Safety
/// `s` is a live handle created on `e`; `mask` points to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_write_mask(s: *const Stream, e: *const Engine, mask: *const u8, len: usize) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        if mask.is_null() {
            set_err("null argument");
            return -1;
        }
        let bytes = unsafe { std::slice::from_raw_parts(mask, len) };
        match unsafe { &*s }.write_mask(unsafe { &*e }.gpu(), bytes) {
            Ok(()) => 0,
            Err(e) => {
                set_err(e);
                -1
            }
        }
    })
}

/// Classifies the pair (`previous`, `slot`) before pushing it, waiting for the GPU: 1 and `out` filled, 0 when the
/// stream has no classifier or every readback is in flight (take the pair as motion), -1 error.
/// # Safety
/// `s` is a live handle created on `e`; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_classify(
    s: *mut Stream,
    e: *const Engine,
    previous: u32,
    slot: u32,
    out: *mut FhClassification,
) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        if out.is_null() {
            set_err("null argument");
            return -1;
        }
        let (s, gpu) = (unsafe { &mut *s }, unsafe { &*e }.gpu());
        if previous as usize >= s.inputs() || slot as usize >= s.inputs() {
            set_err("bad input slot");
            return -1;
        }
        let Some(rb) = s.classify(gpu, previous as usize, slot as usize) else { return 0 };
        match rb.block(gpu) {
            Ok(c) => {
                unsafe {
                    *out = FhClassification {
                        kind: match c.kind {
                            IntervalKind::Motion => 0,
                            IntervalKind::Cut => 1,
                            IntervalKind::Hold => 2,
                        },
                        mean: c.mean,
                        max: c.max,
                        black: c.black as u32,
                        motion: c.motion,
                    };
                }
                1
            }
            Err(err) => {
                set_err(err);
                -1
            }
        }
    })
}

/// The newest source frame is in `slot` with timestamp `pts` (the host's units, usually media ms): runs its pair and
/// fills `out`; 0 or -1.
/// # Safety
/// `s` is a live handle created on `e`; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_push(
    s: *mut Stream,
    e: *const Engine,
    slot: u32,
    pts: f64,
    out: *mut FhPush,
) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        if out.is_null() {
            set_err("null argument");
            return -1;
        }
        match unsafe { &mut *s }.push(unsafe { &*e }.gpu(), slot as usize, pts) {
            Ok(p) => {
                unsafe {
                    *out = FhPush {
                        ready: p.interval.is_some() as u32,
                        base_slot: p.interval.map_or(-1, |i| i.base as i32),
                        stats_pending: p.stats_pending as u32,
                        start_pts: p.interval.map_or(0.0, |i| i.start_pts),
                        end_pts: p.interval.map_or(0.0, |i| i.end_pts),
                    };
                }
                0
            }
            Err(err) => {
                set_err(err);
                -1
            }
        }
    })
}

/// Video + cadence: waits for the pending pair statistics; 0 or -1.
/// # Safety
/// `s` is a live handle created on `e`.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_resolve_stats(s: *mut Stream, e: *const Engine) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        match unsafe { &mut *s }.resolve_stats(unsafe { &*e }.gpu()) {
            Ok(()) => 0,
            Err(err) => {
                set_err(err);
                -1
            }
        }
    })
}

/// Whether the current interval's real slot is re-timed (generate phase 0 instead of presenting the source).
/// # Safety
/// `s` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_real_replaced(s: *const Stream) -> i32 {
    guard(-1, || {
        live!(s);
        unsafe { &*s }.real_replaced() as i32
    })
}

/// Phase `t` of the current interval into output slot `out`: 1 written, 0 the source frame is enough (t = 0 without
/// re-timing), -1 error.
/// # Safety
/// `s` is a live handle created on `e`.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_generate(s: *mut Stream, e: *const Engine, t: f32, out: u32) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        match unsafe { &mut *s }.generate(unsafe { &*e }.gpu(), t, out as usize) {
            Ok(Generated::Written) => 1,
            Ok(Generated::SourceIsEnough) => 0,
            Err(err) => {
                set_err(err);
                -1
            }
        }
    })
}

/// An output slot as RGBA8 bytes into `buf` (cap >= width * height * 4), waiting for the GPU; bytes written or -1.
/// # Safety
/// `s` is a live handle created on `e`; `buf` points to `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_read_rgba(
    s: *const Stream,
    e: *const Engine,
    out: u32,
    buf: *mut u8,
    cap: usize,
) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        if buf.is_null() {
            set_err("null argument");
            return -1;
        }
        let (s, gpu) = (unsafe { &*s }, unsafe { &*e }.gpu());
        if out as usize >= s.outputs() {
            set_err(format!("bad output slot {out} (the stream has {})", s.outputs()));
            return -1;
        }
        let (w, h) = (s.tex_options().w, s.tex_options().h);
        let bpr = (w * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let stage = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fh_stream_read"),
            size: (bpr * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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
        gpu.queue.submit(Some(enc.finish()));
        let rb = flyhigh_engine::gpu::readback::Readback::new(
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
        let v = match rb.block(gpu) {
            Ok(v) => v,
            Err(err) => {
                set_err(err);
                return -1;
            }
        };
        if v.len() > cap {
            set_err(format!("output buffer too small ({} needed)", v.len()));
            return -1;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), buf, v.len());
        }
        v.len() as i32
    })
}

/// Compiles every kernel the stream runs, before the first real frame; 0 or -1.
/// # Safety
/// `s` is a live handle created on `e`.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_warm_up(s: *mut Stream, e: *const Engine) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        match unsafe { &mut *s }.warm_up(unsafe { &*e }.gpu()) {
            Ok(()) => 0,
            Err(err) => {
                set_err(err);
                -1
            }
        }
    })
}

/// A seek, a decoder gap, a new source: forget the frames, keep the plans.
/// # Safety
/// `s` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_reset(s: *mut Stream) -> i32 {
    guard(-1, || {
        live!(s);
        unsafe { &mut *s }.reset();
        0
    })
}

/// Mouse counts at time `t` for the camera warp (`Stream::mouse`): `t` on the clock of the `pts` given to
/// `fh_stream_push`, which must then be the frames' arrival times in host milliseconds. The first call starts the warp;
/// 0 or -1.
/// # Safety
/// `s` is a live handle created on `e`.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_mouse(s: *mut Stream, e: *const Engine, t: f64, dx: f64, dy: f64) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        unsafe { &mut *s }.mouse(unsafe { &*e }.gpu(), t, dx, dy);
        0
    })
}

/// Whether the host has the mouse captured (a game camera) or released (a cursor): released holds the camera warp
/// off at once; 0 or -1.
/// # Safety
/// `s` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_set_mouse_captured(s: *mut Stream, captured: u32) -> i32 {
    guard(-1, || {
        live!(s);
        unsafe { &mut *s }.set_mouse_captured(captured != 0);
        0
    })
}

/// The frame about to be presented at `now` turned by the mouse motion since the moment it shows, into output slot
/// `out` (`Stream::warp`): the frame is input slot `slot` when `output` is 0, output slot `slot` otherwise (`out`
/// itself allowed). 1 = `out` holds the turned frame, present it; 0 = the warp is off, present the frame as it is;
/// -1 = error.
/// # Safety
/// `s` is a live handle created on `e`.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_warp(
    s: *mut Stream,
    e: *const Engine,
    output: u32,
    slot: u32,
    out: u32,
    now: f64,
) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        let frame = if output != 0 { Shown::Output(slot as usize) } else { Shown::Input(slot as usize) };
        match unsafe { &mut *s }.warp(unsafe { &*e }.gpu(), frame, out as usize, now) {
            Ok(w) => w as i32,
            Err(err) => {
                set_err(err);
                -1
            }
        }
    })
}

/// The stream's report as JSON (`StreamReport`); bytes written or -1.
/// # Safety
/// `s` is a live handle; `buf` points to `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_report_json(s: *const Stream, buf: *mut c_char, cap: usize) -> i32 {
    guard(-1, || {
        live!(s);
        if buf.is_null() {
            set_err("null argument");
            return -1;
        }
        match serde_json::to_string(&unsafe { &*s }.report()) {
            Ok(j) => unsafe { write_json(&j, buf, cap) },
            Err(err) => {
                set_err(err);
                -1
            }
        }
    })
}

/// A D3D11 texture shared through an NT handle, imported into the engine's Vulkan device without a copy and added as
/// an input slot of the stream's ring (Windows). Returns the slot, or -1 (on other platforms always: not supported).
/// # Safety
/// `s` is a live handle created on `e`; `handle` is an NT shared handle of an rgba8 texture of the stream's frame size.
#[no_mangle]
pub unsafe extern "C" fn fh_stream_import_d3d11(
    s: *mut Stream,
    e: *const Engine,
    handle: isize,
    width: u32,
    height: u32,
) -> i32 {
    guard(-1, || {
        live!(s);
        live!(e);
        #[cfg(windows)]
        {
            use flyhigh_engine::interop::{d3d11_shared_texture, SharedTexture};
            let tex = d3d11_shared_texture(unsafe { &*e }, handle, &SharedTexture::rgba8(width, height));
            match tex.and_then(|t| unsafe { &mut *s }.add_input_texture(t)) {
                Ok(slot) => slot as i32,
                Err(err) => {
                    set_err(err);
                    -1
                }
            }
        }
        #[cfg(not(windows))]
        {
            let _ = (handle, width, height);
            set_err("D3D11 import is Windows only");
            -1
        }
    })
}
