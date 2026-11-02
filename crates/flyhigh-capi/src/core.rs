//! The Core over C: `fh_engine_*` (the device), `fh_model_*`, `fh_session_*` (two frames in, one frame out, CPU RGBA8
//! frames; GPU textures through the import calls). Single-threaded use per handle; a failed call returns -1 or null and
//! `fh_last_error` has the message.
use crate::{set_err, write_json};
use flyhigh_engine::{
    Engine, EngineOptions, Internal, Model, RenderSampling, Session, SessionOptions, StaticProtection,
};
use std::ffi::{c_char, CStr};

/// Options for `fh_engine_create`; start from `fh_engine_options_default()`. `backend` 0 = automatic, 1 Vulkan,
/// 2 DX12, 3 Metal, 4 OpenGL; `tensor_cores` 1 = cooperative-matrix kernels where available; `autotune` 1 = time
/// the conv kernels per shape when a session is built; `adapter` NULL = the high-performance GPU, else a selector as
/// `EngineOptions::adapter` ("#1", "10de:2803", "4060", "low-power"); `precision` 0 automatic, 1 f16, 2 f32;
/// `tune_cache` NULL or a directory where the autotune results persist; `compact_memory` 1 = small allocator blocks;
/// `dxc_path` NULL or where `dxcompiler.dll` lives (Windows DX12; NULL = beside the library, then the DLL search path).
#[repr(C)]
pub struct FhEngineOptions {
    pub backend: u32,
    pub tensor_cores: u32,
    pub autotune: u32,
    pub adapter: *const c_char,
    pub precision: u32,
    pub tune_cache: *const c_char,
    pub compact_memory: u32,
    pub dxc_path: *const c_char,
}

impl FhEngineOptions {
    /// # Safety
    /// `adapter`, `tune_cache` and `dxc_path` are null or NUL-terminated strings.
    pub unsafe fn to_rust(&self) -> Result<EngineOptions, String> {
        let backend = match self.backend {
            0 => None,
            1 => Some(wgpu::Backends::VULKAN),
            2 => Some(wgpu::Backends::DX12),
            3 => Some(wgpu::Backends::METAL),
            4 => Some(wgpu::Backends::GL),
            b => return Err(format!("unknown backend {b} (0 auto, 1 vulkan, 2 dx12, 3 metal, 4 gl)")),
        };
        let text =
            |p: *const c_char| (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().to_string());
        Ok(EngineOptions {
            backend,
            tensor_cores: self.tensor_cores != 0,
            autotune: self.autotune != 0,
            adapter: text(self.adapter),
            precision: match self.precision {
                1 => flyhigh_engine::Precision::F16,
                2 => flyhigh_engine::Precision::F32,
                _ => flyhigh_engine::Precision::Auto,
            },
            tune_cache: text(self.tune_cache).map(Into::into),
            compact_memory: self.compact_memory != 0,
            dxc_path: text(self.dxc_path).map(Into::into),
        })
    }
}

/// Options for `fh_session_create`; start from `fh_session_options_default(w, h)`. `internal_w/h` 0, 0 = automatic
/// (the 480-line rung above 960 px, else the frame), the frame size = the frame; `k` 1..3; `repair` 0..2; `render`
/// 0 = the model's sampler, 1 bilinear, 2 bicubic; `keep_static` 1 = static protection with `static_frames`
/// (default 8) and `static_tolerance` (default 4); `fast_render` 1 = render from the source textures; `linear_t` 1 =
/// with fast_render, the t-dependent network once per pair; `keep_maps` 1 = keep the repair maps (the Stream's
/// cadence); `tune` NULL or the autotune table text.
#[repr(C)]
pub struct FhSessionOptions {
    pub width: u32,
    pub height: u32,
    pub internal_w: u32,
    pub internal_h: u32,
    pub k: u32,
    pub repair: u32,
    pub render: u32,
    pub keep_static: u32,
    pub static_frames: u32,
    pub static_tolerance: u32,
    pub fast_render: u32,
    pub linear_t: u32,
    pub keep_maps: u32,
    pub protection_mask: u32,
    pub tune: *const c_char,
}

impl FhSessionOptions {
    pub fn from_rust(o: &SessionOptions) -> FhSessionOptions {
        let (iw, ih) = match o.internal {
            Internal::Auto => (0, 0),
            Internal::Frame => (o.width, o.height),
            Internal::Size(w, h) => (w, h),
        };
        let sp = o.static_protection.unwrap_or_default();
        FhSessionOptions {
            width: o.width,
            height: o.height,
            internal_w: iw,
            internal_h: ih,
            k: o.k,
            repair: o.repair,
            render: match o.render {
                RenderSampling::Model => 0,
                RenderSampling::Bilinear => 1,
                RenderSampling::Bicubic => 2,
            },
            keep_static: o.static_protection.is_some() as u32,
            static_frames: sp.frames,
            static_tolerance: sp.tolerance,
            fast_render: o.fast_render as u32,
            linear_t: o.linear_t as u32,
            keep_maps: o.keep_maps as u32,
            protection_mask: o.protection_mask as u32,
            tune: std::ptr::null(),
        }
    }

    /// # Safety
    /// `tune` is null or a NUL-terminated string.
    pub unsafe fn to_rust(&self) -> Result<SessionOptions, String> {
        let mut o = SessionOptions::new(self.width, self.height);
        o.internal = match (self.internal_w, self.internal_h) {
            (0, 0) => Internal::Auto,
            (w, h) if w == self.width && h == self.height => Internal::Frame,
            (w, h) => Internal::Size(w, h),
        };
        o.k = self.k;
        o.repair = self.repair;
        o.render = match self.render {
            0 => RenderSampling::Model,
            1 => RenderSampling::Bilinear,
            2 => RenderSampling::Bicubic,
            r => return Err(format!("unknown render {r} (0 model, 1 bilinear, 2 bicubic)")),
        };
        o.static_protection = (self.keep_static != 0).then(|| {
            let d = StaticProtection::default();
            StaticProtection {
                frames: if self.static_frames > 0 { self.static_frames } else { d.frames },
                tolerance: self.static_tolerance,
            }
        });
        o.fast_render = self.fast_render != 0;
        o.linear_t = self.linear_t != 0;
        o.keep_maps = self.keep_maps != 0;
        o.protection_mask = self.protection_mask != 0;
        if !self.tune.is_null() {
            let text = unsafe { CStr::from_ptr(self.tune) }.to_string_lossy();
            o.tune = flyhigh_engine::tune_table_from(&text);
        }
        o.validate().map_err(|e| e.to_string())?;
        Ok(o)
    }
}

#[no_mangle]
pub extern "C" fn fh_engine_options_default() -> FhEngineOptions {
    FhEngineOptions {
        backend: 0,
        tensor_cores: 1,
        autotune: 1,
        adapter: std::ptr::null(),
        precision: 0,
        tune_cache: std::ptr::null(),
        compact_memory: 1,
        dxc_path: std::ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn fh_session_options_default(width: u32, height: u32) -> FhSessionOptions {
    FhSessionOptions::from_rust(&SessionOptions::new(width, height))
}

/// The device. NULL options = the defaults.
/// # Safety
/// `o` is null or points to a valid `FhEngineOptions` whose strings are null or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn fh_engine_create(o: *const FhEngineOptions) -> *mut Engine {
    let opts = if o.is_null() { Ok(EngineOptions::default()) } else { unsafe { (*o).to_rust() } };
    let opts = match opts {
        Ok(o) => o,
        Err(e) => {
            set_err(e);
            return std::ptr::null_mut();
        }
    };
    match Engine::new(&opts) {
        Ok(e) => Box::into_raw(Box::new(e)),
        Err(e) => {
            set_err(e);
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `e` is null or a handle from `fh_engine_create` that is not used afterwards (sessions on it must be gone).
#[no_mangle]
pub unsafe extern "C" fn fh_engine_destroy(e: *mut Engine) {
    if !e.is_null() {
        unsafe {
            drop(Box::from_raw(e));
        }
    }
}

/// The device as JSON (`DeviceInfo`: name, class, tensor_cores, timestamps, bgra_storage, shader_compiler); bytes
/// written or -1.
/// # Safety
/// `e` is a live handle; `buf` points to `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn fh_engine_info(e: *const Engine, buf: *mut c_char, cap: usize) -> i32 {
    if e.is_null() || buf.is_null() {
        set_err("null argument");
        return -1;
    }
    match serde_json::to_string(&unsafe { &*e }.info()) {
        Ok(s) => unsafe { write_json(&s, buf, cap) },
        Err(err) => {
            set_err(err);
            -1
        }
    }
}

/// The engine's report as JSON (`EngineReport`); bytes written or -1.
/// # Safety
/// `e` is a live handle; `buf` points to `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn fh_engine_report_json(e: *const Engine, buf: *mut c_char, cap: usize) -> i32 {
    if e.is_null() || buf.is_null() {
        set_err("null argument");
        return -1;
    }
    match serde_json::to_string(&unsafe { &*e }.report()) {
        Ok(s) => unsafe { write_json(&s, buf, cap) },
        Err(err) => {
            set_err(err);
            -1
        }
    }
}

/// A model from a directory with `graph.json`, `flyhighfg.json`, `flyhighfg.bin`. NULL on error.
/// # Safety
/// `dir` is a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn fh_model_load(dir: *const c_char) -> *mut Model {
    if dir.is_null() {
        set_err("dir is null");
        return std::ptr::null_mut();
    }
    let dir = unsafe { CStr::from_ptr(dir) }.to_string_lossy().to_string();
    match Model::from_dir(std::path::Path::new(&dir)) {
        Ok(m) => Box::into_raw(Box::new(m)),
        Err(e) => {
            set_err(e);
            std::ptr::null_mut()
        }
    }
}

/// A model from the three files in memory. NULL on error.
/// # Safety
/// Each pointer points to that many readable bytes.
#[no_mangle]
pub unsafe extern "C" fn fh_model_from_bytes(
    graph: *const u8,
    graph_len: usize,
    manifest: *const u8,
    manifest_len: usize,
    weights: *const u8,
    weights_len: usize,
) -> *mut Model {
    if graph.is_null() || manifest.is_null() || weights.is_null() {
        set_err("null argument");
        return std::ptr::null_mut();
    }
    let (g, m, w) = unsafe {
        (
            std::slice::from_raw_parts(graph, graph_len),
            std::slice::from_raw_parts(manifest, manifest_len),
            std::slice::from_raw_parts(weights, weights_len),
        )
    };
    match Model::from_bytes(g, m, w.to_vec()) {
        Ok(model) => Box::into_raw(Box::new(model)),
        Err(e) => {
            set_err(e);
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `m` is null or a handle from `fh_model_*` that is not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn fh_model_destroy(m: *mut Model) {
    if !m.is_null() {
        unsafe {
            drop(Box::from_raw(m));
        }
    }
}

/// The model as JSON (`ModelInfo`); bytes written or -1.
/// # Safety
/// `m` is a live handle; `buf` points to `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn fh_model_info(m: *const Model, buf: *mut c_char, cap: usize) -> i32 {
    if m.is_null() || buf.is_null() {
        set_err("null argument");
        return -1;
    }
    match serde_json::to_string(unsafe { &*m }.info()) {
        Ok(s) => unsafe { write_json(&s, buf, cap) },
        Err(err) => {
            set_err(err);
            -1
        }
    }
}

/// The plans of `m` on `e` for one frame size. NULL on error (an option out of range names the field).
/// # Safety
/// `e` and `m` are live handles; `o` points to a valid `FhSessionOptions` whose `tune` is null or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn fh_session_create(
    e: *const Engine,
    m: *const Model,
    o: *const FhSessionOptions,
) -> *mut Session {
    if e.is_null() || m.is_null() || o.is_null() {
        set_err("null argument");
        return std::ptr::null_mut();
    }
    let opts = match unsafe { (*o).to_rust() } {
        Ok(o) => o,
        Err(err) => {
            set_err(err);
            return std::ptr::null_mut();
        }
    };
    match Session::new(unsafe { &*e }, unsafe { &*m }, &opts) {
        Ok(s) => Box::into_raw(Box::new(s)),
        Err(err) => {
            set_err(err);
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `s` is null or a handle from `fh_session_create` that is not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn fh_session_destroy(s: *mut Session) {
    if !s.is_null() {
        unsafe {
            drop(Box::from_raw(s));
        }
    }
}

macro_rules! live {
    ($p:expr) => {
        if $p.is_null() {
            set_err("null argument");
            return -1;
        }
    };
}

/// Adds an input slot; returns its index or -1.
/// # Safety
/// `s` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_session_add_input(s: *mut Session) -> i32 {
    live!(s);
    unsafe { &mut *s }.add_input() as i32
}

/// Adds an output slot; returns its index or -1.
/// # Safety
/// `s` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_session_add_output(s: *mut Session) -> i32 {
    live!(s);
    unsafe { &mut *s }.add_output() as i32
}

/// A frame (RGBA8, width * height * 4 bytes) into an input slot; 0 or -1.
/// # Safety
/// `s` is a live handle; `rgba` points to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn fh_session_write_rgba(s: *const Session, slot: u32, rgba: *const u8, len: usize) -> i32 {
    live!(s);
    if rgba.is_null() {
        set_err("null argument");
        return -1;
    }
    let frame = unsafe { std::slice::from_raw_parts(rgba, len) };
    match unsafe { &*s }.write_rgba(slot as usize, frame) {
        Ok(()) => 0,
        Err(e) => {
            set_err(e);
            -1
        }
    }
}

/// The protection mask (`protection_mask`): width * height bytes, 0 interpolates, anything else is taken from the
/// source frames; applies from the next pair. 0 or -1.
/// # Safety
/// `s` is a live handle; `mask` points to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn fh_session_write_mask(s: *const Session, mask: *const u8, len: usize) -> i32 {
    live!(s);
    if mask.is_null() {
        set_err("null argument");
        return -1;
    }
    match unsafe { &*s }.write_mask(unsafe { std::slice::from_raw_parts(mask, len) }) {
        Ok(()) => 0,
        Err(e) => {
            set_err(e);
            -1
        }
    }
}

/// An output slot as RGBA8 bytes into `buf` (cap >= width * height * 4), waiting for the GPU; bytes written or -1.
/// # Safety
/// `s` is a live handle; `buf` points to `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn fh_session_read_rgba(s: *const Session, slot: u32, buf: *mut u8, cap: usize) -> i32 {
    live!(s);
    if buf.is_null() {
        set_err("null argument");
        return -1;
    }
    let s = unsafe { &*s };
    let v = match s.read_rgba(slot as usize).and_then(|rb| rb.block(s.gpu())) {
        Ok(v) => v,
        Err(e) => {
            set_err(e);
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
}

/// The motion of the interval (`a`, `b`); 0 or -1.
/// # Safety
/// `s` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_session_pair(s: *const Session, a: u32, b: u32) -> i32 {
    live!(s);
    match unsafe { &*s }.pair(a as usize, b as usize) {
        Ok(()) => 0,
        Err(e) => {
            set_err(e);
            -1
        }
    }
}

/// Phase `t` of the last pair into output slot `out`; 0 or -1.
/// # Safety
/// `s` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_session_mid(s: *const Session, t: f32, out: u32) -> i32 {
    live!(s);
    match unsafe { &*s }.mid(t, out as usize) {
        Ok(()) => 0,
        Err(e) => {
            set_err(e);
            -1
        }
    }
}

/// `pair` and `mid` in one call; 0 or -1.
/// # Safety
/// `s` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_session_generate(s: *const Session, a: u32, b: u32, t: f32, out: u32) -> i32 {
    live!(s);
    match unsafe { &*s }.generate(a as usize, b as usize, t, out as usize) {
        Ok(()) => 0,
        Err(e) => {
            set_err(e);
            -1
        }
    }
}

/// GPU time of the pair stage in ms into `ms`: 1 measured, 0 no timestamp queries on this device, -1 error.
/// # Safety
/// `s` is a live handle; `ms` is writable.
#[no_mangle]
pub unsafe extern "C" fn fh_session_pair_timed(s: *const Session, a: u32, b: u32, ms: *mut f64) -> i32 {
    live!(s);
    if ms.is_null() {
        set_err("null argument");
        return -1;
    }
    match unsafe { &*s }.pair_timed(a as usize, b as usize) {
        Ok(Some(v)) => {
            unsafe { *ms = v };
            1
        }
        Ok(None) => 0,
        Err(e) => {
            set_err(e);
            -1
        }
    }
}

/// GPU time of one generated frame of the last pair in ms into `ms`: 1 measured, 0 no timestamp queries, -1 error.
/// # Safety
/// `s` is a live handle; `ms` is writable.
#[no_mangle]
pub unsafe extern "C" fn fh_session_mid_timed(s: *const Session, t: f32, out: u32, ms: *mut f64) -> i32 {
    live!(s);
    if ms.is_null() {
        set_err("null argument");
        return -1;
    }
    match unsafe { &*s }.mid_timed(t, out as usize) {
        Ok(Some(v)) => {
            unsafe { *ms = v };
            1
        }
        Ok(None) => 0,
        Err(e) => {
            set_err(e);
            -1
        }
    }
}

/// Compiles every kernel the session runs (one pair and one mid on slots added for the purpose); 0 or -1.
/// # Safety
/// `s` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn fh_session_warm_up(s: *mut Session) -> i32 {
    live!(s);
    match unsafe { &mut *s }.warm_up() {
        Ok(()) => 0,
        Err(e) => {
            set_err(e);
            -1
        }
    }
}

/// The session's report as JSON (`SessionReport`); bytes written or -1.
/// # Safety
/// `s` is a live handle; `buf` points to `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn fh_session_report_json(s: *const Session, buf: *mut c_char, cap: usize) -> i32 {
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
}

/// A D3D11 texture shared through an NT handle (`D3D11_RESOURCE_MISC_SHARED_NTHANDLE`), imported into the engine's
/// Vulkan device without a copy and added as an input slot of the session (Windows; `interop` in the Rust API). Returns
/// the slot, or -1 (on other platforms always: not supported).
/// # Safety
/// `s` is a live handle created on `e`; `handle` is an NT shared handle of an rgba8 texture of the session's size.
#[no_mangle]
pub unsafe extern "C" fn fh_session_import_d3d11(
    s: *mut Session,
    e: *const Engine,
    handle: isize,
    width: u32,
    height: u32,
) -> i32 {
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
}
