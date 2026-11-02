//! FlyHigh Engine in the browser: the engine's layers as JavaScript classes over WebGPU through wgpu's web backend,
//! exported with wasm-bindgen. `FhEngine` (the device), `FhModel`, `FhSession`, `FhStream` and `FhPacer`, plus
//! `fh_pacer_replay` for recorded traces. One engine per page; keep calls sequential from JavaScript.
mod core;
mod pacer;

pub use core::*;
pub use pacer::*;
