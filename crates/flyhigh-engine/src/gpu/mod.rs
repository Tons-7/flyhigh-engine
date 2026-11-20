//! The GPU context: device, queue, compiled pipelines, the buffer arena and the autotune table.

pub mod alloc;
pub mod device;
pub mod dxc;
pub mod exec;
pub mod memory;
pub mod pipeline;
pub mod readback;
pub mod timing;

use crate::TuneTable;
use memory::{Buffer, Charge, Memory};
use std::collections::HashMap;
use std::rc::Rc;

pub struct Gpu {
    pub f32_mode: bool,
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub p: std::cell::RefCell<HashMap<&'static str, Rc<wgpu::ComputePipeline>>>, // compiled kernels, see `pipeline`
    pub zero: Rc<Buffer>,
    pub ones: Rc<Buffer>,
    pub name: String,
    pub coop: bool,
    pub autotune: bool, // EngineOptions::autotune (native build-time conv autotune)
    pub tune: std::cell::RefCell<TuneTable>, // conv choices of every build so far
    pub tune_file: Option<std::path::PathBuf>, // EngineOptions::tune_cache: where `tune` persists
    pub(crate) memory: Memory,
    pub(crate) live: std::cell::RefCell<Vec<std::rc::Weak<Charge>>>, // tensor buffers, for live_sizes
    pub bgra_storage: bool, // BGRA8UNORM_STORAGE: canvases that prefer bgra8unorm can be written directly
    pub timestamps: bool,   // TIMESTAMP_QUERY: gpu_time_ms works (pass-level timestamps, WebGPU-standard)
    pub ts_inside: bool,    // + TIMESTAMP_QUERY_INSIDE_ENCODERS (native): per-step profile_plan
    pub(crate) timing: std::cell::RefCell<timing::TimingState>,
    pub shader_compiler: Option<dxc::ShaderCompiler>, // DX12: the HLSL compiler wgpu loaded (None on other backends)
    // buffer arena (Some while a plan is being built): tensors whose last reader has been built release their
    // buffer here and later tensors of the same plan reuse it — a static in-order plan makes this safe
    pub arena: std::cell::RefCell<Option<Vec<Rc<Buffer>>>>,
    pub(crate) workspace: std::cell::RefCell<Option<alloc::Workspace>>,
    // buffers that came from `alloc` (the only ones `release` takes), by address; the Weak keeps the Rc allocation
    // reserved, so a dropped buffer's address cannot be reused by a buffer that never came from the arena
    pub arena_ids: std::cell::RefCell<HashMap<usize, std::rc::Weak<Buffer>>>,
    pub scratch: std::cell::RefCell<Vec<Rc<Buffer>>>, // per-op kernel scratch awaiting `release_scratch`
    pub alloc_bytes: std::cell::Cell<u64>,            // bytes of tensor buffers actually created
    // Frame encoding knobs (experiments on a shared GPU): 0 = one compute pass / one submit per plan (the default);
    // n = a new compute pass every n dispatches / a new command buffer every n dispatches
    pub pass_split: std::cell::Cell<usize>,
    pub submit_split: std::cell::Cell<usize>,
}
