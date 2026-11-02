//! Adapter selection and device creation: features, limits, the f16 / f32 decision; or a device the host already has.

use crate::{parse_tune, EngineOptions, Gpu, Precision};
use anyhow::Result;
use std::collections::HashMap;
use std::rc::Rc;
use wgpu::util::DeviceExt;

/// The adapters the engine can run on, in the order `EngineOptions::adapter = "#N"` indexes them (native).
#[cfg(not(target_arch = "wasm32"))]
pub fn list_adapters(backends: Option<wgpu::Backends>) -> Vec<wgpu::AdapterInfo> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    if let Some(b) = backends {
        desc.backends = b;
    }
    let b = desc.backends;
    pollster::block_on(wgpu::Instance::new(desc).enumerate_adapters(b)).iter().map(|a| a.get_info()).collect()
}
pub(crate) fn pick_adapter(list: Vec<wgpu::Adapter>, sel: &str) -> Result<wgpu::Adapter> {
    let names: Vec<String> = list
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let n = a.get_info();
            format!("#{i} {} ({:?}, {:04x}:{:04x})", n.name, n.backend, n.vendor, n.device)
        })
        .collect();
    let ids = sel
        .split_once(':')
        .and_then(|(v, d)| Some((u32::from_str_radix(v, 16).ok()?, u32::from_str_radix(d, 16).ok()?)));
    let lower = sel.to_lowercase();
    let i = if let Some(n) = sel.strip_prefix('#').and_then(|n| n.parse::<usize>().ok()) {
        (n < list.len()).then_some(n)
    } else if let Some((v, d)) = ids {
        list.iter().position(|a| {
            let n = a.get_info();
            n.vendor == v && n.device == d
        })
    } else {
        list.iter().position(|a| a.get_info().name.to_lowercase().contains(&lower))
    };
    let i = i.ok_or_else(|| anyhow::anyhow!("no adapter matches '{sel}'; available: {}", names.join(", ")))?;
    Ok(list.into_iter().nth(i).unwrap())
}

/// What the kernels may use on a device.
struct Caps {
    f32_mode: bool,
    coop: bool,
    timestamps: bool,
    ts_inside: bool,
    bgra_storage: bool,
}

// The coop kernels are written for 16x16x16 f16 x f16 -> f32 tiles; Metal exposes the feature with 8x8 tiles only and
// its shader compiler rejects ours, so the feature alone is not enough.
fn tile16(adapter: &wgpu::Adapter) -> bool {
    adapter.cooperative_matrix_properties().iter().any(|p| {
        p.m_size == 16
            && p.n_size == 16
            && p.k_size == 16
            && p.ab_type == wgpu::CooperativeScalarType::F16
            && p.cr_type == wgpu::CooperativeScalarType::F32
    })
}

fn caps_of(adapter: &wgpu::Adapter, features: wgpu::Features, opts: &EngineOptions) -> Result<Caps> {
    let has_f16 = features.contains(wgpu::Features::SHADER_F16);
    anyhow::ensure!(
        has_f16 || opts.precision != Precision::F16,
        "precision F16 requested but the device lacks shader-f16"
    );
    let f32_mode = opts.precision == Precision::F32 || !has_f16;
    // the cooperative-matrix kernels are f16 throughout
    let coop = features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        && tile16(adapter)
        && opts.tensor_cores
        && !f32_mode;
    let timestamps = features.contains(wgpu::Features::TIMESTAMP_QUERY);
    let ts_inside = timestamps && features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
    let bgra_storage = features.contains(wgpu::Features::BGRA8UNORM_STORAGE);
    Ok(Caps { f32_mode, coop, timestamps, ts_inside, bgra_storage })
}

impl Gpu {
    /// CLI / developer entry: `EngineOptions::from_env()` and the capability report on stdout
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Gpu> {
        pollster::block_on(Gpu::new_async(&EngineOptions::from_env(), true))
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_with(opts: &EngineOptions) -> Result<Gpu> {
        pollster::block_on(Gpu::new_async(opts, false))
    }
    // `verbose` prints the capability report (CLI); library hosts and the browser shell pass false
    pub async fn new_async(opts: &EngineOptions, verbose: bool) -> Result<Gpu> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        if let Some(b) = opts.backend {
            desc.backends = b;
        } // e.g. DX12 for a DirectComposition overlay
          // WGPU_DX12_COMPILER in the environment is a developer override and wins over the option
        #[cfg(windows)]
        if matches!(desc.backend_options.dx12.shader_compiler, wgpu::Dx12Compiler::Auto) {
            desc.backend_options.dx12.shader_compiler = super::dxc::choose(opts.dxc_path.as_deref());
        }
        let backends = desc.backends;
        let instance = wgpu::Instance::new(desc);
        let adapter = match opts.adapter.as_deref() {
            None | Some("") | Some("high-performance") | Some("low-power") => {
                let power_preference = if opts.adapter.as_deref() == Some("low-power") {
                    wgpu::PowerPreference::LowPower
                } else {
                    wgpu::PowerPreference::HighPerformance
                };
                instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference,
                        force_fallback_adapter: false,
                        compatible_surface: None,
                        apply_limit_buckets: false,
                    })
                    .await
                    .map_err(|e| anyhow::anyhow!("no wgpu adapter: {e:?}"))?
            }
            Some(sel) => pick_adapter(instance.enumerate_adapters(backends).await, sel)?,
        };
        let info = adapter.get_info();
        let f = adapter.features();
        let alim = adapter.limits();
        // tensor_cores = false (CLI: FH_NO_COOP=1) simulates a WebGPU-class device: the browser leg's kernels
        let caps = caps_of(&adapter, f, opts)?;
        let mut feats = if caps.f32_mode { wgpu::Features::empty() } else { wgpu::Features::SHADER_F16 };
        if caps.timestamps {
            feats |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if caps.ts_inside {
            feats |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        }
        if caps.bgra_storage {
            feats |= wgpu::Features::BGRA8UNORM_STORAGE;
        }
        if caps.coop {
            feats |= wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
        }
        // hosts can then import textures from other APIs without a copy (e.g. a D3D11 screen capture into Vulkan)
        if f.contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32) {
            feats |= wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32;
        }
        if verbose {
            println!("  caps: SUBGROUP={} COOP_MATRIX(tensor cores)={} SHADER_F16={} TIMESTAMPS={} (inside encoders {}) wg_storage={}B",
                     f.contains(wgpu::Features::SUBGROUP), f.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX), f.contains(wgpu::Features::SHADER_F16),
                     caps.timestamps, caps.ts_inside, alim.max_compute_workgroup_storage_size);
            println!("  backend={:?} driver='{}' driver_info='{}'", info.backend, info.driver, info.driver_info);
            let cmp = adapter.cooperative_matrix_properties();
            println!("  cooperative_matrix configs exposed: {}", cmp.len());
            for p in cmp.iter().take(6) {
                println!("    {:?}", p);
            }
        }
        // fused4 (coop-only microbench kernel) stages ~23 KB in shared; the WebGPU default is 16 KB and the portable
        // kernels fit in it, so ask for 32 KB only where the adapter has it
        let wg_storage = alim.max_compute_workgroup_storage_size.clamp(16384, 32768);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("flyhigh"),
                required_features: feats,
                required_limits: wgpu::Limits {
                    max_compute_workgroup_storage_size: wg_storage,
                    ..wgpu::Limits::default()
                },
                memory_hints: if opts.compact_memory {
                    wgpu::MemoryHints::MemoryUsage
                } else {
                    wgpu::MemoryHints::Performance
                },
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| anyhow::anyhow!("no wgpu device: {e:?}"))?;
        let gpu = Gpu::assemble(instance, adapter, device, queue, opts, caps);
        if verbose {
            if let Some(c) = gpu.shader_compiler {
                println!("  dx12 shader compiler: {c:?}");
            }
        }
        Ok(gpu)
    }

    /// The engine on a device the host already has (a game's or a player's wgpu device): the kernels use what the
    /// device was created with (shader-f16, else the f32 path; cooperative matrices and timestamp queries when
    /// enabled) and never reconfigure it. `instance` and `adapter` are the ones the device came from.
    pub fn from_device(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        opts: &EngineOptions,
    ) -> Result<Gpu> {
        let caps = caps_of(&adapter, device.features(), opts)?;
        Ok(Gpu::assemble(instance, adapter, device, queue, opts, caps))
    }

    fn assemble(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        opts: &EngineOptions,
        caps: Caps,
    ) -> Gpu {
        let info = adapter.get_info();
        let name = format!("{:?} {} ({:?})", info.backend, info.name, info.device_type);
        #[cfg(windows)]
        let shader_compiler = (info.backend == wgpu::Backend::Dx12).then(super::dxc::loaded);
        #[cfg(not(windows))]
        let shader_compiler = None;
        let zero = Rc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fh_zero"),
            contents: bytemuck::cast_slice(&[0f32; 256]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        }));
        // f16 ones: PReLU slope of 1 turns the fused epilogue into identity, so no-activation convs
        // (conv2d heads, up.w) run through the same Tensor-Core kernel instead of the naive conv.
        let ones_bytes: Vec<u8> = if caps.f32_mode {
            bytemuck::cast_slice(&vec![1f32; 1024]).to_vec()
        } else {
            bytemuck::cast_slice(&vec![half::f16::from_f32(1.0).to_bits(); 1024]).to_vec()
        };
        let ones = Rc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fh_ones"),
            contents: &ones_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        }));
        // the choices depend on the GPU, its driver, the kernel set (precision, tensor cores) and the engine's kernels
        let tune_file = opts.tune_cache.as_ref().map(|d| {
            let drv: u32 = format!("{} {}", info.driver, info.driver_info)
                .bytes()
                .fold(2166136261u32, |h, b| (h ^ b as u32).wrapping_mul(16777619));
            d.join(format!(
                "tune-{:04x}-{:04x}-{:08x}-{}-{}-{}.txt",
                info.vendor,
                info.device,
                drv,
                if caps.f32_mode { "f32" } else { "f16" },
                if caps.coop { "tc" } else { "portable" },
                env!("CARGO_PKG_VERSION")
            ))
        });
        let tune0 = tune_file
            .as_ref()
            .and_then(|f| std::fs::read_to_string(f).ok())
            .map(|t| parse_tune(&t))
            .unwrap_or_default();
        Gpu {
            f32_mode: caps.f32_mode,
            instance,
            adapter,
            device,
            queue,
            p: std::cell::RefCell::new(HashMap::new()),
            zero,
            ones,
            name,
            coop: caps.coop,
            autotune: opts.autotune,
            tune: std::cell::RefCell::new(tune0),
            tune_file,
            live: Default::default(),
            bgra_storage: caps.bgra_storage,
            timestamps: caps.timestamps,
            ts_inside: caps.ts_inside,
            shader_compiler,
            arena: std::cell::RefCell::new(None),
            arena_ids: std::cell::RefCell::new(HashMap::new()),
            scratch: std::cell::RefCell::new(Vec::new()),
            alloc_bytes: std::cell::Cell::new(0),
            last_pb: std::cell::RefCell::new(None),
            pass_split: std::cell::Cell::new(0),
            submit_split: std::cell::Cell::new(0),
        }
    }
}
