//! `Engine`: the device the engine runs on, its compiled kernels and its autotune table (ARCHITECTURE.md). One per device;
//! models and sessions share it. Create one, or bring the device an application already has.
use crate::gpu::dxc::ShaderCompiler;
use crate::{EngineOptions, Gpu};
use anyhow::Result;
use serde::Serialize;
use std::rc::Rc;

/// Which kernel set the device runs: the plans, their performance and the golden fixtures differ per class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum KernelClass {
    /// f16 with cooperative-matrix (tensor core) GEMMs.
    TcF16,
    /// f16 portable kernels (every WebGPU device with shader-f16).
    PortableF16,
    /// f32 everywhere (devices without shader-f16): twice the memory traffic, the same pictures within rounding.
    PortableF32,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeviceInfo {
    /// Backend, adapter name and device type.
    pub name: String,
    pub class: KernelClass,
    pub tensor_cores: bool,
    /// Timestamp queries: GPU time per stage can be measured.
    pub timestamps: bool,
    pub bgra_storage: bool,
    /// DX12 only: the HLSL compiler wgpu loaded. `Fxc` means `dxcompiler.dll` was not found (`EngineOptions::dxc_path`):
    /// slow kernel compiles and the f32 kernels.
    pub shader_compiler: Option<ShaderCompiler>,
}

#[derive(Clone, Debug, Serialize)]
pub struct EngineReport {
    pub device: DeviceInfo,
    /// Kernels compiled so far (pipelines compile on first use).
    pub kernels_compiled: usize,
    /// Conv kernel choices known for this device.
    pub tune_entries: usize,
    /// Bytes of tensor buffers created so far.
    pub tensor_bytes: u64,
}

pub struct Engine {
    gpu: Rc<Gpu>,
}

impl Engine {
    /// A device the engine owns: adapter selection, features and limits per `opts`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(opts: &EngineOptions) -> Result<Engine> {
        Ok(Engine::from_gpu(Gpu::new_with(opts)?))
    }

    /// The browser (and any async host): the same, awaited.
    pub async fn new_async(opts: &EngineOptions) -> Result<Engine> {
        Ok(Engine::from_gpu(Gpu::new_async(opts, false).await?))
    }

    /// The engine on a device the host already has. The kernels use what the device was created with (shader-f16, else
    /// the f32 path; cooperative matrices and timestamp queries when enabled) and never reconfigure it.
    pub fn with_device(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        opts: &EngineOptions,
    ) -> Result<Engine> {
        Ok(Engine::from_gpu(Gpu::from_device(instance, adapter, device, queue, opts)?))
    }

    /// Wraps a GPU context built elsewhere (the CLI tools, the browser shell).
    pub fn from_gpu(gpu: Gpu) -> Engine {
        Engine { gpu: Rc::new(gpu) }
    }

    pub fn from_shared(gpu: Rc<Gpu>) -> Engine {
        Engine { gpu }
    }

    /// The GPU context: device, queue, instance and adapter for the host's own surfaces and copies.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The same, shared (sessions hold one).
    pub fn shared(&self) -> Rc<Gpu> {
        self.gpu.clone()
    }

    pub fn class(&self) -> KernelClass {
        if self.gpu.coop {
            KernelClass::TcF16
        } else if self.gpu.f32_mode {
            KernelClass::PortableF32
        } else {
            KernelClass::PortableF16
        }
    }

    pub fn info(&self) -> DeviceInfo {
        DeviceInfo {
            name: self.gpu.name.clone(),
            class: self.class(),
            tensor_cores: self.gpu.coop,
            timestamps: self.gpu.timestamps,
            bgra_storage: self.gpu.bgra_storage,
            shader_compiler: self.gpu.shader_compiler,
        }
    }

    /// The kernels compiled so far (`Gpu::compiled_kernels`): what a host records to `precompile` next time.
    pub fn compiled_kernels(&self) -> Vec<&'static str> {
        self.gpu.compiled_kernels()
    }

    /// Compiles the named kernels not compiled yet, unknown names skipped (`Gpu::precompile`); returns how many.
    pub fn precompile(&self, names: &[&str]) -> usize {
        self.gpu.precompile(names)
    }

    /// The WGSL and entry point of a kernel as this device compiles it (`Gpu::kernel_wgsl`).
    pub fn kernel_wgsl(&self, name: &str) -> Option<(String, &'static str)> {
        self.gpu.kernel_wgsl(name)
    }

    pub fn report(&self) -> EngineReport {
        EngineReport {
            device: self.info(),
            kernels_compiled: self.gpu.p.borrow().len(),
            tune_entries: self.gpu.tune.borrow().len(),
            tensor_bytes: self.gpu.alloc_bytes.get(),
        }
    }
}
