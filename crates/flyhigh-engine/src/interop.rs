//! External texture interop (docs/design/04-host-adapters.md): textures another API produced, imported into the engine's device without a
//! copy, so a capture or a game hands frames over on the GPU. Windows: a D3D11 texture shared through an NT handle
//! (`D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE`, `IDXGIResource1::CreateSharedHandle`) into
//! the Vulkan device through `VK_KHR_external_memory_win32`. The engine requests that feature whenever the adapter
//! has it; `d3d11_import_supported` says whether this device can.
//!
//! Synchronisation is the host's: D3D11 writes must be complete before the engine samples the texture (flush and wait
//! on the D3D11 side, or fence between the queues), and the engine only reads imported textures, so the image layout
//! Vulkan settles on is kept across later D3D11 writes.
use crate::Engine;
use anyhow::{Context, Result};

/// What the shared texture is: its size and one of `Rgba8Unorm` / `Bgra8Unorm`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedTexture {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
}

impl SharedTexture {
    pub fn rgba8(width: u32, height: u32) -> SharedTexture {
        SharedTexture { width, height, format: wgpu::TextureFormat::Rgba8Unorm }
    }
}

/// Whether the engine's device can import D3D11 shared textures: the Vulkan backend with external memory.
pub fn d3d11_import_supported(engine: &Engine) -> bool {
    let gpu = engine.gpu();
    gpu.adapter.get_info().backend == wgpu::Backend::Vulkan
        && gpu.device.features().contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32)
}

/// Imports the D3D11 texture behind `handle` (an NT shared handle, as `isize`) as a texture the engine samples and
/// copies from (`TEXTURE_BINDING | COPY_SRC`). The texture stays valid while the handle's resource lives; add it to a
/// session or a stream with `add_input_texture`.
pub fn d3d11_shared_texture(engine: &Engine, handle: isize, desc: &SharedTexture) -> Result<wgpu::Texture> {
    anyhow::ensure!(
        matches!(desc.format, wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm),
        "shared texture format {:?}: rgba8unorm or bgra8unorm",
        desc.format
    );
    anyhow::ensure!(handle != 0, "shared texture handle is null");
    anyhow::ensure!(
        d3d11_import_supported(engine),
        "this device cannot import D3D11 textures (Vulkan with VK_KHR_external_memory_win32 needed)"
    );
    let device = &engine.gpu().device;
    let size = wgpu::Extent3d { width: desc.width, height: desc.height, depth_or_array_layers: 1 };
    let hal_desc = wgpu::hal::TextureDescriptor {
        label: Some("fh_imported"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: desc.format,
        usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_SRC,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };
    let hal_tex = unsafe {
        let hal = device.as_hal::<wgpu::hal::api::Vulkan>().context("the engine device is not Vulkan")?;
        hal.texture_from_d3d11_shared_handle(windows::Win32::Foundation::HANDLE(handle as *mut _), &hal_desc)
            .map_err(|e| anyhow::anyhow!("importing the D3D11 texture: {e:?}"))?
    };
    let wgpu_desc = wgpu::TextureDescriptor {
        label: Some("fh_imported"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: desc.format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };
    // the image starts in an undefined layout; after wgpu's first transition to a sampled layout it is only read on
    // this side, so later D3D11 writes land in a layout Vulkan keeps
    Ok(unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_tex, &wgpu_desc, wgpu::TextureUses::UNINITIALIZED)
    })
}
