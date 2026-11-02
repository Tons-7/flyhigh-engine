// A D3D11 texture into the engine without a copy (Windows): a D3D11 device writes two frames into shared textures, the
// Vulkan engine imports them as a session's inputs and generates the in-between. Ignored by default; needs a GPU with
// both D3D11 and Vulkan (`cargo test --release --test interop_d3d11 -- --ignored`).
#![cfg(windows)]
use flyhigh_engine::interop::{d3d11_import_supported, d3d11_shared_texture, SharedTexture};
use flyhigh_engine::{Engine, EngineOptions, Model, Session, SessionOptions};
use std::path::PathBuf;
use windows::core::{Interface, PCWSTR};
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{IDXGIResource1, DXGI_SHARED_RESOURCE_READ, DXGI_SHARED_RESOURCE_WRITE};

const W: u32 = 256;
const H: u32 = 192;

fn frame(dx: f32) -> Vec<u8> {
    let mut v = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let u = (x as f32 - dx) * 0.07;
            let w = y as f32 * 0.05;
            let o = ((y * W + x) * 4) as usize;
            v[o] = (128.0 + 100.0 * u.sin()) as u8;
            v[o + 1] = (128.0 + 100.0 * (u * 0.7 + w).cos()) as u8;
            v[o + 2] = (128.0 + 90.0 * (u + w).sin()) as u8;
            v[o + 3] = 255;
        }
    }
    v
}

fn mae(a: &[u8], b: &[u8]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / a.len() as f64
}

struct D3d {
    device: ID3D11Device,
    ctx: ID3D11DeviceContext,
}

impl D3d {
    fn new() -> Option<D3d> {
        let (mut device, mut ctx) = (None, None);
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                windows::Win32::Foundation::HMODULE::default(),
                D3D11_CREATE_DEVICE_FLAG(0),
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut ctx),
            )
        }
        .ok()?;
        Some(D3d { device: device?, ctx: ctx? })
    }

    /// A shared texture holding `rgba`, written and waited for, and its NT handle.
    fn shared(&self, rgba: &[u8]) -> (ID3D11Texture2D, isize) {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: W,
            Height: H,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: (D3D11_RESOURCE_MISC_SHARED.0 | D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0) as u32,
        };
        let mut tex = None;
        unsafe { self.device.CreateTexture2D(&desc, None, Some(&mut tex)) }.expect("CreateTexture2D");
        let tex = tex.expect("texture");
        unsafe { self.ctx.UpdateSubresource(&tex, 0, None, rgba.as_ptr() as *const _, W * 4, 0) };
        // a staging copy mapped for reading waits for the GPU: the shared texture is complete when we hand it over
        let staging_desc = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
            ..desc
        };
        let mut staging = None;
        unsafe { self.device.CreateTexture2D(&staging_desc, None, Some(&mut staging)) }.expect("staging");
        let staging = staging.expect("staging texture");
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            self.ctx.CopyResource(&staging, &tex);
            self.ctx.Flush();
            self.ctx.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped)).expect("Map");
            self.ctx.Unmap(&staging, 0);
        }
        let res: IDXGIResource1 = tex.cast().expect("IDXGIResource1");
        let handle = unsafe {
            res.CreateSharedHandle(None, DXGI_SHARED_RESOURCE_READ.0 | DXGI_SHARED_RESOURCE_WRITE.0, PCWSTR::null())
        }
        .expect("CreateSharedHandle");
        (tex, handle.0 as isize)
    }
}

#[test]
#[ignore]
fn a_d3d11_texture_feeds_the_engine_without_a_copy() {
    let engine = match Engine::new(&EngineOptions { backend: Some(wgpu::Backends::VULKAN), ..EngineOptions::default() })
    {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipped: no Vulkan engine ({e:#})");
            return;
        }
    };
    if !d3d11_import_supported(&engine) {
        eprintln!("skipped: {} has no VK_KHR_external_memory_win32", engine.info().name);
        return;
    }
    let Some(d3d) = D3d::new() else {
        eprintln!("skipped: no D3D11 hardware device");
        return;
    };
    let (_keep_a, ha) = d3d.shared(&frame(0.0));
    let (_keep_b, hb) = d3d.shared(&frame(8.0));
    let desc = SharedTexture::rgba8(W, H);
    let ta = d3d11_shared_texture(&engine, ha, &desc).expect("import a");
    let tb = d3d11_shared_texture(&engine, hb, &desc).expect("import b");
    assert_eq!((ta.width(), ta.height(), ta.format()), (W, H, wgpu::TextureFormat::Rgba8Unorm));

    let model =
        Model::from_dir(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("models")).unwrap();
    let mut s = Session::new(&engine, &model, &SessionOptions::new(W, H)).unwrap();
    let a = s.add_input_texture(ta).unwrap();
    let b = s.add_input_texture(tb).unwrap();
    let out = s.add_output();
    s.pair(a, b).unwrap();
    s.mid(0.5, out).unwrap();
    let got = s.read_rgba(out).unwrap().block(engine.gpu()).unwrap();
    let (f0, f1, mid) = (frame(0.0), frame(8.0), frame(4.0));
    let (to_mid, to_f0, to_f1) = (mae(&got, &mid), mae(&got, &f0), mae(&got, &f1));
    eprintln!("imported D3D11 frames: MAE to the true middle {to_mid:.2}, to f0 {to_f0:.2}, to f1 {to_f1:.2}");
    assert!(
        to_mid < to_f0 && to_mid < to_f1 && to_mid < 6.0,
        "the in-between of two imported frames lies between them"
    );
    let wrong = s.add_input_texture(engine.gpu().output_texture(W, H));
    assert!(wrong.is_ok(), "an output-class texture is sampleable too");
    let small = engine.gpu().input_texture(64, 64);
    assert!(s.add_input_texture(small).is_err(), "a texture of another size is refused");
}
