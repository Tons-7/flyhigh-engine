//! `Session`: the CPU-frame path, two rgba buffers in and one out, one plan.

use crate::{build_network, map_read, workgroups_2d, Blob, Gpu, MapWait, NetOpts, RgbaParams, Step, Tensor};
use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::rc::Rc;

pub struct Session {
    pub w: u32,
    pub h: u32,
    pub wp: u32,
    pub hp: u32,
    pub k: u32,
    pub t: f32,
    pub in_a: Rc<wgpu::Buffer>,
    pub in_b: Rc<wgpu::Buffer>,
    pub out_rgba: Rc<wgpu::Buffer>,
    pub plan: Vec<Step>,
    pub env: HashMap<String, Tensor>,
}

impl Session {
    /// full resolution, default network options (`NetOpts::new`)
    pub fn new(gpu: &Gpu, blob: &Blob, ops: &[Value], w: u32, h: u32, k: u32, t: f32) -> Session {
        Session::new_int(gpu, blob, ops, w, h, k, t, w, h, &NetOpts::new(blob))
    }
    // Internal-resolution mode (LSFG's trick for high output resolutions): the network runs on the frames downscaled
    // to iw x ih (antialiased), and only the final render (warp + blend + residual) runs at the output size, with the
    // flow scaled by the output/internal ratio per axis. iw == w && ih == h is the plain full-resolution path.
    #[allow(clippy::too_many_arguments)] // the Core `Session` is the public constructor; the tools and the plan tests call this
    pub fn new_int(
        gpu: &Gpu,
        blob: &Blob,
        ops: &[Value],
        w: u32,
        h: u32,
        k: u32,
        t: f32,
        iw: u32,
        ih: u32,
        net: &NetOpts,
    ) -> Session {
        let (hp, wp) = (h.div_ceil(32) * 32, w.div_ceil(32) * 32);
        let u32buf = |n: u64| {
            Rc::new(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fh_rgba"),
                size: n * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))
        };
        let (in_a, in_b, out_rgba) = (u32buf((w * h) as u64), u32buf((w * h) as u64), u32buf((w * h) as u64));
        let a = Tensor { buf: gpu.alloc((3 * hp * wp) as usize), c: 3, h: hp, w: wp, c0: 0, scale: 1.0 };
        let b = Tensor { buf: gpu.alloc((3 * hp * wp) as usize), c: 3, h: hp, w: wp, c0: 0, scale: 1.0 };
        let rp = RgbaParams { w, h, wp, hp };
        let mut plan = vec![
            gpu.compute_step("rgba_in", &[&in_a], bytemuck::bytes_of(&rp), &a.buf, workgroups_2d(wp, hp, 1)),
            gpu.compute_step("rgba_in", &[&in_b], bytemuck::bytes_of(&rp), &b.buf, workgroups_2d(wp, hp, 1)),
        ];
        let (r, env) = build_network(gpu, blob, ops, &a, &b, k, t, iw, ih, &mut plan, None, net);
        plan.push(gpu.compute_step("rgba_out", &[&r.buf], bytemuck::bytes_of(&rp), &out_rgba, workgroups_2d(w, h, 1)));
        Session { w, h, wp, hp, k, t, in_a, in_b, out_rgba, plan, env }
    }
    // upload two RGBA8 frames (w*h*4 bytes each) and run the whole plan in one submit
    pub fn submit(&self, gpu: &Gpu, a_rgba: &[u8], b_rgba: &[u8]) {
        debug_assert!(a_rgba.len() == (self.w * self.h * 4) as usize && b_rgba.len() == a_rgba.len());
        gpu.queue.write_buffer(&self.in_a, 0, a_rgba);
        gpu.queue.write_buffer(&self.in_b, 0, b_rgba);
        gpu.replay(&self.plan);
    }
    /// Starts reading the generated frame back: copies into a staging buffer and requests the mapping; the caller
    /// awaits the returned future (browser: resolved by the mapAsync promise; native: after device.poll) and then
    /// reads `stage`.
    pub fn begin_read(&self, gpu: &Gpu) -> (wgpu::Buffer, MapWait) {
        let n = (self.w * self.h * 4) as u64;
        let stage = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fh_read_stage"),
            size: n,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.out_rgba, 0, &stage, 0, n);
        gpu.queue.submit(Some(enc.finish()));
        let fut = map_read(&stage);
        (stage, fut)
    }
    pub async fn read_rgba(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        let (stage, fut) = self.begin_read(gpu);
        #[cfg(not(target_arch = "wasm32"))]
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        fut.await.map_err(|e| anyhow::anyhow!("map_async: {e:?}"))?;
        let v = {
            let d = stage.slice(..).get_mapped_range().map_err(|e| anyhow::anyhow!("mapped range: {e:?}"))?;
            d.to_vec()
        };
        stage.unmap();
        Ok(v)
    }
}
