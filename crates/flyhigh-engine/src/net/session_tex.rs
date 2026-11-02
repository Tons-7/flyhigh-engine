//! `SessionTex`: the two-texture ping-pong session of the browser canvas path.

use crate::{build_network, clone_step, Blob, Gpu, NetOpts, Step, Tensor, TuneTable};
use serde_json::Value;

// Texture-to-texture session: the two input frames live in GPU textures (filled by copyExternalImageToTexture from a
// video / canvas / ImageBitmap, or by any texture copy), the output goes to a storage texture view (the canvas'
// current texture). Inputs ping-pong: `next_input()` is the texture to fill for the newest frame; `run` reads the
// previous one as A and the newest as B. No CPU bytes anywhere.
pub struct SessionTex {
    pub w: u32,
    pub h: u32,
    pub wp: u32,
    pub hp: u32,
    pub k: u32,
    pub t: f32,
    pub tex: [wgpu::Texture; 2],
    pub view: [wgpu::TextureView; 2],
    pub cur: std::cell::Cell<usize>,
    pub pushed: std::cell::Cell<u32>,
    pub a: Tensor,
    pub b: Tensor,
    pub plan: Vec<Step>,
    pub render: Tensor,
    in_steps: [[Step; 2]; 2],
} // [texture][0 = as A, 1 = as B]: built once, the ping-pong only swaps them
impl SessionTex {
    #[allow(clippy::too_many_arguments)] // the Core `Session` is the public constructor; the tools and the plan tests call this
    pub fn new(
        gpu: &Gpu,
        blob: &Blob,
        ops: &[Value],
        w: u32,
        h: u32,
        k: u32,
        t: f32,
        iw: u32,
        ih: u32,
        tune: Option<&TuneTable>,
        net: &NetOpts,
    ) -> SessionTex {
        let (hp, wp) = (h.div_ceil(32) * 32, w.div_ceil(32) * 32);
        let tex = [gpu.input_texture(w, h), gpu.input_texture(w, h)];
        let view = [
            tex[0].create_view(&wgpu::TextureViewDescriptor::default()),
            tex[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];
        let a = Tensor { buf: gpu.alloc((3 * hp * wp) as usize), c: 3, h: hp, w: wp, c0: 0, scale: 1.0 };
        let b = Tensor { buf: gpu.alloc((3 * hp * wp) as usize), c: 3, h: hp, w: wp, c0: 0, scale: 1.0 };
        let mut plan = Vec::new();
        let (render, _env) = build_network(gpu, blob, ops, &a, &b, k, t, iw, ih, &mut plan, tune, net);
        let in_steps = [0usize, 1].map(|i| [gpu.tex_in_step(&view[i], &a, w, h), gpu.tex_in_step(&view[i], &b, w, h)]);
        SessionTex {
            w,
            h,
            wp,
            hp,
            k,
            t,
            tex,
            view,
            cur: std::cell::Cell::new(0),
            pushed: std::cell::Cell::new(0),
            a,
            b,
            plan,
            render,
            in_steps,
        }
    }
    // the texture to fill with the newest frame; call `commit()` after copying into it
    pub fn next_input(&self) -> &wgpu::Texture {
        &self.tex[self.cur.get() ^ 1]
    }
    pub fn commit(&self) {
        self.cur.set(self.cur.get() ^ 1);
        self.pushed.set(self.pushed.get() + 1);
    }
    pub fn ready(&self) -> bool {
        self.pushed.get() >= 2
    }
    // the full per-frame step list: tex_in(prev -> A), tex_in(cur -> B), network, tex_out(render -> out)
    pub fn frame_steps(&self, gpu: &Gpu, out: &wgpu::TextureView, bgra: bool) -> Vec<Step> {
        let (prev, cur) = (self.cur.get() ^ 1, self.cur.get());
        let mut steps = vec![clone_step(&self.in_steps[prev][0]), clone_step(&self.in_steps[cur][1])];
        steps.extend(self.plan.iter().map(clone_step));
        steps.push(gpu.tex_out_step(&self.render, out, self.w, self.h, bgra));
        steps
    }
    pub fn run(&self, gpu: &Gpu, out: &wgpu::TextureView, bgra: bool) {
        let steps = self.frame_steps(gpu, out, bgra);
        gpu.replay(&steps);
    }
}
