//! `SessionExt`: the texture path with input / output slots, a pair plan, a mid plan and the fast render.

use crate::ops::step::meta;
use crate::{
    build_network_ex, clone_step, set_t, workgroups_2d, Binding, Blob, Gpu, NetOpts, PhaseDep, StaticParams, Step,
    Tensor, TuneTable,
};
use bytemuck::{Pod, Zeroable};
use serde_json::Value;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

// Host-pool session (the browser extension's shape): the host owns N input frame textures (a history ring it also
// presents from) and M output textures; the runtime runs the pair stage on any two input slots and then any number of
// mids (any t, rewritten in place) into an output slot or the canvas. Pair and mid plans are static; t is a uniform.
// The texture in / out steps of a slot are built once and cached (keyed by slot index): building one creates a uniform
// buffer and a bind group, which in the browser are garbage-collected JS objects (per-frame churn otherwise).
pub struct SessionExt {
    pub w: u32,
    pub h: u32,
    pub wp: u32,
    pub hp: u32,
    pub k: u32,
    pub ins: Vec<(wgpu::Texture, wgpu::TextureView)>,
    pub outs: Vec<(wgpu::Texture, wgpu::TextureView)>,
    pub a: Tensor,
    pub b: Tensor,
    pub plan_pair: Vec<Step>,
    pub plan_mid: Vec<Step>,
    pub render: Tensor,
    pub t_deps: Vec<PhaseDep>,
    pub cur_t: std::cell::Cell<f32>,
    pub env: HashMap<String, Tensor>,
    pub net: NetOpts,
    in_steps: std::cell::RefCell<HashMap<(usize, bool), Step>>, // (slot, is B) -> tex_in step
    out_steps: std::cell::RefCell<HashMap<usize, Step>>,        // slot -> tex_out step
    fast: Option<(wgpu::Sampler, f32, f32)>,                    // fast render: sampler, flow scale x / y
    fast_steps: std::cell::RefCell<FastRenderCache>,            // (slot A, slot B, out) -> render_tex
    age_steps: std::cell::RefCell<HashMap<(usize, usize), Step>>, // fast render + static: (slot A, slot B) -> age pass
}
// the cached render_tex step of a (slot A, slot B, out) triple with its uniform buffer and the parameters it holds
type FastRenderCache = HashMap<(usize, usize, usize), (Step, wgpu::Buffer, RenderTexParams)>;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RenderTexParams {
    w: u32,
    h: u32,
    wp: u32,
    hp: u32,
    hq: u32,
    wq: u32,
    f0: u32,
    v0: u32,
    fsx: f32,
    fsy: f32,
    vs: f32,
    min_age: f32,
    use_static: u32,
    sa: f32,
    sb: f32,
    sr: f32,
    cubic: u32,
    _a: u32,
    _b: u32,
    _c: u32,
}
impl SessionExt {
    #[allow(clippy::too_many_arguments)] // the Core `Session` is the public constructor; the tools and the plan tests call this
    pub fn new(
        gpu: &Gpu,
        blob: &Blob,
        ops: &[Value],
        w: u32,
        h: u32,
        k: u32,
        iw: u32,
        ih: u32,
        tune: Option<&TuneTable>,
        net: &NetOpts,
    ) -> SessionExt {
        let (hp, wp) = (h.div_ceil(32) * 32, w.div_ceil(32) * 32);
        // fast render reads the source textures directly: the full-resolution input tensors keep only their shape
        let fast_in = net.fast_render && (iw.div_ceil(32) * 32, ih.div_ceil(32) * 32) != (wp, hp);
        let n = if fast_in { 8 } else { (3 * hp * wp) as usize };
        let a = Tensor { buf: gpu.alloc(n), c: 3, h: hp, w: wp, c0: 0, scale: 1.0 };
        let b = Tensor { buf: gpu.alloc(n), c: 3, h: hp, w: wp, c0: 0, scale: 1.0 };
        let (mut plan_pair, mut plan_mid) = (Vec::new(), Vec::new());
        let (render, env, t_deps) =
            build_network_ex(gpu, blob, ops, &a, &b, k, 0.5, iw, ih, &mut plan_pair, Some(&mut plan_mid), tune, net);
        let fast = env.get("fast:ftq").map(|ftq| {
            let (ihp, iwp) = (ih.div_ceil(32) * 32, iw.div_ceil(32) * 32);
            let samp = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("fh_render_tex"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            (samp, ftq.scale * wp as f32 / iwp as f32, ftq.scale * hp as f32 / ihp as f32)
        });
        SessionExt {
            w,
            h,
            wp,
            hp,
            k,
            ins: Vec::new(),
            outs: Vec::new(),
            a,
            b,
            plan_pair,
            plan_mid,
            render,
            t_deps,
            cur_t: std::cell::Cell::new(0.5),
            env,
            net: *net,
            in_steps: Default::default(),
            out_steps: Default::default(),
            fast,
            fast_steps: Default::default(),
            age_steps: Default::default(),
        }
    }
    /// built with `NetOpts::fast_render` (and an internal size below the frame size): generate with `mid_fast`
    pub fn is_fast(&self) -> bool {
        self.fast.is_some()
    }
    /// The protection mask (`NetOpts::protection_mask`): one byte per pixel of the frame, 0 interpolates, anything else
    /// is taken from the sources. The padded plane replicates the frame's edge like the age buffer.
    pub fn write_mask(&self, gpu: &Gpu, mask: &[u8]) -> anyhow::Result<()> {
        let m = self.env.get("mask").ok_or_else(|| anyhow::anyhow!("the session was built without protection_mask"))?;
        let (w, h, wp, hp) = (self.w as usize, self.h as usize, self.wp as usize, self.hp as usize);
        anyhow::ensure!(mask.len() == w * h, "mask is {} bytes, {w}x{h} = {} expected", mask.len(), w * h);
        let plane: Vec<f32> = (0..hp)
            .flat_map(|y| {
                let row = &mask[y.min(h - 1) * w..][..w];
                (0..wp).map(move |x| if row[x.min(w - 1)] != 0 { 1.0 } else { 0.0 })
            })
            .collect();
        if gpu.f32_mode {
            gpu.queue.write_buffer(&m.buf, 0, bytemuck::cast_slice(&plane));
        } else {
            let bits: Vec<u16> = plane.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
            gpu.queue.write_buffer(&m.buf, 0, bytemuck::cast_slice(&bits));
        }
        Ok(())
    }
    // fast mid stage: FiLM(t) .. maps, then render_tex from the pair's two source textures into output slot `out`
    pub fn mid_steps_fast(
        &self,
        gpu: &Gpu,
        a: (usize, &wgpu::TextureView),
        b: (usize, &wgpu::TextureView),
        out: (usize, &wgpu::TextureView),
    ) -> Vec<Step> {
        let mut s: Vec<Step> = self.plan_mid.iter().map(clone_step).collect();
        s.push(self.render_tex_step(gpu, a, b, out, None));
        s
    }
    // the render_tex step alone; `linear_t`: Some(t) reuses maps computed at t = 0.5 for phase t (else as computed)
    pub fn render_tex_step(
        &self,
        gpu: &Gpu,
        a: (usize, &wgpu::TextureView),
        b: (usize, &wgpu::TextureView),
        out: (usize, &wgpu::TextureView),
        linear_t: Option<f32>,
    ) -> Step {
        let (samp, fsx, fsy) = self.fast.as_ref().expect("session built without fast_render");
        let mut c = self.fast_steps.borrow_mut();
        let step = c.entry((a.0, b.0, out.0)).or_insert_with(|| {
            let (ftq, vtq, rt) = (&self.env["fast:ftq"], &self.env["fast:vtq"], &self.env["fast:rt"]);
            let age = self.env.get("fast:age");
            let p = RenderTexParams {
                w: self.w,
                h: self.h,
                wp: self.wp,
                hp: self.hp,
                hq: ftq.h,
                wq: ftq.w,
                f0: ftq.c0,
                v0: vtq.c0,
                fsx: *fsx,
                fsy: *fsy,
                vs: vtq.scale,
                min_age: age.map_or(0.0, |t| t.scale),
                use_static: age.is_some() as u32,
                sa: 1.0,
                sb: 1.0,
                sr: 1.0,
                cubic: self.net.cubic as u32,
                _a: 0,
                _b: 0,
                _c: 0,
            };
            let pb = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fh_render_tex_params"),
                contents: bytemuck::bytes_of(&p),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let age_buf = age.map_or(&gpu.zero, |t| &t.buf);
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fh_render_tex"),
                layout: &gpu.pipeline("render_tex").get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(a.1) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(b.1) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(samp) },
                    wgpu::BindGroupEntry { binding: 3, resource: ftq.buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: vtq.buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: rt.buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: age_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(out.1) },
                    wgpu::BindGroupEntry { binding: 8, resource: pb.as_entire_binding() },
                ],
            });
            let m = meta(
                vec![
                    Binding::Texture(self.w, self.h),
                    Binding::Texture(self.w, self.h),
                    Binding::Sampler,
                    Binding::Buffer(ftq.buf.size()),
                    Binding::Buffer(vtq.buf.size()),
                    Binding::Buffer(rt.buf.size()),
                    Binding::Buffer(age_buf.size()),
                    Binding::Texture(self.w, self.h),
                    Binding::Params,
                ],
                bytemuck::bytes_of(&p),
            );
            (Step::Compute { pipe: "render_tex", bg, groups: workgroups_2d(self.w, self.h, 1), meta: m }, pb, p)
        });
        let (sa, sb, sr) = match linear_t {
            Some(t) => (2.0 * t, 2.0 * (1.0 - t), 1.0 - (2.0 * t - 1.0).abs()),
            None => (1.0, 1.0, 1.0),
        };
        if (step.2.sa, step.2.sb, step.2.sr) != (sa, sb, sr) {
            step.2.sa = sa;
            step.2.sb = sb;
            step.2.sr = sr;
            gpu.queue.write_buffer(&step.1, 0, bytemuck::bytes_of(&step.2));
        }
        clone_step(&step.0)
    }
    /// linear_t: run the t-dependent network once for the current pair (at t = 0.5)
    pub fn mid_maps_half(&self, gpu: &Gpu) {
        self.set_t(gpu, 0.5);
        gpu.replay(&self.plan_mid.iter().map(clone_step).collect::<Vec<_>>());
    }
    pub fn mid_fast(
        &self,
        gpu: &Gpu,
        t: f32,
        a: (usize, &wgpu::TextureView),
        b: (usize, &wgpu::TextureView),
        out: (usize, &wgpu::TextureView),
    ) {
        self.set_t(gpu, t);
        gpu.replay(&self.mid_steps_fast(gpu, a, b, out));
    }
    pub fn add_input(&mut self, gpu: &Gpu) -> usize {
        let t = gpu.input_texture_l(self.w, self.h, &format!("fh_in_{}", self.ins.len()));
        let v = t.create_view(&wgpu::TextureViewDescriptor::default());
        self.ins.push((t, v));
        self.ins.len() - 1
    }
    pub fn add_output(&mut self, gpu: &Gpu) -> usize {
        let t = gpu.output_texture_l(self.w, self.h, &format!("fh_out_{}", self.outs.len()));
        let v = t.create_view(&wgpu::TextureViewDescriptor::default());
        self.outs.push((t, v));
        self.outs.len() - 1
    }
    fn tex_in_cached(&self, gpu: &Gpu, slot: usize, view: &wgpu::TextureView, is_b: bool) -> Step {
        let mut c = self.in_steps.borrow_mut();
        clone_step(
            c.entry((slot, is_b))
                .or_insert_with(|| gpu.tex_in_step(view, if is_b { &self.b } else { &self.a }, self.w, self.h)),
        )
    }
    // fast render: source texture -> network input at the internal size (tex_down.wgsl), cached per slot
    fn tex_down_cached(&self, gpu: &Gpu, slot: usize, view: &wgpu::TextureView, is_b: bool) -> Step {
        let mut c = self.in_steps.borrow_mut();
        clone_step(c.entry((slot, is_b)).or_insert_with(|| {
            let dst = &self.env[if is_b { "fast:bi" } else { "fast:ai" }];
            #[repr(C)]
            #[derive(Clone, Copy, Pod, Zeroable)]
            struct DownsampleParams {
                w: u32,
                h: u32,
                iw: u32,
                ih: u32,
                ow: u32,
                oh: u32,
                _a: u32,
                _b: u32,
            }
            let p =
                DownsampleParams { w: self.w, h: self.h, iw: self.wp, ih: self.hp, ow: dst.w, oh: dst.h, _a: 0, _b: 0 };
            let pb = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fh_tex_down_params"),
                contents: bytemuck::bytes_of(&p),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &gpu.pipeline("tex_down").get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                    wgpu::BindGroupEntry { binding: 1, resource: dst.buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: pb.as_entire_binding() },
                ],
            });
            Step::Compute {
                pipe: "tex_down",
                bg,
                groups: workgroups_2d(dst.w, dst.h, 1),
                meta: meta(
                    vec![Binding::Texture(self.w, self.h), Binding::Buffer(dst.buf.size()), Binding::Params],
                    bytemuck::bytes_of(&p),
                ),
            }
        }))
    }
    /// the tex_out step into output slot `slot` (whose texture view is `view`), cached
    pub fn tex_out_cached(&self, gpu: &Gpu, slot: usize, view: &wgpu::TextureView) -> Step {
        let mut c = self.out_steps.borrow_mut();
        clone_step(c.entry(slot).or_insert_with(|| gpu.tex_out_step(&self.render, view, self.w, self.h, false)))
    }
    // pair stage: input slots ia (previous) and ib (current) -> A, B tensors -> encoders, flow, fine iterations
    pub fn pair_steps(&self, gpu: &Gpu, ia: usize, ib: usize) -> Vec<Step> {
        self.pair_steps_views(gpu, (ia, &self.ins[ia].1), (ib, &self.ins[ib].1))
    }
    // pair stage on two texture views of a host-owned ring (gen_tex.rs), keyed by the host's slot index (cache key: the
    // view behind a slot must never change)
    pub fn pair_steps_views(
        &self,
        gpu: &Gpu,
        a: (usize, &wgpu::TextureView),
        b: (usize, &wgpu::TextureView),
    ) -> Vec<Step> {
        let mut s = Vec::with_capacity(self.plan_pair.len() + 3);
        if self.fast.is_some() {
            if self.env.contains_key("fast:ai") {
                s.push(self.tex_down_cached(gpu, a.0, a.1, false));
                s.push(self.tex_down_cached(gpu, b.0, b.1, true));
            } else {
                s.push(self.tex_in_cached(gpu, a.0, a.1, false));
                s.push(self.tex_in_cached(gpu, b.0, b.1, true));
            }
            if let Some(age) = self.env.get("fast:age") {
                let mut c = self.age_steps.borrow_mut();
                let step = c.entry((a.0, b.0)).or_insert_with(|| {
                    let p = StaticParams::new(&self.net, self.w, self.h, self.wp, self.hp);
                    let mask: &wgpu::Buffer = self.env.get("mask").map_or(&gpu.zero, |m| &m.buf);
                    let state = &self.env["static:state"].buf;
                    let pb = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("fh_age_params"),
                        contents: bytemuck::bytes_of(&p),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &gpu.pipeline("keep_static_age_tex").get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(a.1) },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(b.1) },
                            wgpu::BindGroupEntry { binding: 2, resource: age.buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 3, resource: pb.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 4, resource: mask.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 5, resource: state.as_entire_binding() },
                        ],
                    });
                    Step::Compute {
                        pipe: "keep_static_age_tex",
                        bg,
                        groups: workgroups_2d(self.wp, self.hp, 1),
                        meta: meta(
                            vec![
                                Binding::Texture(self.w, self.h),
                                Binding::Texture(self.w, self.h),
                                Binding::Buffer(age.buf.size()),
                                Binding::Params,
                                Binding::Buffer(mask.size()),
                                Binding::Buffer(state.size()),
                            ],
                            bytemuck::bytes_of(&p),
                        ),
                    }
                });
                s.push(clone_step(step));
            }
            s.extend(self.plan_pair.iter().map(clone_step));
            return s;
        }
        s.push(self.tex_in_cached(gpu, a.0, a.1, false));
        s.push(self.tex_in_cached(gpu, b.0, b.1, true));
        s.extend(self.plan_pair.iter().map(clone_step));
        s
    }
    // mid stage for the t last set: FiLM(t) .. render -> output slot `slot` (view `out`)
    pub fn mid_steps(&self, gpu: &Gpu, slot: usize, out: &wgpu::TextureView) -> Vec<Step> {
        let mut s: Vec<Step> = self.plan_mid.iter().map(clone_step).collect();
        s.push(self.tex_out_cached(gpu, slot, out));
        s
    }
    // mid stage into a one-off target (the canvas' current texture changes every frame: not cached)
    pub fn mid_steps_view(&self, gpu: &Gpu, out: &wgpu::TextureView, bgra: bool) -> Vec<Step> {
        let mut s: Vec<Step> = self.plan_mid.iter().map(clone_step).collect();
        s.push(gpu.tex_out_step(&self.render, out, self.w, self.h, bgra));
        s
    }
    pub fn set_t(&self, gpu: &Gpu, t: f32) {
        if t != self.cur_t.get() {
            set_t(gpu, &self.t_deps, t);
            self.cur_t.set(t);
        }
    }
    pub fn pair(&self, gpu: &Gpu, ia: usize, ib: usize) {
        gpu.replay(&self.pair_steps(gpu, ia, ib));
    }
    pub fn mid(&self, gpu: &Gpu, t: f32, slot: usize, out: &wgpu::TextureView) {
        self.set_t(gpu, t);
        gpu.replay(&self.mid_steps(gpu, slot, out));
    }
    pub fn mid_view(&self, gpu: &Gpu, t: f32, out: &wgpu::TextureView, bgra: bool) {
        self.set_t(gpu, t);
        gpu.replay(&self.mid_steps_view(gpu, out, bgra));
    }
}
