//! The final render: warped sources, visibility, residual, optional static protection; blend mixing.

use crate::{workgroups_2d, Gpu, LumaParams, NearKParams, NearParams, OpOutput, RenderParams, Tensor};

/// What the render reads: the two source frames (plain, full resolution), the flow and visibility maps of the phase
/// (`flow_tq`, `vis_tq`; views allowed) and the residual `R_t` (plain).
#[derive(Clone, Copy)]
pub struct RenderInputs<'a> {
    pub a: &'a Tensor,
    pub b: &'a Tensor,
    pub flow: &'a Tensor,
    pub vis: &'a Tensor,
    pub residual: &'a Tensor,
}

/// How it renders: `fsx` / `fsy` scale the flow's x / y channels (the view scale times the output / internal
/// resolution ratio when the network ran at a smaller internal size); `cubic` = bicubic source sampling (else
/// bilinear); `static_age` = static protection (`NetOpts::keep_static`): the age buffer and the age that protects,
/// warp sources on protected pixels are not used.
#[derive(Clone, Copy)]
pub struct RenderOpts<'a> {
    pub fsx: f32,
    pub fsy: f32,
    pub cubic: bool,
    /// render.wgsl's mode word: bit 0 decisive blend, bit 1 still test (`NetOpts::render_flags`)
    pub flags: u32,
    pub static_age: Option<(&'a wgpu::Buffer, u32)>,
}

impl Gpu {
    /// The plain render at the network's own resolution, no static protection.
    pub fn render(&self, a: &Tensor, b: &Tensor, ftq: &Tensor, vtq: &Tensor, rt: &Tensor, cubic: bool) -> OpOutput {
        self.render_ex(
            &RenderInputs { a, b, flow: ftq, vis: vtq, residual: rt },
            &RenderOpts { fsx: ftq.scale, fsy: ftq.scale, cubic, flags: 0, static_age: None },
        )
    }
    pub fn render_ex(&self, i: &RenderInputs, o: &RenderOpts) -> OpOutput {
        let (a, b, ftq, vtq, rt) = (i.a, i.b, i.flow, i.vis, i.residual);
        let out = self.alloc((3 * a.h * a.w) as usize);
        debug_assert!(a.is_plain() && b.is_plain() && rt.is_plain());
        let p = RenderParams {
            h: a.h,
            w: a.w,
            hq: ftq.h,
            wq: ftq.w,
            f0: ftq.c0,
            v0: vtq.c0,
            fsx: o.fsx,
            fsy: o.fsy,
            vs: vtq.scale,
            cubic: o.cubic as u32,
            min_age: o.static_age.map_or(0, |s| s.1),
            decisive: o.flags,
        };
        let ins = [&*a.buf, &*b.buf, &*ftq.buf, &*vtq.buf, &*rt.buf];
        let s = match o.static_age {
            Some((age, _)) => self.compute_step_ex(
                "render_static",
                &ins,
                bytemuck::bytes_of(&p),
                &out,
                &[age],
                workgroups_2d(a.w, a.h, 1),
            ),
            None => self.compute_step("render", &ins, bytemuck::bytes_of(&p), &out, workgroups_2d(a.w, a.h, 1)),
        };
        let mut steps = vec![s];
        if o.flags & 4 != 0 {
            // nearest-source regions (near_*.wgsl): where the warped frames disagree over a region, the nearer frame
            let np = NearParams { h: a.h, w: a.w, hq: ftq.h, wq: ftq.w, f0: ftq.c0, fsx: o.fsx, fsy: o.fsy, _a: 0 };
            let cells = (ftq.h * ftq.w) as usize;
            let (d, k) = (self.alloc(2 * cells), self.alloc(2 * cells));
            let groups_q = workgroups_2d(ftq.w, ftq.h, 1);
            steps.push(self.compute_step(
                "near_d",
                &[&*a.buf, &*b.buf, &*ftq.buf],
                bytemuck::bytes_of(&np),
                &d,
                groups_q,
            ));
            let kp = NearKParams { hq: ftq.h, wq: ftq.w, r: 3, _a: 0 };
            steps.push(self.compute_step("near_k", &[&*d], bytemuck::bytes_of(&kp), &k, groups_q));
            let tq = self.alloc(2);
            steps.push(self.compute_step("near_t", &[&*d], bytemuck::bytes_of(&kp), &tq, (1, 1, 1)));
            steps.push(self.compute_step(
                "near_apply",
                &[&*a.buf, &*b.buf, &*ftq.buf, &*k, &*tq],
                bytemuck::bytes_of(&np),
                &out,
                workgroups_2d(a.w, a.h, 1),
            ));
        }
        (Tensor { buf: out, c: 3, h: a.h, w: a.w, c0: 0, scale: 1.0 }, steps)
    }
    // fused blend_weights + 2 mul + add: out[c] = wa[c] * pA + wb[c] * pB with (pA, pB) = blend weights of `vis` (view ok)
    pub fn blendmix(&self, wa: &Tensor, wb: &Tensor, vis: &Tensor) -> OpOutput {
        debug_assert!(wa.is_plain() && wb.is_plain() && wa.c == 3 && wb.c == 3);
        let out = self.alloc(wa.n());
        let p = LumaParams { h: wa.h, w: wa.w, c0: vis.c0, scale: vis.scale };
        let s = self.compute_step(
            "blendmix",
            &[&wa.buf, &wb.buf, &vis.buf],
            bytemuck::bytes_of(&p),
            &out,
            workgroups_2d(wa.w, wa.h, 1),
        );
        (Tensor { buf: out, c: 3, h: wa.h, w: wa.w, c0: 0, scale: 1.0 }, vec![s])
    }
}
