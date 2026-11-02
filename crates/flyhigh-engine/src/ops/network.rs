//! Kernels specific to the FlyHighFG network: backward warp, luma and SAD, convex upsampling, FiLM, blend.

use crate::{workgroups_2d, Gpu, LumaParams, OpOutput, SadParams, Shape3Params, Tensor, WarpParams};

impl Gpu {
    pub fn warp(&self, img: &Tensor, flow: &Tensor) -> OpOutput {
        let out = self.alloc(img.n());
        let p =
            WarpParams { c: img.c, h: img.h, w: img.w, f0: flow.c0, i0: img.c0, fs: flow.scale, is: img.scale, _a: 0 };
        let s = self.compute_step(
            "warp",
            &[&img.buf, &flow.buf],
            bytemuck::bytes_of(&p),
            &out,
            workgroups_2d(img.w, img.h, img.c),
        );
        (Tensor { buf: out, c: img.c, h: img.h, w: img.w, c0: 0, scale: 1.0 }, vec![s])
    }
}

impl Gpu {
    pub fn luma(&self, x: &Tensor, grad: bool) -> OpOutput {
        let out = self.alloc((x.h * x.w) as usize);
        let p = LumaParams { h: x.h, w: x.w, c0: x.c0, scale: x.scale };
        let s = self.compute_step(
            if grad { "grad" } else { "luma" },
            &[&x.buf],
            bytemuck::bytes_of(&p),
            &out,
            workgroups_2d(x.w, x.h, 1),
        );
        (Tensor { buf: out, c: 1, h: x.h, w: x.w, c0: 0, scale: 1.0 }, vec![s])
    }
    pub fn sad(&self, la: &Tensor, lb: &Tensor, r: u32) -> OpOutput {
        let span = 2 * r + 1;
        let cout = span * span;
        let out = self.alloc((cout * la.h * la.w) as usize);
        let p = SadParams { h: la.h, w: la.w, radius: r, span };
        let s = self.compute_step(
            "sad",
            &[&la.buf, &lb.buf],
            bytemuck::bytes_of(&p),
            &out,
            workgroups_2d(la.w, la.h, cout),
        );
        (Tensor { buf: out, c: cout, h: la.h, w: la.w, c0: 0, scale: 1.0 }, vec![s])
    }
    pub fn convex(&self, wt36: &Tensor, x: &Tensor) -> OpOutput {
        let (oh, ow) = (2 * x.h, 2 * x.w);
        let out = self.alloc((x.c * oh * ow) as usize);
        let p = Shape3Params { c: x.c, h: x.h, w: x.w, _a: 0 };
        let s =
            self.compute_step("convex", &[&wt36.buf, &x.buf], bytemuck::bytes_of(&p), &out, workgroups_2d(ow, oh, x.c));
        (Tensor { buf: out, c: x.c, h: oh, w: ow, c0: 0, scale: 1.0 }, vec![s])
    }
    pub fn film(&self, h: &Tensor, g: &Tensor) -> OpOutput {
        let out = self.alloc(h.n());
        let p = Shape3Params { c: h.c, h: h.h, w: h.w, _a: 0 };
        let s =
            self.compute_step("film", &[&h.buf, &g.buf], bytemuck::bytes_of(&p), &out, workgroups_2d(h.w, h.h, h.c));
        (Tensor { buf: out, c: h.c, h: h.h, w: h.w, c0: 0, scale: 1.0 }, vec![s])
    }
    pub fn blend(&self, vis: &Tensor) -> OpOutput {
        let out = self.alloc((2 * vis.h * vis.w) as usize);
        let p = LumaParams { h: vis.h, w: vis.w, c0: vis.c0, scale: vis.scale };
        let s = self.compute_step("blend", &[&vis.buf], bytemuck::bytes_of(&p), &out, workgroups_2d(vis.w, vis.h, 1));
        (Tensor { buf: out, c: 2, h: vis.h, w: vis.w, c0: 0, scale: 1.0 }, vec![s])
    }
}
