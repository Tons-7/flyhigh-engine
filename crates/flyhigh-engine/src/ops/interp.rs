//! Resampling kernels: antialiased interpolation and average pooling.

use crate::{workgroups_2d, Gpu, InterpParams, OpOutput, PoolParams, Tensor};

impl Gpu {
    pub fn interp(&self, x: &Tensor, oh: u32, ow: u32, aa: bool) -> OpOutput {
        let out = self.alloc((x.c * oh * ow) as usize);
        let p = InterpParams { c: x.c, ih: x.h, iw: x.w, oh, ow, aa: aa as u32, c0: x.c0, scale: x.scale };
        let s = self.compute_step("interp", &[&x.buf], bytemuck::bytes_of(&p), &out, workgroups_2d(ow, oh, x.c));
        (Tensor { buf: out, c: x.c, h: oh, w: ow, c0: 0, scale: 1.0 }, vec![s])
    }
    pub fn avgpool(&self, x: &Tensor, k: u32) -> OpOutput {
        let (oh, ow) = (x.h / k, x.w / k);
        let out = self.alloc((x.c * oh * ow) as usize);
        let p = PoolParams { c: x.c, h: x.h, w: x.w, oh, ow, k, _a: 0, _b: 0 };
        let s = self.compute_step("avgpool", &[&x.buf], bytemuck::bytes_of(&p), &out, workgroups_2d(ow, oh, x.c));
        (Tensor { buf: out, c: x.c, h: oh, w: ow, c0: 0, scale: 1.0 }, vec![s])
    }
}
