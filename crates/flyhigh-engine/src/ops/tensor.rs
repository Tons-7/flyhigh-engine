//! `Tensor`: a buffer with its channel / height / width shape, channel offset and scale (views).

use std::rc::Rc;

#[derive(Clone)]
// A tensor is [c][h][w] f16 in `buf`, or a VIEW of it: channels [c0, c0+c) of the buffer, values x `scale`.
// Views make slice / scale / neg / mul_t free (no dispatch): kernels that understand them read with the offset and
// multiply on load; kernels that do not get a materialized copy (Gpu::mat) first.
pub struct Tensor {
    pub buf: Rc<wgpu::Buffer>,
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub c0: u32,
    pub scale: f32,
}
impl Tensor {
    pub fn is_plain(&self) -> bool {
        self.c0 == 0 && self.scale == 1.0
    }
    pub fn view(&self, a: u32, b: u32) -> Tensor {
        Tensor { buf: self.buf.clone(), c: b - a, h: self.h, w: self.w, c0: self.c0 + a, scale: self.scale }
    }
    pub fn scaled(&self, f: f32) -> Tensor {
        Tensor { buf: self.buf.clone(), c: self.c, h: self.h, w: self.w, c0: self.c0, scale: self.scale * f }
    }
}
impl Tensor {
    pub fn n(&self) -> usize {
        (self.c * self.h * self.w) as usize
    }
}
