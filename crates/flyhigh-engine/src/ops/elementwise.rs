//! Binary and unary kernels, channel copies (views materialised), concat, zeros.

use crate::{workgroups_2d, BinaryParams, ChannelCopyParams, Gpu, OpOutput, Step, Tensor, UnaryParams};
use std::rc::Rc;

impl Gpu {
    pub fn binary(&self, a: &Tensor, b: &Tensor, op: u32) -> OpOutput {
        let out = self.alloc(a.n());
        let p = BinaryParams {
            c: a.c,
            h: a.h,
            w: a.w,
            cb: b.c,
            op,
            a0: a.c0,
            b0: b.c0,
            _a: 0,
            sa: a.scale,
            sb: b.scale,
            _b: 0,
            _c: 0,
        };
        let s =
            self.compute_step("binary", &[&a.buf, &b.buf], bytemuck::bytes_of(&p), &out, workgroups_2d(a.w, a.h, a.c));
        (Tensor { buf: out, c: a.c, h: a.h, w: a.w, c0: 0, scale: 1.0 }, vec![s])
    }
    pub fn unary(&self, x: &Tensor, op: u32, a: f32, b: f32) -> OpOutput {
        let out = self.alloc(x.n());
        let p = UnaryParams { c: x.c, h: x.h, w: x.w, op, a, b, c0: x.c0, scale: x.scale };
        let s = self.compute_step("unary", &[&x.buf], bytemuck::bytes_of(&p), &out, workgroups_2d(x.w, x.h, x.c));
        (Tensor { buf: out, c: x.c, h: x.h, w: x.w, c0: 0, scale: 1.0 }, vec![s])
    }
}

impl Gpu {
    // channel-range copy into `out` at channel dst0 from up to 7 inputs (input 0 starting at channel off0), as one
    // `chcopy` dispatch — keeps the whole frame inside one compute pass (buffer copies split passes: ~10% of K=2 time)
    // `first`: (channel offset, channel count) read from input 0; `size`: (h, w) of every input
    fn chcopy(&self, ins: &[&Tensor], first: (u32, u32), out: &Rc<wgpu::Buffer>, dst0: u32, size: (u32, u32)) -> Step {
        let ((off0, nch0), (h, w)) = (first, size);
        debug_assert!(!ins.is_empty() && ins.len() <= 7 && (h * w).is_multiple_of(2));
        let hw2 = h * w / 2;
        let mut ends = [u32::MAX; 7];
        let mut offs = [0u32; 7];
        let mut scl = [1f32; 7];
        let mut acc = 0u32;
        for (k, t) in ins.iter().enumerate() {
            acc += if k == 0 { nch0 } else { t.c };
            ends[k] = acc;
            offs[k] = if k == 0 { off0 } else { t.c0 };
            scl[k] = t.scale; // views: read at their channel offset, scaled
        }
        for e in ends.iter_mut().skip(ins.len()) {
            *e = acc;
        }
        let p = ChannelCopyParams { hw2, ctot: acc, e: ends, o: offs, dst0, s: scl, _a: 0, _b: 0, _c: 0, _d: 0 };
        let bufs: Vec<&wgpu::Buffer> = (0..7).map(|k| if k < ins.len() { &*ins[k].buf } else { &*self.zero }).collect();
        self.compute_step("chcopy", &bufs, bytemuck::bytes_of(&p), out, ((acc * hw2).div_ceil(256), 1, 1))
    }
    pub fn slice(&self, x: &Tensor, a: u32, b: u32) -> OpOutput {
        // a view: no dispatch
        (x.view(a, b), vec![])
    }
    pub fn concat(&self, ts: &[Tensor]) -> OpOutput {
        let (h, w) = (ts[0].h, ts[0].w);
        let c: u32 = ts.iter().map(|t| t.c).sum();
        let out = self.alloc((c * h * w) as usize);
        let mut steps = Vec::new();
        let mut dst0 = 0u32;
        for chunk in ts.chunks(7) {
            let refs: Vec<&Tensor> = chunk.iter().collect();
            steps.push(self.chcopy(&refs, (chunk[0].c0, chunk[0].c), &out, dst0, (h, w)));
            dst0 += chunk.iter().map(|t| t.c).sum::<u32>();
        }
        (Tensor { buf: out, c, h, w, c0: 0, scale: 1.0 }, steps)
    }
    pub fn zeros(&self, c: u32, h: u32, w: u32) -> OpOutput {
        (Tensor { buf: self.buf(&vec![0f32; (c * h * w) as usize]), c, h, w, c0: 0, scale: 1.0 }, vec![])
    }
}

impl Gpu {
    // materialize a view into a plain tensor (one chcopy dispatch); plain tensors pass through for free
    pub fn mat(&self, x: &Tensor) -> OpOutput {
        if x.is_plain() {
            return (x.clone(), vec![]);
        }
        let out = self.alloc(x.n());
        self.note_scratch(&out);
        let s = self.chcopy(&[x], (x.c0, x.c), &out, 0, (x.h, x.w));
        (Tensor { buf: out, c: x.c, h: x.h, w: x.w, c0: 0, scale: 1.0 }, vec![s])
    }
}
