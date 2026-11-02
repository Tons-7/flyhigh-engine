//! The conv kernels: naive, cooperative-matrix (tensor cores), portable tiled FMA with split-K, direct.

use crate::{workgroups_2d, ConvParams, Gpu, ImplicitGemmParams, OpOutput, ShiftParams, SplitKParams, Step, Tensor};
use std::rc::Rc;

/// A 3x3 convolution as the kernels take it: the input, the packed weights with the paddings they were packed with
/// (`buildb`: kp = ceil16(cin * 9), np = cout padded to 32 or 64, cinp 0; `buildb_tap`: kp = 9 * cinp, np = cout padded
/// to 64), bias and PReLU slope per output channel (slope 1 = no activation), the stride.
#[derive(Clone, Copy)]
pub struct ConvArgs<'a> {
    pub x: &'a Tensor,
    pub weights: &'a wgpu::Buffer,
    pub bias: &'a wgpu::Buffer,
    pub slope: &'a wgpu::Buffer,
    pub cin: u32,
    pub cout: u32,
    pub kp: u32,
    pub np: u32,
    pub cinp: u32,
    pub stride: u32,
}

/// How the portable tiled kernel is instantiated (the autotuner's knobs). `variant` (cout > 32): 0 = tconv 64x64
/// (256 thr, 4x4), 1 = tconv8 128x64 (256 thr, 8x4), 2 = tconv88 128x64 (128 thr, 8x8), 3 = tconvm32 32x64 (128 thr,
/// 4x4), 4 = tconv8p (tconv8 with register prefetch). `ks`: split-K count, 0 = heuristic (fill >= 512 workgroups,
/// <= 8 splits). `tn_small` (cout <= 32): N tile 32 / 16 / 8 (B is padded to 32 columns so any of them may read it).
#[derive(Clone, Copy)]
pub struct TiledOpts {
    pub variant: u8,
    pub ks: u32,
    pub tn_small: u32,
}
impl Default for TiledOpts {
    fn default() -> Self {
        TiledOpts { variant: 0, ks: 0, tn_small: 32 }
    }
}

fn out_size(x: &Tensor, stride: u32) -> (u32, u32) {
    ((x.h + 2 - 3) / stride + 1, (x.w + 2 - 3) / stride + 1) // 3x3, pad 1
}

impl Gpu {
    /// The reference conv: one thread per output element, the torch weight layout as is (`conv.wgsl`). `c.weights` is
    /// the raw `[cout][cin][3][3]` buffer, `kp` / `np` / `cinp` are not used; `act` 1 = PReLU with `c.slope`, 0 = none.
    pub fn conv(&self, c: &ConvArgs, act: u32) -> OpOutput {
        let (x, cout, stride) = (c.x, c.cout, c.stride);
        let (hout, wout) = out_size(x, stride);
        let out = self.alloc((cout * hout * wout) as usize);
        let p = ConvParams { cin: x.c, cout, h: x.h, w: x.w, hout, wout, stride, pad: 1, act, _a: 0, _b: 0, _c: 0 };
        let s = self.compute_step(
            "conv",
            &[&x.buf, c.weights, c.bias, c.slope],
            bytemuck::bytes_of(&p),
            &out,
            workgroups_2d(wout, hout, cout),
        );
        (Tensor { buf: out, c: cout, h: hout, w: wout, c0: 0, scale: 1.0 }, vec![s])
    }
}

impl Gpu {
    /// Pack torch `[cout][cin][3][3]` weights as the implicit-GEMM B matrix `[kp][np]` (ci-major K padded to 16, N padded
    /// to 32 or 64); returns the buffer and the paddings.
    pub fn buildb(&self, wf: &[f32], cin: u32, cout: u32) -> (Rc<wgpu::Buffer>, u32, u32) {
        let k = cin * 9;
        let kp = k.div_ceil(16) * 16;
        let nblk = if cout <= 32 { 32 } else { 64 };
        let np = cout.div_ceil(nblk) * nblk;
        let mut b = vec![0f32; (kp * np) as usize];
        for co in 0..cout as usize {
            for kk in 0..(k as usize) {
                b[kk * np as usize + co] = wf[co * k as usize + kk];
            }
        }
        (self.buf(&b), kp, np)
    }
    fn gemm_params(&self, c: &ConvArgs, hout: u32, wout: u32, ksplit: u32, residual: u32) -> ImplicitGemmParams {
        ImplicitGemmParams {
            cin: c.cin,
            h: c.x.h,
            w: c.x.w,
            hout,
            wout,
            stride: c.stride,
            pad: 1,
            k: c.cin * 9,
            kp: c.kp,
            n: c.cout,
            np: c.np,
            m: hout * wout,
            cinp: c.cinp,
            ksplit,
            residual,
            _pad0: 0,
        }
    }
    /// Cooperative-matrix implicit GEMM (`fused` / `fused2`): A gathered from the input per 16-pixel tile.
    pub fn conv_tc(&self, c: &ConvArgs) -> OpOutput {
        let (hout, wout) = out_size(c.x, c.stride);
        let mp = (hout * wout).div_ceil(16) * 16;
        let out = self.alloc((c.cout * hout * wout) as usize);
        let ip = self.gemm_params(c, hout, wout, 0, 0);
        let nblk = if c.cout <= 32 { 32u32 } else { 64u32 };
        let ker = if c.cout <= 32 { "fused2" } else { "fused" };
        let s1 = self.compute_step(
            ker,
            &[&c.x.buf, c.weights, &self.zero, c.bias, c.slope],
            bytemuck::bytes_of(&ip),
            &out,
            (c.np / nblk, mp / 16, 1),
        );
        (Tensor { buf: out, c: c.cout, h: hout, w: wout, c0: 0, scale: 1.0 }, vec![s1])
    }
    /// Portable tiled FMA conv (no cooperative matrix): the browser / non-NVIDIA path, default tile and heuristic
    /// split-K. Same B layout as `conv_tc`.
    pub fn conv_tiled(&self, c: &ConvArgs) -> OpOutput {
        self.conv_tiled_with(c, TiledOpts::default(), None)
    }
    /// The tiled conv with explicit tile / split-K choices (`TiledOpts`). Split-K when the tile grid is too small to
    /// fill the GPU: the kernel writes f32 partials `[z][n][m]` and `splitk_reduce` applies bias, PReLU and the residual.
    /// `FH_KSPLIT=<n>` forces a split count. `res`: fused residual (out = f16(prelu(conv)) + res), a plain tensor of the
    /// output's shape.
    pub fn conv_tiled_with(&self, c: &ConvArgs, opts: TiledOpts, res: Option<&Tensor>) -> OpOutput {
        let (hout, wout) = out_size(c.x, c.stride);
        let m = hout * wout;
        let cout = c.cout;
        let out = self.alloc((cout * hout * wout) as usize);
        debug_assert!(res.is_none_or(|r| r.is_plain() && r.c == cout && r.h == hout && r.w == wout));
        // dummy for the unused residual must differ from the `part` dummy: one buffer cannot be bound read-only and
        // read-write in the same bind group
        let resb: &wgpu::Buffer = res.map(|r| &*r.buf).unwrap_or(&self.ones);
        let res_on = res.is_some() as u32;
        let (ker, tm, tn) = if cout <= 32 {
            (
                match opts.tn_small {
                    8 => "tconvn8",
                    16 => "tconv16",
                    _ => "tconv32",
                },
                128u32,
                opts.tn_small,
            )
        } else {
            match opts.variant {
                1 => ("tconv8", 128u32, 64u32),
                2 => ("tconv88", 128, 64),
                3 => ("tconvm32", 32, 64),
                4 => ("tconv8p", 128, 64),
                _ => ("tconv", 64, 64),
            }
        };
        debug_assert!(c.np.is_multiple_of(tn));
        let (gx, gy) = (if cout <= 32 { cout.div_ceil(tn) } else { c.np / tn }, m.div_ceil(tm));
        let kt_count = c.kp / 16;
        let mut ksplit = if opts.ks > 0 {
            opts.ks.min(kt_count.max(1))
        } else if gx * gy >= 512 {
            1
        } else {
            512_u32.div_ceil(gx * gy).min(8).min(kt_count.max(1))
        };
        if let Some(v) = std::env::var("FH_KSPLIT").ok().and_then(|s| s.parse::<u32>().ok()) {
            ksplit = v.max(1).min(kt_count.max(1));
        }
        // partials are f32 [z][n][m]: keep them under 32 MB (big-M convs never need the split anyway)
        while ksplit > 1 && (ksplit * cout * m) as u64 * 4 > 32 << 20 {
            ksplit -= 1;
        }
        let ip = self.gemm_params(c, hout, wout, ksplit, if ksplit <= 1 { res_on } else { 0 });
        let ins = [&*c.x.buf, c.weights, c.bias, c.slope];
        if ksplit <= 1 {
            let s = self.compute_step_ex(ker, &ins, bytemuck::bytes_of(&ip), &out, &[&self.zero, resb], (gx, gy, 1));
            return (Tensor { buf: out, c: cout, h: hout, w: wout, c0: 0, scale: 1.0 }, vec![s]);
        }
        debug_assert!(m.is_multiple_of(4), "split-K reduce works on vec4 runs along m");
        let part = self.alloc_f32((ksplit * cout * m) as usize);
        self.note_scratch(&part);
        let s1 = self.compute_step_ex(ker, &ins, bytemuck::bytes_of(&ip), &out, &[&part, resb], (gx, gy, ksplit));
        let rp = SplitKParams { n: cout, m, splits: ksplit, residual: res_on };
        let s2 = self.compute_step_ex(
            "skreduce",
            &[&part, c.bias, c.slope],
            bytemuck::bytes_of(&rp),
            &out,
            &[resb],
            ((cout * m / 4).div_ceil(256), 1, 1),
        );
        (Tensor { buf: out, c: cout, h: hout, w: wout, c0: 0, scale: 1.0 }, vec![s1, s2])
    }
    /// Pack weights tap-major for the direct kernels: kcol = tap * cinp + ci with cinp = ceil16(cin), so a 16-wide K
    /// block never straddles a (ky, kx) tap; N padded to the 64-channel tile. Returns the buffer, kp, np and cinp.
    pub fn buildb_tap(&self, wf: &[f32], cin: u32, cout: u32) -> (Rc<wgpu::Buffer>, u32, u32, u32) {
        let cinp = cin.div_ceil(16) * 16;
        let kp = 9 * cinp;
        let np = cout.div_ceil(64) * 64;
        let mut b = vec![0f32; (kp * np) as usize];
        for co in 0..cout as usize {
            for ci in 0..cin as usize {
                for tap in 0..9usize {
                    b[(tap * cinp as usize + ci) * np as usize + co] = wf[(co * cin as usize + ci) * 9 + tap];
                }
            }
        }
        (self.buf(&b), kp, np, cinp)
    }
    /// The 64-pixel cooperative-matrix kernel of the microbench (`fused4`). `mode` is its ablation mask (1 skip gather,
    /// 2 skip B staging, 4 skip MMA; 0 = the real kernel), carried in the `ksplit` slot of the parameters.
    pub fn conv_tc4(&self, c: &ConvArgs, mode: u32) -> OpOutput {
        let (hout, wout) = out_size(c.x, c.stride);
        let mp = (hout * wout).div_ceil(64) * 64;
        let out = self.alloc((c.cout * hout * wout) as usize);
        let ip = self.gemm_params(c, hout, wout, mode, 0);
        let s = self.compute_step(
            "fused4",
            &[&c.x.buf, c.weights, &self.zero, c.bias, c.slope],
            bytemuck::bytes_of(&ip),
            &out,
            (c.np / 64, mp / 64, 1),
        );
        (Tensor { buf: out, c: c.cout, h: hout, w: wout, c0: 0, scale: 1.0 }, vec![s])
    }
    // three x-shifted, y-padded copies of x ([3][cinp][h+2][w]) for the direct conv kernel; the kernel writes the
    // padding too (arena buffers are not zeroed)
    pub fn xshift3(&self, x: &Tensor, cinp: u32) -> (Rc<wgpu::Buffer>, Step) {
        let out = self.alloc((3 * cinp * (x.h + 2) * x.w) as usize);
        self.note_scratch(&out);
        let sp = ShiftParams { cin: x.c, cinp, h: x.h, w: x.w };
        debug_assert!(x.w.is_multiple_of(8)); // one thread per 8 consecutive x (vec4<u32> store)
        let s = self.compute_step(
            "xshift",
            &[&x.buf],
            bytemuck::bytes_of(&sp),
            &out,
            ((x.w / 8).div_ceil(64), x.h + 2, 3 * cinp),
        );
        (out, s)
    }
    /// Stride-1 conv with the A tile loaded straight from the shifted input (no gather); needs x.w % 16 == 0 and the
    /// tap-major weights of `buildb_tap`.
    pub fn conv_direct(&self, c: &ConvArgs) -> OpOutput {
        let x = c.x;
        debug_assert!(x.w.is_multiple_of(16) && c.stride == 1);
        let (h, w) = (x.h, x.w);
        let m = h * w;
        let (sh, s0) = self.xshift3(x, c.cinp);
        let out = self.alloc((c.cout * h * w) as usize);
        let ip = self.gemm_params(c, h, w, 0, 0);
        // 32-pixel register-blocked variant when a tile cannot cross a row; FH_DIRECT16 forces the 16-pixel kernel (A/B)
        let wide = w % 16 == 0 && m % 32 == 0 && std::env::var_os("FH_DIRECT16").is_none();
        let (ker, gy) = if wide { ("fdirect32", m / 32) } else { ("fdirect", m / 16) };
        let s1 = self.compute_step(
            ker,
            &[&sh, c.weights, &self.zero, c.bias, c.slope],
            bytemuck::bytes_of(&ip),
            &out,
            (c.np / 64, gy, 1),
        );
        (Tensor { buf: out, c: c.cout, h, w, c0: 0, scale: 1.0 }, vec![s0, s1])
    }
    /// 64-pixel register-blocked direct kernel (16 accumulators); a partial last tile is allowed (m % 64 != 0).
    pub fn conv_direct64(&self, c: &ConvArgs) -> OpOutput {
        let x = c.x;
        debug_assert!(x.w.is_multiple_of(16) && c.stride == 1);
        let (h, w) = (x.h, x.w);
        let m = h * w;
        let (sh, s0) = self.xshift3(x, c.cinp);
        let out = self.alloc((c.cout * h * w) as usize);
        let ip = self.gemm_params(c, h, w, 0, 0);
        let s1 = self.compute_step(
            "fdirect64",
            &[&sh, c.weights, &self.zero, c.bias, c.slope],
            bytemuck::bytes_of(&ip),
            &out,
            (c.np / 64, m.div_ceil(64), 1),
        );
        (Tensor { buf: out, c: c.cout, h, w, c0: 0, scale: 1.0 }, vec![s0, s1])
    }
}
