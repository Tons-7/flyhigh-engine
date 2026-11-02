//! Per-shape kernel choice at build time (`BuildCtx::conv_pick`) and the asynchronous discovery autotune.

use crate::model::graph::count_uses;
use crate::{
    build, cand_list, cand_name, Blob, BuildCtx, ConvArgs, Gpu, OpOutput, Tensor, TiledOpts, TuneTable, CAND_DIRECT,
    CAND_DIRECT64, CAND_TC, CAND_TILED, CAND_TILED16, CAND_TILED8, CAND_TILED88, CAND_TILED8P, CAND_TILEDM32,
    CAND_TILEDN8,
};
use serde_json::Value;
use std::collections::HashMap;

/// A conv of the graph as the builder sees it: the input, the weight name (the packed B matrix is cached per layout
/// in the build context), bias and slope buffers, the channel counts and the stride.
#[derive(Clone, Copy)]
pub struct ConvOp<'a> {
    pub x: &'a Tensor,
    pub weight: &'a str,
    pub bias: &'a wgpu::Buffer,
    pub slope: &'a wgpu::Buffer,
    pub cin: u32,
    pub cout: u32,
    pub stride: u32,
}

impl<'a> BuildCtx<'a> {
    /// The steps of one conv with a given kernel candidate (weights cached per layout). `res`: an optional fused
    /// residual add after the activation; the FMA kernels fuse it in their epilogue (or the split-K reduce), the
    /// cooperative-matrix kernels get a separate binary add appended, and the autotuner times both forms.
    pub fn conv_cand(&mut self, op: &ConvOp, cand: u8, res: Option<&Tensor>) -> OpOutput {
        let ConvOp { x, weight: wname, bias, slope, cin, cout, stride } = *op;
        let g = self.gpu;
        let ks = (cand >> 4) as u32;
        let kid = cand & 15;
        if kid == CAND_DIRECT || kid == CAND_DIRECT64 || kid == CAND_TC {
            let (y, mut steps) = match kid {
                CAND_DIRECT => {
                    let (bb, kp, np, cinp) = self.bbuf_tap(wname, cin, cout);
                    g.conv_direct(&ConvArgs { x, weights: &bb, bias, slope, cin, cout, kp, np, cinp, stride })
                }
                CAND_DIRECT64 => {
                    let (bb, kp, np, cinp) = self.bbuf_tap(wname, cin, cout);
                    g.conv_direct64(&ConvArgs { x, weights: &bb, bias, slope, cin, cout, kp, np, cinp, stride })
                }
                _ => {
                    let (bb, kp, np) = self.bbuf(wname, cin, cout);
                    g.conv_tc(&ConvArgs { x, weights: &bb, bias, slope, cin, cout, kp, np, cinp: 0, stride })
                }
            };
            if let Some(r) = res {
                let (z, s2) = g.binary(&y, r, 0);
                steps.extend(s2);
                return (z, steps);
            }
            return (y, steps);
        }
        let (bb, kp, np) = self.bbuf(wname, cin, cout);
        let (variant, tn) = match kid {
            CAND_TILED8 => (1, 32),
            CAND_TILED16 => (0, 16),
            CAND_TILEDN8 => (0, 8),
            CAND_TILED88 => (2, 32),
            CAND_TILEDM32 => (3, 32),
            CAND_TILED8P => (4, 32),
            _ => (0, 32),
        };
        let args = ConvArgs { x, weights: &bb, bias, slope, cin, cout, kp, np, cinp: 0, stride };
        g.conv_tiled_with(&args, TiledOpts { variant, ks, tn_small: tn }, res)
    }
    // Per-shape autotune on tensor-core devices: time every applicable candidate on the real buffers (min of 15,
    // GPU timestamps) once per (cin, cout, h, w, stride) and keep the fastest. The plan is static, so the cost is
    // paid once per resolution at build time. EngineOptions::autotune = false (CLI: FH_NO_AUTOTUNE=1) restores the fixed
    // routing (direct if applicable, else tc, else tiled).
    pub fn conv_pick(&mut self, op: &ConvOp, res: Option<&Tensor>) -> OpOutput {
        let ConvOp { x, weight: wname, slope, cin, cout, stride, .. } = *op;
        let g = self.gpu;
        let direct_ok =
            g.coop && stride == 1 && x.w.is_multiple_of(16) && cout > 32 && std::env::var_os("FH_NO_DIRECT").is_none();
        let key = (cin, cout, x.h, x.w, stride);
        if let Some(list) = self.discover.as_mut() {
            // discovery build: record the shape, keep the fixed choice
            let (bname, sname) = (
                format!("{}bias", &wname[..wname.len() - 6]),
                if std::ptr::eq(slope, &*g.ones) {
                    None
                } else {
                    Some(format!("{}1.weight", &wname[..wname.len() - 8]))
                },
            );
            list.push(ConvShape {
                w0: wname.to_string(),
                bias: bname,
                slope: sname,
                cin,
                cout,
                h: x.h,
                w: x.w,
                stride,
            });
        }
        let cand = if let Some(&c) = self.tune.get(&key) {
            c
        } else {
            // the autotuner needs blocking GPU timing: native only (the browser pre-tunes asynchronously via `autotune`)
            let fixed = cfg!(target_arch = "wasm32") || self.discover.is_some() || !g.autotune || !g.timestamps;
            let c = if fixed {
                if direct_ok {
                    CAND_DIRECT
                } else if g.coop {
                    CAND_TC
                } else {
                    CAND_TILED
                }
            } else {
                let mut best = (f64::INFINITY, if g.coop { CAND_TC } else { CAND_TILED });
                let mut line = String::new();
                let (hout, wout) = ((x.h + 2 - 3) / stride + 1, (x.w + 2 - 3) / stride + 1);
                for cand in cand_list(g.coop, cout, hout, wout, direct_ok) {
                    let (_, steps) = self.conv_cand(op, cand, res);
                    #[cfg(not(target_arch = "wasm32"))]
                    let ms = g.gpu_time_ms_n(&steps, 15);
                    #[cfg(target_arch = "wasm32")]
                    let ms = {
                        let _ = &steps;
                        f64::NAN
                    };
                    line += &format!(" {} {:.3}", cand_name(cand), ms);
                    if ms < best.0 {
                        best = (ms, cand);
                    }
                }
                if std::env::var_os("FH_TUNE_VERBOSE").is_some() {
                    println!(
                        "  tune {:>3}->{:<3} @{}x{} s{}:{} -> {}",
                        cin,
                        cout,
                        x.w,
                        x.h,
                        stride,
                        line,
                        cand_name(best.1)
                    );
                }
                best.1
            };
            self.tune.insert(key, c);
            c
        };
        self.conv_cand(op, cand, res)
    }
}

// One conv shape met during a discovery build, with everything needed to time its kernel candidates later
#[derive(Clone)]
pub struct ConvShape {
    pub w0: String,
    pub bias: String,
    pub slope: Option<String>,
    pub cin: u32,
    pub cout: u32,
    pub h: u32,
    pub w: u32,
    pub stride: u32,
}

// Asynchronous autotune (browser: timing must not block): a discovery build records every conv shape of the plan for
// (w, h, K, t, internal size), then each shape's candidates are timed with pass-level timestamp queries and the
// best is kept. The result feeds `SessionTex::new` / `build_network` through `tune`.
/// What to tune: the frame size, fine iterations, phase and internal size of the plan whose conv shapes are timed,
/// and how many timed repetitions per candidate.
#[derive(Clone, Copy, Debug)]
pub struct TuneRequest {
    pub w: u32,
    pub h: u32,
    pub k: u32,
    pub t: f32,
    pub iw: u32,
    pub ih: u32,
    pub reps: usize,
}

pub async fn autotune(gpu: &Gpu, blob: &Blob, ops: &[Value], req: &TuneRequest) -> TuneTable {
    let TuneRequest { w, h, k, t, iw, ih, reps } = *req;
    let mut table: TuneTable = HashMap::new();
    if !gpu.timestamps {
        return table;
    }
    let (hp, wp) = (h.div_ceil(32) * 32, w.div_ceil(32) * 32);
    let a = Tensor { buf: gpu.alloc((3 * hp * wp) as usize), c: 3, h: hp, w: wp, c0: 0, scale: 1.0 };
    let b = Tensor { buf: gpu.alloc((3 * hp * wp) as usize), c: 3, h: hp, w: wp, c0: 0, scale: 1.0 };
    // discovery: build with fixed routing, collecting shapes
    let shapes: Vec<ConvShape> = {
        let (ihp, iwp) = (ih.div_ceil(32) * 32, iw.div_ceil(32) * 32);
        let mut env: HashMap<String, Tensor> = HashMap::new();
        let mut plan = Vec::new();
        if (iwp, ihp) != (wp, hp) {
            let (ai, sa) = gpu.interp(&a, ihp, iwp, true);
            let (bi, sb) = gpu.interp(&b, ihp, iwp, true);
            plan.extend(sa);
            plan.extend(sb);
            env.insert("A".into(), ai);
            env.insert("B".into(), bi);
        } else {
            env.insert("A".into(), a.clone());
            env.insert("B".into(), b.clone());
        }
        let mut cx = BuildCtx::new(gpu, blob, k, t, ihp, iwp);
        cx.discover = Some(Vec::new());
        count_uses(ops, &mut cx.uses);
        let rend = ops.iter().rposition(|o| o["op"] == "render_pix");
        let body = match ((iwp, ihp) != (wp, hp), rend) {
            (true, Some(i)) => &ops[..i],
            _ => ops,
        };
        build(body, &mut env, &mut plan, &mut cx);
        cx.discover.take().unwrap_or_default()
    };
    let mut cx = BuildCtx::new(gpu, blob, k, t, hp, wp);
    for s in shapes {
        let key = (s.cin, s.cout, s.h, s.w, s.stride);
        if table.contains_key(&key) {
            continue;
        }
        let x = Tensor { buf: gpu.alloc((s.cin * s.h * s.w) as usize), c: s.cin, h: s.h, w: s.w, c0: 0, scale: 1.0 };
        let bias = cx.wb(&s.bias);
        let slope = match &s.slope {
            Some(n) => cx.wb(n),
            None => gpu.ones.clone(),
        };
        let direct_ok = gpu.coop && s.stride == 1 && s.w % 16 == 0 && s.cout > 32;
        let (hout, wout) = ((s.h + 2 - 3) / s.stride + 1, (s.w + 2 - 3) / s.stride + 1);
        let mut best = (f64::INFINITY, if gpu.coop { CAND_TC } else { CAND_TILED });
        for cand in cand_list(gpu.coop, s.cout, hout, wout, direct_ok) {
            let op =
                ConvOp { x: &x, weight: &s.w0, bias: &bias, slope: &slope, cin: s.cin, cout: s.cout, stride: s.stride };
            let (_, steps) = cx.conv_cand(&op, cand, None);
            let ms = gpu.gpu_time_ms_async(&steps, reps).await;
            if ms < best.0 {
                best = (ms, cand);
            }
        }
        table.insert(key, best.1);
    }
    table
}
