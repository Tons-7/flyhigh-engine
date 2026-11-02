//! Global-motion hypothesis selection (GMHS): an optional stage at the end of the pair plan that repairs the network's
//! flow on periodic structures. The network matches at 1/8-1/16 of its input, where a fence, a railing or a window
//! grid under a pan aliases: the cost volume cannot tell the true motion from motion +- one period, and the network
//! settles on the wrong one inside a blob while its surroundings are right. The stage fits a robust affine motion to
//! the E8 flow (IRLS, 4 iterations), scores the model flow and the affine hypothesis photometrically on luma at the
//! network resolution (box-filtered mismatch |A[p+F0] - B[p+F1]| on a grid `DS` x coarser), and rewrites the E8 flow in
//! place with the affine hypothesis where it explains the frames better, differs from the model by a real jump
//! (`dmin`, level 1) and the texture is not flat. Every in-between frame of the pair then comes from the corrected
//! motion (mul_t / upsample / refine / render are untouched). The level is a per-session option (`NetOpts::repair`,
//! default 1). Measured 2026-10-26 on anime at 1080p (rung 480): the window-facade scroll goes 34.2 -> 40.1 dB;
//! CS2 / ATD / natural sets unchanged at level 1; +0.2 ms per pair.
//!
//! The same maps feed cadence smoothing (`api::Mode::Video`): the displacement-difference channel is the motion of
//! an element against the global motion, which classifies a pair as a jump / hold of an element animated on 2s and
//! gives the mask for the re-timed composite (`pair_stats`, `composite_frame`).
use crate::{clone_step, workgroups_2d, Gpu, SessionExt, Step, Tensor};
use bytemuck::{Pod, Zeroable};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FitP {
    pub h: u32,
    pub w: u32,
    pub nwg: u32,
    pub use_w: u32,
    pub delta: f32,
    pub _a: u32,
    pub _b: u32,
    pub _c: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SolveP {
    pub nwg: u32,
    pub _a: u32,
    pub _b: u32,
    pub _c: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CostP {
    pub h8: u32,
    pub w8: u32,
    pub h: u32,
    pub w: u32,
    pub hc: u32,
    pub wc: u32,
    pub ds: f32,
    pub _a: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BoxP {
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub r: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SelP {
    pub h8: u32,
    pub w8: u32,
    pub hc: u32,
    pub wc: u32,
    pub eps: f32,
    pub dmin: f32,
    pub tex: f32,
    pub _a: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct StatP {
    pub n: u32,
    pub ch: u32,
    pub nwg: u32,
    pub _a: u32,
    pub thr: f32,
    pub cthr: f32,
    pub _c: u32,
    pub _d: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MaskP {
    pub n: u32,
    pub ch: u32,
    pub mode: u32,
    pub _a: u32,
    pub thr: f32,
    pub _b: u32,
    pub _c: u32,
    pub _d: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CompP {
    pub hp: u32,
    pub wp: u32,
    pub hc: u32,
    pub wc: u32,
    pub sx: f32,
    pub sy: f32,
    pub _a: u32,
    pub _b: u32,
}

/// cost-map grid: `DS` network px per cell (the decision is smoothed over ~60 px anyway; 2 keeps the stage ~0.2 ms)
pub const DS: u32 = 2;

// Levels: 0 = off, 1 = safe (a cell switches only on a real motion jump, `dmin`; natural video untouched),
// 2 = aggressive (no jump gate: ties on repeating texture go to the global motion; +0.4 dB more on the anime clip, but
// natural video loses ~0.8 dB, so drawn content only).

pub const PIPES: [(&str, &str, &str); 9] = [
    ("gm_fit_reduce", include_str!("../../../shaders/gm_fit_reduce.wgsl"), "main"),
    ("gm_fit_solve", include_str!("../../../shaders/gm_fit_solve.wgsl"), "main"),
    ("gm_cost", include_str!("../../../shaders/gm_cost.wgsl"), "main"),
    ("gm_box_h", include_str!("../../../shaders/gm_box.wgsl"), "box_h"),
    ("gm_box_v", include_str!("../../../shaders/gm_box.wgsl"), "box_v"),
    ("gm_select", include_str!("../../../shaders/gm_select.wgsl"), "main"),
    ("gm_stats", include_str!("../../../shaders/gm_stats.wgsl"), "main"),
    ("gm_mask", include_str!("../../../shaders/gm_mask.wgsl"), "main"),
    ("gm_composite", include_str!("../../../shaders/gm_composite.wgsl"), "main"),
];

/// The cost maps kept for cadence smoothing: 5 x hc x wc f32 (`[0]` model cost, `[1]` affine cost, `[2]` displacement
/// difference = element motion against the global motion in network px, `[3]` luma, `[4]` luma^2), all box-filtered.
/// Carried in the graph env as a `Tensor` named "gm_maps" (c = 5, h = hc, w = wc, scale = `s` = network px per
/// half-frame px); the network size is env["A"].
pub fn maps_of(sess: &SessionExt) -> Option<(Tensor, u32, u32)> {
    let m = sess.env.get("gm_maps")?.clone();
    let a = sess.env.get("A")?;
    Some((m, a.h, a.w))
}

/// Steps that rewrite `flow` (the plain E8 flow of the pair stage, `[4][h8][w8]`, network px) in place. `a`, `b`: the
/// network-resolution inputs; `full_w`: the padded full-frame width (the prototype's window sizes are defined at half
/// of the full frame, so they scale with network px per half-frame px). `level`: 1 or 2 (see above). Returns the maps
/// when `keep` (cadence smoothing reads them after the pair).
pub fn gm_steps(
    gpu: &Gpu,
    a: &Tensor,
    b: &Tensor,
    flow: &Tensor,
    full_w: u32,
    level: u32,
    keep: bool,
) -> (Vec<Step>, Option<Tensor>) {
    assert!(flow.is_plain() && flow.c == 4, "gmhs: the E8 flow must be a plain 4-channel tensor");
    let (h8, w8, h, w) = (flow.h, flow.w, a.h, a.w);
    let (hc, wc) = (h.div_ceil(DS), w.div_ceil(DS));
    let s = w as f32 / (full_w as f32 / 2.0);
    let r_blk = ((20.0 * s / DS as f32).round() as u32).max(1);
    let r_dwin = ((32.0 * s / DS as f32).round() as u32).max(1);
    let (dmin, delta, nwg) = (if level >= 2 { 0.0 } else { 8.0 * s }, 1.5 * s, 32u32);
    let coef = gpu.alloc_f32(16);
    let part = gpu.alloc_f32((nwg * 36) as usize);
    // 6 channels: the 5 decision maps (box-filtered below) + the plain frame change (channel 5, unfiltered)
    let maps = gpu.alloc_f32((6 * hc * wc) as usize);
    let tmp = gpu.alloc_f32((5 * hc * wc) as usize);
    for b in [&coef, &part, &tmp] {
        gpu.note_scratch(b);
    }
    if !keep {
        gpu.note_scratch(&maps);
    }
    let mut st = Vec::new();
    for it in 0..4u32 {
        let p = FitP { h: h8, w: w8, nwg, use_w: (it > 0) as u32, delta, _a: 0, _b: 0, _c: 0 };
        st.push(gpu.compute_step("gm_fit_reduce", &[&flow.buf, &coef], bytemuck::bytes_of(&p), &part, (nwg, 1, 1)));
        st.push(gpu.compute_step(
            "gm_fit_solve",
            &[&part],
            bytemuck::bytes_of(&SolveP { nwg, _a: 0, _b: 0, _c: 0 }),
            &coef,
            (1, 1, 1),
        ));
    }
    let cp = CostP { h8, w8, h, w, hc, wc, ds: DS as f32, _a: 0 };
    st.push(gpu.compute_step(
        "gm_cost",
        &[&a.buf, &b.buf, &flow.buf, &coef],
        bytemuck::bytes_of(&cp),
        &maps,
        workgroups_2d(wc, hc, 1),
    ));
    // texture window (all 5 maps), then the decision window (costs + displacement difference only)
    for (r, c) in [(r_blk, 5u32), (r_dwin, 3u32)] {
        let p = BoxP { c, h: hc, w: wc, r };
        st.push(gpu.compute_step("gm_box_h", &[&maps], bytemuck::bytes_of(&p), &tmp, workgroups_2d(wc, hc, c)));
        st.push(gpu.compute_step("gm_box_v", &[&tmp], bytemuck::bytes_of(&p), &maps, workgroups_2d(wc, hc, c)));
    }
    let p = SelP { h8, w8, hc, wc, eps: 0.0, dmin, tex: 0.015, _a: 0 };
    st.push(gpu.compute_step(
        "gm_select",
        &[&maps, &coef],
        bytemuck::bytes_of(&p),
        &flow.buf,
        workgroups_2d(w8, h8, 1),
    ));
    gpu.release_scratch();
    let kept = if keep { Some(Tensor { buf: maps, c: 6, h: hc, w: wc, c0: 0, scale: s }) } else { None };
    (st, kept)
}

// ---------------------------------------------------------------- cadence smoothing (api::Mode::Video) ----

/// Jump / hold classification of a pair: `area` = fraction of the frame whose motion differs from the global affine
/// motion by more than the jump threshold, `residual` = mean of that difference (network px). A pair is a "jump" of
/// an element animated on 2s when area > 0.5 % and residual > 4 * s px (= 8 half-frame px, the prototype's rule).
/// `changed` = fraction of cells whose plain luma change exceeds CHANGE_THR: exactly 0 for a held drawing on a still
/// shot, > 0 for any new drawing however slow (the still-shot cadence detector in `gen_tex`).
#[derive(Clone, Copy, Debug, Default)]
pub struct PairStats {
    pub area: f32,
    pub residual: f32,
    pub jump: bool,
    /// most of the frame moves against the global motion or the residual is huge: a scene cut / history break, never re-timed across
    pub cut: bool,
    pub changed: f32,
    /// mean model mismatch (0..1), diagnostics
    pub cost: f32,
}
/// Plain luma change (0..1) above which a cost cell counts as changed (5/255: above codec noise on re-encoded holds).
pub const CHANGE_THR: f32 = 0.02;

/// Buffers for the re-timed composite, sized for one session's frame size, and the cadence steps built from them
/// (cached: their uniforms never change, and building one allocates GPU objects).
pub struct CadenceScratch {
    pub bg_copy: Rc<wgpu::Buffer>,
    pub mask_a: Rc<wgpu::Buffer>,
    pub mask_b: Rc<wgpu::Buffer>,
    pub out: Tensor,
    pub part: Rc<wgpu::Buffer>,
    steps: RefCell<HashMap<(u8, usize, usize), Vec<Step>>>,
}
pub const STAT_WG: u32 = 32;
const K_STATS: u8 = 0;
const K_MASK: u8 = 1;
const K_COMP: u8 = 2;
const K_OUT: u8 = 3;
fn sid(s: &SessionExt) -> usize {
    s as *const SessionExt as usize
} // sessions live in GenTex::sess and never move
impl CadenceScratch {
    pub fn new(gpu: &Gpu, sess: &SessionExt) -> Option<CadenceScratch> {
        let (m, _, _) = maps_of(sess)?;
        let r = &sess.render;
        let n = (r.c * r.h * r.w) as usize;
        let out = Tensor { buf: gpu.alloc(n), c: r.c, h: r.h, w: r.w, c0: 0, scale: 1.0 };
        Some(CadenceScratch {
            bg_copy: gpu.alloc(n),
            mask_a: gpu.alloc_f32((m.h * m.w) as usize),
            mask_b: gpu.alloc_f32((m.h * m.w) as usize),
            out,
            part: gpu.alloc_f32((STAT_WG * 4) as usize),
            steps: RefCell::new(HashMap::new()),
        })
    }
    fn cached(&self, key: (u8, usize, usize), build: impl FnOnce() -> Vec<Step>) -> Vec<Step> {
        let mut c = self.steps.borrow_mut();
        c.entry(key).or_insert_with(build).iter().map(clone_step).collect()
    }
}

/// A submitted statistics reduction whose 64-float readback is still in flight (`finish` awaits the mapping; on
/// native poll the device first, in the browser the mapAsync promise resolves it).
pub struct StatsRun {
    stage: wgpu::Buffer,
    fut: crate::MapWait,
    s: f32,
    n: u32,
}
impl StatsRun {
    pub async fn finish(self) -> anyhow::Result<PairStats> {
        self.fut.await.map_err(|e| anyhow::anyhow!("map_async: {e:?}"))?;
        let (cnt, sum, cost, chg) = {
            let d = self.stage.slice(..).get_mapped_range().map_err(|e| anyhow::anyhow!("mapped range: {e:?}"))?;
            let f: &[f32] = bytemuck::cast_slice(&d);
            (
                f.iter().step_by(4).sum::<f32>(),
                f.iter().skip(1).step_by(4).sum::<f32>(),
                f.iter().skip(2).step_by(4).sum::<f32>(),
                f.iter().skip(3).step_by(4).sum::<f32>(),
            )
        };
        self.stage.unmap();
        let area = cnt / self.n as f32;
        let residual = if cnt > 0.0 { sum / cnt / self.s * 2.0 } else { 0.0 }; // residual reported in full-frame px
        let mean_cost = cost / self.n as f32; // mean box-filtered luma mismatch under the model flow (0..1)
        let changed = chg / self.n as f32;
        // a scene cut: the model cannot match the frames, or (almost) everything changed and moves against any global
        // motion. A large character walking through a still shot changes a third of the frame and moves fast, but the
        // model still matches it: not a cut. Never re-time across a cut.
        let cut = mean_cost > 0.10 || (changed > 0.6 && area > 0.45);
        Ok(PairStats { area, residual, jump: !cut && area > 0.005 && residual > 8.0, cut, changed, cost: mean_cost })
    }
}
/// Submit the jump / hold statistics reduction of the pair last run in `sess` and start its readback.
pub fn pair_stats_begin(gpu: &Gpu, sess: &SessionExt, sc: &CadenceScratch) -> anyhow::Result<StatsRun> {
    let (m, _, _) = maps_of(sess)
        .ok_or_else(|| anyhow::anyhow!("gmhs: session has no kept maps (cadence needs repair > 0 and keep_maps)"))?;
    let s = m.scale;
    let n = m.h * m.w;
    let st = sc.cached((K_STATS, sid(sess), 0), || {
        let p = StatP { n, ch: 2, nwg: STAT_WG, _a: 0, thr: 4.0 * s, cthr: CHANGE_THR, _c: 0, _d: 0 };
        vec![gpu.compute_step("gm_stats", &[&m.buf], bytemuck::bytes_of(&p), &sc.part, (STAT_WG, 1, 1))]
    });
    gpu.replay(&st);
    let stage = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fh_gm_stats"),
        size: (STAT_WG * 4 * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(&sc.part, 0, &stage, 0, (STAT_WG * 4 * 4) as u64);
    gpu.queue.submit(Some(enc.finish()));
    let fut = crate::map_read(&stage);
    Ok(StatsRun { stage, fut, s, n })
}
/// Native: jump / hold statistics of the pair last run in `sess` (blocking readback of 64 floats).
#[cfg(not(target_arch = "wasm32"))]
pub fn pair_stats(gpu: &Gpu, sess: &SessionExt, sc: &CadenceScratch) -> anyhow::Result<PairStats> {
    let run = pair_stats_begin(gpu, sess, sc)?;
    gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
    pollster::block_on(run.finish())
}

/// One re-timed output frame: background = the `bg` session's mid at its phase, element = the `elem` session's mid at
/// its phase (each given as (session, phase)), composited with the element mask of `mask_from` (dilated jump region),
/// written to output slot `slot` (view `out_view`, rgba8unorm storage).
pub fn composite_frame(
    gpu: &Gpu,
    sc: &CadenceScratch,
    bg: (&SessionExt, f32),
    elem: (&SessionExt, f32),
    mask_from: &SessionExt,
    slot: usize,
    out_view: &wgpu::TextureView,
) -> anyhow::Result<()> {
    let ((bg, t), (elem, te)) = (bg, elem);
    let (m, ih, iw) = maps_of(mask_from).ok_or_else(|| anyhow::anyhow!("gmhs: session has no kept maps"))?;
    let s = m.scale;
    let n = m.h * m.w;
    let nbytes = (bg.render.c * bg.render.h * bg.render.w) as u64 * gpu.esize() as u64;
    // background mid first and copied aside: `bg` and `elem` may be the same session (its t uniform is rewritten in place)
    bg.set_t(gpu, t);
    let mut st: Vec<Step> = bg.plan_mid.iter().map(clone_step).collect();
    st.push(Step::Copy { src: bg.render.buf.clone(), so: 0, dst: sc.bg_copy.clone(), dof: 0, size: nbytes });
    gpu.replay(&st);
    elem.set_t(gpu, te);
    let mut st: Vec<Step> = elem.plan_mid.iter().map(clone_step).collect();
    // element mask: threshold the displacement difference (6 full px), dilate (box 15 full px, > 0.3), soften (box 30 full px)
    st.extend(sc.cached((K_MASK, sid(mask_from), 0), || {
        let spx = s / 2.0 / DS as f32; // cost cells per full-frame px
        let (r1, r2) = (((15.0 * spx).round() as u32).max(1), ((30.0 * spx).round() as u32).max(1));
        let mp = |mode: u32, thr: f32| MaskP { n, ch: 2, mode, _a: 0, thr, _b: 0, _c: 0, _d: 0 };
        let bp = |r: u32| BoxP { c: 1, h: m.h, w: m.w, r };
        let g1 = (n.div_ceil(256), 1, 1);
        vec![
            gpu.compute_step("gm_mask", &[&m.buf], bytemuck::bytes_of(&mp(0, 3.0 * s)), &sc.mask_a, g1),
            gpu.compute_step(
                "gm_box_h",
                &[&sc.mask_a],
                bytemuck::bytes_of(&bp(r1)),
                &sc.mask_b,
                workgroups_2d(m.w, m.h, 1),
            ),
            gpu.compute_step(
                "gm_box_v",
                &[&sc.mask_b],
                bytemuck::bytes_of(&bp(r1)),
                &sc.mask_a,
                workgroups_2d(m.w, m.h, 1),
            ),
            gpu.compute_step("gm_mask", &[&sc.mask_a], bytemuck::bytes_of(&mp(1, 0.3)), &sc.mask_b, g1),
            gpu.compute_step(
                "gm_box_h",
                &[&sc.mask_b],
                bytemuck::bytes_of(&bp(r2)),
                &sc.mask_a,
                workgroups_2d(m.w, m.h, 1),
            ),
            gpu.compute_step(
                "gm_box_v",
                &[&sc.mask_a],
                bytemuck::bytes_of(&bp(r2)),
                &sc.mask_b,
                workgroups_2d(m.w, m.h, 1),
            ),
        ]
    }));
    st.extend(sc.cached((K_COMP, sid(elem), 0), || {
        let cp = CompP {
            hp: bg.render.h,
            wp: bg.render.w,
            hc: m.h,
            wc: m.w,
            sx: iw as f32 / (bg.render.w as f32 * DS as f32),
            sy: ih as f32 / (bg.render.h as f32 * DS as f32),
            _a: 0,
            _b: 0,
        };
        vec![gpu.compute_step(
            "gm_composite",
            &[&sc.bg_copy, &elem.render.buf, &sc.mask_b],
            bytemuck::bytes_of(&cp),
            &sc.out.buf,
            workgroups_2d(bg.render.w, bg.render.h, 1),
        )]
    }));
    st.extend(sc.cached((K_OUT, slot, 0), || vec![gpu.tex_out_step(&sc.out, out_view, bg.w, bg.h, false)]));
    gpu.replay(&st);
    Ok(())
}
