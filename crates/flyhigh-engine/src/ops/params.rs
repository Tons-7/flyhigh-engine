//! Uniform parameter structs of the kernels (`#[repr(C)]`, one per shader).

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ConvParams {
    pub cin: u32,
    pub cout: u32,
    pub h: u32,
    pub w: u32,
    pub hout: u32,
    pub wout: u32,
    pub stride: u32,
    pub pad: u32,
    pub act: u32,
    pub _a: u32,
    pub _b: u32,
    pub _c: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct InterpParams {
    pub c: u32,
    pub ih: u32,
    pub iw: u32,
    pub oh: u32,
    pub ow: u32,
    pub aa: u32,
    pub c0: u32,
    pub scale: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PoolParams {
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub oh: u32,
    pub ow: u32,
    pub k: u32,
    pub _a: u32,
    pub _b: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct WarpParams {
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub f0: u32,
    pub i0: u32,
    pub fs: f32,
    pub is: f32,
    pub _a: u32,
} // flow / image views
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BinaryParams {
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub cb: u32,
    pub op: u32,
    pub a0: u32,
    pub b0: u32,
    pub _a: u32,
    pub sa: f32,
    pub sb: f32,
    pub _b: u32,
    pub _c: u32,
} // 12 u32
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct UnaryParams {
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub op: u32,
    pub a: f32,
    pub b: f32,
    pub c0: u32,
    pub scale: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LumaParams {
    pub h: u32,
    pub w: u32,
    pub c0: u32,
    pub scale: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SadParams {
    pub h: u32,
    pub w: u32,
    pub radius: u32,
    pub span: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Shape3Params {
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub _a: u32,
}
/// `splitk_reduce.wgsl`: sum the split-K partials `[splits][n][m]`, bias, PReLU, optional residual.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SplitKParams {
    pub n: u32,
    pub m: u32,
    pub splits: u32,
    pub residual: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct StaticParams {
    pub w: u32,
    pub h: u32,
    pub wp: u32,
    pub hp: u32,
    pub thr: f32,
    pub min_age: f32,
    pub use_mask: u32,
    pub _b: u32,
}
impl StaticParams {
    /// Automatic protection (unchanged pixels age up to the protecting age) and / or the host's mask (age 255 at
    /// once). Without the automatic part the threshold never matches, and a protecting age of 2 keeps age 0 clear
    /// of the two-pair ramp of the apply pass. `w`, `h`: the plane the kernel covers.
    pub fn new(net: &crate::NetOpts, w: u32, h: u32, wp: u32, hp: u32) -> StaticParams {
        StaticParams {
            w,
            h,
            wp,
            hp,
            thr: if net.keep_static { (net.static_tolerance.min(255) as f32 + 0.5) / 255.0 } else { 0.0 },
            min_age: net.protecting_age() as f32,
            use_mask: net.protection_mask as u32,
            _b: 0,
        }
    }
}
/// near_d.wgsl / near_apply.wgsl
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct NearParams {
    pub h: u32,
    pub w: u32,
    pub hq: u32,
    pub wq: u32,
    pub f0: u32,
    pub fsx: f32,
    pub fsy: f32,
    pub _a: u32,
}
/// near_k.wgsl
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct NearKParams {
    pub hq: u32,
    pub wq: u32,
    pub r: u32,
    pub _a: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RenderParams {
    pub h: u32,
    pub w: u32,
    pub hq: u32,
    pub wq: u32,
    pub f0: u32,
    pub v0: u32,
    pub fsx: f32,
    pub fsy: f32,
    pub vs: f32,
    /// bicubic source sampling (else bilinear)
    pub cubic: u32,
    /// static protection: pixels whose age reaches this are overlays, not world (0 = off)
    pub min_age: u32,
    /// mode word (render.wgsl): bit 0 decisive blend, bit 1 still test
    pub decisive: u32,
} // flow_tq / vis_tq views; per-axis flow scale (internal-resolution mode)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ImplicitGemmParams {
    pub cin: u32,
    pub h: u32,
    pub w: u32,
    pub hout: u32,
    pub wout: u32,
    pub stride: u32,
    pub pad: u32,
    pub k: u32,
    pub kp: u32,
    pub n: u32,
    pub np: u32,
    pub m: u32,
    pub cinp: u32,
    /// split-K count of the tiled kernels (dispatch z = split); the microbench-only fused4 reads this slot as its
    /// ablation mask
    pub ksplit: u32,
    /// fused residual add after the activation (tiled kernels, split-K off)
    pub residual: u32,
    pub _pad0: u32,
} // 16 u32 = 64 B (uniform struct size)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct EpilogueParams {
    pub cout: u32,
    pub hout: u32,
    pub wout: u32,
    pub npad: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ShiftParams {
    pub cin: u32,
    pub cinp: u32,
    pub h: u32,
    pub w: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ChannelCopyParams {
    pub hw2: u32,
    pub ctot: u32,
    pub e: [u32; 7],
    pub o: [u32; 7],
    pub dst0: u32,
    pub s: [f32; 7],
    pub _a: u32,
    pub _b: u32,
    pub _c: u32,
    pub _d: u32,
} // 28 u32: per-input channel offset + scale (views)

// Session: one prepared pipeline for a fixed (W, H, K, t) with RGBA8 frames in and out, entirely on the GPU
// (rgba_in -> graph -> rgba_out). This is what the browser shell and a desktop overlay drive per frame: upload two
// RGBA8 frames, submit the static plan, read back (or later: blit) the generated frame. t is baked into the plan
// (uniforms are built once), so a 3x mode prepares two sessions (t = 1/3, 2/3).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RgbaParams {
    pub w: u32,
    pub h: u32,
    pub wp: u32,
    pub hp: u32,
}
