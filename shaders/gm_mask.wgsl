// Cadence element mask steps on the GMHS cost grid (f32 maps). `mode` 0: mask = (maps[ch] > thr) ? 1 : 0 (reads the
// displacement-difference channel of the maps); `mode` 1: mask = (src > thr) ? 1 : 0 (re-threshold after a box
// dilation). The box filters in between are gm_box_h / gm_box_v with c = 1.
struct Par { n: u32, ch: u32, mode: u32, _a: u32, thr: f32, _b: u32, _c: u32, _d: u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> pp: Par;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pp.n) { return; }
    var v: f32 = 0.0;
    if (pp.mode == 0u) { v = src[pp.ch * pp.n + i]; } else { v = src[i]; }
    dst[i] = select(0.0, 1.0, v > pp.thr);
}
