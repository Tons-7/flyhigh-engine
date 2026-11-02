enable f16;
// GMHS selection on the E8 flow grid: a cell adopts the affine hypothesis where it explains the frames better than
// the model flow (cost margin eps), differs from it by a real motion jump (dmin, network px) and the texture is not
// flat (tex) — the signature of a periodic structure matched one period off. The E8 flow is rewritten in place; the
// mid stage (mul_t, upsample, refine, render) then produces every in-between frame from the corrected motion.
struct Par { h8: u32, w8: u32, hc: u32, wc: u32, eps: f32, dmin: f32, tex: f32, _a: u32 };
@group(0) @binding(0) var<storage, read> maps: array<f32>;
@group(0) @binding(1) var<storage, read> coef: array<f32>;
@group(0) @binding(2) var<storage, read_write> flow: array<f16>;
@group(0) @binding(3) var<uniform> pp: Par;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x; let i = gid.y;
    if (j >= pp.w8 || i >= pp.h8) { return; }
    let x = u32(clamp(i32(floor((f32(j) + 0.5) * f32(pp.wc) / f32(pp.w8))), 0, i32(pp.wc) - 1));
    let y = u32(clamp(i32(floor((f32(i) + 0.5) * f32(pp.hc) / f32(pp.h8))), 0, i32(pp.hc) - 1));
    let n = pp.hc * pp.wc; let idx = y * pp.wc + x;
    let cm = maps[idx]; let ca = maps[n + idx]; let dv = maps[2u * n + idx]; let lm = maps[3u * n + idx]; let l2 = maps[4u * n + idx];
    let sd = sqrt(max(l2 - lm * lm, 0.0));
    if (ca + pp.eps < cm && dv > pp.dmin && sd > pp.tex) {
        let xn = -1.0 + 2.0 * f32(j) / f32(max(pp.w8 - 1u, 1u)); let yn = -1.0 + 2.0 * f32(i) / f32(max(pp.h8 - 1u, 1u));
        let n8 = pp.h8 * pp.w8; let o = i * pp.w8 + j;
        for (var c: u32 = 0u; c < 4u; c = c + 1u) {
            flow[c * n8 + o] = f16(coef[c * 4u] + coef[c * 4u + 1u] * xn + coef[c * 4u + 2u] * yn);
        }
    }
}
