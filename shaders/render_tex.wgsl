enable f16;
// Fast render (option render "fast", texture path): the in-between at full resolution straight from the two source
// textures into the output texture. The exact render reads A / B from three f16 planes per tap and writes a
// full-resolution tensor that tex_out converts afterwards; here a tap is one RGBA8 texel (all three channels), the
// result goes straight to the output, and `cubic` = 0 takes one hardware-filtered bilinear sample per source instead
// of 16 bicubic taps (cheaper, but softer than the bicubic render the model was trained with). Static protection (age buffer, use_static) is folded in: warp sources near protected pixels are
// not used, and protected pixels keep the sources' own value.
// flow_tq / vis_tq / R_t: the network's quarter-resolution maps over the padded plane (wp x hp); R_t is BGR.
// sa / sb / sr: linear-t reuse of maps computed at t = 0.5 (option linear_t): the flows to A and B scale by 2t and
// 2(1 - t), the blend leans towards the nearer frame, the residual fades out towards the source frames; 1 / 1 / 1 = as computed.
// decisive bit 0: where the two warped frames disagree, the one the network trusts more instead of their blend;
// bit 1: the still test (render.wgsl).
struct Par { w: u32, h: u32, wp: u32, hp: u32, hq: u32, wq: u32, f0: u32, v0: u32, fsx: f32, fsy: f32, vs: f32, min_age: f32, use_static: u32, sa: f32, sb: f32, sr: f32, cubic: u32, decisive: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var tex_a: texture_2d<f32>;
@group(0) @binding(1) var tex_b: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read> flow_tq: array<f16>;
@group(0) @binding(4) var<storage, read> vis_tq: array<f16>;
@group(0) @binding(5) var<storage, read> R_t: array<f16>;
@group(0) @binding(6) var<storage, read> age: array<f16>;
@group(0) @binding(7) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(8) var<uniform> pp: Par;

fn bil4(v00: f32, v01: f32, v10: f32, v11: f32, fx: f32, fy: f32) -> f32 {
    return (v00 * (1.0 - fx) + v01 * fx) * (1.0 - fy) + (v10 * (1.0 - fx) + v11 * fx) * fy;
}
fn q(buf_sel: u32, ch: u32, y0: u32, y1: u32, x0: u32, x1: u32, fx: f32, fy: f32) -> f32 {
    let w = pp.wq; let plane = pp.hq * pp.wq;
    if (buf_sel == 0u) {
        let b = (ch + pp.f0) * plane;
        return bil4(f32(flow_tq[b + y0 * w + x0]), f32(flow_tq[b + y0 * w + x1]), f32(flow_tq[b + y1 * w + x0]), f32(flow_tq[b + y1 * w + x1]), fx, fy);
    } else if (buf_sel == 1u) {
        let b = (ch + pp.v0) * plane;
        return bil4(f32(vis_tq[b + y0 * w + x0]), f32(vis_tq[b + y0 * w + x1]), f32(vis_tq[b + y1 * w + x0]), f32(vis_tq[b + y1 * w + x1]), fx, fy);
    }
    let b = ch * plane;
    return bil4(f32(R_t[b + y0 * w + x0]), f32(R_t[b + y0 * w + x1]), f32(R_t[b + y1 * w + x0]), f32(R_t[b + y1 * w + x1]), fx, fy);
}
// pixel position (x + 0.5, y + 0.5) in texels: a bilinear sample between the four neighbours, edges clamped
// (= grid_sample bilinear, align_corners=True, border padding, for a position given in pixel indices)
fn sample(t: texture_2d<f32>, gx: f32, gy: f32) -> vec3<f32> {
    if (pp.cubic != 0u) { return sample_cubic(t, gx, gy); }
    return textureSampleLevel(t, samp, vec2<f32>((gx + 0.5) / f32(pp.w), (gy + 0.5) / f32(pp.h)), 0.0).rgb;
}
// Keys cubic convolution, a = -0.75, per-tap border clamp (= grid_sample bicubic, align_corners=True, border padding)
fn cub1(x: f32) -> f32 { let a = -0.75; return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0; }
fn cub2(x: f32) -> f32 { let a = -0.75; return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a; }
fn cubw(t: f32) -> vec4<f32> { return vec4<f32>(cub2(t + 1.0), cub1(t), cub1(1.0 - t), cub2(2.0 - t)); }
fn tap(t: texture_2d<f32>, x: i32, y: i32) -> vec3<f32> {
    return textureLoad(t, vec2<i32>(clamp(x, 0, i32(pp.w) - 1), clamp(y, 0, i32(pp.h) - 1)), 0).rgb;
}
fn row(t: texture_2d<f32>, px: i32, y: i32, cx: vec4<f32>) -> vec3<f32> {
    return tap(t, px - 1, y) * cx.x + tap(t, px, y) * cx.y + tap(t, px + 1, y) * cx.z + tap(t, px + 2, y) * cx.w;
}
fn sample_cubic(t: texture_2d<f32>, gx: f32, gy: f32) -> vec3<f32> {
    let px = i32(floor(gx)); let py = i32(floor(gy));
    let cx = cubw(gx - f32(px)); let cy = cubw(gy - f32(py));   // unrolled rows: no dynamic vector index
    return row(t, px, py - 1, cx) * cy.x + row(t, px, py, cx) * cy.y + row(t, px, py + 1, cx) * cy.z + row(t, px, py + 2, cx) * cy.w;
}
fn on_static(gx: f32, gy: f32) -> bool {
    let cx = i32(round(gx)); let cy = i32(round(gy));
    for (var dy = -1; dy <= 1; dy++) { for (var dx = -1; dx <= 1; dx++) {
        let xx = u32(clamp(cx + dx, 0, i32(pp.w) - 1)); let yy = u32(clamp(cy + dy, 0, i32(pp.h) - 1));
        if (f32(age[yy * pp.wp + xx]) >= pp.min_age) { return true; }
    } }
    return false;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= pp.w || y >= pp.h) { return; }
    let iy = (f32(y) + 0.5) * f32(pp.hq) / f32(pp.hp) - 0.5; let ix = (f32(x) + 0.5) * f32(pp.wq) / f32(pp.wp) - 0.5;
    let qy0 = i32(floor(iy)); let qx0 = i32(floor(ix));
    let fy = iy - f32(qy0); let fx = ix - f32(qx0);
    let y0 = u32(clamp(qy0, 0, i32(pp.hq) - 1)); let y1 = u32(clamp(qy0 + 1, 0, i32(pp.hq) - 1));
    let x0 = u32(clamp(qx0, 0, i32(pp.wq) - 1)); let x1 = u32(clamp(qx0 + 1, 0, i32(pp.wq) - 1));
    let gxa = f32(x) + q(0u, 0u, y0, y1, x0, x1, fx, fy) * pp.fsx * pp.sa; let gya = f32(y) + q(0u, 1u, y0, y1, x0, x1, fx, fy) * pp.fsy * pp.sa;
    let gxb = f32(x) + q(0u, 2u, y0, y1, x0, x1, fx, fy) * pp.fsx * pp.sb; let gyb = f32(y) + q(0u, 3u, y0, y1, x0, x1, fx, fy) * pp.fsy * pp.sb;
    let v0 = q(1u, 0u, y0, y1, x0, x1, fx, fy) * pp.vs; let v1 = q(1u, 1u, y0, y1, x0, x1, fx, fy) * pp.vs; let v2 = q(1u, 2u, y0, y1, x0, x1, fx, fy) * pp.vs;
    let m = max(v0, max(v1, v2));
    let e0 = exp(v0 - m); let e1 = exp(v1 - m); let e2 = exp(v2 - m); let s = e0 + e1 + e2;
    var wa = (e0 + 0.5 * e2) / s * (2.0 - pp.sa); var wb = (e1 + 0.5 * e2) / s * (2.0 - pp.sb);
    let wn = max(wa + wb, 1e-6); wa = wa / wn; wb = wb / wn;
    var r_k = pp.sr; var raw = 0.0;
    let ca = sample(tex_a, gxa, gya); let cb = sample(tex_b, gxb, gyb);
    var stillw = 0.0;
    if ((pp.decisive & 2u) != 0u) {
        var es = 0.0;
        for (var dy = -1; dy <= 1; dy++) { for (var dx = -1; dx <= 1; dx++) {
            let q = vec2<i32>(clamp(i32(x) + dx, 0, i32(pp.w) - 1), clamp(i32(y) + dy, 0, i32(pp.h) - 1));
            es += dot(abs(textureLoad(tex_a, q, 0).rgb - textureLoad(tex_b, q, 0).rgb), vec3<f32>(1.0 / 3.0));
        } }
        es = es / 9.0;
        let ef = dot(abs(ca - cb), vec3<f32>(1.0 / 3.0));
        stillw = clamp((ef - es) / 0.04, 0.0, 1.0) * select(0.0, 1.0, es < 0.03);
    }
    if ((pp.decisive & 1u) != 0u) {
        let k = clamp((dot(abs(ca - cb), vec3<f32>(1.0 / 3.0)) - 0.04) / 0.08, 0.0, 1.0);
        let pa = select(0.0, 1.0, wa >= wb);
        wa = mix(wa, pa, k); wb = mix(wb, 1.0 - pa, k); r_k = r_k * (1.0 - k);
    }
    if (pp.use_static != 0u) {
        let sa = on_static(gxa, gya); let sb = on_static(gxb, gyb);
        if (sa && sb) { r_k = 0.0; raw = 1.0; } else if (sa) { r_k = 0.0; wa = 0.0; wb = 1.0; } else if (sb) { r_k = 0.0; wa = 1.0; wb = 0.0; }
    }
    let res = vec3<f32>(q(2u, 2u, y0, y1, x0, x1, fx, fy), q(2u, 1u, y0, y1, x0, x1, fx, fy), q(2u, 0u, y0, y1, x0, x1, fx, fy));   // BGR -> RGB
    var c = ca * wa + cb * wb + res * r_k;
    if (stillw > 0.0) {
        let here = vec2<i32>(i32(x), i32(y));
        c = mix(c, 0.5 * (textureLoad(tex_a, here, 0).rgb + textureLoad(tex_b, here, 0).rgb), stillw);
    }
    if (pp.use_static != 0u) {
        let own = 0.5 * (textureLoad(tex_a, vec2<i32>(i32(x), i32(y)), 0).rgb + textureLoad(tex_b, vec2<i32>(i32(x), i32(y)), 0).rgb);
        c = mix(c, own, raw);
        let k = clamp((f32(age[y * pp.wp + x]) - pp.min_age + 2.0) / 2.0, 0.0, 1.0);
        c = mix(c, own, k);
    }
    textureStore(dst, vec2<i32>(i32(x), i32(y)), vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
