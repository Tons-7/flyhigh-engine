enable f16;
// Fused render_pix: upsample flow_tq/vis_tq/R_t (Q res) to full res, warp A/B, visibility
// blend, + residual, clamp — the whole full-res tail in one kernel. One thread per output
// pixel (x,y), loops the 3 image channels. f16 I/O, f32 math. (WGSL forbids storage-pointer
// function params, so samplers read their binding directly.)
// pp.cubic selects the A/B source sampler: 0 = bilinear, 1 = bicubic (cubic convolution a=-0.75,
// per-tap border clamp = torch.grid_sample(mode="bicubic", align_corners=True, padding="border")).
// flow/vis/R upsampling stays bilinear in both modes (the refine-head warps are bilinear too).
// pp.decisive: where the two warped frames disagree (a motion the network could not follow), the one it trusts more
// instead of their blend, and no residual there; a blend of two misplaced copies reads as fog, one copy reads as
// motion. Where they agree the blend is the network's own. Bit 0 of pp.decisive.
// Still test (pp.decisive bit 1): per pixel, two explanations of the pair compete: the pixel stayed (A and B agree in
// place, over 3 x 3) or it moved with the network's flow (the two warped frames agree). A still overlay over a moving
// world (HUD text, a crosshair) is explained far better by staying, but the flow lives on an 8 x 8 px grid and drags
// it along with the world; where staying wins the in-between keeps the still pixels. Only nearly identical pixels
// count as still (0.03 of the range), so a world the network got wrong is not frozen into a blend of two frames.
struct Par { H: u32, W: u32, hq: u32, wq: u32, f0: u32, v0: u32, fsx: f32, fsy: f32, vs: f32, cubic: u32, min_age: u32, decisive: u32 };   // flow_tq / vis_tq may be views; fsx/fsy: flow scale per axis
@group(0) @binding(0) var<storage, read> A: array<f16>;
@group(0) @binding(1) var<storage, read> B: array<f16>;
@group(0) @binding(2) var<storage, read> flow_tq: array<f16>;
@group(0) @binding(3) var<storage, read> vis_tq: array<f16>;
@group(0) @binding(4) var<storage, read> R_t: array<f16>;
@group(0) @binding(5) var<storage, read_write> outp: array<f16>;
@group(0) @binding(6) var<uniform> pp: Par;

fn bil4(v00: f32, v01: f32, v10: f32, v11: f32, fx: f32, fy: f32) -> f32 {
    return (v00 * (1.0 - fx) + v01 * fx) * (1.0 - fy) + (v10 * (1.0 - fx) + v11 * fx) * fy;
}
fn qflow(ch: u32, y0: u32, y1: u32, x0: u32, x1: u32, fx: f32, fy: f32) -> f32 {
    let b = (ch + pp.f0) * pp.hq * pp.wq; let w = pp.wq;
    return bil4(f32(flow_tq[b + y0 * w + x0]), f32(flow_tq[b + y0 * w + x1]), f32(flow_tq[b + y1 * w + x0]), f32(flow_tq[b + y1 * w + x1]), fx, fy) * select(pp.fsy, pp.fsx, (ch & 1u) == 0u);
}
fn qvis(ch: u32, y0: u32, y1: u32, x0: u32, x1: u32, fx: f32, fy: f32) -> f32 {
    let b = (ch + pp.v0) * pp.hq * pp.wq; let w = pp.wq;
    return bil4(f32(vis_tq[b + y0 * w + x0]), f32(vis_tq[b + y0 * w + x1]), f32(vis_tq[b + y1 * w + x0]), f32(vis_tq[b + y1 * w + x1]), fx, fy) * pp.vs;
}
fn qr(ch: u32, y0: u32, y1: u32, x0: u32, x1: u32, fx: f32, fy: f32) -> f32 {
    let b = ch * pp.hq * pp.wq; let w = pp.wq;
    return bil4(f32(R_t[b + y0 * w + x0]), f32(R_t[b + y0 * w + x1]), f32(R_t[b + y1 * w + x0]), f32(R_t[b + y1 * w + x1]), fx, fy);
}
fn wa_s(ch: u32, y0: u32, y1: u32, x0: u32, x1: u32, ax: f32, ay: f32) -> f32 {
    let b = ch * pp.H * pp.W; let w = pp.W;
    return bil4(f32(A[b + y0 * w + x0]), f32(A[b + y0 * w + x1]), f32(A[b + y1 * w + x0]), f32(A[b + y1 * w + x1]), ax, ay);
}
fn wb_s(ch: u32, y0: u32, y1: u32, x0: u32, x1: u32, ax: f32, ay: f32) -> f32 {
    let b = ch * pp.H * pp.W; let w = pp.W;
    return bil4(f32(B[b + y0 * w + x0]), f32(B[b + y0 * w + x1]), f32(B[b + y1 * w + x0]), f32(B[b + y1 * w + x1]), ax, ay);
}
// Keys cubic convolution, a = -0.75 (PyTorch bicubic). The four tap weights for a point at
// fractional offset t in [0,1): taps at floor-1, floor, floor+1, floor+2 -> distances 1+t, t, 1-t, 2-t.
fn cub1(x: f32) -> f32 { let a = -0.75; return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0; }   // |x| <= 1
fn cub2(x: f32) -> f32 { let a = -0.75; return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a; }   // 1 < |x| < 2
fn cubw(t: f32) -> vec4<f32> { return vec4<f32>(cub2(t + 1.0), cub1(t), cub1(1.0 - t), cub2(2.0 - t)); }
fn clampx(v: i32) -> u32 { return u32(max(0, min(v, i32(pp.W) - 1))); }
fn clampy(v: i32) -> u32 { return u32(max(0, min(v, i32(pp.H) - 1))); }
// one bicubic row: 4 clamped x-taps on row `yy` of `base`, weighted by cx (caller supplies A or B via the two wrappers)
fn rowa(base: u32, yy: u32, px: i32, cx: vec4<f32>) -> f32 {
    return f32(A[base + yy + clampx(px - 1)]) * cx.x + f32(A[base + yy + clampx(px)]) * cx.y
         + f32(A[base + yy + clampx(px + 1)]) * cx.z + f32(A[base + yy + clampx(px + 2)]) * cx.w;
}
fn rowb(base: u32, yy: u32, px: i32, cx: vec4<f32>) -> f32 {
    return f32(B[base + yy + clampx(px - 1)]) * cx.x + f32(B[base + yy + clampx(px)]) * cx.y
         + f32(B[base + yy + clampx(px + 1)]) * cx.z + f32(B[base + yy + clampx(px + 2)]) * cx.w;
}
fn wa_cub(ch: u32, gx: f32, gy: f32) -> f32 {
    let base = ch * pp.H * pp.W; let w = pp.W;
    let px = i32(floor(gx)); let py = i32(floor(gy));
    let cx = cubw(gx - f32(px)); let cy = cubw(gy - f32(py));   // unrolled: no dynamic vector index (portable to all WGSL backends)
    return rowa(base, clampy(py - 1) * w, px, cx) * cy.x + rowa(base, clampy(py) * w, px, cx) * cy.y
         + rowa(base, clampy(py + 1) * w, px, cx) * cy.z + rowa(base, clampy(py + 2) * w, px, cx) * cy.w;
}
fn wb_cub(ch: u32, gx: f32, gy: f32) -> f32 {
    let base = ch * pp.H * pp.W; let w = pp.W;
    let px = i32(floor(gx)); let py = i32(floor(gy));
    let cx = cubw(gx - f32(px)); let cy = cubw(gy - f32(py));
    return rowb(base, clampy(py - 1) * w, px, cx) * cy.x + rowb(base, clampy(py) * w, px, cx) * cy.y
         + rowb(base, clampy(py + 1) * w, px, cx) * cy.z + rowb(base, clampy(py + 2) * w, px, cx) * cy.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= pp.W || y >= pp.H) { return; }
    // Q-res sample corners (shared by flow/vis/R)
    let sh = f32(pp.hq) / f32(pp.H); let sw = f32(pp.wq) / f32(pp.W);
    let iy = (f32(y) + 0.5) * sh - 0.5; let ix = (f32(x) + 0.5) * sw - 0.5;
    let qy0 = i32(floor(iy)); let qx0 = i32(floor(ix));
    let qfy = iy - f32(qy0); let qfx = ix - f32(qx0);
    let qy0c = u32(max(0, min(qy0, i32(pp.hq) - 1))); let qy1c = u32(max(0, min(qy0 + 1, i32(pp.hq) - 1)));
    let qx0c = u32(max(0, min(qx0, i32(pp.wq) - 1))); let qx1c = u32(max(0, min(qx0 + 1, i32(pp.wq) - 1)));

    let f0 = qflow(0u, qy0c, qy1c, qx0c, qx1c, qfx, qfy);
    let f1 = qflow(1u, qy0c, qy1c, qx0c, qx1c, qfx, qfy);
    let f2 = qflow(2u, qy0c, qy1c, qx0c, qx1c, qfx, qfy);
    let f3 = qflow(3u, qy0c, qy1c, qx0c, qx1c, qfx, qfy);
    let v0 = qvis(0u, qy0c, qy1c, qx0c, qx1c, qfx, qfy);
    let v1 = qvis(1u, qy0c, qy1c, qx0c, qx1c, qfx, qfy);
    let v2 = qvis(2u, qy0c, qy1c, qx0c, qx1c, qfx, qfy);
    let m = max(v0, max(v1, v2));
    let e0 = exp(v0 - m); let e1 = exp(v1 - m); let e2 = exp(v2 - m); let s = e0 + e1 + e2;
    let wa_w = (e0 + 0.5 * e2) / s; let wb_w = (e1 + 0.5 * e2) / s;

    // warp positions for A (by fwd flow) and B (by bwd flow)
    let gxa = f32(x) + f0; let gya = f32(y) + f1;
    let gxb = f32(x) + f2; let gyb = f32(y) + f3;
    // bilinear corners (used when pp.cubic == 0)
    let axx0 = i32(floor(gxa)); let axy0 = i32(floor(gya));
    let aax = gxa - f32(axx0); let aay = gya - f32(axy0);
    let ax0 = u32(max(0, min(axx0, i32(pp.W) - 1))); let ax1 = u32(max(0, min(axx0 + 1, i32(pp.W) - 1)));
    let ay0 = u32(max(0, min(axy0, i32(pp.H) - 1))); let ay1 = u32(max(0, min(axy0 + 1, i32(pp.H) - 1)));
    let bxx0 = i32(floor(gxb)); let bxy0 = i32(floor(gyb));
    let bax = gxb - f32(bxx0); let bay = gyb - f32(bxy0);
    let bx0 = u32(max(0, min(bxx0, i32(pp.W) - 1))); let bx1 = u32(max(0, min(bxx0 + 1, i32(pp.W) - 1)));
    let by0 = u32(max(0, min(bxy0, i32(pp.H) - 1))); let by1 = u32(max(0, min(bxy0 + 1, i32(pp.H) - 1)));

    let cubic = pp.cubic != 0u;
    var ca = vec3<f32>(0.0); var cb = vec3<f32>(0.0); var cr = vec3<f32>(0.0);
    for (var c: u32 = 0u; c < 3u; c = c + 1u) {
        var wa: f32; var wb: f32;
        if (cubic) {
            wa = wa_cub(c, gxa, gya);
            wb = wb_cub(c, gxb, gyb);
        } else {
            wa = wa_s(c, ay0, ay1, ax0, ax1, aax, aay);
            wb = wb_s(c, by0, by1, bx0, bx1, bax, bay);
        }
        ca[c] = wa; cb[c] = wb; cr[c] = qr(c, qy0c, qy1c, qx0c, qx1c, qfx, qfy);
    }
    var bwa = wa_w; var bwb = wb_w; var rk = 1.0;
    var stillw = 0.0;
    var own = vec3<f32>(0.0);
    if ((pp.decisive & 2u) != 0u) {
        var es = 0.0;
        for (var dy = -1; dy <= 1; dy++) { for (var dx = -1; dx <= 1; dx++) {
            let yy = u32(clamp(i32(y) + dy, 0, i32(pp.H) - 1)); let xx = u32(clamp(i32(x) + dx, 0, i32(pp.W) - 1));
            for (var c: u32 = 0u; c < 3u; c = c + 1u) {
                let i = (c * pp.H + yy) * pp.W + xx;
                es += abs(f32(A[i]) - f32(B[i]));
            }
        } }
        es = es / 27.0;
        for (var c: u32 = 0u; c < 3u; c = c + 1u) {
            let i = (c * pp.H + y) * pp.W + x;
            own[c] = 0.5 * (f32(A[i]) + f32(B[i]));
        }
        let ef = dot(abs(ca - cb), vec3<f32>(1.0 / 3.0));
        stillw = clamp((ef - es) / 0.04, 0.0, 1.0) * select(0.0, 1.0, es < 0.03);
    }
    if ((pp.decisive & 1u) != 0u) {
        let k = clamp((dot(abs(ca - cb), vec3<f32>(1.0 / 3.0)) - 0.04) / 0.08, 0.0, 1.0);
        let pa = select(0.0, 1.0, wa_w >= wb_w);
        bwa = mix(wa_w, pa, k); bwb = mix(wb_w, 1.0 - pa, k); rk = 1.0 - k;
    }
    // static hook
    let o = clamp(mix(ca * bwa + cb * bwb + cr * rk, own, stillw), vec3<f32>(0.0), vec3<f32>(1.0));
    for (var c: u32 = 0u; c < 3u; c = c + 1u) {
        outp[(c * pp.H + y) * pp.W + x] = f16(o[c]);
    }
}
