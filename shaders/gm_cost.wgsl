enable f16;
// GMHS cost maps on a grid `ds` x coarser than the network input (positions and flows in network px): for every cell,
// the luma mismatch |A[p + F0] - B[p + F1]| under the model flow (channel 0) and under the affine hypothesis (channel
// 1; = the model cost where the hypothesis samples outside the frame, so borders never switch), the A->B displacement
// difference between the two hypotheses (2), the unwarped luma and its square (3, 4: texture), the plain frame change
// |A - B| (5). f32 maps; the flow is the E8 flow upsampled bilinearly.
struct Par { h8: u32, w8: u32, h: u32, w: u32, hc: u32, wc: u32, ds: f32, _a: u32 };
@group(0) @binding(0) var<storage, read> A: array<f16>;
@group(0) @binding(1) var<storage, read> B: array<f16>;
@group(0) @binding(2) var<storage, read> flow: array<f16>;
@group(0) @binding(3) var<storage, read> coef: array<f32>;
@group(0) @binding(4) var<storage, read_write> maps: array<f32>;
@group(0) @binding(5) var<uniform> pp: Par;

fn lumA(y: u32, x: u32) -> f32 {
    let p = pp.h * pp.w; let i = y * pp.w + x;
    return 0.299 * f32(A[2u * p + i]) + 0.587 * f32(A[p + i]) + 0.114 * f32(A[i]);
}
fn lumB(y: u32, x: u32) -> f32 {
    let p = pp.h * pp.w; let i = y * pp.w + x;
    return 0.299 * f32(B[2u * p + i]) + 0.587 * f32(B[p + i]) + 0.114 * f32(B[i]);
}
fn clampi(v: i32, lo: i32, hi: i32) -> i32 { return max(lo, min(v, hi)); }
fn sampA(gx: f32, gy: f32) -> f32 {
    let x0 = i32(floor(gx)); let y0 = i32(floor(gy)); let ax = gx - f32(x0); let ay = gy - f32(y0);
    let x0c = u32(clampi(x0, 0, i32(pp.w) - 1)); let x1c = u32(clampi(x0 + 1, 0, i32(pp.w) - 1));
    let y0c = u32(clampi(y0, 0, i32(pp.h) - 1)); let y1c = u32(clampi(y0 + 1, 0, i32(pp.h) - 1));
    let top = lumA(y0c, x0c) * (1.0 - ax) + lumA(y0c, x1c) * ax; let bot = lumA(y1c, x0c) * (1.0 - ax) + lumA(y1c, x1c) * ax;
    return top * (1.0 - ay) + bot * ay;
}
fn sampB(gx: f32, gy: f32) -> f32 {
    let x0 = i32(floor(gx)); let y0 = i32(floor(gy)); let ax = gx - f32(x0); let ay = gy - f32(y0);
    let x0c = u32(clampi(x0, 0, i32(pp.w) - 1)); let x1c = u32(clampi(x0 + 1, 0, i32(pp.w) - 1));
    let y0c = u32(clampi(y0, 0, i32(pp.h) - 1)); let y1c = u32(clampi(y0 + 1, 0, i32(pp.h) - 1));
    let top = lumB(y0c, x0c) * (1.0 - ax) + lumB(y0c, x1c) * ax; let bot = lumB(y1c, x0c) * (1.0 - ax) + lumB(y1c, x1c) * ax;
    return top * (1.0 - ay) + bot * ay;
}
// E8 flow sampled at network-resolution position (rx, ry), bilinear, align_corners = false
fn flow8(c: u32, rx: f32, ry: f32) -> f32 {
    let ix = (rx + 0.5) * f32(pp.w8) / f32(pp.w) - 0.5; let iy = (ry + 0.5) * f32(pp.h8) / f32(pp.h) - 0.5;
    let x0 = i32(floor(ix)); let y0 = i32(floor(iy)); let fx = ix - f32(x0); let fy = iy - f32(y0);
    let x0c = u32(clampi(x0, 0, i32(pp.w8) - 1)); let x1c = u32(clampi(x0 + 1, 0, i32(pp.w8) - 1));
    let y0c = u32(clampi(y0, 0, i32(pp.h8) - 1)); let y1c = u32(clampi(y0 + 1, 0, i32(pp.h8) - 1));
    let b = c * pp.h8 * pp.w8; let w = pp.w8;
    let top = f32(flow[b + y0c * w + x0c]) * (1.0 - fx) + f32(flow[b + y0c * w + x1c]) * fx;
    let bot = f32(flow[b + y1c * w + x0c]) * (1.0 - fx) + f32(flow[b + y1c * w + x1c]) * fx;
    return top * (1.0 - fy) + bot * fy;
}
fn aff(c: u32, rx: f32, ry: f32) -> f32 {
    let ix = (rx + 0.5) * f32(pp.w8) / f32(pp.w) - 0.5; let iy = (ry + 0.5) * f32(pp.h8) / f32(pp.h) - 0.5;
    let xn = -1.0 + 2.0 * ix / f32(max(pp.w8 - 1u, 1u)); let yn = -1.0 + 2.0 * iy / f32(max(pp.h8 - 1u, 1u));
    return coef[c * 4u] + coef[c * 4u + 1u] * xn + coef[c * 4u + 2u] * yn;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= pp.wc || y >= pp.hc) { return; }
    let n = pp.hc * pp.wc; let i = y * pp.wc + x;
    let rx = (f32(x) + 0.5) * pp.ds - 0.5; let ry = (f32(y) + 0.5) * pp.ds - 0.5;   // network-px position of this cell
    let f0x = flow8(0u, rx, ry); let f0y = flow8(1u, rx, ry); let f1x = flow8(2u, rx, ry); let f1y = flow8(3u, rx, ry);
    let cm = abs(sampA(rx + f0x, ry + f0y) - sampB(rx + f1x, ry + f1y));
    let g0x = aff(0u, rx, ry); let g0y = aff(1u, rx, ry); let g1x = aff(2u, rx, ry); let g1y = aff(3u, rx, ry);
    let wm = f32(pp.w) - 1.0; let hm = f32(pp.h) - 1.0;
    let ax = rx + g0x; let ay = ry + g0y; let bx = rx + g1x; let by = ry + g1y;
    let inside = ax >= 0.0 && ax <= wm && ay >= 0.0 && ay <= hm && bx >= 0.0 && bx <= wm && by >= 0.0 && by <= hm;
    var ca: f32 = cm;
    if (inside) { ca = abs(sampA(ax, ay) - sampB(bx, by)); }
    let l = sampA(rx, ry);
    let dvx = (g1x - g0x) - (f1x - f0x); let dvy = (g1y - g0y) - (f1y - f0y);
    maps[i] = cm; maps[n + i] = ca; maps[2u * n + i] = sqrt(dvx * dvx + dvy * dvy); maps[3u * n + i] = l; maps[4u * n + i] = l * l;
    // channel 5 (not box-filtered): plain change |A - B| at the cell, no motion hypothesis. A held drawing on a still
    // shot gives exactly 0; any new drawing, however slow, changes some cells (cadence step detection).
    maps[5u * n + i] = abs(l - sampB(rx, ry));
}
