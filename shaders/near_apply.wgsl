enable f16;
// Nearest-source regions, the last step: where the smoothed disagreement says the network could not follow the motion
// (a flick), the in-between becomes the nearer source frame alone, warped by the network's own flow: one coherent
// frame instead of the blend of two misplaced copies (fog) or a per-pixel choice (patches). Updates the render in
// place; elsewhere the render stays as it is.
struct Par { H: u32, W: u32, hq: u32, wq: u32, f0: u32, fsx: f32, fsy: f32, _a: u32 };
@group(0) @binding(0) var<storage, read> A: array<f16>;
@group(0) @binding(1) var<storage, read> B: array<f16>;
@group(0) @binding(2) var<storage, read> flow_tq: array<f16>;
@group(0) @binding(3) var<storage, read> K: array<f16>;
@group(0) @binding(4) var<storage, read> T: array<f16>;   // near_t.wgsl: the in-between's phase
@group(0) @binding(5) var<storage, read_write> outp: array<f16>;
@group(0) @binding(6) var<uniform> pp: Par;

fn bil(buf: u32, b: u32, x0: u32, x1: u32, y0: u32, y1: u32, fx: f32, fy: f32) -> f32 {
    var v00: f32; var v01: f32; var v10: f32; var v11: f32;
    if (buf == 0u) {
        let w = pp.wq;
        v00 = f32(flow_tq[b + y0 * w + x0]); v01 = f32(flow_tq[b + y0 * w + x1]);
        v10 = f32(flow_tq[b + y1 * w + x0]); v11 = f32(flow_tq[b + y1 * w + x1]);
    } else {
        let w = pp.wq;
        v00 = f32(K[b + y0 * w + x0]); v01 = f32(K[b + y0 * w + x1]);
        v10 = f32(K[b + y1 * w + x0]); v11 = f32(K[b + y1 * w + x1]);
    }
    return (v00 * (1.0 - fx) + v01 * fx) * (1.0 - fy) + (v10 * (1.0 - fx) + v11 * fx) * fy;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= pp.W || y >= pp.H) { return; }
    let iy = (f32(y) + 0.5) * f32(pp.hq) / f32(pp.H) - 0.5; let ix = (f32(x) + 0.5) * f32(pp.wq) / f32(pp.W) - 0.5;
    let qy0 = i32(floor(iy)); let qx0 = i32(floor(ix)); let fy = iy - f32(qy0); let fx = ix - f32(qx0);
    let y0 = u32(clamp(qy0, 0, i32(pp.hq) - 1)); let y1 = u32(clamp(qy0 + 1, 0, i32(pp.hq) - 1));
    let x0 = u32(clamp(qx0, 0, i32(pp.wq) - 1)); let x1 = u32(clamp(qx0 + 1, 0, i32(pp.wq) - 1));
    let plane = pp.hq * pp.wq;
    let k = clamp(2.0 * bil(1u, 0u, x0, x1, y0, y1, fx, fy), 0.0, 1.0);
    if (k <= 0.0) { return; }
    // the nearer frame for the whole in-between; the middle (t = 0.5) goes to A
    let use_a = f32(T[0]) <= 0.52;
    let c0 = select(2u, 0u, use_a);
    let gx = f32(x) + bil(0u, (pp.f0 + c0) * plane, x0, x1, y0, y1, fx, fy) * pp.fsx;
    let gy = f32(y) + bil(0u, (pp.f0 + c0 + 1u) * plane, x0, x1, y0, y1, fx, fy) * pp.fsy;
    let ax = i32(floor(gx)); let ay = i32(floor(gy)); let ffx = gx - f32(ax); let ffy = gy - f32(ay);
    let xa = u32(clamp(ax, 0, i32(pp.W) - 1)); let xb = u32(clamp(ax + 1, 0, i32(pp.W) - 1));
    let ya = u32(clamp(ay, 0, i32(pp.H) - 1)); let yb = u32(clamp(ay + 1, 0, i32(pp.H) - 1));
    for (var c: u32 = 0u; c < 3u; c = c + 1u) {
        let b = c * pp.H * pp.W;
        var v: f32;
        if (use_a) {
            v = (f32(A[b + ya * pp.W + xa]) * (1.0 - ffx) + f32(A[b + ya * pp.W + xb]) * ffx) * (1.0 - ffy)
              + (f32(A[b + yb * pp.W + xa]) * (1.0 - ffx) + f32(A[b + yb * pp.W + xb]) * ffx) * ffy;
        } else {
            v = (f32(B[b + ya * pp.W + xa]) * (1.0 - ffx) + f32(B[b + ya * pp.W + xb]) * ffx) * (1.0 - ffy)
              + (f32(B[b + yb * pp.W + xa]) * (1.0 - ffx) + f32(B[b + yb * pp.W + xb]) * ffx) * ffy;
        }
        let i = b + y * pp.W + x;
        outp[i] = f16(mix(f32(outp[i]), v, k));
    }
}
