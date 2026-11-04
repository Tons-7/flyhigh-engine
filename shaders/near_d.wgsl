enable f16;
// Nearest-source regions, step 1 of 3 (ops/render.rs `render_ex`): per cell of the network's quarter-resolution flow,
// how far the two frames warped by the flow disagree there (0..1 of the way from agreement at 0.04 to clear
// disagreement at 0.12, luma), and the cell's estimate of the in-between's phase, |flow to A| / (|flow to A| + |flow
// to B|) (-1 where the cell barely moves and says nothing). Out: [2][hq][wq].
struct Par { H: u32, W: u32, hq: u32, wq: u32, f0: u32, fsx: f32, fsy: f32, _a: u32 };
@group(0) @binding(0) var<storage, read> A: array<f16>;
@group(0) @binding(1) var<storage, read> B: array<f16>;
@group(0) @binding(2) var<storage, read> flow_tq: array<f16>;
@group(0) @binding(3) var<storage, read_write> outp: array<f16>;
@group(0) @binding(4) var<uniform> pp: Par;

fn luma_at(img: u32, gx: f32, gy: f32) -> f32 {
    let x0 = i32(floor(gx)); let y0 = i32(floor(gy)); let fx = gx - f32(x0); let fy = gy - f32(y0);
    var l = 0.0;
    var wts = array<f32, 3>(0.299, 0.587, 0.114);
    for (var c: u32 = 0u; c < 3u; c = c + 1u) {
        let b = c * pp.H * pp.W;
        var v = 0.0;
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let xx = u32(clamp(x0 + i32(k & 1u), 0, i32(pp.W) - 1)); let yy = u32(clamp(y0 + i32(k >> 1u), 0, i32(pp.H) - 1));
            let wx = select(1.0 - fx, fx, (k & 1u) == 1u); let wy = select(1.0 - fy, fy, (k >> 1u) == 1u);
            let s = select(f32(B[b + yy * pp.W + xx]), f32(A[b + yy * pp.W + xx]), img == 0u);
            v = v + s * wx * wy;
        }
        l = l + v * wts[c];
    }
    return l;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qx = gid.x; let qy = gid.y;
    if (qx >= pp.wq || qy >= pp.hq) { return; }
    let plane = pp.hq * pp.wq; let q = qy * pp.wq + qx;
    let fax = f32(flow_tq[(pp.f0 + 0u) * plane + q]) * pp.fsx; let fay = f32(flow_tq[(pp.f0 + 1u) * plane + q]) * pp.fsy;
    let fbx = f32(flow_tq[(pp.f0 + 2u) * plane + q]) * pp.fsx; let fby = f32(flow_tq[(pp.f0 + 3u) * plane + q]) * pp.fsy;
    let px = (f32(qx) + 0.5) * f32(pp.W) / f32(pp.wq) - 0.5; let py = (f32(qy) + 0.5) * f32(pp.H) / f32(pp.hq) - 0.5;
    let d = abs(luma_at(0u, px + fax, py + fay) - luma_at(1u, px + fbx, py + fby));
    outp[q] = f16(clamp((d - 0.04) / 0.08, 0.0, 1.0));
    let ma = length(vec2<f32>(fax, fay)); let mb = length(vec2<f32>(fbx, fby));
    outp[plane + q] = f16(select(-1.0, ma / max(ma + mb, 1e-3), ma + mb > 1.0));
}
