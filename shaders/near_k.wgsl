enable f16;
// Nearest-source regions, step 2: both maps of near_d averaged over 7 x 7 cells (about 28 px), so whole regions
// decide together: a per-pixel choice between two frames shows as patches.
struct Par { hq: u32, wq: u32, r: u32, _a: u32 };
@group(0) @binding(0) var<storage, read> D: array<f16>;
@group(0) @binding(1) var<storage, read_write> outp: array<f16>;
@group(0) @binding(2) var<uniform> pp: Par;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qx = i32(gid.x); let qy = i32(gid.y);
    if (gid.x >= pp.wq || gid.y >= pp.hq) { return; }
    let plane = pp.hq * pp.wq; let r = i32(pp.r);
    var s0 = 0.0; var s1 = 0.0; var n = 0.0;
    for (var dy = -r; dy <= r; dy++) { for (var dx = -r; dx <= r; dx++) {
        let x = qx + dx; let y = qy + dy;
        if (x < 0 || y < 0 || x >= i32(pp.wq) || y >= i32(pp.hq)) { continue; }
        let i = u32(y) * pp.wq + u32(x);
        s0 = s0 + f32(D[i]); s1 = s1 + f32(D[plane + i]); n = n + 1.0;
    } }
    let q = gid.y * pp.wq + gid.x;
    outp[q] = f16(s0 / n); outp[plane + q] = f16(s1 / n);
}
