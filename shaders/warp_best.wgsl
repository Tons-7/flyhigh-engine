// Camera warp, the best shift of a pyramid level (warp/gpu.rs): the candidate with the least difference among those overlapping at least
// 40 % of the grid, refined to a fraction of a cell by a parabola through its neighbours. Writes [sx, sy, valid,
// difference] in pixels. Run per level: over the coarse search, where the shift is valid only when its minimum stands out
// (a flat or changing picture has none), and over each finer search around it (warp_refine.wgsl), which keeps that
// verdict.
struct P { fine: u32, _a: u32, rx: i32, ry: i32, ncx: u32, ncy: u32, cw: f32, ch: f32 };
@group(0) @binding(0) var<storage, read> sad: array<f32>;
@group(0) @binding(1) var<storage, read_write> best: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@group(0) @binding(3) var<storage, read> coarse: array<f32>;   // the coarse result, read by the fine pass

var<workgroup> s_v: array<f32, 256>;
var<workgroup> s_i: array<u32, 256>;
var<workgroup> s_sum: array<f32, 256>;
var<workgroup> s_n: array<f32, 256>;

fn value(cx: i32, cy: i32) -> f32 {
    if (cx < 0 || cy < 0 || cx >= i32(p.ncx) || cy >= i32(p.ncy)) { return 1e9; }
    let i = u32(cy) * p.ncx + u32(cx);
    return select(1e9, sad[i * 2u], sad[i * 2u + 1u] >= 0.4);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let n = p.ncx * p.ncy;
    var bv = 1e9; var bi = 0u; var sum = 0.0; var cnt = 0.0;
    for (var i = lid.x; i < n; i += 256u) {
        if (sad[i * 2u + 1u] < 0.4) { continue; }
        let v = sad[i * 2u];
        sum += v; cnt += 1.0;
        if (v < bv) { bv = v; bi = i; }
    }
    s_v[lid.x] = bv; s_i[lid.x] = bi; s_sum[lid.x] = sum; s_n[lid.x] = cnt;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off >>= 1u) {
        if (lid.x < off) {
            if (s_v[lid.x + off] < s_v[lid.x]) { s_v[lid.x] = s_v[lid.x + off]; s_i[lid.x] = s_i[lid.x + off]; }
            s_sum[lid.x] += s_sum[lid.x + off]; s_n[lid.x] += s_n[lid.x + off];
        }
        workgroupBarrier();
    }
    if (lid.x != 0u) { return; }
    let v0 = s_v[0];
    let mean = s_sum[0] / max(s_n[0], 1.0);
    let cx = i32(s_i[0] % p.ncx); let cy = i32(s_i[0] / p.ncx);
    // sub-cell: the vertex of the parabola through the minimum and its two neighbours, per axis
    var fx = 0.0; var fy = 0.0;
    let l = value(cx - 1, cy); let r = value(cx + 1, cy);
    if (l < 1e8 && r < 1e8 && l + r - 2.0 * v0 > 1e-6) { fx = clamp(0.5 * (l - r) / (l + r - 2.0 * v0), -0.5, 0.5); }
    let u = value(cx, cy - 1); let d = value(cx, cy + 1);
    if (u < 1e8 && d < 1e8 && u + d - 2.0 * v0 > 1e-6) { fy = clamp(0.5 * (u - d) / (u + d - 2.0 * v0), -0.5, 0.5); }
    var base = vec2<f32>(0.0);
    var valid = v0 < 1e8 && v0 < 0.6 * mean;
    if (p.fine != 0u) {
        base = vec2<f32>(round(coarse[0] / p.cw), round(coarse[1] / p.ch));
        valid = v0 < 1e8 && coarse[2] > 0.5;
    }
    best[0] = (base.x + f32(cx - p.rx) + fx) * p.cw;
    best[1] = (base.y + f32(cy - p.ry) + fy) * p.ch;
    best[2] = select(0.0, 1.0, valid);
    best[3] = v0;
}
