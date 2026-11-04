enable f16;
// Nearest-source regions, the in-between's phase for the whole frame (near_d's per-cell estimates where the warped
// frames agree, averaged): which source frame is the nearer one must be one decision per frame, or regions of one
// frame come from A and regions next to them from B. 0.5 when nothing moves. Out: [1].
struct Par { hq: u32, wq: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> D: array<f16>;
@group(0) @binding(1) var<storage, read_write> outp: array<f16>;
@group(0) @binding(2) var<uniform> pp: Par;

var<workgroup> s_t: array<f32, 256>;
var<workgroup> s_n: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let plane = pp.hq * pp.wq;
    var t = 0.0; var n = 0.0;
    for (var i = lid.x; i < plane; i += 256u) {
        let r = f32(D[plane + i]);
        if (r >= 0.0 && f32(D[i]) < 0.5) { t += r; n += 1.0; }
    }
    s_t[lid.x] = t; s_n[lid.x] = n;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off >>= 1u) {
        if (lid.x < off) { s_t[lid.x] += s_t[lid.x + off]; s_n[lid.x] += s_n[lid.x + off]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { outp[0] = f16(select(0.5, s_t[0] / s_n[0], s_n[0] > 0.0)); }
}
