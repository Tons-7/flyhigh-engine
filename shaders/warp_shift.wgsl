// Camera warp, global shift of a pair (warp/gpu.rs): the two frames' cell grids (warp_cells.wgsl) compared under every
// candidate shift of whole cells; one workgroup per candidate writes the mean absolute difference over the overlap and
// the overlap's share of the grid (cells outside the frame, -1, left out). Content at cell x in frame A sits at cell x + shift in frame B.
struct P { gw: u32, gh: u32, rx: i32, ry: i32, ncx: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read> cells_a: array<f32>;
@group(0) @binding(1) var<storage, read> cells_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;   // [candidate][2]: mean difference, overlap share
@group(0) @binding(3) var<uniform> p: P;

var<workgroup> s_sum: array<f32, 256>;
var<workgroup> s_cnt: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let cand = wid.x;
    let dx = i32(cand % p.ncx) - p.rx;
    let dy = i32(cand / p.ncx) - p.ry;
    // only the cells of A whose shifted position stays inside B
    let x0 = max(0, -dx); let x1 = min(i32(p.gw), i32(p.gw) - dx);
    let y0 = max(0, -dy); let y1 = min(i32(p.gh), i32(p.gh) - dy);
    let ow = max(x1 - x0, 0); let n = u32(ow * max(y1 - y0, 0));
    var sum = 0.0; var cnt = 0.0;
    for (var i = lid.x; i < n; i += 256u) {
        let ax = x0 + i32(i % u32(ow)); let ay = y0 + i32(i / u32(ow));
        let a = cells_a[u32(ay) * p.gw + u32(ax)]; let b = cells_b[u32(ay + dy) * p.gw + u32(ax + dx)];
        if (a < 0.0 || b < 0.0) { continue; }
        sum += abs(a - b);
        cnt += 1.0;
    }
    s_sum[lid.x] = sum; s_cnt[lid.x] = cnt;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off >>= 1u) {
        if (lid.x < off) { s_sum[lid.x] += s_sum[lid.x + off]; s_cnt[lid.x] += s_cnt[lid.x + off]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        out[cand * 2u] = s_sum[0] / max(s_cnt[0], 1.0);
        out[cand * 2u + 1u] = s_cnt[0] / f32(p.gw * p.gh);
    }
}
