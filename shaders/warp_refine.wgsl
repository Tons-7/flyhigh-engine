// Camera warp, refining the shift one pyramid level down (warp/gpu.rs): around the level above's shift (warp_best.wgsl),
// the two frames' cell grids at this level compared under every candidate within `rx` x `ry` cells, on every
// `stride`-th cell each way and leaving out the blocks the mask holds screen-locked (a weapon or HUD would pull small
// shifts towards zero). Same output as warp_shift.wgsl, for this level's warp_best pass.
struct P {
    gw: u32, gh: u32, rx: i32, ry: i32, ncx: u32, mgw: u32, mgh: u32, stride: u32,
    cw: f32, ch: f32, ha: f32, he: f32, w: u32, h: u32, _a: u32, _b: u32,
};
@group(0) @binding(0) var<storage, read> cells_a: array<f32>;
@group(0) @binding(1) var<storage, read> cells_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read> prior: array<f32>;
@group(0) @binding(5) var<storage, read> mask: array<f32>;

var<workgroup> s_sum: array<f32, 256>;
var<workgroup> s_cnt: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let cand = wid.x;
    let base = vec2<i32>(i32(round(prior[0] / p.cw)), i32(round(prior[1] / p.ch)));
    let dx = base.x + i32(cand % p.ncx) - p.rx;
    let dy = base.y + i32(cand / p.ncx) - p.ry;
    let hw = (p.gw + p.stride - 1u) / p.stride;
    let n = hw * ((p.gh + p.stride - 1u) / p.stride);
    var sum = 0.0; var cnt = 0.0;
    // f: the focal length in centre pixels (a cell spans cw / f radians)
    let f = p.cw * f32(p.gw) / (2.0 * p.ha);
    for (var i = lid.x; i < n; i += 256u) {
        let ax = i32((i % hw) * p.stride); let ay = i32((i / hw) * p.stride);
        let az = ((f32(ax) + 0.5) / f32(p.gw) * 2.0 - 1.0) * p.ha;
        // yaw moves every column by the same cells; pitch moves a column at azimuth az by cos(az) of it
        let bx = ax + dx;
        let byf = f32(ay) + f32(dy) * cos(az);
        let by = i32(floor(byf)); let fy = byf - f32(by);
        if (bx < 0 || by < 0 || bx >= i32(p.gw) || by + 1 >= i32(p.gh)) { continue; }
        let a = cells_a[u32(ay) * p.gw + u32(ax)];
        let b0 = cells_b[u32(by) * p.gw + u32(bx)]; let b1 = cells_b[u32(by + 1) * p.gw + u32(bx)];
        if (a < 0.0 || b0 < 0.0 || b1 < 0.0) { continue; }
        // the cell's pixel in frame A (warp_cells.wgsl's mapping)
        let el = ((f32(ay) + 0.5) / f32(p.gh) * 2.0 - 1.0) * p.he;
        let px = 0.5 * vec2<f32>(f32(p.w), f32(p.h)) + f * vec2<f32>(tan(az), tan(el) / cos(az));
        let mx = min(u32(max(px.x, 0.0) / 8.0), p.mgw - 1u);
        let my = min(u32(max(px.y, 0.0) / 8.0), p.mgh - 1u);
        if (mask[my * p.mgw + mx] > 0.5) { continue; }
        sum += abs(a - mix(b0, b1, fy));
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
        out[cand * 2u + 1u] = s_cnt[0] / f32(n);
    }
}
