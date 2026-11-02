// Cadence statistics of a pair from the GMHS maps (f32, network px): per-workgroup partials of (count of cells whose
// displacement difference (channel `ch`) exceeds `thr`, sum of those values, sum of the model's photometric cost
// (channel 0) over all cells, count of cells whose plain frame change (channel 5) exceeds `cthr`). The host sums the
// partials: area and mean residual motion of the "element" that moves against the global affine motion, the mean
// mismatch that flags a scene cut, and the share of the frame where the drawing changed at all.
struct Par { n: u32, ch: u32, nwg: u32, _a: u32, thr: f32, cthr: f32, _c: u32, _d: u32 };
@group(0) @binding(0) var<storage, read> maps: array<f32>;
@group(0) @binding(1) var<storage, read_write> part: array<f32>;   // [nwg][4]
@group(0) @binding(2) var<uniform> pp: Par;

var<workgroup> sc: array<f32, 256>;
var<workgroup> ss: array<f32, 256>;
var<workgroup> sm: array<f32, 256>;
var<workgroup> sk: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x; let stride = 256u * pp.nwg;
    var cnt: f32 = 0.0; var sum: f32 = 0.0; var cost: f32 = 0.0; var chg: f32 = 0.0;
    var i: u32 = wid.x * 256u + tid;
    loop {
        if (i >= pp.n) { break; }
        let v = maps[pp.ch * pp.n + i];
        if (v > pp.thr) { cnt = cnt + 1.0; sum = sum + v; }
        cost = cost + maps[i];
        if (maps[5u * pp.n + i] > pp.cthr) { chg = chg + 1.0; }
        i = i + stride;
    }
    sc[tid] = cnt; ss[tid] = sum; sm[tid] = cost; sk[tid] = chg;
    workgroupBarrier();
    for (var off: u32 = 128u; off > 0u; off = off >> 1u) {
        if (tid < off) { sc[tid] = sc[tid] + sc[tid + off]; ss[tid] = ss[tid] + ss[tid + off]; sm[tid] = sm[tid] + sm[tid + off]; sk[tid] = sk[tid] + sk[tid + off]; }
        workgroupBarrier();
    }
    if (tid == 0u) { part[wid.x * 4u] = sc[0]; part[wid.x * 4u + 1u] = ss[0]; part[wid.x * 4u + 2u] = sm[0]; part[wid.x * 4u + 3u] = sk[0]; }
}
