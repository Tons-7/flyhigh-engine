enable f16;
// Global-motion hypothesis (GMHS), fit step 1: per-workgroup partial sums of the weighted normal equations of an
// affine fit f(x, y) = a0 + a1 x + a2 y for each of the 4 flow channels of the E8 flow (x, y = normalized cell
// coordinates in [-1, 1]). Weights: 1 on the first iteration, Huber-like delta / max(delta, |r|) afterwards (r = the
// residual against the previous iteration's coefficients) — a robust fit that ignores the blobs we want to replace.
struct Par { h: u32, w: u32, nwg: u32, use_w: u32, delta: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read> flow: array<f16>;
@group(0) @binding(1) var<storage, read> coef: array<f32>;        // [4][4]: a0, a1, a2, pad per channel
@group(0) @binding(2) var<storage, read_write> part: array<f32>;  // [nwg][4][9]
@group(0) @binding(3) var<uniform> pp: Par;

var<workgroup> sh: array<f32, 2304>;   // 9 sums x 256 threads

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = pp.h * pp.w; let tid = lid.x; let stride = 256u * pp.nwg;
    let sx = 2.0 / f32(max(pp.w - 1u, 1u)); let sy = 2.0 / f32(max(pp.h - 1u, 1u));
    for (var c: u32 = 0u; c < 4u; c = c + 1u) {
        var s: array<f32, 9>;
        var i: u32 = wid.x * 256u + tid;
        loop {
            if (i >= n) { break; }
            let yy = i / pp.w; let xx = i - yy * pp.w;
            let x = -1.0 + sx * f32(xx); let y = -1.0 + sy * f32(yy);
            let f = f32(flow[c * n + i]);
            var w2: f32 = 1.0;
            if (pp.use_w != 0u) {
                let r = abs(f - (coef[c * 4u] + coef[c * 4u + 1u] * x + coef[c * 4u + 2u] * y));
                w2 = pp.delta / max(pp.delta, r);
            }
            s[0] = s[0] + w2; s[1] = s[1] + w2 * x; s[2] = s[2] + w2 * y;
            s[3] = s[3] + w2 * x * x; s[4] = s[4] + w2 * x * y; s[5] = s[5] + w2 * y * y;
            s[6] = s[6] + w2 * f; s[7] = s[7] + w2 * x * f; s[8] = s[8] + w2 * y * f;
            i = i + stride;
        }
        for (var k: u32 = 0u; k < 9u; k = k + 1u) { sh[k * 256u + tid] = s[k]; }
        workgroupBarrier();
        for (var off: u32 = 128u; off > 0u; off = off >> 1u) {
            if (tid < off) { for (var k: u32 = 0u; k < 9u; k = k + 1u) { sh[k * 256u + tid] = sh[k * 256u + tid] + sh[k * 256u + tid + off]; } }
            workgroupBarrier();
        }
        if (tid == 0u) { for (var k: u32 = 0u; k < 9u; k = k + 1u) { part[(wid.x * 4u + c) * 9u + k] = sh[k * 256u]; } }
        workgroupBarrier();
    }
}
