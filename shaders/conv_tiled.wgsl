enable f16;
// Portable tiled implicit-GEMM 3x3 conv (pad 1, stride 1 or 2) for WebGPU-class devices: no cooperative
// matrix, no subgroups, f16 I/O, f32 FMA, 12 KiB of workgroup memory at most (WebGPU default limit 16 KiB).
//   Out[m][n] = sum_k A[m][k] * B[k][n]      m = output pixel (oy*wout + ox), n = output channel,
//   k = ci*9 + tap (ci-major, K padded to 16), B = [kp][np] row-major f16 (host: buildb).
// One 256-thread workgroup computes a TM x TN tile (TM = 4*TX pixels, TN = 4*TY channels); each thread owns a
// 4 (pixels) x 4 (channels) register block. Per K-step of 16: the A tile (gathered from the input with zero
// padding) and the B tile are staged in shared memory as f32, then every thread does 16 x (2 vec4 loads +
// 4 vec4 FMAs). Consecutive threads (tx) own consecutive pixels, so the epilogue stores are coalesced per
// channel row and B reads are workgroup-uniform per ty (broadcast). Fused bias + PReLU (slope 1 = identity).
// Variants: TX=16,TY=16 (64 px x 64 ch, cout > 32) and TX=32,TY=8 (128 px x 32 ch, cout <= 32); the host
// derives the second from this source by rewriting the TX/TY line.
struct P { cin: u32, h: u32, w: u32, hout: u32, wout: u32, stride: u32, pad: u32, k: u32, kp: u32, n: u32, np: u32, m: u32, cinp: u32, ksplit: u32, residual: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read> inp: array<f16>;          // [cin][h][w]
@group(0) @binding(1) var<storage, read> Bw: array<vec4<f16>>;     // [kp][np] row-major, np % 4 == 0
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read> slope: array<f16>;
@group(0) @binding(4) var<storage, read_write> outp: array<f16>;   // [n][hout][wout] = [n][m]
@group(0) @binding(5) var<uniform> p: P;
// split-K (p.ksplit = number of K splits > 1, dispatch z = split): raw f32 partial sums [z][n][m], no bias/PReLU;
// splitk_reduce.wgsl finishes them. Small-M large-K convs (coarse stage: ~3.7k pixels) otherwise launch too
// few workgroups to fill a big GPU. Bound to a 1-element dummy when unused.
@group(0) @binding(6) var<storage, read_write> part: array<f32>;
// optional fused residual (p.residual == 1): out = f16(prelu(conv)) + res, same shape as the output (dummy when unused)
@group(0) @binding(7) var<storage, read> res: array<f16>;

const TX: u32 = 16u; const TY: u32 = 16u;
const TM: u32 = TX * 4u; const TN: u32 = TY * 4u;
const KS: u32 = 16u;
var<workgroup> As: array<vec4<f32>, KS * TX>;   // [k][m/4]: 4 consecutive pixels
var<workgroup> Bs: array<vec4<f32>, KS * TY>;   // [k][n/4]: 4 consecutive channels

// Per-thread pixel bases: each thread always stages the same 4 consecutive pixels mb..mb+4 (for every k), so their
// input coordinates (oy*stride - pad, ox*stride - pad) and validity are computed once, outside the K loop.
struct Pix4 { yb: vec4<i32>, xb: vec4<i32>, ok: vec4<bool> };
fn pix4(mb: u32) -> Pix4 {
    var r: Pix4;
    var oy = mb / p.wout; var ox = mb - oy * p.wout;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        r.yb[i] = i32(oy * p.stride) - i32(p.pad); r.xb[i] = i32(ox * p.stride) - i32(p.pad); r.ok[i] = oy < p.hout;
        ox += 1u; if (ox == p.wout) { ox = 0u; oy += 1u; }
    }
    return r;
}
fn ld(ci: u32, yi: i32, xi: i32, ok: bool) -> f32 {
    if (!ok || yi < 0 || yi >= i32(p.h) || xi < 0 || xi >= i32(p.w)) { return 0.0; }
    return f32(inp[(ci * p.h + u32(yi)) * p.w + u32(xi)]);
}
// A[mb..mb+4][k] for one K column (one tap of one input channel)
fn gather4(k: u32, px: Pix4) -> vec4<f32> {
    if (k >= p.k) { return vec4<f32>(0.0); }
    let ci = k / 9u; let tap = k - ci * 9u; let ky = i32(tap / 3u); let kx = i32(tap - u32(ky) * 3u);
    return vec4<f32>(ld(ci, px.yb.x + ky, px.xb.x + kx, px.ok.x), ld(ci, px.yb.y + ky, px.xb.y + kx, px.ok.y),
                     ld(ci, px.yb.z + ky, px.xb.z + kx, px.ok.z), ld(ci, px.yb.w + ky, px.xb.w + kx, px.ok.w));
}

fn fin(r: f32, idx: u32) -> f16 {   // f16 round of the activation, then the residual add (bit-equal to the unfused path)
    let h = f16(r);
    if (p.residual == 1u) { return f16(f32(h) + f32(res[idx])); }
    return h;
}
fn store4(co: u32, mb: u32, acc: vec4<f32>) {
    if (co >= p.n) { return; }
    let bv = f32(bias[co]); let sl = f32(slope[co]);
    let v = acc + vec4<f32>(bv);
    let r = select(sl * v, v, v > vec4<f32>(0.0));
    let base = co * p.m + mb;
    if (mb + 4u <= p.m) {
        outp[base] = fin(r.x, base); outp[base + 1u] = fin(r.y, base + 1u); outp[base + 2u] = fin(r.z, base + 2u); outp[base + 3u] = fin(r.w, base + 3u);
    } else {
        if (mb < p.m) { outp[base] = fin(r.x, base); }
        if (mb + 1u < p.m) { outp[base + 1u] = fin(r.y, base + 1u); }
        if (mb + 2u < p.m) { outp[base + 2u] = fin(r.z, base + 2u); }
    }
}

fn store_part(z: u32, co: u32, mb: u32, acc: vec4<f32>) {
    if (co >= p.n) { return; }
    let base = (z * p.n + co) * p.m + mb;
    if (mb + 4u <= p.m) {
        part[base] = acc.x; part[base + 1u] = acc.y; part[base + 2u] = acc.z; part[base + 3u] = acc.w;
    } else {
        if (mb < p.m) { part[base] = acc.x; }
        if (mb + 1u < p.m) { part[base + 1u] = acc.y; }
        if (mb + 2u < p.m) { part[base + 2u] = acc.z; }
    }
}

@compute @workgroup_size(TX * TY)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let m0 = wid.y * TM; let n0 = wid.x * TN;
    let tx = lid.x % TX; let ty = lid.x / TX;
    var acc0 = vec4<f32>(0.0); var acc1 = vec4<f32>(0.0); var acc2 = vec4<f32>(0.0); var acc3 = vec4<f32>(0.0);
    let kt_count = p.kp / KS;
    let ksplit = max(p.ksplit, 1u);
    let per = (kt_count + ksplit - 1u) / ksplit;
    let kt0 = wid.z * per; let kt1 = min(kt0 + per, kt_count);
    let px = pix4(m0 + (lid.x % TX) * 4u);   // (TX*TY) % TX == 0: this thread's staging pixel group is fixed
    for (var kt: u32 = kt0; kt < kt1; kt = kt + 1u) {
        let kb = kt * KS;
        for (var e: u32 = lid.x; e < KS * TX; e = e + TX * TY) {
            As[e] = gather4(kb + e / TX, px);
        }
        for (var e: u32 = lid.x; e < KS * TY; e = e + TX * TY) {
            let kk = e / TY; let nq = e - kk * TY;
            Bs[e] = vec4<f32>(Bw[((kb + kk) * p.np + n0 + nq * 4u) >> 2u]);
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < KS; kk = kk + 1u) {
            let a = As[kk * TX + tx];
            let b = Bs[kk * TY + ty];
            acc0 = fma(a, vec4<f32>(b.x), acc0);
            acc1 = fma(a, vec4<f32>(b.y), acc1);
            acc2 = fma(a, vec4<f32>(b.z), acc2);
            acc3 = fma(a, vec4<f32>(b.w), acc3);
        }
        workgroupBarrier();
    }
    let mb = m0 + tx * 4u;
    let nb = n0 + ty * 4u;
    if (ksplit > 1u) {
        store_part(wid.z, nb, mb, acc0); store_part(wid.z, nb + 1u, mb, acc1);
        store_part(wid.z, nb + 2u, mb, acc2); store_part(wid.z, nb + 3u, mb, acc3);
    } else {
        store4(nb, mb, acc0); store4(nb + 1u, mb, acc1); store4(nb + 2u, mb, acc2); store4(nb + 3u, mb, acc3);
    }
}
