enable f16;
// conv_tiled.wgsl with an 8 (pixels) x 8 (channels) register block per thread: TM = 128 px x TN = 64 ch per 128-thread
// workgroup, 16 vec4 FMAs per 4 vec4 shared loads. The 4x4 and 8x4 kernels are shared-memory-bandwidth bound on
// NVIDIA (2-3 LDS.128 per 16-32 FMAs); 8x8 balances shared traffic against FMA issue. Same contracts as conv_tiled.
struct P { cin: u32, h: u32, w: u32, hout: u32, wout: u32, stride: u32, pad: u32, k: u32, kp: u32, n: u32, np: u32, m: u32, cinp: u32, ksplit: u32, residual: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read> Bw: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read> slope: array<f16>;
@group(0) @binding(4) var<storage, read_write> outp: array<f16>;
@group(0) @binding(5) var<uniform> p: P;
@group(0) @binding(6) var<storage, read_write> part: array<f32>;
@group(0) @binding(7) var<storage, read> res: array<f16>;   // optional fused residual (p.residual == 1)

const TX: u32 = 16u; const TY: u32 = 8u;           // 128 threads
const TM: u32 = TX * 8u; const TN: u32 = TY * 8u;  // 128 px x 64 ch
const KS: u32 = 16u;
const AW: u32 = TX * 2u;                           // vec4 per k row of As (2 groups of 64 px)
const BWD: u32 = TY * 2u;                          // vec4 per k row of Bs (2 groups of 32 ch)
var<workgroup> As: array<vec4<f32>, KS * AW>;      // 8 KiB
var<workgroup> Bs: array<vec4<f32>, KS * BWD>;     // 4 KiB

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
fn gather4(k: u32, px: Pix4) -> vec4<f32> {
    if (k >= p.k) { return vec4<f32>(0.0); }
    let ci = k / 9u; let tap = k - ci * 9u; let ky = i32(tap / 3u); let kx = i32(tap - u32(ky) * 3u);
    return vec4<f32>(ld(ci, px.yb.x + ky, px.xb.x + kx, px.ok.x), ld(ci, px.yb.y + ky, px.xb.y + kx, px.ok.y),
                     ld(ci, px.yb.z + ky, px.xb.z + kx, px.ok.z), ld(ci, px.yb.w + ky, px.xb.w + kx, px.ok.w));
}
fn fin(r: f32, idx: u32) -> f16 {
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
fn out8(z: u32, split: bool, nb: u32, mb: u32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, c3: vec4<f32>) {
    if (split) { store_part(z, nb, mb, c0); store_part(z, nb + 1u, mb, c1); store_part(z, nb + 2u, mb, c2); store_part(z, nb + 3u, mb, c3); }
    else { store4(nb, mb, c0); store4(nb + 1u, mb, c1); store4(nb + 2u, mb, c2); store4(nb + 3u, mb, c3); }
}

@compute @workgroup_size(TX * TY)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let m0 = wid.y * TM; let n0 = wid.x * TN;
    let tx = lid.x % TX; let ty = lid.x / TX;
    var c00 = vec4<f32>(0.0); var c01 = vec4<f32>(0.0); var c02 = vec4<f32>(0.0); var c03 = vec4<f32>(0.0);
    var c04 = vec4<f32>(0.0); var c05 = vec4<f32>(0.0); var c06 = vec4<f32>(0.0); var c07 = vec4<f32>(0.0);
    var c10 = vec4<f32>(0.0); var c11 = vec4<f32>(0.0); var c12 = vec4<f32>(0.0); var c13 = vec4<f32>(0.0);
    var c14 = vec4<f32>(0.0); var c15 = vec4<f32>(0.0); var c16 = vec4<f32>(0.0); var c17 = vec4<f32>(0.0);
    let kt_count = p.kp / KS;
    let ksplit = max(p.ksplit, 1u);
    let per = (kt_count + ksplit - 1u) / ksplit;
    let kt0 = wid.z * per; let kt1 = min(kt0 + per, kt_count);
    let px = pix4(m0 + (lid.x % AW) * 4u);   // (TX*TY) % AW == 0: fixed pixel group per thread
    for (var kt: u32 = kt0; kt < kt1; kt = kt + 1u) {
        let kb = kt * KS;
        for (var e: u32 = lid.x; e < KS * AW; e = e + TX * TY) {
            As[e] = gather4(kb + e / AW, px);
        }
        for (var e: u32 = lid.x; e < KS * BWD; e = e + TX * TY) {
            let kk = e / BWD; let nq = e - kk * BWD;
            Bs[e] = vec4<f32>(Bw[((kb + kk) * p.np + n0 + nq * 4u) >> 2u]);
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < KS; kk = kk + 1u) {
            let a0 = As[kk * AW + tx];
            let a1 = As[kk * AW + TX + tx];
            let b0 = Bs[kk * BWD + ty];
            let b1 = Bs[kk * BWD + TY + ty];
            c00 = fma(a0, vec4<f32>(b0.x), c00); c10 = fma(a1, vec4<f32>(b0.x), c10);
            c01 = fma(a0, vec4<f32>(b0.y), c01); c11 = fma(a1, vec4<f32>(b0.y), c11);
            c02 = fma(a0, vec4<f32>(b0.z), c02); c12 = fma(a1, vec4<f32>(b0.z), c12);
            c03 = fma(a0, vec4<f32>(b0.w), c03); c13 = fma(a1, vec4<f32>(b0.w), c13);
            c04 = fma(a0, vec4<f32>(b1.x), c04); c14 = fma(a1, vec4<f32>(b1.x), c14);
            c05 = fma(a0, vec4<f32>(b1.y), c05); c15 = fma(a1, vec4<f32>(b1.y), c15);
            c06 = fma(a0, vec4<f32>(b1.z), c06); c16 = fma(a1, vec4<f32>(b1.z), c16);
            c07 = fma(a0, vec4<f32>(b1.w), c07); c17 = fma(a1, vec4<f32>(b1.w), c17);
        }
        workgroupBarrier();
    }
    let mb0 = m0 + tx * 4u; let mb1 = m0 + TX * 4u + tx * 4u;
    let nb0 = n0 + ty * 4u; let nb1 = n0 + TY * 4u + ty * 4u;
    let split = ksplit > 1u;
    out8(wid.z, split, nb0, mb0, c00, c01, c02, c03);
    out8(wid.z, split, nb1, mb0, c04, c05, c06, c07);
    out8(wid.z, split, nb0, mb1, c10, c11, c12, c13);
    out8(wid.z, split, nb1, mb1, c14, c15, c16, c17);
}
