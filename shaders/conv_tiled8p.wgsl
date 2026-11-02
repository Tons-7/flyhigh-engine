enable f16;
// conv_tiled8.wgsl (8x4 register block, 128 px x 64 ch, 256 threads) with software pipelining: the global loads for
// K-step t+1 (gathered A, B row) are issued into registers before the FMAs of step t and written to shared after
// them, so global-memory latency overlaps the inner loop instead of serialising with it. One shared tile set (12 KiB)
// keeps the WebGPU 16 KiB default; two barriers per K-step remain. Same contracts as conv_tiled.wgsl.
struct P { cin: u32, h: u32, w: u32, hout: u32, wout: u32, stride: u32, pad: u32, k: u32, kp: u32, n: u32, np: u32, m: u32, cinp: u32, ksplit: u32, residual: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read> Bw: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read> slope: array<f16>;
@group(0) @binding(4) var<storage, read_write> outp: array<f16>;
@group(0) @binding(5) var<uniform> p: P;
@group(0) @binding(6) var<storage, read_write> part: array<f32>;
@group(0) @binding(7) var<storage, read> res: array<f16>;

const TX: u32 = 16u; const TY: u32 = 16u;
const TM: u32 = TX * 8u; const TN: u32 = TY * 4u;
const KS: u32 = 16u;
const AW: u32 = TX * 2u;                           // 32 vec4 per k row of As; 512 total = 2 per thread
var<workgroup> As: array<vec4<f32>, KS * AW>;
var<workgroup> Bs: array<vec4<f32>, KS * TY>;      // 256 = 1 per thread

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

@compute @workgroup_size(TX * TY)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let m0 = wid.y * TM; let n0 = wid.x * TN;
    let tx = lid.x % TX; let ty = lid.x / TX;
    var c00 = vec4<f32>(0.0); var c01 = vec4<f32>(0.0); var c02 = vec4<f32>(0.0); var c03 = vec4<f32>(0.0);
    var c10 = vec4<f32>(0.0); var c11 = vec4<f32>(0.0); var c12 = vec4<f32>(0.0); var c13 = vec4<f32>(0.0);
    let kt_count = p.kp / KS;
    let ksplit = max(p.ksplit, 1u);
    let per = (kt_count + ksplit - 1u) / ksplit;
    let kt0 = wid.z * per; let kt1 = min(kt0 + per, kt_count);
    // this thread's staging slots: As[e0], As[e1] (same pixel group, k rows lid/32 and lid/32 + 8), Bs[lid]
    let e0 = lid.x; let e1 = lid.x + TX * TY;
    let ka0 = e0 / AW; let ka1 = e1 / AW;
    let px = pix4(m0 + (lid.x % AW) * 4u);
    let kbb = lid.x / TY; let nq = lid.x - kbb * TY;
    let bcol = (n0 + nq * 4u) >> 2u;
    // prologue: stage K-step kt0
    if (kt0 < kt1) {
        let kb = kt0 * KS;
        As[e0] = gather4(kb + ka0, px); As[e1] = gather4(kb + ka1, px);
        Bs[lid.x] = vec4<f32>(Bw[((kb + kbb) * p.np) / 4u + bcol]);
    }
    workgroupBarrier();
    for (var kt: u32 = kt0; kt < kt1; kt = kt + 1u) {
        // prefetch K-step kt+1 into registers (global loads in flight during the FMAs below)
        let more = kt + 1u < kt1;
        var na0 = vec4<f32>(0.0); var na1 = vec4<f32>(0.0); var nb = vec4<f32>(0.0);
        if (more) {
            let kb = (kt + 1u) * KS;
            na0 = gather4(kb + ka0, px); na1 = gather4(kb + ka1, px);
            nb = vec4<f32>(Bw[((kb + kbb) * p.np) / 4u + bcol]);
        }
        for (var kk: u32 = 0u; kk < KS; kk = kk + 1u) {
            let a0 = As[kk * AW + tx];
            let a1 = As[kk * AW + TX + tx];
            let b = Bs[kk * TY + ty];
            c00 = fma(a0, vec4<f32>(b.x), c00); c10 = fma(a1, vec4<f32>(b.x), c10);
            c01 = fma(a0, vec4<f32>(b.y), c01); c11 = fma(a1, vec4<f32>(b.y), c11);
            c02 = fma(a0, vec4<f32>(b.z), c02); c12 = fma(a1, vec4<f32>(b.z), c12);
            c03 = fma(a0, vec4<f32>(b.w), c03); c13 = fma(a1, vec4<f32>(b.w), c13);
        }
        workgroupBarrier();
        if (more) { As[e0] = na0; As[e1] = na1; Bs[lid.x] = nb; }
        workgroupBarrier();
    }
    let mb0 = m0 + tx * 4u; let mb1 = m0 + TX * 4u + tx * 4u;
    let nb0 = n0 + ty * 4u;
    if (ksplit > 1u) {
        store_part(wid.z, nb0, mb0, c00); store_part(wid.z, nb0 + 1u, mb0, c01); store_part(wid.z, nb0 + 2u, mb0, c02); store_part(wid.z, nb0 + 3u, mb0, c03);
        store_part(wid.z, nb0, mb1, c10); store_part(wid.z, nb0 + 1u, mb1, c11); store_part(wid.z, nb0 + 2u, mb1, c12); store_part(wid.z, nb0 + 3u, mb1, c13);
    } else {
        store4(nb0, mb0, c00); store4(nb0 + 1u, mb0, c01); store4(nb0 + 2u, mb0, c02); store4(nb0 + 3u, mb0, c03);
        store4(nb0, mb1, c10); store4(nb0 + 1u, mb1, c11); store4(nb0 + 2u, mb1, c12); store4(nb0 + 3u, mb1, c13);
    }
}
