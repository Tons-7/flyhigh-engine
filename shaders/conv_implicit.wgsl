enable f16;
enable wgpu_cooperative_matrix;
// Implicit-GEMM conv with a cheap gather: precompute (oy,ox) for the 16 M-rows ONCE into shared
// (kills the runtime div/mod by Wout from the hot gather loop), then gather A-tiles from input.
struct P { cin: u32, h: u32, w: u32, hout: u32, wout: u32, stride: u32, pad: u32, k: u32, kp: u32, n: u32, np: u32, m: u32 };
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read> Bw: array<f16>;
@group(0) @binding(2) var<storage, read> zero: array<f32>;
@group(0) @binding(3) var<storage, read_write> Cout: array<f32>;
@group(0) @binding(4) var<uniform> p: P;

var<workgroup> tA: array<f16, 256>;
var<workgroup> oys: array<i32, 16>;
var<workgroup> oxs: array<i32, 16>;
var<workgroup> valid: array<u32, 16>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let m0 = wid.y * 16u;
    let n0 = wid.x * 16u;
    let snp = p.np;
    if (lid.x < 16u) {
        let m = m0 + lid.x;
        if (m < p.m) { oys[lid.x] = i32(m / p.wout); oxs[lid.x] = i32(m % p.wout); valid[lid.x] = 1u; }
        else { oys[lid.x] = 0; oxs[lid.x] = 0; valid[lid.x] = 0u; }
    }
    workgroupBarrier();
    let pz = &zero[0];
    var acc = coopLoad<coop_mat16x16<f32, C>>(pz);
    let kt_count = p.kp / 16u;
    for (var kt: u32 = 0u; kt < kt_count; kt = kt + 1u) {
        for (var e: u32 = lid.x; e < 256u; e = e + 32u) {
            let r = e / 16u; let c = e % 16u;
            let kcol = kt * 16u + c;
            var v: f16 = f16(0.0);
            if (valid[r] == 1u && kcol < p.k) {
                let ci = kcol / 9u; let rem = kcol % 9u; let ky = i32(rem / 3u); let kx = i32(rem % 3u);
                let yi = oys[r] * i32(p.stride) + ky - i32(p.pad);
                let xi = oxs[r] * i32(p.stride) + kx - i32(p.pad);
                if (yi >= 0 && yi < i32(p.h) && xi >= 0 && xi < i32(p.w)) { v = inp[(ci * p.h + u32(yi)) * p.w + u32(xi)]; }
            }
            tA[e] = v;
        }
        workgroupBarrier();
        let pa = &tA[0];
        let a = coopLoadT<coop_mat16x16<f16, A>>(pa, 16u);
        let pb = &Bw[(kt * 16u) * snp + n0];
        let b = coopLoadT<coop_mat16x16<f16, B>>(pb, snp);
        acc = coopMultiplyAdd(a, b, acc);
        workgroupBarrier();
    }
    let fa = acc;
    let pc = &Cout[m0 * snp + n0];
    coopStoreT(fa, pc, snp);
}
