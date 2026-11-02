enable f16;
enable wgpu_cooperative_matrix;
// Single-pass Tensor-Core conv: implicit-GEMM (register-blocked 16x64) with the bias+PReLU
// epilogue FUSED in — coopStore the 4 accumulators to shared, then scatter to [Cout,H,W].
struct P { cin: u32, h: u32, w: u32, hout: u32, wout: u32, stride: u32, pad: u32, k: u32, kp: u32, n: u32, np: u32, m: u32 };
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read> Bw: array<f16>;
@group(0) @binding(2) var<storage, read> zero: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f16>;
@group(0) @binding(4) var<storage, read> slope: array<f16>;
@group(0) @binding(5) var<storage, read_write> outp: array<f16>;
@group(0) @binding(6) var<uniform> p: P;

var<workgroup> tA: array<f16, 256>;
var<workgroup> tC: array<f32, 1024>;
var<workgroup> oys: array<i32, 16>;
var<workgroup> oxs: array<i32, 16>;
var<workgroup> valid: array<u32, 16>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let m0 = wid.y * 16u;
    let n0 = wid.x * 64u;
    let snp = p.np;
    if (lid.x < 16u) {
        let m = m0 + lid.x;
        if (m < p.m) { oys[lid.x] = i32(m / p.wout); oxs[lid.x] = i32(m % p.wout); valid[lid.x] = 1u; }
        else { oys[lid.x] = 0; oxs[lid.x] = 0; valid[lid.x] = 0u; }
    }
    workgroupBarrier();
    let pz = &zero[0];
    var a0 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a1 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a2 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a3 = coopLoad<coop_mat16x16<f32, C>>(pz);
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
        let kb = (kt * 16u) * snp + n0;
        let pb0 = &Bw[kb]; a0 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(pb0, snp), a0);
        let pb1 = &Bw[kb + 16u]; a1 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(pb1, snp), a1);
        let pb2 = &Bw[kb + 32u]; a2 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(pb2, snp), a2);
        let pb3 = &Bw[kb + 48u]; a3 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(pb3, snp), a3);
        workgroupBarrier();
    }
    let f0 = a0; let f1 = a1; let f2 = a2; let f3 = a3;
    let q0 = &tC[0]; coopStoreT(f0, q0, 16u);
    let q1 = &tC[256]; coopStoreT(f1, q1, 16u);
    let q2 = &tC[512]; coopStoreT(f2, q2, 16u);
    let q3 = &tC[768]; coopStoreT(f3, q3, 16u);
    workgroupBarrier();
    for (var e: u32 = lid.x; e < 1024u; e = e + 32u) {
        let nt = e / 256u; let rem = e % 256u; let r = rem / 16u; let c = rem % 16u;
        let co = n0 + nt * 16u + c;
        if (valid[r] == 1u && co < p.n) {
            var val = tC[e] + f32(bias[co]);
            val = select(f32(slope[co]) * val, val, val > 0.0);
            outp[(co * p.hout + u32(oys[r])) * p.wout + u32(oxs[r])] = f16(val);
        }
    }
}
