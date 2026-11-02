enable f16;
enable wgpu_cooperative_matrix;
// conv_fused_direct32.wgsl blocked twice as deep in M: one subgroup computes 64 pixels (four 16-pixel quarters, each
// with its own row/x) x 64 output channels with 16 accumulators; per K-step 4 A + 4 B loads feed 16 MMAs (0.5 loads
// per MMA vs 0.75 in the 32-px kernel). Quarters past the last pixel (m % 64 != 0) read row 0 and are not stored.
// Same layout contract: shifted input [3][cinp][h+2][w], tap-major K (cin padded to 16), B [kp][np] row-major.
struct P { cin: u32, h: u32, w: u32, hout: u32, wout: u32, stride: u32, pad: u32, k: u32, kp: u32, n: u32, np: u32, m: u32, cinp: u32, ksplit: u32, residual: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read> sh: array<f16>;
@group(0) @binding(1) var<storage, read> Bw: array<f16>;
@group(0) @binding(2) var<storage, read> zero: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f16>;
@group(0) @binding(4) var<storage, read> slope: array<f16>;
@group(0) @binding(5) var<storage, read_write> outp: array<f16>;
@group(0) @binding(6) var<uniform> p: P;

var<workgroup> tC: array<f32, 4096>;   // [quarter][nt][16][16]

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let m0 = wid.y * 64u;
    let n0 = wid.x * 64u;
    let v0 = m0 < p.m; let v1 = m0 + 16u < p.m; let v2 = m0 + 32u < p.m; let v3 = m0 + 48u < p.m;
    let oy0 = select(0u, m0 / p.wout, v0);         let x0 = select(0u, m0 % p.wout, v0);
    let oy1 = select(0u, (m0 + 16u) / p.wout, v1); let x1 = select(0u, (m0 + 16u) % p.wout, v1);
    let oy2 = select(0u, (m0 + 32u) / p.wout, v2); let x2 = select(0u, (m0 + 32u) % p.wout, v2);
    let oy3 = select(0u, (m0 + 48u) / p.wout, v3); let x3 = select(0u, (m0 + 48u) % p.wout, v3);
    let plane = (p.h + 2u) * p.w;
    let pz = &zero[0];
    var a00 = coopLoad<coop_mat16x16<f32, C>>(pz); var a01 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a02 = coopLoad<coop_mat16x16<f32, C>>(pz); var a03 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a10 = coopLoad<coop_mat16x16<f32, C>>(pz); var a11 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a12 = coopLoad<coop_mat16x16<f32, C>>(pz); var a13 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a20 = coopLoad<coop_mat16x16<f32, C>>(pz); var a21 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a22 = coopLoad<coop_mat16x16<f32, C>>(pz); var a23 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a30 = coopLoad<coop_mat16x16<f32, C>>(pz); var a31 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a32 = coopLoad<coop_mat16x16<f32, C>>(pz); var a33 = coopLoad<coop_mat16x16<f32, C>>(pz);
    let kt_count = p.kp / 16u;
    for (var kt: u32 = 0u; kt < kt_count; kt = kt + 1u) {
        let kbase = kt * 16u;
        let tap = kbase / p.cinp;
        let ci0 = kbase - tap * p.cinp;
        let ky = tap / 3u; let kx = tap % 3u;
        let cbase = (kx * p.cinp + ci0) * plane;
        let pa0 = &sh[cbase + (oy0 + ky) * p.w + x0]; let A0 = coopLoad<coop_mat16x16<f16, A>>(pa0, plane);
        let pa1 = &sh[cbase + (oy1 + ky) * p.w + x1]; let A1 = coopLoad<coop_mat16x16<f16, A>>(pa1, plane);
        let pa2 = &sh[cbase + (oy2 + ky) * p.w + x2]; let A2 = coopLoad<coop_mat16x16<f16, A>>(pa2, plane);
        let pa3 = &sh[cbase + (oy3 + ky) * p.w + x3]; let A3 = coopLoad<coop_mat16x16<f16, A>>(pa3, plane);
        let kb = kbase * p.np + n0;
        let pb0 = &Bw[kb];       let B0 = coopLoadT<coop_mat16x16<f16, B>>(pb0, p.np);
        a00 = coopMultiplyAdd(A0, B0, a00); a10 = coopMultiplyAdd(A1, B0, a10); a20 = coopMultiplyAdd(A2, B0, a20); a30 = coopMultiplyAdd(A3, B0, a30);
        let pb1 = &Bw[kb + 16u]; let B1 = coopLoadT<coop_mat16x16<f16, B>>(pb1, p.np);
        a01 = coopMultiplyAdd(A0, B1, a01); a11 = coopMultiplyAdd(A1, B1, a11); a21 = coopMultiplyAdd(A2, B1, a21); a31 = coopMultiplyAdd(A3, B1, a31);
        let pb2 = &Bw[kb + 32u]; let B2 = coopLoadT<coop_mat16x16<f16, B>>(pb2, p.np);
        a02 = coopMultiplyAdd(A0, B2, a02); a12 = coopMultiplyAdd(A1, B2, a12); a22 = coopMultiplyAdd(A2, B2, a22); a32 = coopMultiplyAdd(A3, B2, a32);
        let pb3 = &Bw[kb + 48u]; let B3 = coopLoadT<coop_mat16x16<f16, B>>(pb3, p.np);
        a03 = coopMultiplyAdd(A0, B3, a03); a13 = coopMultiplyAdd(A1, B3, a13); a23 = coopMultiplyAdd(A2, B3, a23); a33 = coopMultiplyAdd(A3, B3, a33);
    }
    let q00 = &tC[0];    coopStoreT(a00, q00, 16u); let q01 = &tC[256];  coopStoreT(a01, q01, 16u);
    let q02 = &tC[512];  coopStoreT(a02, q02, 16u); let q03 = &tC[768];  coopStoreT(a03, q03, 16u);
    let q10 = &tC[1024]; coopStoreT(a10, q10, 16u); let q11 = &tC[1280]; coopStoreT(a11, q11, 16u);
    let q12 = &tC[1536]; coopStoreT(a12, q12, 16u); let q13 = &tC[1792]; coopStoreT(a13, q13, 16u);
    let q20 = &tC[2048]; coopStoreT(a20, q20, 16u); let q21 = &tC[2304]; coopStoreT(a21, q21, 16u);
    let q22 = &tC[2560]; coopStoreT(a22, q22, 16u); let q23 = &tC[2816]; coopStoreT(a23, q23, 16u);
    let q30 = &tC[3072]; coopStoreT(a30, q30, 16u); let q31 = &tC[3328]; coopStoreT(a31, q31, 16u);
    let q32 = &tC[3584]; coopStoreT(a32, q32, 16u); let q33 = &tC[3840]; coopStoreT(a33, q33, 16u);
    workgroupBarrier();
    for (var e: u32 = lid.x; e < 4096u; e = e + 32u) {
        let q = e / 1024u; let rem1 = e % 1024u;
        let nt = rem1 / 256u; let rem = rem1 % 256u; let r = rem / 16u; let c = rem % 16u;
        let co = n0 + nt * 16u + c;
        let valid = select(select(v3, v2, q == 2u), select(v1, v0, q == 0u), q < 2u);
        if (co < p.n && valid) {
            var val = tC[e] + f32(bias[co]);
            val = select(f32(slope[co]) * val, val, val > 0.0);
            let oy = select(select(oy3, oy2, q == 2u), select(oy1, oy0, q == 0u), q < 2u);
            let xs = select(select(x3, x2, q == 2u), select(x1, x0, q == 0u), q < 2u);
            outp[(co * p.hout + oy) * p.wout + xs + r] = f16(val);
        }
    }
}
