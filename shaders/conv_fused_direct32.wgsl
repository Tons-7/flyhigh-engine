enable f16;
enable wgpu_cooperative_matrix;
// Direct (no-gather) Tensor-Core conv, register-blocked in M: one subgroup computes 32 pixels x 64 output
// channels with 8 accumulators. Per K-step it loads 2 A tiles (two 16-pixel halves of the same image row)
// and 4 B tiles, and issues 8 MMAs, so each B tile is reused twice and the coop-load issue rate (the
// measured floor of the 16-pixel kernel: 5 loads per 4 MMAs) drops to 6 loads per 8 MMAs.
// Same layout contract as conv_fused_direct.wgsl: shifted input [3][cinp][h+2][w], K tap-major with cin padded
// to 16, B [kp][np] row-major. Requires stride 1 and wout % 32 == 0 (a 32-pixel tile never crosses a row).
struct P { cin: u32, h: u32, w: u32, hout: u32, wout: u32, stride: u32, pad: u32, k: u32, kp: u32, n: u32, np: u32, m: u32, cinp: u32, ksplit: u32, residual: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read> sh: array<f16>;
@group(0) @binding(1) var<storage, read> Bw: array<f16>;
@group(0) @binding(2) var<storage, read> zero: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f16>;
@group(0) @binding(4) var<storage, read> slope: array<f16>;
@group(0) @binding(5) var<storage, read_write> outp: array<f16>;
@group(0) @binding(6) var<uniform> p: P;

var<workgroup> tC: array<f32, 2048>;   // [half][nt][16][16]

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let m0 = wid.y * 32u;
    let n0 = wid.x * 64u;
    // each 16-pixel half gets its own (row, x): the two halves may sit in different rows, so only wout % 16 == 0
    // (and m % 32 == 0) is required, not wout % 32
    let oy0 = m0 / p.wout; let x0 = m0 % p.wout;
    let oy1 = (m0 + 16u) / p.wout; let x1 = (m0 + 16u) % p.wout;
    let plane = (p.h + 2u) * p.w;
    let pz = &zero[0];
    var a00 = coopLoad<coop_mat16x16<f32, C>>(pz); var a01 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a02 = coopLoad<coop_mat16x16<f32, C>>(pz); var a03 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a10 = coopLoad<coop_mat16x16<f32, C>>(pz); var a11 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a12 = coopLoad<coop_mat16x16<f32, C>>(pz); var a13 = coopLoad<coop_mat16x16<f32, C>>(pz);
    let kt_count = p.kp / 16u;
    for (var kt: u32 = 0u; kt < kt_count; kt = kt + 1u) {
        let kbase = kt * 16u;
        let tap = kbase / p.cinp;
        let ci0 = kbase - tap * p.cinp;
        let ky = tap / 3u; let kx = tap % 3u;
        let cbase = (kx * p.cinp + ci0) * plane;
        let pa0 = &sh[cbase + (oy0 + ky) * p.w + x0]; let A0 = coopLoad<coop_mat16x16<f16, A>>(pa0, plane);
        let pa1 = &sh[cbase + (oy1 + ky) * p.w + x1]; let A1 = coopLoad<coop_mat16x16<f16, A>>(pa1, plane);
        let kb = kbase * p.np + n0;
        let pb0 = &Bw[kb];       let B0 = coopLoadT<coop_mat16x16<f16, B>>(pb0, p.np);
        a00 = coopMultiplyAdd(A0, B0, a00); a10 = coopMultiplyAdd(A1, B0, a10);
        let pb1 = &Bw[kb + 16u]; let B1 = coopLoadT<coop_mat16x16<f16, B>>(pb1, p.np);
        a01 = coopMultiplyAdd(A0, B1, a01); a11 = coopMultiplyAdd(A1, B1, a11);
        let pb2 = &Bw[kb + 32u]; let B2 = coopLoadT<coop_mat16x16<f16, B>>(pb2, p.np);
        a02 = coopMultiplyAdd(A0, B2, a02); a12 = coopMultiplyAdd(A1, B2, a12);
        let pb3 = &Bw[kb + 48u]; let B3 = coopLoadT<coop_mat16x16<f16, B>>(pb3, p.np);
        a03 = coopMultiplyAdd(A0, B3, a03); a13 = coopMultiplyAdd(A1, B3, a13);
    }
    let q00 = &tC[0];    coopStoreT(a00, q00, 16u); let q01 = &tC[256];  coopStoreT(a01, q01, 16u);
    let q02 = &tC[512];  coopStoreT(a02, q02, 16u); let q03 = &tC[768];  coopStoreT(a03, q03, 16u);
    let q10 = &tC[1024]; coopStoreT(a10, q10, 16u); let q11 = &tC[1280]; coopStoreT(a11, q11, 16u);
    let q12 = &tC[1536]; coopStoreT(a12, q12, 16u); let q13 = &tC[1792]; coopStoreT(a13, q13, 16u);
    workgroupBarrier();
    for (var e: u32 = lid.x; e < 2048u; e = e + 32u) {
        let half = e / 1024u; let rem1 = e % 1024u;
        let nt = rem1 / 256u; let rem = rem1 % 256u; let r = rem / 16u; let c = rem % 16u;
        let co = n0 + nt * 16u + c;
        if (co < p.n) {
            var val = tC[e] + f32(bias[co]);
            val = select(f32(slope[co]) * val, val, val > 0.0);
            let oy = select(oy0, oy1, half == 1u); let xs = select(x0, x1, half == 1u);
            outp[(co * p.hout + oy) * p.wout + xs + r] = f16(val);
        }
    }
}
