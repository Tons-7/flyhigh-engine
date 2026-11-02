enable f16;
enable wgpu_cooperative_matrix;
// Tensor-Core implicit-GEMM 3x3 stride-1 conv with NO gather. The input arrives pre-shifted (xshift3 layout
// [3][cinp][h+2][w]), so for one K-step (one tap, 16 consecutive channels) and 16 consecutive pixels of one
// image row the A tile is a column-major 16x16 window of global memory (column stride = one channel plane)
// loaded with a single coopLoad. B is loaded straight from global as well: no shared memory and no barriers
// in the K loop. Workgroup = one subgroup = 16 pixels x 64 output channels (4 accumulators).
// Requirements (checked by the host): stride 1, wout % 16 == 0 (a tile never crosses a row, m % 16 == 0),
// K tap-major with cin padded to 16 (kcol = tap*cinp + ci). The bias + PReLU epilogue is fused (slope 1 = identity).
struct P { cin: u32, h: u32, w: u32, hout: u32, wout: u32, stride: u32, pad: u32, k: u32, kp: u32, n: u32, np: u32, m: u32, cinp: u32, ksplit: u32, residual: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read> sh: array<f16>;      // shifted input [3][cinp][h+2][w]
@group(0) @binding(1) var<storage, read> Bw: array<f16>;      // [kp][np] row-major, tap-major K
@group(0) @binding(2) var<storage, read> zero: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f16>;
@group(0) @binding(4) var<storage, read> slope: array<f16>;
@group(0) @binding(5) var<storage, read_write> outp: array<f16>;
@group(0) @binding(6) var<uniform> p: P;

var<workgroup> tC: array<f32, 1024>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let m0 = wid.y * 16u;
    let n0 = wid.x * 64u;
    let oy = m0 / p.wout; let x0 = m0 % p.wout;
    let plane = (p.h + 2u) * p.w;
    let pz = &zero[0];
    var a0 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a1 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a2 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a3 = coopLoad<coop_mat16x16<f32, C>>(pz);
    let kt_count = p.kp / 16u;
    for (var kt: u32 = 0u; kt < kt_count; kt = kt + 1u) {
        let kbase = kt * 16u;
        let tap = kbase / p.cinp;
        let ci0 = kbase - tap * p.cinp;
        let ky = tap / 3u; let kx = tap % 3u;
        // A[r][c] = sh[(kx*cinp + ci0 + c) * plane + (oy + ky) * w + x0 + r]  -> column-major, stride = plane
        let pa = &sh[(kx * p.cinp + ci0) * plane + (oy + ky) * p.w + x0];
        let a = coopLoad<coop_mat16x16<f16, A>>(pa, plane);
        let kb = kbase * p.np + n0;
        let pb0 = &Bw[kb];       a0 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(pb0, p.np), a0);
        let pb1 = &Bw[kb + 16u]; a1 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(pb1, p.np), a1);
        let pb2 = &Bw[kb + 32u]; a2 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(pb2, p.np), a2);
        let pb3 = &Bw[kb + 48u]; a3 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(pb3, p.np), a3);
    }
    let q0 = &tC[0];   coopStoreT(a0, q0, 16u);
    let q1 = &tC[256]; coopStoreT(a1, q1, 16u);
    let q2 = &tC[512]; coopStoreT(a2, q2, 16u);
    let q3 = &tC[768]; coopStoreT(a3, q3, 16u);
    workgroupBarrier();
    for (var e: u32 = lid.x; e < 1024u; e = e + 32u) {
        let nt = e / 256u; let rem = e % 256u; let r = rem / 16u; let c = rem % 16u;
        let co = n0 + nt * 16u + c;
        if (co < p.n) {   // every pixel of the tile is valid: m % 16 == 0
            var val = tC[e] + f32(bias[co]);
            val = select(f32(slope[co]) * val, val, val > 0.0);
            outp[(co * p.hout + oy) * p.wout + x0 + r] = f16(val);
        }
    }
}
