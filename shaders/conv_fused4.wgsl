enable f16;
enable wgpu_cooperative_matrix;
// Tensor-Core implicit-GEMM 3x3 conv, retiled and software-pipelined. One workgroup = 128 threads = 4 subgroups,
// each owning a 16-pixel row slab of a 64-pixel M tile, all four sharing one 64-channel N tile:
//  * the B (weight) tile of a K-step is staged in shared ONCE per workgroup, loaded as one vec4<u32> (8 f16)
//    per thread instead of eight scalar loads;
//  * K is tap-major with cin padded to 16 (kcol = tap*cinp + ci), so a 16-wide K block never straddles a
//    (ky,kx) tap and the bounds test is one per K-step per pixel instead of one per gathered element;
//  * double-buffered A/B tiles: step k+1 is prefetched into registers before the MMA of step k and staged
//    after it, so one barrier per K-step and the gather latency overlaps the tensor-core work;
//  * gathers and stores map thread -> pixel, so image reads/writes along x are coalesced.
// Shared row strides are multiples of 16 bytes (coop load/store alignment) and padded to spread banks.
// The bias + PReLU epilogue is fused; a slope of 1 makes it the identity (no-activation convs).
// p.mode is the microbench ablation mask (uniform): 1 = skip A gather, 2 = skip B load, 4 = skip MMA.
struct P { cin: u32, h: u32, w: u32, hout: u32, wout: u32, stride: u32, pad: u32, k: u32, kp: u32, n: u32, np: u32, m: u32, cinp: u32, mode: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read> Bw: array<vec4<u32>>;   // [kp][np] f16 row-major, 8 f16 per element
@group(0) @binding(2) var<storage, read> zero: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f16>;
@group(0) @binding(4) var<storage, read> slope: array<f16>;
@group(0) @binding(5) var<storage, read_write> outp: array<f16>;
@group(0) @binding(6) var<uniform> p: P;

const TA_STRIDE: u32 = 24u;   // 16 k-columns + pad (48 B rows)
const TA_BUF: u32 = 1536u;    // 64 rows x 24 per buffer
const TB_BUF: u32 = 1024u;    // 16 rows x 64 per buffer
const TC_STRIDE: u32 = 68u;   // 64 channels + pad (272 B rows)
var<workgroup> tA: array<f16, 3072>;   // 2 buffers
var<workgroup> tB: array<f16, 2048>;   // 2 buffers
var<workgroup> tC: array<f32, 4352>;   // 64 rows x 68
var<workgroup> oys: array<i32, 64>;
var<workgroup> oxs: array<i32, 64>;
var<workgroup> valid: array<u32, 64>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let sg = t / 32u;                 // 32-wide subgroups packed in invocation order
    let m0 = wid.y * 64u;
    let n0 = wid.x * 64u;
    if (t < 64u) {
        let m = m0 + t;
        if (m < p.m) { oys[t] = i32(m / p.wout); oxs[t] = i32(m % p.wout); valid[t] = 1u; }
        else { oys[t] = 0; oxs[t] = 0; valid[t] = 0u; }
    }
    workgroupBarrier();
    let pz = &zero[0];
    var a0 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a1 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a2 = coopLoad<coop_mat16x16<f32, C>>(pz);
    var a3 = coopLoad<coop_mat16x16<f32, C>>(pz);
    // gather mapping: thread t owns pixel r = t % 64 and k-columns c = 2*i + t/64, i = 0..7
    let r = t % 64u;
    let chalf = t / 64u;
    let oy = oys[r]; let ox = oxs[r]; let vr = valid[r];
    let plane = p.h * p.w;
    // B mapping: thread t loads row t/8 of the 16 x 64 tile, channels (t%8)*8 .. +8 as one vec4<u32>
    let brow = t / 8u; let bcol = (t % 8u) * 8u;
    let skip_g = (p.mode & 1u) != 0u;
    let skip_b = (p.mode & 2u) != 0u;
    let skip_m = (p.mode & 4u) != 0u;
    let kt_count = p.kp / 16u;

    var rv: array<f16, 8>;
    var bv: vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);

    // prefetch + stage step 0 into buffer 0
    {
        let tap = 0u; let ci0 = 0u;
        let yi = oy * i32(p.stride) - i32(p.pad);
        let xi = ox * i32(p.stride) - i32(p.pad);
        let inb = vr == 1u && yi >= 0 && yi < i32(p.h) && xi >= 0 && xi < i32(p.w);
        let pix = u32(yi) * p.w + u32(xi);
        if (!skip_g) {
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                let ci = ci0 + 2u * i + chalf;
                var v: f16 = f16(0.0);
                if (inb && ci < p.cin) { v = inp[ci * plane + pix]; }
                rv[i] = v;
            }
        }
        if (!skip_b) { bv = Bw[((tap * 0u + brow) * p.np + n0 + bcol) / 8u]; }
        for (var i: u32 = 0u; i < 8u; i = i + 1u) { tA[r * TA_STRIDE + 2u * i + chalf] = rv[i]; }
        let f01 = unpack2x16float(bv.x); let f23 = unpack2x16float(bv.y);
        let f45 = unpack2x16float(bv.z); let f67 = unpack2x16float(bv.w);
        let bo = brow * 64u + bcol;
        tB[bo] = f16(f01.x); tB[bo + 1u] = f16(f01.y); tB[bo + 2u] = f16(f23.x); tB[bo + 3u] = f16(f23.y);
        tB[bo + 4u] = f16(f45.x); tB[bo + 5u] = f16(f45.y); tB[bo + 6u] = f16(f67.x); tB[bo + 7u] = f16(f67.y);
    }
    workgroupBarrier();

    var cur: u32 = 0u;
    for (var kt: u32 = 0u; kt < kt_count; kt = kt + 1u) {
        let nxt = cur ^ 1u;
        let has_next = kt + 1u < kt_count;
        if (has_next) {   // issue next step's loads before this step's MMA
            let kbase = (kt + 1u) * 16u;
            let tap = kbase / p.cinp;
            let ci0 = kbase - tap * p.cinp;
            let ky = i32(tap / 3u); let kx = i32(tap % 3u);
            let yi = oy * i32(p.stride) + ky - i32(p.pad);
            let xi = ox * i32(p.stride) + kx - i32(p.pad);
            let inb = vr == 1u && yi >= 0 && yi < i32(p.h) && xi >= 0 && xi < i32(p.w);
            let pix = u32(yi) * p.w + u32(xi);
            if (!skip_g) {
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    let ci = ci0 + 2u * i + chalf;
                    var v: f16 = f16(0.0);
                    if (inb && ci < p.cin) { v = inp[ci * plane + pix]; }
                    rv[i] = v;
                }
            }
            if (!skip_b) { bv = Bw[((kbase + brow) * p.np + n0 + bcol) / 8u]; }
        }
        if (!skip_m) {
            let pa = &tA[cur * TA_BUF + sg * 16u * TA_STRIDE];
            let a = coopLoadT<coop_mat16x16<f16, A>>(pa, TA_STRIDE);
            let bb = cur * TB_BUF;
            let pb0 = &tB[bb];       a0 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(pb0, 64u), a0);
            let pb1 = &tB[bb + 16u]; a1 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(pb1, 64u), a1);
            let pb2 = &tB[bb + 32u]; a2 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(pb2, 64u), a2);
            let pb3 = &tB[bb + 48u]; a3 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(pb3, 64u), a3);
        }
        if (has_next) {   // stage the prefetched registers into the other buffer
            let ab = nxt * TA_BUF;
            for (var i: u32 = 0u; i < 8u; i = i + 1u) { tA[ab + r * TA_STRIDE + 2u * i + chalf] = rv[i]; }
            let f01 = unpack2x16float(bv.x); let f23 = unpack2x16float(bv.y);
            let f45 = unpack2x16float(bv.z); let f67 = unpack2x16float(bv.w);
            let bo = nxt * TB_BUF + brow * 64u + bcol;
            tB[bo] = f16(f01.x); tB[bo + 1u] = f16(f01.y); tB[bo + 2u] = f16(f23.x); tB[bo + 3u] = f16(f23.y);
            tB[bo + 4u] = f16(f45.x); tB[bo + 5u] = f16(f45.y); tB[bo + 6u] = f16(f67.x); tB[bo + 7u] = f16(f67.y);
        }
        workgroupBarrier();
        cur = nxt;
    }
    // epilogue: every subgroup stores its 16 x 64 slab into its own tC rows, then all 128 threads write the
    // 64 x 64 tile with thread -> pixel mapping (64 contiguous pixels per channel).
    let base = sg * 16u * TC_STRIDE;
    let q0 = &tC[base];        coopStoreT(a0, q0, TC_STRIDE);
    let q1 = &tC[base + 16u];  coopStoreT(a1, q1, TC_STRIDE);
    let q2 = &tC[base + 32u];  coopStoreT(a2, q2, TC_STRIDE);
    let q3 = &tC[base + 48u];  coopStoreT(a3, q3, TC_STRIDE);
    workgroupBarrier();
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        let e = i * 128u + t;
        let rr = e % 64u; let cc = e / 64u;
        let co = n0 + cc;
        if (valid[rr] == 1u && co < p.n) {
            var val = tC[rr * TC_STRIDE + cc] + f32(bias[co]);
            val = select(f32(slope[co]) * val, val, val > 0.0);
            outp[(co * p.hout + u32(oys[rr])) * p.wout + u32(oxs[rr])] = f16(val);
        }
    }
}
