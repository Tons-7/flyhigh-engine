enable f16;
// im2col: input [Cin,H,W] -> A [M = Hout*Wout, K = Cin*9] row-major, for a 3x3 conv.
// A[m, ci*9 + ky*3 + kx] = input[ci, oy*stride-pad+ky, ox*stride-pad+kx] (0 if out of bounds).
struct P { cin: u32, h: u32, w: u32, hout: u32, wout: u32, stride: u32, pad: u32, k: u32 };
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read_write> outA: array<f16>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x; let oy = gid.y;
    if (ox >= p.wout || oy >= p.hout) { return; }
    let m = oy * p.wout + ox;
    for (var ci: u32 = 0u; ci < p.cin; ci = ci + 1u) {
        for (var ky: u32 = 0u; ky < 3u; ky = ky + 1u) {
            for (var kx: u32 = 0u; kx < 3u; kx = kx + 1u) {
                let yi = i32(oy * p.stride + ky) - i32(p.pad);
                let xi = i32(ox * p.stride + kx) - i32(p.pad);
                var v: f16 = f16(0.0);
                if (yi >= 0 && yi < i32(p.h) && xi >= 0 && xi < i32(p.w)) {
                    v = inp[(ci * p.h + u32(yi)) * p.w + u32(xi)];
                }
                outA[m * p.k + ci * 9u + ky * 3u + kx] = v;
            }
        }
    }
}
