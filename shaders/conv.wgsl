enable f16;
// 3x3 conv (stride 1 or 2, pad 1) + per-channel PReLU. f16 I/O, f32 accumulate.
// (A shared-memory tiled variant gave no clear gain over this naive one — within the ~1 ms
//  run-to-run noise. The convs are small; the bottleneck is the op-count / barrier chain,
//  not conv memory latency or bandwidth. The lever is fusion. See the run doc.)
struct Params { cin: u32, cout: u32, h: u32, w: u32, hout: u32, wout: u32, stride: u32, pad: u32, act: u32 };
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read> wgt: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read> slope: array<f16>;
@group(0) @binding(4) var<storage, read_write> outp: array<f16>;
@group(0) @binding(5) var<uniform> P: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let xo = gid.x; let yo = gid.y; let co = gid.z;
    if (xo >= P.wout || yo >= P.hout || co >= P.cout) { return; }
    var acc: f32 = f32(bias[co]);
    for (var ci: u32 = 0u; ci < P.cin; ci = ci + 1u) {
        for (var ky: u32 = 0u; ky < 3u; ky = ky + 1u) {
            for (var kx: u32 = 0u; kx < 3u; kx = kx + 1u) {
                let yi = i32(yo * P.stride + ky) - i32(P.pad);
                let xi = i32(xo * P.stride + kx) - i32(P.pad);
                if (yi >= 0 && yi < i32(P.h) && xi >= 0 && xi < i32(P.w)) {
                    let xv = f32(inp[(ci * P.h + u32(yi)) * P.w + u32(xi)]);
                    let wv = f32(wgt[((co * P.cin + ci) * 3u + ky) * 3u + kx]);
                    acc = acc + xv * wv;
                }
            }
        }
    }
    var v = acc;
    if (P.act == 1u) { let s = f32(slope[co]); v = select(s * acc, acc, acc > 0.0); }
    outp[(co * P.hout + yo) * P.wout + xo] = f16(v);
}
