enable f16;
// Finish a split-K conv (conv_tiled*.wgsl with p.ksplit > 1): sum the f32 partials [z][n][m], add bias, PReLU, f16 out.
// One thread per 4 consecutive outputs (vec4 loads; n*m is a multiple of 4 whenever w is).
struct P { n: u32, m: u32, z: u32, res_on: u32 };
@group(0) @binding(0) var<storage, read> part: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bias: array<f16>;
@group(0) @binding(2) var<storage, read> slope: array<f16>;
@group(0) @binding(3) var<storage, read_write> outp: array<f16>;   // [n][m]
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> res: array<f16>;         // optional fused residual (res_on == 1)
fn fin(r: f32, idx: u32) -> f16 { let h = f16(r); if (p.res_on == 1u) { return f16(f32(h) + f32(res[idx])); } return h; }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x; let total4 = (p.n * p.m) / 4u;
    if (q >= total4) { return; }
    var s = part[q];
    for (var z: u32 = 1u; z < p.z; z = z + 1u) { s = s + part[z * total4 + q]; }
    let i = q * 4u;
    let co = i / p.m;                       // 4 consecutive m never cross a channel when m % 4 == 0
    let v = s + vec4<f32>(f32(bias[co]));
    let r = select(f32(slope[co]) * v, v, v > vec4<f32>(0.0));
    outp[i] = fin(r.x, i); outp[i + 1u] = fin(r.y, i + 1u); outp[i + 2u] = fin(r.z, i + 2u); outp[i + 3u] = fin(r.w, i + 3u);
}
