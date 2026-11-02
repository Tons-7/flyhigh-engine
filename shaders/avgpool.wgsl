enable f16;
// Average pooling, kernel k, stride k. f16 I/O, f32 accumulate.
struct Par { c: u32, h: u32, w: u32, oh: u32, ow: u32, k: u32 };
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read_write> outp: array<f16>;
@group(0) @binding(2) var<uniform> pp: Par;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x; let oy = gid.y; let c = gid.z;
    if (ox >= pp.ow || oy >= pp.oh || c >= pp.c) { return; }
    var acc: f32 = 0.0;
    for (var ky: u32 = 0u; ky < pp.k; ky = ky + 1u) {
        for (var kx: u32 = 0u; kx < pp.k; kx = kx + 1u) {
            acc = acc + f32(inp[(c * pp.h + oy * pp.k + ky) * pp.w + ox * pp.k + kx]);
        }
    }
    outp[(c * pp.oh + oy) * pp.ow + ox] = f16(acc / f32(pp.k * pp.k));
}
