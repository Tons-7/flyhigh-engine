enable f16;
// FiLM modulate: out = h * (1 + sc) + bi, sc = g[c], bi = g[c + cr]. g is [2*cr]. f16 I/O, f32 math.
struct Par { c: u32, h: u32, w: u32 };
@group(0) @binding(0) var<storage, read> hh: array<f16>;
@group(0) @binding(1) var<storage, read> g: array<f16>;
@group(0) @binding(2) var<storage, read_write> outp: array<f16>;
@group(0) @binding(3) var<uniform> pp: Par;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y; let c = gid.z;
    if (x >= pp.w || y >= pp.h || c >= pp.c) { return; }
    let i = (c * pp.h + y) * pp.w + x;
    outp[i] = f16(f32(hh[i]) * (1.0 + f32(g[c])) + f32(g[c + pp.c]));
}
