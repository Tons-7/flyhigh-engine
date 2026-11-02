enable f16;
// Elementwise unary: op 0 = scale(*a), 1 = clamp(a,b), 2 = affine_out(2*sigmoid-1), 3 = neg.
struct Par { c: u32, h: u32, w: u32, op: u32, a: f32, b: f32, c0: u32, scale: f32 };   // input may be a view (c0, scale)
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read_write> outp: array<f16>;
@group(0) @binding(2) var<uniform> pp: Par;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y; let c = gid.z;
    if (x >= pp.w || y >= pp.h || c >= pp.c) { return; }
    let i = (c * pp.h + y) * pp.w + x;
    let v = f32(inp[((c + pp.c0) * pp.h + y) * pp.w + x]) * pp.scale;
    var r: f32;
    if (pp.op == 0u) { r = v * pp.a; }
    else if (pp.op == 1u) { r = clamp(v, pp.a, pp.b); }
    else if (pp.op == 2u) { r = 2.0 / (1.0 + exp(-v)) - 1.0; }
    else { r = -v; }
    outp[i] = f16(r);
}
