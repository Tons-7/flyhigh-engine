enable f16;
// Elementwise binary: op 0 = add, op 1 = mul. cb==1 broadcasts b over all channels of a.
struct Par { c: u32, h: u32, w: u32, cb: u32, op: u32, a0: u32, b0: u32, _a: u32, sa: f32, sb: f32, _b: u32, _c: u32 };   // a, b may be views
@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> outp: array<f16>;
@group(0) @binding(3) var<uniform> pp: Par;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y; let c = gid.z;
    if (x >= pp.w || y >= pp.h || c >= pp.c) { return; }
    let i = (c * pp.h + y) * pp.w + x;
    let cb = select(c, 0u, pp.cb == 1u);
    let av = f32(a[((c + pp.a0) * pp.h + y) * pp.w + x]) * pp.sa; let bv = f32(b[((cb + pp.b0) * pp.h + y) * pp.w + x]) * pp.sb;
    outp[i] = f16(select(av + bv, av * bv, pp.op == 1u));
}
