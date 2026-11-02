enable f16;
// BGR -> luma (entry `luma`) and edge magnitude of luma (entry `grad`). 3ch in, 1ch out. f16 I/O.
struct Par { h: u32, w: u32, c0: u32, scale: f32 };   // the RGB input may be a view (channel offset c0, scale)
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read_write> outp: array<f16>;
@group(0) @binding(2) var<uniform> pp: Par;

fn lum(y: u32, x: u32) -> f32 {
    let p = pp.h * pp.w; let i = pp.c0 * p + y * pp.w + x;
    return (0.299 * f32(inp[2u * p + i]) + 0.587 * f32(inp[1u * p + i]) + 0.114 * f32(inp[i])) * pp.scale;
}

@compute @workgroup_size(8, 8, 1)
fn luma(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= pp.w || y >= pp.h) { return; }
    outp[y * pp.w + x] = f16(lum(y, x));
}

@compute @workgroup_size(8, 8, 1)
fn grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= pp.w || y >= pp.h) { return; }
    let c = lum(y, x);
    let gx = select(0.0, lum(y, x + 1u) - c, x + 1u < pp.w);
    let gy = select(0.0, lum(y + 1u, x) - c, y + 1u < pp.h);
    outp[y * pp.w + x] = f16(abs(gx) + abs(gy));
}
