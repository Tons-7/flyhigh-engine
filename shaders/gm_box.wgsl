// GMHS separable box filter over f32 maps (entry `box_h` along x, `box_v` along y), radius r, edge replicate.
struct Par { c: u32, h: u32, w: u32, r: u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> pp: Par;

@compute @workgroup_size(8, 8, 1)
fn box_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y; let c = gid.z;
    if (x >= pp.w || y >= pp.h || c >= pp.c) { return; }
    let row = (c * pp.h + y) * pp.w; let r = i32(pp.r); let wm = i32(pp.w) - 1;
    var acc: f32 = 0.0;
    for (var d: i32 = -r; d <= r; d = d + 1) { acc = acc + src[row + u32(max(0, min(i32(x) + d, wm)))]; }
    dst[row + x] = acc / f32(2 * r + 1);
}

@compute @workgroup_size(8, 8, 1)
fn box_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y; let c = gid.z;
    if (x >= pp.w || y >= pp.h || c >= pp.c) { return; }
    let plane = c * pp.h * pp.w; let r = i32(pp.r); let hm = i32(pp.h) - 1;
    var acc: f32 = 0.0;
    for (var d: i32 = -r; d <= r; d = d + 1) { acc = acc + src[plane + u32(max(0, min(i32(y) + d, hm))) * pp.w + x]; }
    dst[plane + y * pp.w + x] = acc / f32(2 * r + 1);
}
