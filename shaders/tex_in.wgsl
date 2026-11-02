enable f16;
// GPU texture (rgba8unorm, e.g. a video frame copied with copyExternalImageToTexture) -> model input [3][hp][wp] f16,
// replicate-padded to the 32-aligned working size, BGR channel order (the training order, see rgba_in.wgsl).
// No CPU round trip.
struct P { w: u32, h: u32, wp: u32, hp: u32 };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f16>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= p.wp || y >= p.hp) { return; }
    let c = textureLoad(src, vec2<i32>(i32(min(x, p.w - 1u)), i32(min(y, p.h - 1u))), 0);
    let plane = p.hp * p.wp; let o = y * p.wp + x;
    outp[o] = f16(c.b);
    outp[plane + o] = f16(c.g);
    outp[2u * plane + o] = f16(c.r);
}
