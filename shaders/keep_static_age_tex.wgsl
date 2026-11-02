enable f16;
// Static protection, age pass on the fast render path: as keep_static_age.wgsl, reading the pair's two source textures
// (replicate-padded to the working size) instead of full-resolution tensors.
struct P { w: u32, h: u32, wp: u32, hp: u32, thr: f32, min_age: f32, use_mask: u32, _b: u32 };
@group(0) @binding(0) var tex_a: texture_2d<f32>;
@group(0) @binding(1) var tex_b: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f16>;
@group(0) @binding(3) var<uniform> p: P;
// the host's protection mask (option protection_mask): 1 where never to interpolate, read as age 255; the zero
// buffer stands in when the session has none (use_mask 0)
@group(0) @binding(4) var<storage, read> mask: array<f16>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.wp || gid.y >= p.hp) { return; }
    let t = vec2<i32>(i32(min(gid.x, p.w - 1u)), i32(min(gid.y, p.h - 1u)));
    let d = abs(textureLoad(tex_a, t, 0).rgb - textureLoad(tex_b, t, 0).rgb);
    let o = gid.y * p.wp + gid.x;
    let aged = select(f16(0.0), f16(min(f32(outp[o]) + 1.0, 255.0)), max(d.r, max(d.g, d.b)) < p.thr);
    outp[o] = select(aged, f16(255.0), p.use_mask != 0u && f32(mask[o]) > 0.5);
}
