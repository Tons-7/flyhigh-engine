enable f16;
// Rendered frame [3][hp][wp] f16 (BGR, see rgba_in.wgsl) -> RGBA8 [h][w] u32 (crop of the padded working size,
// alpha 255), ready for putImageData / texture upload.
struct P { w: u32, h: u32, wp: u32, hp: u32 };
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read_write> outp: array<u32>;
@group(0) @binding(2) var<uniform> p: P;

fn q(v: f16) -> u32 { return u32(clamp(f32(v), 0.0, 1.0) * 255.0 + 0.5); }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= p.w || y >= p.h) { return; }
    let plane = p.hp * p.wp; let o = y * p.wp + x;
    outp[y * p.w + x] = q(inp[2u * plane + o]) | (q(inp[plane + o]) << 8u) | (q(inp[o]) << 16u) | 0xff000000u;
}
