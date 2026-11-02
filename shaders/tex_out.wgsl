enable f16;
// Rendered frame [3][hp][wp] f16 (BGR) -> storage texture (the canvas' current texture or an offscreen target), cropped to
// w x h, alpha 1. The host derives a bgra8unorm variant of this source for canvases that prefer BGRA.
struct P { w: u32, h: u32, wp: u32, hp: u32 };
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= p.w || y >= p.h) { return; }
    let plane = p.hp * p.wp; let o = y * p.wp + x;
    let c = vec4<f32>(clamp(f32(inp[2u * plane + o]), 0.0, 1.0), clamp(f32(inp[plane + o]), 0.0, 1.0), clamp(f32(inp[o]), 0.0, 1.0), 1.0);
    textureStore(dst, vec2<i32>(i32(x), i32(y)), c);
}
