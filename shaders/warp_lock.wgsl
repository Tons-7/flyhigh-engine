// Camera warp, the screen-locked mask (warp/gpu.rs): per 8 x 8 px block, whether the newest pair is explained better
// standing still (a weapon in hand, a HUD, a cockpit) than turned by the pair's own camera motion (warp_best.wgsl).
// Those blocks are not warped. Written only after a pair with a valid shift of at least 4 px: a still pair says
// nothing, and the mask it would replace stays.
struct P { w: u32, h: u32, gw: u32, gh: u32, margin: f32, rel: f32, f: f32, _b: f32 };
@group(0) @binding(0) var tex_a: texture_2d<f32>;
@group(0) @binding(1) var tex_b: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> mask: array<f32>;   // [gh][gw], 1 = locked to the screen
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> best: array<f32>;

// Where frame A had the content that a camera turned by `s` (pixels at the centre, warp_render.wgsl) shows at `px`.
fn turned_back(px: vec2<f32>, s: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    let c = 0.5 * size;
    let ty = s.x / p.f; let tp = s.y / p.f;
    var d = vec3<f32>((px - c) / p.f, 1.0);
    d = vec3<f32>(d.x * cos(ty) - d.z * sin(ty), d.y, d.x * sin(ty) + d.z * cos(ty));
    d = vec3<f32>(d.x, d.y * cos(tp) - d.z * sin(tp), d.y * sin(tp) + d.z * cos(tp));
    return c + p.f * d.xy / max(d.z, 1e-3);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.gw || gid.y >= p.gh) { return; }
    let shift = vec2<f32>(best[0], best[1]);
    if (best[2] < 0.5 || length(shift) < 4.0) { return; }
    let size = vec2<f32>(f32(p.w), f32(p.h));
    var e_still = 0.0; var e_moved = 0.0;
    for (var j = 0u; j < 4u; j++) { for (var i = 0u; i < 4u; i++) {
        let px = vec2<f32>(f32(gid.x * 8u + i * 2u) + 1.0, f32(gid.y * 8u + j * 2u) + 1.0);
        let b = textureSampleLevel(tex_b, samp, px / size, 0.0).rgb;
        let a = textureSampleLevel(tex_a, samp, px / size, 0.0).rgb;
        let am = textureSampleLevel(tex_a, samp, turned_back(px, shift, size) / size, 0.0).rgb;
        e_still += dot(abs(a - b), vec3<f32>(1.0 / 3.0));
        e_moved += dot(abs(am - b), vec3<f32>(1.0 / 3.0));
    } }
    e_still /= 16.0; e_moved /= 16.0;
    mask[gid.y * p.gw + gid.x] = select(0.0, 1.0, e_moved - e_still > max(p.margin, p.rel * e_moved));
}
