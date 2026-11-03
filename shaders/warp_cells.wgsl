// Camera warp, a frame's luma on a gw x gh grid of cells (warp/gpu.rs), 2 x 2 bilinear samples per cell: what the
// shift search compares. The grid is even in angle, not in pixels: cell columns step the azimuth and rows the
// elevation of a camera with focal length f, so a turn of the camera is the same shift of cells everywhere (on a flat
// grid the edges of a wide view move nearly three times as far as the centre). Cells whose samples fall outside the
// frame (the corners) are -1 and skipped. Kept per frame, so each pair only pays for its newer frame.
struct P { gw: u32, gh: u32, w: u32, h: u32, f: f32, ha: f32, he: f32, _a: f32 };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read_write> cells: array<f32>;   // [gh][gw]
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.gw || gid.y >= p.gh) { return; }
    let size = vec2<f32>(f32(p.w), f32(p.h));
    let g = vec2<f32>(f32(p.gw), f32(p.gh));
    var s = 0.0;
    var outside = false;
    for (var j = 0u; j < 2u; j++) { for (var i = 0u; i < 2u; i++) {
        let u = (vec2<f32>(gid.xy) + vec2<f32>(0.25 + 0.5 * f32(i), 0.25 + 0.5 * f32(j))) / g;
        let az = (2.0 * u.x - 1.0) * p.ha;
        let el = (2.0 * u.y - 1.0) * p.he;
        let px = 0.5 * size + p.f * vec2<f32>(tan(az), tan(el) / cos(az));
        outside = outside || any(px < vec2<f32>(0.0)) || any(px > size);
        s += dot(textureSampleLevel(src, samp, px / size, 0.0).rgb, vec3<f32>(0.299, 0.587, 0.114));
    } }
    cells[gid.y * p.gw + gid.x] = select(0.25 * s, -1.0, outside);
}
