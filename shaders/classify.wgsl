// Pair classification for the Stream (stream/classify.rs): the two source frames sampled on a gw x gh grid (one
// bilinear sample at each cell centre), the sum over cells of the rgb absolute difference in 1/255 steps (0..765 per
// cell) and its maximum. A scene cut has a large mean, a held frame a tiny mean and maximum. The 48 x 27 grid and
// these units are what the extension measured its thresholds on.
struct Par { gw: u32, gh: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var tex_a: texture_2d<f32>;
@group(0) @binding(1) var tex_b: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> stats: array<atomic<u32>, 2>;
@group(0) @binding(4) var<uniform> pp: Par;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= pp.gw || gid.y >= pp.gh) { return; }
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / vec2<f32>(f32(pp.gw), f32(pp.gh));
    let a = textureSampleLevel(tex_a, samp, uv, 0.0).rgb;
    let b = textureSampleLevel(tex_b, samp, uv, 0.0).rgb;
    let d = u32(dot(abs(a - b), vec3<f32>(255.0, 255.0, 255.0)));
    atomicAdd(&stats[0], d);
    atomicMax(&stats[1], d);
}
