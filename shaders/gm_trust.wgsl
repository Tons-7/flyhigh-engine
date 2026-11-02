// Cadence trust mask: the element mask (or the whole frame on a still shot, `full`) minus the cells where the transition
// pair's model could not match the frames (a disocclusion: an arm appearing from behind the body). There the re-timed
// render is a guess, so the plain interpolation of the current interval, which shows the real drawing, is kept instead.
struct Par { n: u32, full: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> elem: array<f32>;
@group(0) @binding(1) var<storage, read> bad: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> pp: Par;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pp.n) { return; }
    let e = select(elem[i], 1.0, pp.full != 0u);
    dst[i] = e * (1.0 - clamp(bad[i], 0.0, 1.0));
}
