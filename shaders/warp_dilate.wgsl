// Camera warp, the mask the render reads (warp/gpu.rs): the screen-locked blocks grown by one block each way, so the
// render's soft edge falls outside the weapon rather than across it. Once per measured frame instead of per pixel.
struct P { gw: u32, gh: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> mask: array<f32>;
@group(0) @binding(1) var<storage, read_write> grown: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.gw || gid.y >= p.gh) { return; }
    var m = 0.0;
    for (var j = -1; j <= 1; j++) { for (var i = -1; i <= 1; i++) {
        let x = clamp(i32(gid.x) + i, 0, i32(p.gw) - 1); let y = clamp(i32(gid.y) + j, 0, i32(p.gh) - 1);
        m = max(m, mask[u32(y) * p.gw + u32(x)]);
    } }
    grown[gid.y * p.gw + gid.x] = m;
}
