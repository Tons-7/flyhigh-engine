// Camera warp, the render (warp/gpu.rs): the frame turned by the camera motion since it was made. Each output pixel's
// ray is turned back by the yaw and pitch of the turn (sx, sy) pixels at the centre at focal length f, and the frame is
// sampled there; where that falls outside the frame (the strip the camera turned towards) the clamped edge is
// blurred; blocks locked to the screen (the grown mask, softened) keep their own pixels.
struct P { w: u32, h: u32, gw: u32, gh: u32, sx: f32, sy: f32, f: f32, _a: f32 };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read> mask: array<f32>;   // warp_dilate.wgsl
@group(0) @binding(3) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> p: P;

fn locked(cx: i32, cy: i32) -> f32 {
    let x = clamp(cx, 0, i32(p.gw) - 1); let y = clamp(cy, 0, i32(p.gh) - 1);
    return mask[u32(y) * p.gw + u32(x)];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) { return; }
    let size = vec2<f32>(f32(p.w), f32(p.h));
    let c = 0.5 * size;
    let px = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    // camera space: x right, y down, z forward
    // the calibrated gain is pixels at the centre per count over small steps, so the angle grows linearly with the
    // counts: a 300 px turn is 300 / f radians, not atan(300 / f)
    let ty = p.sx / p.f; let tp = p.sy / p.f;
    var d = vec3<f32>((px - c) / p.f, 1.0);
    let cy = cos(ty); let sy = sin(ty);
    d = vec3<f32>(d.x * cy - d.z * sy, d.y, d.x * sy + d.z * cy);
    let cp = cos(tp); let sp = sin(tp);
    d = vec3<f32>(d.x, d.y * cp - d.z * sp, d.y * sp + d.z * cp);
    let s = c + p.f * d.xy / max(d.z, 1e-3);
    var col: vec3<f32>;
    if (d.z > 1e-3 && all(s >= vec2<f32>(0.0)) && all(s <= size)) {
        col = textureSampleLevel(src, samp, s / size, 0.0).rgb;
    } else {
        let e = clamp(s, vec2<f32>(0.0), size);
        col = vec3<f32>(0.0);
        for (var j = -2; j <= 2; j++) { for (var i = -2; i <= 2; i++) {
            col += textureSampleLevel(src, samp, clamp(e + 6.0 * vec2<f32>(f32(i), f32(j)), vec2<f32>(0.0), size) / size, 0.0).rgb;
        } }
        col /= 25.0;
    }
    // the grown mask, interpolated between block centres
    let g = px / 8.0 - 0.5;
    let g0 = vec2<i32>(floor(g)); let fr = g - floor(g);
    let m = mix(mix(locked(g0.x, g0.y), locked(g0.x + 1, g0.y), fr.x),
                mix(locked(g0.x, g0.y + 1), locked(g0.x + 1, g0.y + 1), fr.x), fr.y);
    let own = textureSampleLevel(src, samp, px / size, 0.0).rgb;
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(mix(col, own, m), 1.0));
}
