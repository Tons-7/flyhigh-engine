enable f16;
// Cadence composite: out = bg * (1 - m) + elem * m over the 3 image channels of two rendered frames ([3][hp][wp] f16),
// with the element mask m (f32, cost grid hc x wc covering the network frame) sampled bilinearly at the full-frame
// pixel's position in the network frame.
struct Par { hp: u32, wp: u32, hc: u32, wc: u32, sx: f32, sy: f32, w: u32, h: u32 };   // sx, sy: cost-grid cells per full-frame px
@group(0) @binding(0) var<storage, read> bg: array<f16>;
@group(0) @binding(1) var<storage, read> elem: array<f16>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> pp: Par;

fn clampi(v: i32, lo: i32, hi: i32) -> i32 { return max(lo, min(v, hi)); }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= pp.w || y >= pp.h) { return; }
    let cx = (f32(x) + 0.5) * pp.sx - 0.5; let cy = (f32(y) + 0.5) * pp.sy - 0.5;
    let x0 = i32(floor(cx)); let y0 = i32(floor(cy)); let fx = cx - f32(x0); let fy = cy - f32(y0);
    let x0c = u32(clampi(x0, 0, i32(pp.wc) - 1)); let x1c = u32(clampi(x0 + 1, 0, i32(pp.wc) - 1));
    let y0c = u32(clampi(y0, 0, i32(pp.hc) - 1)); let y1c = u32(clampi(y0 + 1, 0, i32(pp.hc) - 1));
    let top = mask[y0c * pp.wc + x0c] * (1.0 - fx) + mask[y0c * pp.wc + x1c] * fx;
    let bot = mask[y1c * pp.wc + x0c] * (1.0 - fx) + mask[y1c * pp.wc + x1c] * fx;
    let m = clamp(top * (1.0 - fy) + bot * fy, 0.0, 1.0);
    let plane = pp.hp * pp.wp; let i = y * pp.wp + x;
    // Preserve the old planar buffer's precision rounding before RGBA8 quantization.
    var bgr: vec3<f16>;
    for (var c: u32 = 0u; c < 3u; c = c + 1u) {
        let o = c * plane + i;
        bgr[c] = f16(f32(bg[o]) * (1.0 - m) + f32(elem[o]) * m);
    }
    let rgba = vec4<f32>(clamp(vec3<f32>(bgr.zyx), vec3(0.0), vec3(1.0)), 1.0);
    textureStore(dst, vec2<i32>(i32(x), i32(y)), rgba);
}
