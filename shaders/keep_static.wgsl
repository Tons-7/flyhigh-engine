enable f16;
// Static protection (option keep_static). An overlay that did not move (a HUD, chat, buttons, a logo) should stay
// exactly where it is, but the network warps it along the motion behind it. Two passes:
// - age (pair stage, once per source pair, keep_static_age.wgsl): the number of consecutive pairs in which the pixel
//   did not change, capped at 255. A coincidence inside a moving texture does not last 8 pairs; a 1 px crosshair line does.
// - the render (render.wgsl, static variant) does not take warp sources from protected pixels, so the overlay is not
//   dragged along the motion onto the world around it (a ghost copy of the crosshair).
// - mask (option protection_mask): the host's own map of pixels never to interpolate enters the age pass as age 255,
//   with or without the automatic detection (whose threshold then never matches, min_age 2).
// - apply (after the render, every generated frame): pixels static for at least `min_age` pairs take the sources'
//   own value, with a two-pair ramp. Agreement of one pair alone is not enough: an object moving over a background
//   (a weapon swaying in first person) can sit, in the middle of the pair, where both source frames show background.
// a, b: [3][hp][wp] padded source frames; age: [hp][wp]; render: [3][hp][wp], rewritten in place.
struct P { w: u32, h: u32, wp: u32, hp: u32, thr: f32, min_age: f32, use_mask: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> ages: array<f16>;
@group(0) @binding(1) var<storage, read> sa: array<f16>;
@group(0) @binding(2) var<storage, read> sb: array<f16>;
@group(0) @binding(3) var<storage, read_write> render: array<f16>;
@group(0) @binding(4) var<uniform> q: P;

@compute @workgroup_size(8, 8, 1)
fn apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= q.w || gid.y >= q.h) { return; }
    let o = gid.y * q.wp + gid.x;
    let k = clamp((f32(ages[o]) - q.min_age + 2.0) / 2.0, 0.0, 1.0);
    if (k <= 0.0) { return; }
    let plane = q.hp * q.wp;
    for (var c = 0u; c < 3u; c++) {
        let i = c * plane + o;
        render[i] = f16(mix(f32(render[i]), 0.5 * (f32(sa[i]) + f32(sb[i])), k));
    }
}
