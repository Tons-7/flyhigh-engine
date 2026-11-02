enable f16;
// Static protection, age pass: see keep_static.wgsl.
struct P { w: u32, h: u32, wp: u32, hp: u32, thr: f32, min_age: f32, use_mask: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> outp: array<f16>;
@group(0) @binding(3) var<uniform> p: P;
// the host's protection mask (option protection_mask): 1 where never to interpolate, read as age 255; the zero
// buffer stands in when the session has none (use_mask 0)
@group(0) @binding(4) var<storage, read> mask: array<f16>;

fn diff(o: u32, plane: u32) -> f32 {
    var d = 0.0;
    for (var c = 0u; c < 3u; c++) { d = max(d, abs(f32(a[c * plane + o]) - f32(b[c * plane + o]))); }
    return d;
}

// outp = age
@compute @workgroup_size(8, 8, 1)
fn age(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) { return; }
    let o = gid.y * p.wp + gid.x;
    let still = diff(o, p.hp * p.wp) < p.thr;
    let aged = select(f16(0.0), f16(min(f32(outp[o]) + 1.0, 255.0)), still);
    outp[o] = select(aged, f16(255.0), p.use_mask != 0u && f32(mask[o]) > 0.5);
}
