enable f16;
// Backward warp: out[p] = img[p + flow[p]], flow in pixels. Bilinear, border clamp
// (grid_sample align_corners=True). f16 I/O, f32 math.
struct Par { c: u32, h: u32, w: u32, f0: u32, i0: u32, fs: f32, is: f32, _a: u32 };   // flow / image may be views
@group(0) @binding(0) var<storage, read> img: array<f16>;
@group(0) @binding(1) var<storage, read> flow: array<f16>;
@group(0) @binding(2) var<storage, read_write> outp: array<f16>;
@group(0) @binding(3) var<uniform> pp: Par;

fn clampi(v: i32, lo: i32, hi: i32) -> i32 { return max(lo, min(v, hi)); }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y; let c = gid.z;
    if (x >= pp.w || y >= pp.h || c >= pp.c) { return; }
    let plane = pp.h * pp.w;
    let fx = f32(flow[(pp.f0 * pp.h + y) * pp.w + x]) * pp.fs;
    let fy = f32(flow[((pp.f0 + 1u) * pp.h + y) * pp.w + x]) * pp.fs;
    let gx = f32(x) + fx; let gy = f32(y) + fy;
    let x0 = i32(floor(gx)); let y0 = i32(floor(gy));
    let ax = gx - f32(x0); let ay = gy - f32(y0);
    let x0c = clampi(x0, 0, i32(pp.w) - 1); let x1c = clampi(x0 + 1, 0, i32(pp.w) - 1);
    let y0c = clampi(y0, 0, i32(pp.h) - 1); let y1c = clampi(y0 + 1, 0, i32(pp.h) - 1);
    let base = (c + pp.i0) * plane;
    let v00 = f32(img[base + u32(y0c) * pp.w + u32(x0c)]); let v01 = f32(img[base + u32(y0c) * pp.w + u32(x1c)]);
    let v10 = f32(img[base + u32(y1c) * pp.w + u32(x0c)]); let v11 = f32(img[base + u32(y1c) * pp.w + u32(x1c)]);
    let top = v00 * (1.0 - ax) + v01 * ax; let bot = v10 * (1.0 - ax) + v11 * ax;
    outp[c * plane + y * pp.w + x] = f16((top * (1.0 - ay) + bot * ay) * pp.is);
}
