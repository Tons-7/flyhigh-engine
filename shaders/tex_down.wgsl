enable f16;
// Fast render's input stage: source texture (rgba8unorm) -> network input [3][oh][ow] f16 (BGR), the antialiased
// downscale of interp.wgsl (separable triangle = torch interpolate antialias=True) applied to the replicate-padded
// frame (iw x ih = the 32-aligned working size) in one pass, all three channels per thread. Replaces tex_in at full
// resolution followed by interp: no full-resolution tensor is written or read.
struct P { w: u32, h: u32, iw: u32, ih: u32, ow: u32, oh: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f16>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x; let oy = gid.y;
    if (ox >= p.ow || oy >= p.oh) { return; }
    let sh = f32(p.ih) / f32(p.oh); let sw = f32(p.iw) / f32(p.ow);
    let fsh = max(sh, 1.0); let fsw = max(sw, 1.0);
    let cy = (f32(oy) + 0.5) * sh; let cx = (f32(ox) + 0.5) * sw;
    let ymin = max(0, i32(floor(cy - fsh + 0.5))); let ymax = min(i32(p.ih), i32(floor(cy + fsh + 0.5)));
    let xmin = max(0, i32(floor(cx - fsw + 0.5))); let xmax = min(i32(p.iw), i32(floor(cx + fsw + 0.5)));
    var acc = vec3<f32>(0.0); var norm: f32 = 0.0;
    for (var y: i32 = ymin; y < ymax; y = y + 1) {
        let wy = max(0.0, 1.0 - abs((f32(y) + 0.5 - cy) / fsh));
        let ty = min(y, i32(p.h) - 1);
        for (var x: i32 = xmin; x < xmax; x = x + 1) {
            let wx = max(0.0, 1.0 - abs((f32(x) + 0.5 - cx) / fsw));
            let w = wx * wy;
            acc = acc + textureLoad(src, vec2<i32>(min(x, i32(p.w) - 1), ty), 0).rgb * w; norm = norm + w;
        }
    }
    let v = acc / norm;
    let plane = p.oh * p.ow; let o = oy * p.ow + ox;
    outp[o] = f16(v.b); outp[plane + o] = f16(v.g); outp[2u * plane + o] = f16(v.r);
}
