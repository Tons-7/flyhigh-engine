enable f16;
// Bilinear resize. aa=0: bilinear align_corners=False. aa=1: antialias downsample (separable
// triangle, matches torch F.interpolate antialias=True). f16 I/O, f32 math.
struct Par { c: u32, ih: u32, iw: u32, oh: u32, ow: u32, aa: u32, c0: u32, scale: f32 };   // input may be a view
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read_write> outp: array<f16>;
@group(0) @binding(2) var<uniform> pp: Par;

fn at(i: u32) -> f32 { return f32(inp[i]); }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x; let oy = gid.y; let c = gid.z;
    if (ox >= pp.ow || oy >= pp.oh || c >= pp.c) { return; }
    let sh = f32(pp.ih) / f32(pp.oh);
    let sw = f32(pp.iw) / f32(pp.ow);
    let base = (c + pp.c0) * pp.ih * pp.iw;
    var val: f32;
    if (pp.aa == 0u) {
        let iy = (f32(oy) + 0.5) * sh - 0.5;
        let ix = (f32(ox) + 0.5) * sw - 0.5;
        let y0 = i32(floor(iy)); let x0 = i32(floor(ix));
        let fy = iy - f32(y0); let fx = ix - f32(x0);
        let y0c = max(0, min(y0, i32(pp.ih) - 1)); let y1c = max(0, min(y0 + 1, i32(pp.ih) - 1));
        let x0c = max(0, min(x0, i32(pp.iw) - 1)); let x1c = max(0, min(x0 + 1, i32(pp.iw) - 1));
        let v00 = at(base + u32(y0c) * pp.iw + u32(x0c)); let v01 = at(base + u32(y0c) * pp.iw + u32(x1c));
        let v10 = at(base + u32(y1c) * pp.iw + u32(x0c)); let v11 = at(base + u32(y1c) * pp.iw + u32(x1c));
        val = (v00 * (1.0 - fx) + v01 * fx) * (1.0 - fy) + (v10 * (1.0 - fx) + v11 * fx) * fy;
    } else {
        let fsh = max(sh, 1.0); let fsw = max(sw, 1.0);
        let cy = (f32(oy) + 0.5) * sh; let cx = (f32(ox) + 0.5) * sw;
        let ymin = max(0, i32(floor(cy - fsh + 0.5))); let ymax = min(i32(pp.ih), i32(floor(cy + fsh + 0.5)));
        let xmin = max(0, i32(floor(cx - fsw + 0.5))); let xmax = min(i32(pp.iw), i32(floor(cx + fsw + 0.5)));
        var acc: f32 = 0.0; var norm: f32 = 0.0;
        for (var y: i32 = ymin; y < ymax; y = y + 1) {
            let wy = max(0.0, 1.0 - abs((f32(y) + 0.5 - cy) / fsh));
            for (var x: i32 = xmin; x < xmax; x = x + 1) {
                let wx = max(0.0, 1.0 - abs((f32(x) + 0.5 - cx) / fsw));
                let w = wx * wy; acc = acc + at(base + u32(y) * pp.iw + u32(x)) * w; norm = norm + w;
            }
        }
        val = acc / norm;
    }
    outp[(c * pp.oh + oy) * pp.ow + ox] = f16(val * pp.scale);
}
