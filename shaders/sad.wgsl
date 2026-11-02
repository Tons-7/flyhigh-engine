enable f16;
// Luma SAD cost volume: for displacement c (dy,dx in [-r,r]), out = 3x3 average (zero-padded,
// /9) of |la(y,x) - lb(y+dy,x+dx)| with lb edge-clamped. (2r+1)^2 channels. f16 I/O, f32 math.
struct Par { h: u32, w: u32, radius: u32, span: u32 };
@group(0) @binding(0) var<storage, read> la: array<f16>;
@group(0) @binding(1) var<storage, read> lb: array<f16>;
@group(0) @binding(2) var<storage, read_write> outp: array<f16>;
@group(0) @binding(3) var<uniform> pp: Par;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y; let c = gid.z;
    if (x >= pp.w || y >= pp.h || c >= pp.span * pp.span) { return; }
    let dy = i32(c / pp.span) - i32(pp.radius);
    let dx = i32(c % pp.span) - i32(pp.radius);
    let hh = i32(pp.h); let ww = i32(pp.w);
    var acc: f32 = 0.0;
    for (var wy: i32 = -1; wy <= 1; wy = wy + 1) {
        for (var wx: i32 = -1; wx <= 1; wx = wx + 1) {
            let yy = i32(y) + wy; let xx = i32(x) + wx;
            if (yy >= 0 && yy < hh && xx >= 0 && xx < ww) {
                let lav = f32(la[u32(yy) * pp.w + u32(xx)]);
                let syy = max(0, min(yy + dy, hh - 1)); let sxx = max(0, min(xx + dx, ww - 1));
                acc = acc + abs(lav - f32(lb[u32(syy) * pp.w + u32(sxx)]));
            }
        }
    }
    outp[(c * pp.h + y) * pp.w + x] = f16(acc / 9.0);
}
