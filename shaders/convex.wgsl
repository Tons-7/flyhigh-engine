enable f16;
// Convex upsample E8->Q (x2): softmax(over 9)-weighted sum of the 3x3 neighbourhood of x
// (zero-padded), weights from wt[36,h,w] as channel k9*4 + sy*2 + sx. f16 I/O, f32 math.
struct Par { c: u32, h: u32, w: u32 };
@group(0) @binding(0) var<storage, read> wt: array<f16>;
@group(0) @binding(1) var<storage, read> x: array<f16>;
@group(0) @binding(2) var<storage, read_write> outp: array<f16>;
@group(0) @binding(3) var<uniform> pp: Par;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x; let oy = gid.y; let co = gid.z;
    if (ox >= 2u * pp.w || oy >= 2u * pp.h || co >= pp.c) { return; }
    let ihh = oy / 2u; let iww = ox / 2u;
    let sy = oy % 2u; let sx = ox % 2u;
    let plane = pp.h * pp.w;
    var m: f32 = -3.0e38; var ws: array<f32, 9>;
    for (var k: u32 = 0u; k < 9u; k = k + 1u) {
        let ch = k * 4u + sy * 2u + sx;
        let v = f32(wt[(ch * pp.h + ihh) * pp.w + iww]); ws[k] = v; m = max(m, v);
    }
    var denom: f32 = 0.0;
    for (var k: u32 = 0u; k < 9u; k = k + 1u) { ws[k] = exp(ws[k] - m); denom = denom + ws[k]; }
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < 9u; k = k + 1u) {
        let ky = i32(k / 3u); let kx = i32(k % 3u);
        let ny = i32(ihh) - 1 + ky; let nx = i32(iww) - 1 + kx;
        var xv: f32 = 0.0;
        if (ny >= 0 && ny < i32(pp.h) && nx >= 0 && nx < i32(pp.w)) {
            xv = f32(x[co * plane + u32(ny) * pp.w + u32(nx)]);
        }
        acc = acc + (ws[k] / denom) * xv;
    }
    outp[(co * 2u * pp.h + oy) * 2u * pp.w + ox] = f16(acc);
}
