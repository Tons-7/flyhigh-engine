enable f16;
// Build the three x-shifted, y-padded copies of a [cin][h][w] f16 tensor that let the direct conv kernel load
// its A tile with one column-major coopLoad straight from global memory:
//   out[kx][ci][y + 1][x] = inp[ci][y][x + kx - 1]   (zero outside the image; channels >= cin are zero)
// Layout [3][cinp][h + 2][w]. Every element is written, padding rows / channels included: the buffer comes from the
// plan's arena and holds a dead tensor's data (stale f32 maps read as f16 are NaN).
// Vectorised: one thread per 8 consecutive x (requires w % 8 == 0), gathers 8 shifted f16 and writes them as
// a single vec4<u32> (8x fewer store instructions than one thread per element).
struct P { cin: u32, cinp: u32, h: u32, w: u32 };
@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read_write> outp: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> p: P;

fn fetch(ci: u32, y: u32, xs: i32) -> f32 {
    if (xs < 0 || xs >= i32(p.w)) { return 0.0; }
    return f32(inp[(ci * p.h + y) * p.w + u32(xs)]);
}

// dispatch: x = w / 8 (in groups of 64), y = h + 2 padded rows, z = 3 * cinp
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x; let yp = gid.y; let kci = gid.z;
    let x0 = o * 8u;
    if (x0 >= p.w || yp >= p.h + 2u || kci >= 3u * p.cinp) { return; }
    let kx = kci / p.cinp; let ci = kci % p.cinp;
    let dst = (((kx * p.cinp + ci) * (p.h + 2u) + yp) * p.w + x0) / 8u;
    if (ci >= p.cin || yp == 0u || yp == p.h + 1u) { outp[dst] = vec4<u32>(0u); return; }
    let y = yp - 1u;
    let xs = i32(x0) + i32(kx) - 1;
    outp[dst] = vec4<u32>(
        pack2x16float(vec2<f32>(fetch(ci, y, xs),      fetch(ci, y, xs + 1))),
        pack2x16float(vec2<f32>(fetch(ci, y, xs + 2),  fetch(ci, y, xs + 3))),
        pack2x16float(vec2<f32>(fetch(ci, y, xs + 4),  fetch(ci, y, xs + 5))),
        pack2x16float(vec2<f32>(fetch(ci, y, xs + 6),  fetch(ci, y, xs + 7))));
}
