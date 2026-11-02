enable f16;
// Visibility blend weights: softmax over the 3 logits, wA = p0 + 0.5*p2, wB = p1 + 0.5*p2.
// Input [3,h,w] -> output [2,h,w]. f16 I/O, f32 math.
struct Par { h: u32, w: u32, c0: u32, scale: f32 };   // vis may be a view
@group(0) @binding(0) var<storage, read> vis: array<f16>;
@group(0) @binding(1) var<storage, read_write> outp: array<f16>;
@group(0) @binding(2) var<uniform> pp: Par;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= pp.w || y >= pp.h) { return; }
    let p = pp.h * pp.w; let i = y * pp.w + x; let vb = pp.c0 * p + i;
    let a = f32(vis[vb]) * pp.scale; let b = f32(vis[vb + p]) * pp.scale; let c = f32(vis[vb + 2u * p]) * pp.scale;
    let m = max(a, max(b, c));
    let ea = exp(a - m); let eb = exp(b - m); let ec = exp(c - m);
    let s = ea + eb + ec;
    let p0 = ea / s; let p1 = eb / s; let p2 = ec / s;
    outp[i] = f16(p0 + 0.5 * p2);
    outp[p + i] = f16(p1 + 0.5 * p2);
}
