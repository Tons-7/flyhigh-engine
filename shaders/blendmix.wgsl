enable f16;
// Fused blend: out[c] = wa[c] * pA + wb[c] * pB, with (pA, pB) the blend weights of the 3-channel visibility logits
// (softmax; the third class is split half/half) — replaces blend_weights + 2 slices + 2 mul + add. `vis` may be a view
// (channel offset c0, scale).
struct Par { h: u32, w: u32, c0: u32, scale: f32 };
@group(0) @binding(0) var<storage, read> wa: array<f16>;
@group(0) @binding(1) var<storage, read> wb: array<f16>;
@group(0) @binding(2) var<storage, read> vis: array<f16>;
@group(0) @binding(3) var<storage, read_write> outp: array<f16>;
@group(0) @binding(4) var<uniform> pp: Par;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= pp.w || y >= pp.h) { return; }
    let p = pp.h * pp.w; let i = y * pp.w + x;
    let vb = pp.c0 * p + i;
    let a = f32(vis[vb]) * pp.scale; let b = f32(vis[vb + p]) * pp.scale; let c = f32(vis[vb + 2u * p]) * pp.scale;
    let m = max(a, max(b, c));
    let ea = exp(a - m); let eb = exp(b - m); let ec = exp(c - m);
    let s = ea + eb + ec;
    let pa = f32(f16((ea + 0.5 * ec) / s)); let pb = f32(f16((eb + 0.5 * ec) / s));   // f16 round-trip = the unfused path
    for (var ch: u32 = 0u; ch < 3u; ch = ch + 1u) {
        let o = ch * p + i;
        let t1 = f32(f16(f32(wa[o]) * pa)); let t2 = f32(f16(f32(wb[o]) * pb));
        outp[o] = f16(t1 + t2);
    }
}
