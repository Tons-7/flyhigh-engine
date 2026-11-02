// Channel-range copy / up-to-7-way concat as a compute dispatch, replacing copy_buffer_to_buffer steps (each of
// those split the frame's single compute pass and cost a pipeline drain). All tensors are [C][H][W] f16 with the same
// H, W, viewed as u32 pairs (hw is even). Output channel c (relative to dst0) comes from input k with
// ends[k-1] <= c < ends[k], read at that input's channel offset o_k (views) and multiplied by s_k when s_k != 1.
// 7 inputs + 1 output = the WebGPU default limit of 8 storage buffers per stage; unused inputs are bound to a dummy.
struct P { hw2: u32, ctot: u32, e0: u32, e1: u32, e2: u32, e3: u32, e4: u32, e5: u32, e6: u32,
           o0: u32, o1: u32, o2: u32, o3: u32, o4: u32, o5: u32, o6: u32, dst0: u32,
           s0: f32, s1: f32, s2: f32, s3: f32, s4: f32, s5: f32, s6: f32, _a: u32, _b: u32, _c: u32, _d: u32 };
@group(0) @binding(0) var<storage, read> i0: array<u32>;
@group(0) @binding(1) var<storage, read> i1: array<u32>;
@group(0) @binding(2) var<storage, read> i2: array<u32>;
@group(0) @binding(3) var<storage, read> i3: array<u32>;
@group(0) @binding(4) var<storage, read> i4: array<u32>;
@group(0) @binding(5) var<storage, read> i5: array<u32>;
@group(0) @binding(6) var<storage, read> i6: array<u32>;
@group(0) @binding(7) var<storage, read_write> outp: array<u32>;
@group(0) @binding(8) var<uniform> p: P;

fn scl(v: u32, s: f32) -> u32 {
    if (s == 1.0) { return v; }
    return pack2x16float(unpack2x16float(v) * s);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.ctot * p.hw2) { return; }
    let c = i / p.hw2; let r = i - c * p.hw2;
    var v: u32;
    if (c < p.e0) { v = scl(i0[(c + p.o0) * p.hw2 + r], p.s0); }
    else if (c < p.e1) { v = scl(i1[(c - p.e0 + p.o1) * p.hw2 + r], p.s1); }
    else if (c < p.e2) { v = scl(i2[(c - p.e1 + p.o2) * p.hw2 + r], p.s2); }
    else if (c < p.e3) { v = scl(i3[(c - p.e2 + p.o3) * p.hw2 + r], p.s3); }
    else if (c < p.e4) { v = scl(i4[(c - p.e3 + p.o4) * p.hw2 + r], p.s4); }
    else if (c < p.e5) { v = scl(i5[(c - p.e4 + p.o5) * p.hw2 + r], p.s5); }
    else { v = scl(i6[(c - p.e5 + p.o6) * p.hw2 + r], p.s6); }
    outp[(c + p.dst0) * p.hw2 + r] = v;
}
