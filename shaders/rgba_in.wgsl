enable f16;
// RGBA8 frame (canvas ImageData / RGBA texture readout, [h][w] u32 little-endian R|G<<8|B<<16|A<<24) -> model input
// [3][hp][wp] f16 in [0,1], replicate-padded to the 32-aligned working size (same as the PyTorch reference pad32).
// Channel order is BGR: the model was trained on cv2 frames (channel 0 = blue), and luma / render assume it.
struct P { w: u32, h: u32, wp: u32, hp: u32 };
@group(0) @binding(0) var<storage, read> inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f16>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= p.wp || y >= p.hp) { return; }
    let v = inp[min(y, p.h - 1u) * p.w + min(x, p.w - 1u)];
    let plane = p.hp * p.wp; let o = y * p.wp + x;
    outp[o] = f16(f32((v >> 16u) & 0xffu) / 255.0);
    outp[plane + o] = f16(f32((v >> 8u) & 0xffu) / 255.0);
    outp[2u * plane + o] = f16(f32(v & 0xffu) / 255.0);
}
