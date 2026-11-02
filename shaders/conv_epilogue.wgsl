enable f16;
// Conv epilogue: C [M, Npad] (f32, GEMM output) -> output [Cout,Hout,Wout] (f16), + bias, PReLU.
// output[co, oy, ox] = prelu(C[(oy*Wout+ox)*Npad + co] + bias[co]).
struct P { cout: u32, hout: u32, wout: u32, npad: u32 };
@group(0) @binding(0) var<storage, read> C: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f16>;
@group(0) @binding(2) var<storage, read> slope: array<f16>;
@group(0) @binding(3) var<storage, read_write> outp: array<f16>;
@group(0) @binding(4) var<uniform> p: P;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x; let oy = gid.y; let co = gid.z;
    if (ox >= p.wout || oy >= p.hout || co >= p.cout) { return; }
    let m = oy * p.wout + ox;
    var v = C[m * p.npad + co] + f32(bias[co]);
    v = select(f32(slope[co]) * v, v, v > 0.0);
    outp[(co * p.hout + oy) * p.wout + ox] = f16(v);
}
