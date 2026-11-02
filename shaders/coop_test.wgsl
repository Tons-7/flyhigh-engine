enable f16;
enable wgpu_cooperative_matrix;
// Minimal Tensor-Core proof: one 16x16x16 f16 matmul C = A * B via cooperative matrix
// (hardware Tensor Cores). One subgroup cooperatively holds each matrix.
@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> c: array<f16>;

@compute @workgroup_size(32)
fn main() {
    let ma = coopLoad<coop_mat16x16<f16, A>>(&a[0]);       // stride inferred = 16
    let mb = coopLoad<coop_mat16x16<f16, B>>(&b[0]);
    let mc0 = coopLoad<coop_mat16x16<f16, C>>(&c[0]);      // c is pre-zeroed on the host
    let mc = coopMultiplyAdd(ma, mb, mc0);
    coopStore(mc, &c[0]);
}
