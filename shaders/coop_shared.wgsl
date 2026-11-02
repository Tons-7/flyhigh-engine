enable f16;
enable wgpu_cooperative_matrix;
// Implicit-GEMM key test: can coopLoad read a matrix from SHARED (workgroup) memory?
@group(0) @binding(0) var<storage, read> src: array<f16>;    // A 16x16 row-major
@group(0) @binding(1) var<storage, read> bmat: array<f16>;   // B 16x16
@group(0) @binding(2) var<storage, read> zero: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
var<workgroup> tile: array<f16, 256>;
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    for (var i: u32 = lid.x; i < 256u; i = i + 32u) { tile[i] = src[i]; }
    workgroupBarrier();
    let pa = &tile[0];
    let a = coopLoadT<coop_mat16x16<f16, A>>(pa, 16u);   // <-- coopLoad from shared memory
    let pb = &bmat[0];
    let b = coopLoadT<coop_mat16x16<f16, B>>(pb, 16u);
    let pz = &zero[0];
    var acc = coopLoad<coop_mat16x16<f32, C>>(pz);
    acc = coopMultiplyAdd(a, b, acc);
    let fa = acc;
    let pd = &dst[0];
    coopStoreT(fa, pd, 16u);
}
