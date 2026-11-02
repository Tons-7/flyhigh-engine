enable f16;
enable wgpu_cooperative_matrix;
struct Dims { m: u32, n: u32, k: u32, _p: u32 };
@group(0) @binding(0) var<storage, read> matA: array<f16>;
@group(0) @binding(1) var<storage, read> matB: array<f16>;
@group(0) @binding(2) var<storage, read> Zero: array<f32>;
@group(0) @binding(3) var<storage, read_write> Cout: array<f32>;
@group(0) @binding(4) var<uniform> d: Dims;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    let sk = d.k;
    let sn = d.n;
    let m0 = wid.y * 16u;
    let n0 = wid.x * 16u;
    var acc = coopLoad<coop_mat16x16<f32, C>>(&Zero[0]);
    let kt_count = sk / 16u;
    for (var kt: u32 = 0u; kt < kt_count; kt = kt + 1u) {
        let pa = &matA[m0 * sk + kt * 16u];
        let pb = &matB[(kt * 16u) * sn + n0];
        acc = coopMultiplyAdd(coopLoadT<coop_mat16x16<f16, A>>(pa, sk), coopLoadT<coop_mat16x16<f16, B>>(pb, sn), acc);
    }
    let final_acc = acc;
    let pc = &Cout[m0 * sn + n0];
    coopStoreT(final_acc, pc, sn);
}
