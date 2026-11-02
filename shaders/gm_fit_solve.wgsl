// GMHS fit step 2: sum the workgroup partials and solve the 3x3 normal equations per flow channel (one thread each).
struct Par { nwg: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read> part: array<f32>;        // [nwg][4][9]
@group(0) @binding(1) var<storage, read_write> coef: array<f32>;  // [4][4]
@group(0) @binding(2) var<uniform> pp: Par;

@compute @workgroup_size(4, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let c = lid.x;
    var s: array<f32, 9>;
    for (var g: u32 = 0u; g < pp.nwg; g = g + 1u) {
        for (var k: u32 = 0u; k < 9u; k = k + 1u) { s[k] = s[k] + part[(g * 4u + c) * 9u + k]; }
    }
    let m00 = s[0] + 1e-6; let m01 = s[1]; let m02 = s[2]; let m11 = s[3] + 1e-6; let m12 = s[4]; let m22 = s[5] + 1e-6;
    let det = m00 * (m11 * m22 - m12 * m12) - m01 * (m01 * m22 - m12 * m02) + m02 * (m01 * m12 - m11 * m02);
    if (abs(det) < 1e-12) {
        coef[c * 4u] = s[6] / max(s[0], 1e-6); coef[c * 4u + 1u] = 0.0; coef[c * 4u + 2u] = 0.0; coef[c * 4u + 3u] = 0.0;
        return;
    }
    let i00 = (m11 * m22 - m12 * m12) / det; let i01 = (m02 * m12 - m01 * m22) / det; let i02 = (m01 * m12 - m02 * m11) / det;
    let i11 = (m00 * m22 - m02 * m02) / det; let i12 = (m01 * m02 - m00 * m12) / det; let i22 = (m00 * m11 - m01 * m01) / det;
    coef[c * 4u] = i00 * s[6] + i01 * s[7] + i02 * s[8];
    coef[c * 4u + 1u] = i01 * s[6] + i11 * s[7] + i12 * s[8];
    coef[c * 4u + 2u] = i02 * s[6] + i12 * s[7] + i22 * s[8];
    coef[c * 4u + 3u] = 0.0;
}
