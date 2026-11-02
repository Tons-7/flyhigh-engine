//! The kernel candidates the autotuner times for a conv shape, and how a choice is encoded.

// conv kernel candidates for the autotuner (tensor-core devices): see BuildCtx::conv_cand
// candidate code = kernel id (low 4 bits) | split-K code << 4 (0 = heuristic, 1, 2, 4)
pub const CAND_DIRECT: u8 = 0; // xshift3 + fdirect32 (no gather), stride 1, w % 16 == 0, cout > 32
pub const CAND_TC: u8 = 1; // gathering Tensor-Core implicit GEMM (fused / fused2)
pub const CAND_TILED: u8 = 2; // portable FMA tiled conv (+ split-K), the browser-class kernel
pub const CAND_TILED8: u8 = 3; // portable FMA, 8x4 register block (128 px x 64 ch), cout > 32
pub const CAND_DIRECT64: u8 = 4; // xshift3 + fdirect64 (16 accumulators); never won on the 5090, kept for other GPUs (FH_TRY_DIRECT64=1)
pub const CAND_TILED16: u8 = 5; // portable FMA, 128 px x 16 ch tile (cout <= 32)
pub const CAND_TILEDN8: u8 = 6; // portable FMA, 128 px x 8 ch tile (cout <= 16)
pub const CAND_TILED88: u8 = 7; // portable FMA, 8x8 register block, 128 px x 64 ch / 128 thr (cout > 32)
pub const CAND_TILEDM32: u8 = 8; // portable FMA, 4x4 block, 32 px x 64 ch / 128 thr (cout > 32)
pub const CAND_TILED8P: u8 = 9; // tiled8 with register prefetch of the next K-step (software pipelining)
pub const CAND_NAMES: [&str; 10] =
    ["direct", "tc", "tiled", "tiled8", "direct64", "tiled16", "tiledn8", "tiled88", "tiledm32", "tiled8p"];
pub fn cand_name(c: u8) -> String {
    let ks = c >> 4;
    if ks == 0 {
        CAND_NAMES[(c & 15) as usize].to_string()
    } else {
        format!("{}/k{}", CAND_NAMES[(c & 15) as usize], ks)
    }
}
// the kernel candidates worth timing for a conv of this shape on this device (FH_TRY_ALL=1 adds the ones that never
// won on the 5090: direct64, tiledm32, tiled8p); the FMA kernels are also tried with split-K forced to 1/2/4 when the
// heuristic would split
pub fn cand_list(coop: bool, cout: u32, hout: u32, wout: u32, direct_ok: bool) -> Vec<u8> {
    let try_all = std::env::var_os("FH_TRY_ALL").is_some() || std::env::var_os("FH_TRY_DIRECT64").is_some();
    let mut cands: Vec<u8> = vec![CAND_DIRECT, CAND_DIRECT64, CAND_TC];
    for k in [CAND_TILED, CAND_TILED8, CAND_TILED16, CAND_TILEDN8, CAND_TILED88, CAND_TILEDM32, CAND_TILED8P] {
        cands.push(k);
        let (tm, tn) = if cout <= 32 {
            (
                128,
                match k {
                    CAND_TILED16 => 16,
                    CAND_TILEDN8 => 8,
                    _ => 32,
                },
            )
        } else {
            (
                match k {
                    CAND_TILED8 | CAND_TILED88 | CAND_TILED8P => 128,
                    CAND_TILEDM32 => 32,
                    _ => 64,
                },
                64,
            )
        };
        let ntiles = if cout <= 32 { cout.div_ceil(tn) } else { cout.div_ceil(64) };
        if ntiles * (hout * wout).div_ceil(tm) < 512 {
            for ks in [1u8, 2, 4] {
                cands.push(k | (ks << 4));
            }
        }
    }
    cands
        .into_iter()
        .filter(|&cand| {
            let kid = cand & 15;
            if (kid == CAND_DIRECT || kid == CAND_DIRECT64) && !direct_ok {
                return false;
            }
            if (kid == CAND_DIRECT64 || kid == CAND_TILEDM32 || kid == CAND_TILED8P) && !try_all {
                return false;
            }
            if kid == CAND_TC && !coop {
                return false;
            }
            if (kid == CAND_TILED8 || kid == CAND_TILED88 || kid == CAND_TILEDM32 || kid == CAND_TILED8P) && cout <= 32
            {
                return false;
            }
            if kid == CAND_TILED16 && cout > 32 {
                return false;
            }
            if kid == CAND_TILEDN8 && cout > 16 {
                return false;
            }
            true
        })
        .collect()
}
