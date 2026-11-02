// Pure logic, no GPU: the f32 shader rewrite, conv kernel candidate lists, the cadence (re-timing) decision.
use flyhigh_engine::gen_tex::{cadence_retime, Retime};
use flyhigh_engine::gmhs::PairStats;
use flyhigh_engine::{
    cand_list, shader_f32, CAND_DIRECT, CAND_TC, CAND_TILED, CAND_TILED16, CAND_TILED8, CAND_TILED88, CAND_TILEDN8,
};
use std::collections::VecDeque;

#[test]
fn f32_shader_swaps_whole_words_only() {
    let src = "enable f16;\nvar<storage> a: array<f16>;\nfn f(x: f16) -> f16 { let half16 = 1u; return f16(x) + unpack2x16float(0u).x; }";
    let out = shader_f32(src);
    assert!(!out.contains("enable f16"), "the f16 extension is dropped");
    assert!(out.contains("array<f32>") && out.contains("fn f(x: f32) -> f32") && out.contains("return f32(x)"));
    assert!(
        out.contains("half16") && out.contains("unpack2x16float"),
        "identifiers that merely contain f16 or 16 stay"
    );
    assert!(!out.split(|c: char| !c.is_alphanumeric() && c != '_').any(|w| w == "f16"), "no f16 word is left");
}

#[test]
fn f32_chcopy_moves_float_pairs_instead_of_packed_halves() {
    let out = shader_f32(include_str!("../../../shaders/chcopy.wgsl"));
    assert!(out.contains("array<vec2<f32>>") && out.contains("return v * s;"));
    assert!(!out.contains("pack2x16float"));
}

fn has(list: &[u8], kind: u8) -> bool {
    list.iter().any(|&c| c & 15 == kind)
}

#[test]
fn portable_candidates_for_a_wide_conv() {
    // fine.r1-like shape: 64 channels out at 60 x 108, no tensor cores
    let l = cand_list(false, 64, 60, 108, false);
    assert!(has(&l, CAND_TILED) && has(&l, CAND_TILED8) && has(&l, CAND_TILED88));
    assert!(!has(&l, CAND_TC) && !has(&l, CAND_DIRECT), "no tensor-core kernels without the feature");
    assert!(!has(&l, CAND_TILED16) && !has(&l, CAND_TILEDN8), "the narrow tiles are for small cout");
    // 102 tiles < 512 workgroups: the split-K variants (k = 1 / 2 / 4) are tried as well
    assert!(
        l.contains(&(CAND_TILED | 1 << 4)) && l.contains(&(CAND_TILED | 2 << 4)) && l.contains(&(CAND_TILED | 4 << 4))
    );
}

#[test]
fn narrow_candidates_for_a_small_conv() {
    let l = cand_list(false, 16, 120, 216, false);
    assert!(has(&l, CAND_TILED) && has(&l, CAND_TILED16) && has(&l, CAND_TILEDN8));
    assert!(!has(&l, CAND_TILED8) && !has(&l, CAND_TILED88), "the wide register blocks need cout > 32");
}

#[test]
fn tensor_core_candidates_need_the_feature_and_direct_needs_its_shape() {
    let tc = cand_list(true, 64, 60, 108, false);
    assert!(has(&tc, CAND_TC) && !has(&tc, CAND_DIRECT));
    let direct = cand_list(true, 64, 60, 112, true);
    assert!(has(&direct, CAND_TC) && has(&direct, CAND_DIRECT));
    assert!(has(&tc, CAND_TILED), "the portable kernels stay in the race");
}

fn pairs(spec: &[(f32, bool, bool)]) -> VecDeque<PairStats> {
    // (changed, jump, cut)
    spec.iter().map(|&(changed, jump, cut)| PairStats { changed, jump, cut, ..PairStats::default() }).collect()
}
const ORDER: [usize; 3] = [0, 1, 2];

#[test]
fn still_shot_on_twos_renders_from_the_transition_pair() {
    // a drawing changes every other frame on a static camera: pairs alternate "new drawing" / "held copy";
    // the current pair (index 4) is a step, the lookahead pair a hold
    let h = pairs(&[
        (0.1, false, false),
        (0.0, false, false),
        (0.1, false, false),
        (0.0, false, false),
        (0.1, false, false),
        (0.0, false, false),
    ]);
    let r = cadence_retime(&h, &ORDER, 0.25).expect("re-timed");
    assert_eq!(r.sess, ORDER[1], "the current pair is the transition into the next drawing");
    assert!(r.full, "a still shot is rendered whole from that pair");
    // the drawing centres are at -0.5 and 1.5: phase 0.25 sits 3/8 of the way
    assert!((r.te - 0.375).abs() < 1e-6, "te {}", r.te);
    assert!((cadence_retime(&h, &ORDER, 0.75).unwrap().te - 0.625).abs() < 1e-6);
}

#[test]
fn nothing_is_retimed_across_a_cut_or_without_history() {
    let cut_next = pairs(&[(0.1, false, false), (0.0, false, false), (0.1, false, false), (0.0, false, true)]);
    assert_eq!(cadence_retime(&cut_next, &ORDER, 0.5), None, "the lookahead pair is a cut");
    let cut_now = pairs(&[(0.1, false, false), (0.0, false, true), (0.1, false, false)]);
    assert_eq!(cadence_retime(&cut_now, &ORDER, 0.5), None, "the current pair is a cut");
    assert_eq!(cadence_retime(&pairs(&[(0.1, false, false)]), &ORDER, 0.5), None, "one pair is no history");
    assert_eq!(cadence_retime(&VecDeque::new(), &ORDER, 0.5), None);
}

#[test]
fn continuous_motion_and_plain_holds_are_left_alone() {
    // camera move with an element jumping every frame: steps everywhere, hold length 1 = continuous motion
    let every =
        pairs(&[(0.5, true, false), (0.5, true, false), (0.5, true, false), (0.5, true, false), (0.5, true, false)]);
    assert_eq!(cadence_retime(&every, &ORDER, 0.5), None);
    // nothing changes in the window: a hold, nothing to re-time
    let hold = pairs(&[(0.0, false, false), (0.0, false, false), (0.0, false, false), (0.0, false, false)]);
    assert_eq!(cadence_retime(&hold, &ORDER, 0.5), None);
}

#[test]
fn camera_move_on_twos_uses_the_jump_rule_with_a_mask() {
    // the whole frame changes every pair (a pan), the element jumps every other pair; the lookahead pair jumps
    let h = pairs(&[
        (0.6, true, false),
        (0.6, false, false),
        (0.6, true, false),
        (0.6, false, false),
        (0.6, true, false),
        (0.6, false, false),
    ]);
    let r: Retime = cadence_retime(&h, &ORDER, 0.25).expect("re-timed");
    assert!(!r.full, "during a camera move only the element is re-timed (masked composite)");
    assert_eq!(r.sess, ORDER[1]);
}
