// Option defaults and pure helpers: no GPU needed (runs in CI everywhere).
use flyhigh_engine::api::{default_internal, Mode, Options, DEFAULT_REPAIR};
use flyhigh_engine::gen_tex::{TexMode, TexOptions};
use flyhigh_engine::{format_tune, parse_tune, tune_table_from, EngineOptions};

#[test]
fn internal_size_is_the_480_line_rung_above_960_px() {
    assert_eq!(default_internal(800, 600), (800, 600));
    assert_eq!(default_internal(960, 540), (960, 540));
    assert_eq!(default_internal(1280, 720), (856, 480));
    assert_eq!(default_internal(1920, 1080), (856, 480));
    assert_eq!(default_internal(3840, 2160), (856, 480));
    assert_eq!(default_internal(1080, 1920), (272, 480)); // portrait
}

#[test]
fn generator_defaults() {
    let o = Options::new(1920, 1080);
    assert_eq!((o.k, o.repair, o.mode, o.cadence, o.cubic), (1, DEFAULT_REPAIR, Mode::Realtime, false, None));
    assert_eq!(o.internal, Some((856, 480)));
    assert!(!o.keep_static && !o.tex_options().keep_static && !TexOptions::new(64, 64).keep_static);
    assert_eq!(Options::new(800, 600).internal, None);
    let t = o.tex_options();
    assert_eq!((t.w, t.h, t.iw, t.ih, t.k, t.repair), (1920, 1080, 856, 480, 1, 1));
    let n = TexOptions::new(1920, 1080);
    assert_eq!(
        (n.iw, n.ih, n.k, n.repair, n.mode, n.cadence, n.cubic),
        (856, 480, 1, 1, TexMode::Realtime, false, None)
    );
}

#[test]
fn engine_defaults() {
    let e = EngineOptions::default();
    assert!(
        e.backend.is_none()
            && e.tensor_cores
            && e.autotune
            && e.adapter.is_none()
            && e.precision == flyhigh_engine::Precision::Auto
    );
}

#[test]
fn tune_table_round_trips_and_skips_garbage() {
    let text = "96,96,60,108,1=18\nnot a line\n36,36,180,320,1=2\n1,2,3=4\n48,48,60,108,1=x\n";
    let t = parse_tune(text);
    assert_eq!(t.len(), 2);
    assert_eq!(t[&(96, 96, 60, 108, 1)], 18);
    assert_eq!(parse_tune(&format_tune(&t)), t);
    assert!(tune_table_from("").is_none());
    assert!(tune_table_from("garbage").is_none());
}
