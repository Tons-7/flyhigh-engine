// The C-side defaults match the library's: no GPU needed.
use flyhigh::{fh_engine_options_default, fh_options_default};
use flyhigh_engine::api::DEFAULT_REPAIR;

#[test]
fn engine_options_default() {
    let c = fh_engine_options_default();
    assert_eq!((c.backend, c.tensor_cores, c.autotune, c.compact_memory), (0, 1, 1, 1));
    assert!(c.adapter.is_null() && c.tune_cache.is_null() && c.dxc_path.is_null() && c.precision == 0);
}

#[test]
fn options_default_matches_rust() {
    let c = fh_options_default(1920, 1080);
    assert_eq!(
        (c.width, c.height, c.k, c.internal_w, c.internal_h, c.mode, c.repair, c.cadence, c.render_sampling),
        (1920, 1080, 1, 856, 480, 0, DEFAULT_REPAIR, 0, 0)
    );
    assert_eq!((c.keep_static, c.fast_render, c.linear_t), (0, 0, 0));
    let s = fh_options_default(640, 480);
    assert_eq!((s.internal_w, s.internal_h), (0, 0)); // 0, 0 = the frame size
}
