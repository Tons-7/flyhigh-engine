// The Core and the Stream over C: the option structs map both ways without loss (no GPU), and on a real GPU a C host
// generates an in-between through fh_engine / fh_model / fh_session and through fh_stream (ignored by default).
use flyhigh::{
    fh_engine_create, fh_engine_destroy, fh_engine_info, fh_engine_options_default, fh_model_destroy,
    fh_model_from_bytes, fh_model_info, fh_model_load, fh_session_add_input, fh_session_add_output, fh_session_create,
    fh_session_destroy, fh_session_mid, fh_session_options_default, fh_session_pair, fh_session_read_rgba,
    fh_session_report_json, fh_session_write_rgba, fh_stream_classify, fh_stream_create, fh_stream_destroy,
    fh_stream_generate, fh_stream_next_input, fh_stream_options_default, fh_stream_push, fh_stream_read_rgba,
    fh_stream_write_rgba, FhClassification, FhPush, FhSessionOptions, FhStreamOptions,
};
use flyhigh_engine::stream::{HeldFrames, Mode, SceneCut, StreamOptions};
use flyhigh_engine::{Internal, RenderSampling, SessionOptions, StaticProtection};
use std::ffi::{c_char, CStr, CString};

const W: u32 = 256;
const H: u32 = 192;

fn last_error() -> String {
    unsafe { CStr::from_ptr(flyhigh::fh_last_error()) }.to_string_lossy().to_string()
}

#[test]
fn session_options_round_trip_through_the_c_struct() {
    let d = fh_session_options_default(1920, 1080);
    assert_eq!((d.width, d.height, d.internal_w, d.internal_h, d.k, d.repair, d.render), (1920, 1080, 0, 0, 1, 1, 0));
    assert_eq!(unsafe { d.to_rust() }.unwrap(), SessionOptions::new(1920, 1080));
    for o in [
        SessionOptions { internal: Internal::Frame, k: 2, ..SessionOptions::new(640, 480) },
        SessionOptions {
            internal: Internal::Size(428, 240),
            render: RenderSampling::Bicubic,
            ..SessionOptions::new(1280, 720)
        },
        SessionOptions {
            static_protection: Some(StaticProtection { frames: 12, tolerance: 6 }),
            fast_render: true,
            linear_t: true,
            keep_maps: true,
            ..SessionOptions::new(1920, 1080)
        },
    ] {
        assert_eq!(unsafe { FhSessionOptions::from_rust(&o).to_rust() }.unwrap(), o);
    }
    let bad = FhSessionOptions { k: 7, ..fh_session_options_default(W, H) };
    assert!(unsafe { bad.to_rust() }.unwrap_err().contains("k"));
}

#[test]
fn stream_options_round_trip_through_the_c_struct() {
    let d = fh_stream_options_default();
    assert_eq!((d.mode, d.cadence, d.held_frames, d.inputs, d.outputs), (0, 0, 0, 4, 4));
    assert!((d.scene_cut - 0.08).abs() < 1e-6);
    assert_eq!(d.to_rust().unwrap(), StreamOptions::default());
    let o = StreamOptions {
        mode: Mode::Video,
        cadence: true,
        scene_cut: None,
        held_frames: Some(HeldFrames::default()),
        inputs: 6,
        outputs: 3,
    };
    assert_eq!(FhStreamOptions::from_rust(&o).to_rust().unwrap(), o);
    let custom = StreamOptions { scene_cut: Some(SceneCut { threshold: 0.2 }), ..StreamOptions::default() };
    let back = FhStreamOptions::from_rust(&custom).to_rust().unwrap();
    assert!((back.scene_cut.unwrap().threshold - 0.2).abs() < 1e-6);
    assert!(FhStreamOptions { mode: 5, ..fh_stream_options_default() }.to_rust().is_err());
}

fn frame(dx: f32) -> Vec<u8> {
    let mut v = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let u = (x as f32 - dx) * 0.07;
            let w = y as f32 * 0.05;
            let o = ((y * W + x) * 4) as usize;
            v[o] = (128.0 + 100.0 * u.sin()) as u8;
            v[o + 1] = (128.0 + 100.0 * (u * 0.7 + w).cos()) as u8;
            v[o + 2] = (128.0 + 90.0 * (u + w).sin()) as u8;
            v[o + 3] = 255;
        }
    }
    v
}

fn mae(a: &[u8], b: &[u8]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / a.len() as f64
}

fn assert_inbetween(got: &[u8]) {
    let (f0, f1, mid) = (frame(0.0), frame(8.0), frame(4.0));
    let (to_mid, to_f0, to_f1) = (mae(got, &mid), mae(got, &f0), mae(got, &f1));
    eprintln!("MAE to the true middle {to_mid:.2}, to f0 {to_f0:.2}, to f1 {to_f1:.2}");
    assert!(to_mid < to_f0 && to_mid < to_f1 && to_mid < 6.0, "the in-between lies between its neighbours");
}

#[test]
#[ignore]
fn a_c_host_generates_an_inbetween_through_the_core_and_the_stream() {
    let o = fh_engine_options_default();
    let e = unsafe { fh_engine_create(&o) };
    if e.is_null() {
        eprintln!("skipped: {}", last_error());
        return;
    }
    let dir = CString::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models")).unwrap();
    let m = unsafe { fh_model_load(dir.as_ptr()) };
    assert!(!m.is_null(), "{}", last_error());
    let mut json = vec![0u8; 4096];
    let n = unsafe { fh_engine_info(e, json.as_mut_ptr() as *mut c_char, json.len()) };
    assert!(n > 0, "{}", last_error());
    eprintln!("{}", std::str::from_utf8(&json[..n as usize]).unwrap());
    let n = unsafe { fh_model_info(m, json.as_mut_ptr() as *mut c_char, json.len()) };
    assert!(n > 0 && std::str::from_utf8(&json[..n as usize]).unwrap().contains("\"graph_ops\""));

    // the Core: own slots, pair, mid, read back
    let so = fh_session_options_default(W, H);
    let s = unsafe { fh_session_create(e, m, &so) };
    assert!(!s.is_null(), "{}", last_error());
    let (a, b, out) = unsafe { (fh_session_add_input(s), fh_session_add_input(s), fh_session_add_output(s)) };
    assert_eq!((a, b, out), (0, 1, 0));
    let (f0, f1) = (frame(0.0), frame(8.0));
    assert_eq!(unsafe { fh_session_write_rgba(s, 0, f0.as_ptr(), f0.len()) }, 0);
    assert_eq!(unsafe { fh_session_write_rgba(s, 1, f1.as_ptr(), f1.len()) }, 0);
    assert_eq!(unsafe { fh_session_pair(s, 0, 1) }, 0, "{}", last_error());
    assert_eq!(unsafe { fh_session_mid(s, 0.5, 0) }, 0, "{}", last_error());
    let mut got = vec![0u8; (W * H * 4) as usize];
    let n = unsafe { fh_session_read_rgba(s, 0, got.as_mut_ptr(), got.len()) };
    assert_eq!(n as usize, got.len(), "{}", last_error());
    assert_inbetween(&got);
    assert!(unsafe { fh_session_mid(s, 0.5, 9) } < 0 && last_error().contains("bad output slot"));
    let n = unsafe { fh_session_report_json(s, json.as_mut_ptr() as *mut c_char, json.len()) };
    assert!(n > 0 && std::str::from_utf8(&json[..n as usize]).unwrap().contains("\"pair_steps\""));
    unsafe { fh_session_destroy(s) };

    // the Stream: ring slots, classification, push with a timestamp, generate
    let sto = fh_stream_options_default();
    let st = unsafe { fh_stream_create(e, m, &so, &sto) };
    assert!(!st.is_null(), "{}", last_error());
    let s0 = unsafe { fh_stream_next_input(st, std::ptr::null(), 0) };
    assert_eq!(unsafe { fh_stream_write_rgba(st, e, s0 as u32, f0.as_ptr(), f0.len()) }, 0, "{}", last_error());
    let mut push = FhPush { ready: 0, base_slot: 0, stats_pending: 0, start_pts: 0.0, end_pts: 0.0 };
    assert_eq!(unsafe { fh_stream_push(st, e, s0 as u32, 0.0, &mut push) }, 0, "{}", last_error());
    assert_eq!(push.ready, 0, "one frame is no interval");
    let s1 = unsafe { fh_stream_next_input(st, &[s0 as u32] as *const u32, 1) };
    assert!(s1 >= 0 && s1 != s0);
    assert_eq!(unsafe { fh_stream_write_rgba(st, e, s1 as u32, f1.as_ptr(), f1.len()) }, 0);
    let mut c = FhClassification { kind: 9, mean: 0.0, max: 0.0, black: 0, motion: 0.0 };
    assert_eq!(unsafe { fh_stream_classify(st, e, s0 as u32, s1 as u32, &mut c) }, 1, "{}", last_error());
    assert_eq!(c.kind, 1, "an 8 px shift of the soft pattern is a cut at the 0.08 threshold: {c:?}");
    assert_eq!(unsafe { fh_stream_push(st, e, s1 as u32, 41.7, &mut push) }, 0, "{}", last_error());
    assert_eq!((push.ready, push.base_slot, push.start_pts, push.end_pts), (1, s0, 0.0, 41.7));
    assert_eq!(unsafe { fh_stream_generate(st, e, 0.0, 0) }, 0, "the source frame is enough at t 0");
    assert_eq!(unsafe { fh_stream_generate(st, e, 0.5, 0) }, 1, "{}", last_error());
    let n = unsafe { fh_stream_read_rgba(st, e, 0, got.as_mut_ptr(), got.len()) };
    assert_eq!(n as usize, got.len(), "{}", last_error());
    assert_inbetween(&got);
    unsafe {
        fh_stream_destroy(st);
        fh_model_destroy(m);
        fh_engine_destroy(e);
    }
}

#[test]
fn a_malformed_model_is_an_error_message_over_c() {
    let graph = br#"{"ops": [{"op": "conv", "in": ["a"], "out": ["b"], "w": ["w0"]}]}"#;
    let manifest = br#"{"tensors": {"w0": {"shape": [2, "3"], "offset": 0}}}"#;
    let weights = [0u8; 64];
    let m = unsafe {
        fh_model_from_bytes(
            graph.as_ptr(),
            graph.len(),
            manifest.as_ptr(),
            manifest.len(),
            weights.as_ptr(),
            weights.len(),
        )
    };
    assert!(m.is_null());
    assert!(last_error().contains("tensor w0"), "{}", last_error());
}

#[test]
fn a_panic_inside_a_call_becomes_an_error_not_an_abort() {
    // a NUL in the path cannot become a CString: the one way to make fh_model_load panic from the outside
    let bad = CString::new("x").unwrap();
    let m = unsafe { fh_model_load(bad.as_ptr()) };
    assert!(m.is_null());
    let r = flyhigh::guard(-1, || -> i32 { panic!("boom") });
    assert_eq!(r, -1);
    assert!(last_error().contains("boom"), "{}", last_error());
}
