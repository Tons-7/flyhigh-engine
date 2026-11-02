// The C surface of the Pacer against the Rust one: the options map both ways without loss, and a recorded host trace
// driven through the C functions (as a C host would) ends in the same report as the Rust replay. No GPU.
use flyhigh::{
    fh_pacer_create, fh_pacer_destroy, fh_pacer_generated, fh_pacer_options_default, fh_pacer_report_json,
    fh_pacer_requests, fh_pacer_source_frame, fh_pacer_tick, FhPacerOptions, FhRequest, FhSourceFrame, FhTick,
};
use flyhigh_engine::pacer::trace::{Trace, TraceEvent};
use flyhigh_engine::pacer::{Frame, IntervalKind, Pacer, PacerOptions, RateMode};

#[test]
fn options_round_trip_through_the_c_struct() {
    let c = fh_pacer_options_default();
    assert_eq!(c.to_rust().unwrap(), PacerOptions::default());
    assert_eq!((c.mode, c.max_factor, c.queue_capacity, c.limit_to_display), (0, 6, 24, 1));
    for o in [
        PacerOptions { mode: RateMode::Factor(3), ..PacerOptions::default() },
        PacerOptions { mode: RateMode::Target(90.0), fps_limit: Some(72.0), ..PacerOptions::default() },
        PacerOptions {
            mode: RateMode::DisplayHz,
            lookahead: true,
            min_present_interval_ms: 7.0,
            ..PacerOptions::default()
        },
    ] {
        assert_eq!(FhPacerOptions::from_rust(&o).to_rust().unwrap(), o);
    }
    let bad = FhPacerOptions { mode: 9, ..fh_pacer_options_default() };
    assert!(bad.to_rust().is_err());
}

fn recorded() -> Trace {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/traces/chrome-anime-auto.json");
    Trace::from_json(&std::fs::read_to_string(path).expect("the recorded trace")).expect("trace v1")
}

#[test]
fn a_recorded_trace_through_the_c_api_matches_the_rust_pacer() {
    let trace = recorded();
    let opts = trace.options.clone().expect("the host recorded its options");
    let mut rust = Pacer::new(&opts).unwrap();
    let c_opts = FhPacerOptions::from_rust(&opts);
    let c = unsafe { fh_pacer_create(&c_opts) };
    assert!(!c.is_null(), "{}", unsafe { std::ffi::CStr::from_ptr(flyhigh::fh_last_error()) }.to_string_lossy());
    let mut buf = vec![FhRequest { id: 0, previous: 0, current: 0, t: 0.0, due_ms: 0.0, retime: 0, probe: 0 }; 64];
    let (mut rust_out, mut c_out) = (0u32, 0u32);
    let mut generate_c = |now: f64, n: i32| {
        assert!(n >= 0);
        let got = unsafe { fh_pacer_requests(c, buf.as_mut_ptr(), buf.len()) };
        assert_eq!(got, n);
        for r in &buf[..n as usize] {
            c_out += 1;
            assert_eq!(unsafe { fh_pacer_generated(c, now, r.id, c_out) }, 0);
        }
    };
    let (mut presents, mut c_presents) = (0u64, 0u64);
    for e in &trace.events {
        match *e {
            TraceEvent::Options { t, ref options } => {
                rust.set_options(t, options).unwrap();
                let o = FhPacerOptions::from_rust(options);
                assert_eq!(unsafe { flyhigh::fh_pacer_set_options(c, t, &o) }, 0);
            }
            TraceEvent::Tick { t } => {
                let tick = rust.display_tick(t);
                presents += tick.present.is_some() as u64;
                for r in tick.generate {
                    rust_out += 1;
                    rust.generated(t, r.id, Frame::Output(rust_out));
                }
                let mut out = FhTick {
                    has_present: 0,
                    frame_kind: 0,
                    frame_id: 0,
                    due_ms: 0.0,
                    late_ms: 0.0,
                    mid: 0,
                    fallback: 0,
                    factor: 0,
                    dropped: 0,
                    requests: 0,
                };
                assert_eq!(unsafe { fh_pacer_tick(c, t, &mut out) }, 0);
                c_presents += out.has_present as u64;
                generate_c(t, out.requests as i32);
            }
            TraceEvent::Source { t, id, pts, kind, motion, skipped } => {
                let flush =
                    rust.source_frame(t, flyhigh_engine::pacer::SourceFrame { id, pts_ms: pts, kind, motion, skipped });
                for r in flush {
                    rust_out += 1;
                    rust.generated(t, r.id, Frame::Output(rust_out));
                }
                let f = FhSourceFrame {
                    id,
                    pts_ms: pts,
                    kind: match kind {
                        IntervalKind::Motion => 0,
                        IntervalKind::Cut => 1,
                        IntervalKind::Hold => 2,
                    },
                    has_motion: motion.is_some() as u32,
                    motion: motion.unwrap_or(0.0),
                    skipped,
                };
                let n = unsafe { fh_pacer_source_frame(c, t, &f) };
                generate_c(t, n);
            }
            TraceEvent::Cost { t, stage, ms } => {
                rust.cost_sample(t, stage, ms);
                let s = match stage {
                    flyhigh_engine::pacer::Stage::Pair => 0,
                    flyhigh_engine::pacer::Stage::Mid => 1,
                    flyhigh_engine::pacer::Stage::Present => 2,
                };
                assert!(unsafe { flyhigh::fh_pacer_cost_sample(c, t, s, ms) } >= 0);
            }
            TraceEvent::Stall { t, ms } => {
                rust.host_stall(t, ms);
                assert_eq!(unsafe { flyhigh::fh_pacer_host_stall(c, t, ms) }, 0);
            }
            TraceEvent::ModeChange { t } => {
                rust.mode_change(t);
                assert_eq!(unsafe { flyhigh::fh_pacer_mode_change(c, t) }, 0);
            }
            TraceEvent::Reset { t } => {
                rust.reset(t);
                assert_eq!(unsafe { flyhigh::fh_pacer_reset(c, t) }, 0);
            }
        }
    }
    assert_eq!(presents, c_presents);
    assert!(presents > 2500, "the session presented frames: {presents}");
    let mut json = vec![0u8; 1 << 16];
    let n = unsafe { fh_pacer_report_json(c, json.as_mut_ptr() as *mut std::ffi::c_char, json.len()) };
    assert!(n > 0, "{}", unsafe { std::ffi::CStr::from_ptr(flyhigh::fh_last_error()) }.to_string_lossy());
    let c_report = std::str::from_utf8(&json[..n as usize]).unwrap();
    assert_eq!(c_report, rust.report().to_json(), "the two surfaces decided the same");
    unsafe { fh_pacer_destroy(c) };
}
