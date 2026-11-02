// Golden plans: the structure of every plan the engine builds for a fixed set of sessions, compared with a text
// fixture (`tests/golden/<config>.<class>.txt`). A refactor that drops a step, picks another kernel, changes a
// buffer size or a parameter word fails here, on any GPU, before parity or quality are run. Autotune is off, so the
// kernel routing is fixed per class: `tc-f16` (tensor cores), `portable-f16`, `portable-f32` (any device).
// Ignored by default (needs a wgpu adapter; lavapipe suffices): `cargo test --release -- --ignored`.
// After a deliberate change: `FH_UPDATE_GOLDENS=1 cargo test --release --test golden_plans -- --ignored`.
use flyhigh_engine::api::{default_internal, Engine};
use flyhigh_engine::gen_tex::{GenTex, TexMode, TexOptions};
use flyhigh_engine::{plan_dump, EngineOptions, NetOpts, PlanSession, Precision, SessionExt};
use std::path::PathBuf;

struct Cfg {
    name: &'static str,
    w: u32,
    h: u32,
    k: u32,
    internal: Option<(u32, u32)>,
    repair: u32,
    keep_static: bool,
    fast: bool,
    linear: bool,
}
const RT: Cfg =
    Cfg { name: "", w: 1280, h: 720, k: 1, internal: None, repair: 1, keep_static: false, fast: false, linear: false };
const CONFIGS: [Cfg; 7] = [
    Cfg { name: "rt-720p-default", ..RT },
    Cfg { name: "rt-720p-repair0", repair: 0, ..RT },
    Cfg { name: "rt-720p-repair2", repair: 2, ..RT },
    Cfg { name: "rt-720p-fast-static", keep_static: true, fast: true, ..RT },
    Cfg { name: "rt-720p-full-fast-linear", internal: Some((1280, 720)), fast: true, linear: true, ..RT },
    Cfg { name: "rt-1080p-fast-linear-static", w: 1920, h: 1080, keep_static: true, fast: true, linear: true, ..RT },
    Cfg { name: "rt-480p-k2", w: 856, h: 480, k: 2, ..RT },
];

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("models")
}
fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("golden")
}

// the realtime texture path as the hosts drive it (web.rs `ext_*`, gen_tex.rs): pair on slots 0 / 1, one mid into slot 0
fn dump_ext(e: &Engine, c: &Cfg) -> String {
    let (iw, ih) = c.internal.unwrap_or_else(|| default_internal(c.w, c.h));
    let mut net = NetOpts::new(&e.blob);
    net.repair = c.repair;
    net.keep_static = c.keep_static;
    net.fast_render = c.fast;
    let mut s = SessionExt::new(&e.gpu, &e.blob, &e.ops, c.w, c.h, c.k, iw, ih, None, &net);
    s.add_input(&e.gpu);
    s.add_input(&e.gpu);
    s.add_output(&e.gpu);
    let (a, b, o) = ((0, &s.ins[0].1), (1, &s.ins[1].1), (0, &s.outs[0].1));
    let pair = s.pair_steps(&e.gpu, 0, 1);
    let mid = if !s.is_fast() {
        s.mid_steps(&e.gpu, 0, o.1)
    } else if c.linear {
        vec![s.render_tex_step(&e.gpu, a, b, o, Some(0.25))]
    } else {
        s.mid_steps_fast(&e.gpu, a, b, o)
    };
    format!(
        "# {} {}x{} k={} internal={}x{} repair={} static={} fast={} linear={}\n# pair\n{}# mid\n{}",
        c.name,
        c.w,
        c.h,
        c.k,
        iw,
        ih,
        c.repair,
        c.keep_static,
        c.fast,
        c.linear,
        plan_dump(&pair),
        plan_dump(&mid)
    )
}

// video mode with cadence: three sessions with the GMHS maps kept; they must be identical
fn dump_video(e: &Engine) -> String {
    let mut o = TexOptions::new(1280, 720);
    o.mode = TexMode::Video;
    o.cadence = true;
    let g = GenTex::new(&e.gpu, &e.blob, &e.ops, o, None).expect("video session");
    let d = |s: &SessionExt| format!("# pair\n{}# mid\n{}", plan_dump(&s.plan_pair), plan_dump(&s.plan_mid));
    assert_eq!(g.sess.len(), 3);
    assert!(d(&g.sess[1]) == d(&g.sess[0]) && d(&g.sess[2]) == d(&g.sess[0]), "the three video sessions differ");
    format!("# video-720p-cadence 1280x720 k=1 repair>=1 cadence\n{}", d(&g.sess[0]))
}

// the CPU path (rgba buffers in and out), parity-fixture size
fn dump_cpu(e: &Engine) -> String {
    let s = PlanSession::new(&e.gpu, &e.blob, &e.ops, 256, 192, 1, 0.5);
    format!("# cpu-256x192 k=1 t=0.5\n{}", plan_dump(&s.plan))
}

fn check(name: &str, class: &str, actual: &str) {
    let path = golden_dir().join(format!("{name}.{class}.txt"));
    if std::env::var_os("FH_UPDATE_GOLDENS").is_some() {
        std::fs::create_dir_all(golden_dir()).unwrap();
        std::fs::write(&path, actual).unwrap();
        eprintln!("wrote {}", path.display());
        return;
    }
    let Ok(expected) = std::fs::read_to_string(&path) else {
        // every device can build the portable f32 plans, so that golden has to exist
        assert!(class != "portable-f32", "missing golden {}", path.display());
        eprintln!("no golden {} for this device class, skipped", path.display());
        return;
    };
    if expected.replace("\r\n", "\n") == actual {
        eprintln!("{name}.{class}: {} lines, identical", actual.lines().count());
        return;
    }
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target").join("golden-actual");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join(format!("{name}.{class}.txt")), actual).unwrap();
    let (el, al): (Vec<&str>, Vec<&str>) = (expected.lines().collect(), actual.lines().collect());
    let i = el.iter().zip(&al).position(|(x, y)| x != y).unwrap_or(el.len().min(al.len()));
    panic!("golden plan {name}.{class} differs at line {} ({} vs {} lines):\n  expected: {}\n  actual:   {}\n(actual written to {})",
           i + 1, el.len(), al.len(), el.get(i).unwrap_or(&"<end>"), al.get(i).unwrap_or(&"<end>"), out.display());
}

#[test]
#[ignore]
fn plans_match_their_goldens() {
    let mut seen = Vec::new();
    // the three kernel classes; a device without shader-f16 or tensor cores yields fewer (duplicates are skipped)
    for (tensor_cores, precision) in [(false, Precision::F32), (false, Precision::F16), (true, Precision::Auto)] {
        let opts =
            EngineOptions { tensor_cores, precision, autotune: false, tune_cache: None, ..EngineOptions::default() };
        let e = match Engine::load_with(&model_dir(), &opts) {
            Ok(e) => e,
            Err(e) if format!("{e:#}").contains("shader-f16") || format!("{e:#}").contains("adapter") => {
                eprintln!("skipped a class: {e:#}");
                continue;
            }
            Err(e) => panic!("engine: {e:#}"),
        };
        let class =
            format!("{}-{}", if e.gpu.coop { "tc" } else { "portable" }, if e.gpu.f32_mode { "f32" } else { "f16" });
        if seen.contains(&class) {
            continue;
        }
        eprintln!("class {class} on {}", e.gpu.name);
        for c in &CONFIGS {
            check(c.name, &class, &dump_ext(&e, c));
        }
        check("video-720p-cadence", &class, &dump_video(&e));
        check("cpu-256x192", &class, &dump_cpu(&e));
        seen.push(class);
    }
    assert!(!seen.is_empty(), "no device class could be built");
}
