//! Replays a `trace v1` recording through the Pacer and prints what it decided, second by second.
use anyhow::{Context, Result};
use clap::Args as ClapArgs;
use flyhigh_engine::pacer::trace::{replay_with, Compute, Trace};
use flyhigh_engine::pacer::PacerOptions;
use std::path::PathBuf;

#[derive(ClapArgs)]
pub struct Args {
    /// The trace (JSON, `trace v1`)
    pub trace: PathBuf,
    /// Pacer options (JSON, any subset of fields); default: the trace's own, else the defaults
    #[arg(long)]
    pub options: Option<PathBuf>,
    /// Print the whole replay (report and per-second table) as JSON
    #[arg(long)]
    pub json: bool,
    /// Charge this much GPU time per source frame (the pair) before its requests count as generated
    #[arg(long)]
    pub pair_ms: Option<f64>,
    /// Charge this much GPU time per generated frame
    #[arg(long)]
    pub mid_ms: Option<f64>,
}

pub fn run(a: Args) -> Result<()> {
    let text = std::fs::read_to_string(&a.trace).with_context(|| format!("reading {}", a.trace.display()))?;
    let trace = Trace::from_json(&text).map_err(anyhow::Error::msg)?;
    let opts: PacerOptions = match &a.options {
        Some(p) => {
            let text = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
            serde_json::from_str(&text).with_context(|| format!("options in {}", p.display()))?
        }
        None => trace.options.clone().unwrap_or_default(),
    };
    let compute = (a.pair_ms.is_some() || a.mid_ms.is_some())
        .then(|| Compute { pair_ms: a.pair_ms.unwrap_or(0.0), mid_ms: a.mid_ms.unwrap_or(0.0) });
    let r = replay_with(&trace, &opts, compute)?;
    if a.json {
        println!("{}", serde_json::to_string_pretty(&r)?);
        return Ok(());
    }
    println!("{} ({:.1} s, {} events)", trace.host, trace.duration_ms() / 1000.0, trace.events.len());
    println!(
        "{:>4} {:>8} {:>6} {:>6} {:>8} {:>6} {:>7}",
        "s", "presents", "mids", "drops", "requests", "factor", "penalty"
    );
    for (i, s) in r.seconds.iter().enumerate() {
        println!(
            "{i:>4} {:>8} {:>6} {:>6} {:>8} {:>6} {:>7}",
            s.presents, s.mids, s.drops, s.requests, s.factor, s.penalty
        );
    }
    let c = &r.report.counters;
    println!(
        "factor {}..{}, penalty max {}, late max {:.1} ms; drops {} (stall-exempt {}, settle-exempt {}), late fallbacks {}, delay {:.0} ms",
        r.factor_min,
        r.factor_max,
        r.penalty_max,
        r.max_late_ms,
        c.drops,
        c.drops_stall_exempt,
        c.drops_settle_exempt,
        c.late_fallbacks,
        r.report.delay_ms
    );
    println!(
        "source frames on screen after arrival: mean {:.1} ms, p95 {:.1}, max {:.1} ({} frames)",
        r.latency.mean_ms, r.latency.p95_ms, r.latency.max_ms, r.latency.frames
    );
    Ok(())
}
