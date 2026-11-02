//! Replays a `trace v1` recording through the Pacer and prints what it decided, second by second.
use anyhow::{Context, Result};
use clap::Args as ClapArgs;
use flyhigh_engine::pacer::trace::{replay, Trace};
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
    let r = replay(&trace, &opts)?;
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
    Ok(())
}
