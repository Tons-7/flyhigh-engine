//! `trace v1`: the host's events in JSON, so a recorded session replays through the Pacer bit for bit in a test.
//!
//! ```json
//! { "version": 1, "host": "firefox 159", "notes": "...", "events": [
//!   { "kind": "tick", "t": 16.7 },
//!   { "kind": "source", "t": 20.1, "id": 3, "pts": 41.7, "kind_of": "motion", "motion": 12.5 },
//!   { "kind": "cost", "t": 25.0, "stage": "mid", "ms": 1.8 },
//!   { "kind": "stall", "t": 900.0, "ms": 62.0 },
//!   { "kind": "mode_change", "t": 3000.0 }
//! ] }
//! ```
use super::{Frame, IntervalKind, Pacer, PacerOptions, PacerReport, Stage};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraceEvent {
    Tick {
        t: f64,
    },
    Source {
        t: f64,
        id: u32,
        pts: f64,
        #[serde(default, rename = "kind_of")]
        kind: IntervalKind,
        #[serde(default)]
        motion: Option<f64>,
        #[serde(default)]
        skipped: u32,
    },
    Cost {
        t: f64,
        stage: Stage,
        ms: f64,
    },
    Stall {
        t: f64,
        ms: f64,
    },
    ModeChange {
        t: f64,
    },
    Reset {
        t: f64,
    },
    /// `set_options` while recording; a replay applies them at the same point.
    Options {
        t: f64,
        options: Box<PacerOptions>,
    },
}

impl TraceEvent {
    pub fn t(&self) -> f64 {
        match self {
            TraceEvent::Tick { t }
            | TraceEvent::Source { t, .. }
            | TraceEvent::Cost { t, .. }
            | TraceEvent::Stall { t, .. }
            | TraceEvent::ModeChange { t }
            | TraceEvent::Reset { t }
            | TraceEvent::Options { t, .. } => *t,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trace {
    pub version: u32,
    #[serde(default)]
    pub host: String,
    #[serde(default)]
    pub notes: String,
    /// The options the host ran with; a replay uses them unless given others.
    #[serde(default)]
    pub options: Option<PacerOptions>,
    pub events: Vec<TraceEvent>,
}

impl Trace {
    pub fn new(host: &str) -> Trace {
        Trace { version: 1, host: host.into(), notes: String::new(), options: None, events: Vec::new() }
    }

    pub fn from_json(text: &str) -> Result<Trace, String> {
        let t: Trace = serde_json::from_str(text).map_err(|e| format!("trace: {e}"))?;
        if t.version != 1 {
            return Err(format!("trace: version {} not supported, this engine reads 1", t.version));
        }
        Ok(t)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    pub fn duration_ms(&self) -> f64 {
        let first = self.events.first().map_or(0.0, TraceEvent::t);
        let last = self.events.last().map_or(0.0, TraceEvent::t);
        last - first
    }
}

/// One second of a replay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Second {
    pub presents: u64,
    pub mids: u64,
    pub drops: u64,
    pub requests: u64,
    pub factor: u32,
    pub penalty: u32,
}

/// How long after its arrival each source frame reached the screen, from the third second of the trace on (the first
/// two are the start-up): the latency frame generation adds to the picture (the host's own capture and scanout come
/// on top).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Latency {
    pub mean_ms: f64,
    pub p95_ms: f64,
    pub max_ms: f64,
    /// Source frames presented.
    pub frames: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Replay {
    pub report: PacerReport,
    pub seconds: Vec<Second>,
    pub factor_min: u32,
    pub factor_max: u32,
    pub penalty_max: u32,
    pub max_late_ms: f64,
    pub latency: Latency,
}

/// GPU time a replay charges before a requested frame counts as generated (`replay_with`): one pair per source frame,
/// one frame per request, on a single queue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Compute {
    pub pair_ms: f64,
    pub mid_ms: f64,
}

impl Replay {
    /// Seconds after the first `skip` ones.
    pub fn steady(&self, skip: usize) -> &[Second] {
        &self.seconds[skip.min(self.seconds.len())..]
    }
}

/// Replays a trace through a Pacer standing in for the host: every request is generated at once (the real host
/// queues the blit behind the compute, so from the scheduler's view a submitted frame is a frame) and every present
/// happens on its tick. A pacer records such a trace itself with `Pacer::record`.
pub fn replay(trace: &Trace, opts: &PacerOptions) -> Result<Replay, super::OptionsError> {
    replay_with(trace, opts, None)
}

/// `replay`, optionally charging GPU time: with `compute`, a request is reported generated only once the queue has
/// worked through the pair of its source frame and the requests before it, so a presentation delay too short for the
/// work shows up as late fallbacks and drops.
pub fn replay_with(
    trace: &Trace,
    opts: &PacerOptions,
    compute: Option<Compute>,
) -> Result<Replay, super::OptionsError> {
    let mut pacer = Pacer::new(opts)?;
    let mut arrivals: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
    let mut latencies: Vec<f64> = Vec::new();
    let mut gpu_free = f64::NEG_INFINITY;
    let mut done: std::collections::VecDeque<(f64, u64)> = std::collections::VecDeque::new();
    let t0 = trace.events.first().map_or(0.0, TraceEvent::t);
    let mut seconds: Vec<Second> = Vec::new();
    let mut outputs = 0u32;
    let mut factor_min = u32::MAX;
    let mut factor_max = 0;
    let mut penalty_max = 0;
    let mut max_late = 0.0f64;
    let mut generate = |pacer: &mut Pacer,
                        now: f64,
                        reqs: Vec<super::Request>,
                        sec: &mut Second,
                        done: &mut std::collections::VecDeque<(f64, u64)>,
                        gpu_free: &mut f64| {
        for r in reqs {
            sec.requests += 1;
            match compute {
                Some(c) => {
                    *gpu_free = gpu_free.max(now) + c.mid_ms;
                    done.push_back((*gpu_free, r.id));
                }
                None => {
                    outputs = outputs.wrapping_add(1);
                    pacer.generated(now, r.id, Frame::Output(outputs));
                }
            }
        }
    };
    let mut next_output = 0u32;
    for e in &trace.events {
        let now = e.t();
        while done.front().is_some_and(|(at, _)| *at <= now) {
            let (at, id) = done.pop_front().unwrap();
            next_output = next_output.wrapping_add(1);
            pacer.generated(at, id, Frame::Output(u32::MAX - next_output));
        }
        let s = ((now - t0) / 1000.0).floor().max(0.0) as usize;
        while seconds.len() <= s {
            seconds.push(Second::default());
        }
        match *e {
            TraceEvent::Tick { t } => {
                let tick = pacer.display_tick(t);
                let sec = &mut seconds[s];
                if let Some(p) = tick.present {
                    sec.presents += 1;
                    if p.mid {
                        sec.mids += 1;
                    }
                    max_late = max_late.max(p.late_ms);
                    if let (Frame::Source(id), false) = (p.frame, p.mid) {
                        if let Some(a) = arrivals.remove(&id) {
                            if a - t0 >= 2000.0 {
                                latencies.push(t - a);
                            }
                        }
                    }
                }
                sec.drops += tick.dropped as u64;
                sec.factor = tick.factor;
                sec.penalty = pacer.penalty();
                factor_min = factor_min.min(tick.factor);
                factor_max = factor_max.max(tick.factor);
                penalty_max = penalty_max.max(pacer.penalty());
                generate(&mut pacer, t, tick.generate, sec, &mut done, &mut gpu_free);
            }
            TraceEvent::Source { t, id, pts, kind, motion, skipped } => {
                arrivals.insert(id, t);
                if let Some(c) = compute {
                    gpu_free = gpu_free.max(t) + c.pair_ms;
                }
                let flush = pacer.source_frame(t, super::SourceFrame { id, pts_ms: pts, kind, motion, skipped });
                generate(&mut pacer, t, flush, &mut seconds[s], &mut done, &mut gpu_free);
            }
            TraceEvent::Cost { t, stage, ms } => {
                pacer.cost_sample(t, stage, ms);
            }
            TraceEvent::Stall { t, ms } => pacer.host_stall(t, ms),
            TraceEvent::ModeChange { t } => pacer.mode_change(t),
            TraceEvent::Reset { t } => pacer.reset(t),
            TraceEvent::Options { t, ref options } => pacer.set_options(t, options)?,
        }
    }
    if factor_min == u32::MAX {
        factor_min = 0;
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let latency = match latencies.len() {
        0 => Latency::default(),
        n => Latency {
            mean_ms: latencies.iter().sum::<f64>() / n as f64,
            p95_ms: latencies[(n * 95 / 100).min(n - 1)],
            max_ms: latencies[n - 1],
            frames: n as u64,
        },
    };
    Ok(Replay { report: pacer.report(), seconds, factor_min, factor_max, penalty_max, max_late_ms: max_late, latency })
}
