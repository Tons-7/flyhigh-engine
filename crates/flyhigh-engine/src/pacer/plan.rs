//! Presentations on an exact output grid: every target tick becomes one presentation, an anchor on a source frame
//! or an interpolation at phase t.
use super::rate::MAX_TICKS_PER_INTERVAL;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Previous,
    Current,
    Interpolate,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slot {
    pub at: f64,
    pub t: f64,
    pub kind: Kind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GridPlan {
    pub next_at: f64,
    pub next_phase_ms: f64,
    pub slots: Vec<Slot>,
    pub step_ms: f64,
    pub resynced: bool,
    pub overflowed: bool,
}

/// Interpolations everywhere except the tick nearest a source frame, which stays that source frame.
pub fn assign_kinds(slots: &mut [Slot], interpolate: bool) {
    for s in slots.iter_mut() {
        s.kind = if interpolate {
            Kind::Interpolate
        } else if s.t < 0.5 {
            Kind::Previous
        } else {
            Kind::Current
        };
    }
    if interpolate && !slots.is_empty() {
        let mut anchor = 0;
        let mut distance = slots[0].t.abs().min((1.0 - slots[0].t).abs());
        for (i, s) in slots.iter().enumerate().skip(1) {
            let d = s.t.abs().min((1.0 - s.t).abs());
            if d < distance {
                anchor = i;
                distance = d;
            }
        }
        slots[anchor].kind = if slots[anchor].t <= 0.5 { Kind::Previous } else { Kind::Current };
    }
}

/// The GPU could not afford the interpolations: keep the cadence and fail safe to the nearest anchor instead of an
/// off-grid frame.
pub fn fallback_to_anchors(slots: &mut [Slot]) {
    for s in slots.iter_mut() {
        if s.kind == Kind::Interpolate {
            s.kind = if s.t < 0.5 { Kind::Previous } else { Kind::Current };
        }
    }
}

/// Decoded media time decides how many target ticks belong to a source pair; wall time only anchors the absolute
/// output grid. Keeping the phases apart stops callback jitter from turning nominal 60 to 120 into third-of-a-tick
/// bursts while every deadline stays exactly one step apart. `next_at` 0 starts a new grid.
pub fn plan_source_cadence(
    next_at: f64,
    phase_ms: f64,
    start_at: f64,
    source_interval_ms: f64,
    output_hz: f64,
    interpolate: bool,
) -> GridPlan {
    let step = 1000.0 / output_hz;
    let mut phase = phase_ms;
    let mut at = next_at;
    let mut resynced = false;
    if at <= 0.0 {
        phase = step;
        at = start_at + step;
        resynced = true;
    } else {
        // keep the absolute clock near the wall-time place of this media phase: a stall may move start_at by many
        // periods; move the cursor by whole periods only, so the grid and the phase survive without a backlog of
        // stale deadlines
        let desired = start_at + phase;
        let epsilon = (step * 1e-9).max(1e-7);
        if desired - at > step + epsilon {
            let shift = ((desired - at) / step - 1e-10).ceil();
            at += shift * step;
            resynced = true;
        } else if at - desired > step + epsilon {
            // never rewind the clock (that would re-emit a scheduled deadline): carry its lead as media phase
            phase = (at - start_at).max(0.0);
            resynced = true;
        }
    }
    let mut slots = Vec::new();
    let stop_before = source_interval_ms - (step * 1e-9).max(1e-7);
    while phase < stop_before && slots.len() < MAX_TICKS_PER_INTERVAL {
        slots.push(Slot { at, t: phase / source_interval_ms, kind: Kind::Interpolate });
        phase += step;
        at += step;
    }
    let overflowed = phase < stop_before;
    if overflowed {
        return GridPlan {
            next_at: start_at + source_interval_ms + step,
            next_phase_ms: step,
            slots: Vec::new(),
            step_ms: step,
            resynced,
            overflowed,
        };
    }
    assign_kinds(&mut slots, interpolate);
    GridPlan {
        next_at: at,
        next_phase_ms: (phase - source_interval_ms).max(0.0),
        slots,
        step_ms: step,
        resynced,
        overflowed,
    }
}
