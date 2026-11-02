//! The GPU cost of one stage (pair, mid, present) as the scheduler believes it.
use std::collections::VecDeque;

/// Median of the last 16 timestamp samples, joined with attack and slow release: GPU timestamps exclude queue
/// backlog, so a slower content-dependent sample is real work and must tighten admission at once; relaxation is
/// gradual so one fast scene does not re-inflate the factor.
#[derive(Default, Debug, Clone)]
pub struct RollingCost {
    samples: VecDeque<f64>,
    value: f64,
}

impl RollingCost {
    /// Samples above this are measurement artefacts (a timestamp pair spanning a stall) and are ignored.
    pub const MAX_SAMPLE_MS: f64 = 80.0;

    pub fn push(&mut self, ms: f64) {
        if !(ms > 0.0 && ms <= Self::MAX_SAMPLE_MS) {
            return;
        }
        self.samples.push_back(ms);
        if self.samples.len() > 16 {
            self.samples.pop_front();
        }
        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[(sorted.len() - 1) / 2];
        self.value = if self.value == 0.0 {
            median
        } else if ms > self.value {
            self.value * 0.5 + ms * 0.5
        } else {
            self.value * 0.9 + median * 0.1
        };
    }

    /// 0 until the first sample.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// The value, or `seed` until the first sample.
    pub fn or(&self, seed: f64) -> f64 {
        if self.value > 0.0 {
            self.value
        } else {
            seed
        }
    }

    pub fn reset(&mut self) {
        self.samples.clear();
        self.value = 0.0;
    }
}
