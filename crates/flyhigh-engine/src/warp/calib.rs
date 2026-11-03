//! Input calibration of the camera warp: whether the mouse turns the camera at all, how many pixels of image shift a
//! count gives, and how late the picture shows it. Pure logic on the host's clock: mouse counts in, the measured image
//! shift between consecutive source frames in, a shift to apply to a frame out.
//!
//! The picture of a streamed or captured game shows the input of `lag` ms before its arrival (the network, the encoder,
//! the game's own frame). The shift between two frames is therefore explained by the counts of the same interval moved
//! back by `lag`; the calibrator searches the lag that explains the recent shifts best and fits the gain through the
//! origin. It turns the warp on only when the fit is strong, and off at once when the mouse moved and the picture did
//! not (a menu, a map, a game whose mouse drives a cursor), or when the host says the mouse is not captured.
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibOptions {
    /// Frames the fit looks back over.
    pub window_ms: f64,
    /// The latest lag searched; a cloud stream with a slow network can show input 150 ms late.
    pub max_lag_ms: f64,
    pub lag_step_ms: f64,
    /// Correlation that turns the warp on, and the one below which it turns off.
    pub on_r: f64,
    pub off_r: f64,
    /// The largest shift the warp applies, in pixels: a misfit never throws the picture across the screen.
    pub max_shift_px: f64,
}

impl Default for CalibOptions {
    fn default() -> Self {
        CalibOptions {
            window_ms: 2000.0,
            max_lag_ms: 200.0,
            lag_step_ms: 2.0,
            on_r: 0.9,
            off_r: 0.6,
            max_shift_px: 400.0,
        }
    }
}

/// What the calibrator believes now.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Calibration {
    pub on: bool,
    /// The host captured the mouse (relative input, no cursor).
    pub captured: bool,
    /// Correlation of the last fit between counts and image shifts.
    pub r: f64,
    pub lag_ms: f64,
    /// Image pixels per count, horizontal and vertical (signed: a game may invert either axis).
    pub gain_x: f64,
    pub gain_y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Pair {
    t0: f64,
    t1: f64,
    sx: f64,
    sy: f64,
}

pub struct InputCalibrator {
    opts: CalibOptions,
    /// Cumulative counts at each mouse event (time, x, y), oldest first.
    mouse: VecDeque<(f64, f64, f64)>,
    pairs: VecDeque<Pair>,
    last_frame: Option<f64>,
    last_fit: f64,
    misses: u32,
    /// After a fast off, the fit only looks at pairs from here on: the ones before still say "on".
    fresh_from: f64,
    state: Calibration,
}

impl InputCalibrator {
    pub fn new(opts: CalibOptions) -> InputCalibrator {
        InputCalibrator {
            opts,
            mouse: VecDeque::new(),
            pairs: VecDeque::new(),
            last_frame: None,
            last_fit: f64::NEG_INFINITY,
            misses: 0,
            fresh_from: f64::NEG_INFINITY,
            state: Calibration { captured: true, ..Calibration::default() },
        }
    }

    pub fn report(&self) -> Calibration {
        self.state
    }

    /// Mouse counts at host time `t` (relative input, as the game receives it).
    pub fn mouse(&mut self, t: f64, dx: f64, dy: f64) {
        let (x, y) = self.mouse.back().map_or((0.0, 0.0), |m| (m.1, m.2));
        if self.mouse.is_empty() {
            self.mouse.push_back((t - 1.0, 0.0, 0.0));
        }
        self.mouse.push_back((t, x + dx, y + dy));
        let keep = self.opts.window_ms + self.opts.max_lag_ms + 1000.0;
        while self.mouse.len() > 2 && self.mouse[1].0 < t - keep {
            self.mouse.pop_front();
        }
    }

    /// Whether the host has the mouse captured; released (a cursor on screen) turns the warp off at once.
    pub fn set_captured(&mut self, captured: bool) {
        self.state.captured = captured;
        if !captured {
            self.state.on = false;
        }
    }

    /// Counts between host times `a` and `b`.
    pub fn counts_between(&self, a: f64, b: f64) -> (f64, f64) {
        let (xa, ya) = self.cumulative(a);
        let (xb, yb) = self.cumulative(b);
        (xb - xa, yb - ya)
    }

    fn cumulative(&self, t: f64) -> (f64, f64) {
        let m = &self.mouse;
        if m.is_empty() || t <= m[0].0 {
            return m.front().map_or((0.0, 0.0), |e| (e.1, e.2));
        }
        if t >= m[m.len() - 1].0 {
            let e = m[m.len() - 1];
            return (e.1, e.2);
        }
        let i = m.partition_point(|e| e.0 <= t);
        let (p, q) = (m[i - 1], m[i]);
        let f = (t - p.0) / (q.0 - p.0).max(1e-9);
        (p.1 + f * (q.1 - p.1), p.2 + f * (q.2 - p.2))
    }

    /// A source frame arrived at host time `t`; `shift` is the image shift (pixels, x and y) against the previous one,
    /// None when it could not be measured (a cut, a flat picture).
    pub fn frame(&mut self, t: f64, shift: Option<(f64, f64)>) {
        if let (Some(t0), Some((sx, sy))) = (self.last_frame, shift) {
            self.check(t0, t, sx);
            self.pairs.push_back(Pair { t0, t1: t, sx, sy });
            while self.pairs.front().is_some_and(|p| p.t1 < t - self.opts.window_ms) {
                self.pairs.pop_front();
            }
        }
        self.last_frame = Some(t);
        if t - self.last_fit >= 100.0 {
            self.last_fit = t;
            self.fit();
        }
    }

    // Every pair: with the current gain and lag, did the picture move the way the mouse said?
    fn check(&mut self, t0: f64, t1: f64, sx: f64) {
        if !self.state.on {
            return;
        }
        let pred = self.state.gain_x * self.counts_between(t0 - self.state.lag_ms, t1 - self.state.lag_ms).0;
        if pred.abs() > 5.0 && sx.abs() < 0.3 * pred.abs() {
            self.misses += 1;
        } else {
            self.misses = 0;
        }
        if self.misses >= 2 {
            self.state.on = false;
            self.misses = 0;
            self.fresh_from = t1;
        }
    }

    fn fit(&mut self) {
        let pairs: Vec<Pair> = self.pairs.iter().copied().filter(|p| p.t0 >= self.fresh_from).collect();
        if pairs.len() < 30 {
            return;
        }
        let mut best: Option<(f64, f64, f64)> = None; // (r, lag, gain_x)
        let mut lag = 0.0;
        while lag <= self.opts.max_lag_ms {
            let x: Vec<f64> = pairs.iter().map(|p| self.counts_between(p.t0 - lag, p.t1 - lag).0).collect();
            let y: Vec<f64> = pairs.iter().map(|p| p.sx).collect();
            if let Some((r, g)) = robust_fit(&x, &y) {
                if best.is_none_or(|b| r > b.0) {
                    best = Some((r, lag, g));
                }
            }
            lag += self.opts.lag_step_ms;
        }
        let Some((r, lag, gx)) = best else { return };
        self.state.r = r;
        self.state.lag_ms = lag;
        self.state.gain_x = gx;
        // the vertical gain at the same lag; when the player barely moved vertically, keep the last one (or the
        // horizontal one: the same sensitivity, unknown sign until seen)
        let x: Vec<f64> = pairs.iter().map(|p| self.counts_between(p.t0 - lag, p.t1 - lag).1).collect();
        let y: Vec<f64> = pairs.iter().map(|p| p.sy).collect();
        match robust_fit(&x, &y) {
            Some((ry, gy)) if ry > self.opts.off_r => self.state.gain_y = gy,
            _ if self.state.gain_y == 0.0 => self.state.gain_y = gx,
            _ => {}
        }
        if !self.state.on && r > self.opts.on_r && self.state.captured {
            self.state.on = true;
        } else if self.state.on && r < self.opts.off_r {
            self.state.on = false;
        }
    }

    /// The shift to apply to a frame that arrived at host time `t_frame` when it is shown at `now`: the counts since
    /// the input the frame shows, through the gain. None while the warp is off.
    pub fn shift(&self, t_frame: f64, now: f64) -> Option<(f64, f64)> {
        if !self.state.on || !self.state.captured {
            return None;
        }
        let (cx, cy) = self.counts_between(t_frame - self.state.lag_ms, now);
        let (sx, sy) = (self.state.gain_x * cx, self.state.gain_y * cy);
        let m = (sx * sx + sy * sy).sqrt();
        let k = if m > self.opts.max_shift_px { self.opts.max_shift_px / m } else { 1.0 };
        Some((sx * k, sy * k))
    }
}

/// Slope through the origin and correlation of `y` on `x`, once refitted without the residuals beyond three times the
/// median (a cut or a mismeasured shift). None when the counts barely moved.
fn robust_fit(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let sxx: f64 = x.iter().map(|v| v * v).sum();
    if x.iter().map(|v| v.abs()).sum::<f64>() < 50.0 || sxx <= 1e-9 {
        return None;
    }
    let g0 = x.iter().zip(y).map(|(a, b)| a * b).sum::<f64>() / sxx;
    let res: Vec<f64> = x.iter().zip(y).map(|(a, b)| (b - g0 * a).abs()).collect();
    let mut sorted = res.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cut = 3.0 * sorted[sorted.len() / 2] + 0.5;
    let keep: Vec<usize> = (0..x.len()).filter(|&i| res[i] < cut).collect();
    if keep.len() < 10 {
        return None;
    }
    let (kx, ky): (Vec<f64>, Vec<f64>) = keep.iter().map(|&i| (x[i], y[i])).unzip();
    let sxx: f64 = kx.iter().map(|v| v * v).sum();
    let g = kx.iter().zip(&ky).map(|(a, b)| a * b).sum::<f64>() / sxx.max(1e-9);
    let n = kx.len() as f64;
    let (mx, my) = (kx.iter().sum::<f64>() / n, ky.iter().sum::<f64>() / n);
    let cov: f64 = kx.iter().zip(&ky).map(|(a, b)| (a - mx) * (b - my)).sum();
    let vx: f64 = kx.iter().map(|a| (a - mx).powi(2)).sum();
    let vy: f64 = ky.iter().map(|b| (b - my).powi(2)).sum();
    let r = if vx > 1e-12 && vy > 1e-12 { cov / (vx * vy).sqrt() } else { 0.0 };
    // the strength of the relation; the sign is the gain's (a game may invert an axis)
    Some((r.abs(), g))
}
