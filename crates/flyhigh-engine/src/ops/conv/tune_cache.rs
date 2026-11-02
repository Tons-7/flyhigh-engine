//! The autotune table: its key, text format and persistence per GPU.

use crate::Gpu;
use std::collections::HashMap;

/// A conv shape the autotuner keys its choice by: (cin, cout, h, w, stride).
pub type TuneKey = (u32, u32, u32, u32, u32);
/// The kernel candidate chosen per conv shape (`CAND_*` in the low nibble, the forced split-K count in the high one).
pub type TuneTable = HashMap<TuneKey, u8>;

impl Gpu {
    /// Merge a build's conv choices into the engine-wide table and persist it when something new was timed.
    pub fn remember_tune(&self, t: &TuneTable) {
        let mut cur = self.tune.borrow_mut();
        let before = cur.len();
        for (k, v) in t {
            cur.entry(*k).or_insert(*v);
        }
        if cur.len() > before {
            self.persist_tune(&cur);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn persist_tune(&self, table: &TuneTable) {
        if let Some(f) = &self.tune_file {
            if let Some(d) = f.parent() {
                let _ = std::fs::create_dir_all(d);
            }
            let _ = std::fs::write(f, format_tune(table));
        }
    }

    /// The browser keeps the table through the host (`engine.autotune`, the `tune` option), not through files.
    #[cfg(target_arch = "wasm32")]
    fn persist_tune(&self, _: &TuneTable) {}
}

/// Parse a persisted autotune table ("cin,cout,h,w,stride=candidate" per line, the text of the `tune` option);
/// malformed lines are skipped.
pub fn parse_tune(text: &str) -> TuneTable {
    let mut t = HashMap::new();
    for line in text.lines() {
        let (k, v) = match line.split_once('=') {
            Some(p) => p,
            None => continue,
        };
        let f: Vec<u32> = k.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if f.len() != 5 {
            continue;
        }
        if let Ok(c) = v.trim().parse::<u8>() {
            t.insert((f[0], f[1], f[2], f[3], f[4]), c);
        }
    }
    t
}
/// The inverse of `parse_tune`.
pub fn format_tune(t: &TuneTable) -> String {
    let mut v: Vec<String> =
        t.iter().map(|((a, b, c, d, e), x)| format!("{},{},{},{},{}={}", a, b, c, d, e, x)).collect();
    v.sort();
    v.join("\n")
}
/// `parse_tune`, None for an empty table (= default kernel routing).
pub fn tune_table_from(text: &str) -> Option<TuneTable> {
    let t = parse_tune(text);
    if t.is_empty() {
        None
    } else {
        Some(t)
    }
}
