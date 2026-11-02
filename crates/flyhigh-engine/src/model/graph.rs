//! Helpers over the operation graph (`graph.json`): operand names, read counts, the pair / mid boundary.

use serde_json::Value;
use std::collections::HashMap;

pub fn names(op: &Value, key: &str) -> Vec<String> {
    op[key].as_array().map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect()).unwrap_or_default()
}

// count how many ops read each tensor name (recursing into repeat bodies once: peepholes and dead-output detection)
pub(crate) fn count_uses(ops: &[Value], uses: &mut HashMap<String, usize>) {
    for op in ops {
        for n in names(op, "in") {
            *uses.entry(n).or_insert(0) += 1;
        }
        if op["op"].as_str() == Some("repeat") {
            if let Some(b) = op["body"].as_array() {
                count_uses(b, uses);
            }
        }
    }
}
// the same over the UNROLLED plan (repeat bodies K times): how many reads each name still has (buffer liveness)
pub(crate) fn count_uses_unrolled(ops: &[Value], k: u32, uses: &mut HashMap<String, usize>) {
    for op in ops {
        for n in names(op, "in") {
            *uses.entry(n).or_insert(0) += 1;
        }
        if op["op"].as_str() == Some("repeat") {
            if let Some(b) = op["body"].as_array() {
                for _ in 0..k {
                    count_uses_unrolled(b, k, uses);
                }
            }
        }
    }
}
// the first op of the t-dependent (per generated frame) stage: everything from here on is rebuilt per mid in a split plan
pub fn mid_start(ops: &[Value]) -> usize {
    ops.iter()
        .position(|o| o["op"] == "mul_t" || o["op"] == "film_mlp" || names(o, "in").iter().any(|n| n == "t"))
        .unwrap_or(ops.len())
}
