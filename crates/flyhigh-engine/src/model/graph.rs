//! Helpers over the operation graph (`graph.json`): operand names, read counts, the pair / mid boundary.

use serde_json::{Map, Value};
use std::collections::HashMap;

/// The operand names under `key` of an op. A graph that passed [`validate`] has only strings there.
pub fn names(op: &Value, key: &str) -> Vec<String> {
    op[key].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default()
}

/// The weights the FiLM MLP reads by fixed name.
const FILM_WEIGHTS: [&str; 4] = ["film.film.0.weight", "film.film.0.bias", "film.film.2.weight", "film.film.2.bias"];

/// Checks what the builder takes for granted: every op names itself, its operands are strings, a repeat body is a
/// list of ops, and every weight an op reads exists in the manifest; so a model that loads can be built without a
/// panic, whatever directory it came from.
pub(crate) fn validate(ops: &[Value], tensors: &Map<String, Value>) -> anyhow::Result<()> {
    for (i, op) in ops.iter().enumerate() {
        let name = op["op"].as_str().ok_or_else(|| anyhow::anyhow!("graph.json: op {i} has no name"))?;
        for key in ["in", "out", "w"] {
            if let Some(list) = op.get(key) {
                let list = list.as_array().ok_or_else(|| anyhow::anyhow!("graph.json: {name}.{key} is not a list"))?;
                anyhow::ensure!(list.iter().all(Value::is_string), "graph.json: {name}.{key} has a non-string entry");
            }
        }
        for w in names(op, "w") {
            anyhow::ensure!(tensors.contains_key(&w), "graph.json: {name} reads tensor {w}, which the manifest lacks");
        }
        match name {
            "repeat" => {
                let body = op["body"].as_array().ok_or_else(|| anyhow::anyhow!("graph.json: repeat without a body"))?;
                validate(body, tensors)?;
            }
            "film_mlp" => {
                for w in FILM_WEIGHTS {
                    anyhow::ensure!(
                        tensors.contains_key(w),
                        "graph.json: film_mlp needs tensor {w}, which the manifest lacks"
                    );
                }
            }
            _ => {}
        }
    }
    Ok(())
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
