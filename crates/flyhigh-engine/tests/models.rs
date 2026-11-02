// The model catalogue (models/models.json) and the shipped manifest (models/flyhighfg.json) agree, and a malformed
// model directory is refused by the loader instead of reaching the builder; no GPU needed.
use flyhigh_engine::Model;
use serde_json::{json, Value};
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("models")
}

#[test]
fn catalogue_matches_the_shipped_manifest() {
    let cat: Value = serde_json::from_slice(&std::fs::read(models_dir().join("models.json")).unwrap()).unwrap();
    let shipped: Vec<&Value> = cat["models"].as_array().unwrap().iter().filter(|m| m["shipped"] == true).collect();
    assert_eq!(shipped.len(), 1, "exactly one shipped model");
    let m = shipped[0];
    let dir = models_dir().join(m["dir"].as_str().unwrap());
    for f in m["files"].as_array().unwrap() {
        assert!(dir.join(f.as_str().unwrap()).exists(), "missing {}", f);
    }
    let man: Value = serde_json::from_slice(&std::fs::read(dir.join("flyhighfg.json")).unwrap()).unwrap();
    let meta = &man["meta"];
    assert_eq!(m["name"], meta["model"]);
    assert_eq!(m["train_step"], meta["train_step"]);
    assert_eq!(m["params"], meta["n_params"]);
    assert_eq!(m["k_range"], meta["K_range"]);
    assert_eq!(m["default_k"], meta["default_K"]);
    assert_eq!(m["render_sampling"], meta["render_sampling"]);
    assert_eq!(m["dtype"], meta["dtype"]);
}

fn load(graph: &Value, manifest: &Value, weights: Vec<u8>) -> Result<usize, String> {
    Model::from_bytes(graph.to_string().as_bytes(), manifest.to_string().as_bytes(), weights)
        .map(|m| m.info().tensors)
        .map_err(|e| e.to_string())
}

fn manifest(tensors: Value) -> Value {
    json!({ "meta": { "model": "t", "format_version": 1 }, "tensors": tensors })
}

#[test]
fn a_well_formed_model_loads_and_the_shipped_one_passes_the_same_checks() {
    let graph = json!({ "ops": [{ "op": "conv", "in": ["a"], "out": ["b"], "w": ["w0", "b0"] }] });
    let man = manifest(json!({ "w0": { "shape": [2, 3], "offset": 0 }, "b0": { "shape": [], "offset": 6 } }));
    assert_eq!(load(&graph, &man, vec![0; 14]), Ok(2));
    assert!(Model::from_dir(&models_dir()).is_ok());
}

#[test]
fn malformed_manifests_are_refused_not_panicked_on() {
    let graph = json!({ "ops": [{ "op": "conv", "in": ["a"], "out": ["b"], "w": ["w0"] }] });
    let cases: [(&str, Value, usize); 7] = [
        ("non-integer dimension", json!({ "w0": { "shape": [2, "3"], "offset": 0 } }), 64),
        ("float dimension", json!({ "w0": { "shape": [2, 1.5], "offset": 0 } }), 64),
        ("zero dimension", json!({ "w0": { "shape": [0], "offset": 32 } }), 64),
        ("negative offset", json!({ "w0": { "shape": [2], "offset": -4 } }), 64),
        ("past the end", json!({ "w0": { "shape": [2], "offset": 31 } }), 64),
        ("overflowing shape", json!({ "w0": { "shape": [u64::MAX, 2], "offset": 0 } }), 64),
        ("overflowing offset", json!({ "w0": { "shape": [2], "offset": u64::MAX } }), 64),
    ];
    for (what, tensors, n) in cases {
        let r = load(&graph, &manifest(tensors), vec![0; n]);
        assert!(r.is_err(), "{what}: loaded");
        assert!(r.unwrap_err().contains("tensor w0"), "{what}: the error names the tensor");
    }
    assert!(load(&graph, &json!({ "tensors": [] }), vec![0; 64]).is_err());
    assert!(Model::from_bytes(b"{\"ops\": [{\"op\": \"conv\"}]}", b"\x00\xff not json", vec![0; 8]).is_err());
}

#[test]
fn a_graph_that_names_what_the_manifest_lacks_is_refused() {
    let man = manifest(json!({ "w0": { "shape": [2], "offset": 0 } }));
    let missing = json!({ "ops": [{ "op": "conv", "in": ["a"], "out": ["b"], "w": ["w0", "nope"] }] });
    assert!(load(&missing, &man, vec![0; 4]).unwrap_err().contains("nope"));
    let film = json!({ "ops": [{ "op": "film_mlp", "in": ["t"], "out": ["f"] }] });
    assert!(load(&film, &man, vec![0; 4]).unwrap_err().contains("film.film.0.weight"));
    let nameless = json!({ "ops": [{ "in": ["a"], "out": ["b"] }] });
    assert!(load(&nameless, &man, vec![0; 4]).unwrap_err().contains("no name"));
    let bad_operand = json!({ "ops": [{ "op": "add", "in": ["a", 7], "out": ["b"] }] });
    assert!(load(&bad_operand, &man, vec![0; 4]).unwrap_err().contains("non-string"));
    let nested = json!({ "ops": [{ "op": "repeat", "times": 2, "body": [{ "op": "conv", "in": ["a"], "out": ["b"], "w": ["gone"] }] }] });
    assert!(load(&nested, &man, vec![0; 4]).unwrap_err().contains("gone"));
    let bodyless = json!({ "ops": [{ "op": "repeat", "times": 2 }] });
    assert!(load(&bodyless, &man, vec![0; 4]).unwrap_err().contains("body"));
}
