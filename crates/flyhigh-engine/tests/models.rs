// The model catalogue (models/models.json) and the shipped manifest (models/flyhighfg.json) agree; no GPU needed.
use serde_json::Value;
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
