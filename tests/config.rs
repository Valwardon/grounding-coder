use grounding_coder::config::{AppConfig, load_config, save_config};

#[test]
fn config_path_always_resolves() {
    // Must never return Err — not even with an empty environment, which
    // is what an Android app process looks like to the `dirs` crate.
    assert!(grounding_coder::config::config_path().is_ok());
}

#[test]
fn save_load_roundtrip_with_missing_parent_dirs() {
    let dir = std::env::temp_dir().join(format!(
        "gc-cfg-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let path = dir.join("deep").join("nested").join("config.json");
    let cfg = AppConfig {
        project_path: "/tmp/proj".to_string(),
        max_retries: 7,
        ..AppConfig::default()
    };
    save_config(&cfg, &path.to_string_lossy()).expect("save must create parents");
    let loaded = load_config(&path.to_string_lossy());
    assert_eq!(loaded.project_path, "/tmp/proj");
    assert_eq!(loaded.max_retries, 7);
    // Old config files without the new fields still load.
    std::fs::write(&path, "{\"project_path\":\"/tmp/o\"}").unwrap();
    let old = load_config(&path.to_string_lossy());
    assert_eq!(old.project_path, "/tmp/o");
    assert_eq!(old.max_retries, 5);
    let _ = std::fs::remove_dir_all(&dir);
}
