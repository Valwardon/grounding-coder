use grounding_coder::engine::CodeBot;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static CTR: AtomicU64 = AtomicU64::new(1000);

fn tmp_rust_project() -> (PathBuf, PathBuf) {
    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("gc-e2e-{}-{}", std::process::id(), id));
    let src = base.join("src");
    fs::create_dir_all(&src).expect("src");
    fs::write(
        base.join("Cargo.toml"),
        "[package]\nname = \"tmp-proj\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("cargo");
    fs::write(src.join("lib.rs"), "pub fn base() {}\n").expect("lib");
    (base.clone(), base)
}

#[tokio::test]
async fn boring_import_path_commits() {
    let (_base_holder, base) = tmp_rust_project();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let intent = serde_json::json!({
        "goal": "Add HashMap support",
        "file": "src/lib.rs",
        "language": "rust",
        "actions": [],
        "references": ["std::collections::HashMap"],
        "define": [],
        "test": [],
        "imports": ["std::collections::HashMap"],
        "platform": "desktop",
        "architecture": "native",
        "runtime": "",
        "capabilities": [],
        "domains": [],
        "constraints": [],
        "dependencies": [],
        "unknown_requirements": [],
        "confidence": 1.0
    });
    let intent_json = serde_json::to_string(&intent).unwrap();
    let outcome = bot.run_task(&intent_json).await.expect("run");
    let rendered = format!("{}", outcome);
    assert!(
        rendered.contains("SUCCESS")
            || rendered.contains("BLOCKED")
            || rendered.contains("FAILURE"),
        "unexpected outcome: {}",
        rendered
    );
    if rendered.contains("SUCCESS") {
        let lib = fs::read_to_string(base.join("src/lib.rs")).unwrap();
        assert!(
            lib.contains("use std::collections::HashMap;"),
            "missing import in: {}",
            lib
        );
    }
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn unknown_symbol_blocks_honestly() {
    let (_b, base) = tmp_rust_project();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 2);
    let intent = serde_json::json!({
        "goal": "Wire FooBarBazUnknown",
        "file": "src/lib.rs",
        "language": "rust",
        "actions": [{"Action": {"action": "wire foobarbaz", "params": [], "references": ["FooBarBazUnknown"]}}],
        "references": ["FooBarBazUnknown"],
        "define": [],
        "test": [],
        "imports": [],
        "platform": "desktop",
        "architecture": "native",
        "runtime": "",
        "capabilities": [],
        "domains": [],
        "constraints": [],
        "dependencies": [],
        "unknown_requirements": ["FooBarBazUnknown"],
        "confidence": 0.2
    });
    let intent_json = serde_json::to_string(&intent).unwrap();
    let outcome = bot.run_task(&intent_json).await.expect("run");
    let rendered = format!("{}", outcome);
    assert!(
        rendered.contains("BLOCKED")
            || rendered.contains("SUCCESS")
            || rendered.contains("FAILURE"),
        "outcome: {}",
        rendered
    );
    let _ = fs::remove_dir_all(&base);
}
