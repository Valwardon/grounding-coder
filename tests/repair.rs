use grounding_coder::engine::CodeBot;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

static CTR: AtomicU64 = AtomicU64::new(30000);

fn tmp_rust_project_with(lib_rs: &str) -> std::path::PathBuf {
    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("gc-repair-{}-{}", std::process::id(), id));
    let src = base.join("src");
    fs::create_dir_all(&src).expect("src");
    fs::write(
        base.join("Cargo.toml"),
        "[package]\nname = \"gc-repair\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("cargo");
    fs::write(src.join("lib.rs"), lib_rs).expect("lib");
    base
}

fn fix_intent() -> String {
    serde_json::to_string(&serde_json::json!({
        "goal": "fix this repo",
        "language": "rust",
        "actions": [],
        "references": [],
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
        "unknown_requirements": [],
        "confidence": 1.0
    }))
    .unwrap()
}

#[tokio::test]
async fn repairs_missing_import_then_compiles() {
    let base = tmp_rust_project_with(
        "pub fn new_map() -> HashMap<String, u32> {\n    HashMap::new()\n}\n",
    );
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let outcome = bot.run_task(&fix_intent()).await.expect("run");
    let rendered = format!("{}", outcome);
    assert!(
        rendered.contains("SUCCESS"),
        "expected repair SUCCESS, got: {}",
        rendered
    );
    let lib = fs::read_to_string(base.join("src/lib.rs")).unwrap();
    assert!(
        lib.contains("use std::collections::HashMap;"),
        "missing compiler-suggested import:\n{}",
        lib
    );
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn unfixable_error_blocks_without_touching_disk() {
    let broken = "pub fn broken() -> i32 {\n    undefined_function_xyz()\n}\n";
    let base = tmp_rust_project_with(broken);
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let outcome = bot.run_task(&fix_intent()).await.expect("run");
    let rendered = format!("{}", outcome);
    assert!(
        rendered.contains("BLOCKED"),
        "expected BLOCKED, got: {}",
        rendered
    );
    assert_eq!(
        fs::read_to_string(base.join("src/lib.rs")).unwrap(),
        broken,
        "disk must be untouched"
    );
    let _ = fs::remove_dir_all(&base);
}
