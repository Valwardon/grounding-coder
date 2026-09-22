use grounding_coder::engine::CodeBot;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

static CTR: AtomicU64 = AtomicU64::new(9000);

fn tmp_dir() -> std::path::PathBuf {
    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("gc-poly-{}-{}", std::process::id(), id));
    fs::create_dir_all(&base).expect("tmp");
    base
}

fn intent(file: &str, imports: &[&str], language: &str) -> String {
    serde_json::to_string(&serde_json::json!({
        "goal": "add imports",
        "file": file,
        "language": language,
        "actions": [],
        "references": imports,
        "define": [],
        "test": [],
        "imports": imports,
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
async fn python_import_verified_by_py_compile() {
    let base = tmp_dir();
    fs::write(base.join("main.py"), "def greet():\n    return \"hi\"\n").unwrap();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let outcome = bot
        .run_task(&intent("main.py", &["os"], "python"))
        .await
        .expect("run");
    let rendered = format!("{}", outcome);
    assert!(rendered.contains("SUCCESS"), "got: {}", rendered);
    let content = fs::read_to_string(base.join("main.py")).unwrap();
    assert!(
        content.contains("import os"),
        "missing import:\n{}",
        content
    );
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn python_unknown_import_blocked() {
    let base = tmp_dir();
    fs::write(base.join("main.py"), "x = 1\n").unwrap();
    let before = fs::read_to_string(base.join("main.py")).unwrap();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let outcome = bot
        .run_task(&intent("main.py", &["nonexistent_pkg_xyz"], "python"))
        .await
        .expect("run");
    let rendered = format!("{}", outcome);
    assert!(rendered.contains("BLOCKED"), "got: {}", rendered);
    assert_eq!(fs::read_to_string(base.join("main.py")).unwrap(), before);
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn registry_node_import_verified_by_node_check() {
    // JavaScript arrived via PURE DATA (registry spec) — no engine code.
    let base = tmp_dir();
    fs::write(base.join("package.json"), "{\"name\":\"t\"}\n").unwrap();
    fs::write(base.join("index.js"), "console.log('hi');\n").unwrap();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let outcome = bot
        .run_task(&intent("index.js", &["fs"], "javascript"))
        .await
        .expect("run");
    let rendered = format!("{}", outcome);
    assert!(rendered.contains("SUCCESS"), "got: {}", rendered);
    let content = fs::read_to_string(base.join("index.js")).unwrap();
    assert!(
        content.contains("require('fs')"),
        "missing require:\n{}",
        content
    );
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn toml_defined_language_works_with_zero_engine_changes() {
    // A language the engine has never heard of, taught by one TOML file.
    let base = tmp_dir();
    fs::write(
        base.join(".grounding.toml"),
        "[language]\n\
         name = \"zz\"\n\
         extensions = [\"zz\"]\n\
         import_template = \"need {p};\"\n\
         import_prefixes = [\"need \"]\n\
         std_exact = [\"std/io\"]\n\
         verify_cmd = [\"true\"]\n",
    )
    .unwrap();
    fs::write(base.join("prog.zz"), "say hello\n").unwrap();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let outcome = bot
        .run_task(&intent("prog.zz", &["std/io"], "zz"))
        .await
        .expect("run");
    let rendered = format!("{}", outcome);
    assert!(rendered.contains("SUCCESS"), "got: {}", rendered);
    let content = fs::read_to_string(base.join("prog.zz")).unwrap();
    assert!(
        content.contains("need std/io;"),
        "missing need:\n{}",
        content
    );
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn missing_toolchain_blocks_honestly() {
    // Go has a registry spec but no toolchain here: import plans fine,
    // verification refuses loudly, disk untouched.
    let base = tmp_dir();
    fs::write(base.join("go.mod"), "module example.com/t\n\ngo 1.21\n").unwrap();
    fs::write(base.join("main.go"), "package main\n\nfunc main() {}\n").unwrap();
    let before = fs::read_to_string(base.join("main.go")).unwrap();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let outcome = bot
        .run_task(&intent("main.go", &["fmt"], "go"))
        .await
        .expect("run");
    let rendered = format!("{}", outcome);
    assert!(rendered.contains("BLOCKED"), "got: {}", rendered);
    assert!(rendered.contains("GO_MISSING"), "got: {}", rendered);
    assert_eq!(fs::read_to_string(base.join("main.go")).unwrap(), before);
    let _ = fs::remove_dir_all(&base);
}
