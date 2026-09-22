use grounding_coder::engine::CodeBot;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static CTR: AtomicU64 = AtomicU64::new(5000);

fn tmp_rust_project() -> PathBuf {
    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("gc-synth-{}-{}", std::process::id(), id));
    let src = base.join("src");
    fs::create_dir_all(&src).expect("src");
    fs::write(
        base.join("Cargo.toml"),
        "[package]\nname = \"gc-synth\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("cargo");
    fs::write(src.join("lib.rs"), "pub fn base() {}\n").expect("lib");
    base
}

#[tokio::test]
async fn synthesizes_word_counts_from_contract() {
    let base = tmp_rust_project();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let intent = serde_json::json!({
        "goal": "Count word occurrences",
        "file": "src/lib.rs",
        "language": "rust",
        "actions": [],
        "references": [],
        "define": [{
            "name": "word_counts",
            "kind": "function",
            "references": [],
            "signature": "fn word_counts(text: &str) -> HashMap<String, usize>",
            "cases": [
                {"input": "\"a a b\"", "expected": "HashMap::from([(\"a\".to_string(), 2usize), (\"b\".to_string(), 1usize)])"},
                {"input": "\"hello\"", "expected": "HashMap::from([(\"hello\".to_string(), 1usize)])"}
            ]
        }],
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
    let outcome = bot
        .run_task(&serde_json::to_string(&intent).unwrap())
        .await
        .expect("run");
    let rendered = format!("{}", outcome);
    assert!(
        rendered.contains("SUCCESS"),
        "expected synthesis SUCCESS, got: {}",
        rendered
    );
    let lib = fs::read_to_string(base.join("src/lib.rs")).unwrap();
    // Real logic: built only from verified std ingredients.
    assert!(lib.contains("fn word_counts"), "missing fn:\n{}", lib);
    assert!(
        lib.contains("split_whitespace"),
        "missing ingredient:\n{}",
        lib
    );
    assert!(lib.contains(".entry("), "missing ingredient:\n{}", lib);
    assert!(
        lib.contains("grounded_contract_word_counts"),
        "missing contract test:\n{}",
        lib
    );
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn unsupported_shape_blocks_honestly() {
    let base = tmp_rust_project();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let intent = serde_json::json!({
        "goal": "Do async I/O",
        "file": "src/lib.rs",
        "language": "rust",
        "actions": [],
        "references": [],
        "define": [{
            "name": "fetch_everything",
            "kind": "function",
            "references": [],
            "signature": "fn fetch_everything(url: Url) -> Result<Response, Error>",
            "cases": [{"input": "x", "expected": "y"}]
        }],
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
    });
    let outcome = bot
        .run_task(&serde_json::to_string(&intent).unwrap())
        .await
        .expect("run");
    let rendered = format!("{}", outcome);
    assert!(
        rendered.contains("BLOCKED"),
        "expected BLOCKED for unsupported shape, got: {}",
        rendered
    );
    let lib = fs::read_to_string(base.join("src/lib.rs")).unwrap();
    assert!(
        !lib.contains("fn fetch_everything"),
        "unverified fn must not remain:\n{}",
        lib
    );
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn synthesizes_counter_struct_from_contract() {
    let base = tmp_rust_project();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let intent = serde_json::json!({
        "goal": "A counter with increment behavior",
        "file": "src/lib.rs",
        "language": "rust",
        "actions": [],
        "references": [],
        "define": [{
            "name": "Counter",
            "kind": "struct",
            "references": [],
            "fields": [{"name": "count", "type": "i32"}],
            "methods": [
                {"name": "new", "self": "none", "params": ["count: i32"], "op": "new"},
                {"name": "incr", "self": "mut", "params": [], "op": "add_assign", "field": "count"},
                {"name": "incr_by", "self": "mut", "params": ["n: i32"], "op": "add_assign", "field": "count", "amount": "n"},
                {"name": "value", "self": "ref", "params": [], "ret": "i32", "op": "get", "field": "count"}
            ],
            "cases": [
                {"input": "{ let mut c = Counter::new(0); c.incr(); c.value() }", "expected": "1"},
                {"input": "{ let mut c = Counter::new(10); c.incr_by(5); c.incr(); c.value() }", "expected": "16"}
            ]
        }],
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
    });
    let outcome = bot
        .run_task(&serde_json::to_string(&intent).unwrap())
        .await
        .expect("run");
    let rendered = format!("{}", outcome);
    assert!(
        rendered.contains("SUCCESS"),
        "expected synthesis SUCCESS, got: {}",
        rendered
    );
    let lib = fs::read_to_string(base.join("src/lib.rs")).unwrap();
    assert!(
        lib.contains("pub struct Counter"),
        "missing struct:\n{}",
        lib
    );
    assert!(lib.contains("fn incr_by"), "missing method:\n{}", lib);
    assert!(
        lib.contains("grounded_contract_Counter"),
        "missing contract test:\n{}",
        lib
    );
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn synthesizes_fallible_parse_from_contract() {
    let base = tmp_rust_project();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let intent = serde_json::json!({
        "goal": "Parse a port number",
        "file": "src/lib.rs",
        "language": "rust",
        "actions": [],
        "references": [],
        "define": [{
            "name": "parse_port",
            "kind": "function",
            "references": [],
            "signature": "fn parse_port(s: &str) -> Result<u16, String>",
            "cases": [
                {"input": "\"8080\"", "expected": "Ok(8080u16)"},
                {"input": "\"abc\"", "expected": "Err(\"invalid digit found in string\".to_string())"}
            ]
        }],
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
    });
    let outcome = bot
        .run_task(&serde_json::to_string(&intent).unwrap())
        .await
        .expect("run");
    let rendered = format!("{}", outcome);
    assert!(
        rendered.contains("SUCCESS"),
        "expected synthesis SUCCESS, got: {}",
        rendered
    );
    let lib = fs::read_to_string(base.join("src/lib.rs")).unwrap();
    assert!(lib.contains("fn parse_port"), "missing fn:\n{}", lib);
    assert!(lib.contains("parse::<u16>"), "missing ingredient:\n{}", lib);
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn unknown_method_op_blocks_without_trace() {
    let base = tmp_rust_project();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let intent = serde_json::json!({
        "goal": "Counter with magic",
        "file": "src/lib.rs",
        "language": "rust",
        "actions": [],
        "references": [],
        "define": [{
            "name": "Magic",
            "kind": "struct",
            "references": [],
            "fields": [{"name": "x", "type": "i32"}],
            "methods": [
                {"name": "conjure", "self": "mut", "params": [], "op": "teleport", "field": "x"}
            ],
            "cases": [{"input": "0", "expected": "0"}]
        }],
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
    });
    let outcome = bot
        .run_task(&serde_json::to_string(&intent).unwrap())
        .await
        .expect("run");
    let rendered = format!("{}", outcome);
    assert!(
        rendered.contains("BLOCKED"),
        "expected BLOCKED for unknown op, got: {}",
        rendered
    );
    let lib = fs::read_to_string(base.join("src/lib.rs")).unwrap();
    assert!(
        !lib.contains("struct Magic"),
        "unverified struct must not remain:\n{}",
        lib
    );
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn synthesizes_new_module_with_wiring() {
    let base = tmp_rust_project();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let intent = serde_json::json!({
        "goal": "Geometry helpers in their own module",
        "file": "src/geometry.rs",
        "language": "rust",
        "actions": [],
        "references": [],
        "define": [{
            "name": "word_counts",
            "kind": "function",
            "references": [],
            "signature": "fn word_counts(text: &str) -> HashMap<String, usize>",
            "cases": [
                {"input": "\"a a b\"", "expected": "HashMap::from([(\"a\".to_string(), 2usize), (\"b\".to_string(), 1usize)])"}
            ]
        }],
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
    let outcome = bot
        .run_task(&serde_json::to_string(&intent).unwrap())
        .await
        .expect("run");
    let rendered = format!("{}", outcome);
    assert!(
        rendered.contains("SUCCESS"),
        "expected SUCCESS, got: {}",
        rendered
    );
    // New module created with real logic + contract.
    let geo = fs::read_to_string(base.join("src/geometry.rs")).unwrap();
    assert!(geo.contains("fn word_counts"), "missing fn:\n{}", geo);
    assert!(geo.contains("split_whitespace"), "missing logic:\n{}", geo);
    // Crate root wired in the same transaction.
    let lib = fs::read_to_string(base.join("src/lib.rs")).unwrap();
    assert!(
        lib.lines().any(|l| l.trim() == "mod geometry;"),
        "missing mod wiring:\n{}",
        lib
    );
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn failed_second_task_rolls_back_first() {
    let base = tmp_rust_project();
    let lib_before = fs::read_to_string(base.join("src/lib.rs")).unwrap();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    // Task 1 (valid import) applies and verifies clean. Task 2 synthesizes
    // a body but its contract is WRONG, so verification fails after apply.
    // Atomicity demands task 1's import is rolled back too.
    let intent = serde_json::json!({
        "goal": "Import plus a miscounted contract",
        "file": "src/lib.rs",
        "language": "rust",
        "actions": [],
        "references": [],
        "define": [{
            "name": "word_counts",
            "kind": "function",
            "references": [],
            "signature": "fn word_counts(text: &str) -> HashMap<String, usize>",
            "cases": [
                {"input": "\"a a b\"", "expected": "HashMap::new()"}
            ]
        }],
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
    let outcome = bot
        .run_task(&serde_json::to_string(&intent).unwrap())
        .await
        .expect("run");
    let rendered = format!("{}", outcome);
    assert!(
        rendered.contains("BLOCKED") || rendered.contains("FAILURE"),
        "expected BLOCKED/FAILURE, got: {}",
        rendered
    );
    let lib_after = fs::read_to_string(base.join("src/lib.rs")).unwrap();
    assert_eq!(
        lib_before, lib_after,
        "lib.rs must be byte-identical after rollback — import included"
    );
    let _ = fs::remove_dir_all(&base);
}
