use grounding_coder::engine::CodeBot;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

static CTR: AtomicU64 = AtomicU64::new(70000);

fn tmp_dir() -> std::path::PathBuf {
    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("gc-kt-{}-{}", std::process::id(), id));
    fs::create_dir_all(&base).expect("tmp");
    base
}

/// Full Kotlin RNG flow through the engine: import planning with Kotlin
/// conventions, synthesized class + contract main, kotlinc compile, JVM
/// run of the checks. Deterministic seeds keep the contract fixed.
#[tokio::test]
async fn kotlin_rng_synthesized_compiled_and_run() {
    let base = tmp_dir();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let intent = serde_json::json!({
        "goal": "Kotlin random number generator",
        "file": "Rng.kt",
        "language": "kotlin",
        "actions": [],
        "references": [],
        "define": [{
            "name": "RandomNumberGenerator",
            "kind": "struct",
            "references": ["kotlin.random.Random"],
            "fields": [{"name": "rng", "type": "Random"}],
            "methods": [
                {"name": "constructor", "self": "none", "params": ["seed: Long"], "op": "knew"},
                {"name": "nextInt", "self": "mut", "params": [], "ret": "Int", "op": "kcall", "field": "rng"}
            ],
            "cases": [
                {"input": "{ val r = RandomNumberGenerator(42); r.nextInt() }", "expected": "972016666"},
                {"input": "{ val r = RandomNumberGenerator(42); r.nextInt(); r.nextInt() }", "expected": "1740578880"}
            ]
        }],
        "test": [],
        "imports": ["kotlin.random.Random"],
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
    let src = fs::read_to_string(base.join("Rng.kt")).unwrap();
    assert!(src.contains("class RandomNumberGenerator"), "missing class");
    assert!(src.contains("rng.nextInt()"), "missing verified body");
    assert!(src.contains("fun main()"), "missing contract");
    let _ = fs::remove_dir_all(&base);
}

/// Unknown Kotlin call shapes block with no file left behind.
#[tokio::test]
async fn kotlin_unknown_call_blocks_cleanly() {
    let base = tmp_dir();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let intent = serde_json::json!({
        "goal": "magic",
        "file": "Magic.kt",
        "language": "kotlin",
        "actions": [],
        "references": [],
        "define": [{
            "name": "Magic",
            "kind": "struct",
            "references": [],
            "fields": [{"name": "rng", "type": "Random"}],
            "methods": [
                {"name": "teleport", "self": "mut", "params": [], "ret": "Int", "op": "kcall", "field": "rng"}
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
    assert!(format!("{}", outcome).contains("BLOCKED"));
    assert!(!base.join("Magic.kt").exists());
    let _ = fs::remove_dir_all(&base);
}

/// Provisioner really downloads + verifies (isolated tools dir).
/// Cached afterwards: reruns are offline.
#[tokio::test]
async fn provisioner_downloads_pinned_toolchain() {
    let dir = std::env::temp_dir().join(format!(
        "gc-tools-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    unsafe {
        std::env::set_var("GROUNDING_TOOLS_DIR", &dir);
    }
    let tool_dir = grounding_coder::tool::ensure_tool(&grounding_coder::tool::KOTLIN_TOOL)
        .await
        .expect("provision kotlin");
    for f in grounding_coder::tool::KOTLIN_FILES {
        let p = tool_dir.join(f.name);
        assert!(p.exists(), "missing {}", f.name);
        // Non-empty and plausible (annotations jar is legitimately ~18KB).
        assert!(p.metadata().unwrap().len() > 1_000, "suspicious {}", f.name);
    }
    unsafe {
        std::env::remove_var("GROUNDING_TOOLS_DIR");
    }
    let _ = fs::remove_dir_all(&dir);
}
