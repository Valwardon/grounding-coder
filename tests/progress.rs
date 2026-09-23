use grounding_coder::engine::{CodeBot, ProgressCallback, ProgressEvent};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static CTR: AtomicU64 = AtomicU64::new(50000);

fn tmp_rust_project() -> std::path::PathBuf {
    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("gc-prog-{}-{}", std::process::id(), id));
    let src = base.join("src");
    fs::create_dir_all(&src).expect("src");
    fs::write(
        base.join("Cargo.toml"),
        "[package]\nname = \"gc-prog\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("cargo");
    fs::write(src.join("lib.rs"), "pub fn base() {}\n").expect("lib");
    base
}

/// The feedback loop: every stage narrates in order, so a stuck run names
/// where it stopped instead of spinning silently.
#[tokio::test]
async fn progress_trail_narrates_in_order() {
    let base = tmp_rust_project();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let trail: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = trail.clone();
    let cb: ProgressCallback = Arc::new(move |ev: ProgressEvent| {
        sink.lock().unwrap().push(ev.to_string());
    });
    bot.set_progress_listener(cb);
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
    let outcome = bot
        .run_task(&serde_json::to_string(&intent).unwrap())
        .await
        .expect("run");
    assert!(format!("{}", outcome).contains("SUCCESS"));
    let trail = trail.lock().unwrap().join("\n");
    for marker in ["task ", "planned", "applied", "verify", "committed"] {
        assert!(
            trail.contains(marker),
            "trail missing '{}':\n{}",
            marker,
            trail
        );
    }
    // Order: planning before applying before verifying.
    let (mut pi, mut ai, mut vi) = (0, 0, 0);
    for (i, line) in trail.lines().enumerate() {
        if line.contains("planned") {
            pi = i;
        }
        if line.contains("applied") {
            ai = i;
        }
        if line.contains("verify") {
            vi = i;
        }
    }
    assert!(pi < ai && ai < vi, "stages out of order:\n{}", trail);
    let _ = fs::remove_dir_all(&base);
}

/// Silence by default: no listener, no trail, same outcome.
#[tokio::test]
async fn silent_without_listener() {
    let base = tmp_rust_project();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let intent = serde_json::json!({
        "goal": "noop verify",
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
    });
    let outcome = bot
        .run_task(&serde_json::to_string(&intent).unwrap())
        .await
        .expect("run");
    assert!(format!("{}", outcome).contains("SUCCESS"));
    let _ = fs::remove_dir_all(&base);
}
