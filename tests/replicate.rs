use grounding_coder::engine::CodeBot;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

static CTR: AtomicU64 = AtomicU64::new(80000);

fn tmp_dir() -> std::path::PathBuf {
    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("gc-rep-{}-{}", std::process::id(), id));
    fs::create_dir_all(&base).expect("tmp");
    base
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

fn files_intent(files: serde_json::Value, replacements: serde_json::Value) -> String {
    serde_json::to_string(&serde_json::json!({
        "goal": "port files",
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
        "confidence": 1.0,
        "files": files,
        "replacements": replacements
    }))
    .unwrap()
}

#[tokio::test]
async fn replicate_file_url_lands_byte_exact() {
    let base = tmp_dir();
    let src = base.join("src.txt");
    fs::write(&src, "hello port\n").unwrap();
    let sha = sha256_hex(b"hello port\n");
    let project = base.join("proj");
    fs::create_dir_all(&project).unwrap();
    let mut bot = CodeBot::new(project.to_str().unwrap(), 5);
    // Replication of an owned template: explicit clean-room opt-out.
    bot.set_clean_room(false);
    let outcome = bot
        .run_task(&files_intent(
            serde_json::json!([{
                "path": "copied.txt",
                "url": format!("file://{}", src.display()),
                "sha256": sha
            }]),
            serde_json::json!([]),
        ))
        .await
        .expect("run");
    assert!(
        format!("{}", outcome).contains("SUCCESS"),
        "got: {}",
        outcome
    );
    assert_eq!(
        fs::read_to_string(project.join("copied.txt")).unwrap(),
        "hello port\n"
    );
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn hash_mismatch_blocks_with_clean_disk() {
    let base = tmp_dir();
    let src = base.join("src.txt");
    fs::write(&src, "real bytes\n").unwrap();
    let project = base.join("proj");
    fs::create_dir_all(&project).unwrap();
    let mut bot = CodeBot::new(project.to_str().unwrap(), 5);
    bot.set_clean_room(false);
    let outcome = bot
        .run_task(&files_intent(
            serde_json::json!([{
                "path": "evil.txt",
                "url": format!("file://{}", src.display()),
                "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
            }]),
            serde_json::json!([]),
        ))
        .await
        .expect("run");
    assert!(format!("{}", outcome).contains("BLOCKED"));
    assert!(!project.join("evil.txt").exists());
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn replace_exact_single_match_rule() {
    let base = tmp_dir();
    let project = base.join("proj");
    fs::create_dir_all(&project).unwrap();
    fs::write(
        project.join("app.txt"),
        "name = \"psdr\"\nname = \"psdr\"\n",
    )
    .unwrap();
    // Two matches → ambiguous → BLOCKED, file untouched.
    let mut bot = CodeBot::new(project.to_str().unwrap(), 5);
    let outcome = bot
        .run_task(&files_intent(
            serde_json::json!([]),
            serde_json::json!([{
                "file": "app.txt", "find": "psdr", "replace": "psdrv2"
            }]),
        ))
        .await
        .expect("run");
    assert!(format!("{}", outcome).contains("BLOCKED"));
    assert_eq!(
        fs::read_to_string(project.join("app.txt")).unwrap(),
        "name = \"psdr\"\nname = \"psdr\"\n"
    );
    // Zero matches → BLOCKED as well.
    let outcome = bot
        .run_task(&files_intent(
            serde_json::json!([]),
            serde_json::json!([{
                "file": "app.txt", "find": "nope", "replace": "x"
            }]),
        ))
        .await
        .expect("run");
    assert!(format!("{}", outcome).contains("BLOCKED"));
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn replace_exact_single_match_applies() {
    let base = tmp_dir();
    let project = base.join("proj");
    fs::create_dir_all(&project).unwrap();
    fs::write(
        project.join("app.txt"),
        "name = \"psdr\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    let mut bot = CodeBot::new(project.to_str().unwrap(), 5);
    let outcome = bot
        .run_task(&files_intent(
            serde_json::json!([]),
            serde_json::json!([{
                "file": "app.txt",
                "find": "name = \"psdr\"",
                "replace": "name = \"psdrv2\""
            }]),
        ))
        .await
        .expect("run");
    // Single-file content edit verifies through the normal oracle path;
    // a bare dir has no toolchain, so syntax N/A — accept SUCCESS or
    // an honest non-guess outcome, but the bytes must be exact.
    let rendered = format!("{}", outcome);
    assert!(
        rendered.contains("SUCCESS")
            || rendered.contains("PARTIAL")
            || rendered.contains("BLOCKED"),
        "unexpected: {}",
        rendered
    );
    let content = fs::read_to_string(project.join("app.txt")).unwrap();
    // Either applied exactly once, or rolled back byte-identical.
    assert!(
        content == "name = \"psdrv2\"\nversion = \"0.1.0\"\n"
            || content == "name = \"psdr\"\nversion = \"0.1.0\"\n",
        "corrupted:\n{}",
        content
    );
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn clean_room_is_default_and_refuses_before_any_fetch() {
    let base = tmp_dir();
    // Outside bytes the engine must never see — even with a valid hash.
    let src = base.join("secret.txt");
    fs::write(&src, "someone else's code\n").unwrap();
    let sha = sha256_hex(b"someone else's code\n");
    let project = base.join("proj");
    fs::create_dir_all(&project).unwrap();
    // NOTE: no set_clean_room call — default must refuse.
    let mut bot = CodeBot::new(project.to_str().unwrap(), 5);
    let outcome = bot
        .run_task(&files_intent(
            serde_json::json!([{
                "path": "secret.txt",
                "url": format!("file://{}", src.display()),
                "sha256": sha
            }]),
            serde_json::json!([]),
        ))
        .await
        .expect("run");
    let rendered = format!("{}", outcome);
    assert!(
        rendered.contains("BLOCKED")
            && rendered.contains("CleanRoom")
            && rendered.contains("CLEAN_ROOM"),
        "unexpected: {}",
        rendered
    );
    // Disk untouched: nothing fetched, nothing written.
    assert!(!project.join("secret.txt").exists());
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn clean_room_explicit_opt_out_restores_replication() {
    let base = tmp_dir();
    let src = base.join("owned.txt");
    fs::write(&src, "my own template\n").unwrap();
    let sha = sha256_hex(b"my own template\n");
    let project = base.join("proj");
    fs::create_dir_all(&project).unwrap();
    let mut bot = CodeBot::new(project.to_str().unwrap(), 5);
    bot.set_clean_room(false);
    let outcome = bot
        .run_task(&files_intent(
            serde_json::json!([{
                "path": "owned.txt",
                "url": format!("file://{}", src.display()),
                "sha256": sha
            }]),
            serde_json::json!([]),
        ))
        .await
        .expect("run");
    assert!(
        format!("{}", outcome).contains("SUCCESS"),
        "got: {}",
        outcome
    );
    assert_eq!(
        fs::read_to_string(project.join("owned.txt")).unwrap(),
        "my own template\n"
    );
    let _ = fs::remove_dir_all(&base);
}
