use grounding_coder::engine::CodeBot;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

static CTR: AtomicU64 = AtomicU64::new(60000);

fn tmp_cargo_project() -> std::path::PathBuf {
    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("gc-rng-{}-{}", std::process::id(), id));
    let src = base.join("src");
    fs::create_dir_all(&src).expect("src");
    fs::write(
        base.join("Cargo.toml"),
        "[package]\nname = \"gc-rng\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\n",
    )
    .expect("cargo");
    fs::write(src.join("lib.rs"), "pub fn base() {}\n").expect("lib");
    base
}

/// Full external-crate flow: crates.io verification → Cargo.toml dep →
/// verified imports → synthesized RNG struct with a real `next_u32` body
/// judged by a fixed contract value (seeded StdRng is deterministic).
#[tokio::test]
async fn rng_struct_from_registry_verified_crate() {
    let base = tmp_cargo_project();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let intent = serde_json::json!({
        "goal": "Random number generator struct",
        "file": "src/lib.rs",
        "language": "rust",
        "actions": [],
        "references": [],
        "define": [{
            "name": "RandomNumberGenerator",
            "kind": "struct",
            "references": ["rand::rngs::StdRng"],
            "fields": [{"name": "rng", "type": "StdRng"}],
            "methods": [
                {"name": "new", "self": "none", "params": ["rng: StdRng"], "op": "new"},
                {"name": "next_u32", "self": "mut", "params": [], "ret": "u32", "op": "call", "field": "rng"}
            ],
            "cases": [
                {"input": "{ let mut r = RandomNumberGenerator::new(<rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(42)); r.next_u32() }", "expected": "572990626u32"}
            ]
        }],
        "test": [],
        "imports": ["rand::rngs::StdRng", "rand::SeedableRng", "rand::Rng"],
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
    let manifest = fs::read_to_string(base.join("Cargo.toml")).unwrap();
    assert!(
        manifest.contains("rand = "),
        "missing registry dep:\n{}",
        manifest
    );
    let lib = fs::read_to_string(base.join("src/lib.rs")).unwrap();
    assert!(lib.contains("use rand::rngs::StdRng;"), "missing import");
    assert!(
        lib.contains("pub struct RandomNumberGenerator"),
        "missing struct"
    );
    assert!(lib.contains("self.rng.next_u32()"), "missing verified body");
    let _ = fs::remove_dir_all(&base);
}

/// Unknown crate root: no registry entry → import blocked, disk untouched.
#[tokio::test]
async fn bogus_crate_blocks_cleanly() {
    let base = tmp_cargo_project();
    let before_manifest = fs::read_to_string(base.join("Cargo.toml")).unwrap();
    let before_lib = fs::read_to_string(base.join("src/lib.rs")).unwrap();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let intent = serde_json::json!({
        "goal": "use bogus",
        "file": "src/lib.rs",
        "language": "rust",
        "actions": [],
        "references": [],
        "define": [],
        "test": [],
        "imports": ["definitely_not_a_real_crate_xyz::Thing"],
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
    assert_eq!(
        fs::read_to_string(base.join("Cargo.toml")).unwrap(),
        before_manifest
    );
    assert_eq!(
        fs::read_to_string(base.join("src/lib.rs")).unwrap(),
        before_lib
    );
    let _ = fs::remove_dir_all(&base);
}
