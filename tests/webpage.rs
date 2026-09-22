use grounding_coder::engine::CodeBot;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

static CTR: AtomicU64 = AtomicU64::new(20000);

fn tmp_dir() -> std::path::PathBuf {
    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("gc-web-{}-{}", std::process::id(), id));
    fs::create_dir_all(&base).expect("tmp");
    base
}

fn page_intent(file: &str, title: &str, sections: serde_json::Value, footer: &str) -> String {
    serde_json::to_string(&serde_json::json!({
        "goal": "dentist homepage",
        "file": file,
        "language": "html",
        "actions": [],
        "references": [],
        "define": [{
            "name": "homepage",
            "kind": "page",
            "references": [],
            "title": title,
            "sections": sections,
            "footer": footer
        }],
        "test": [],
        "imports": [],
        "platform": "web",
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
async fn dentist_homepage_builds_and_verifies() {
    let base = tmp_dir();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let outcome = bot
        .run_task(&page_intent(
            "index.html",
            "Bright Smile Dental — Miami, FL",
            serde_json::json!([
                {"heading": "Our Services", "body": "Cleanings, whitening, braces, and implants for the whole family."},
                {"heading": "Visit Us", "body": "Open Monday to Saturday in Little Havana. Call (305) 555-0142."}
            ]),
            "Bright Smile Dental, 1234 SW 8th St, Miami, FL 33135",
        ))
        .await
        .expect("run");
    let rendered = format!("{}", outcome);
    assert!(rendered.contains("SUCCESS"), "got: {}", rendered);
    let html = fs::read_to_string(base.join("index.html")).unwrap();
    // Structure from the engine template.
    assert!(html.contains("<!DOCTYPE html>"), "no doctype");
    assert!(html.contains("<title>Bright Smile Dental"), "no title");
    // Every content slot present verbatim.
    assert!(html.contains("Our Services"), "missing section");
    assert!(html.contains("(305) 555-0142"), "missing phone");
    assert!(
        html.contains("1234 SW 8th St, Miami, FL 33135"),
        "missing address"
    );
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn hostile_content_is_escaped_not_executed() {
    let base = tmp_dir();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let outcome = bot
        .run_task(&page_intent(
            "index.html",
            "Test Page",
            serde_json::json!([
                {"heading": "Hi", "body": "<script>alert(1)</script>"}
            ]),
            "foot",
        ))
        .await
        .expect("run");
    assert!(format!("{}", outcome).contains("SUCCESS"));
    let html = fs::read_to_string(base.join("index.html")).unwrap();
    assert!(
        !html.contains("<script>"),
        "live script tag leaked:\n{}",
        html
    );
    assert!(html.contains("&lt;script&gt;"), "not escaped:\n{}", html);
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn empty_title_blocks_without_file() {
    let base = tmp_dir();
    let project = base.to_string_lossy().to_string();
    let mut bot = CodeBot::new(&project, 5);
    let outcome = bot
        .run_task(&page_intent(
            "index.html",
            "",
            serde_json::json!([{"heading": "H", "body": "B"}]),
            "F",
        ))
        .await
        .expect("run");
    assert!(format!("{}", outcome).contains("BLOCKED"));
    assert!(!base.join("index.html").exists());
    let _ = fs::remove_dir_all(&base);
}
