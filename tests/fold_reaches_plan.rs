//! Prove the fold reaches the plan: clarified code answers (language,
//! platform, every other aspect) must change the sub-tasks the planner
//! forms — not sit in the intent as inert prose. The fold function
//! under test is the same one `gc chat --clarify-answers` /
//! `gc encounter --answer` call (`apply_intent_answers`), so what is
//! proven here is what the loop actually does.
use grounding_coder::engine::arena::CodeArena;
use grounding_coder::engine::tasks::{TaskDecomposer, TaskKind, apply_intent_answers};
use grounding_coder::engine::understand;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static CTR: AtomicU64 = AtomicU64::new(8000);

fn tmp_project() -> PathBuf {
    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("gc-fold-{}-{}", std::process::id(), id));
    let src = base.join("src");
    fs::create_dir_all(&src).expect("src");
    fs::write(
        base.join("Cargo.toml"),
        "[package]\nname = \"gc-fold\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("cargo");
    fs::write(src.join("lib.rs"), "pub fn base() {}\n").expect("lib");
    base
}

fn build_task_payload_language(prose: &str, answers: &[(&str, &str)]) -> Option<String> {
    let base = tmp_project();
    let under = understand::understand(prose, Some(&base));
    let mut intent = under.intent;
    let folded: Vec<(String, String)> = answers
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    apply_intent_answers(&mut intent, &folded);
    let json = serde_json::to_string(&intent).expect("intent serializes");
    let mut d = TaskDecomposer::new();
    let arena = CodeArena::new();
    let tasks = d.decompose(&json, &arena);
    let _ = fs::remove_dir_all(&base);
    tasks
        .iter()
        .find(|t| t.kind == TaskKind::Build)
        .and_then(|t| {
            t.payload
                .get("language")
                .and_then(|v| v.as_str())
                .map(String::from)
        })
}

#[test]
fn a_folded_language_reshapes_the_build_plan() {
    // Without an answer, no language rides the build task: the backend
    // must infer it from the project's needs.
    assert_eq!(
        build_task_payload_language("build the app", &[]),
        None,
        "an unanswered build carries no language"
    );
    // Answer "rust": the fold reaches the plan and the build oracle
    // will pick the Rust backend from the payload.
    assert_eq!(
        build_task_payload_language("build the app", &[("language", "rust")]),
        Some("rust".to_string()),
        "a folded language must ride the build task"
    );
    // A different answer makes a different plan — the answer is not
    // decorative.
    assert_eq!(
        build_task_payload_language("build the app", &[("language", "python")]),
        Some("python".to_string()),
        "a different folded language must give a different build task"
    );
}

#[test]
fn unfolded_answers_still_plan_but_carry_the_fold() {
    // "no preference" for language is an attitude, not a selection: it
    // rides as a constraint and the build task stays language-free.
    assert_eq!(
        build_task_payload_language("build the app", &[("language", "no preference")]),
        None,
        "no preference must not pin a language"
    );
}

#[test]
fn folded_platform_and_constraints_survive_into_the_plan_input() {
    let base = tmp_project();
    let under = understand::understand("build an android apk", Some(&base));
    let mut intent = under.intent;
    apply_intent_answers(
        &mut intent,
        &[
            ("platform".to_string(), "android".to_string()),
            ("interface".to_string(), "command-line tool".to_string()),
        ],
    );
    let json = serde_json::to_string(&intent).expect("intent serializes");
    // Both the platform and the folded aspect ride the reconstructed
    // intent the planner reads; nothing is dropped in the round trip.
    assert!(json.contains("\"platform\":\"android\""), "{json}");
    assert!(json.contains("\"interface: command-line tool\""), "{json}");
    // The intent still decomposes into a build task — the fold never
    // blocks or corrupts planning.
    let mut d = TaskDecomposer::new();
    let arena = CodeArena::new();
    let tasks = d.decompose(&json, &arena);
    let _ = fs::remove_dir_all(&base);
    assert!(
        tasks.iter().any(|t| t.kind == TaskKind::Build),
        "an answered build intent must still plan a build"
    );
}
