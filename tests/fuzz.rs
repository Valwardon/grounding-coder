use grounding_coder::engine::{CodeWriter, EditPlan, SourceEdit, tasks::SubTask};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

static CTR: AtomicU64 = AtomicU64::new(40000);

fn tmp_file(name: &str, content: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("gc-fuzz-{}-{}", std::process::id(), id));
    fs::create_dir_all(&base).expect("tmp");
    let file = base.join(name);
    fs::write(&file, content).expect("write");
    (base, file)
}

/// Deeply nested hostile JSON must not overflow the stack: serde caps
/// recursion and the decomposer returns no tasks.
#[test]
fn deep_json_does_not_kill_decomposer() {
    let mut s = String::from("{\"a\":");
    for _ in 0..1000 {
        s.push('[');
    }
    s.push('1');
    for _ in 0..1000 {
        s.push(']');
    }
    s.push('}');
    let mut decomposer = grounding_coder::engine::tasks::TaskDecomposer::new();
    let arena = grounding_coder::engine::CodeArena::new();
    // Must return safely, never panic or abort. Deep input either fails
    // to parse (no tasks) or parses to defaults (verify-only task).
    let tasks = decomposer.decompose(&s, &arena);
    assert!(
        tasks
            .iter()
            .all(|t| t.kind == grounding_coder::engine::tasks::TaskKind::VerifyOnly)
    );
}

/// Unicode everywhere: content, import payload, evidence logging.
#[test]
fn unicode_content_and_imports_do_not_panic() {
    let (base, file) = tmp_file("lib.rs", "// héllo wörld — 日本語 🎉\nfn main() {}\n");
    let writer = CodeWriter::new(base.clone());
    let st = grounding_coder::engine::SymbolTable::new();
    let task_json = serde_json::json!({
        "id": 1, "kind": "AddImport", "description": "t",
        "target_symbols": ["lib.rs"], "required_symbols": [],
        "priority": 0.5, "source": "fuzz", "payload": {"import": "std::collections::HashMap"},
        "deadline": 0
    });
    let mut task: SubTask = serde_json::from_value(task_json).unwrap();
    task.target_symbols = vec![
        file.strip_prefix(&base)
            .unwrap()
            .to_string_lossy()
            .to_string(),
    ];
    // Must Ok or honest Err — never panic, even with multibyte content.
    let _ = writer.plan(&task, &st);
    let _ = fs::remove_dir_all(&base);
}

/// Adversarial byte ranges: all rejected, none panics.
#[test]
fn hostile_ranges_are_rejected_not_fatal() {
    let (base, file) = tmp_file("a.rs", "héllo\n");
    let writer = CodeWriter::new(base.clone());
    for (start, end, old) in [
        (3usize, 4usize, "x".to_string()),       // mid-char split
        (100usize, 200usize, String::new()),     // out of range
        (5usize, 2usize, String::new()),         // inverted
        (0usize, 7usize, "héllo\n".to_string()), // exact multibyte ok
    ] {
        let plan = EditPlan {
            task_id: 1,
            edits: vec![SourceEdit {
                file: file.clone(),
                start,
                end,
                expected_old: old,
                replacement: "y".to_string(),
            }],
            evidence: vec![],
        };
        let snap = writer.snapshot(&plan).expect("snapshot");
        let r = writer.apply_plan(&plan);
        // Exact-multibyte case applies; the rest must Err, never panic.
        if start == 0 {
            assert!(r.is_ok());
        } else {
            assert!(r.is_err());
        }
        let _ = writer.rollback(&snap);
    }
    let _ = fs::remove_dir_all(&base);
}

/// Garbage intent shapes decompose to nothing (or VerifyOnly), never panic.
#[test]
fn garbage_intents_decompose_safely() {
    let cases = [
        "null",
        "[]",
        "{}",
        "{\"actions\":[null, 42, \"x\"]}",
        "{\"define\":[null, null]}",
        "{\"goal\": \"ünïcodé 🎉 test — dash\"}",
    ];
    for c in cases {
        let mut decomposer = grounding_coder::engine::tasks::TaskDecomposer::new();
        let arena = grounding_coder::engine::CodeArena::new();
        let _ = decomposer.decompose(c, &arena);
    }
}
