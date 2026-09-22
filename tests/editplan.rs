use grounding_coder::engine::{CodeWriter, EditPlan, Evidence, SourceEdit, SymbolTable};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static CTR: AtomicU64 = AtomicU64::new(1);

fn tmp_project(initial: &str) -> (PathBuf, PathBuf) {
    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("gc-test-{}-{}", std::process::id(), id));
    let src = base.join("src");
    fs::create_dir_all(&src).expect("src dir");
    let file = src.join("lib.rs");
    fs::write(&file, initial).expect("write");
    (base, file)
}

fn cleanup(base: &PathBuf) {
    let _ = fs::remove_dir_all(base);
}

#[test]
fn insert_import_applies_with_empty_expected_old() {
    let (base, file) = tmp_project("fn main() {}\n");
    let project_dir = file.parent().unwrap().parent().unwrap().to_path_buf();
    let writer = CodeWriter::new(project_dir);
    let content = fs::read_to_string(&file).unwrap();
    assert!(content.contains("fn main"));

    let plan = EditPlan {
        task_id: 1,
        edits: vec![SourceEdit {
            file: file.clone(),
            start: 0,
            end: 0,
            expected_old: String::new(),
            replacement: "use std::collections::HashMap;\n".to_string(),
        }],
        evidence: vec![Evidence::VerifiedSymbol {
            qname: "std::collections::HashMap".to_string(),
            source: "std".to_string(),
        }],
    };
    let snap = writer.snapshot(&plan).expect("snapshot");
    let changed = writer.apply_plan(&plan).expect("apply");
    assert_eq!(changed.len(), 1);
    let after = fs::read_to_string(&file).unwrap();
    assert!(after.contains("use std::collections::HashMap;"));
    assert!(after.contains("fn main"));
    writer.rollback(&snap).expect("rollback");
    let restored = fs::read_to_string(&file).unwrap();
    assert_eq!(restored, content);
    cleanup(&base);
}

#[test]
fn stale_edit_is_blocked() {
    let (base, file) = tmp_project("fn a() {}\n");
    let project_dir = file.parent().unwrap().parent().unwrap().to_path_buf();
    let writer = CodeWriter::new(project_dir);

    let plan = EditPlan {
        task_id: 2,
        edits: vec![SourceEdit {
            file: file.clone(),
            start: 0,
            end: 4,
            expected_old: "WRONG".to_string(),
            replacement: "fn b() {}".to_string(),
        }],
        evidence: vec![],
    };
    let err = writer.apply_plan(&plan).expect_err("should be stale");
    assert!(err.contains("STALE"), "got: {}", err);
    cleanup(&base);
}

#[test]
fn invalid_range_is_blocked() {
    let (base, file) = tmp_project("hi\n");
    let project_dir = file.parent().unwrap().parent().unwrap().to_path_buf();
    let writer = CodeWriter::new(project_dir);
    let plan = EditPlan {
        task_id: 3,
        edits: vec![SourceEdit {
            file: file.clone(),
            start: 100,
            end: 200,
            expected_old: String::new(),
            replacement: "x".to_string(),
        }],
        evidence: vec![],
    };
    let err = writer.apply_plan(&plan).expect_err("should be invalid");
    assert!(err.contains("Invalid byte range"), "got: {}", err);
    cleanup(&base);
}

#[test]
fn planner_rejects_llm_code() {
    let (base, _file) = tmp_project("fn main() {}\n");
    let writer = CodeWriter::new(base.clone());
    let st = SymbolTable::new();
    let task_json = serde_json::json!({
        "id": 99,
        "kind": "WriteFunction",
        "description": "evil",
        "target_symbols": ["src/lib.rs"],
        "required_symbols": [],
        "priority": 0.5,
        "source": "test",
        "payload": {"code": "fn evil() {}"},
        "deadline": 0
    });
    let task: grounding_coder::engine::tasks::SubTask =
        serde_json::from_value(task_json).expect("task");
    let err = writer.plan(&task, &st).expect_err("should reject code");
    assert!(err.contains("LLM code"), "got: {}", err);
    cleanup(&base);
}
