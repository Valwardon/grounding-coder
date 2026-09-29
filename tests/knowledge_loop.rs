//! The idle learning loop, end to end: a finished task leaves gaps,
//! gaps become questions, the loop investigates on spare cycles, and
//! only verified results become knowledge. No network anywhere here —
//! every dream that cannot verify stays a dream.
use grounding_coder::engine::knowledge::{self, CompletedTask, KnowledgeState, KnowledgeStore};

fn scratch_project(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "gc-dream-it-{}-{}",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

fn demo_task() -> CompletedTask {
    CompletedTask {
        goal: "build a retrying fetcher".to_string(),
        concepts_used: vec!["retry-budget".to_string(), "http-client".to_string()],
        unknowns_seen: vec!["backoff".to_string()],
        failed_attempts: 2,
        related: vec!["jitter".to_string()],
    }
}

#[test]
fn idle_loop_respects_budget_and_persists() {
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let project = scratch_project("budget");
    let mut store = KnowledgeStore::open(&project);
    let gaps = knowledge::extract_gaps(&demo_task(), &mut store);
    assert_eq!(gaps.len(), 4, "{:?}", gaps);
    drop(store);

    // Offline pass: hypotheses form, nothing verifies, budget binds.
    let outcomes = rt.block_on(knowledge::dream_loop(&project, 2, false));
    assert_eq!(outcomes.len(), 2, "{:?}", outcomes);
    assert!(
        outcomes.iter().all(|o| o.to == KnowledgeState::Hypothesis),
        "{:?}",
        outcomes
    );

    // The store file is the truth: reopen and check states survived.
    let store = KnowledgeStore::open(&project);
    let hyps = store
        .all()
        .iter()
        .filter(|i| i.state == KnowledgeState::Hypothesis)
        .count();
    assert_eq!(hyps, 2);
    assert!(
        store
            .all()
            .iter()
            .all(|i| i.state != KnowledgeState::Verified),
        "nothing verifies without evidence"
    );
}

#[test]
fn idle_loop_sleeps_when_nothing_open() {
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let project = scratch_project("empty");
    let outcomes = rt.block_on(knowledge::dream_loop(&project, 5, false));
    assert!(outcomes.is_empty(), "{:?}", outcomes);
}

#[test]
fn failures_deprioritize_without_deleting() {
    let project = scratch_project("prio");
    let mut store = KnowledgeStore::open(&project);
    knowledge::extract_gaps(&demo_task(), &mut store);
    // First round: every concept surfaces once (breadth-first).
    let mut round1 = Vec::new();
    for _ in 0..4 {
        round1.push(knowledge::prioritize(&mut store).expect("open question"));
    }
    assert_eq!(round1[0], "backoff", "{:?}", round1);
    // Fail "backoff" twice. Second round must skip it: equal visits,
    // higher failures sorts last.
    store.record_failure("backoff");
    store.record_failure("backoff");
    let next = knowledge::prioritize(&mut store).expect("open question");
    assert_ne!(next, "backoff", "failures must deprioritize");
    // Nothing was deleted or retired by failing.
    assert!(store.get("backoff").is_some());
    assert_ne!(
        store.get("backoff").unwrap().state,
        KnowledgeState::Rejected
    );
}
