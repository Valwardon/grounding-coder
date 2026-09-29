//! The first real experiment: a family of word-count-shaped problems
//! runs in two arms — dreaming (shared store + idle passes) vs
//! amnesiac (fresh store per task). Solving is real in both arms
//! (fresh bot + scratch project every task); only the knowledge
//! store carries state. Durations are recorded, never asserted.
use grounding_coder::engine::experiment::{TaskObservation, run_family};

fn work_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "gc-exp-{}-{}",
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("work dir");
    dir
}

fn count_task(fn_name: &str, input: &str, expected: &str) -> (String, TaskObservation) {
    let intent = serde_json::json!({
        "goal": format!("Count with {}", fn_name),
        "file": "src/lib.rs",
        "language": "rust",
        "actions": [],
        "references": [],
        "define": [{
            "name": fn_name,
            "kind": "function",
            "references": [],
            "signature": format!("fn {}(text: &str) -> HashMap<String, usize>", fn_name),
            "cases": [{"input": input, "expected": expected}]
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
    (
        fn_name.to_string(),
        TaskObservation {
            intent_json: serde_json::to_string(&intent).unwrap(),
            concepts_used: vec!["hashmap".to_string(), "hashmap-entry".to_string()],
            unknowns_seen: vec!["entry-api".to_string(), "or-insert".to_string()],
            related: vec!["hashset".to_string()],
            research_requests: 0,
        },
    )
}

/// Same pattern, new instance — the family the synthesizer can
/// honestly solve. A proves the pattern; B and C test whether the
/// loop remembers it.
fn family() -> Vec<(String, TaskObservation)> {
    vec![
        count_task(
            "word_counts",
            "\"a a b\"",
            "HashMap::from([(\"a\".to_string(), 2usize), (\"b\".to_string(), 1usize)])",
        ),
        count_task(
            "word_tally",
            "\"foo foo bar\"",
            "HashMap::from([(\"foo\".to_string(), 2usize), (\"bar\".to_string(), 1usize)])",
        ),
        count_task(
            "token_counts",
            "\"a b a c a\"",
            "HashMap::from([(\"a\".to_string(), 3usize), (\"b\".to_string(), 1usize), (\"c\".to_string(), 1usize)])",
        ),
    ]
}

#[tokio::test]
async fn dreaming_arm_recognizes_family_patterns() {
    let work = work_dir("dream");
    let report = run_family("dream", &work, family(), true, 3, false, 5).await;
    eprintln!(
        "dream arm: solved={} recognized={} new={} rate={:.2} verified={} durations={:?}",
        report.solved_all(),
        report.recognized_total(),
        report.new_total(),
        report.rediscovery_rate(),
        report.verified_total(),
        report
            .tasks
            .iter()
            .map(|t| t.duration_ms)
            .collect::<Vec<_>>(),
    );
    assert!(report.solved_all(), "family must solve: {:?}", report.tasks);
    // The pattern carries: B and C re-meet entry-api/or-insert/hashset
    // as recognized gaps, not fresh discoveries.
    assert!(
        report.recognized_total() >= 3,
        "dreaming must carry the pattern: {:?}",
        report.tasks
    );
    assert!(
        report.rediscovery_rate() > 0.0,
        "rate must be nonzero: {:?}",
        report.tasks
    );
    // Offline dreaming holds hypotheses but verifies nothing — the
    // DREAM != KNOWLEDGE firewall holds inside the experiment too.
    assert_eq!(report.verified_total(), 0);
    assert!(
        report.dreams.iter().all(|d| d.investigations <= 3),
        "budget binds every pass: {:?}",
        report.dreams
    );
    assert_eq!(
        report.dreams.len(),
        2,
        "dream between tasks, never after the last"
    );
}

#[tokio::test]
async fn amnesiac_arm_starts_from_zero() {
    let work = work_dir("amnesiac");
    let report = run_family("amnesiac", &work, family(), false, 0, false, 5).await;
    eprintln!(
        "amnesiac arm: solved={} recognized={} new={} rate={:.2}",
        report.solved_all(),
        report.recognized_total(),
        report.new_total(),
        report.rediscovery_rate(),
    );
    assert!(
        report.solved_all(),
        "baseline must solve too: {:?}",
        report.tasks
    );
    assert_eq!(
        report.recognized_total(),
        0,
        "nothing carries without the store"
    );
    assert_eq!(report.rediscovery_rate(), 0.0);
    assert!(report.dreams.is_empty(), "no dream passes in the baseline");
    assert!(report.new_total() > 0, "every gap is fresh without memory");
}
