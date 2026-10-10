//! Temporal validity of committed facts: a claim can be true and then
//! refuted or superseded. `valid_from`/`valid_to` record that; reads
//! skip the expired, nothing is ever deleted, and the palace can show
//! both the active and the retired.

use chrono::{Duration, Utc};
use grounding_coder::engine::knowledge::{
    CompletedTask, KnowledgeItem, KnowledgeState, KnowledgeStore, extract_gaps, prioritize,
};
use grounding_coder::engine::palace::Palace;
use std::sync::atomic::{AtomicU64, Ordering};

static CTR: AtomicU64 = AtomicU64::new(95000);

fn dir() -> std::path::PathBuf {
    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("gc-temporal-{}-{}", std::process::id(), id));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).expect("tmp");
    base
}

fn verified_expiring(concept: &str, expired: bool) -> KnowledgeItem {
    let mut item = KnowledgeItem::new(concept, KnowledgeState::Verified, "test");
    item.provenance.verification = Some("probe green".to_string());
    item.valid_from = Some(Utc::now() - Duration::days(10));
    item.valid_to = Some(if expired {
        Utc::now() - Duration::days(1)
    } else {
        Utc::now() + Duration::days(30)
    });
    item
}

#[test]
fn only_live_questions_are_prioritized() {
    let d = dir();
    let mut store = KnowledgeStore::open(&d);
    let mut past = KnowledgeItem::new("legacy-question", KnowledgeState::Question, "test");
    past.valid_to = Some(Utc::now() - Duration::days(1));
    store.insert(past);
    let mut future = KnowledgeItem::new("future-question", KnowledgeState::Question, "test");
    future.valid_to = Some(Utc::now() + Duration::days(1));
    store.insert(future);
    store.insert(KnowledgeItem::new(
        "open-question",
        KnowledgeState::Question,
        "test",
    ));

    let mut seen = Vec::new();
    for _ in 0..6 {
        match prioritize(&mut store) {
            Some(c) => seen.push(c),
            None => break,
        }
    }
    assert!(seen.contains(&"open-question".to_string()), "{:?}", seen);
    assert!(seen.contains(&"future-question".to_string()), "{:?}", seen);
    assert!(
        !seen.contains(&"legacy-question".to_string()),
        "an expired question must never surface: {:?}",
        seen
    );
}

#[test]
fn expiring_never_deletes() {
    let d = dir();
    let mut store = KnowledgeStore::open(&d);
    store.insert(KnowledgeItem::new("fact", KnowledgeState::Question, "test"));
    assert!(store.expire("fact"), "known concept retires");
    let item = store.get("fact").expect("kept on disk");
    assert!(item.is_expired());
    assert!(item.valid_to.is_some());
    // Re-open: the append-only file still carries the retired record.
    let reopened = KnowledgeStore::open(&d);
    assert!(reopened.get("fact").expect("persisted").is_expired());
    assert!(
        !store.expire("ghost"),
        "unknown concept is not an error, just false"
    );
}

#[test]
fn superseding_stamps_valid_to_on_the_old_fact() {
    let d = dir();
    let mut store = KnowledgeStore::open(&d);
    let mut old = KnowledgeItem::new("pinned-version", KnowledgeState::Verified, "test");
    old.provenance.verification = Some("probe green".to_string());
    store.insert(old);

    let mut new = KnowledgeItem::new("next-version", KnowledgeState::Verified, "test");
    new.provenance.verification = Some("probe green".to_string());
    assert!(store.supersede("pinned-version", new));

    let old = store.get("pinned-version").expect("old kept");
    assert!(old.valid_to.is_some(), "replacing must stamp the old");
    assert!(old.is_expired());
    let new = store.get("next-version").expect("replacement committed");
    assert!(!new.is_expired());
    assert!(new.valid_from.is_some(), "new fact records when it began");
}

#[test]
fn gaps_do_not_revive_expired_memory() {
    let d = dir();
    let mut store = KnowledgeStore::open(&d);
    let mut stale = KnowledgeItem::new("cron", KnowledgeState::Question, "test");
    stale.valid_to = Some(Utc::now() - Duration::days(1));
    store.insert(stale);

    let task = CompletedTask {
        goal: "build a scheduler".to_string(),
        concepts_used: vec![],
        unknowns_seen: vec!["cron".to_string()],
        failed_attempts: 0,
        related: vec![],
    };
    let gaps = extract_gaps(&task, &mut store);
    assert!(!gaps.contains(&"cron".to_string()), "{:?}", gaps);
    // Retired, not deleted.
    assert!(store.get("cron").unwrap().is_expired());
}

#[test]
fn expired_verified_members_are_not_consolidated() {
    let d = dir();
    let mut store = KnowledgeStore::open(&d);
    let deps = vec![
        "d1".to_string(),
        "d2".to_string(),
        "d3".to_string(),
        "d4".to_string(),
    ];
    let mut alpha = KnowledgeItem::new("alpha", KnowledgeState::Verified, "test");
    alpha.provenance.verification = Some("probe green".to_string());
    alpha.dependencies = deps.clone();
    store.insert(alpha);
    let mut beta = verified_expiring("beta", false);
    beta.dependencies = deps.clone();
    store.insert(beta);
    let mut gamma = verified_expiring("gamma", true);
    gamma.dependencies = deps;
    store.insert(gamma);

    // Two live supporters (alpha, beta) cluster; the expired gamma is
    // not a supporter and is not swept into the pattern.
    let created = store.consolidate();
    assert_eq!(created, vec!["cluster:alpha+beta".to_string()]);
    assert!(
        !store.get("gamma").unwrap().dormant,
        "expiry is not dormancy"
    );
    assert!(store.get("gamma").unwrap().is_expired());
}

#[test]
fn legacy_lines_load_without_timestamps() {
    let item: KnowledgeItem = serde_json::from_str(
        r#"{"concept":"flux","state":"Question","dependencies":[],
            "provenance":{"source":"s","experiment":null,"verification":null,"rejection":null},
            "visits":0,"failures":0}"#,
    )
    .expect("legacy line loads");
    assert!(item.valid_from.is_none());
    assert!(item.valid_to.is_none());
    assert!(!item.is_expired(), "no expiry means always valid");
}

#[test]
fn palace_separates_active_and_expired_knowledge() {
    let d = dir();
    let mut store = KnowledgeStore::open(&d);
    store.insert(KnowledgeItem::new("live", KnowledgeState::Question, "test"));
    let mut retired = KnowledgeItem::new("retired", KnowledgeState::Verified, "test");
    retired.valid_to = Some(Utc::now() - Duration::days(1));
    store.insert(retired);

    let palace = Palace::build(&d);
    let active: Vec<&str> = palace.active().iter().map(|l| l.room.as_str()).collect();
    let expired: Vec<&str> = palace.expired().iter().map(|l| l.room.as_str()).collect();
    assert_eq!(active, vec!["live"]);
    assert_eq!(expired, vec!["retired"]);
    assert!(
        palace.unresolved().is_empty(),
        "retired memory is still viewable"
    );
}
