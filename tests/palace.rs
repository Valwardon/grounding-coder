//! The palace in the real repository: every committed record becomes a
//! leaf, every leaf resolves back to its source line, and the same
//! files always build the same index. If a leaf cannot be viewed, the
//! test fails — memory that cannot be viewed is not trusted.
use grounding_coder::engine::palace::Palace;
use std::path::Path;

fn repo_palace() -> Palace {
    // `cargo test` runs with the package root as the working directory.
    Palace::build(Path::new("."))
}

#[test]
fn indexes_every_committed_record() {
    let palace = repo_palace();
    let counts = palace.wing_counts();
    let get = |w: &str| {
        counts
            .iter()
            .find(|(name, _)| name == w)
            .map(|(_, n)| *n)
            .unwrap_or(0)
    };
    // One leaf per non-blank line across the four journals; a whole-file
    // leaf per measured plate. Invented nothing, dropped nothing.
    assert_eq!(get("decisions"), 20, "{:?}", counts);
    assert_eq!(get("lessons"), 7, "{:?}", counts);
    assert_eq!(get("encounters"), 5, "{:?}", counts);
    assert_eq!(get("clarifications"), 30, "{:?}", counts);
    assert_eq!(get("visual"), 12, "{:?}", counts);
}

#[test]
fn every_leaf_resolves_to_its_source() {
    let palace = repo_palace();
    let orphans = palace.unresolved();
    assert!(
        orphans.is_empty(),
        "orphan leaves (memory that cannot be viewed): {:?}",
        orphans
    );
    assert!(
        palace.leaves.len() >= 74,
        "expected every record indexed, got {}",
        palace.leaves.len()
    );
}

#[test]
fn build_is_deterministic() {
    let a = repo_palace();
    let b = repo_palace();
    assert_eq!(a.render(), b.render());
}

#[test]
fn scoped_query_finds_committed_memory_by_name() {
    let palace = repo_palace();
    // A decision by topic.
    assert!(
        palace
            .query("discriminator-collapse")
            .iter()
            .any(|l| l.wing == "decisions"),
        "decision not found"
    );
    // A filed rule by its lesson key.
    assert!(
        palace
            .query("L-ground-escape")
            .iter()
            .any(|l| l.wing == "lessons"),
        "lesson not found"
    );
    // The same aspect living in two different stores at once.
    let region: Vec<&str> = palace
        .query("subject-region")
        .iter()
        .map(|l| l.wing.as_str())
        .collect();
    assert!(region.contains(&"encounters"), "{:?}", region);
    assert!(region.contains(&"clarifications"), "{:?}", region);
    // A measured plate by its researched title.
    assert!(
        palace
            .query("Portrait of a Woman Standing")
            .iter()
            .any(|l| l.wing == "visual"),
        "plate not found"
    );
}
