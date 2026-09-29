//! Compiler-probe evidence: the stronger verification path for code
//! knowledge.
//!
//! A sourced summary says "the documentation claims X". A probe says
//! "X was experimentally demonstrated in this environment": the
//! smallest program exhibiting the behavior, run against the real
//! toolchain oracle (`cargo check` + `clippy` + `cargo test` via
//! [`super::verifier::CodeVerifier`]).
//!
//! ```text
//! research → candidate claim → tiny probe → oracle → evidence → verified
//! ```
//!
//! Probes run in disposable scratch projects under the caller's work
//! dir, with zero dependencies so no network is ever needed. The
//! oracle runs on its own thread under a real timeout — a hanging
//! toolchain waits out the clock, it doesn't own the loop (an
//! orphaned cargo finishes on its own; we stop waiting and say so).
//! A failing probe records honest negative evidence — it never
//! promotes, and it never deletes.
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::knowledge::{KnowledgeState, KnowledgeStore, Provenance, VerificationTier};
use super::verifier::CodeVerifier;

/// One experimental demonstration of one claim.
pub struct ProbeSpec {
    /// Short slug: used for the scratch dir and the evidence string.
    pub name: String,
    /// The claim under test, in words ("HashMap::entry or_insert
    /// accumulates counts").
    pub claim: String,
    /// Complete `src/main.rs` content, tests included. Must be
    /// self-contained (std only) so probes never need the network.
    pub code: String,
    /// Stdout markers the oracle output must contain (e.g.
    /// `"test result: ok"`). Empty means the oracle verdict alone
    /// decides.
    pub must_contain: Vec<String>,
}

/// What the oracle said about the probe.
#[derive(Debug, Clone)]
pub struct ProbeOutcome {
    pub name: String,
    pub passed: bool,
    /// The evidence string on success (goes into the store), or the
    /// honest failure reason on failure.
    pub evidence: String,
    pub duration_ms: u64,
}

fn slug(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

fn unique_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time")
        .as_nanos()
}

/// Build the probe project, run the oracle under `timeout`, judge the
/// verdict. The directory survives the run — evidence you can re-read
/// beats a transcript you must trust.
pub async fn run_probe(spec: &ProbeSpec, work_dir: &Path, timeout: Duration) -> ProbeOutcome {
    let start = Instant::now();
    let elapsed = || start.elapsed().as_millis() as u64;
    let dir: PathBuf = work_dir.join(format!("probe-{}-{}", slug(&spec.name), unique_suffix()));
    let fail = |reason: String| ProbeOutcome {
        name: spec.name.clone(),
        passed: false,
        evidence: format!("probe {:?}: {}", spec.name, reason),
        duration_ms: elapsed(),
    };
    if std::fs::create_dir_all(dir.join("src")).is_err() {
        return fail("could not create scratch project".to_string());
    }
    let manifest = "[package]\nname = \"gc-probe\"\nversion = \"0.1.0\"\nedition = \"2021\"\n";
    if std::fs::write(dir.join("Cargo.toml"), manifest).is_err()
        || std::fs::write(dir.join("src/main.rs"), &spec.code).is_err()
    {
        return fail("could not write probe sources".to_string());
    }
    // The verifier is synchronous inside (blocking cargo invocations),
    // so an async timeout could never fire mid-run. A fresh OS thread
    // with its own one-shot channel gives a real deadline instead.
    let (tx, rx) = std::sync::mpsc::channel();
    let thread_dir = dir.clone();
    std::thread::spawn(move || {
        let verifier = CodeVerifier::new(thread_dir);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build();
        let verdict = match rt {
            Ok(rt) => rt.block_on(verifier.verify()),
            Err(e) => {
                let _ = tx.send(Err(format!("probe runtime failed: {}", e)));
                return;
            }
        };
        let _ = tx.send(Ok(verdict));
    });
    let verdict = match rx.recv_timeout(timeout) {
        Err(_) => {
            return fail(format!(
                "oracle timed out after {}s (orphaned toolchain left to finish)",
                timeout.as_secs()
            ));
        }
        Ok(Err(e)) => return fail(e),
        Ok(Ok(v)) => v,
    };
    let duration_ms = elapsed();
    if !(verdict.clean && verdict.errors.is_empty() && verdict.full) {
        let first = verdict
            .errors
            .first()
            .map(|e| format!("{}: {}", e.code, e.message))
            .unwrap_or_else(|| "oracle not clean".to_string());
        return ProbeOutcome {
            name: spec.name.clone(),
            passed: false,
            evidence: format!("probe {:?} failed oracle: {}", spec.name, first),
            duration_ms,
        };
    }
    for marker in &spec.must_contain {
        if !verdict.stdout.contains(marker) {
            return ProbeOutcome {
                name: spec.name.clone(),
                passed: false,
                evidence: format!(
                    "probe {:?} clean but missing marker {:?} in oracle output",
                    spec.name, marker
                ),
                duration_ms,
            };
        }
    }
    ProbeOutcome {
        name: spec.name.clone(),
        passed: true,
        evidence: format!(
            "probe {:?}: oracle clean (full) in {}ms at {}",
            spec.name,
            duration_ms,
            dir.display()
        ),
        duration_ms,
    }
}

/// Walk a concept through Hypothesis → Experiment → Evidence, then
/// promote to Verified only on a passing probe. Accepts concepts in
/// `Question` (hypothesis is formed from the claim) or `Hypothesis`
/// state; anything else is refused with the reason — probes test
/// unknowns, not trophies, and disproven concepts stay disproven
/// until explicitly reopened (no reopen API in v1).
pub async fn verify_concept_by_probe(
    store: &mut KnowledgeStore,
    concept: &str,
    spec: &ProbeSpec,
    work_dir: &Path,
    timeout: Duration,
) -> Result<ProbeOutcome, String> {
    let state = store
        .get(concept)
        .map(|i| i.state)
        .ok_or_else(|| format!("unknown concept {:?}", concept))?;
    match state {
        KnowledgeState::Question => {
            store.transition(
                concept,
                KnowledgeState::Hypothesis,
                Provenance {
                    source: format!("probe hypothesis: {}", spec.claim),
                    experiment: None,
                    verification: None,
                    rejection: None,
                    tier: VerificationTier::Sourced,
                },
            )?;
        }
        KnowledgeState::Hypothesis => {}
        KnowledgeState::Verified | KnowledgeState::Generalized => {
            return Err(format!(
                "{:?} is already known — probes test unknowns, not trophies",
                concept
            ));
        }
        KnowledgeState::Rejected => {
            return Err(format!(
                "{:?} is disproven — reopen it explicitly before re-probing",
                concept
            ));
        }
        _ => {
            return Err(format!(
                "{:?} is mid-investigation ({:?}) — let the pass finish",
                concept, state
            ));
        }
    }
    store.transition(
        concept,
        KnowledgeState::Experiment,
        Provenance {
            source: format!("probe experiment: {}", spec.claim),
            experiment: Some(format!("probe {:?}", spec.name)),
            verification: None,
            rejection: None,
            tier: VerificationTier::Sourced,
        },
    )?;
    let outcome = run_probe(spec, work_dir, timeout).await;
    // Evidence either way: the run happened and said something.
    store.transition(
        concept,
        KnowledgeState::Evidence,
        Provenance {
            source: format!("probe ran: {}", spec.claim),
            experiment: Some(format!("probe {:?}", spec.name)),
            verification: None,
            rejection: None,
            tier: VerificationTier::Sourced,
        },
    )?;
    if outcome.passed {
        store.transition(
            concept,
            KnowledgeState::Verified,
            Provenance {
                source: format!("probe ran: {}", spec.claim),
                experiment: Some(format!("probe {:?}", spec.name)),
                verification: Some(outcome.evidence.clone()),
                rejection: None,
                tier: VerificationTier::Demonstrated,
            },
        )?;
    } else {
        // Negative evidence is still evidence: counted as a failure
        // (deprioritizes future passes), promotion refused. The
        // concept can be re-probed later; nothing is deleted.
        store.record_failure(concept);
    }
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::super::knowledge::KnowledgeItem;
    use super::*;
    use std::time::Duration;

    fn work_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("gc-probe-test-{}", unique_suffix()));
        std::fs::create_dir_all(&dir).expect("work dir");
        dir
    }

    fn mem_store() -> KnowledgeStore {
        KnowledgeStore::open(&work_dir().join("store"))
    }

    fn entry_spec() -> ProbeSpec {
        ProbeSpec {
            name: "entry-accumulates".to_string(),
            claim: "HashMap::entry or_insert accumulates counts".to_string(),
            code: "use std::collections::HashMap;\nfn main() {}\n#[test]\nfn accumulates() {\n    let mut m: HashMap<String, usize> = HashMap::new();\n    for w in \"a a b\".split_whitespace() {\n        *m.entry(w.to_string()).or_insert(0) += 1;\n    }\n    assert_eq!(m[\"a\"], 2);\n    assert_eq!(m[\"b\"], 1);\n}\n"
                .to_string(),
            must_contain: vec!["test result: ok".to_string()],
        }
    }

    #[tokio::test]
    async fn green_probe_verifies() {
        let work = work_dir();
        let out = run_probe(&entry_spec(), &work, Duration::from_secs(240)).await;
        assert!(out.passed, "green probe must pass: {}", out.evidence);
        assert!(out.evidence.contains("oracle clean"), "{}", out.evidence);
    }

    #[tokio::test]
    async fn red_probe_records_negative_evidence() {
        let work = work_dir();
        let mut spec = entry_spec();
        spec.name = "entry-red".to_string();
        spec.code = spec
            .code
            .replace("assert_eq!(m[\"a\"], 2);", "assert_eq!(m[\"a\"], 99);");
        let out = run_probe(&spec, &work, Duration::from_secs(240)).await;
        assert!(!out.passed, "failing assert must not verify");
        assert!(
            out.evidence.contains("failed oracle") || out.evidence.contains("missing marker"),
            "{}",
            out.evidence
        );
    }

    #[tokio::test]
    async fn uncompilable_probe_never_verifies() {
        let work = work_dir();
        let mut spec = entry_spec();
        spec.name = "entry-broken".to_string();
        spec.code = "fn main() { let _x: u32 = \"not a number\"; }\n".to_string();
        let out = run_probe(&spec, &work, Duration::from_secs(240)).await;
        assert!(!out.passed, "type error must not verify");
        assert!(out.evidence.contains("failed oracle"), "{}", out.evidence);
    }

    #[tokio::test]
    async fn hanging_probe_waits_out_the_clock() {
        let work = work_dir();
        let mut spec = entry_spec();
        spec.name = "entry-slow".to_string();
        spec.code = "fn main() {}\n#[test]\nfn slow() { std::thread::sleep(std::time::Duration::from_secs(120)); }\n"
            .to_string();
        spec.must_contain = Vec::new();
        let out = run_probe(&spec, &work, Duration::from_secs(8)).await;
        assert!(!out.passed, "timeout must not verify");
        assert!(out.evidence.contains("timed out"), "{}", out.evidence);
        assert!(
            out.duration_ms < 60_000,
            "bounded wait, got {}ms",
            out.duration_ms
        );
    }

    #[tokio::test]
    async fn verify_concept_walks_the_chain() {
        let work = work_dir();
        let mut store = mem_store();
        store.insert(KnowledgeItem::new(
            "entry-api",
            KnowledgeState::Question,
            "test",
        ));
        let out = verify_concept_by_probe(
            &mut store,
            "entry-api",
            &entry_spec(),
            &work,
            Duration::from_secs(240),
        )
        .await
        .expect("walk");
        assert!(out.passed, "{}", out.evidence);
        let item = store.get("entry-api").expect("stored");
        assert_eq!(item.state, KnowledgeState::Verified);
        assert_eq!(
            item.provenance.tier,
            super::super::knowledge::VerificationTier::Demonstrated,
            "probe evidence is demonstrated, not merely sourced"
        );
        assert!(
            item.provenance
                .verification
                .as_ref()
                .is_some_and(|v| v.contains("oracle clean"))
        );
        // Known concepts refuse re-probing.
        assert!(
            verify_concept_by_probe(
                &mut store,
                "entry-api",
                &entry_spec(),
                &work,
                Duration::from_secs(60),
            )
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn failed_probe_leaves_concept_unverified() {
        let work = work_dir();
        let mut store = mem_store();
        store.insert(KnowledgeItem::new(
            "shaky",
            KnowledgeState::Hypothesis,
            "test",
        ));
        let mut spec = entry_spec();
        spec.name = "shaky-red".to_string();
        spec.code = spec
            .code
            .replace("assert_eq!(m[\"a\"], 2);", "assert_eq!(m[\"a\"], 99);");
        let out =
            verify_concept_by_probe(&mut store, "shaky", &spec, &work, Duration::from_secs(240))
                .await
                .expect("walk records failure, not error");
        assert!(!out.passed);
        let item = store.get("shaky").expect("stored");
        assert_eq!(item.state, KnowledgeState::Evidence);
        assert_eq!(item.failures, 1);
        assert!(item.provenance.verification.is_none());
    }
}
