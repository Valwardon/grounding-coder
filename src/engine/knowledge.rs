//! Proactive curiosity and the idle learning loop.
//!
//! The request path already goes understand → research → build →
//! verify → remember. This module is the second half of that arc:
//! solved work is inspected for gaps, gaps become curiosities,
//! curiosities are investigated on spare cycles, and only verified
//! results become knowledge.
//!
//! ```text
//! CompletedTask → gaps → questions → hypotheses → experiments
//!     → evidence → VERIFIED → generalized patterns → new gaps
//! ```
//!
//! The defining invariant is **DREAM ≠ KNOWLEDGE**: a dream can be
//! wrong, a hypothesis can be wrong, and nothing reaches `Verified`
//! without evidence that passes [`KnowledgeStore::promote`]. Invalid
//! generalizations are `Rejected` — recorded, so they are never
//! re-learned — instead of silently kept.
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};

/// Lifecycle of one concept. Forward motion always passes through
/// evidence; anything else is a dream wearing a lab coat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KnowledgeState {
    /// Heard of, nothing more (a related concept, an import name).
    Unknown,
    /// Surfaced as a gap worth investigating.
    Question,
    /// A guess formed, no evidence yet. Dreams live here.
    Hypothesis,
    /// An investigation designed (probe description, lookup plan).
    Experiment,
    /// The experiment ran; the honest result is recorded, pass or fail.
    Evidence,
    /// Evidence passed the verification rules. Only this and
    /// `Generalized` may guide future builds.
    Verified,
    /// A pattern across two or more verified items.
    Generalized,
    /// Disproven or unverifiable. Terminal: never re-questioned.
    Rejected,
}

/// Strength of the evidence behind a Verified item. A sourced
/// summary and a green compiler probe both verify — but they are not
/// the same claim, and generalizing across tiers without noting it
/// is how overconfidence accumulates. The tier rides with the item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum VerificationTier {
    /// A reputable source states it (web lookup, citation).
    Sourced,
    /// It ran green against a toolchain (compiler probe).
    Demonstrated,
    /// It measured out on our own renders (anatomy study).
    Measured,
}

impl Default for VerificationTier {
    /// Old journal lines carry no tier; they read as sourced — the
    /// weaker claim, never the stronger.
    fn default() -> Self {
        VerificationTier::Sourced
    }
}

impl VerificationTier {
    /// Strength order: a cited summary < a measured render < a green
    /// toolchain run. Generalizations take the weakest supporter.
    fn rank(self) -> u8 {
        match self {
            VerificationTier::Sourced => 0,
            VerificationTier::Measured => 1,
            VerificationTier::Demonstrated => 2,
        }
    }

    fn from_rank(rank: u8) -> Self {
        match rank {
            0 => VerificationTier::Sourced,
            1 => VerificationTier::Measured,
            _ => VerificationTier::Demonstrated,
        }
    }
}

/// Where a claim came from. Every edge in the graph carries this.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Provenance {
    /// Human-readable origin ("gap from task: ...", "web lookup", ...).
    pub source: String,
    /// What was tried ("lookup on verified sources", "probe compiles").
    pub experiment: Option<String>,
    /// What passed ("source_url=...", "cargo test green"). Required
    /// for promotion to [`KnowledgeState::Verified`].
    pub verification: Option<String>,
    /// How strongly it passed. Defaults to Sourced for old lines.
    #[serde(default)]
    pub tier: VerificationTier,
    /// Why a concept was rejected, if it was.
    pub rejection: Option<String>,
}

/// One concept and everything proven (or disproven) about it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KnowledgeItem {
    pub concept: String,
    pub state: KnowledgeState,
    /// Concepts this one builds on — the adjacency the
    /// what-next engine walks ("Task → CancellationToken").
    pub dependencies: Vec<String>,
    pub provenance: Provenance,
    /// Times surfaced as a curiosity (drives prioritization).
    pub visits: u32,
    /// Failed investigation attempts (deprioritizes, never deletes).
    pub failures: u32,
    /// Sleeping, not gone. A dormant item keeps its provenance and its
    /// edges but never surfaces: its content was either compressed
    /// into a pattern (`absorbed_by`) or primed as unusable (rejected,
    /// or worn out by failures). Attention stays on live memory.
    #[serde(default)]
    pub dormant: bool,
    /// The pattern that subsumes this item, when compression put it to
    /// sleep. `None` for items primed for a reason other than
    /// consolidation.
    #[serde(default)]
    pub absorbed_by: Option<String>,
    /// When this item's claim took effect. Set at creation; `None` on
    /// legacy lines reads as "always".
    #[serde(default)]
    pub valid_from: Option<DateTime<Utc>>,
    /// When the claim stopped holding — refuted or superseded. `None`
    /// means it still holds. Expired memory stays on disk but never
    /// surfaces; it is evidence, not garbage.
    #[serde(default)]
    pub valid_to: Option<DateTime<Utc>>,
}

impl KnowledgeItem {
    pub fn new(concept: &str, state: KnowledgeState, source: &str) -> Self {
        KnowledgeItem {
            concept: concept.to_string(),
            state,
            dependencies: Vec::new(),
            provenance: Provenance {
                source: source.to_string(),
                experiment: None,
                verification: None,
                rejection: None,
                tier: VerificationTier::Sourced,
            },
            visits: 0,
            failures: 0,
            dormant: false,
            absorbed_by: None,
            valid_from: Some(Utc::now()),
            valid_to: None,
        }
    }

    /// Whether the claim had stopped holding at `now` (superseded or
    /// refuted). `now` is explicit so tests measure, never race a clock.
    pub fn is_expired_at(&self, now: DateTime<Utc>) -> bool {
        self.valid_to.is_some_and(|t| t <= now)
    }

    /// Expired as of the moment of asking: off attention, still on disk.
    pub fn is_expired(&self) -> bool {
        self.is_expired_at(Utc::now())
    }
}

/// Append-only JSONL store, one line per item snapshot. The file is
/// the truth; memory is a cache rebuilt by [`KnowledgeStore::open`].
pub struct KnowledgeStore {
    path: PathBuf,
    items: HashMap<String, KnowledgeItem>,
}

impl KnowledgeStore {
    /// Open (or create) `<project>/.grounding/knowledge.jsonl`.
    pub fn open(project_dir: &Path) -> Self {
        let dir = project_dir.join(".grounding");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("knowledge.jsonl");
        let mut items = HashMap::new();
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines() {
                if let Ok(item) = serde_json::from_str::<KnowledgeItem>(line) {
                    items.insert(item.concept.clone(), item);
                }
            }
        }
        KnowledgeStore { path, items }
    }

    fn persist(&self) {
        let mut out = String::new();
        let mut concepts: Vec<&String> = self.items.keys().collect();
        concepts.sort();
        for c in concepts {
            if let Ok(line) = serde_json::to_string(&self.items[c]) {
                out.push_str(&line);
                out.push('\n');
            }
        }
        let _ = std::fs::write(&self.path, out);
    }

    pub fn get(&self, concept: &str) -> Option<&KnowledgeItem> {
        self.items.get(concept)
    }

    pub fn all(&self) -> Vec<&KnowledgeItem> {
        let mut v: Vec<&KnowledgeItem> = self.items.values().collect();
        v.sort_by(|a, b| a.concept.cmp(&b.concept));
        v
    }

    /// Insert a fresh item. Existing items are never overwritten —
    /// use [`KnowledgeStore::transition`] to move them.
    pub fn insert(&mut self, item: KnowledgeItem) {
        self.items.entry(item.concept.clone()).or_insert(item);
        self.persist();
    }

    /// Legal transitions. Anything else is rejected with the reason,
    /// and promotion to `Verified` additionally requires verification
    /// evidence — this is the DREAM ≠ KNOWLEDGE firewall.
    pub fn transition(
        &mut self,
        concept: &str,
        to: KnowledgeState,
        provenance: Provenance,
    ) -> Result<(), String> {
        let item = self
            .items
            .get_mut(concept)
            .ok_or_else(|| format!("unknown concept {:?}", concept))?;
        let from = item.state;
        let legal = matches!(
            (from, to),
            (KnowledgeState::Unknown, KnowledgeState::Question)
                | (KnowledgeState::Question, KnowledgeState::Hypothesis)
                | (KnowledgeState::Hypothesis, KnowledgeState::Experiment)
                | (KnowledgeState::Experiment, KnowledgeState::Evidence)
                | (KnowledgeState::Evidence, KnowledgeState::Verified)
                | (KnowledgeState::Evidence, KnowledgeState::Rejected)
                | (KnowledgeState::Hypothesis, KnowledgeState::Rejected)
                | (KnowledgeState::Verified, KnowledgeState::Generalized)
        );
        if !legal {
            return Err(format!(
                "illegal transition {:?} -> {:?} for {:?}",
                from, to, concept
            ));
        }
        if to == KnowledgeState::Verified
            && provenance
                .verification
                .as_ref()
                .is_none_or(|v| v.trim().is_empty())
        {
            return Err(format!(
                "DREAM != KNOWLEDGE: {:?} reaches Verified only with verification evidence",
                concept
            ));
        }
        item.state = to;
        if to == KnowledgeState::Evidence || to == KnowledgeState::Verified {
            item.provenance = provenance;
        } else if to == KnowledgeState::Rejected {
            item.provenance.rejection = provenance.rejection;
            if item.provenance.source.is_empty() {
                item.provenance.source = provenance.source;
            }
        } else {
            if !provenance.source.is_empty() {
                item.provenance.source = provenance.source;
            }
            if provenance.experiment.is_some() {
                item.provenance.experiment = provenance.experiment;
            }
        }
        self.persist();
        Ok(())
    }

    /// Record a failed investigation. Failures deprioritize; only an
    /// explicit [`KnowledgeStore::transition`] to `Rejected` retires.
    pub fn record_failure(&mut self, concept: &str) {
        if let Some(item) = self.items.get_mut(concept) {
            item.failures += 1;
            self.persist();
        }
    }

    /// Retire a fact as of now without deleting it: `valid_to` records
    /// when it stopped holding and every read path skips it afterwards.
    /// The append-only file keeps the record — dormant and expired alike
    /// are evidence. Returns false for an unknown concept.
    pub fn expire(&mut self, concept: &str) -> bool {
        if let Some(item) = self.items.get_mut(concept) {
            item.valid_to = Some(Utc::now());
            self.persist();
            true
        } else {
            false
        }
    }

    /// Supersede `old` with a freshly committed `replacement`: the old
    /// claim stops holding now, the new one starts (it carries its own
    /// `valid_from`). The old record is kept, never deleted. Returns
    /// whether `old` existed.
    pub fn supersede(&mut self, old: &str, replacement: KnowledgeItem) -> bool {
        let retired = self.expire(old);
        self.insert(replacement);
        retired
    }

    /// Promote a pattern across verified items. Requires at least two
    /// distinct verified supporters — a generalization from one
    /// example is a dream, and is `Rejected` instead of kept.
    pub fn generalize(&mut self, pattern: &str, supporting: &[String]) -> Result<(), String> {
        let verified: Vec<String> = supporting
            .iter()
            .filter(|c| {
                self.items.get(c.as_str()).is_some_and(|i| {
                    !i.is_expired()
                        && matches!(
                            i.state,
                            KnowledgeState::Verified | KnowledgeState::Generalized
                        )
                })
            })
            .cloned()
            .collect();
        if verified.len() >= 2 {
            let mut item = KnowledgeItem::new(
                pattern,
                KnowledgeState::Generalized,
                &format!("generalized from {:?}", verified),
            );
            item.dependencies = verified;
            item.provenance.verification = Some("2+ verified supporters".to_string());
            // A pattern is only as strong as its weakest supporter:
            // sourced claims don't launder into demonstrated ones.
            let weakest = item
                .dependencies
                .iter()
                .filter_map(|c| self.items.get(c.as_str()))
                .map(|i| i.provenance.tier.rank())
                .min()
                .unwrap_or(0);
            item.provenance.tier = VerificationTier::from_rank(weakest);
            self.insert(item);
            Ok(())
        } else {
            let mut item = KnowledgeItem::new(
                pattern,
                KnowledgeState::Rejected,
                &format!("generalization attempt from {:?}", supporting),
            );
            item.provenance.rejection = Some(format!(
                "only {}/2+ supporters verified — a dream, not a pattern",
                verified.len()
            ));
            self.insert(item);
            Err(format!(
                "rejected {:?}: only {}/2+ supporters verified",
                pattern,
                verified.len()
            ))
        }
    }

    /// Jaccard overlap of two dependency sets: shared edges over all
    /// edges. Two items with no dependencies score 0.0 — sharing
    /// nothing is not kinship.
    pub fn signature_overlap(a: &[String], b: &[String]) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }
        let set_a: std::collections::HashSet<&String> = a.iter().collect();
        let set_b: std::collections::HashSet<&String> = b.iter().collect();
        let inter = set_a.intersection(&set_b).count() as f64;
        let union = set_a.union(&set_b).count() as f64;
        if union == 0.0 { 0.0 } else { inter / union }
    }

    /// Consolidation: cluster verified items by dependency-signature
    /// overlap (≥0.80) and generalize each cluster. Discoveries, not
    /// declarations — the dream loop finds patterns instead of being
    /// handed supporters. Each cluster still passes through
    /// [`KnowledgeStore::generalize`], so the 2-supporter rule and
    /// the rejection path hold unchanged. On success the members are
    /// **compressed**: put to sleep (`dormant`, `absorbed_by` the
    /// pattern) so the pattern carries them and active memory holds N
    /// facts as one. Returns created patterns.
    pub fn consolidate(&mut self) -> Vec<String> {
        let mut verified: Vec<String> = self
            .items
            .values()
            .filter(|i| i.state == KnowledgeState::Verified && !i.dormant && !i.is_expired())
            .map(|i| i.concept.clone())
            .collect();
        verified.sort();
        let mut clustered = std::collections::HashSet::new();
        let mut created = Vec::new();
        for concept in &verified {
            if clustered.contains(concept) {
                continue;
            }
            let deps_c = self
                .items
                .get(concept)
                .map(|i| i.dependencies.clone())
                .unwrap_or_default();
            let mut cluster = vec![concept.clone()];
            for other in verified.iter() {
                if other == concept || clustered.contains(other) {
                    continue;
                }
                let deps_o = self
                    .items
                    .get(other)
                    .map(|i| i.dependencies.clone())
                    .unwrap_or_default();
                if Self::signature_overlap(&deps_c, &deps_o) >= 0.80 {
                    cluster.push(other.clone());
                }
            }
            if cluster.len() >= 2 {
                cluster.sort();
                let pattern = format!("cluster:{}", cluster.join("+"));
                let fresh = self.get(&pattern).is_none();
                if fresh && self.generalize(&pattern, &cluster).is_ok() {
                    created.push(pattern.clone());
                }
                // Compression: the freshly built pattern carries the
                // members, so put them to sleep. A pattern that already
                // existed is left as it was (converged, not re-merged).
                if fresh && self.get(&pattern).is_some() {
                    for member in &cluster {
                        self.mark_dormant(member, Some(pattern.clone()));
                    }
                    self.persist();
                }
                for member in &cluster {
                    clustered.insert(member.clone());
                }
            }
        }
        created
    }

    /// Put a concept to sleep: keep its provenance and edges, stop it
    /// surfacing. `absorbed_by` names the pattern that compressed it,
    /// when there is one.
    fn mark_dormant(&mut self, concept: &str, absorbed_by: Option<String>) -> bool {
        if let Some(item) = self.items.get_mut(concept) {
            item.dormant = true;
            item.absorbed_by = absorbed_by;
            true
        } else {
            false
        }
    }

    /// Priming: forget what will not be used, without deleting it.
    /// `Rejected` concepts (disproven) and hypotheses/evidence worn out
    /// by repeated failures go dormant — kept as evidence, off the
    /// active graph. Returns the concepts primed, sorted.
    pub fn prime(&mut self) -> Vec<String> {
        const WORN: u32 = 3;
        let mut primed: Vec<String> = self
            .items
            .values()
            .filter(|i| {
                !i.dormant
                    && !i.is_expired()
                    && (i.state == KnowledgeState::Rejected
                        || (matches!(
                            i.state,
                            KnowledgeState::Hypothesis | KnowledgeState::Evidence
                        ) && i.failures >= WORN))
            })
            .map(|i| i.concept.clone())
            .collect();
        primed.sort();
        if !primed.is_empty() {
            for concept in &primed {
                self.mark_dormant(concept, None);
            }
            self.persist();
        }
        primed
    }

    /// Concepts structurally adjacent to this one: its dependencies
    /// plus everything depending on it. The what-next engine walks
    /// these edges, not a syllabus.
    pub fn neighbors(&self, concept: &str) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(item) = self.items.get(concept) {
            out.extend(item.dependencies.iter().cloned());
        }
        for (name, item) in &self.items {
            if item.dependencies.iter().any(|d| d == concept) && name != concept {
                out.push(name.clone());
            }
        }
        out.sort();
        out.dedup();
        out
    }
}

/// What a finished task leaves behind for the dream loop: what the
/// solution depended on, what it brushed past without understanding,
/// what failed, and what nearby concepts the caller already suspects.
pub struct CompletedTask {
    pub goal: String,
    pub concepts_used: Vec<String>,
    pub unknowns_seen: Vec<String>,
    pub failed_attempts: u32,
    pub related: Vec<String>,
}

/// Turn a finished task into questions. Concepts already Verified,
/// Generalized, or Rejected are left alone — solved stays solved,
/// disproven stays disproven. Everything new enters as `Question`
/// with the task's used concepts as its dependency seeds.
pub fn extract_gaps(task: &CompletedTask, store: &mut KnowledgeStore) -> Vec<String> {
    let mut gaps = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let candidates: Vec<(&String, bool)> = task
        .unknowns_seen
        .iter()
        .map(|c| (c, false))
        .chain(task.related.iter().map(|c| (c, false)))
        .chain(
            task.concepts_used
                .iter()
                .filter(|c| store.get(c.as_str()).is_none())
                .map(|c| (c, true)),
        )
        .collect();
    for (concept, used) in candidates {
        if !seen.insert(concept.clone()) {
            continue;
        }
        if let Some(existing) = store.get(concept.as_str()) {
            if existing.dormant || existing.is_expired() {
                // Sleeping or retired memory does not wake just because
                // a task brushed past it — it was compressed, primed, or
                // superseded.
                continue;
            }
            match existing.state {
                KnowledgeState::Verified
                | KnowledgeState::Generalized
                | KnowledgeState::Rejected => continue,
                _ => {
                    gaps.push(concept.clone());
                    continue;
                }
            }
        }
        let mut item = KnowledgeItem::new(
            concept,
            KnowledgeState::Question,
            &format!(
                "gap from task {:?} ({} failed attempts{})",
                task.goal,
                task.failed_attempts,
                if used { "; concept was used" } else { "" }
            ),
        );
        item.dependencies = task
            .concepts_used
            .iter()
            .filter(|c| c.as_str() != concept.as_str())
            .cloned()
            .collect();
        store.insert(item);
        gaps.push(concept.clone());
    }
    gaps.sort();
    gaps
}

/// What to learn next: open questions first (least-visited, then
/// least-failed), then untouched unknowns. Verified, Generalized,
/// and Rejected items never surface — the loop studies, not replays.
pub fn prioritize(store: &mut KnowledgeStore) -> Option<String> {
    let mut open: Vec<&KnowledgeItem> = store
        .items
        .values()
        .filter(|i| {
            !i.dormant
                && !i.is_expired()
                && matches!(i.state, KnowledgeState::Question | KnowledgeState::Unknown)
        })
        .collect();
    open.sort_by(|a, b| {
        (a.state as u8)
            .cmp(&(b.state as u8))
            .then(a.visits.cmp(&b.visits))
            .then(a.failures.cmp(&b.failures))
            .then(a.concept.cmp(&b.concept))
    });
    let next = open.first()?.concept.clone();
    if let Some(item) = store.items.get_mut(&next) {
        item.visits += 1;
        store.persist();
    }
    Some(next)
}

/// One idle-loop step, for logs and tests.
#[derive(Debug)]
pub struct DreamOutcome {
    pub concept: String,
    pub from: KnowledgeState,
    pub to: KnowledgeState,
    pub note: String,
}

fn prov(source: String) -> Provenance {
    Provenance {
        source,
        experiment: None,
        verification: None,
        rejection: None,
        tier: VerificationTier::Sourced,
    }
}

/// The idle loop: while budget remains and open questions exist,
/// form a hypothesis, investigate it (web lookup when `research` is
/// on), record honest evidence, and promote only what verifies.
/// Afterwards, seed the verified concept's unstored neighbors as
/// `Unknown` so the next idle pass has somewhere to go, then **sleep**:
/// consolidate what was verified into patterns (compressing members
/// into dormancy) and prime what will never be used. Deterministic
/// and bounded: at most `budget` investigations, then it sleeps.
pub async fn dream_loop(project_dir: &Path, budget: u32, research: bool) -> Vec<DreamOutcome> {
    let mut store = KnowledgeStore::open(project_dir);
    let mut outcomes = Vec::new();
    let mut oracle = super::research::ResearchOracle::new(budget * 2);
    for _ in 0..budget {
        let Some(concept) = prioritize(&mut store) else {
            break;
        };
        let from = store
            .get(&concept)
            .map(|i| i.state)
            .unwrap_or(KnowledgeState::Unknown);
        if from == KnowledgeState::Unknown {
            let _ = store.transition(
                &concept,
                KnowledgeState::Question,
                prov(format!("surfaced by idle prioritization ({} visits)", 1)),
            );
        }
        let r = dream_one(&mut store, &mut oracle, &concept, research).await;
        outcomes.push(r);
    }
    // The sleep that follows the day's work: make the connections the
    // verified facts imply (compress each cluster into one pattern and
    // put its members to sleep), then prime what will never be used.
    // Side effects on the store; the returned outcomes stay exactly the
    // investigations, so budget and metrics describe the studying.
    store.consolidate();
    store.prime();
    outcomes
}

async fn dream_one(
    store: &mut KnowledgeStore,
    oracle: &mut super::research::ResearchOracle,
    concept: &str,
    research: bool,
) -> DreamOutcome {
    let from = store
        .get(concept)
        .map(|i| i.state)
        .unwrap_or(KnowledgeState::Unknown);
    // Every investigation starts as a hypothesis — a dream with a label.
    if store
        .transition(
            concept,
            KnowledgeState::Hypothesis,
            prov(format!("idle hypothesis: {:?} is learnable", concept)),
        )
        .is_err()
    {
        return DreamOutcome {
            concept: concept.to_string(),
            from,
            to: from,
            note: "not in a hypothesiable state; skipped".to_string(),
        };
    }
    if !research {
        return DreamOutcome {
            concept: concept.to_string(),
            from,
            to: KnowledgeState::Hypothesis,
            note: "research off — hypothesis held for a later pass".to_string(),
        };
    }
    let _ = store.transition(
        concept,
        KnowledgeState::Experiment,
        Provenance {
            source: format!("idle experiment on {:?}", concept),
            experiment: Some("lookup on verified sources".to_string()),
            verification: None,
            rejection: None,
            tier: VerificationTier::Sourced,
        },
    );
    match oracle.research_word(concept).await {
        Some(def) => {
            let _ = store.transition(
                concept,
                KnowledgeState::Evidence,
                Provenance {
                    source: format!("web lookup: {}", def.source_url),
                    experiment: Some("lookup on verified sources".to_string()),
                    verification: None,
                    rejection: None,
                    tier: VerificationTier::Sourced,
                },
            );
            // The oracle only queries verified sources, so a sourced
            // summary counts as verification evidence for word
            // knowledge. Compiler probes (harder evidence) are the
            // follow-up project; the firewall — no evidence, no
            // promotion — already holds either way.
            let url = def.source_url.clone();
            let summary = def.summary.clone();
            match store.transition(
                concept,
                KnowledgeState::Verified,
                Provenance {
                    source: format!("web lookup: {}", url),
                    experiment: Some("lookup on verified sources".to_string()),
                    verification: Some(format!("source_url={} summary={}", url, summary)),
                    rejection: None,
                    tier: VerificationTier::Sourced,
                },
            ) {
                Ok(()) => {
                    // Solved items seed their unstored neighbors so the
                    // graph grows along structure, not a syllabus.
                    let neighbors: Vec<String> = guess_neighbors(concept)
                        .into_iter()
                        .filter(|n| store.get(n).is_none())
                        .collect();
                    for n in &neighbors {
                        store.insert(KnowledgeItem::new(
                            n,
                            KnowledgeState::Unknown,
                            &format!("adjacent to verified {:?}", concept),
                        ));
                    }
                    DreamOutcome {
                        concept: concept.to_string(),
                        from,
                        to: KnowledgeState::Verified,
                        note: format!("verified via {} (+{} neighbors)", url, neighbors.len()),
                    }
                }
                Err(e) => DreamOutcome {
                    concept: concept.to_string(),
                    from,
                    to: KnowledgeState::Evidence,
                    note: format!("promotion refused: {}", e),
                },
            }
        }
        None => {
            store.record_failure(concept);
            DreamOutcome {
                concept: concept.to_string(),
                from,
                to: KnowledgeState::Hypothesis,
                note: "no verified source found — hypothesis held, deprioritized".to_string(),
            }
        }
    }
}

/// Structural adjacency without a model: split compound concepts on
/// separators and treat the parts as neighbors ("CancellationToken"
/// → "cancellation", "token" is noise, so instead we link the full
/// dependency seeds from gap extraction). v1 keeps this honest and
/// small: neighbors come from the store graph; this helper only fires
/// for concepts that arrived with dependency seeds.
fn guess_neighbors(concept: &str) -> Vec<String> {
    // Deliberately empty in v1: inventing neighbors from spelling
    // would be dreaming disguised as structure. Real adjacency comes
    // from `extract_gaps` dependency seeds and the store graph walked
    // by `neighbors()`. This hook exists so the rule stays explicit.
    let _ = concept;
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem_store() -> KnowledgeStore {
        let dir = std::env::temp_dir().join(format!(
            "gc-knowledge-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        let _ = std::fs::create_dir_all(&dir);
        KnowledgeStore::open(&dir)
    }

    fn task() -> CompletedTask {
        CompletedTask {
            goal: "build a scheduler".to_string(),
            concepts_used: vec!["asynctask".to_string(), "config".to_string()],
            unknowns_seen: vec!["cron".to_string()],
            failed_attempts: 1,
            related: vec!["retry-budget".to_string()],
        }
    }

    #[test]
    fn gaps_become_questions_with_dependency_seeds() {
        let mut store = mem_store();
        let gaps = extract_gaps(&task(), &mut store);
        assert!(gaps.contains(&"cron".to_string()), "{:?}", gaps);
        assert!(gaps.contains(&"retry-budget".to_string()), "{:?}", gaps);
        // Used-but-unstored concepts are gaps too — use is not proof.
        assert!(gaps.contains(&"asynctask".to_string()), "{:?}", gaps);
        let cron = store.get("cron").expect("stored");
        assert_eq!(cron.state, KnowledgeState::Question);
        assert!(cron.dependencies.contains(&"asynctask".to_string()));
    }

    #[test]
    fn solved_and_disproven_stay_put() {
        let mut store = mem_store();
        let mut done = KnowledgeItem::new("asynctask", KnowledgeState::Verified, "test");
        done.provenance.verification = Some("test".to_string());
        store.insert(done);
        store.insert(KnowledgeItem::new("cron", KnowledgeState::Rejected, "test"));
        let gaps = extract_gaps(&task(), &mut store);
        assert!(!gaps.contains(&"asynctask".to_string()), "{:?}", gaps);
        assert!(!gaps.contains(&"cron".to_string()), "{:?}", gaps);
        // Rejected is never re-questioned by prioritization either:
        // the open set holds the remaining questions — "cron" must
        // never surface no matter how many times we ask.
        for _ in 0..4 {
            let next = prioritize(&mut store).expect("open question");
            assert_ne!(next, "cron", "rejected concept resurfaced");
        }
    }

    #[test]
    fn dream_is_not_knowledge() {
        let mut store = mem_store();
        store.insert(KnowledgeItem::new(
            "flux",
            KnowledgeState::Hypothesis,
            "test",
        ));
        for (to, p) in [
            (KnowledgeState::Experiment, prov("probe".to_string())),
            (
                KnowledgeState::Evidence,
                Provenance {
                    source: "probe ran".to_string(),
                    experiment: Some("probe".to_string()),
                    verification: None,
                    rejection: None,
                    tier: VerificationTier::Sourced,
                },
            ),
        ] {
            store.transition("flux", to, p).expect("walk to evidence");
        }
        // Evidence with no verification must NOT promote.
        let err = store
            .transition(
                "flux",
                KnowledgeState::Verified,
                prov("wishful".to_string()),
            )
            .expect_err("firewall must hold");
        assert!(err.contains("DREAM != KNOWLEDGE"), "{}", err);
        assert_eq!(store.get("flux").unwrap().state, KnowledgeState::Evidence);
        // ...but real evidence promotes.
        store
            .transition(
                "flux",
                KnowledgeState::Verified,
                Provenance {
                    source: "probe".to_string(),
                    experiment: Some("probe".to_string()),
                    verification: Some("cargo test green".to_string()),
                    rejection: None,
                    tier: VerificationTier::Sourced,
                },
            )
            .expect("evidence promotes");
        assert_eq!(store.get("flux").unwrap().state, KnowledgeState::Verified);
    }

    #[test]
    fn illegal_jumps_rejected() {
        let mut store = mem_store();
        store.insert(KnowledgeItem::new("hop", KnowledgeState::Question, "test"));
        assert!(
            store
                .transition("hop", KnowledgeState::Verified, prov("skip".to_string()))
                .is_err()
        );
        assert!(
            store
                .transition("hop", KnowledgeState::Evidence, prov("skip".to_string()))
                .is_err()
        );
        assert_eq!(store.get("hop").unwrap().state, KnowledgeState::Question);
    }

    #[test]
    fn invalid_generalization_rejected_not_kept() {
        let mut store = mem_store();
        // One verified supporter is a dream, not a pattern.
        let mut a = KnowledgeItem::new("task-a", KnowledgeState::Verified, "test");
        a.provenance.verification = Some("test".to_string());
        store.insert(a);
        let err = store
            .generalize(
                "all-tasks-need-x",
                &["task-a".to_string(), "task-b".to_string()],
            )
            .expect_err("must reject");
        assert!(err.contains("1/2"), "{}", err);
        assert_eq!(
            store.get("all-tasks-need-x").unwrap().state,
            KnowledgeState::Rejected
        );
        // Two verified supporters generalize.
        let mut b = KnowledgeItem::new("task-b", KnowledgeState::Verified, "test");
        b.provenance.verification = Some("test".to_string());
        store.insert(b);
        store
            .generalize(
                "all-tasks-need-x2",
                &["task-a".to_string(), "task-b".to_string()],
            )
            .expect("two supporters");
        let g = store.get("all-tasks-need-x2").unwrap();
        assert_eq!(g.state, KnowledgeState::Generalized);
        assert_eq!(g.dependencies.len(), 2);
    }

    #[test]
    fn overlap_grades_kinship() {
        use super::KnowledgeStore as KS;
        assert_eq!(KS::signature_overlap(&[], &[]), 0.0);
        let (a, b): (Vec<String>, Vec<String>) = (vec![], vec!["x".to_string()]);
        assert_eq!(KS::signature_overlap(&a, &b), 0.0);
        let (a, b) = (vec!["x".to_string()], vec!["x".to_string()]);
        assert_eq!(KS::signature_overlap(&a, &b), 1.0);
        let (a, b) = (
            vec![
                "x".to_string(),
                "y".to_string(),
                "z".to_string(),
                "w".to_string(),
            ],
            vec![
                "x".to_string(),
                "y".to_string(),
                "z".to_string(),
                "v".to_string(),
            ],
        );
        assert!((KS::signature_overlap(&a, &b) - 0.6).abs() < 1e-9);
    }

    #[test]
    fn consolidation_compresses_members_into_pattern() {
        let mut store = mem_store();
        for name in ["alpha", "beta"] {
            let mut item = KnowledgeItem::new(name, KnowledgeState::Verified, "test");
            item.provenance.verification = Some("probe green".to_string());
            item.dependencies = vec![
                "d1".to_string(),
                "d2".to_string(),
                "d3".to_string(),
                "d4".to_string(),
            ];
            store.insert(item);
        }
        let created = store.consolidate();
        assert_eq!(created, vec!["cluster:alpha+beta".to_string()]);
        let pattern = &created[0];
        // The pattern is awake; the members sleep beneath it.
        let mut active: Vec<&str> = store
            .all()
            .iter()
            .filter(|i| !i.dormant)
            .map(|i| i.concept.as_str())
            .collect();
        active.sort();
        assert_eq!(active, vec![pattern.as_str()]);
        for member in ["alpha", "beta"] {
            let item = store.get(member).expect("member kept");
            assert!(item.dormant, "{} must sleep", member);
            assert_eq!(item.absorbed_by.as_deref(), Some(pattern.as_str()));
            // Compression keeps the evidence: provenance survives sleep.
            assert!(item.provenance.verification.is_some());
        }
    }

    #[test]
    fn prime_forgets_rejected_and_worn_but_not_live_knowledge() {
        let mut store = mem_store();
        store.insert(KnowledgeItem::new("dud", KnowledgeState::Rejected, "test"));
        let mut worn = KnowledgeItem::new("chased", KnowledgeState::Hypothesis, "test");
        worn.failures = 3;
        store.insert(worn);
        let mut young = KnowledgeItem::new("fresh", KnowledgeState::Hypothesis, "test");
        young.failures = 1;
        store.insert(young);
        let mut live = KnowledgeItem::new("fact", KnowledgeState::Verified, "test");
        live.provenance.verification = Some("probe green".to_string());
        store.insert(live);

        let primed = store.prime();
        assert_eq!(primed, vec!["chased".to_string(), "dud".to_string()]);
        assert!(store.get("dud").unwrap().dormant);
        assert!(store.get("chased").unwrap().dormant);
        assert!(
            !store.get("fresh").unwrap().dormant,
            "one failure is not worn out"
        );
        assert!(
            !store.get("fact").unwrap().dormant,
            "verified memory stays live"
        );
        // Second pass converges: nothing left to prime.
        assert!(store.prime().is_empty());
    }

    #[test]
    fn consolidation_discovers_clusters() {
        let mut store = mem_store();
        // Two verified items with identical signatures: kin.
        for name in ["task-a", "task-b"] {
            let mut item = KnowledgeItem::new(name, KnowledgeState::Verified, "test");
            item.provenance.verification = Some("test".to_string());
            item.dependencies = vec![
                "d1".to_string(),
                "d2".to_string(),
                "d3".to_string(),
                "d4".to_string(),
            ];
            store.insert(item);
        }
        // Near-miss at 0.6 overlap: similar is not kin.
        let mut near = KnowledgeItem::new("task-d", KnowledgeState::Verified, "test");
        near.provenance.verification = Some("test".to_string());
        near.dependencies = vec![
            "d1".to_string(),
            "d2".to_string(),
            "d3".to_string(),
            "e1".to_string(),
        ];
        store.insert(near);
        // One verified loner: nothing shared, never clustered.
        let mut lone = KnowledgeItem::new("task-c", KnowledgeState::Verified, "test");
        lone.provenance.verification = Some("test".to_string());
        lone.dependencies = vec!["elsewhere".to_string()];
        store.insert(lone);
        let created = store.consolidate();
        assert_eq!(created.len(), 1, "{:?}", created);
        assert_eq!(created[0], "cluster:task-a+task-b");
        let g = store.get(&created[0]).expect("pattern stored");
        assert_eq!(g.state, KnowledgeState::Generalized);
        assert_eq!(g.dependencies.len(), 2);
        // Second pass finds nothing new — consolidation converges.
        assert!(store.consolidate().is_empty());
    }

    #[test]
    fn tiers_distinguish_evidence_strength() {
        let old: KnowledgeItem = serde_json::from_str(
            r#"{"concept":"x","state":"Verified","dependencies":[],
                "provenance":{"source":"s","experiment":null,"verification":"v","rejection":null},
                "visits":0,"failures":0}"#,
        )
        .expect("legacy line loads");
        assert_eq!(old.provenance.tier, VerificationTier::Sourced);
    }

    #[test]
    fn generalization_takes_weakest_supporter() {
        let mut store = mem_store();
        let mut a = KnowledgeItem::new("ga", KnowledgeState::Verified, "test");
        a.provenance.verification = Some("probe green".to_string());
        a.provenance.tier = VerificationTier::Demonstrated;
        a.dependencies = vec!["d".to_string()];
        store.insert(a);
        let mut b = KnowledgeItem::new("gb", KnowledgeState::Verified, "test");
        b.provenance.verification = Some("source says".to_string());
        b.provenance.tier = VerificationTier::Sourced;
        b.dependencies = vec!["d".to_string()];
        store.insert(b);
        store
            .generalize("gp", &["ga".to_string(), "gb".to_string()])
            .expect("two supporters");
        // Sourced claims don't launder into demonstrated patterns.
        assert_eq!(
            store.get("gp").unwrap().provenance.tier,
            VerificationTier::Sourced
        );
    }

    #[test]
    fn neighbors_walk_the_graph() {
        let mut store = mem_store();
        let mut t = KnowledgeItem::new("task", KnowledgeState::Verified, "test");
        t.dependencies = vec!["cancellation".to_string()];
        store.insert(t);
        let mut c = KnowledgeItem::new("cancellation", KnowledgeState::Question, "test");
        c.dependencies = vec!["task".to_string()];
        store.insert(c);
        let n = store.neighbors("task");
        assert!(n.contains(&"cancellation".to_string()), "{:?}", n);
    }
}
