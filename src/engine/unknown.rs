//! Unknown Resolution: investigate, don't just refuse.
//!
//! Sits between understanding and execution. An unknown is a state
//! to investigate — research it until resolved, clarified, or
//! provably exhausted. BLOCKED is terminal output after a real
//! attempt, never the first response to uncertainty.
//!
//! ```text
//! ENCOUNTER → INVESTIGATE → PLAN → RESOLVE/VERIFY → DECIDE
//!                                        proceed | continue | blocked
//! ```
//!
//! Every investigation keeps a record (question, evidence for and
//! against, attempted routes, what's still open). Attempted routes
//! are load-bearing: without them the system re-searches dead ends
//! forever instead of recognizing exhaustion.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// What kind of unknown this is — the kind decides the routes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnknownKind {
    /// Code-shaped (CamelCase, snake_case, dotted, ::): search code,
    /// docs, dependencies first, web second.
    SymbolOrApi,
    /// Ordinary word/claim: web sources, corroborate.
    FactualClaim,
    /// Vague or contradictory prose: only the user can resolve.
    Ambiguous,
    /// Nothing to investigate (vacuous/destructive input).
    Unknowable,
}

/// One piece of evidence, with its reliability stated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub source: String,
    pub content: String,
    pub reliable: bool,
}

/// One attempted route: what was tried, where, what came back.
/// This record is what stops endless re-search loops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchAttempt {
    pub method: String,
    pub query: String,
    pub found: bool,
    pub reliable: bool,
    pub remaining: String,
}

/// Lifecycle of one unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnknownStatus {
    Open,
    Investigating,
    AwaitingUser,
    Resolved,
    Exhausted,
}

/// The investigation record. Serializable so trails persist across
/// turns and the system never re-walks a dead end.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnknownRecord {
    pub question: String,
    pub kind: UnknownKind,
    pub known_facts: Vec<Evidence>,
    pub missing_facts: Vec<String>,
    pub hypotheses: Vec<String>,
    pub attempted_routes: Vec<ResearchAttempt>,
    pub unresolved: Vec<String>,
    pub followups: Vec<String>,
    pub status: UnknownStatus,
}

impl UnknownRecord {
    pub fn open(question: &str) -> Self {
        UnknownRecord {
            question: question.to_string(),
            kind: classify_unknown(question),
            known_facts: Vec::new(),
            missing_facts: vec![question.to_string()],
            hypotheses: Vec::new(),
            attempted_routes: Vec::new(),
            unresolved: vec![question.to_string()],
            followups: Vec::new(),
            status: UnknownStatus::Open,
        }
    }

    /// Routes still untried for this record's kind. Attempted
    /// (method, query) pairs never repeat — the loop killer.
    pub fn open_routes(&self) -> Vec<(&'static str, String)> {
        let tried: Vec<(&str, &str)> = self
            .attempted_routes
            .iter()
            .map(|a| (a.method.as_str(), a.query.as_str()))
            .collect();
        let mut out = Vec::new();
        let queries = default_queries(&self.question);
        let methods: &[&str] = match self.kind {
            UnknownKind::SymbolOrApi => &["codebase", "docs", "web"],
            UnknownKind::FactualClaim => &["web", "codebase"],
            UnknownKind::Ambiguous => &["user"],
            UnknownKind::Unknowable => &[],
        };
        for m in methods {
            for q in &queries {
                if !tried.contains(&(*m, q.as_str())) {
                    out.push((*m, q.clone()));
                }
            }
        }
        out
    }

    pub fn record_attempt(&mut self, attempt: ResearchAttempt) {
        if attempt.found && attempt.reliable {
            self.status = UnknownStatus::Resolved;
        }
        self.attempted_routes.push(attempt);
    }

    pub fn record_evidence(&mut self, ev: Evidence) {
        if ev.reliable {
            let content = ev.content.to_lowercase();
            self.missing_facts
                .retain(|m| !content.contains(&m.to_lowercase()));
            self.unresolved
                .retain(|u| !content.contains(&u.to_lowercase()));
            if self.missing_facts.is_empty() && self.unresolved.is_empty() {
                self.status = UnknownStatus::Resolved;
            }
        }
        self.known_facts.push(ev);
    }
}

/// Classify an unknown word by shape. Code-shaped tokens go to
/// code routes; ordinary words to web sources; nothing usable
/// stays unknowable.
pub fn classify_unknown(word: &str) -> UnknownKind {
    let w = word.trim();
    if w.len() <= 2 {
        return UnknownKind::Unknowable;
    }
    let code_shaped = w.contains("::")
        || w.contains('.')
        || w.contains('_')
        || w.contains('-')
        || (w.chars().next().is_some_and(|c| c.is_uppercase())
            && w.chars().any(|c| c.is_lowercase()));
    if code_shaped {
        UnknownKind::SymbolOrApi
    } else {
        UnknownKind::FactualClaim
    }
}

/// Default queries for an unknown: itself plus split parts
/// (`flux-capacitor` also tries `flux` and `capacitor`).
fn default_queries(question: &str) -> Vec<String> {
    let mut out = vec![question.to_string()];
    for part in question.split(['-', '_', ' ']) {
        let p = part.trim().to_string();
        if p.len() > 2 && !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

/// The decision after investigation. Proceed only on verified
/// resolution; Continue names the next route; Blocked carries the
/// structured honest-failure report (what was established, what
/// was tried, what stays open, what could unblock it).
#[derive(Debug, Clone)]
pub enum Decision {
    Proceed { evidence: Evidence },
    Continue { method: &'static str, query: String },
    Blocked { report: BlockReport },
}

#[derive(Debug, Clone)]
pub struct BlockReport {
    pub objective: String,
    pub established: Vec<String>,
    pub investigated: Vec<String>,
    pub unresolved: Vec<String>,
    pub next_step: String,
}

/// Decide: first verified evidence wins; else the next untried
/// route; else principled exhaustion — every applicable method
/// tried or ruled out, budget spent, user unavailable (or asked
/// already), uncertainty still blocking safe execution.
pub fn decide(record: &UnknownRecord, budget_left: u32, user_available: bool) -> Decision {
    if let Some(ev) = record.known_facts.iter().find(|e| e.reliable) {
        return Decision::Proceed {
            evidence: ev.clone(),
        };
    }
    let mut routes = record.open_routes();
    // Ambiguous unknowns route to the user — but only if one is
    // actually there; otherwise they exhaust immediately with the
    // reason stated.
    routes.retain(|(m, _)| *m != "user" || user_available);
    if budget_left > 0
        && let Some((method, query)) = routes.into_iter().next()
    {
        return Decision::Continue { method, query };
    }
    // Exhausted: say what was actually exhausted, never claim
    // universal impossibility.
    let investigated: Vec<String> = record
        .attempted_routes
        .iter()
        .map(|a| {
            format!(
                "{} '{}' → {}",
                a.method,
                a.query,
                if a.found { "miss" } else { "nothing" }
            )
        })
        .collect();
    let next_step = match record.kind {
        UnknownKind::Ambiguous => {
            "answer the clarification question with the missing detail".to_string()
        }
        UnknownKind::Unknowable => "rephrase with actionable content".to_string(),
        _ => format!(
            "provide an authoritative source for '{}' or confirm the term",
            record.question
        ),
    };
    Decision::Blocked {
        report: BlockReport {
            objective: format!("resolve '{}'", record.question),
            established: record
                .known_facts
                .iter()
                .map(|e| format!("{}: {}", e.source, snippet(&e.content)))
                .collect(),
            investigated,
            unresolved: if record.unresolved.is_empty() {
                vec![record.question.clone()]
            } else {
                record.unresolved.clone()
            },
            next_step,
        },
    }
}

fn snippet(s: &str) -> String {
    const N: usize = 120;
    if s.len() <= N {
        s.to_string()
    } else {
        format!("{}…", &s[..N])
    }
}

/// Follow-up curiosity for a resolved record: capitalized terms in
/// the verified finding that are neither the question (any case) nor
/// everyday words, appearing at least twice (single mentions are
/// usually sentence-start noise like "Colloquially"). At most 2.
/// Suggests, never asserts — the next investigation decides them.
pub fn followups(record: &UnknownRecord, ev: &Evidence) -> Vec<String> {
    const STOP: &[&str] = &[
        "the", "and", "with", "from", "that", "this", "home", "page", "counter", "settings",
    ];
    followups_from(&ev.content, std::slice::from_ref(&record.question), STOP)
}

/// Follow-up curiosity core: capitalized non-lexicon terms inside a
/// verified finding become bounded new questions (at most 2).
/// Candidates must repeat (single mentions are sentence-start
/// noise) and match case-insensitively against known terms.
/// Pure function of text.
pub fn followups_from(finding: &str, known: &[String], lexicon: &[&str]) -> Vec<String> {
    use std::collections::HashMap;
    let mut freq: HashMap<String, usize> = HashMap::new();
    let mut display: HashMap<String, String> = HashMap::new();
    for w in finding.split(|c: char| !c.is_alphanumeric()) {
        let mut cs = w.chars();
        if !matches!(cs.next(), Some(c) if c.is_uppercase()) || w.len() <= 3 {
            continue;
        }
        let key = w.to_lowercase();
        *freq.entry(key.clone()).or_insert(0) += 1;
        display.entry(key).or_insert_with(|| w.to_string());
    }
    let known_lc: Vec<String> = known.iter().map(|k| k.to_lowercase()).collect();
    let mut out = Vec::new();
    let mut keys: Vec<String> = freq.keys().cloned().collect();
    keys.sort();
    for key in keys {
        if out.len() >= 2 {
            break;
        }
        if freq[&key] < 2 || lexicon.contains(&key.as_str()) || known_lc.iter().any(|k| k == &key) {
            continue;
        }
        out.push(display[&key].clone());
    }
    out
}

/// Merge persisted trail rows (term → summary/source) back into
/// quick lookup. Unknown file or bad rows load as empty — a trail
/// is an aid, never load-bearing.
pub fn load_trail(path: &std::path::Path) -> HashMap<String, (String, String)> {
    let mut out = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return out;
    };
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if let (Some(t), Some(s)) = (
            v.get("term").and_then(|x| x.as_str()),
            v.get("summary").and_then(|x| x.as_str()),
        ) {
            let src = v
                .get("source")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            out.insert(t.to_string(), (s.to_string(), src));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typing_routes_code_and_prose_apart() {
        assert_eq!(classify_unknown("HashMap"), UnknownKind::SymbolOrApi);
        assert_eq!(classify_unknown("std::vec::Vec"), UnknownKind::SymbolOrApi);
        assert_eq!(classify_unknown("flux-capacitor"), UnknownKind::SymbolOrApi);
        assert_eq!(classify_unknown("capacitor"), UnknownKind::FactualClaim);
        assert_eq!(classify_unknown("hi"), UnknownKind::Unknowable);
    }

    #[test]
    fn routes_never_repeat() {
        let mut r = UnknownRecord::open("flux-capacitor");
        let first = r.open_routes();
        assert!(!first.is_empty());
        let (m, q) = first[0].clone();
        r.record_attempt(ResearchAttempt {
            method: m.to_string(),
            query: q.clone(),
            found: false,
            reliable: false,
            remaining: "still unknown".to_string(),
        });
        let second = r.open_routes();
        assert!(!second.contains(&(m, q)));
    }

    #[test]
    fn verified_evidence_proceeds() {
        let mut r = UnknownRecord::open("capacitor");
        r.record_evidence(Evidence {
            source: "wikipedia".to_string(),
            content: "A capacitor stores charge. capacitor".to_string(),
            reliable: true,
        });
        assert!(matches!(decide(&r, 5, true), Decision::Proceed { .. }));
        assert_eq!(r.status, UnknownStatus::Resolved);
    }

    #[test]
    fn exhaustion_reports_honestly() {
        let mut r = UnknownRecord::open("flux-capacitor");
        // Burn every route with misses.
        loop {
            let next = match decide(&r, 99, false) {
                Decision::Continue { method, query } => Some((method, query)),
                Decision::Blocked { report } => {
                    assert!(report.objective.contains("flux-capacitor"));
                    assert!(!report.investigated.is_empty());
                    assert!(!report.unresolved.is_empty());
                    assert!(!report.next_step.is_empty());
                    break;
                }
                Decision::Proceed { .. } => panic!("nothing verified"),
            };
            let (m, q) = next.unwrap();
            r.record_attempt(ResearchAttempt {
                method: m.to_string(),
                query: q.clone(),
                found: false,
                reliable: false,
                remaining: "miss".to_string(),
            });
        }
        assert!(matches!(decide(&r, 0, false), Decision::Blocked { .. }));
    }

    #[test]
    fn followups_need_repeats_not_noise() {
        // "Colloquially" appears once (sentence start) — dropped.
        // "Capacitor" repeats and isn't the question — kept.
        let f = followups_from(
            "Colloquially, a Capacitor may be called a cap. The Capacitor stores charge.",
            &["flux".to_string()],
            &[],
        );
        assert_eq!(f, vec!["Capacitor".to_string()]);
        // Question itself never returns, any case.
        let f = followups_from("Flux Flux flux", &["flux".to_string()], &[]);
        assert!(f.is_empty());
    }

    #[test]
    fn ambiguous_needs_a_user() {
        let r = UnknownRecord {
            kind: UnknownKind::Ambiguous,
            ..UnknownRecord::open("thing")
        };
        // No user around: exhausts immediately with the reason.
        assert!(matches!(decide(&r, 5, false), Decision::Blocked { .. }));
    }
}
