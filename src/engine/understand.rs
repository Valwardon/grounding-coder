//! Understanding without an LLM: lexical database + grammar frames +
//! history retrieval.
//!
//! No model reads the prose. A hand-built verb/noun lexicon normalizes
//! words, a tiny grammar extracts an action frame with slots, and a
//! per-project history of verified translations offers precedent by
//! lexical overlap. Output is a StructuredIntent plus a confidence that
//! says exactly which slots were quoted (solid) versus inferred (soft).
//! Below threshold the caller must ask, not act — the number is on the
//! output for exactly that decision.

use super::tasks::{IntentAction, IntentDefinition, SectionDef, StructuredIntent};
/// Normalized actions the frame grammar can express.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameAction {
    Build,
    Create,
    Fix,
    Publish,
    Verify,
    Unknown,
}

/// Verb lexicon: surface forms → normalized action. Hand-curated and
/// auditable; adding a row is a product decision, never learning.
const VERBS: &[(&str, FrameAction)] = &[
    ("build", FrameAction::Build),
    ("make", FrameAction::Build),
    ("compile", FrameAction::Build),
    ("assemble", FrameAction::Build),
    ("create", FrameAction::Create),
    ("add", FrameAction::Create),
    ("new", FrameAction::Create),
    ("generate", FrameAction::Create),
    ("write", FrameAction::Create),
    ("fix", FrameAction::Fix),
    ("repair", FrameAction::Fix),
    ("mend", FrameAction::Fix),
    ("correct", FrameAction::Fix),
    ("publish", FrameAction::Publish),
    ("upload", FrameAction::Publish),
    ("release", FrameAction::Publish),
    ("share", FrameAction::Publish),
    ("test", FrameAction::Verify),
    ("check", FrameAction::Verify),
    ("verify", FrameAction::Verify),
    ("run", FrameAction::Verify),
];

/// Kind lexicon: surface forms → definition kind.
const KINDS: &[(&str, &str)] = &[
    ("page", "page"),
    ("screen", "page"),
    ("struct", "struct"),
    ("class", "struct"),
    ("config", "config"),
    ("settings", "config"),
    ("component", "component"),
    ("widget", "component"),
    ("statemachine", "statemachine"),
    ("machine", "statemachine"),
    ("function", "function"),
    ("fn", "function"),
    ("test", "test"),
    ("repo", "repo"),
    ("repository", "repo"),
    ("apk", "apk"),
];

/// Stopwords ignored by history similarity (small, documented).
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "to", "for", "with", "and", "of", "in", "on", "please", "my", "me", "it",
    "this", "that", "is", "are", "be",
];

/// One verified translation remembered per project.
#[derive(Debug, Clone)]
pub struct HistoryRow {
    pub prose: String,
    pub goal: String,
    pub defines: Vec<(String, String)>,
}

/// The parse result: an intent plus how it was derived.
#[derive(Debug, Clone)]
pub struct Understood {
    pub intent: StructuredIntent,
    pub confidence: f64,
    pub frame: String,
    pub similar: Option<SimilarHit>,
}

/// Closest verified precedent, if any clears the bar.
#[derive(Debug, Clone)]
pub struct SimilarHit {
    pub score: f64,
    pub goal: String,
}

/// Lowercased alphanumeric tokens in order.
fn words(prose: &str) -> Vec<String> {
    prose
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// Double-quoted literals in order (the solid slots).
fn quoted(prose: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = prose;
    while let Some(start) = rest.find('"') {
        let after = &rest[start + 1..];
        if let Some(end) = after.find('"') {
            let lit = after[..end].trim();
            if !lit.is_empty() {
                out.push(lit.to_string());
            }
            rest = &after[end + 1..];
        } else {
            break;
        }
    }
    out
}

/// Capitalized words in original case (candidate proper names).
fn capitals(prose: &str) -> Vec<String> {
    prose
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| {
            let mut cs = w.chars();
            matches!(cs.next(), Some(c) if c.is_uppercase()) && !w.chars().all(|c| c.is_uppercase())
        })
        .map(|w| w.to_string())
        .collect()
}

fn history_path(project_dir: &std::path::Path) -> std::path::PathBuf {
    project_dir.join(".grounding").join("translations.jsonl")
}

/// Load verified translation history (missing file = no precedent).
pub fn load_history(project_dir: &std::path::Path) -> Vec<HistoryRow> {
    let Ok(content) = std::fs::read_to_string(history_path(project_dir)) else {
        return Vec::new();
    };
    content
        .lines()
        .filter_map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).ok()?;
            Some(HistoryRow {
                prose: v.get("prose")?.as_str()?.to_string(),
                goal: v.get("goal")?.as_str()?.to_string(),
                defines: v
                    .get("defines")?
                    .as_array()?
                    .iter()
                    .filter_map(|d| {
                        Some((
                            d.get("name")?.as_str()?.to_string(),
                            d.get("kind")?.as_str()?.to_string(),
                        ))
                    })
                    .collect(),
            })
        })
        .collect()
}

/// Record a verified translation (SUCCESS outcomes only — the caller
/// decides; history must never learn from failures or guesses).
/// Capped at 200 rows, oldest dropped.
pub fn save_history(project_dir: &std::path::Path, prose: &str, intent: &StructuredIntent) {
    let dir = project_dir.join(".grounding");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let mut rows = load_history(project_dir);
    rows.push(HistoryRow {
        prose: prose.to_string(),
        goal: intent.goal.clone(),
        defines: intent
            .define
            .iter()
            .flatten()
            .map(|d| (d.name.clone(), d.kind.clone()))
            .collect(),
    });
    if rows.len() > 200 {
        let drop = rows.len() - 200;
        rows.drain(..drop);
    }
    let mut out = String::new();
    for r in &rows {
        out.push_str(
            &serde_json::json!({
                "prose": r.prose,
                "goal": r.goal,
                "defines": r.defines.iter().map(|(n, k)| serde_json::json!({"name": n, "kind": k})).collect::<Vec<_>>(),
            })
            .to_string(),
        );
        out.push('\n');
    }
    let _ = std::fs::write(history_path(project_dir), out);
}

/// Lexical overlap of content tokens, 0.0–1.0. Deterministic.
pub fn similarity(a: &str, b: &str) -> f64 {
    let qa: Vec<String> = words(a)
        .into_iter()
        .filter(|w| !STOPWORDS.contains(&w.as_str()))
        .collect();
    if qa.is_empty() {
        return 0.0;
    }
    let qb: Vec<String> = words(b)
        .into_iter()
        .filter(|w| !STOPWORDS.contains(&w.as_str()))
        .collect();
    let hits = qa.iter().filter(|w| qb.contains(w)).count();
    hits as f64 / qa.len() as f64
}

/// Best precedent at or above 0.6, else None. Identical prose is
/// replay, not precedent, and never matches itself.
pub fn find_similar(project_dir: &std::path::Path, prose: &str) -> Option<SimilarHit> {
    let mut best_score = 0.0;
    let mut best_row: Option<HistoryRow> = None;
    for row in load_history(project_dir) {
        if row.prose.trim() == prose.trim() {
            continue;
        }
        let s = similarity(prose, &row.prose);
        if s >= 0.6 && s > best_score {
            best_score = s;
            best_row = Some(row);
        }
    }
    best_row.map(|r| SimilarHit {
        score: best_score,
        goal: r.goal,
    })
}

fn empty_intent(goal: String) -> StructuredIntent {
    StructuredIntent {
        platform: String::new(),
        architecture: String::new(),
        runtime: String::new(),
        capabilities: Vec::new(),
        domains: Vec::new(),
        constraints: Vec::new(),
        dependencies: Vec::new(),
        unknown_requirements: Vec::new(),
        goal,
        file: None,
        language: None,
        imports: Vec::new(),
        confidence: 0.0,
        actions: Vec::new(),
        define: Vec::new(),
        test: Vec::new(),
        references: Vec::new(),
        files: Vec::new(),
        replacements: Vec::new(),
    }
}

fn empty_define(name: String, kind: &str) -> IntentDefinition {
    use super::tasks::IntentDefinition;
    IntentDefinition {
        name,
        kind: kind.to_string(),
        references: Vec::new(),
        signature: None,
        cases: Vec::new(),
        fields: Vec::new(),
        methods: Vec::new(),
        states: Vec::new(),
        transitions: Vec::new(),
        title: None,
        sections: Vec::new(),
        footer: None,
    }
}

/// Parse prose into an intent with a confidence receipt. Pure:
/// same input, same output, no I/O except optional history lookup.
pub fn understand(prose: &str, project_dir: Option<&std::path::Path>) -> Understood {
    let ws = words(prose);
    let quotes = quoted(prose);
    let names = capitals(prose);
    let verb = ws
        .iter()
        .find_map(|w| VERBS.iter().find(|(v, _)| v == w).map(|(_, a)| *a));
    let kind = ws
        .iter()
        .find_map(|w| KINDS.iter().find(|(k, _)| k == w).map(|(_, k)| *k));

    let similar = project_dir.and_then(|p| find_similar(p, prose));

    let mut frame = String::from("unknown");
    let mut filled = 0u32;
    let mut total = 1u32; // the verb slot always counts
    let mut intent = empty_intent(prose.trim().to_string());

    match verb {
        None => {
            intent.unknown_requirements = vec![
                "no action verb recognized — state the verb (build/create/fix/publish/test)"
                    .to_string(),
            ];
        }
        Some(FrameAction::Build) => {
            frame = "build".to_string();
            total = 2; // verb + target
            let target = if ws.iter().any(|w| w == "apk" || w == "android") {
                filled += 1;
                "android-apk"
            } else if ws.iter().any(|w| w == "app" || w == "binary") {
                filled += 1;
                "native"
            } else {
                "native"
            };
            filled += 1; // verb present
            intent.platform = if target == "android-apk" {
                "android".to_string()
            } else {
                "desktop".to_string()
            };
            intent.actions = vec![IntentAction::Action {
                action: format!("build {}", target.replace('-', " ")),
                params: vec![target.to_string()],
                references: Vec::new(),
            }];
        }
        Some(FrameAction::Create) => {
            frame = "create".to_string();
            total = 3; // verb + kind + name
            filled += 1;
            let name = names
                .first()
                .cloned()
                .or_else(|| quotes.first().cloned())
                .unwrap_or_default();
            match (kind, name.is_empty()) {
                (Some(k), false) => {
                    filled += 2;
                    let mut def = empty_define(name.clone(), k);
                    if k == "page" {
                        def.title = Some(name.clone());
                        def.sections = quotes
                            .iter()
                            .skip(if quotes.first().map(|q| q == &name).unwrap_or(false) {
                                1
                            } else {
                                0
                            })
                            .take(8)
                            .map(|q| {
                                let heading: String =
                                    q.split_whitespace().take(4).collect::<Vec<_>>().join(" ");
                                SectionDef {
                                    heading,
                                    body: q.clone(),
                                }
                            })
                            .collect();
                    }
                    intent.define = vec![Some(def)];
                    intent.file = Some("src/lib.rs".to_string());
                }
                _ => {
                    intent.unknown_requirements = vec![
                        "create needs a kind (page/struct/config/component/statemachine/function) and a Capitalized name".to_string(),
                    ];
                }
            }
        }
        Some(FrameAction::Fix) => {
            frame = "fix".to_string();
            total = 1;
            filled += 1;
            // Goal-only intents run the verify+repair loop as-is.
        }
        Some(FrameAction::Publish) => {
            frame = "publish".to_string();
            total = 2; // verb + repo
            filled += 1;
            let repo = names.first().cloned().or_else(|| quotes.first().cloned());
            let mut text = format!(
                "publish github repository {}",
                repo.clone().unwrap_or_default()
            );
            if let Some(r) = &repo {
                filled += 1;
                text = format!("publish github repository repo:{}", r.to_lowercase());
            } else {
                intent.unknown_requirements = vec!["publish needs a repository name".to_string()];
            }
            if ws.iter().any(|w| w == "release" || w == "apk") {
                text.push_str(" release upload apk tag v0.1.0");
            }
            intent.goal = text.clone();
            intent.actions = vec![IntentAction::Action {
                action: text,
                params: Vec::new(),
                references: Vec::new(),
            }];
        }
        Some(FrameAction::Verify) => {
            frame = "verify".to_string();
            total = 1;
            filled += 1;
        }
        Some(FrameAction::Unknown) => {
            intent.unknown_requirements =
                vec!["action not mappable to build/create/fix/publish/verify".to_string()];
        }
    }

    let confidence = (filled as f64 / total.max(1) as f64).min(1.0);
    intent.confidence = confidence;
    Understood {
        intent,
        confidence,
        frame,
        similar,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verbs_normalize() {
        assert_eq!(
            understand("please compile the thing", None).frame,
            "build".to_string()
        );
        assert_eq!(understand("repair it", None).frame, "fix".to_string());
    }

    #[test]
    fn quoted_literals_are_solid_slots() {
        let u = understand("create a page called \"Hello\" with \"say hi\"", None);
        assert_eq!(u.frame, "create".to_string());
        assert!(u.confidence >= 0.99, "got {}", u.confidence);
        let def = u.intent.define.into_iter().flatten().next().unwrap();
        assert_eq!(def.name, "Hello");
        assert_eq!(def.title, Some("Hello".to_string()));
        assert_eq!(def.sections.len(), 1);
        assert_eq!(def.sections[0].body, "say hi".to_string());
    }

    #[test]
    fn missing_name_blocks_with_receipt() {
        let u = understand("create a struct", None);
        assert!(u.confidence < 0.75, "got {}", u.confidence);
        assert!(!u.intent.unknown_requirements.is_empty());
    }

    #[test]
    fn build_apk_maps_target() {
        let u = understand("build the android apk", None);
        assert_eq!(u.intent.platform, "android".to_string());
        match &u.intent.actions[..] {
            [IntentAction::Action { params, .. }] => assert_eq!(params, &["android-apk"]),
            other => panic!("unexpected actions: {:?}", other),
        }
    }

    #[test]
    fn history_match_and_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "gc-und-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut intent = empty_intent("make a counter page".to_string());
        intent.define = vec![Some(empty_define("Counter".to_string(), "page"))];
        save_history(&dir, "please make me a counter page now", &intent);
        let hit = find_similar(&dir, "make a counter page please").expect("should match");
        assert!(hit.score >= 0.6, "score {}", hit.score);
        assert!(find_similar(&dir, "please make me a counter page now").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn similarity_is_deterministic() {
        let a = similarity("build the apk now", "build apk please");
        let b = similarity("build the apk now", "build apk please");
        assert_eq!(a, b);
        assert!(a > 0.5);
        assert_eq!(similarity("", "x"), 0.0);
    }
}
