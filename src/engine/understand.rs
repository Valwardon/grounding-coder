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

/// Kinds the bot can actually define (structs, pages, configs,
/// components, machines, functions). Other kind words (`apk`, `repo`,
/// `test`) name targets and artifacts — they never trigger creation.
const DEFINABLE: &[&str] = &[
    "page",
    "struct",
    "config",
    "component",
    "statemachine",
    "function",
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

/// One thing the parser could not place, with its best guesses.
/// Curiosity is bounded and transparent: nearest lexicon neighbors by
/// edit distance, any verified history containing the word, and whether
/// the prose holds multiple requests. Never a guess acted on — only
/// questions the user can answer.
#[derive(Debug, Clone)]
pub struct Curiosity {
    pub unknown: String,
    pub suggestions: Vec<String>,
    pub history_note: Option<String>,
}

/// Split markers for multi-request prose ("do X and also Y").
const CLAUSE_SPLITS: &[&str] = &[" and also ", " and then ", " then "];

/// Lexicon surface for neighbor search: verbs and kind words alike.
fn lexicon_words() -> Vec<(&'static str, &'static str)> {
    let mut out: Vec<(&str, &str)> = Vec::new();
    for (v, _) in VERBS {
        out.push((*v, "verb"));
    }
    for (k, _) in KINDS {
        out.push((*k, "kind"));
    }
    out
}

/// Nearest lexicon words within distance 4, ordered by (distance,
/// word). Deterministic; empty when nothing is close.
fn neighbors(word: &str) -> Vec<String> {
    let mut scored: Vec<(usize, String, String)> = Vec::new();
    for (lex, role) in lexicon_words() {
        let d = edit_distance(word, lex);
        if d > 0 && d <= 4 {
            scored.push((d, lex.to_string(), role.to_string()));
        }
    }
    scored.sort();
    scored
        .into_iter()
        .take(3)
        .map(|(_, w, r)| format!("{} ({})", w, r))
        .collect()
}

/// Ask about what the parse could not place. At most 3 curiosities,
/// longest-unknown-first is wrong — order follows prose order so the
/// questions read naturally. Pure function of prose + project history.
pub fn curiosities(prose: &str, project_dir: Option<&std::path::Path>) -> Vec<Curiosity> {
    let ws = words(prose);
    let mut out = Vec::new();
    for w in &ws {
        if out.len() >= 3 {
            break;
        }
        // Words the parser already accepted — exactly or through
        // typo tolerance — are understood, never questioned.
        if w.len() <= 2
            || STOPWORDS.contains(&w.as_str())
            || VERBS.iter().any(|(v, _)| v == w)
            || KINDS.iter().any(|(k, _)| k == w)
            || fuzzy_verb(w).is_some()
            || w.chars().all(|c| c.is_ascii_digit())
        {
            continue;
        }
        // Words already consumed as slots (quotes, names, numbers) are
        // understood — curiosity is only for the leftovers.
        let quoted_txt = quoted(prose).join(" ").to_lowercase();
        let caps: Vec<String> = capitals(prose)
            .into_iter()
            .map(|s| s.to_lowercase())
            .collect();
        if quoted_txt.contains(w) || caps.iter().any(|c| c == w) {
            continue;
        }
        let history_note = project_dir.and_then(|p| {
            load_history(p)
                .into_iter()
                .find(|r| words(&r.prose).iter().any(|hw| hw == w))
                .map(|r| format!("last time {:?} meant: {}", r.prose, r.goal))
        });
        let suggestions = neighbors(w);
        out.push(Curiosity {
            unknown: w.clone(),
            suggestions,
            history_note,
        });
    }
    out
}

/// Detect multiple requests hiding in one breath ("do X and also Y").
/// Returns the clause count (1 = single). Deterministic substring scan.
pub fn clause_count(prose: &str) -> usize {
    let lower = prose.to_lowercase();
    let mut count = 1;
    let mut rest = lower.as_str();
    loop {
        let mut found = None;
        for split in CLAUSE_SPLITS {
            if let Some(pos) = rest.find(split)
                && found.map(|(_, p)| pos < p).unwrap_or(true)
            {
                found = Some((*split, pos));
            }
        }
        match found {
            Some((split, pos)) => {
                count += 1;
                rest = &rest[pos + split.len()..];
            }
            None => break,
        }
    }
    count
}

/// Stopwords ignored by history similarity (small, documented).
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "to", "for", "with", "and", "of", "in", "on", "please", "my", "me", "it",
    "this", "that", "is", "are", "be", "also", "then",
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

/// Edit distance over chars. Used only to forgive typos in the small
/// verb lexicon — never for names, kinds, or literals, which pass
/// through verbatim (correcting those would invent meaning).
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != cb))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Nearest verb within typo tolerance (≤1 for short words, ≤2 above),
/// or None. Exact matches always win — this runs only when none fired.
fn fuzzy_verb(word: &str) -> Option<FrameAction> {
    let max_dist = if word.len() <= 4 { 1 } else { 2 };
    let mut best: Option<(usize, FrameAction)> = None;
    for (v, a) in VERBS {
        let d = edit_distance(word, v);
        if d <= max_dist && best.map(|(bd, _)| d < bd).unwrap_or(true) {
            best = Some((d, *a));
        }
    }
    best.map(|(_, a)| a)
}

/// Constraint keywords: performance/shape words that become intent
/// constraints instead of vanishing.
const CONSTRAINTS: &[(&str, &str)] = &[
    ("fast", "latency"),
    ("faster", "latency"),
    ("quick", "latency"),
    ("slow", "throughput"),
    ("small", "size"),
    ("tiny", "size"),
    ("secure", "security"),
    ("safe", "security"),
];

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

/// Noun adjacent to a kind word: "a counter page" / "the RiskBoard
/// component" name the referent right before the kind. Deterministic
/// positional slot-filling (articles skipped, first letter capitalized
/// for Rust convention) — visible in the output, never silent.
fn adjacent_name(words: &[String], kind: &str) -> Option<String> {
    let pos = words
        .iter()
        .position(|w| KINDS.iter().any(|(k, v)| *v == kind && k == w))?;
    let articles = [
        "a", "an", "the", "my", "this", "that", "some", "any", "each", "every",
    ];
    let mut i = pos;
    while i > 0 {
        i -= 1;
        // Articles, verbs, and kind words are grammar, never names —
        // "create a struct" must not name the struct "Create".
        if articles.contains(&words[i].as_str())
            || VERBS.iter().any(|(v, _)| v == &words[i])
            || KINDS.iter().any(|(k, _)| k == &words[i])
        {
            continue;
        }
        let mut cs = words[i].chars();
        let head: String = cs
            .next()
            .map(|c| c.to_uppercase().collect::<String>())
            .unwrap_or_default();
        let name = format!("{}{}", head, cs.as_str());
        if !name.is_empty() {
            return Some(name);
        }
    }
    None
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
    // Typo tolerance on verbs only: "buld" still means build. Names,
    // kinds, and literals are never corrected — that would invent.
    let fuzzy = if verb.is_none() {
        ws.iter().find_map(|w| fuzzy_verb(w))
    } else {
        None
    };
    let verb = verb.or(fuzzy);
    // Specificity wins: bare "make" with a kind + name in the same
    // breath ("make a counter page") is creation, not compilation.
    // "make the thing work faster" (no kind, no name) stays a build.
    let kind = ws
        .iter()
        .find_map(|w| KINDS.iter().find(|(k, _)| k == w).map(|(_, k)| *k));
    // Constraint words ride along as intent constraints.
    let constraints: Vec<String> = CONSTRAINTS
        .iter()
        .filter(|(w, _)| ws.iter().any(|x| x == *w))
        .map(|(_, c)| c.to_string())
        .collect();

    let similar = project_dir.and_then(|p| find_similar(p, prose));

    let mut frame = String::from("unknown");
    let mut filled = 0u32;
    let mut total = 1u32; // the verb slot always counts
    let mut intent = empty_intent(prose.trim().to_string());
    intent.constraints = constraints;
    // Content gate: a create frame only completes locally when it
    // carries buildable content (page sections from quotes). Bare
    // "create a Counter struct" parses fine but has nothing to build
    // with — confidence caps below the act threshold so the model (or
    // a clarifying question) takes it instead of a guaranteed block.
    let mut content_ok = true;

    // Name resolution, most explicit first: `titled`/`named`/`called`
    // X patterns, then Capitalized words, then quoted literals, then
    // the noun adjacent to the kind word ("a counter page" → Counter).
    let titled: Option<String> = {
        let mut out = None;
        let wslice: Vec<&str> = ws.iter().map(|s| s.as_str()).collect();
        for (i, w) in wslice.iter().enumerate() {
            if (*w == "titled" || *w == "named" || *w == "called") && i + 1 < wslice.len() {
                let raw = wslice[i + 1];
                if !KINDS.iter().any(|(k, _)| *k == raw) {
                    let mut cs = raw.chars();
                    let head: String = cs
                        .next()
                        .map(|c| c.to_uppercase().collect::<String>())
                        .unwrap_or_default();
                    out = Some(format!("{}{}", head, cs.as_str()));
                    break;
                }
            }
        }
        out
    };
    let resolved_name: Option<String> = titled
        .or_else(|| capitals(prose).first().cloned())
        .or_else(|| quoted(prose).first().cloned())
        .or_else(|| kind.and_then(|k| adjacent_name(&ws, k)));
    // Any build verb + kind + name reads as creation ("build a counter
    // page", "make a token struct"). Bare builds without kind+name
    // ("build the apk" has a target, not a name) stay builds.
    let verb = if verb == Some(FrameAction::Build)
        && kind.is_some_and(|k| DEFINABLE.contains(&k))
        && resolved_name.is_some()
    {
        Some(FrameAction::Create)
    } else {
        verb
    };
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
            // One name resolution for the whole parse (titled, then
            // Capitalized, then quoted, then adjacent) — two competing
            // resolutions once named different names for the same slot.
            let name = resolved_name.clone().unwrap_or_default();
            match (kind, name.is_empty()) {
                (Some(k), false) if DEFINABLE.contains(&k) => {
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
                    // Content gate: pages need quoted sections, other
                    // definables need fields the prose never carries —
                    // without buildable content the local path would
                    // walk into a guaranteed block.
                    if (k == "page" && def.sections.is_empty()) || (k != "page" && k != "function")
                    {
                        content_ok = false;
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

    let mut confidence = (filled as f64 / total.max(1) as f64).min(1.0);
    if !content_ok {
        // Parsable but not locally buildable — route to the model
        // (or a question), never into a guaranteed block.
        confidence = confidence.min(0.49);
    }
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
    fn typo_verbs_still_parse() {
        let u = understand("buld the android apk", None);
        assert_eq!(u.frame, "build".to_string());
        assert_eq!(u.intent.platform, "android".to_string());
        // Names and literals are never typo-corrected.
        let u = understand("create a page called Coutner", None);
        let def = u.intent.define.into_iter().flatten().next().unwrap();
        assert_eq!(def.name, "Coutner");
    }

    #[test]
    fn make_with_kind_and_name_is_creation() {
        let u = understand("make a counter page", None);
        assert_eq!(u.frame, "create".to_string());
        // Bare make without kind/name stays a build.
        let u = understand("make the thing work faster", None);
        assert_eq!(u.frame, "build".to_string());
        assert!(u.intent.constraints.contains(&"latency".to_string()));
    }

    #[test]
    fn contentless_creates_cap_below_threshold() {
        // Parsable but unbuildable locally: routes to model/questions.
        let u = understand("create a Counter struct", None);
        assert_eq!(u.frame, "create".to_string());
        assert!(u.confidence < 0.75, "got {}", u.confidence);
        // Quoted sections complete the page locally.
        let u = understand("create a page called P with \"hello world\"", None);
        assert!(u.confidence >= 0.75, "got {}", u.confidence);
    }

    #[test]
    fn curiosity_suggests_nearest_lexicon() {
        let cs = curiosities("please flibber the widget", None);
        assert!(!cs.is_empty());
        assert_eq!(cs[0].unknown, "flibber");
        // Nothing within distance 4 of "flibber" in the lexicon.
        assert!(cs[0].suggestions.is_empty());
        let cs = curiosities("buld it now", None);
        let all: Vec<&str> = cs.iter().map(|c| c.unknown.as_str()).collect();
        // "buld" is consumed by fuzzy verbs, so curiosity skips it;
        // "now" is unknown but has no close neighbor either.
        assert!(!all.contains(&"buld"));
    }

    #[test]
    fn curiosity_recalls_history() {
        let dir = std::env::temp_dir().join(format!(
            "gc-und-cur-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut intent = empty_intent("make a counter page".to_string());
        intent.define = vec![Some(empty_define("Counter".to_string(), "page"))];
        save_history(&dir, "please make me a counter page now", &intent);
        let cs = curiosities("counter thing", Some(dir.as_path()));
        assert!(
            cs.iter().any(|c| c.history_note.is_some()),
            "expected a history note, got {:?}",
            cs
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clause_count_splits_requests() {
        assert_eq!(clause_count("do this"), 1);
        assert_eq!(clause_count("build the apk and also fix the bug"), 2);
        assert_eq!(clause_count("make a page then make it blue"), 2);
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
