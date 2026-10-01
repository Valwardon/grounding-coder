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
    ("building", FrameAction::Build),
    ("built", FrameAction::Build),
    ("made", FrameAction::Build),
    ("makes", FrameAction::Build),
    ("making", FrameAction::Build),
    ("create", FrameAction::Create),
    ("add", FrameAction::Create),
    ("new", FrameAction::Create),
    ("generate", FrameAction::Create),
    ("write", FrameAction::Create),
    ("creating", FrameAction::Create),
    ("created", FrameAction::Create),
    ("creates", FrameAction::Create),
    ("adding", FrameAction::Create),
    ("added", FrameAction::Create),
    ("generating", FrameAction::Create),
    ("generated", FrameAction::Create),
    ("writing", FrameAction::Create),
    ("wrote", FrameAction::Create),
    ("written", FrameAction::Create),
    ("imagine", FrameAction::Create),
    ("render", FrameAction::Create),
    ("compose", FrameAction::Create),
    ("draw", FrameAction::Create),
    ("paint", FrameAction::Create),
    ("show", FrameAction::Create),
    ("fix", FrameAction::Fix),
    ("repair", FrameAction::Fix),
    ("mend", FrameAction::Fix),
    ("correct", FrameAction::Fix),
    ("fixing", FrameAction::Fix),
    ("fixed", FrameAction::Fix),
    ("fixes", FrameAction::Fix),
    ("repairing", FrameAction::Fix),
    ("repaired", FrameAction::Fix),
    ("publish", FrameAction::Publish),
    ("upload", FrameAction::Publish),
    ("release", FrameAction::Publish),
    ("share", FrameAction::Publish),
    ("publishing", FrameAction::Publish),
    ("published", FrameAction::Publish),
    ("uploading", FrameAction::Publish),
    ("uploaded", FrameAction::Publish),
    ("test", FrameAction::Verify),
    ("check", FrameAction::Verify),
    ("verify", FrameAction::Verify),
    ("run", FrameAction::Verify),
    ("testing", FrameAction::Verify),
    ("tested", FrameAction::Verify),
    ("tests", FrameAction::Verify),
    ("checking", FrameAction::Verify),
    ("checked", FrameAction::Verify),
    ("verifying", FrameAction::Verify),
    ("verified", FrameAction::Verify),
    ("running", FrameAction::Verify),
    ("runs", FrameAction::Verify),
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

/// Kind lexicon: surface forms → definition kind. Hand-curated noun
/// layer (the deterministic stand-in for a WordNet synset table):
/// everyday spellings map to the canonical kind the engine can build.
/// WordNet / Wikidata / Wikipedia stay external research sources —
/// nothing here claims to be those databases, only auditable rows.
const KINDS: &[(&str, &str)] = &[
    ("page", "page"),
    ("screen", "page"),
    ("webpage", "page"),
    ("website", "page"),
    ("homepage", "page"),
    ("site", "page"),
    ("struct", "struct"),
    ("class", "struct"),
    ("record", "struct"),
    ("model", "struct"),
    ("config", "config"),
    ("settings", "config"),
    ("options", "config"),
    ("preferences", "config"),
    ("component", "component"),
    ("widget", "component"),
    ("control", "component"),
    ("element", "component"),
    ("statemachine", "statemachine"),
    ("machine", "statemachine"),
    ("workflow", "statemachine"),
    ("fsm", "statemachine"),
    ("function", "function"),
    ("fn", "function"),
    ("func", "function"),
    ("method", "function"),
    ("routine", "function"),
    ("procedure", "function"),
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

/// What the system should do with a parse: act on it, ask about it,
/// or refuse it. Decided from the receipt, never from vibes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// Confidence clears the bar with a known frame: execute.
    Execute,
    /// Something specific is missing or contradictory: ask one question.
    Ask,
    /// Nothing actionable (vacuous input) or destructive ambiguity:
    /// refuse with the reason stated.
    Block,
}

/// Act threshold shared by every caller (CLI, chat, API).
pub const ACT_THRESHOLD: f64 = 0.75;

/// A frame schema: the slots that must fill for the frame to mean
/// anything, plus slots that merely help. Confidence is derived from
/// slot satisfaction — never tuned per phrase.
pub struct FrameSchema {
    pub name: &'static str,
    pub mandatory: &'static [&'static str],
    pub optional: &'static [&'static str],
}

/// Slot inventory per frame. "content" is optional everywhere it
/// appears: a fully-specified create without quoted sections still
/// parses at 1.0 and lets the content gate below route it.
pub const FRAME_SCHEMAS: &[FrameSchema] = &[
    FrameSchema {
        name: "build",
        mandatory: &["verb", "target"],
        optional: &[],
    },
    FrameSchema {
        name: "create",
        mandatory: &["verb", "kind", "name"],
        optional: &["content"],
    },
    FrameSchema {
        name: "fix",
        mandatory: &["verb"],
        optional: &[],
    },
    FrameSchema {
        name: "publish",
        mandatory: &["verb", "repo"],
        optional: &["release"],
    },
    FrameSchema {
        name: "verify",
        mandatory: &["verb"],
        optional: &[],
    },
    FrameSchema {
        name: "unknown",
        mandatory: &[],
        optional: &[],
    },
];

/// Slot satisfaction for a frame given filled slot names: mandatory
/// fraction plus a small bonus per optional slot, capped at 1.0.
/// Returns the score and the missing mandatory slots (the actual
/// question to ask). An empty mandatory set (unknown frame) scores
/// 0.0 — nothing to satisfy.
pub fn evaluate_frame(frame: &str, filled: &[&str]) -> (f64, Vec<&'static str>) {
    let schema = match FRAME_SCHEMAS.iter().find(|s| s.name == frame) {
        Some(s) => s,
        None => return (0.0, Vec::new()),
    };
    if schema.mandatory.is_empty() {
        return (0.0, Vec::new());
    }
    let mut missing = Vec::new();
    let mut mandatory_filled = 0u32;
    for slot in schema.mandatory {
        if filled.contains(slot) {
            mandatory_filled += 1;
        } else {
            missing.push(*slot);
        }
    }
    let mut optional_filled = 0u32;
    for slot in schema.optional {
        if filled.contains(slot) {
            optional_filled += 1;
        }
    }
    let score = (mandatory_filled as f64 / schema.mandatory.len() as f64
        + 0.05 * optional_filled as f64)
        .min(1.0);
    (score, missing)
}

/// Destructive verbs: acting on these from prose ambiguity deletes
/// user data. They never execute — at most a question naming the
/// exact target, usually a refusal.
const DESTRUCTIVE: &[&str] = &["delete", "remove", "destroy", "wipe", "drop", "erase", "rm"];

/// Contradiction pairs: both sides present means the request fights
/// itself. Each side is a set of equivalent words.
const CONTRADICTIONS: &[(&[&str], &[&str])] = &[
    (
        &["readonly", "read-only", "immutable"],
        &["edit", "write", "change", "mutate"],
    ),
    (&["create", "make", "add"], &["delete", "remove", "destroy"]),
    (
        &["allow", "permit", "enable"],
        &["forbid", "deny", "block", "disable"],
    ),
    (&["fast", "faster", "quick"], &["slow", "slower"]),
];

/// Context words: the request points at conversation history.
const CONTEXT_WORDS: &[&str] = &["previous", "last", "prior", "same", "other", "that"];

/// Decide what to do with a parse. Pure and total.
pub fn disposition(confidence: f64, frame: &str, prose: &str) -> Disposition {
    let ws = words(prose);
    // Destructive ambiguity refuses first: no threshold can bless it.
    if ws.iter().any(|w| DESTRUCTIVE.contains(&w.as_str())) {
        return Disposition::Block;
    }
    // Contradictions ask: the user must pick a side. Inflections
    // match too ("editing" counts as "edit") via a tiny local stemmer
    // — full Snowball would be a dependency for one comparison.
    fn stem_lite(w: &str) -> &str {
        if w.len() > 5 && w.ends_with("ing") {
            &w[..w.len() - 3]
        } else if w.len() > 4 && w.ends_with("ed") {
            &w[..w.len() - 2]
        } else if w.len() > 3 && w.ends_with('s') && !w.ends_with("ss") {
            &w[..w.len() - 1]
        } else {
            w
        }
    }
    let lower: Vec<String> = ws.clone();
    for (a, b) in CONTRADICTIONS {
        if lower.iter().any(|w| a.contains(&stem_lite(w)))
            && lower.iter().any(|w| b.contains(&stem_lite(w)))
        {
            return Disposition::Ask;
        }
    }
    // Multiple requests in one breath stage as questions, not as a
    // blind batch: half-failing a compound is worse than asking where
    // to start. (Staged execution is the follow-up project.)
    if clause_count(prose) > 1 {
        return Disposition::Ask;
    }
    if confidence >= ACT_THRESHOLD && frame != "unknown" {
        return Disposition::Execute;
    }
    // Anything with content words left to ask about gets a question;
    // vacuous input gets a refusal.
    let content = ws.iter().any(|w| {
        w.len() > 2
            && !STOPWORDS.contains(&w.as_str())
            && fuzzy_verb(w).is_none()
            && !VERBS.iter().any(|(v, _)| v == w)
            && exact_kind(w).is_none()
    });
    if content {
        Disposition::Ask
    } else {
        Disposition::Block
    }
}

/// Names of previously defined items of a kind, newest first, from
/// verified history. Powers "the previous screen" without guessing:
/// the caller shows these as options, never silently reuses them.
pub fn history_names(project_dir: &std::path::Path, kind: &str) -> Vec<String> {
    load_history(project_dir)
        .into_iter()
        .rev()
        .filter_map(|r| {
            r.defines
                .into_iter()
                .find(|(_, k)| k == kind)
                .map(|(n, _)| n)
        })
        .collect::<Vec<_>>()
}

/// True when the prose points at conversation history.
pub fn wants_context(prose: &str) -> bool {
    words(prose)
        .iter()
        .any(|w| CONTEXT_WORDS.contains(&w.as_str()))
}

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

/// Canonical kind for a surface word: exact match, then a single
/// trailing-`s` plural strip (`pages` → `page`, `structs` → `struct`).
/// Deterministic; names never consult this (only kind-slot search).
fn exact_kind(word: &str) -> Option<&'static str> {
    if let Some((_, k)) = KINDS.iter().find(|(s, _)| *s == word) {
        return Some(*k);
    }
    if word.len() > 3 && word.ends_with('s') && !word.ends_with("ss") {
        let singular = &word[..word.len() - 1];
        if let Some((_, k)) = KINDS.iter().find(|(s, _)| *s == singular) {
            return Some(*k);
        }
    }
    None
}

/// Nearest kind within typo tolerance (≤1 for short words, ≤2 above),
/// returning the canonical kind. Consulted only after exact + plural
/// matching miss — spelling paths keep priority, and names never
/// consult kinds (a meaning match is not spelling evidence for a name).
fn fuzzy_kind(word: &str) -> Option<&'static str> {
    if exact_kind(word).is_some() {
        return exact_kind(word);
    }
    let max_dist = if word.len() <= 4 { 1 } else { 2 };
    let mut best: Option<(usize, &'static str)> = None;
    for (surface, canon) in KINDS {
        let d = edit_distance(word, surface);
        if d <= max_dist && best.map(|(bd, _)| d < bd).unwrap_or(true) {
            best = Some((d, *canon));
        }
    }
    best.map(|(_, k)| k)
}

/// Material-relation grammar (the deterministic stand-in for a
/// Wikidata `material-used` edge): `made of X`, `made out of X`,
/// `made from X`, `out of X`, `built from X`, `built of X`.
/// Returns (base object, material), both lowercase. Base is the last
/// content word before the marker; material is the first content word
/// after it. Articles are skipped, nothing is invented.
pub fn extract_material(prose: &str) -> Option<(String, String)> {
    let ws = words(prose);
    let articles = ["a", "an", "the", "some", "any"];
    // Marker patterns, longest first so `made out of` wins over `out of`.
    const MARKERS: &[&[&str]] = &[
        &["made", "out", "of"],
        &["built", "out", "of"],
        &["made", "of"],
        &["made", "from"],
        &["built", "from"],
        &["built", "of"],
        &["constructed", "from"],
        &["constructed", "of"],
        &["out", "of"],
    ];
    for marker in MARKERS {
        if let Some(pos) = ws
            .windows(marker.len())
            .position(|w| w.iter().zip(marker.iter()).all(|(a, b)| a == *b))
        {
            let after = ws
                .iter()
                .skip(pos + marker.len())
                .find(|w| !articles.contains(&w.as_str()) && !STOPWORDS.contains(&w.as_str()));
            let before = ws[..pos]
                .iter()
                .rev()
                .find(|w| !articles.contains(&w.as_str()) && !STOPWORDS.contains(&w.as_str()));
            if let (Some(b), Some(m)) = (before, after) {
                return Some(((*b).clone(), (*m).clone()));
            } else if let Some(m) = after {
                return Some((String::new(), (*m).clone()));
            }
        }
    }
    None
}

/// PrimitiveVector: meaning as 5 numbers (mass, velocity, spatial,
/// valence, temporal), after grounded's PrimitiveMatrix. Synonyms
/// join by coordinates — auditable numbers, not bare table rows.
/// Cosine decides; the 0.90 bar is fixed and the vectors carry the
/// calibration, all visible below.
pub type PrimitiveVector = [f64; 5];

pub fn cosine(a: &PrimitiveVector, b: &PrimitiveVector) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let la: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let lb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if la == 0.0 || lb == 0.0 {
        0.0
    } else {
        (dot / (la * lb)).clamp(-1.0, 1.0)
    }
}

/// Verb vocabulary in meaning space: one vector per VERBS entry, in
/// the same order. Families cluster by construction intent.
const VERB_VECTORS: &[PrimitiveVector] = &[
    // build family
    [0.8, 0.4, 0.4, 0.5, 0.5],      // build
    [0.7, 0.4, 0.3, 0.5, 0.4],      // make
    [0.6, 0.5, 0.2, 0.4, 0.6],      // compile
    [0.7, 0.5, 0.5, 0.5, 0.5],      // assemble
    [0.75, 0.45, 0.40, 0.45, 0.55], // building
    [0.70, 0.45, 0.35, 0.45, 0.60], // built
    [0.68, 0.42, 0.32, 0.48, 0.45], // made
    [0.69, 0.41, 0.31, 0.49, 0.42], // makes
    [0.71, 0.43, 0.33, 0.47, 0.48], // making
    // create family
    [0.9, 0.2, 0.3, 0.6, 0.3],      // create
    [0.7, 0.2, 0.2, 0.5, 0.2],      // add
    [0.8, 0.1, 0.2, 0.5, 0.1],      // new
    [0.9, 0.3, 0.2, 0.6, 0.4],      // generate
    [0.6, 0.4, 0.2, 0.5, 0.4],      // write
    [0.88, 0.22, 0.28, 0.58, 0.32], // creating
    [0.86, 0.21, 0.27, 0.57, 0.33], // created
    [0.87, 0.23, 0.26, 0.56, 0.31], // creates
    [0.72, 0.21, 0.21, 0.51, 0.22], // adding
    [0.71, 0.20, 0.20, 0.50, 0.23], // added
    [0.89, 0.29, 0.21, 0.59, 0.39], // generating
    [0.88, 0.28, 0.20, 0.58, 0.40], // generated
    [0.62, 0.39, 0.21, 0.51, 0.39], // writing
    [0.61, 0.38, 0.20, 0.50, 0.38], // wrote
    [0.60, 0.37, 0.20, 0.49, 0.39], // written
    [0.82, 0.28, 0.28, 0.56, 0.34], // imagine
    [0.78, 0.32, 0.34, 0.54, 0.42], // render
    [0.76, 0.34, 0.32, 0.53, 0.40], // compose
    [0.74, 0.30, 0.28, 0.55, 0.36], // draw
    [0.73, 0.29, 0.27, 0.54, 0.35], // paint
    [0.70, 0.32, 0.30, 0.52, 0.33], // show
    // fix family
    [0.3, 0.3, 0.2, 0.4, 0.6],      // fix
    [0.3, 0.3, 0.2, 0.5, 0.6],      // repair
    [0.3, 0.2, 0.2, 0.4, 0.5],      // mend
    [0.2, 0.3, 0.1, 0.4, 0.6],      // correct
    [0.29, 0.29, 0.19, 0.41, 0.59], // fixing
    [0.28, 0.28, 0.18, 0.40, 0.60], // fixed
    [0.29, 0.30, 0.19, 0.42, 0.58], // fixes
    [0.30, 0.29, 0.19, 0.49, 0.59], // repairing
    [0.29, 0.28, 0.18, 0.48, 0.60], // repaired
    // publish family
    [0.2, 0.7, 0.8, 0.3, 0.4],      // publish
    [0.2, 0.8, 0.9, 0.3, 0.4],      // upload
    [0.3, 0.7, 0.8, 0.4, 0.4],      // release
    [0.2, 0.6, 0.8, 0.5, 0.3],      // share
    [0.21, 0.69, 0.79, 0.31, 0.39], // publishing
    [0.20, 0.68, 0.78, 0.30, 0.40], // published
    [0.21, 0.79, 0.89, 0.31, 0.39], // uploading
    [0.20, 0.78, 0.88, 0.30, 0.40], // uploaded
    // verify family
    [0.1, 0.5, 0.1, 0.0, 0.5],      // test
    [0.1, 0.4, 0.1, 0.0, 0.5],      // check
    [0.1, 0.4, 0.1, 0.1, 0.5],      // verify
    [0.2, 0.6, 0.3, 0.1, 0.5],      // run
    [0.11, 0.49, 0.11, 0.01, 0.51], // testing
    [0.10, 0.48, 0.10, 0.00, 0.50], // tested
    [0.11, 0.50, 0.11, 0.01, 0.49], // tests
    [0.10, 0.41, 0.10, 0.01, 0.51], // checking
    [0.09, 0.40, 0.09, 0.00, 0.50], // checked
    [0.10, 0.41, 0.10, 0.10, 0.51], // verifying
    [0.09, 0.40, 0.09, 0.09, 0.50], // verified
    [0.19, 0.59, 0.29, 0.10, 0.51], // running
    [0.20, 0.60, 0.30, 0.11, 0.50], // runs
];

/// Unlisted synonyms placed by meaning. Held-out probes for the
/// mechanism: none of these appear in VERBS or the tuning above.
const SYNONYM_VECTORS: &[(&str, PrimitiveVector)] = &[
    ("construct", [0.75, 0.45, 0.40, 0.45, 0.50]),
    ("fabricate", [0.75, 0.50, 0.35, 0.40, 0.50]),
    ("craft", [0.80, 0.30, 0.30, 0.55, 0.35]),
    ("produce", [0.85, 0.30, 0.25, 0.55, 0.35]),
    ("author", [0.70, 0.35, 0.20, 0.55, 0.35]),
    ("examine", [0.15, 0.40, 0.15, 0.05, 0.50]),
    ("inspect", [0.15, 0.40, 0.20, 0.05, 0.50]),
    ("audit", [0.20, 0.40, 0.15, 0.00, 0.55]),
];

/// Nearest table action by meaning. Consulted only after exact and
/// edit-fuzzy matching miss — spelling paths keep priority, and the
/// name-exclusion filters never consult vectors (a meaning match is
/// not spelling evidence).
pub fn vector_verb(word: &str) -> Option<FrameAction> {
    const BAR: f64 = 0.90;
    let query = SYNONYM_VECTORS.iter().find(|(w, _)| *w == word)?.1;
    let mut best: Option<(f64, FrameAction)> = None;
    for ((_, action), vec) in VERBS.iter().zip(VERB_VECTORS.iter()) {
        let s = cosine(&query, vec);
        if s >= BAR && best.map(|(bs, _)| s > bs).unwrap_or(true) {
            best = Some((s, *action));
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

/// Lowercased alphanumeric tokens in order. Intra-word hyphens fuse
/// first (`read-only` → `readonly`), so hyphenated compounds match the
/// lexicon instead of shattering into misleading pieces.
fn words(prose: &str) -> Vec<String> {
    let fused: String = {
        let chars: Vec<char> = prose.chars().collect();
        let mut out = String::new();
        for (i, c) in chars.iter().enumerate() {
            if *c == '-'
                && i > 0
                && i + 1 < chars.len()
                && chars[i - 1].is_alphanumeric()
                && chars[i + 1].is_alphanumeric()
            {
                continue;
            }
            out.push(*c);
        }
        out
    };
    fused
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
    let kind_of = |w: &str| -> Option<&'static str> { exact_kind(w).or_else(|| fuzzy_kind(w)) };
    let pos = words.iter().position(|w| kind_of(w) == Some(kind))?;
    let articles = [
        "a", "an", "the", "my", "this", "that", "some", "any", "each", "every",
    ];
    let mut i = pos;
    while i > 0 {
        i -= 1;
        // Articles and verbs (exact or typo'd — "buld" is an action,
        // never a name) are grammar. The anchor kind itself is skipped,
        // but OTHER kind words may be the referent ("settings page"
        // names Settings; "create a struct" names nothing).
        let is_anchor = KINDS.iter().any(|(k, v)| k == &words[i] && *v == kind);
        if articles.contains(&words[i].as_str())
            || VERBS.iter().any(|(v, _)| v == &words[i])
            || fuzzy_verb(&words[i]).is_some()
            || is_anchor
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
/// Words that are exactly verbs — or typos of verbs ("Buld") — are
/// actions, never names, and are excluded so a capitalized typo does
/// not become a struct called Buld.
fn capitals(prose: &str) -> Vec<String> {
    prose
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| {
            let mut cs = w.chars();
            matches!(cs.next(), Some(c) if c.is_uppercase()) && !w.chars().all(|c| c.is_uppercase())
        })
        .filter(|w| {
            let lower = w.to_lowercase();
            !VERBS.iter().any(|(v, _)| v == &lower) && fuzzy_verb(&lower).is_none()
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
    // Meaning fallback, last resort only: unlisted synonyms score by
    // vector similarity after spelling paths miss.
    let verb = verb
        .or(fuzzy)
        .or_else(|| ws.iter().find_map(|w| vector_verb(w)));
    // Specificity wins: bare "make" with a kind + name in the same
    // breath ("make a counter page") is creation, not compilation.
    // "make the thing work faster" (no kind, no name) stays a build.
    // Head-noun rule: the LAST kind word governs ("settings page" is a
    // page about settings, not a config). Exact + plural forms win
    // first; typo'd kinds (`stuct`) forgive second. Single pass each,
    // deterministic.
    let kind = ws
        .iter()
        .rev()
        .find_map(|w| exact_kind(w))
        .or_else(|| ws.iter().rev().find_map(|w| fuzzy_kind(w)));
    // Constraint words ride along as intent constraints.
    let constraints: Vec<String> = CONSTRAINTS
        .iter()
        .filter(|(w, _)| ws.iter().any(|x| x == *w))
        .map(|(_, c)| c.to_string())
        .collect();

    let similar = project_dir.and_then(|p| find_similar(p, prose));

    let mut frame = String::from("unknown");
    // Filled slot names — confidence derives from schema satisfaction
    // below, never from ad-hoc tallies.
    let mut slots: Vec<&'static str> = Vec::new();
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
            let target = if ws.iter().any(|w| w == "apk" || w == "android") {
                slots.push("target");
                "android-apk"
            } else if ws.iter().any(|w| w == "app" || w == "binary") {
                slots.push("target");
                "native"
            } else {
                "native"
            };
            slots.push("verb");
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
            slots.push("verb");
            // One name resolution for the whole parse (titled, then
            // Capitalized, then quoted, then adjacent) — two competing
            // resolutions once named different names for the same slot.
            let name = resolved_name.clone().unwrap_or_default();
            match (kind, name.is_empty()) {
                (Some(k), false) if DEFINABLE.contains(&k) => {
                    slots.push("kind");
                    slots.push("name");
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
                    if !def.sections.is_empty() {
                        slots.push("content");
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
            slots.push("verb");
            // Goal-only intents run the verify+repair loop as-is.
        }
        Some(FrameAction::Publish) => {
            frame = "publish".to_string();
            slots.push("verb");
            let repo = names.first().cloned().or_else(|| quotes.first().cloned());
            let mut text = format!(
                "publish github repository {}",
                repo.clone().unwrap_or_default()
            );
            if let Some(r) = &repo {
                slots.push("repo");
                text = format!("publish github repository repo:{}", r.to_lowercase());
            } else {
                intent.unknown_requirements = vec!["publish needs a repository name".to_string()];
            }
            if ws.iter().any(|w| w == "release" || w == "apk") {
                slots.push("release");
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
            slots.push("verb");
        }
        Some(FrameAction::Unknown) => {
            intent.unknown_requirements =
                vec!["action not mappable to build/create/fix/publish/verify".to_string()];
        }
    }

    // Material relation (deterministic Wikidata-`material-used` analogue):
    // `X made of Y` names a construction requirement, not a typo. The
    // parser banks it as a constraint + references + research questions
    // so the research planner can chase silhouettes, geometry, and
    // placement rules instead of dropping the words as noise.
    if let Some((base, material)) = extract_material(prose)
        && !material.is_empty()
    {
        let tag = format!("material:{}", material);
        if !intent.constraints.iter().any(|c| c == &tag) {
            intent.constraints.push(tag);
        }
        for r in [&base, &material] {
            if !r.is_empty() && !intent.references.iter().any(|x| x == r) {
                intent.references.push(r.clone());
            }
        }
        // Research plan, fully generic over base and material (any
        // material works the same — straw, glass, steel, pasta):
        // shape of the base, geometry of the material, placement rules.
        let mut plan = Vec::new();
        if !base.is_empty() {
            plan.push(format!("research {} shape and dimensions", base));
        }
        plan.push(format!("research {} geometry and appearance", material));
        if !base.is_empty() {
            plan.push(format!("research placement of {} over {}", material, base));
        }
        for p in plan {
            if !intent.unknown_requirements.iter().any(|x| x == &p) {
                intent.unknown_requirements.push(p);
            }
        }
    }

    // Confidence is slot satisfaction, derived from the frame schema
    // — the content gate still caps parsable-but-unbuildable parses.
    let (mut confidence, _missing) = evaluate_frame(&frame, &slots);
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
    fn vectors_cover_the_table() {
        // Zip safety: every verb word carries exactly one vector.
        assert_eq!(super::VERBS.len(), super::VERB_VECTORS.len());
    }

    #[test]
    fn table_is_geometrically_coherent() {
        // Every verb is nearer its own family than any other action:
        // misassigned rows would surface here, not in production.
        for (i, (_, action)) in super::VERBS.iter().enumerate() {
            let v = super::VERB_VECTORS[i];
            let mut best: Option<(f64, super::FrameAction)> = None;
            for (j, (_, a)) in super::VERBS.iter().enumerate() {
                let s = super::cosine(&v, &super::VERB_VECTORS[j]);
                if best.map(|(bs, _)| s > bs).unwrap_or(true) {
                    best = Some((s, *a));
                }
            }
            assert_eq!(
                best.map(|(_, a)| a),
                Some(*action),
                "verb {}",
                super::VERBS[i].0
            );
        }
    }

    #[test]
    fn held_out_synonyms_map_by_meaning() {
        use super::FrameAction as FA;
        // None of these are in VERBS and none tuned the vectors.
        for (word, action) in [
            ("construct", FA::Build),
            ("fabricate", FA::Build),
            ("craft", FA::Create),
            ("produce", FA::Create),
            ("author", FA::Create),
            ("examine", FA::Verify),
            ("inspect", FA::Verify),
            ("audit", FA::Verify),
        ] {
            assert_eq!(super::vector_verb(word), Some(action), "word {}", word);
        }
        assert_eq!(super::vector_verb("banana"), None);
        // End to end: an unlisted synonym drives the frame.
        let u = super::understand("Construct a settings page.", None);
        assert_eq!(u.frame, "create");
    }

    #[test]
    fn frame_satisfaction_scores_slots() {
        // Full create: all mandatory, content bonus caps at 1.0.
        let (sat, missing) = super::evaluate_frame("create", &["verb", "kind", "name", "content"]);
        assert_eq!((sat, missing.len()), (1.0, 0));
        // Missing name: 2/3 with the question named.
        let (sat, missing) = super::evaluate_frame("create", &["verb", "kind"]);
        assert!((sat - 2.0 / 3.0).abs() < 1e-9, "{}", sat);
        assert_eq!(missing, vec!["name"]);
        // Unknown frame: nothing to satisfy.
        assert_eq!(super::evaluate_frame("unknown", &["verb"]).0, 0.0);
        assert_eq!(super::evaluate_frame("nope", &["verb"]).0, 0.0);
        // Optional slots never dilute mandatory: publish+repo is whole.
        let (sat, _) = super::evaluate_frame("publish", &["verb", "repo"]);
        assert_eq!(sat, 1.0);
    }

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
        // Grown lexicon: `flibber` now sits within distance 4 of the
        // fix-family inflections, so it demonstrates neighbor search;
        // `zxqxjv` is the far-from-everything probe for the empty case.
        let cs = curiosities("please flibber the widget", None);
        assert!(!cs.is_empty());
        assert_eq!(cs[0].unknown, "flibber");
        assert!(
            cs[0].suggestions.iter().any(|s| s.contains("fix")),
            "expected a fix-family neighbor, got {:?}",
            cs[0].suggestions
        );
        let cs = curiosities("please zxqxjv the widget", None);
        assert!(!cs.is_empty());
        assert_eq!(cs[0].unknown, "zxqxjv");
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
    fn disposition_execute_ask_block() {
        use super::Disposition::*;
        use super::disposition;
        assert_eq!(
            disposition(0.9, "create", "create a Counter struct"),
            Execute
        );
        assert_eq!(disposition(0.9, "unknown", "flibber the widget"), Ask);
        assert_eq!(disposition(0.3, "create", "create a thing"), Ask);
        assert_eq!(disposition(0.0, "unknown", "!!!"), Block);
        // Destructive beats confident.
        assert_eq!(disposition(0.99, "fix", "delete the old files now"), Block);
        // Contradictions ask.
        assert_eq!(
            disposition(0.9, "fix", "make it read-only but allow editing"),
            Ask
        );
        // Multi-clause stages as a question even when parseable.
        assert_eq!(
            disposition(1.0, "create", "create a config and then add tests"),
            Ask
        );
    }

    #[test]
    fn history_names_lists_known_defines() {
        let dir = std::env::temp_dir().join(format!(
            "gc-und-hn-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut intent = empty_intent("make a counter page".to_string());
        intent.define = vec![Some(empty_define("Counter".to_string(), "page"))];
        save_history(&dir, "make a counter page", &intent);
        assert_eq!(
            super::history_names(&dir, "page"),
            vec!["Counter".to_string()]
        );
        assert!(super::history_names(&dir, "struct").is_empty());
        assert!(super::wants_context("same layout as the previous screen"));
        assert!(!super::wants_context("build the apk"));
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
