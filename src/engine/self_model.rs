//! Self-model: the engine's inward semantic graph (grounded, turned
//! on itself).
//!
//! Grounded keeps a graph of the WORLD; the self-model keeps a graph
//! of the AGENT: every capability (task kinds, synthesis families,
//! oracles, picture pipeline), what each needs, and where each
//! honestly stops. Answers about the bot come from this table —
//! never generated, never guessed. Limits are first-class entries,
//! not apologies: "can you X" resolves to capable-with-receipt or
//! named-missing-capability, same honesty as everywhere else.
//!
//! Track record comes from the project's own judged logs
//! (rank/outcome journals when present): what the bot attempted,
//! what verified. No history file = "no history yet", never a
//! fabricated past.

/// One capability row: what it does, what it needs, where it stops.
#[derive(Debug, Clone)]
pub struct Capability {
    /// Stable id, e.g. "repair.import", "synthesize.function".
    pub id: &'static str,
    /// What it does, one line.
    pub does: &'static str,
    /// What must exist for it to work.
    pub needs: &'static str,
    /// Where it honestly stops (empty = no stated limit).
    pub limit: &'static str,
}

/// The full capability table. Adding a row is a product decision
/// with a test; removing one without removing the code fails review.
pub const CAPABILITIES: &[Capability] = &[
    Capability {
        id: "verify.project",
        does: "check, lint, and test a project; report errors with locations",
        needs: "a project directory with a toolchain",
        limit: "cannot judge aesthetics, only compiler and test output",
    },
    Capability {
        id: "repair.import",
        does: "restore missing imports from compiler suggestions",
        needs: "an E0425/E0432 diagnostic naming the path",
        limit: "only exact compiler bytes; never invents paths",
    },
    Capability {
        id: "repair.parameter",
        does: "flip exact-byte switches from known recipes (photo mottling)",
        needs: "a seeded recipe plus a matching diagnostic anchor",
        limit: "cannot author new logic, only apply known switches",
    },
    Capability {
        id: "synthesize.function",
        does: "write function bodies from input/output contract cases",
        needs: "cases plus an inferred signature inside a known family",
        limit: "unknown shapes block; names come from the user",
    },
    Capability {
        id: "research.unknown",
        does: "investigate unknown words via codebase, docs, and web",
        needs: "a bounded fetch budget; user questions stay questions",
        limit: "web text never becomes verified knowledge by itself",
    },
    Capability {
        id: "picture.construct",
        does: "build posed figures from researched plates and oracles",
        needs: "sourced plates with provenance; sufficiency budgets",
        limit: "no generative model; below-minimum collections refuse",
    },
    Capability {
        id: "picture.verify",
        does: "measure renders (variance, features, symmetry) with thresholds",
        needs: "a rendered image plus stated expectations",
        limit: "pixel facts only; taste stays with the human",
    },
    Capability {
        id: "intent.parse",
        does: "turn messy prose into frames, slots, and confidence receipts",
        needs: "verbs/kinds in its lexicon; quotes for literals",
        limit: "below 0.75 it asks or researches instead of acting",
    },
];

/// Stopwords ignored in capability matching (small, documented).
const MATCH_STOP: &[&str] = &[
    "can", "you", "your", "my", "the", "a", "an", "to", "do", "does", "it", "and", "or", "me",
    "please", "what", "how",
];

/// Can the bot do the described thing? Scores each capability by
/// distinct content-word hits (id or description); the unique best
/// wins, ties and zeroes refuse honestly.
pub fn can_do(ask: &str) -> Result<(&'static str, &'static str), String> {
    let words: Vec<String> = ask
        .to_lowercase()
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|w| w.len() > 3 && !MATCH_STOP.contains(w))
        .map(|w| w.to_string())
        .collect();
    let mut scored: Vec<(usize, &Capability)> = CAPABILITIES
        .iter()
        .map(|c| {
            let blob = format!("{} {}", c.id.replace('.', " "), c.does.to_lowercase());
            let n = words.iter().filter(|w| blob.contains(w.as_str())).count();
            (n, c)
        })
        .filter(|(n, _)| *n > 0)
        .collect();
    scored.sort_by_key(|a| std::cmp::Reverse(a.0));
    // A lone shared word ("write", "code") proves nothing — demand
    // two distinct hits and a unique winner, else refuse honestly.
    match scored.as_slice() {
        [(n, c)] if *n >= 2 => Ok((c.id, c.limit)),
        [(n, c), (m, _), ..] if *n >= 2 && n > m => Ok((c.id, c.limit)),
        [] => Err("no capability matches — that is outside what I do".to_string()),
        _ => {
            Err("too vague: several capabilities match or too few words — be specific".to_string())
        }
    }
}

/// Full self-description, generated from the table (count included
/// so drift between code and claims fails visibly).
pub fn describe() -> String {
    let mut out = format!(
        "I am grounding-coder: a deterministic coding engine ({} capabilities). ",
        CAPABILITIES.len()
    );
    for c in CAPABILITIES {
        out.push_str(&format!("\n- {}: {}.", c.id, c.does));
    }
    out
}

/// Stated limits, one per capability that has any.
pub fn limits() -> Vec<(&'static str, &'static str)> {
    CAPABILITIES
        .iter()
        .filter(|c| !c.limit.is_empty())
        .map(|c| (c.id, c.limit))
        .collect()
}

/// Track record from the project's own judged logs. Returns
/// (attempts, successes) across rank outcomes, or None when the
/// project has no history — never a fabricated past.
pub fn track_record(project_dir: &std::path::Path) -> Option<(usize, usize)> {
    let path = project_dir.join(".grounding").join("outcomes.jsonl");
    let text = std::fs::read_to_string(path).ok()?;
    let mut attempts = 0usize;
    let mut successes = 0usize;
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        attempts += 1;
        let fixed = v.get("fixed").and_then(|x| x.as_bool()).unwrap_or(false);
        let success = v.get("success").and_then(|x| x.as_bool()).unwrap_or(false);
        if fixed || success {
            successes += 1;
        }
    }
    if attempts == 0 {
        return None;
    }
    Some((attempts, successes))
}

/// Is this prose a question about the bot itself? Tight matcher:
/// self-words plus capability-asking phrases. Never fires on
/// ordinary tasks ("give it the same layout" has no self words).
pub fn is_self_question(prompt: &str) -> bool {
    let p = prompt.to_lowercase();
    let words: Vec<&str> = p.split(|c: char| !c.is_alphanumeric()).collect();
    let self_word = ["you", "your", "yourself", "yours"]
        .iter()
        .any(|w| words.contains(w));
    let ask_word = [
        "can you",
        "what can",
        "who are",
        "your limit",
        "yourself",
        "what are you",
        "do you do",
    ]
    .iter()
    .any(|w| p.contains(w));
    self_word && ask_word
}

/// Answer a self-question from the table. Pure function of prompt +
/// project dir (for the track record only).
pub fn answer_self(prompt: &str, project_dir: &std::path::Path) -> String {
    let p = prompt.to_lowercase();
    if p.contains("limit") {
        let mut out = String::from("My stated limits:\n");
        for (id, limit) in limits() {
            out.push_str(&format!("- {}: {}\n", id, limit));
        }
        return out;
    }
    if p.contains("who are") || p.contains("what are you") {
        return describe();
    }
    if p.contains("what can you do") || p.contains("list") {
        let mut out = describe();
        out.push_str(
            "\nAsk about any one by name and I'll state what it needs and where it stops.\n",
        );
        return out;
    }
    match can_do(prompt) {
        Ok((id, limit)) => {
            let cap = CAPABILITIES.iter().find(|c| c.id == id).unwrap();
            let mut out = format!("Yes — {}: {}.\nNeeds: {}.\n", id, cap.does, cap.needs);
            if !limit.is_empty() {
                out.push_str(&format!("Limit: {}.\n", limit));
            }
            if let Some((a, s)) = track_record(project_dir) {
                out.push_str(&format!(
                    "Track record here: {}/{} judged fixes held.\n",
                    s, a
                ));
            } else {
                out.push_str("No track record in this project yet.\n");
            }
            out
        }
        Err(why) => format!("{}\n{}", why, describe()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_covers_core_repertoire() {
        let ids: Vec<&str> = CAPABILITIES.iter().map(|c| c.id).collect();
        for want in [
            "verify.project",
            "repair.import",
            "synthesize.function",
            "research.unknown",
            "picture.construct",
            "intent.parse",
        ] {
            assert!(ids.contains(&want), "missing {}", want);
        }
        assert!(limits().len() >= 6);
    }

    #[test]
    fn can_do_matches_and_misses() {
        assert!(can_do("can you repair my import").is_ok());
        assert!(can_do("can you write poetry").is_err());
    }

    #[test]
    fn self_questions_route_only() {
        assert!(is_self_question("what can you do?"));
        assert!(is_self_question("who are you?"));
        assert!(is_self_question("what are your limits?"));
        assert!(!is_self_question("Give it the same layout as before."));
        assert!(!is_self_question("Build the android apk."));
        assert!(!is_self_question("Create a Counter struct."));
    }

    #[test]
    fn no_history_says_so() {
        let dir = std::env::temp_dir().join(format!(
            "gc-self-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert_eq!(track_record(&dir), None);
        let ans = answer_self("who are you?", &dir);
        assert!(ans.contains("grounding-coder"));
        assert!(ans.contains("8 capabilities"));
    }
}
