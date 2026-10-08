//! Clarification: ambiguous intent → questions → answers → repeat.
//!
//! The loop in `AGENTS.md` says failure is never terminal — a request
//! the engine cannot ground comes back as questions, and each answer
//! is another fact, until nothing ambiguous is left. This module is
//! that loop, and it is the same in every domain: sense an ambiguity,
//! name it, offer what the source can offer, repeat.
//!
//! Nothing here guesses. For a scene the ambiguities are the parser's
//! own: what it left unresolved, what it flagged ambiguous, how much
//! of the prose it could ground, and which things sit behind a
//! spatial word whose sense the prose never fixed ("in a lake" —
//! in the water, or the lake behind her?). For a request that is not
//! a scene at all (the parser grounds none of it and it reads like a
//! program) the domain's own unknowns are named instead.
//!
//! [`clarify`] states the questions; [`resolve`] applies answers and
//! re-runs, so calling it with every answer yields `resolved: true`
//! and the request is ready to act on.

use super::scene_intent::{self, SceneSpec};

/// Which kind of request this is; decides which unknowns exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// A scene to be researched and rendered.
    Scene,
    /// A program to be written.
    Code,
    /// Neither — the engine has nothing to measure it against.
    Unknown,
}

/// One ambiguity, named with the aspect that resolves it. The aspect
/// is the key: an answer for it retires the question.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ambiguity {
    /// Stable key used to match an answer ("relation:lake_1").
    pub aspect: String,
    /// What to ask the user.
    pub question: String,
    /// What the source can offer, best first; empty means free text.
    pub options: Vec<String>,
}

/// The state of one clarification round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clarify {
    pub request: String,
    pub domain: Domain,
    /// Still-open ambiguities, in a deterministic order.
    pub ambiguities: Vec<Ambiguity>,
    /// True when nothing is left to ask — safe to act.
    pub resolved: bool,
}

/// Spatial words whose sense a bare noun never fixes. Language, not
/// subject knowledge: the same table serves every scene.
const SPATIAL: &[(&str, &str, &str)] = &[
    (
        "in",
        "in contact with it (inside the water/place)",
        "it is the setting behind or around the subject",
    ),
    (
        "on",
        "standing on top of it",
        "it is merely near or behind the subject",
    ),
    ("under", "beneath it", "lower in the frame than it"),
    ("behind", "occluded by it", "further away with it in front"),
];

/// Words that make a request read like a program rather than a scene.
const CODE_CUES: &[&str] = &[
    "program",
    "script",
    "code",
    "function",
    "app",
    "application",
    "tool",
    "automate",
    "automation",
    "cli",
    "daemon",
    "service",
    "bot",
    "endpoint",
];

/// Content words of a request, lowercase, in order.
fn words(prose: &str) -> Vec<String> {
    prose
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

fn looks_like_code(ws: &[String]) -> bool {
    ws.iter().any(|w| {
        CODE_CUES.contains(&w.as_str()) || w.starts_with("macro") || w.starts_with("keystroke")
    })
}

/// Sense the ambiguities of a request, then retire whichever the
/// answers already close. Same request + same answers → same round.
pub fn resolve(request: &str, answers: &[(String, String)]) -> Clarify {
    let mut c = clarify(request);
    c.ambiguities
        .retain(|a| !answers.iter().any(|(k, _)| k == &a.aspect));
    c.resolved = c.ambiguities.is_empty();
    c
}

/// Sense the ambiguities of a request as they stand, unresolved.
pub fn clarify(request: &str) -> Clarify {
    let spec = scene_intent::parse_scene(request);
    let ws = words(request);
    let domain = if looks_like_code(&ws) && spec.confidence < 0.5 {
        Domain::Code
    } else if !spec.subjects.is_empty() || !spec.actions.is_empty() {
        // A scene needs a subject doing an action. Bare noun lists with
        // nothing grounded (confidence 0) are not a scene — they are
        // the parser admitting it had no picture to read.
        Domain::Scene
    } else {
        Domain::Unknown
    };

    let mut ambiguities = match domain {
        Domain::Scene => scene_ambiguities(request, &ws, &spec),
        Domain::Code => code_ambiguities(&ws),
        Domain::Unknown => vec![Ambiguity {
            aspect: "intent".into(),
            question: format!("I cannot tell what to do with: {request:?}. What is it?"),
            options: Vec::new(),
        }],
    };
    ambiguities.sort_by(|a, b| a.aspect.cmp(&b.aspect));
    ambiguities.dedup_by(|a, b| a.aspect == b.aspect);
    Clarify {
        request: request.to_string(),
        domain,
        resolved: ambiguities.is_empty(),
        ambiguities,
    }
}

/// Scene ambiguities, all from the parser's own signals.
fn scene_ambiguities(request: &str, ws: &[String], spec: &SceneSpec) -> Vec<Ambiguity> {
    let mut out = Vec::new();

    // The parser could not ground all the prose: confirm the reading.
    if spec.confidence < 0.6 {
        let subjects: Vec<&str> = spec.subjects.iter().map(|s| s.stype.as_str()).collect();
        let objects: Vec<&str> = spec.objects.iter().map(|o| o.otype.as_str()).collect();
        out.push(Ambiguity {
            aspect: "reading".into(),
            question: format!(
                "I read {request:?} as {} — is that the scene you mean?",
                describe(&subjects, &objects)
            ),
            options: vec![
                "yes, read it that way".into(),
                "no — I will rephrase".into(),
            ],
        });
    }

    // Everything the parser explicitly left open.
    for u in &spec.unresolved {
        let key = u.split(':').next().unwrap_or(u).trim().to_string();
        out.push(Ambiguity {
            aspect: format!("unresolved:{key}"),
            question: format!("{u}. What setting should it have?"),
            options: Vec::new(),
        });
    }

    // An action the prose underdetermined ("waving" — hand or flag?).
    for a in spec.actions.iter().filter(|a| a.ambiguous) {
        out.push(Ambiguity {
            aspect: format!("action:{}", a.atype),
            question: format!("How is '{}' done here?", a.atype),
            options: Vec::new(),
        });
    }

    // A thing sitting behind a spatial word whose sense is unfixed
    // ("in a lake" — in the water, or the lake behind her?).
    for o in &spec.objects {
        let noun = o.otype.to_lowercase();
        let Some(i) = ws.iter().position(|w| *w == noun) else {
            continue;
        };
        // Walk back past function words ("in A lake") to the preposition.
        let prep = ws[..i].iter().rev().find(|w| !is_function_word(w)).cloned();
        let Some(prep) = prep else { continue };
        if let Some((p, sense_a, sense_b)) = SPATIAL.iter().find(|(p, _, _)| *p == prep) {
            out.push(Ambiguity {
                aspect: format!("relation:{}", o.id),
                question: format!(
                    "'{p} a {noun}' — is the {} {sense_a}, or {sense_b}?",
                    if spec.subjects.is_empty() {
                        "subject".to_string()
                    } else {
                        spec.subjects
                            .iter()
                            .map(|s| s.stype.clone())
                            .collect::<Vec<_>>()
                            .join(", ")
                    }
                ),
                options: vec![sense_a.to_string(), sense_b.to_string()],
            });
        }
    }

    out
}

/// Function words the scan walks past when looking for the word that
/// governs a noun ("in **a** lake").
fn is_function_word(w: &str) -> bool {
    matches!(w, "a" | "an" | "the" | "of" | "and")
}

/// Code ambiguities: the unknowns every program request carries, plus
/// the one its own words name. Domain knowledge, not a subject table.
fn code_ambiguities(ws: &[String]) -> Vec<Ambiguity> {
    let has = |w: &str| ws.iter().any(|x| x == w || x.starts_with(w));
    let mut out = vec![
        Ambiguity {
            aspect: "language".into(),
            question: "What language should it be written in?".into(),
            options: vec![
                "Rust".into(),
                "Python".into(),
                "Shell (bash)".into(),
                "JavaScript".into(),
                "no preference".into(),
            ],
        },
        Ambiguity {
            aspect: "platform".into(),
            question: "What platform must it run on?".into(),
            options: vec![
                "Linux".into(),
                "macOS".into(),
                "Windows".into(),
                "Android".into(),
                "cross-platform".into(),
            ],
        },
        Ambiguity {
            aspect: "interface".into(),
            question: "How should it be used?".into(),
            options: vec![
                "a command-line tool".into(),
                "a background service".into(),
                "a GUI".into(),
            ],
        },
    ];
    if has("keystroke") || has("key") || has("hotkey") {
        out.push(Ambiguity {
            aspect: "which_keystroke".into(),
            question: "Which keystroke — a single key, or a combination?".into(),
            options: Vec::new(),
        });
    }
    if ws.iter().any(|w| w.starts_with("macro")) {
        out.push(Ambiguity {
            aspect: "macro_meaning".into(),
            question: "What should the macro do?".into(),
            options: vec![
                "send a fixed sequence of keys".into(),
                "remap one key to another".into(),
                "record my keys and replay them".into(),
            ],
        });
    }
    out
}

fn describe(subjects: &[&str], objects: &[&str]) -> String {
    let s = if subjects.is_empty() {
        "no subject".to_string()
    } else {
        subjects.join(", ")
    };
    if objects.is_empty() {
        s
    } else {
        format!("{s} with {}", objects.join(", "))
    }
}

/// Human-readable log of a round: what is still open. Deterministic.
pub fn log_lines(c: &Clarify) -> Vec<String> {
    let mut out = vec![format!(
        "clarify [{}]: {} open — {}",
        match c.domain {
            Domain::Scene => "scene",
            Domain::Code => "code",
            Domain::Unknown => "unknown",
        },
        c.ambiguities.len(),
        if c.resolved {
            "resolved"
        } else {
            "needs answers"
        }
    )];
    for a in &c.ambiguities {
        let opts = if a.options.is_empty() {
            String::new()
        } else {
            format!("  [{}]", a.options.join(" | "))
        };
        out.push(format!("  {}: {}{}", a.aspect, a.question, opts));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answered(aspects: &[&str]) -> Vec<(String, String)> {
        aspects
            .iter()
            .map(|a| (a.to_string(), "answer".to_string()))
            .collect()
    }

    #[test]
    fn a_woman_in_a_lake_asks_how_she_relates_to_the_lake() {
        let c = clarify("A woman standing in a lake.");
        assert_eq!(c.domain, Domain::Scene);
        assert!(!c.resolved, "an underfed scene is not resolved");
        let rel = c
            .ambiguities
            .iter()
            .find(|a| a.aspect == "relation:lake_1")
            .expect("the lake relation must be asked");
        assert!(
            rel.options.iter().any(|o| o.contains("contact")),
            "the water sense must be offered: {:?}",
            rel.options
        );
    }

    #[test]
    fn a_bare_scene_asks_for_the_place_the_parser_left_open() {
        let c = clarify("A woman standing.");
        assert_eq!(c.domain, Domain::Scene);
        assert!(
            c.ambiguities
                .iter()
                .any(|a| a.aspect.starts_with("unresolved:")),
            "the unplaced action must be asked: {:?}",
            c.ambiguities
        );
    }

    #[test]
    fn a_program_request_becomes_code_questions_not_a_scene() {
        let c = clarify("Write a program that macros a keystroke.");
        assert_eq!(c.domain, Domain::Code);
        let aspects: Vec<&str> = c.ambiguities.iter().map(|a| a.aspect.as_str()).collect();
        for want in ["language", "platform", "which_keystroke", "macro_meaning"] {
            assert!(aspects.contains(&want), "missing {want}: {aspects:?}");
        }
        assert!(
            c.ambiguities
                .iter()
                .find(|a| a.aspect == "language")
                .unwrap()
                .options
                .contains(&"Rust".to_string()),
            "language options must be offered"
        );
    }

    #[test]
    fn answering_every_aspect_resolves_the_round() {
        let c = clarify("Write a program that macros a keystroke.");
        let keys: Vec<String> = c.ambiguities.iter().map(|a| a.aspect.clone()).collect();
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        let done = resolve("Write a program that macros a keystroke.", &answered(&keys));
        assert!(
            done.resolved,
            "all answered → resolved: {:?}",
            done.ambiguities
        );
        assert!(done.ambiguities.is_empty());
    }

    #[test]
    fn a_partial_answer_leaves_only_the_rest() {
        let c = resolve(
            "Write a program that macros a keystroke.",
            &answered(&["language", "platform"]),
        );
        assert!(!c.resolved);
        let aspects: Vec<&str> = c.ambiguities.iter().map(|a| a.aspect.as_str()).collect();
        assert!(!aspects.contains(&"language"), "answered aspect retired");
        assert!(aspects.contains(&"interface"), "unanswered aspect stays");
    }

    #[test]
    fn clarification_is_deterministic() {
        assert_eq!(
            clarify("A woman standing in a lake."),
            clarify("A woman standing in a lake.")
        );
        assert_eq!(
            log_lines(&clarify("Write a program that macros a keystroke.")),
            log_lines(&clarify("Write a program that macros a keystroke."))
        );
    }

    #[test]
    fn unreadable_prose_asks_what_it_is() {
        let c = clarify("Zzz qwerty floob.");
        assert_eq!(c.domain, Domain::Unknown);
        assert_eq!(c.ambiguities.len(), 1);
        assert_eq!(c.ambiguities[0].aspect, "intent");
    }
}
