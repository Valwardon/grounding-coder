//! Intent research: intent → research intent → research subjects.
//!
//! The chat picture path used to jump from prose straight to pixels.
//! This module is the missing middle: given prose, state WHAT the
//! intent is, WHAT research would satisfy it, WHICH subjects need
//! researching, and WHICH structural questions plates must answer
//! (pose, skin, hair, proportions — never pixels). Plates are
//! research material: they answer questions with numbers, and fresh
//! construction renders from those numbers.
//!
//! Pure and offline: same prose, same bundle. Live fetching happens
//! downstream in `imagine::attempt`, driven by these intents.

use super::scene_intent::{self, ResearchQuery, SceneSpec};

/// What kind of subject research must resolve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubjectClass {
    Person,
    Animal,
    Flora,
    Thing,
    Unknown,
}

impl SubjectClass {
    pub fn of(stype: &str) -> Self {
        match stype {
            "man" | "woman" | "human" => SubjectClass::Person,
            "cat" | "dog" | "elephant" | "horse" | "bird" | "fish" | "lion" | "tiger" | "bear" => {
                SubjectClass::Animal
            }
            "tree" => SubjectClass::Flora,
            "car" | "house" | "cake" | "jetski" => SubjectClass::Thing,
            _ => SubjectClass::Unknown,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SubjectClass::Person => "person",
            SubjectClass::Animal => "animal",
            SubjectClass::Flora => "flora",
            SubjectClass::Thing => "thing",
            SubjectClass::Unknown => "unknown",
        }
    }
}

/// One structural question plates must answer with measurements.
/// `answered` is true when the research bundle already carries a
/// measurement path (study medians, learned pose, measured rig);
/// false names the acquisition gap for the seeker.
#[derive(Debug, Clone)]
pub struct StructuralQuestion {
    pub question: String,
    pub topic: &'static str,
    pub answered: bool,
    pub detail: String,
}

/// The researched intent: everything downstream needs, decided
/// before any pixel or any fetch.
#[derive(Debug, Clone)]
pub struct IntentResearch {
    /// Canonical prose understanding (subjects, actions, objects).
    pub spec: SceneSpec,
    /// Per-requirement research queries + capability tags.
    pub plan: Vec<ResearchQuery>,
    /// Where each requirement should be looked up.
    pub seeks: Vec<super::seek::SeekIntent>,
    /// Each subject with its research class.
    pub subjects: Vec<(String, SubjectClass)>,
    /// Structural questions for plates (pose, skin, hair…).
    pub structural: Vec<StructuralQuestion>,
    /// True when at least one human subject needs a poseable action.
    pub people_subject: bool,
}

/// Research the intent behind prose: parse → plan → seek → subjects
/// → structural questions. No I/O, no fetching — fetching answers
/// these questions downstream.
pub fn research_intent(prose: &str) -> IntentResearch {
    let spec = scene_intent::parse_scene(prose);
    let plan = scene_intent::plan_research(&spec);
    let people_subject = spec.subjects.iter().any(|s| {
        matches!(s.stype.as_str(), "man" | "woman" | "human")
            || s.attributes.iter().any(|a| a == "woman" || a == "man")
    });
    let seeks = super::seek::intents_for_plan(&plan, people_subject);
    let subjects: Vec<(String, SubjectClass)> = spec
        .subjects
        .iter()
        .map(|s| (s.stype.clone(), SubjectClass::of(&s.stype)))
        .collect();

    // Structural questions: what research must resolve before a
    // photograph can be selected for this scene. Nothing here is
    // answered from encoded body knowledge — anatomy lives in
    // research (plates, oracle definitions), never in tables.
    // Pose questions resolve through pose-reference plates;
    // proportions through measured plates; skin through plate-study
    // medians downstream; hair stays a named gap until research
    // measures it.
    let mut structural = Vec::new();
    for a in &spec.actions {
        structural.push(StructuralQuestion {
            question: format!("which photographs show '{}'?", a.atype),
            topic: "pose",
            answered: false,
            detail: format!(
                "acquire '{}' pose-reference plates for this prompt",
                a.atype
            ),
        });
    }
    for (stype, class) in &subjects {
        match class {
            SubjectClass::Person => {
                structural.push(StructuralQuestion {
                    question: format!("which photographs show '{}'?", stype),
                    topic: "proportions",
                    answered: false,
                    detail: "acquire proportion-measurable plates for this prompt".to_string(),
                });
                structural.push(StructuralQuestion {
                    question: format!("what skin tones do '{}' plates measure?", stype),
                    topic: "skin",
                    answered: false,
                    detail: "acquire plate-study median tone for this prompt".to_string(),
                });
                structural.push(StructuralQuestion {
                    question: format!("which photographs show '{}' hair clearly?", stype),
                    topic: "hair",
                    answered: false,
                    detail: "hair structure unmeasured — needs hair research".to_string(),
                });
            }
            SubjectClass::Animal => {
                structural.push(StructuralQuestion {
                    question: format!("which photographs show '{}' anatomy?", stype),
                    topic: "pose",
                    answered: false,
                    detail: format!("acquire '{}' anatomy references for this prompt", stype),
                });
                structural.push(StructuralQuestion {
                    question: format!("what coat tones do '{}' plates measure?", stype),
                    topic: "skin",
                    answered: false,
                    detail: "acquire plate-study palette for this prompt".to_string(),
                });
            }
            SubjectClass::Flora | SubjectClass::Thing => {
                structural.push(StructuralQuestion {
                    question: format!("which photographs show '{}'?", stype),
                    topic: "proportions",
                    answered: false,
                    detail: format!("acquire '{}' silhouette references", stype),
                });
            }
            SubjectClass::Unknown => {
                structural.push(StructuralQuestion {
                    question: format!("what is '{}'?", stype),
                    topic: "identity",
                    answered: false,
                    detail: "unclassified subject — research must resolve before delivery"
                        .to_string(),
                });
            }
        }
    }

    IntentResearch {
        spec,
        plan,
        seeks,
        subjects,
        structural,
        people_subject,
    }
}

/// Log lines for the researched intent: intent, research plan,
/// subjects, and structural questions. Deterministic over prose.
pub fn log_lines(ir: &IntentResearch) -> Vec<String> {
    let mut out = Vec::new();
    out.push(format!(
        "intent: {} subject(s), {} action(s), {} object(s), confidence {:.2}",
        ir.spec.subjects.len(),
        ir.spec.actions.len(),
        ir.spec.objects.len(),
        ir.spec.confidence
    ));
    for (stype, class) in &ir.subjects {
        out.push(format!("research subject: {} [{}]", stype, class.as_str()));
    }
    for q in &ir.plan {
        out.push(format!(
            "research intent: {} → {} [{}]",
            q.requirement,
            q.queries.join(" / "),
            q.capability
        ));
    }
    for s in &ir.seeks {
        out.push(format!(
            "seek: {} via {:?} (closes '{}')",
            s.question, s.source, s.closes_gap
        ));
    }
    for q in &ir.structural {
        out.push(format!(
            "structural [{}]: {} — {} ({})",
            q.topic,
            q.question,
            if q.answered { "answered" } else { "gap" },
            q.detail
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intent_names_subjects_and_questions() {
        let ir = research_intent("A man standing on a mountain.");
        assert!(
            ir.subjects
                .iter()
                .any(|(s, c)| s == "man" && *c == SubjectClass::Person)
        );
        assert!(ir.people_subject);
        assert!(!ir.plan.is_empty());
        assert!(!ir.seeks.is_empty());
        // Structural questions name pose, skin, and hair — all gaps
        // until research answers them; nothing answered from tables.
        let topics: Vec<&str> = ir.structural.iter().map(|q| q.topic).collect();
        assert!(topics.contains(&"pose"), "{:?}", topics);
        assert!(topics.contains(&"skin"), "{:?}", topics);
        assert!(topics.contains(&"hair"), "{:?}", topics);
        assert!(
            ir.structural.iter().all(|q| !q.answered),
            "nothing is answered before research: {:?}",
            ir.structural
        );
    }

    #[test]
    fn animal_subjects_research_anatomy() {
        let ir = research_intent("An elephant crossing a river.");
        assert!(
            ir.subjects
                .iter()
                .any(|(s, c)| s == "elephant" && *c == SubjectClass::Animal),
            "{:?}",
            ir.subjects
        );
        assert!(!ir.people_subject);
        assert!(
            ir.plan.iter().any(|q| q.requirement.contains("elephant")
                && q.queries.iter().any(|s| s.contains("anatomy"))),
            "{:?}",
            ir.plan
        );
        // No human hair questions for an elephant scene.
        assert!(
            !ir.structural.iter().any(|q| q.topic == "hair"),
            "{:?}",
            ir.structural
        );
    }

    #[test]
    fn multi_subjects_all_researched() {
        let ir = research_intent("Cat sitting in human's lap.");
        let names: Vec<&str> = ir.subjects.iter().map(|(s, _)| s.as_str()).collect();
        assert!(names.contains(&"cat"), "{:?}", names);
        assert!(names.contains(&"human"), "{:?}", names);
        assert!(
            ir.seeks.iter().any(|s| s.closes_gap.contains("cat")),
            "{:?}",
            ir.seeks
        );
    }

    #[test]
    fn logging_is_deterministic() {
        let a = log_lines(&research_intent("A man standing on a mountain."));
        let b = log_lines(&research_intent("A man standing on a mountain."));
        assert_eq!(a, b);
        assert!(a.iter().any(|l| l.starts_with("intent:")));
        assert!(a.iter().any(|l| l.starts_with("research subject:")));
        assert!(a.iter().any(|l| l.starts_with("structural [pose]:")));
    }
}
