//! Intent research: intent → research intent → research subjects.
//!
//! The chat picture path used to jump from prose straight to pixels.
//! This module is the missing middle: given prose, state WHAT the
//! intent is, WHAT research would satisfy it, and WHICH subjects
//! need researching. Every subject gets the SAME generic questions
//! regardless of what it is — person, animal, or thing are research
//! problems, never code branches.
//!
//! Pure and offline: same prose, same bundle. Live fetching happens
//! downstream in `imagine::attempt`, driven by these intents.

use super::scene_intent::{self, ResearchQuery, SceneSpec};

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
    /// Each subject to research (plain nouns — no classes).
    pub subjects: Vec<String>,
    /// Structural questions for plates (pose, appearance…).
    pub structural: Vec<StructuralQuestion>,
    /// True when a human subject is present: the people-only index
    /// (OpenImages) joins the source sweep. Source plumbing, not a
    /// subject branch — the questions stay identical either way.
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
    let subjects: Vec<String> = spec.subjects.iter().map(|s| s.stype.clone()).collect();

    // Structural questions: what research must resolve before a
    // photograph can be selected for this scene. One generic set
    // per action and per subject — identical for person, animal, or
    // thing. What nouns MEAN is research's problem; the code never
    // branches on it.
    let mut structural = Vec::new();
    for a in &spec.actions {
        structural.push(StructuralQuestion {
            question: format!("which photographs show '{}'?", a.atype),
            topic: "action",
            answered: false,
            detail: format!("acquire '{}' photographs for this prompt", a.atype),
        });
    }
    for stype in &subjects {
        structural.push(StructuralQuestion {
            question: format!("which photographs show '{}'?", stype),
            topic: "subject",
            answered: false,
            detail: format!("acquire '{}' photographs for this prompt", stype),
        });
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
    for stype in &ir.subjects {
        out.push(format!("research subject: {}", stype));
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
        assert!(ir.subjects.iter().any(|s| s == "man"), "{:?}", ir.subjects);
        assert!(ir.people_subject);
        assert!(!ir.plan.is_empty());
        assert!(!ir.seeks.is_empty());
        // Generic questions, all gaps until research answers them.
        let topics: Vec<&str> = ir.structural.iter().map(|q| q.topic).collect();
        assert!(topics.contains(&"action"), "{:?}", topics);
        assert!(topics.contains(&"subject"), "{:?}", topics);
        assert!(
            ir.structural.iter().all(|q| !q.answered),
            "nothing is answered before research: {:?}",
            ir.structural
        );
    }

    #[test]
    fn animal_subjects_research_generically() {
        // No animal rows encoded: the open-vocabulary fallback names
        // the subject, the generic arm researches it.
        let ir = research_intent("An elephant crossing a river.");
        assert!(
            ir.subjects.iter().any(|s| s == "elephant"),
            "{:?}",
            ir.subjects
        );
        assert!(!ir.people_subject);
        assert!(
            ir.plan.iter().any(|q| q.requirement.contains("elephant")
                && q.queries.iter().any(|s| s.contains("elephant"))),
            "{:?}",
            ir.plan
        );
        // Same question shape as a person scene — no animal branch.
        let man = research_intent("A man standing on a mountain.");
        let topics_a: Vec<&str> = ir.structural.iter().map(|q| q.topic).collect();
        let topics_b: Vec<&str> = man.structural.iter().map(|q| q.topic).collect();
        assert_eq!(
            topics_a, topics_b,
            "question shape must not vary by subject"
        );
    }

    #[test]
    fn multi_subjects_all_researched() {
        let ir = research_intent("Cat sitting in human's lap.");
        assert!(ir.subjects.iter().any(|s| s == "cat"), "{:?}", ir.subjects);
        assert!(
            ir.subjects.iter().any(|s| s == "human"),
            "{:?}",
            ir.subjects
        );
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
        assert!(a.iter().any(|l| l.starts_with("structural [subject]:")));
    }
}
