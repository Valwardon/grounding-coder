//! Seek: gaps ask their own research questions.
//!
//! The truth-seeking loop's question half. Given what is unknown —
//! truth gaps ("no eye-locked donors — acquire frontal faces") or
//! evidence shortfalls — emit research intents: what to ask, where
//! to look, and which gap an answer would close. Seeking never
//! answers from knowledge and never shapes anything; it only asks
//! better. Acquisition (fetching, measuring) answers; the loop
//! reconciles.
//!
//! Example ("cat sitting in human's lap"): the scene's unknowns fan
//! out into reference images, anatomy, body physics, sitting, lap —
//! research, research, research — each intent naming the gap it
//! would close. After research and source images are exhausted,
//! construction draws from scratch on the accumulated knowledge.

/// Where an intent looks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Curated metadata (Wikimedia Commons).
    Commons,
    /// General web image search.
    Web,
    /// Adult-filtered people boxes (people subjects only).
    OpenImages,
    /// Word/method definitions (research oracle).
    Oracle,
}

/// One research question: what to ask, where, and what it closes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeekIntent {
    pub question: String,
    pub queries: Vec<String>,
    pub source: Source,
    /// The gap text this would close. Empty only when the gap itself
    /// was empty — intents always name their target.
    pub closes_gap: String,
}

/// Content words of gap text (lowercased, stopwords dropped, long
/// first): deterministic query material, stated as such.
fn gap_keywords(gap: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "no", "a", "an", "the", "of", "on", "in", "with", "and", "or", "to", "for", "is", "are",
        "it", "at", "by", "from", "that", "this", "needs", "need",
    ];
    let mut words: Vec<String> = gap
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .filter(|w| !STOP.contains(&w.as_str()))
        .collect();
    words.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
    words.truncate(3);
    words
}

/// Ask from gaps: each unknown becomes one or more intents. Known
/// gap shapes map to their acquisition routes; anything else becomes
/// an oracle intent over the gap's own keywords — the system asking
/// about its confusion in its own words, never inventing an answer.
pub fn questions_for_gaps(gaps: &[String]) -> Vec<SeekIntent> {
    let mut out = Vec::new();
    for gap in gaps {
        let lower = gap.to_lowercase();
        if lower.contains("frontal faces") || lower.contains("skin-visible") {
            out.push(SeekIntent {
                question: "which photographs show measurable faces?".to_string(),
                queries: vec![
                    "portrait photograph".to_string(),
                    "face forward portrait".to_string(),
                ],
                source: Source::OpenImages,
                closes_gap: gap.clone(),
            });
        } else if lower.contains("hair") {
            out.push(SeekIntent {
                question: "how is hair structure measured classically?".to_string(),
                queries: vec!["hair segmentation reference".to_string()],
                source: Source::Oracle,
                closes_gap: gap.clone(),
            });
            out.push(SeekIntent {
                question: "which photographs show hair clearly?".to_string(),
                queries: vec!["hairstyle portrait photograph".to_string()],
                source: Source::Commons,
                closes_gap: gap.clone(),
            });
        } else if lower.contains("age") {
            out.push(SeekIntent {
                question: "which photographs span ages?".to_string(),
                queries: vec!["portrait by age photograph".to_string()],
                source: Source::Commons,
                closes_gap: gap.clone(),
            });
        } else if !gap.trim().is_empty() {
            let keys = gap_keywords(gap);
            if !keys.is_empty() {
                out.push(SeekIntent {
                    question: format!("how is this resolved: {}?", gap),
                    queries: keys.iter().map(|k| format!("{} reference", k)).collect(),
                    source: Source::Oracle,
                    closes_gap: gap.clone(),
                });
            }
        }
    }
    out
}

/// Ask from a research plan: each planned requirement becomes
/// intents aimed at the sources that serve its capability. The plan
/// says WHAT is needed; seeking says WHERE to look and WHAT gap an
/// answer closes. Capability-to-source is a stated table, and
/// people-subjects additionally sweep Open Images (boxes beat
/// segmentation) while everything else stays on Commons/Web.
pub fn intents_for_plan(
    plan: &[super::scene_intent::ResearchQuery],
    people_subject: bool,
) -> Vec<SeekIntent> {
    let mut out = Vec::new();
    for q in plan {
        let sources: &[Source] = match q.capability {
            "photographic-subject" if people_subject => &[Source::OpenImages, Source::Commons],
            "photographic-subject" => &[Source::Commons, Source::Web],
            "photographic-place" | "photo-asset" => &[Source::Commons, Source::Web],
            "fresh-construction" => &[Source::Web, Source::Commons],
            _ => &[Source::Commons],
        };
        for source in sources {
            out.push(SeekIntent {
                question: format!("{}?", q.requirement),
                queries: q.queries.clone(),
                source: *source,
                closes_gap: q.requirement.clone(),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaps_ask_their_own_questions() {
        let intents = questions_for_gaps(&[
            "no eye-locked donors — acquire frontal faces".to_string(),
            "hair structure unmeasured — needs hair extractor".to_string(),
        ]);
        // Frontal-faces gap asks OpenImages; hair gap asks methods
        // AND photographs. Every intent names its gap.
        assert!(
            intents
                .iter()
                .any(|i| i.source == Source::OpenImages && i.closes_gap.contains("frontal faces"))
        );
        assert!(
            intents
                .iter()
                .any(|i| i.source == Source::Oracle && i.closes_gap.contains("hair"))
        );
        assert!(
            intents
                .iter()
                .any(|i| i.source == Source::Commons && i.closes_gap.contains("hair"))
        );
        for i in &intents {
            assert!(!i.closes_gap.is_empty());
            assert!(!i.queries.is_empty());
        }
    }

    #[test]
    fn unknown_gaps_ask_in_their_own_words() {
        // Nothing programmed for bridge trusses: the system still
        // asks, using the gap's own keywords, via the oracle.
        let intents = questions_for_gaps(&["truss geometry unmeasured".to_string()]);
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].source, Source::Oracle);
        assert!(intents[0].queries.iter().any(|q| q.contains("truss")));
        assert!(questions_for_gaps(&["   ".to_string()]).is_empty());
        assert!(questions_for_gaps(&[]).is_empty());
    }

    #[test]
    fn plans_route_to_sources() {
        let spec = crate::engine::scene_intent::parse_scene("An elephant crossing a river.");
        let plan = crate::engine::scene_intent::plan_research(&spec);
        assert!(!plan.is_empty());
        // Non-people subject: Commons/Web only, never OpenImages
        // (a people-only index must not answer elephant questions).
        let intents = intents_for_plan(&plan, false);
        assert!(!intents.is_empty());
        assert!(intents.iter().all(|i| i.source != Source::OpenImages));
        assert!(intents.iter().any(|i| i.closes_gap.contains("elephant")));
        // People subjects sweep OpenImages first.
        let spec = crate::engine::scene_intent::parse_scene("A man standing on a mountain.");
        let plan = crate::engine::scene_intent::plan_research(&spec);
        let intents = intents_for_plan(&plan, true);
        assert!(intents.iter().any(|i| i.source == Source::OpenImages));
    }

    #[test]
    fn seeking_is_deterministic() {
        let gaps = vec!["hair structure unmeasured — needs hair extractor".to_string()];
        assert_eq!(questions_for_gaps(&gaps), questions_for_gaps(&gaps));
    }
}
