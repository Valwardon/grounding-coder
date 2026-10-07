//! One command from prose to photo: imagine.
//!
//! ```text
//! prose → intent → research → rank → deliver
//! ```
//!
//! Research finds the photograph; the photograph is the answer.
//!
//! There is no pixel generation here of any kind — no montage, no
//! meshes, no statistical synthesis, no per-subject extractors. A
//! person-synthesizer would demand a cat-synthesizer and a
//! tree-synthesizer next: infinite programming for every living
//! thing, which research already answers. So the pipeline parses
//! intent, researches licensed plates for ANY subject through the
//! same generic path, ranks them by title/query overlap, and hands
//! over the winner's own bytes labeled exactly what they are: a
//! sourced photograph by a real photographer, not a generated image.
//! No code here varies by subject — subjects live in research.
//!
//! Refusal is honest: with no usable plate the command refuses with
//! its research trail instead of fabricating pixels.

use super::plates::SourcedPlate;
use super::vision::Image;

/// Who stands in the photo. Subjects come from researched
/// photographs with complete provenance — no likenesses generated,
/// no minors delivered, by construction of the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubjectKind {
    Man,
    Woman,
    Person,
}

/// Props shape research queries, never geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropKind {
    Roses,
    Flag,
}

/// The parsed request: deterministic keyword scan, stated limits —
/// no grammar beyond "contains the word". Subject, pose, and props
/// shape RESEARCH queries, never pixels.
#[derive(Debug, Clone)]
pub struct Brief {
    pub subject: SubjectKind,
    pub pose: String,
    pub props: Vec<PropKind>,
    pub raw: String,
}

pub fn parse_brief(prose: &str) -> Brief {
    // Whole-token matching (never substrings): "human" contains
    // "man" as characters but is not a man, and that bleed once sent
    // a cat prompt down the man-portrait path. Verbs match base and
    // -ing forms explicitly; props match singulars and plurals.
    let lower = prose.to_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let has = |w: &str| tokens.contains(&w);
    let subject = if has("woman") || has("female") || has("lady") || has("girl") {
        SubjectKind::Woman
    } else if has("man") || has("male") || has("gentleman") || has("boy") {
        SubjectKind::Man
    } else {
        SubjectKind::Person
    };
    let pose = if has("peace") {
        "peace sign".to_string()
    } else if has("salute") || has("salutes") || has("saluting") || has("saluted") {
        "saluting".to_string()
    } else if has("cross") || has("crosses") || has("crossing") || has("crossed") {
        "crossing".to_string()
    } else if has("sit") || has("sits") || has("sitting") || has("sat") {
        "sitting".to_string()
    } else if has("wave") || has("waves") || has("waving") || has("waved") {
        "waving".to_string()
    } else {
        "standing".to_string()
    };
    let mut props = Vec::new();
    if has("rose") || has("roses") || has("flower") || has("flowers") || has("bouquet") {
        props.push(PropKind::Roses);
    }
    if has("flag") || has("flags") {
        props.push(PropKind::Flag);
    }
    Brief {
        subject,
        pose,
        props,
        raw: prose.to_string(),
    }
}

/// A bare personal name as author-or-title ("Firstname Lastname",
/// optional trailing number, nothing else): evidence of an
/// identified subject. Selection skips these — delivering strangers
/// is the job; named individuals are a human decision.
pub fn looks_like_person_name(s: &str) -> bool {
    // Author strings glue photographer credit and photo title with
    // commas ("lifrita lifi,Christina Hendricks 3"). Only segments
    // AFTER the first can be the title: a lone "John Smith" reads as
    // photographer credit (legitimate), while a trailing bare name
    // reads as subject identity (selection skips it).
    // Single-segment strings therefore pass — residual misses stay
    // visible in the logged author string for human curation.
    let segments: Vec<&str> = s.split(',').collect();
    segments.iter().skip(1).any(|segment| {
        let words: Vec<&str> = segment.split_whitespace().collect();
        let words = if words.len() == 3 && words[2].chars().all(|c| c.is_ascii_digit()) {
            &words[..2]
        } else {
            &words[..]
        };
        words.len() == 2
            && words.iter().all(|w| {
                let mut cs = w.chars();
                matches!(cs.next(), Some(c) if c.is_uppercase()) && cs.all(|c| c.is_lowercase())
            })
    })
}

/// Rank researched plates for delivery: title/query word overlap
/// first (the plate's own words matching the prompt's words), frame
/// area as tiebreak. Name-titled plates are excluded (logged) —
/// identified people are a human decision, never automatic. Pure
/// text matching over titles: identical machinery for a cat, a
/// person, or a tree — no subject branches anywhere.
/// Returns delivery order as indices. Pure over inputs.
pub fn rank_for_delivery(
    plates: &[SourcedPlate],
    query_words: &[String],
    log: &mut Vec<String>,
) -> Vec<usize> {
    let mut scored: Vec<(usize, usize, u64)> = Vec::new();
    for (i, p) in plates.iter().enumerate() {
        if looks_like_person_name(&p.provenance.author) {
            log.push(format!(
                "skipped name-titled plate: {:?}",
                p.provenance.author
            ));
            continue;
        }
        let title_words = evidence_words(&p.title);
        let overlap = query_words
            .iter()
            .filter(|w| title_words.contains(w))
            .count();
        let area = (p.image.width as u64) * (p.image.height as u64);
        scored.push((i, overlap, area));
    }
    scored.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
    scored.into_iter().map(|(i, _, _)| i).collect()
}

/// A sourced photograph: the winning plate's own bytes plus the
/// provenance the license audit needs. Labeled exactly what it is —
/// a real photograph by a real photographer, found by research, not
/// generated.
#[derive(Debug, Clone)]
pub struct SourcedPhoto {
    pub image: Image,
    pub plate_index: usize,
    pub author: String,
    pub license: String,
    pub page_url: String,
    pub title: String,
}

/// Deliver from already-sourced plates (no network): rank by title
/// overlap with the prompt's words, hand over the winner's own bytes
/// with its provenance. Pure over its inputs — the offline-testable
/// core. Refuses (naming the shortfall) when nothing is deliverable.
pub fn deliver_from_plates(
    brief: &Brief,
    plates: &[SourcedPlate],
) -> Result<(SourcedPhoto, Vec<String>), String> {
    let mut log = vec![format!(
        "brief: {:?} {:?} {:?}",
        brief.subject, brief.pose, brief.props
    )];
    log.push(format!(
        "research-only: {} plate(s) — ranking by title overlap, never rendering",
        plates.len()
    ));
    let query_words = evidence_words(&brief.raw);
    let order = rank_for_delivery(plates, &query_words, &mut log);
    let idx = order.into_iter().next().ok_or_else(|| {
        "no deliverable photograph — every plate refused (name-titled or absent)".to_string()
    })?;
    let plate = &plates[idx];
    log.push(format!(
        "delivered: plate {} by {} ({}) — sourced photograph, not generated",
        idx, plate.provenance.author, plate.provenance.license
    ));
    log.push(format!("provenance: {}", plate.provenance.page_url));
    Ok((
        SourcedPhoto {
            image: plate.image.clone(),
            plate_index: idx,
            author: plate.provenance.author.clone(),
            license: plate.provenance.license.clone(),
            page_url: plate.provenance.page_url.clone(),
            title: plate.title.clone(),
        },
        log,
    ))
}

/// Place word in prose, if any: backdrops and query anchors come
/// from the scene's own nouns, never from a default list consulted
/// blindly. Small closed set (product decision); unknown places
/// ride the generic landscape fallback downstream.
pub fn place_word(prose: &str) -> Option<&'static str> {
    let lower = prose.to_lowercase();
    ["river", "mountain", "lake", "ocean", "forest", "desert"]
        .into_iter()
        .find(|w| lower.contains(w))
}

/// Subject query for research. People take pose-portraits ("waving
/// woman portrait"); generic subjects take noun queries with their
/// place ("elephant river") and NEVER pose words — "crossing"
/// drifts web search into road signs and game wikis, measured live.
/// Pure over the brief: testable without network.
pub fn build_subject_query(
    brief: &Brief,
    subject_word: &str,
    generic: bool,
    place: Option<&str>,
) -> String {
    if generic {
        return match place {
            Some(p) => format!("{} {}", subject_word, p),
            // No pose words, no habitat words: the noun photographs
            // itself. ("Wildlife" editorializing was wrong for laps,
            // laps, and living rooms — removed.)
            None => format!("{} photograph", subject_word),
        };
    }
    // The pose word always rides along ("standing man portrait"):
    // it is the prompt's own word shaping search, identical machinery
    // for every pose and every subject.
    let mut query = format!("{} {} portrait", brief.pose, subject_word);
    if brief.props.contains(&PropKind::Roses) {
        query = format!("{} with roses", query);
    }
    if brief.props.contains(&PropKind::Flag) && !query.contains("flag") {
        query = format!("{} with flag", query);
    }
    query
}

/// Search-query hygiene: the safety review refuses titles containing
/// minor words ("girl", "boy", ...), so searching those words verbatim
/// only harvests refusals. Rewrite to the adult equivalent for search —
/// the Open Images Girl/Boy exclusion + title review still keep actual
/// minors out of the library. Pure string rewrite, logged by callers.
fn sanitize_search_query(prose: &str) -> String {
    let mut q = prose.to_lowercase();
    for (from, to) in [
        ("girls", "women"),
        ("girl", "woman"),
        ("boys", "men"),
        ("boy", "man"),
    ] {
        q = q.replace(from, to);
    }
    q
}

/// Coverage check: which scene requirements have evidence behind
/// them. Pure over (spec, evidence words): a requirement is covered
/// when its own words appear in the round's queries or plate titles.
/// Uncovered requirements are gaps that fund another research round —
/// never a refusal by themselves. Action words match their pose
/// forms too ("sit" rides on "sitting").
pub fn verify_coverage(spec: &super::scene_intent::SceneSpec, evidence: &[String]) -> Vec<String> {
    let has = |w: &str| evidence.iter().any(|e| e == w);
    let mut gaps = Vec::new();
    for s in &spec.subjects {
        if !has(&s.stype) && !s.attributes.iter().any(|a| has(a)) {
            gaps.push(format!("subject:{}", s.stype));
        }
    }
    for o in &spec.objects {
        if !has(&o.otype) {
            gaps.push(format!("object:{}", o.id));
        }
    }
    for a in &spec.actions {
        let forms: &[&str] = match a.atype.as_str() {
            "salute" => &["salute", "saluting"],
            "wave" => &["wave", "waving"],
            "stand" => &["stand", "standing"],
            "sit" => &["sit", "sitting"],
            "cross" => &["cross", "crossing"],
            "peace_sign" => &["peace", "peace_sign"],
            _ => &[],
        };
        if !has(&a.atype) && !forms.iter().any(|w| has(w)) {
            gaps.push(format!("action:{}", a.atype));
        }
    }
    gaps
}

/// Refined queries from coverage gaps: action gaps pair the pose
/// with the subject for people ("sitting cat" finds sitters), but
/// NEVER for generic subjects — pose words drift search into road
/// signs and game wikis (measured live: "crossing elephant" won a
/// road sign). Generic actions re-ask noun + place instead. Subject
/// gaps re-ask the noun as a photograph, object gaps ask the object.
/// Capped at three, deduped, deterministic — the funded second round
/// asks exactly what the first round failed to show.
pub fn gap_queries(
    gaps: &[String],
    subject_word: &str,
    pose: &str,
    generic: bool,
    place: Option<&str>,
) -> Vec<String> {
    let mut out = Vec::new();
    for g in gaps {
        if out.len() >= 3 {
            break;
        }
        if let Some(action) = g.strip_prefix("action:") {
            let _ = action;
            let q = if generic {
                match place {
                    Some(p) => format!("{} {}", subject_word, p),
                    None => format!("{} photograph", subject_word),
                }
            } else if pose != "standing" {
                format!("{} {}", pose, subject_word)
            } else {
                format!("{} photograph", subject_word)
            };
            if !out.contains(&q) {
                out.push(q);
            }
        } else if let Some(noun) = g.strip_prefix("subject:") {
            let q = format!("{} photograph", noun);
            if !out.contains(&q) {
                out.push(q);
            }
        } else if let Some(id) = g.strip_prefix("object:") {
            let noun = id.split('_').next().unwrap_or(id);
            let q = noun.to_string();
            if !out.contains(&q) {
                out.push(q);
            }
        }
    }
    out
}

/// Lowercase content words of a string: the evidence vocabulary.
/// Queries, titles, and authors all reduce to this for coverage.
fn evidence_words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// Usable OpenImages plates: name-titled plates name an identified
/// person — delivery skips them (logged) while the library keeps
/// them listed. No box geometry is read: the whole plate image is
/// the candidate, whatever the box says. Returns usable indices.
fn usable_openimages(
    oi_plates: &[super::plates::openimages::OpenPlate],
    log: &mut Vec<String>,
) -> Vec<usize> {
    oi_plates
        .iter()
        .enumerate()
        .filter_map(|(i, hit)| {
            if looks_like_person_name(&hit.plate.provenance.author) {
                log.push(format!(
                    "skipped name-titled plate: {:?}",
                    hit.plate.provenance.author
                ));
                return None;
            }
            Some(i)
        })
        .collect()
}

/// The full trajectory: research the brief, rank the plates,
/// deliver the winner's own bytes. One command, full receipts.
/// Per-request training, narrow scope: research only what this prompt
/// needs (subject + pose + props) — grounded, never from noise,
/// never generated.
/// Two bounded rounds: round 1 researches the brief; coverage gaps
/// fund round 2 with refined queries; the round closing more gaps
/// wins (ties keep round 1). Refusal is never terminal while gaps
/// name what is missing — the loop only stops at two rounds.
pub async fn imagine(
    prose: &str,
    _width: u32,
    _height: u32,
) -> Result<(SourcedPhoto, Vec<String>), Vec<String>> {
    let brief = parse_brief(prose);
    let mut log = vec![format!(
        "brief: {:?} {:?} {:?}",
        brief.subject, brief.pose, brief.props
    )];
    let scene = super::scene_intent::parse_scene(prose);
    let scene_word = scene
        .subjects
        .iter()
        .find(|s| !matches!(s.stype.as_str(), "man" | "woman" | "human"))
        .map(|s| s.stype.clone());
    let (subject_word, generic) = match (&brief.subject, scene_word) {
        (SubjectKind::Woman, _) => ("woman".to_string(), false),
        (SubjectKind::Man, _) => ("man".to_string(), false),
        (SubjectKind::Person, Some(w)) => (w, true),
        (SubjectKind::Person, None) => ("person".to_string(), false),
    };
    let place = place_word(prose);
    // All subjects are researched for candidacy — the main noun plus
    // any companions the scene names. Nobody is mounted or rendered.
    let extra_nouns: Vec<String> = scene
        .subjects
        .iter()
        .map(|s| s.stype.clone())
        .filter(|s| s != &subject_word)
        .collect();
    // Round 1: research the brief.
    let (o1, e1) = attempt(
        prose,
        &brief,
        &subject_word,
        generic,
        place,
        extra_nouns.clone(),
        None,
    )
    .await;
    let g1 = verify_coverage(&scene, &e1);
    let (mut log1, ok1) = match o1 {
        Ok((img, l)) => (l, Some(img)),
        Err(l) => (l, None),
    };
    log.append(&mut log1);
    if g1.is_empty() {
        log.push("verified: full coverage — no second round".to_string());
        return match ok1 {
            Some(img) => Ok((img, log)),
            None => Err(log),
        };
    }
    // Round 2, funded by the gaps round 1 left open.
    let gq = gap_queries(&g1, &subject_word, &brief.pose, generic, place);
    log.push(format!("round 2 (gaps reach further): queries {:?}", gq));
    let (o2, e2) = attempt(
        prose,
        &brief,
        &subject_word,
        generic,
        place,
        extra_nouns,
        Some(gq),
    )
    .await;
    let g2 = verify_coverage(&scene, &e2);
    let (mut log2, ok2) = match o2 {
        Ok((img, l)) => (l, Some(img)),
        Err(l) => (l, None),
    };
    // Lower score wins: (open gaps, failed). Ties keep round 1.
    let s1 = (g1.len(), ok1.is_none() as usize);
    let s2 = (g2.len(), ok2.is_none() as usize);
    if s2 < s1 {
        log.push(format!(
            "round 2 wins: gaps {}→{} ({})",
            g1.len(),
            g2.len(),
            g2.join(", ")
        ));
        log.append(&mut log2);
        match ok2 {
            Some(img) => Ok((img, log)),
            None => Err(log),
        }
    } else {
        log.push(format!(
            "round 1 stands: gaps {} vs {} ({})",
            g1.len(),
            g2.len(),
            g1.join(", ")
        ));
        log.append(&mut log2);
        match ok1 {
            Some(img) => Ok((img, log)),
            None => Err(log),
        }
    }
}

/// One research round: research every named subject as
/// candidates, rank them, deliver the winner. Override queries
/// replace the default subject query (tried in order until plates
/// land). Returns the outcome plus evidence words for coverage
/// checking. Round 1 passes None and researches the brief. No
/// copying into new pixels, no montage, no rendering: the winning
/// plate's own bytes, labeled sourced.
#[allow(clippy::too_many_arguments)]
async fn attempt(
    prose: &str,
    brief: &Brief,
    subject_word: &str,
    generic: bool,
    place: Option<&str>,
    extra_nouns: Vec<String>,
    override_queries: Option<Vec<String>>,
) -> (
    Result<(SourcedPhoto, Vec<String>), Vec<String>>,
    Vec<String>,
) {
    let mut log = Vec::new();
    let mut evidence = Vec::new();
    // Intent first: WHAT the prose asks, WHAT research satisfies it,
    // WHICH subjects need it. Fetching below serves these intents.
    let researched = super::intent_research::research_intent(prose);
    for l in super::intent_research::log_lines(&researched) {
        log.push(l);
    }
    // Every named subject is researched as a candidate — the main
    // noun plus any companions the scene names. Nobody is rendered.
    let mut all_nouns = vec![subject_word.to_string()];
    for n in &extra_nouns {
        if !all_nouns.contains(n) {
            all_nouns.push(n.clone());
        }
    }
    log.push(format!("subjects researched: {:?}", all_nouns));
    let default_query =
        sanitize_search_query(&build_subject_query(brief, subject_word, generic, place));
    // Override queries (round 2) tried in order until plates land;
    // round 1 researches the default query.
    let subject_queries = override_queries.unwrap_or_else(|| vec![default_query]);
    let mut plates = Vec::new();
    for query in &subject_queries {
        log.push(format!("research: query {:?}", query));
        evidence.extend(evidence_words(query));
        let (commons_plates, commons_refused) = super::plates::source_plates(query, 10).await;
        for r in &commons_refused {
            log.push(format!("commons refused: {}", r));
        }
        let (mut web_plates, refused) = super::plates::source_plates_web(query, 10).await;
        for r in &refused {
            log.push(format!("refused: {}", r));
        }
        let mut all = commons_plates;
        all.append(&mut web_plates);
        plates = all;
        if !plates.is_empty() {
            break;
        }
    }
    // Companion nouns research their own candidacy too.
    for noun in all_nouns.iter().skip(1).take(2) {
        let q = format!("{} photograph", noun);
        log.push(format!("research: query {:?}", q));
        evidence.extend(evidence_words(&q));
        let (more, refused) = super::plates::source_plates(&q, 3).await;
        for r in &refused {
            log.push(format!("companion refused: {}", r));
        }
        for p in &more {
            evidence.extend(evidence_words(&p.title));
        }
        plates.extend(more);
        if plates.len() >= 20 {
            break;
        }
    }
    // Stock-photo queries wall off entirely; fall back to plain
    // portraiture once rather than study nothing. Logged either way.
    if plates.is_empty() {
        log.push("research: fallback query \"portrait\"".to_string());
        let (fallback, refused2) = super::plates::source_plates_web("portrait", 10).await;
        for r in &refused2 {
            log.push(format!("refused: {}", r));
        }
        plates = fallback;
    }
    for p in &plates {
        evidence.extend(evidence_words(&p.title));
    }
    log.push(format!(
        "references: {} plate(s) — candidates, never altered",
        plates.len()
    ));
    // Open Images is a people-only index: for non-person subjects it
    // would return strangers' photos as false candidates, so the
    // sweep skips it with the reason stated. For people, usable
    // plates join as candidates whole — boxes unread, bytes as-is.
    let oi_cache = std::path::Path::new(".grounding/openimages");
    let oi_query = sanitize_search_query(prose);
    let (oi_plates, oi_refused) = if generic {
        log.push("openimages: skipped (people-only index, non-person subject)".to_string());
        (Vec::new(), Vec::new())
    } else {
        super::plates::openimages::search_openimages(oi_cache, &oi_query, 10).await
    };
    for r in &oi_refused {
        log.push(format!("openimages refused: {}", r));
    }
    let usable = usable_openimages(&oi_plates, &mut log);
    log.push(format!(
        "openimages: {} plate(s), {} usable candidate(s)",
        oi_plates.len(),
        usable.len()
    ));
    let mut all_plates = plates;
    for idx in usable.iter().take(8) {
        let hit = &oi_plates[*idx];
        evidence.extend(evidence_words(&hit.plate.title));
        all_plates.push(hit.plate.clone());
    }
    // Rank, then deliver the winner's own bytes.
    match deliver_from_plates(brief, &all_plates) {
        Ok((photo, mut chain)) => {
            log.append(&mut chain);
            (Ok((photo, log)), evidence)
        }
        Err(e) => {
            log.push(format!("refused: {}", e));
            (Err(log), evidence)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::plates::{PlateProvenance, SourcedPlate};
    use super::super::vision::Rgb;
    use super::*;

    #[test]
    fn brief_parses_plain_words() {
        // Poses are query words now, not geometry: they shape research.
        let b = parse_brief("man holding peace sign");
        assert_eq!(b.subject, SubjectKind::Man);
        assert_eq!(b.pose, "peace sign");
        assert!(b.props.is_empty());
        let b = parse_brief("woman with roses");
        assert_eq!(b.subject, SubjectKind::Woman);
        assert_eq!(b.pose, "standing");
        assert_eq!(b.props, vec![PropKind::Roses]);
        let b = parse_brief("person waving");
        assert_eq!(b.pose, "waving");
        let b = parse_brief("do the thing");
        assert_eq!(
            (b.subject, b.pose.as_str()),
            (SubjectKind::Person, "standing")
        );
    }

    #[test]
    fn brief_parses_salute_flag_and_girl() {
        // The failing prompt: girl saluting a flag must reach research
        // as an adult woman saluting with a flag — never as a bare
        // "portrait" with the pose and prop dropped.
        let b = parse_brief("girl saluting a flag");
        assert_eq!(b.subject, SubjectKind::Woman);
        assert_eq!(b.pose, "saluting");
        assert!(b.props.contains(&PropKind::Flag));
        let b = parse_brief("a woman saluting");
        assert_eq!(
            (b.subject, b.pose.as_str()),
            (SubjectKind::Woman, "saluting")
        );
    }

    #[test]
    fn search_queries_rewrite_minor_words() {
        // Safety hygiene: search text rewrites minor words to adult
        // equivalents so queries don't only harvest title refusals.
        // The Girl/Boy image exclusion still applies downstream.
        assert_eq!(sanitize_search_query("girl saluting"), "woman saluting");
        assert_eq!(sanitize_search_query("boy standing"), "man standing");
        assert_eq!(sanitize_search_query("woman portrait"), "woman portrait");
    }

    #[test]
    fn brief_parses_generic_subjects() {
        // Open vocabulary: an elephant stays an elephant (never a
        // "person"), crossing is the pose, and the river shapes the
        // backdrop search downstream.
        let b = parse_brief("an elephant crossing a river");
        assert_eq!(b.subject, SubjectKind::Person);
        assert_eq!(b.pose, "crossing");
        let b = parse_brief("person waving");
        assert_eq!(b.pose, "waving");
        let b = parse_brief("cat sitting in human's lap");
        assert_eq!(b.pose, "sitting");
    }

    #[test]
    fn name_titled_plates_skip_delivery() {
        use super::super::plates::openimages::OpenPlate;
        // Name-titled plates name an identified person — delivery
        // skips them (logged); everything else is a candidate whole.
        // No box geometry is read, for any subject.
        let mk = |author: &str| OpenPlate {
            plate: SourcedPlate {
                image: Image::blank(100, 100, Rgb::new(10, 10, 10)),
                provenance: PlateProvenance {
                    source_url: "s".to_string(),
                    page_url: "s".to_string(),
                    author: author.to_string(),
                    license: "t".to_string(),
                },
                basis: "t".to_string(),
                title: "t".to_string(),
            },
            bbox: (0.0, 0.0, 1.0, 1.0),
            label: "x".to_string(),
            total_boxes: 2,
        };
        let plates = vec![
            mk("lifrita lifi,Christina Hendricks 3"), // named: out
            mk("plain author"),                       // clean: usable
        ];
        let mut log = Vec::new();
        assert_eq!(usable_openimages(&plates, &mut log), vec![1]);
        assert!(!log.is_empty(), "skips must be logged");
    }

    #[test]
    fn subject_queries_use_scene_not_pose_for_generics() {
        // "crossing" in a web query harvests road signs (measured
        // live): generic subjects query noun + place, never the pose.
        let b = parse_brief("an elephant crossing a river");
        assert_eq!(
            build_subject_query(&b, "elephant", true, Some("river")),
            "elephant river"
        );
        assert_eq!(
            build_subject_query(&b, "elephant", true, None),
            "elephant photograph"
        );
        // People keep pose-portraits.
        let w = parse_brief("woman waving");
        assert_eq!(
            build_subject_query(&w, "woman", false, None),
            "waving woman portrait"
        );
        assert_eq!(place_word("an elephant crossing a river"), Some("river"));
        assert_eq!(place_word("a man standing"), None);
    }

    #[test]
    fn coverage_gaps_fund_refined_queries() {
        use super::super::scene_intent::parse_scene;
        // Full evidence: no gaps, no second round.
        let spec = parse_scene("An elephant crossing a river.");
        let ev: Vec<String> = ["elephant", "crossing", "river"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(verify_coverage(&spec, &ev).is_empty());
        // Cat photo only: human, lap, and sitting stay open.
        let spec = parse_scene("Cat sitting in human's lap.");
        let ev: Vec<String> = ["cat", "photograph"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let gaps = verify_coverage(&spec, &ev);
        assert!(gaps.iter().any(|g| g.contains("human")), "{:?}", gaps);
        assert!(gaps.iter().any(|g| g.contains("lap")), "{:?}", gaps);
        assert!(gaps.iter().any(|g| g.contains("sit")), "{:?}", gaps);
        assert!(!gaps.iter().any(|g| g.contains("cat")), "{:?}", gaps);
        // People pair pose + subject; generic subjects re-ask noun
        // (+ place), never the pose word that harvests road signs.
        let qs = gap_queries(&gaps, "cat", "sitting", true, None);
        assert!(qs.len() <= 3 && !qs.is_empty());
        assert!(
            qs.iter()
                .any(|q| q.contains("cat") && !q.contains("sitting")),
            "{:?}",
            qs
        );
        assert!(qs.iter().any(|q| q.contains("human")), "{:?}", qs);
        let people = gap_queries(&gaps, "woman", "sitting", false, None);
        assert!(
            people
                .iter()
                .any(|q| q.contains("sitting") && q.contains("woman")),
            "{:?}",
            people
        );
    }

    fn titled_plate(title: &str, w: u32, h: u32) -> SourcedPlate {
        SourcedPlate {
            image: Image::blank(w, h, Rgb::new(60, 80, 120)),
            provenance: PlateProvenance {
                source_url: "synthetic".to_string(),
                page_url: "synthetic".to_string(),
                author: "test".to_string(),
                license: "test".to_string(),
            },
            basis: "test".to_string(),
            title: title.to_string(),
        }
    }

    #[test]
    fn rank_prefers_title_overlap_then_size() {
        // Same generic machinery for any subject: title words
        // matching the prompt win; ties break by frame area.
        let plates = vec![
            titled_plate("desert dunes", 200, 200),
            titled_plate("elephant river crossing", 120, 160),
            titled_plate("elephant river", 100, 100),
        ];
        let query = evidence_words("an elephant crossing a river");
        let mut log = Vec::new();
        let order = rank_for_delivery(&plates, &query, &mut log);
        assert_eq!(order, vec![1, 2, 0], "overlap then size: {:?}", order);
    }

    #[test]
    fn delivery_hands_over_winner_bytes_with_provenance() {
        // The product is the winning plate's own bytes, labeled
        // sourced — byte-identical to that donor, never rendered.
        let plates = vec![
            titled_plate("desert dunes", 120, 160),
            titled_plate("standing man portrait", 120, 160),
        ];
        let brief = parse_brief("a man standing");
        let (photo, log) = deliver_from_plates(&brief, &plates).expect("delivers");
        assert_eq!(photo.title, "standing man portrait");
        assert_eq!((photo.image.width, photo.image.height), (120, 160));
        assert!(
            log.iter()
                .any(|l| l.contains("sourced photograph, not generated")),
            "{:?}",
            log
        );
        // Delivered bytes are the winner's own bytes.
        let donor = &plates[photo.plate_index];
        assert_eq!(donor.title, photo.title);
        for (x, y) in [(10, 10), (60, 70), (100, 140)] {
            assert_eq!(photo.image.get(x, y), donor.image.get(x, y));
        }
        // With no plates at all: honest refusal, never fabrication.
        let err = deliver_from_plates(&brief, &[]).expect_err("must refuse");
        assert!(err.contains("no deliverable photograph"), "{:?}", err);
    }

    #[test]
    fn name_titles_skip_but_credits_pass() {
        assert!(looks_like_person_name("lifrita lifi,Christina Hendricks 3"));
        assert!(!looks_like_person_name("Bob Park,ADP-AHA Golf 012"));
        assert!(!looks_like_person_name("Aaron Shikler"));
        assert!(!looks_like_person_name("John Smith"));
        assert!(!looks_like_person_name("beach portrait sunset"));
        assert!(!looks_like_person_name(""));
    }
}
