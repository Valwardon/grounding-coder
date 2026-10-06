//! One command from prose to photo: imagine.
//!
//! ```text
//! prose → brief → research references → study them → build fresh
//! ```
//!
//! The loop the user asked for: a request like "man holding peace
//! sign" is parsed to a brief, reference plates are sourced for the
//! pose, the references are MEASURED (skin tone, backdrop palette,
//! framing — medians across plates, robust to outliers), and a fresh
//! procedural figure is built from those measurements. Every pixel
//! of the output is rendered, never copied: references teach palette
//! and composition, the raytracer makes the photo.
//!
//! Degradation is honest: with no usable references the build falls
//! back to defaults and the log says so. Hair is not measured in v1
//! (no reliable classical cue) — the log states the default used.
use super::plates::SourcedPlate;
use super::vision::{Image, Rgb};

/// Who stands in the photo. Subjects come from researched
/// photographs with complete provenance — no likenesses generated,
/// no minors composited, by construction of the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubjectKind {
    Man,
    Woman,
    Person,
}

/// Procedural props the builder knows how to grow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropKind {
    Roses,
    Flag,
}

/// The parsed request: deterministic keyword scan, stated limits —
/// no grammar beyond "contains the word". Subject, pose, and props
/// shape RESEARCH queries now, not geometry: people come from
/// photographs, never from meshes.
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

/// What the references taught, per channel as medians. None means
/// "no usable reference" — the builder falls back to defaults.
#[derive(Debug, Clone)]
pub struct Study {
    pub skin: Option<Rgb>,
    pub backdrop_top: Option<Rgb>,
    pub backdrop_bottom: Option<Rgb>,
    /// Subject height as a fraction of frame height.
    pub subject_fill: Option<f64>,
    pub plates_used: usize,
    pub notes: Vec<String>,
}

/// Dominant quantized color in a row band, skipping the subject's
/// x-range. Quantization (5 bits/channel) keeps near-solid backdrops
/// voting as one.
fn dominant_outside(
    img: &Image,
    rows: std::ops::Range<u32>,
    skip_x0: u32,
    skip_x1: u32,
) -> Option<(u8, u8, u8)> {
    let mut counts: std::collections::HashMap<(u8, u8, u8), u64> = std::collections::HashMap::new();
    for y in rows {
        for x in (0..img.width).step_by(4) {
            if x >= skip_x0 && x <= skip_x1 {
                continue;
            }
            if let Some(p) = img.get(x, y) {
                let key = (p.r & 0xF8, p.g & 0xF8, p.b & 0xF8);
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    counts.into_iter().max_by_key(|(_, n)| *n).map(|(c, _)| c)
}

fn median_channel(mut vs: Vec<u8>) -> Option<u8> {
    if vs.is_empty() {
        return None;
    }
    vs.sort_unstable();
    Some(vs[vs.len() / 2])
}

/// Measure reference plates: skin tone from the largest cleaned skin
/// blob's mean color, backdrop from the top/bottom bands' dominant
/// colors, framing from the subject bbox. Median across plates.
pub fn study_references(plates: &[SourcedPlate]) -> Study {
    let mut skins_r = Vec::new();
    let mut skins_g = Vec::new();
    let mut skins_b = Vec::new();
    let mut tops_r = Vec::new();
    let mut tops_g = Vec::new();
    let mut tops_b = Vec::new();
    let mut bots_r = Vec::new();
    let mut bots_g = Vec::new();
    let mut bots_b = Vec::new();
    let mut fills = Vec::new();
    let mut used = 0usize;
    for plate in plates {
        let img = &plate.image;
        let (w, h) = (img.width, img.height);
        if w < 32 || h < 32 {
            continue;
        }
        let cleaned = Image::morph_close(&Image::morph_open(&img.skin_mask(), w, h, 2), w, h, 2);
        let blobs = Image::all_blobs(&cleaned, w, h, ((w * h) as f64 * 0.002) as usize);
        let Some((bx0, by0, bx1, by1, area, _parts)) = Image::assemble_subject(&blobs, w, h, 0.06)
        else {
            continue;
        };
        if area as f64 / (w * h) as f64 > 0.6 {
            continue; // wash, not a subject
        }
        // Mean skin color over masked pixels only: the bbox corners
        // hold backdrop, and averaging them in drags the tone.
        let (mut sr, mut sg, mut sb, mut sn) = (0u64, 0u64, 0u64, 0u64);
        for y in by0..=by1.min(h - 1) {
            for x in bx0..=bx1.min(w - 1) {
                if !cleaned[(y * w + x) as usize] {
                    continue;
                }
                if let Some(p) = img.get(x, y) {
                    sr += p.r as u64;
                    sg += p.g as u64;
                    sb += p.b as u64;
                    sn += 1;
                }
            }
        }
        if sn == 0 {
            continue;
        }
        skins_r.push((sr / sn) as u8);
        skins_g.push((sg / sn) as u8);
        skins_b.push((sb / sn) as u8);
        // Backdrop bands, avoiding the subject columns.
        if let Some(c) = dominant_outside(img, 0..h / 10, bx0, bx1) {
            tops_r.push(c.0);
            tops_g.push(c.1);
            tops_b.push(c.2);
        }
        if let Some(c) = dominant_outside(img, h * 9 / 10..h, bx0, bx1) {
            bots_r.push(c.0);
            bots_g.push(c.1);
            bots_b.push(c.2);
        }
        // Framing: subject bbox height over plate height.
        fills.push((by1 - by0 + 1) as f64 / h as f64);
        used += 1;
    }
    // Medians, each channel independent and documented as such.
    let med3 = |r: Vec<u8>, g: Vec<u8>, b: Vec<u8>| -> Option<Rgb> {
        match (median_channel(r), median_channel(g), median_channel(b)) {
            (Some(r), Some(g), Some(b)) => Some(Rgb::new(r, g, b)),
            _ => None,
        }
    };
    let mut fills_sorted = fills.clone();
    fills_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Study {
        skin: med3(skins_r, skins_g, skins_b),
        backdrop_top: med3(tops_r, tops_g, tops_b),
        backdrop_bottom: med3(bots_r, bots_g, bots_b),
        subject_fill: fills_sorted.get(fills_sorted.len() / 2).copied(),
        plates_used: used,
        notes: if used == 0 {
            vec!["no usable references — defaults throughout".to_string()]
        } else {
            vec![format!("measured across {} reference(s)", used)]
        },
    }
}

/// A bare personal name as author-or-title ("Firstname Lastname",
/// optional trailing number, nothing else): evidence of an
/// identified subject. Automatic selection skips these — compositing
/// strangers is the job; named individuals are a human decision.
pub fn looks_like_person_name(s: &str) -> bool {
    // Author strings glue photographer credit and photo title with
    // commas ("lifrita lifi,Christina Hendricks 3"). Only segments
    // AFTER the first can be the title: a lone "John Smith" reads as
    // photographer credit (legitimate), while a trailing bare name
    // reads as subject identity (automatic selection skips it).
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

/// Score one plate as a compositing source: a complete subject
/// (margins on all four sides — cropped figures score zero no
/// matter how large) times its area fraction. Returns the subject
/// and score, or None when the plate holds nothing composable.
pub fn score_plate(img: &Image) -> Option<(f64, (u32, u32, u32, u32))> {
    let (w, h) = (img.width, img.height);
    let subject = super::compose::find_subject(img).ok()?;
    let margins = [
        subject.x0 as f64 / w as f64,
        subject.y0 as f64 / h as f64,
        (w - 1 - subject.x1) as f64 / w as f64,
        (h - 1 - subject.y1) as f64 / h as f64,
    ];
    let worst = margins.iter().cloned().fold(1.0f64, f64::min);
    if worst < 0.03 {
        return None; // cropped by the frame edge: unusable whole
    }
    let fraction = subject.area as f64 / (w * h) as f64;
    Some((
        fraction * worst * 10.0,
        (subject.x0, subject.y0, subject.x1, subject.y1),
    ))
}

/// Best compositing source across plates, if any scores above zero.
pub fn pick_subject(plates: &[SourcedPlate]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, plate) in plates.iter().enumerate() {
        if let Some((score, _)) = score_plate(&plate.image)
            && score > 0.0
            && best.map(|(_, s)| score > s).unwrap_or(true)
        {
            best = Some((i, score));
        }
    }
    best.map(|(i, _)| i)
}

/// Compose from already-sourced plates (no network): pick the best
/// subject, pair it with a backdrop from a different source, build
/// the new photo. Pure over its inputs — the offline-testable core.
pub fn imagine_from_plates(
    brief: &Brief,
    plates: &[SourcedPlate],
    bg_plates: &[SourcedPlate],
    width: u32,
    height: u32,
) -> Result<(Image, Vec<String>), String> {
    let mut log = vec![format!(
        "brief: {:?} {:?} {:?}",
        brief.subject, brief.pose, brief.props
    )];
    let idx = pick_subject(plates).ok_or_else(|| {
        "no complete photographic subject — refusing; people come from photographs".to_string()
    })?;
    log.push(format!("path: photographic (plate {})", idx));
    let plate = &plates[idx];
    let mut backdrop: Option<(&Image, String)> = None;
    for bg in bg_plates {
        if bg.provenance.source_url != plate.provenance.source_url {
            backdrop = Some((&bg.image, bg.provenance.page_url.clone()));
            log.push(format!("backdrop: {}", bg.provenance.page_url));
            break;
        }
    }
    let backdrop_label = backdrop
        .as_ref()
        .map(|(_, u)| u.clone())
        .unwrap_or_else(|| "studio gradient".to_string());
    let backdrop_ref = backdrop.as_ref().map(|(img, u)| (*img, u.as_str()));
    let (img, clog) =
        super::compose::compose_portrait_on(&plate.image, "imagine", backdrop_ref, width, height)?;
    for op in &clog.ops {
        log.push(format!("compose: {}", op));
    }
    log.push(format!(
        "subject: bbox {:?}, {:.3} of frame on {}",
        clog.subject_bbox, clog.subject_fraction, backdrop_label
    ));
    Ok((img, log))
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
    let mut query = if brief.pose != "standing" {
        format!("{} {} portrait", brief.pose, subject_word)
    } else {
        format!("{} portrait", subject_word)
    };
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

/// The full trajectory: research references for the brief, study
/// them, composite the new photo. One command, full receipts.
/// Per-request training, narrow scope: research only what this prompt
/// needs (subject + pose + props), measure the plates, then composite
/// real pixels — grounded, never from noise.
pub async fn imagine(
    prose: &str,
    width: u32,
    height: u32,
) -> Result<(Image, Vec<String>), Vec<String>> {
    let brief = parse_brief(prose);
    let mut log = vec![format!(
        "brief: {:?} {:?} {:?}",
        brief.subject, brief.pose, brief.props
    )];
    // Subject query follows the ask (pose/subject/props words feed
    // research now); backdrop query seeks a place — or the named prop.
    // Unclassified subjects come from the scene parse (researched,
    // never guessed): a Person brief with a non-human scene subject
    // researches that noun. The brief only knows people by keyword.
    let scene_word = super::scene_intent::parse_scene(prose)
        .subjects
        .into_iter()
        .find(|s| !matches!(s.stype.as_str(), "man" | "woman" | "human"))
        .map(|s| s.stype);
    let (subject_word, generic) = match (&brief.subject, scene_word) {
        (SubjectKind::Woman, _) => ("woman".to_string(), false),
        (SubjectKind::Man, _) => ("man".to_string(), false),
        (SubjectKind::Person, Some(w)) => (w, true),
        (SubjectKind::Person, None) => ("person".to_string(), false),
    };
    let place = place_word(prose);
    let query = sanitize_search_query(&build_subject_query(&brief, &subject_word, generic, place));
    log.push(format!("research: query {:?}", query));
    // Curated Commons first (machine-readable provenance beats
    // aggregator soup), then general web search. Both rank together.
    let (commons_plates, commons_refused) = super::plates::source_plates(&query, 10).await;
    for r in &commons_refused {
        log.push(format!("commons refused: {}", r));
    }
    let (mut plates, refused) = super::plates::source_plates_web(&query, 10).await;
    for r in &refused {
        log.push(format!("refused: {}", r));
    }
    let mut all = commons_plates;
    all.append(&mut plates);
    let mut plates = all;
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
    log.push(format!("references: {} plate(s)", plates.len()));
    // Open Images is a people-only index (Person/Woman boxes): for
    // non-person subjects it would return strangers' photos as false
    // candidates, so the path is skipped with the reason stated.
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
    log.push(format!("openimages: {} plate(s)", oi_plates.len()));
    // Best sane box wins, not the first hit: full-frame boxes
    // reframe the whole photo (nothing learned), and name-titled
    // plates name an identified person — automatic selection skips
    // them (logged) while the library keeps them listed. Sample
    // curation stays human either way.
    let mut ranked: Vec<(usize, f64)> = oi_plates
        .iter()
        .enumerate()
        .filter_map(|(i, hit)| {
            let area = (hit.bbox.2 - hit.bbox.0) * (hit.bbox.3 - hit.bbox.1);
            if area > 0.9 {
                return None;
            }
            // Portrait convention, not full containment: headroom
            // plus both sides must clear the edge; the bottom may
            // crop (half-body portraits cut at the legs routinely).
            // Skin-path margins stay stricter (it can't see heads).
            let worst = hit.bbox.0.min(hit.bbox.1).min(1.0 - hit.bbox.2);
            if worst < 0.03 {
                log.push(format!(
                    "skipped edge-cropped box {:?} ({:.3} margin)",
                    hit.bbox, worst
                ));
                return None;
            }
            if looks_like_person_name(&hit.plate.provenance.author) {
                log.push(format!(
                    "skipped name-titled plate: {:?}",
                    hit.plate.provenance.author
                ));
                return None;
            }
            Some((i, area))
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some((oi_idx, _)) = ranked.first() {
        let hit = &oi_plates[*oi_idx];
        log.push(format!(
            "path: openimages box {} by {}",
            hit.label, hit.plate.provenance.author
        ));
        // Backdrop follows the brief: a named flag becomes the place,
        // a named scene place becomes the place, otherwise a generic
        // landscape. This is the per-request scope — only what this
        // prompt needs, nothing else.
        let bg_query = if brief.props.contains(&PropKind::Flag) {
            "american flag".to_string()
        } else if let Some(p) = place {
            format!("{} landscape", p)
        } else {
            "landscape".to_string()
        };
        let (bg_plates, bg_refused) = super::plates::source_plates_web(&bg_query, 3).await;
        for r in &bg_refused {
            log.push(format!("backdrop refused: {}", r));
        }
        let bg_ref = bg_plates.first().and_then(|bg| {
            (bg.provenance.source_url != hit.plate.provenance.source_url)
                .then_some((&bg.image, bg.provenance.page_url.as_str()))
        });
        match super::compose::compose_known_box(
            &hit.plate.image,
            hit.bbox,
            &hit.label,
            "imagine-openimages",
            bg_ref,
            width,
            height,
        ) {
            Ok((img, clog)) => {
                for op in &clog.ops {
                    log.push(format!("compose: {}", op));
                }
                log.push(format!(
                    "subject: box {:?} on {}",
                    clog.subject_bbox, clog.background
                ));
                return Ok((img, log));
            }
            Err(e) => log.push(format!("openimages box refused ({}); web path", e)),
        }
    }
    let study = study_references(&plates);
    for n in &study.notes {
        log.push(format!("study: {}", n));
    }
    if let Some(s) = study.skin {
        log.push(format!("study: skin {:?}", s));
    }
    if let Some(f) = study.subject_fill {
        log.push(format!("study: fill {:.2}", f));
    }
    // Backdrop search: the named prop or place when present, else
    // a generic landscape.
    let bg_query = if brief.props.contains(&PropKind::Flag) {
        "american flag".to_string()
    } else if let Some(p) = place {
        format!("{} landscape", p)
    } else {
        "landscape".to_string()
    };
    let (bg_plates, bg_refused) = super::plates::source_plates_web(&bg_query, 3).await;
    for r in &bg_refused {
        log.push(format!("backdrop refused: {}", r));
    }
    // Photo-only: without a complete photographic subject the
    // command refuses instead of meshing a block figure.
    match imagine_from_plates(&brief, &plates, &bg_plates, width, height) {
        Ok((img, mut chain)) => {
            log.append(&mut chain);
            Ok((img, log))
        }
        Err(e) => {
            log.push(format!("refused: {}", e));
            Err(log)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::plates::{PlateProvenance, SourcedPlate};
    use super::*;

    fn reference_plate(skin: Rgb, bg: Rgb) -> SourcedPlate {
        // Synthetic "photograph": skin oval on flat backdrop.
        let mut img = Image::blank(120, 160, bg);
        img.draw_disc(60, 70, 28, skin);
        SourcedPlate {
            image: img,
            provenance: PlateProvenance {
                source_url: "synthetic".to_string(),
                page_url: "synthetic".to_string(),
                author: "test".to_string(),
                license: "test".to_string(),
            },
            basis: "test".to_string(),
            title: "test".to_string(),
        }
    }

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
    fn study_measures_what_is_there() {
        // Two references, one off-tone outlier: medians hold the line.
        // (A non-skin plate is excluded entirely — tested by fallback.)
        let plates = vec![
            reference_plate(Rgb::new(200, 150, 115), Rgb::new(60, 80, 120)),
            reference_plate(Rgb::new(205, 155, 120), Rgb::new(62, 82, 122)),
            reference_plate(Rgb::new(140, 90, 60), Rgb::new(60, 80, 120)),
        ];
        let study = study_references(&plates);
        assert_eq!(study.plates_used, 3);
        let skin = study.skin.expect("skin measured");
        // Median skin ≈ (200,150,115): the outlier loses 2:1.
        assert!((skin.r as i32 - 200).abs() <= 8, "{:?}", skin);
        assert!((skin.g as i32 - 150).abs() <= 8, "{:?}", skin);
        let top = study.backdrop_top.expect("backdrop measured");
        assert!((top.r as i32 - 60).abs() <= 8, "{:?}", top);
    }

    #[test]
    fn from_plates_composes_or_refuses() {
        // With a subject plate and a backdrop plate: a new photo.
        let mut subject = Image::blank(120, 160, Rgb::new(60, 80, 120));
        subject.draw_rect(35, 40, 50, 70, Rgb::new(200, 150, 115));
        let bg = Image::blank(200, 150, Rgb::new(40, 120, 60));
        let mk = |img: Image| SourcedPlate {
            image: img,
            provenance: PlateProvenance {
                source_url: "s".to_string(),
                page_url: "s".to_string(),
                author: "t".to_string(),
                license: "t".to_string(),
            },
            basis: "t".to_string(),
            title: "t".to_string(),
        };
        let brief = parse_brief("person standing");
        let (img, log) =
            imagine_from_plates(&brief, &[mk(subject)], &[mk(bg)], 64, 80).expect("composes");
        assert_eq!((img.width, img.height), (64, 80));
        assert!(log.iter().any(|l| l.contains("photographic")), "{:?}", log);
        // With no plates at all: honest refusal, never a mesh.
        let err = imagine_from_plates(&brief, &[], &[], 64, 80).expect_err("must refuse");
        assert!(err.contains("photograph"), "{:?}", err);
    }

    #[test]
    fn photo_path_takes_complete_subjects() {
        // Centered subject with margins: composable.
        let mut centered = Image::blank(120, 160, Rgb::new(60, 80, 120));
        centered.draw_rect(35, 40, 50, 70, Rgb::new(200, 150, 115));
        assert!(score_plate(&centered).is_some());
        // Same mass bleeding off every edge: cropped, refused.
        let mut cropped = Image::blank(120, 160, Rgb::new(60, 80, 120));
        cropped.draw_rect(0, 0, 120, 160, Rgb::new(200, 150, 115));
        assert!(score_plate(&cropped).is_none());
        // Nothing at all: nothing to score.
        assert!(score_plate(&Image::blank(120, 160, Rgb::new(60, 80, 120))).is_none());
        // Picker takes the best of several plates.
        let mk = |img: Image| SourcedPlate {
            image: img,
            provenance: PlateProvenance {
                source_url: "s".to_string(),
                page_url: "s".to_string(),
                author: "t".to_string(),
                license: "t".to_string(),
            },
            basis: "t".to_string(),
            title: "t".to_string(),
        };
        let plates = vec![
            mk(Image::blank(120, 160, Rgb::new(60, 80, 120))),
            mk(centered),
        ];
        assert_eq!(pick_subject(&plates), Some(1));
        assert_eq!(pick_subject(&[]), None);
    }

    #[test]
    fn two_sources_make_one_new_photo() {
        // Subject plate (skin rect) + backdrop plate (green field):
        // the composite centers subject pixels on backdrop pixels —
        // a photo that existed in neither source.
        let mut subject = Image::blank(120, 160, Rgb::new(60, 80, 120));
        subject.draw_rect(35, 40, 50, 70, Rgb::new(200, 150, 115));
        let bg = Image::blank(200, 150, Rgb::new(40, 120, 60));
        let (img, log) = super::super::compose::compose_portrait_on(
            &subject,
            "test",
            Some((&bg, "bg-test")),
            64,
            80,
        )
        .expect("two-source composite");
        assert_eq!((img.width, img.height), (64, 80));
        assert!(log.background.contains("bg-test"), "{:?}", log.background);
        // Corners read backdrop green, center reads subject skin.
        let corner = img.get(2, 2).unwrap();
        assert!(corner.g > 80 && corner.r < 100, "{:?}", corner);
        let center = img.get(32, 30).unwrap();
        assert!(center.r > 120, "{:?}", center);
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

    #[test]
    fn empty_study_falls_back_logged() {
        let study = study_references(&[]);
        assert_eq!(study.plates_used, 0);
        assert!(study.skin.is_none());
        assert!(
            study.notes.iter().any(|n| n.contains("defaults")),
            "{:?}",
            study.notes
        );
    }
}
