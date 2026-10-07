//! One command from prose to photo: imagine.
//!
//! ```text
//! prose → intent → research → rank → deliver
//! ```
//!
//! Brand-new photos from research — never a donor's bytes, never
//! a rendered mesh. Researched plates are measured, person crops
//! are box-aligned (translation/scale only), and the crops collapse
//! to a per-pixel MEDIAN with seeded detail bounded by measured
//! deviation. Every output pixel is statistics; novelty against
//! every donor is asserted, blur is carried on the receipt. No
//! montage, no claymation, no copying.
//!
//! Refusal is honest: with too few usable donors the command refuses
//! with the shortfall instead of averaging strangers thinly — and
//! the receipt states plainly that classical alignment blur is not
//! yet photographer-indistinguishable.

use super::plates::SourcedPlate;
use super::vision::{Image, Rgb};

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

/// What the references taught, per channel as medians. None means
/// "no usable reference" — ranking degrades to size order and the
/// log says so.
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
/// Measurements describe the donors — never pixels for output.
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

/// Ground-truth donor box: plate index plus fractional person box.
pub type DonorBox = (usize, (f64, f64, f64, f64));

/// Person box of one plate as fractions: measured skin-blob bbox
/// over frame dims. `None` when no measurable subject (wash,
/// speckle, or too small) — synthesis skips it, never guesses one.
fn person_box_frac(img: &Image) -> Option<(f64, f64, f64, f64)> {
    let (w, h) = (img.width, img.height);
    if w < 32 || h < 32 {
        return None;
    }
    let frame_area = (w * h) as f64;
    let cleaned = Image::morph_close(&Image::morph_open(&img.skin_mask(), w, h, 2), w, h, 2);
    let blobs = Image::all_blobs(&cleaned, w, h, (frame_area * 0.002) as usize);
    let (bx0, by0, bx1, by1, area, _parts) = Image::assemble_subject(&blobs, w, h, 0.06)?;
    let fraction = area as f64 / frame_area;
    if fraction > 0.6 || fraction < 0.001 {
        return None;
    }
    Some((
        bx0 as f64 / w as f64,
        by0 as f64 / h as f64,
        (bx1 + 1) as f64 / w as f64,
        (by1 + 1) as f64 / h as f64,
    ))
}

/// A synthesized photograph: brand-new pixels computed from
/// researched donors, plus the receipt that proves it. No donor
/// pixel is copied; novelty against every donor is asserted.
#[derive(Debug, Clone)]
pub struct SynthPhoto {
    pub image: Image,
    /// Donor plate indices behind the pixels.
    pub donors: Vec<usize>,
    /// Provenance lines (author + license + page) per donor.
    pub donor_provenance: Vec<String>,
    /// Sharpness proxy (higher = crisper; pose-mixed medians score low).
    pub sharpness: f64,
    /// Minimum distance to any donor (0 = copy — must never be).
    pub min_novelty: f64,
    /// Why skipped inputs were skipped.
    pub refused: Vec<String>,
}

/// Synthesize from already-sourced plates (no network): extract
/// person donors (ground-truth boxes where given, measured skin-blob
/// boxes otherwise), synthesize one novel figure from their median.
/// Pure over its inputs — the offline-testable core. Refuses (naming
/// the shortfall) below the donor gate.
pub fn synthesize_from_plates(
    brief: &Brief,
    plates: &[SourcedPlate],
    oi_boxes: &[DonorBox],
) -> Result<(SynthPhoto, Vec<String>), String> {
    let mut log = vec![format!(
        "brief: {:?} {:?} {:?}",
        brief.subject, brief.pose, brief.props
    )];
    log.push(format!(
        "research-only: {} plate(s) + {} box(es) — synthesizing, never copying",
        plates.len(),
        oi_boxes.len()
    ));
    // Donor extraction: name-titled plates excluded (identified
    // people are a human decision); ground-truth boxes win wherever
    // given, measured skin-blob boxes otherwise; unmeasurable plates
    // skipped with reasons.
    let mut refused: Vec<String> = Vec::new();
    let mut cand_imgs: Vec<Image> = Vec::new();
    let mut cand_boxes: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut cand_ids: Vec<usize> = Vec::new();
    let mut cand_gt: Vec<bool> = Vec::new();
    for (i, p) in plates.iter().enumerate() {
        if looks_like_person_name(&p.provenance.author) {
            refused.push(format!(
                "plate {}: name-titled ({:?})",
                i, p.provenance.author
            ));
            continue;
        }
        if let Some((_, bbox)) = oi_boxes.iter().find(|(pi, _)| *pi == i) {
            cand_imgs.push(p.image.clone());
            cand_boxes.push(*bbox);
            cand_ids.push(i);
            cand_gt.push(true);
        } else {
            match person_box_frac(&p.image) {
                Some(m) => {
                    cand_imgs.push(p.image.clone());
                    cand_boxes.push(m);
                    cand_ids.push(i);
                    cand_gt.push(false);
                }
                None => refused.push(format!("plate {}: no measurable person — skipped", i)),
            }
        }
    }
    let study = study_references(plates);
    for n in &study.notes {
        log.push(format!("study: {}", n));
    }
    if let Some(s) = study.skin {
        log.push(format!("study: skin {:?}", s));
    }
    if let Some(f) = study.subject_fill {
        log.push(format!("study: fill {:.2}", f));
    }
    let n_gt = cand_gt.iter().filter(|g| **g).count();
    log.push(format!(
        "donors: {} measurable person(s) ({} ground-truth boxes), {} refused",
        cand_imgs.len(),
        n_gt,
        refused.len()
    ));
    for r in &refused {
        log.push(format!("donor refused: {}", r));
    }
    let seed: u64 = 0xF00D;
    match super::synth::synthesize_body(&cand_imgs, &cand_boxes, seed) {
        Ok((image, slog)) => {
            for op in &slog.ops {
                log.push(format!("synth: {}", op));
            }
            for r in &slog.refused {
                log.push(format!("synth refused: {}", r));
            }
            // Donor provenance travels into the product receipt.
            let donor_provenance: Vec<String> = cand_ids
                .iter()
                .map(|id| {
                    let p = &plates[*id];
                    format!(
                        "{} | {} | {}",
                        p.provenance.author, p.provenance.license, p.provenance.page_url
                    )
                })
                .collect();
            for d in &donor_provenance {
                log.push(format!("donor: {}", d));
            }
            log.push(
                "product: brand-new pixels from donor statistics — not a donor copy, not a render"
                    .to_string(),
            );
            Ok((
                SynthPhoto {
                    image,
                    donors: cand_ids,
                    donor_provenance,
                    sharpness: slog.sharpness,
                    min_novelty: slog.min_novelty,
                    refused: slog.refused.clone(),
                },
                log,
            ))
        }
        Err(e) => {
            log.push(format!("refused: {}", e));
            Err(format!("synthesis refused: {}", e))
        }
    }
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
/// with the subject ("sitting cat"), subject gaps re-ask the noun
/// as a photograph, object gaps ask the object. Capped at three,
/// deduped, deterministic — the funded second round asks exactly
/// what the first round failed to show.
pub fn gap_queries(gaps: &[String], subject_word: &str, pose: &str) -> Vec<String> {
    let mut out = Vec::new();
    for g in gaps {
        if out.len() >= 3 {
            break;
        }
        if let Some(action) = g.strip_prefix("action:") {
            let _ = action;
            let q = if pose != "standing" {
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

/// Sane OpenImages boxes for synthesis: full-frame boxes reframe
/// the whole photo (nothing learned), edge-cropped boxes bleed, and
/// name-titled plates name an identified person — synthesis skips
/// them (logged) while the library keeps them listed. Returns the
/// indices of usable donor plates: their crops feed the median,
/// never the output.
fn usable_openimages(
    oi_plates: &[super::plates::openimages::OpenPlate],
    log: &mut Vec<String>,
) -> Vec<usize> {
    let mut ranked: Vec<(usize, f64)> = oi_plates
        .iter()
        .enumerate()
        .filter_map(|(i, hit)| {
            let area = (hit.bbox.2 - hit.bbox.0) * (hit.bbox.3 - hit.bbox.1);
            if area > 0.9 {
                return None;
            }
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
    ranked.into_iter().map(|(i, _)| i).collect()
}

/// The full trajectory: research the brief, study the plates,
/// synthesize one novel figure from their median. One command, full
/// receipts. Per-request training, narrow scope: research only what
/// this prompt needs (subject + pose + props) — grounded, never from
/// noise, never a donor copy, never a render.
/// Two bounded rounds: round 1 researches the brief; coverage gaps
/// fund round 2 with refined queries; the round closing more gaps
/// wins (ties keep round 1). Refusal is never terminal while gaps
/// name what is missing — the loop only stops at two rounds.
pub async fn imagine(
    prose: &str,
    _width: u32,
    _height: u32,
) -> Result<(SynthPhoto, Vec<String>), Vec<String>> {
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
    let gq = gap_queries(&g1, &subject_word, &brief.pose);
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

/// One research round: research every named subject as donors,
/// study them, synthesize one novel figure from their median.
/// Override queries replace the default subject query (tried in
/// order until plates land). Returns the outcome plus evidence words
/// for coverage checking. Round 1 passes None and researches the
/// brief. No copying, no montage, no rendering: brand-new pixels
/// from donor statistics, with donor provenance on the receipt.
#[allow(clippy::too_many_arguments)]
async fn attempt(
    prose: &str,
    brief: &Brief,
    subject_word: &str,
    generic: bool,
    place: Option<&str>,
    extra_nouns: Vec<String>,
    override_queries: Option<Vec<String>>,
) -> (Result<(SynthPhoto, Vec<String>), Vec<String>>, Vec<String>) {
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
        "references: {} plate(s) — donors for synthesis, never delivered",
        plates.len()
    ));
    // Open Images is a people-only index (Person/Woman boxes): for
    // non-person subjects it would return strangers' photos as false
    // donors, so candidacy skips it with the reason stated. For
    // people, sane boxes join as donors with their ground-truth
    // boxes — measured, never delivered.
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
        "openimages: {} plate(s), {} usable donor(s)",
        oi_plates.len(),
        usable.len()
    ));
    // Ground-truth boxes, indexed against the combined donor list:
    // OI plates append after the Commons/Web plates.
    let base = plates.len();
    let mut all_plates = plates;
    let mut oi_boxes: Vec<DonorBox> = Vec::new();
    for idx in usable.iter().take(8) {
        let hit = &oi_plates[*idx];
        evidence.extend(evidence_words(&hit.plate.title));
        oi_boxes.push((base + oi_boxes.len(), hit.bbox));
        all_plates.push(hit.plate.clone());
    }
    // Study, then synthesize one novel figure from the donors.
    match synthesize_from_plates(brief, &all_plates, &oi_boxes) {
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
    fn measurement_skips_unusable_boxes() {
        use super::super::plates::openimages::OpenPlate;
        // Full-frame, edge-cropped, and name-titled boxes are skipped
        // for delivery (logged); clean boxes are usable candidates —
        // delivered as-is, never altered.
        let mk = |box_: (f64, f64, f64, f64), author: &str| OpenPlate {
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
            bbox: box_,
            label: "x".to_string(),
            total_boxes: 2,
        };
        let plates = vec![
            mk((0.0, 0.0, 1.0, 1.0), "plain author"), // full-frame: out
            mk((0.0, 0.2, 0.6, 0.8), "plain author"), // edge-cropped: out
            mk((0.2, 0.2, 0.6, 0.6), "lifrita lifi,Christina Hendricks 3"), // named: out
            mk((0.2, 0.2, 0.6, 0.6), "plain author"), // clean: usable
        ];
        let mut log = Vec::new();
        assert_eq!(usable_openimages(&plates, &mut log), vec![3]);
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
        // Refined queries pair the pose with the subject, capped.
        let qs = gap_queries(&gaps, "cat", "sitting");
        assert!(qs.len() <= 3 && !qs.is_empty());
        assert!(
            qs.iter()
                .any(|q| q.contains("sitting") && q.contains("cat")),
            "{:?}",
            qs
        );
        assert!(qs.iter().any(|q| q.contains("human")), "{:?}", qs);
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

    fn body_donors(n: usize) -> Vec<SourcedPlate> {
        // Alternating close tones: the median holds, graft noise keeps
        // novelty above zero, aspect coherence holds (same geometry).
        (0..n)
            .map(|i| {
                let tone = if i % 2 == 0 {
                    Rgb::new(200, 150, 115)
                } else {
                    Rgb::new(205, 155, 120)
                };
                reference_plate(tone, Rgb::new(60, 80, 120))
            })
            .collect()
    }

    #[test]
    fn synthesis_makes_new_pixels_from_donors() {
        // Eight measured donors in, one novel figure out: canon
        // dimensions, asserted novelty (not a copy of any donor),
        // donor provenance on the product, synthesis ops in the log.
        let donors = body_donors(8);
        let brief = parse_brief("a man standing");
        let (photo, log) = synthesize_from_plates(&brief, &donors, &[]).expect("synthesizes");
        assert_eq!(
            (photo.image.width, photo.image.height),
            (crate::engine::synth::BODY_W, crate::engine::synth::BODY_H)
        );
        assert_eq!(photo.donors.len(), 8);
        assert_eq!(photo.donor_provenance.len(), 8);
        assert!(photo.min_novelty > 0.0, "must not be a copy");
        assert!(photo.sharpness >= 0.0);
        assert!(
            log.iter().any(|l| l.contains("brand-new pixels")),
            "{:?}",
            log
        );
        assert!(
            !log.iter().any(|l| l.contains("delivered: plate")),
            "no delivery path: {:?}",
            log
        );
        // Product pixels differ from every donor crop: novelty is
        // measured, not claimed.
        for d in &donors {
            let mut same = 0usize;
            let mut total = 0usize;
            for y in (0..photo.image.height).step_by(7) {
                for x in (0..photo.image.width).step_by(7) {
                    // Donors are 120x160; sample the overlapping window.
                    if x < d.image.width && y < d.image.height {
                        total += 1;
                        if photo.image.get(x, y) == d.image.get(x, y) {
                            same += 1;
                        }
                    }
                }
            }
            assert!(
                (same as f64) < (total as f64) * 0.99,
                "product copies a donor: {}/{} match",
                same,
                total
            );
        }
        // Below the donor gate: honest refusal with the shortfall.
        let thin = body_donors(3);
        let err = synthesize_from_plates(&brief, &thin, &[]).expect_err("must refuse");
        assert!(err.contains("INSUFFICIENT"), "{:?}", err);
        // With no plates at all: honest refusal, never fabrication.
        let err = synthesize_from_plates(&brief, &[], &[]).expect_err("must refuse");
        assert!(err.contains("INSUFFICIENT"), "{:?}", err);
    }

    #[test]
    fn study_feeds_synthesis_not_copying() {
        // The study measures donors; synthesis consumes the crops.
        // Two agreeing donors plus one outlier hold the median.
        let donors = vec![
            reference_plate(Rgb::new(200, 150, 115), Rgb::new(60, 80, 120)),
            reference_plate(Rgb::new(205, 155, 120), Rgb::new(62, 82, 122)),
            reference_plate(Rgb::new(140, 90, 60), Rgb::new(60, 80, 120)),
        ];
        let study = study_references(&donors);
        assert_eq!(study.plates_used, 3);
        let skin = study.skin.expect("skin measured");
        assert!((skin.r as i32 - 200).abs() <= 8, "{:?}", skin);
        // Every synthetic donor yields a measurable person box.
        for d in &donors {
            assert!(person_box_frac(&d.image).is_some());
        }
        assert!(person_box_frac(&Image::blank(120, 160, Rgb::new(60, 80, 120))).is_none());
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
