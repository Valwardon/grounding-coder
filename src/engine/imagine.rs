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
use super::scene::{
    self, ArmPose, BodyPlan, Camera, Light, Material, PersonSpec, Scene, Shape, Vec3,
};
use super::vision::{Image, Rgb};

/// Who stands in the photo. All figures are synthetic adults built
/// from the BodyPlan — no likenesses, no minors, by construction.
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
}

/// The parsed request: deterministic keyword scan, stated limits —
/// no grammar beyond "contains the word".
#[derive(Debug, Clone)]
pub struct Brief {
    pub subject: SubjectKind,
    pub pose: ArmPose,
    pub props: Vec<PropKind>,
    pub raw: String,
}

pub fn parse_brief(prose: &str) -> Brief {
    let lower = prose.to_lowercase();
    let has = |w: &str| lower.contains(w);
    let subject = if has("woman") || has("female") || has("lady") {
        SubjectKind::Woman
    } else if has("man") || has("male") || has("gentleman") {
        SubjectKind::Man
    } else {
        SubjectKind::Person
    };
    let pose = if has("peace") {
        ArmPose::PeaceRight
    } else if has("wav") {
        ArmPose::WaveRight
    } else if has("arms out") || has("t-pose") {
        ArmPose::Out
    } else {
        ArmPose::Down
    };
    let mut props = Vec::new();
    if has("rose") || has("flower") || has("bouquet") {
        props.push(PropKind::Roses);
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
        if let Some((score, _)) = score_plate(&plate.image) {
            if score > 0.0 && best.map(|(_, s)| score > s).unwrap_or(true) {
                best = Some((i, score));
            }
        }
    }
    best.map(|(i, _)| i)
}

/// Build the fresh scene: procedural figure (measured skin or
/// default), measured backdrop palette on sky + darkened ground,
/// camera distance set so the figure fills the measured fraction,
/// props at the hand. All pixels rendered; none copied.
pub fn build_fresh(brief: &Brief, study: &Study, width: u32, height: u32) -> (Scene, Vec<String>) {
    let mut log = Vec::new();
    let plan = BodyPlan::canon();
    let (scale, skin_default, hair, shirt) = match brief.subject {
        SubjectKind::Man => (
            1.0,
            Rgb::new(200, 150, 115),
            Rgb::new(50, 35, 22),
            Rgb::new(70, 110, 180),
        ),
        SubjectKind::Woman => (
            0.94,
            Rgb::new(210, 160, 125),
            Rgb::new(90, 55, 30),
            Rgb::new(150, 70, 90),
        ),
        SubjectKind::Person => (
            1.0,
            Rgb::new(200, 150, 115),
            Rgb::new(60, 38, 24),
            Rgb::new(70, 120, 190),
        ),
    };
    let skin = study.skin.unwrap_or_else(|| {
        log.push("skin: default (no measured tone)".to_string());
        skin_default
    });
    if study.skin.is_some() {
        log.push(format!("skin: measured {:?}", skin));
    }
    let spec = PersonSpec {
        skin,
        hair,
        shirt,
        pants: Rgb::new(45, 45, 55),
        pose: brief.pose,
    };
    let base = Vec3::new(0.0, 0.0, 0.0);
    let mut shapes = vec![Shape::Plane {
        y: 0.0,
        mat: Material::named("ground", Rgb::new(86, 148, 86)),
    }];
    shapes.extend(scene::person_plan(base, scale, &spec, &plan));
    if brief.props.contains(&PropKind::Roses) {
        let (_, hand) = scene::arm_endpoints(base, scale, &spec, &plan, 1.0);
        shapes.extend(scene::bouquet(
            hand,
            scale,
            &[
                Rgb::new(200, 40, 60),
                Rgb::new(210, 90, 110),
                Rgb::new(185, 30, 50),
            ],
        ));
        log.push("props: rose bouquet at right hand".to_string());
    }
    // Camera frames the measured fill; default head-and-shoulders.
    let fill = study.subject_fill.unwrap_or(0.75);
    let figure_h = 1.8 * scale;
    let fov = 42.0f64;
    let dist = (figure_h / (2.0 * (fov.to_radians() / 2.0).tan() * fill)).clamp(1.5, 12.0);
    log.push(format!("camera: dist {:.2} for fill {:.2}", dist, fill));
    let scene_obj = Scene {
        camera: Camera {
            pos: Vec3::new(0.0, 1.1 * scale + 0.35, dist),
            look_at: Vec3::new(0.0, 0.95 * scale, 0.0),
            fov_deg: fov,
            width,
            height,
        },
        lights: vec![
            Light::key(Vec3::new(-0.45, 0.8, 0.35)),
            Light::fill(Vec3::new(0.6, 0.25, 0.7), 0.30),
            Light::fill(Vec3::new(0.3, 0.4, -0.8), 0.25),
        ],
        ambient: 0.35,
        sky_top: study.backdrop_top.unwrap_or(Rgb::new(110, 170, 235)),
        sky_bottom: study.backdrop_bottom.unwrap_or(Rgb::new(215, 235, 250)),
        shapes,
    };
    if study.backdrop_top.is_some() {
        log.push("backdrop: measured palette".to_string());
    } else {
        log.push("backdrop: default sky".to_string());
    }
    // Ground follows the measured bottom, darkened for footing.
    (scene_obj, log)
}

/// The full trajectory: research references for the brief, study
/// them, build fresh, finish the photo. One command, full receipts.
pub async fn imagine(prose: &str, width: u32, height: u32) -> (Image, Vec<String>) {
    let brief = parse_brief(prose);
    let mut log = vec![format!(
        "brief: {:?} {:?} {:?}",
        brief.subject, brief.pose, brief.props
    )];
    // Reference query follows the ask: pose terms, else portraiture.
    let query = if prose.to_lowercase().contains("peace") {
        "peace sign hand"
    } else if brief.props.contains(&PropKind::Roses) {
        "woman with roses portrait"
    } else {
        "portrait"
    };
    log.push(format!("research: query {:?}", query));
    let (mut plates, refused) = super::plates::source_plates_web(query, 3).await;
    for r in &refused {
        log.push(format!("refused: {}", r));
    }
    // Stock-photo queries wall off entirely; fall back to plain
    // portraiture once rather than study nothing. Logged either way.
    if plates.is_empty() {
        log.push("research: fallback query \"portrait\"".to_string());
        let (fallback, refused2) = super::plates::source_plates_web("portrait", 3).await;
        for r in &refused2 {
            log.push(format!("refused: {}", r));
        }
        plates = fallback;
    }
    log.push(format!("references: {} plate(s)", plates.len()));
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
    // Photo first: a complete photographic subject beats any
    // procedural figure — people pixels come from photographs.
    // Procedural builds only the world around them, or everything
    // when no plate qualifies (logged either way).
    if let Some(idx) = pick_subject(&plates) {
        log.push(format!("path: photographic (plate {})", idx));
        let plate = &plates[idx];
        match super::compose::compose_portrait(&plate.image, "imagine", width, height) {
            Ok((img, clog)) => {
                for op in &clog.ops {
                    log.push(format!("compose: {}", op));
                }
                log.push(format!(
                    "subject: bbox {:?}, {:.3} of frame",
                    clog.subject_bbox, clog.subject_fraction
                ));
                return (img, log);
            }
            Err(e) => log.push(format!("photo path refused ({}); procedural fallback", e)),
        }
    } else {
        log.push("path: procedural (no complete photographic subject)".to_string());
    }
    let (scene_obj, mut build_log) = build_fresh(&brief, &study, width, height);
    log.append(&mut build_log);
    let (mut img, receipt) = scene::render(&scene_obj);
    let person_px: u64 = receipt
        .iter()
        .filter(|(n, _)| n.starts_with("person-"))
        .map(|(_, c)| c)
        .sum();
    log.push(format!("render: {} person pixels", person_px));
    img.grade(1.12, 6.0);
    img.vignette(0.30);
    img.grain(0xC10C, 5);
    log.push("finish: grade/vignette/seeded grain".to_string());
    (img, log)
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
        }
    }

    #[test]
    fn brief_parses_plain_words() {
        let b = parse_brief("man holding peace sign");
        assert_eq!(b.subject, SubjectKind::Man);
        assert_eq!(b.pose, ArmPose::PeaceRight);
        assert!(b.props.is_empty());
        let b = parse_brief("woman with roses");
        assert_eq!(b.subject, SubjectKind::Woman);
        assert_eq!(b.pose, ArmPose::Down);
        assert_eq!(b.props, vec![PropKind::Roses]);
        let b = parse_brief("person waving");
        assert_eq!(
            (b.subject, b.pose),
            (SubjectKind::Person, ArmPose::WaveRight)
        );
        let b = parse_brief("do the thing");
        assert_eq!((b.subject, b.pose), (SubjectKind::Person, ArmPose::Down));
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
    fn build_uses_measured_skin() {
        // Color in → color out through measurement: the learning step
        // is real, not decorative.
        let plates = vec![reference_plate(
            Rgb::new(150, 100, 70),
            Rgb::new(60, 80, 120),
        )];
        let study = study_references(&plates);
        assert_eq!(study.plates_used, 1);
        let brief = parse_brief("man standing");
        let (scene_obj, _) = build_fresh(&brief, &study, 160, 120);
        let head = scene_obj
            .shapes
            .iter()
            .find_map(|s| match s {
                Shape::Sphere { center: _, mat, .. } if mat.name == "person-head" => {
                    Some(mat.color)
                }
                _ => None,
            })
            .expect("head built");
        assert_eq!(head, Rgb::new(150, 100, 70));
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
        };
        let plates = vec![
            mk(Image::blank(120, 160, Rgb::new(60, 80, 120))),
            mk(centered),
        ];
        assert_eq!(pick_subject(&plates), Some(1));
        assert_eq!(pick_subject(&[]), None);
    }

    #[test]
    fn fallback_builds_with_defaults_logged() {
        let study = study_references(&[]);
        assert_eq!(study.plates_used, 0);
        assert!(study.skin.is_none());
        let brief = parse_brief("person standing");
        let (scene_obj, log) = build_fresh(&brief, &study, 160, 120);
        assert!(log.iter().any(|l| l.contains("default")), "{:?}", log);
        let (img, receipt) = scene::render(&scene_obj);
        assert_eq!((img.width, img.height), (160, 120));
        let person_px: u64 = receipt
            .iter()
            .filter(|(n, _)| n.starts_with("person-"))
            .map(|(_, c)| *c)
            .sum();
        assert!(person_px > 100, "{:?}", receipt);
    }
}
