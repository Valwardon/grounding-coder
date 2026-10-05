//! Truth representation: measured facts, never hallucinations.
//!
//! Doctrine: training turns photographs into facts, measurements,
//! and relationships — a growing world model, not weights. Every
//! field in every truth struct is either MEASURED (with its method
//! and donor evidence) or INSUFFICIENT (with a named gap). Gaps are
//! not refusals: they are acquisition work orders for the
//! truth-seeking loop — "eye-lock 4 more frontal donors" is a task,
//! never a verdict.
//!
//! A truth struct answers three questions per field: what was
//! measured, how (method + donors), and what is still missing. The
//! generator reasons over these answers; anything unmeasured stays
//! out of construction. The system cannot legitimately produce
//! information its truth cannot support — and says exactly which
//! support is missing.

use super::vision::Rgb;

/// How one measured value came to be known.
#[derive(Debug, Clone)]
pub struct Provenance {
    /// Measurement method (`skin-locus-median`, `eye-pair-geometry`…).
    pub method: String,
    /// Donor indices (into the training input) behind the value.
    pub donors: Vec<usize>,
}

/// A value that is either measured (with provenance) or missing with
/// a named acquisition order. There is no third state — no defaults
/// smuggled in as knowledge.
#[derive(Debug, Clone)]
pub enum Truth<T> {
    Measured { value: T, provenance: Provenance },
    Insufficient { gap: String },
}

impl<T> Truth<T> {
    pub fn is_measured(&self) -> bool {
        matches!(self, Truth::Measured { .. })
    }

    pub fn gap(&self) -> Option<&str> {
        match self {
            Truth::Insufficient { gap } => Some(gap),
            _ => None,
        }
    }
}

/// Skin appearance truth: median tone plus the measured spread
/// (per-channel min/max across donors). No reflectance claims —
/// tone + spread is what classical measurement actually yields.
#[derive(Debug, Clone)]
pub struct SkinModel {
    pub median: Rgb,
    pub spread: (u8, u8, u8),
}

/// Cranial proportion truth: head height/width and eye-line
/// fractions, each as (median, min, max) across donors. Ratios, not
/// pixels — resolution-independent by construction.
#[derive(Debug, Clone)]
pub struct ProportionModel {
    /// Eye-line distance / head width.
    pub eye_spacing: (f64, f64, f64),
    /// Head height / head width.
    pub head_aspect: (f64, f64, f64),
}

/// Landmark truth: measured eye-pair centers in normalized
/// head-square coordinates (0..1), plus head-square side in source
/// pixels (scale evidence travels with the landmarks).
#[derive(Debug, Clone)]
pub struct LandmarkSet {
    pub left_eye: (f64, f64),
    pub right_eye: (f64, f64),
    pub head_side_px: u32,
}

/// Deterministic description of a face derived from training truth.
/// Fields the evidence cannot yet support stay INSUFFICIENT with the
/// acquisition order attached — orbital/nasal/oral/jaw/ear geometry,
/// hair structure, and age effects are named gaps today, populated
/// by future extractors through this same shape.
#[derive(Debug, Clone)]
pub struct FaceTruth {
    pub skin: Truth<SkinModel>,
    pub proportions: Truth<ProportionModel>,
    pub landmarks: Truth<LandmarkSet>,
    pub hair: Truth<String>,
    pub age_effects: Truth<String>,
}

impl FaceTruth {
    /// Every gap in the struct, in field order: the truth-seeker's
    /// work list. Empty means fully constrained.
    pub fn gaps(&self) -> Vec<String> {
        [
            self.skin.gap(),
            self.proportions.gap(),
            self.landmarks.gap(),
            self.hair.gap(),
            self.age_effects.gap(),
        ]
        .into_iter()
        .flatten()
        .map(|g| g.to_string())
        .collect()
    }

    pub fn measured_fields(&self) -> usize {
        5 - self.gaps().len()
    }
}

fn median3(mut vs: Vec<u8>) -> u8 {
    vs.sort_unstable();
    vs[vs.len() / 2]
}

/// Measure skin truth from face crops: median tone + per-channel
/// spread across all masked skin pixels. Donors are crop indices.
pub fn measure_skin(crops: &[super::vision::Image]) -> Truth<SkinModel> {
    let mut rs = Vec::new();
    let mut gs = Vec::new();
    let mut bs = Vec::new();
    for crop in crops {
        let mask = crop.skin_mask();
        for y in 0..crop.height {
            for x in 0..crop.width {
                if mask[(y * crop.width + x) as usize]
                    && let Some(p) = crop.get(x, y)
                {
                    rs.push(p.r);
                    gs.push(p.g);
                    bs.push(p.b);
                }
            }
        }
    }
    if rs.is_empty() {
        return Truth::Insufficient {
            gap: "no skin pixels measured — acquire skin-visible donors".to_string(),
        };
    }
    let (r0, g0, b0) = (
        *rs.iter().min().unwrap(),
        *gs.iter().min().unwrap(),
        *bs.iter().min().unwrap(),
    );
    let (r1, g1, b1) = (
        *rs.iter().max().unwrap(),
        *gs.iter().max().unwrap(),
        *bs.iter().max().unwrap(),
    );
    Truth::Measured {
        value: SkinModel {
            median: Rgb::new(median3(rs), median3(gs), median3(bs)),
            spread: (r1 - r0, g1 - g0, b1 - b0),
        },
        provenance: Provenance {
            method: "skin-locus-median".to_string(),
            donors: (0..crops.len()).collect(),
        },
    }
}

/// Measure proportion truth from eye-locked head squares: eye
/// spacing and aspect as (median, min, max). Unlocked donors carry
/// no landmark truth and are excluded with the reason — they would
/// blur the relationships, not refine them.
pub fn measure_proportions(
    squares: &[(super::vision::Image, Option<super::synth::EyePair>)],
) -> Truth<ProportionModel> {
    let mut spacings = Vec::new();
    let mut aspects = Vec::new();
    let mut donors = Vec::new();
    for (k, (square, pair)) in squares.iter().enumerate() {
        let Some(((lx, ly), (rx, ry))) = *pair else {
            continue;
        };
        let side = square.width.max(square.height) as f64;
        if side < 1.0 {
            continue;
        }
        let ied = ((rx as f64 - lx as f64).hypot(ry as f64 - ly as f64) / side).clamp(0.0, 1.0);
        // Aspect of the head square itself is 1 by construction; the
        // informative ratio is eye-line height over side.
        let eyeline = ((ly as f64 + ry as f64) / 2.0 / side).clamp(0.0, 1.0);
        spacings.push(ied);
        aspects.push(eyeline);
        donors.push(k);
    }
    if spacings.is_empty() {
        return Truth::Insufficient {
            gap: "no eye-locked donors — acquire frontal faces".to_string(),
        };
    }
    let stat = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        (v[v.len() / 2], v[0], v[v.len() - 1])
    };
    Truth::Measured {
        value: ProportionModel {
            eye_spacing: stat(spacings),
            head_aspect: stat(aspects),
        },
        provenance: Provenance {
            method: "eye-pair-geometry".to_string(),
            donors,
        },
    }
}

/// Build face truth from head squares with optional eye pairs:
/// skin from all crops, proportions from locked ones, named gaps for
/// everything the extractors cannot yet measure. Deterministic over
/// its inputs — same crops, same truth.
pub fn face_truth(
    crops: &[super::vision::Image],
    pairs: &[Option<super::synth::EyePair>],
) -> FaceTruth {
    let squares: Vec<(super::vision::Image, Option<super::synth::EyePair>)> = crops
        .iter()
        .zip(pairs.iter())
        .map(|(c, p)| (c.clone(), *p))
        .collect();
    FaceTruth {
        skin: measure_skin(crops),
        proportions: measure_proportions(&squares),
        landmarks: match squares
            .iter()
            .enumerate()
            .filter_map(|(k, (sq, p))| (*p).map(|((lx, ly), (rx, ry))| (k, sq, lx, ly, rx, ry)))
            .next()
        {
            // First locked donor (deterministic tie-break: lowest
            // index) lends the landmark exemplar; the full
            // multi-donor landmark model is staged behind it.
            Some((k, sq, lx, ly, rx, ry)) => {
                let side = sq.width.max(sq.height) as f64;
                Truth::Measured {
                    value: LandmarkSet {
                        left_eye: (lx as f64 / side, ly as f64 / side),
                        right_eye: (rx as f64 / side, ry as f64 / side),
                        head_side_px: sq.width.min(sq.height),
                    },
                    provenance: Provenance {
                        method: "eye-pair-exemplar".to_string(),
                        donors: vec![k],
                    },
                }
            }
            None => Truth::Insufficient {
                gap: "no eye-locked donors — acquire frontal faces".to_string(),
            },
        },
        hair: Truth::Insufficient {
            gap: "hair structure unmeasured — needs hair extractor".to_string(),
        },
        age_effects: Truth::Insufficient {
            gap: "age effects unmeasured — needs age-stratified donors".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::super::vision::{Image, Rgb};
    use super::*;

    fn skin_crop(tone: Rgb) -> Image {
        let mut img = Image::blank(40, 40, Rgb::new(60, 80, 120));
        img.draw_disc(20, 20, 14, tone);
        img
    }

    #[test]
    fn skin_truth_measures_median_and_spread() {
        let crops = vec![
            skin_crop(Rgb::new(200, 150, 115)),
            skin_crop(Rgb::new(205, 155, 120)),
            skin_crop(Rgb::new(140, 90, 60)),
        ];
        match measure_skin(&crops) {
            Truth::Measured { value, provenance } => {
                assert!((value.median.r as i32 - 200).abs() <= 8);
                assert_eq!(provenance.method, "skin-locus-median");
                assert_eq!(provenance.donors.len(), 3);
                assert!(value.spread.0 > 0, "spread must record disagreement");
            }
            Truth::Insufficient { gap } => panic!("must measure: {}", gap),
        }
        // No skin anywhere: named gap, never a default tone.
        match measure_skin(&[Image::blank(16, 16, Rgb::new(60, 80, 120))]) {
            Truth::Insufficient { gap } => assert!(gap.contains("acquire")),
            _ => panic!("must name the gap"),
        }
    }

    #[test]
    fn proportions_need_locks_and_name_the_gap() {
        let sq = Image::blank(100, 100, Rgb::new(200, 150, 115));
        let pairs = [Some(((30u32, 40u32), (70u32, 40u32))), None];
        let squares = [(sq.clone(), pairs[0]), (sq, pairs[1])];
        match measure_proportions(&squares) {
            Truth::Measured { value, provenance } => {
                assert!((value.eye_spacing.0 - 0.4).abs() < 0.01);
                assert_eq!(provenance.donors, vec![0]);
            }
            Truth::Insufficient { gap } => panic!("must measure: {}", gap),
        }
        let squares: Vec<(Image, Option<crate::engine::synth::EyePair>)> = vec![];
        assert!(measure_proportions(&squares).gap().is_some());
    }

    #[test]
    fn face_truth_reports_gaps_as_work_orders() {
        let crops = vec![skin_crop(Rgb::new(200, 150, 115))];
        let truth = face_truth(&crops, &[None]);
        // Skin measured; everything structural open with orders.
        assert!(truth.skin.is_measured());
        assert!(!truth.proportions.is_measured());
        let gaps = truth.gaps();
        assert!(gaps.len() >= 3, "gaps must be listed: {:?}", gaps);
        assert!(
            gaps.iter()
                .any(|g| g.contains("acquire") || g.contains("needs"))
        );
        assert_eq!(truth.measured_fields(), 5 - gaps.len());
    }

    #[test]
    fn face_truth_is_deterministic() {
        let crops = vec![
            skin_crop(Rgb::new(200, 150, 115)),
            skin_crop(Rgb::new(205, 155, 120)),
        ];
        let a = face_truth(&crops, &[None, None]);
        let b = face_truth(&crops, &[None, None]);
        assert_eq!(a.gaps(), b.gaps());
        match (a.skin, b.skin) {
            (Truth::Measured { value: va, .. }, Truth::Measured { value: vb, .. }) => {
                assert_eq!((va.median, va.spread), (vb.median, vb.spread))
            }
            _ => panic!("skin must measure twice identically"),
        }
    }
}
