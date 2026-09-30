//! Human measurements from plates — Phase 1 evidence.
//!
//! Coarse, classical, honest: given a person box, the module reports
//! thirds-band skin fractions, the skin centroid, and the median skin
//! tone (Chai & Ngan locus + Y>40 floor, the same classifier the
//! composer uses). Aggregation across plates yields ranges with a
//! sample count — and a gate: below 20 examples the model is
//! `Insufficient`, never "verified".
//!
//! STATED LIMIT: fine joints (elbows, wrists, knees, ankles) are NOT
//! extractable from arbitrary photographs with classical operators
//! and no model. `fine_joints_available()` returns false and a test
//! pins it — flipping that bit without the extractor and its tests
//! is a lie the suite will catch, not a shortcut.

use super::vision::{Image, Rgb};

/// Minimum examples before ranges may be called a model (§3).
pub const MINIMUM_EXAMPLES: usize = 20;

/// Person bounding box in pixels (ground-truth boxes, never guesses).
#[derive(Debug, Clone, Copy)]
pub struct PersonBox {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl PersonBox {
    pub fn width(&self) -> u32 {
        self.x1.saturating_sub(self.x0)
    }

    pub fn height(&self) -> u32 {
        self.y1.saturating_sub(self.y0)
    }
}

/// One plate's coarse measurements. Bands split the box into head /
/// torso / legs thirds — positions are box-relative, never claimed
/// as anatomical landmarks.
#[derive(Debug, Clone)]
pub struct PlateMeasure {
    /// Skin fraction in the head third.
    pub skin_head: f64,
    /// Skin fraction in the torso third.
    pub skin_torso: f64,
    /// Skin fraction in the legs third.
    pub skin_legs: f64,
    /// Skin centroid as box fractions (x, y), if any skin found.
    pub skin_center: Option<(f64, f64)>,
    /// Median skin-tone RGB among skin pixels, if any.
    pub skin_tone: Option<Rgb>,
    /// Box aspect (width / height).
    pub aspect: f64,
}

/// Classical measurement: skin mask inside the box, counted per
/// thirds-band. Deterministic; empty boxes report zeros, not errors.
pub fn measure_plate(img: &Image, pb: &PersonBox) -> PlateMeasure {
    let mask = img.skin_mask();
    let w = pb.width().max(1) as f64;
    let h = pb.height().max(1) as f64;
    let mut band_skin = [0u64; 3];
    let mut band_total = [0u64; 3];
    let mut sx = 0u64;
    let mut sy = 0u64;
    let mut skin_n = 0u64;
    let mut rs = Vec::new();
    let mut gs = Vec::new();
    let mut bs = Vec::new();
    for y in pb.y0..pb.y1.min(img.height) {
        // Band by box-relative height: 0 head, 1 torso, 2 legs.
        let band = (((y - pb.y0) as f64 / h) * 3.0) as usize;
        let band = band.min(2);
        for x in pb.x0..pb.x1.min(img.width) {
            let idx = (y * img.width + x) as usize;
            if idx >= mask.len() {
                continue;
            }
            band_total[band] += 1;
            if mask[idx] {
                band_skin[band] += 1;
                skin_n += 1;
                sx += x as u64;
                sy += y as u64;
                if let Some(p) = img.get(x, y) {
                    rs.push(p.r);
                    gs.push(p.g);
                    bs.push(p.b);
                }
            }
        }
    }
    let frac = |b: usize| {
        if band_total[b] == 0 {
            0.0
        } else {
            band_skin[b] as f64 / band_total[b] as f64
        }
    };
    let skin_center = if skin_n == 0 {
        None
    } else {
        Some((
            (sx as f64 / skin_n as f64 - pb.x0 as f64) / w,
            (sy as f64 / skin_n as f64 - pb.y0 as f64) / h,
        ))
    };
    let skin_tone = if rs.is_empty() {
        None
    } else {
        Some(Rgb::new(median(&mut rs), median(&mut gs), median(&mut bs)))
    };
    PlateMeasure {
        skin_head: frac(0),
        skin_torso: frac(1),
        skin_legs: frac(2),
        skin_center,
        skin_tone,
        aspect: w / h,
    }
}

fn median(vs: &mut [u8]) -> u8 {
    vs.sort_unstable();
    vs[vs.len() / 2]
}

/// Min / max / mean over aggregated plates.
#[derive(Debug, Clone)]
pub struct Range {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub n: usize,
}

fn range_of(mut vs: Vec<f64>) -> Range {
    let n = vs.len();
    if n == 0 {
        return Range {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            n: 0,
        };
    }
    vs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Range {
        min: vs[0],
        max: vs[n - 1],
        mean: vs.iter().sum::<f64>() / n as f64,
        n,
    }
}

/// Model maturity: counts are the entire argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStatus {
    /// Fewer than MINIMUM_EXAMPLES — numbers, not a model.
    Insufficient { n: usize, need: usize },
    /// Enough examples to call the ranges a candidate model.
    Candidate { n: usize },
}

/// Aggregated body evidence across plates.
#[derive(Debug, Clone)]
pub struct BodyModel {
    pub skin_head: Range,
    pub skin_torso: Range,
    pub skin_legs: Range,
    pub aspect: Range,
    pub tones: Vec<Rgb>,
    pub status: ModelStatus,
}

/// Aggregate plate measures. Pure statistics — no single plate is
/// ever treated as representative (§3).
pub fn aggregate(measures: &[PlateMeasure]) -> BodyModel {
    let n = measures.len();
    BodyModel {
        skin_head: range_of(measures.iter().map(|m| m.skin_head).collect()),
        skin_torso: range_of(measures.iter().map(|m| m.skin_torso).collect()),
        skin_legs: range_of(measures.iter().map(|m| m.skin_legs).collect()),
        aspect: range_of(measures.iter().map(|m| m.aspect).collect()),
        tones: measures.iter().filter_map(|m| m.skin_tone).collect(),
        status: if n >= MINIMUM_EXAMPLES {
            ModelStatus::Candidate { n }
        } else {
            ModelStatus::Insufficient {
                n,
                need: MINIMUM_EXAMPLES,
            }
        },
    }
}

/// Only Candidate models verify. Anything else names its shortfall.
pub fn verified(model: &BodyModel) -> Result<(), String> {
    match &model.status {
        ModelStatus::Candidate { n } => {
            if model.tones.is_empty() {
                return Err("candidate has no skin tones — refusing".to_string());
            }
            let _ = n;
            Ok(())
        }
        ModelStatus::Insufficient { n, need } => Err(format!(
            "only {} examples (need {}) — not a model yet",
            n, need
        )),
    }
}

/// Machine-checked honesty gate: fine-joint extraction does not
/// exist yet. See module docs.
pub fn fine_joints_available() -> bool {
    false
}

/// Finest landmark this module reports. Today: box thirds only.
pub fn supported_detail() -> &'static str {
    "coarse-bands"
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn synthetic_plate(skin_rows: std::ops::Range<u32>) -> Image {
        // 100x150 portrait: neutral backdrop, one skin-tone block.
        let mut img = Image::blank(100, 150, Rgb::new(40, 60, 90));
        if !skin_rows.is_empty() {
            img.draw_rect(
                30,
                skin_rows.start,
                40,
                skin_rows.end - skin_rows.start,
                Rgb::new(200, 150, 115),
            );
        }
        img
    }

    #[test]
    fn thirds_split_skin_by_band() {
        let img = synthetic_plate(10..40);
        let pb = PersonBox {
            x0: 0,
            y0: 0,
            x1: 100,
            y1: 150,
        };
        let m = measure_plate(&img, &pb);
        // Head third (rows 0..50) holds all the skin.
        assert!(m.skin_head > 0.2, "head band {}", m.skin_head);
        assert_eq!(m.skin_torso, 0.0);
        assert_eq!(m.skin_legs, 0.0);
        assert!(m.skin_tone.is_some());
        let (cx, cy) = m.skin_center.expect("centroid");
        assert!((cx - 0.5).abs() < 0.05, "cx {}", cx);
        assert!(cy < 0.33, "cy {}", cy);
    }

    #[test]
    fn gate_refuses_small_collections() {
        let img = synthetic_plate(10..40);
        let pb = PersonBox {
            x0: 0,
            y0: 0,
            x1: 100,
            y1: 150,
        };
        let few: Vec<PlateMeasure> = (0..3).map(|_| measure_plate(&img, &pb)).collect();
        let model = aggregate(&few);
        assert_eq!(model.status, ModelStatus::Insufficient { n: 3, need: 20 });
        assert!(verified(&model).is_err());
        let many: Vec<PlateMeasure> = (0..20).map(|_| measure_plate(&img, &pb)).collect();
        let model = aggregate(&many);
        assert_eq!(model.status, ModelStatus::Candidate { n: 20 });
        assert!(verified(&model).is_ok());
        assert!((model.skin_head.mean - few[0].skin_head).abs() < 1e-9);
    }

    #[test]
    fn fine_joints_stay_unavailable() {
        assert!(!fine_joints_available());
        assert_eq!(supported_detail(), "coarse-bands");
    }
}
