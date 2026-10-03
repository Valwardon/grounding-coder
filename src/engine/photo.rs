//! Photo verification oracle: deterministic image diagnostics.
//!
//! The compiler verifies code; this verifies pictures. It measures
//! pixel facts — skin-tone variance, feature contrast, figure
//! presence — and reports named diagnostics with severity (how far
//! off), confidence (how much evidence), and a repair target (which
//! component owns the defect). Measurements only: whether a defect
//! is worth repairing is decided by recipes + budgets downstream,
//! never here.
//!
//! Caution, enforced by thresholds: low variance is not
//! automatically a defect (a gray wall is flat and fine) and perfect
//! symmetry is not realism. Every diagnostic carries the numbers it
//! was computed from, so a bad threshold gets caught, not obeyed.

use super::vision::{Image, Rgb};

/// What "good" looks like for one render. Thresholds are explicit
/// arguments — never tuned per image, always stated per call.
#[derive(Debug, Clone)]
pub struct PhotoExpect {
    /// Skin reference color + match tolerance (Manhattan).
    pub skin: (Rgb, u32),
    /// Minimum high-pass energy (mean neighbor difference) over
    /// skin pixels. Calibrated on real renders: finish grain alone
    /// floors ~0.0117, mottled skin measures ~0.0139 — the oracle
    /// must sit above the finish floor, never in it. Below the
    /// floor, no tonal modeling exists.
    pub min_skin_detail: f64,
    /// Minimum dark-pixel fraction inside the face band (eyes, mouth,
    /// brows, nostrils must exist as pixels).
    pub min_feature_fraction: f64,
    /// Face band as fractions of image height (top, bottom).
    pub face_band: (f64, f64),
}

impl Default for PhotoExpect {
    fn default() -> Self {
        PhotoExpect {
            skin: (Rgb::new(200, 150, 115), 90),
            min_skin_detail: 0.0128,
            min_feature_fraction: 0.005,
            face_band: (0.05, 0.30),
        }
    }
}

/// One measured defect: what, how bad, how sure, who owns it.
#[derive(Debug, Clone)]
pub struct PhotoDiagnostic {
    /// PHOTO_FLAT_SKIN | PHOTO_NO_FEATURES | PHOTO_NO_FIGURE.
    pub code: String,
    /// 0..1 shortfall (1 = total absence of the property).
    pub severity: f64,
    /// 0..1 evidence weight (sample sizes behind the numbers).
    pub confidence: f64,
    /// Human + numerical account.
    pub message: String,
    /// Owning component: "studio.skin.mottle" | "studio.face.contrast".
    /// Unowned measurements stay informational (no target).
    pub repair_target: Option<String>,
    /// The numbers behind the verdict.
    pub evidence: Vec<(String, f64)>,
}

/// Verify a rendered image against expectations. Pure: same pixels
/// twice, same diagnostics. Empty vec = the photo holds.
pub fn verify_photo(img: &Image, exp: &PhotoExpect) -> Vec<PhotoDiagnostic> {
    let mut out = Vec::new();
    let (sw, sh) = (img.width, img.height);
    if sw == 0 || sh == 0 {
        return vec![PhotoDiagnostic {
            code: "PHOTO_NO_FIGURE".to_string(),
            severity: 1.0,
            confidence: 1.0,
            message: "empty image: nothing rendered".to_string(),
            repair_target: None,
            evidence: vec![("pixels".to_string(), 0.0)],
        }];
    }
    // Skin mask + brightness stats.
    let (sr, sg, sb) = (
        exp.skin.0.r as i32,
        exp.skin.0.g as i32,
        exp.skin.0.b as i32,
    );
    let mut skin_n = 0u64;
    for y in 0..sh {
        for x in 0..sw {
            if let Some(p) = img.get(x, y) {
                let d = (p.r as i32 - sr).abs() + (p.g as i32 - sg).abs() + (p.b as i32 - sb).abs();
                if d < exp.skin.1 as i32 {
                    skin_n += 1;
                }
            }
        }
    }
    if skin_n < 100 {
        return vec![PhotoDiagnostic {
            code: "PHOTO_NO_FIGURE".to_string(),
            severity: 1.0,
            confidence: 1.0,
            message: format!("no figure: {} skin pixels", skin_n),
            repair_target: None,
            evidence: vec![("skin_pixels".to_string(), skin_n as f64)],
        }];
    }
    // Flat skin: high-pass energy far below what mottled finish
    // produces. Overall variance can't discriminate (shading
    // gradients swamp it); neighbor differences can — smooth clay
    // varies only by finish grain, modeled skin varies more.
    let mut diff_sum = 0.0;
    let mut diff_n = 0u64;
    for y in 0..sh {
        for x in 0..sw.saturating_sub(1) {
            let a = img.get(x, y);
            let b = img.get(x + 1, y);
            if let (Some(p), Some(q)) = (a, b) {
                let da =
                    (p.r as i32 - sr).abs() + (p.g as i32 - sg).abs() + (p.b as i32 - sb).abs();
                let db =
                    (q.r as i32 - sr).abs() + (q.g as i32 - sg).abs() + (q.b as i32 - sb).abs();
                if da < exp.skin.1 as i32 && db < exp.skin.1 as i32 {
                    diff_sum += ((p.r as f64 - q.r as f64).abs()
                        + (p.g as f64 - q.g as f64).abs()
                        + (p.b as f64 - q.b as f64).abs())
                        / (3.0 * 255.0);
                    diff_n += 1;
                }
            }
        }
    }
    let detail = if diff_n == 0 {
        0.0
    } else {
        diff_sum / diff_n as f64
    };
    let confidence = ((skin_n as f64) / 5000.0).min(1.0);
    if detail < exp.min_skin_detail {
        out.push(PhotoDiagnostic {
            code: "PHOTO_FLAT_SKIN".to_string(),
            severity: ((exp.min_skin_detail - detail) / exp.min_skin_detail).clamp(0.0, 1.0),
            confidence,
            message: format!(
                "flat skin: tonal detail {:.4} < {:.3} over {} px — single-tone clay (repair-target: studio.skin.mottle)",
                detail, exp.min_skin_detail, skin_n
            ),
            repair_target: Some("studio.skin.mottle".to_string()),
            evidence: vec![
                ("detail".to_string(), detail),
                ("threshold".to_string(), exp.min_skin_detail),
                ("skin_pixels".to_string(), skin_n as f64),
            ],
        });
    }
    // Features: dark pixels must exist in the face band (eyes,
    // brows, mouth, nostrils are all darker than skin).
    let y0 = (exp.face_band.0 * sh as f64) as u32;
    let y1 = (exp.face_band.1 * sh as f64) as u32;
    let mut band_n = 0u64;
    let mut dark_n = 0u64;
    for y in y0..y1.min(sh) {
        for x in 0..sw {
            if let Some(p) = img.get(x, y) {
                band_n += 1;
                let b = (p.r as u32 + p.g as u32 + p.b as u32) / 3;
                if b < 110 {
                    dark_n += 1;
                }
            }
        }
    }
    let frac = if band_n == 0 {
        0.0
    } else {
        dark_n as f64 / band_n as f64
    };
    if frac < exp.min_feature_fraction {
        out.push(PhotoDiagnostic {
            code: "PHOTO_NO_FEATURES".to_string(),
            severity: ((exp.min_feature_fraction - frac) / exp.min_feature_fraction)
                .clamp(0.0, 1.0),
            confidence: ((band_n as f64) / 20000.0).min(1.0),
            message: format!(
                "no features: dark fraction {:.4} < {:.3} in face band ({} px)",
                frac, exp.min_feature_fraction, band_n
            ),
            repair_target: Some("studio.face.contrast".to_string()),
            evidence: vec![
                ("dark_fraction".to_string(), frac),
                ("threshold".to_string(), exp.min_feature_fraction),
                ("band_pixels".to_string(), band_n as f64),
            ],
        });
    }
    out
}

/// Resolve a repair target to a code anchor: file + line of the
/// switch that owns the defect. Scans project sources for the
/// target's marker — deterministic, no guessing.
pub fn anchor_for(project_dir: &std::path::Path, target: &str) -> Option<(String, u32)> {
    // Note: markers must resolve on broken trees too (both switch
    // states), or diagnostics can't name their repair site.
    if target == "studio.face.contrast" {
        let rel = "src/engine/studio.rs";
        let text = std::fs::read_to_string(project_dir.join(rel)).ok()?;
        for (i, line) in text.lines().enumerate() {
            if line.contains("brow_mat") {
                return Some((rel.to_string(), (i + 1) as u32));
            }
        }
        return None;
    }
    if target == "studio.skin.mottle" {
        // The skin constructor specifically (not the matte one):
        // first "mottled:" after "pub fn skin".
        let rel = "src/engine/scene.rs";
        let text = std::fs::read_to_string(project_dir.join(rel)).ok()?;
        let mut in_skin = false;
        for (i, line) in text.lines().enumerate() {
            if line.contains("pub fn skin(") {
                in_skin = true;
            }
            if in_skin && line.contains("mottled:") {
                return Some((rel.to_string(), (i + 1) as u32));
            }
        }
        return None;
    }
    None
}

/// Convert a diagnostic into the engine's error shape so the
/// repair loop can consume it: code, measured message, anchored
/// file/line, and the measured numbers as the suggestion payload
/// recipes match on. Kind is assigned by the caller (PhotoDefect).
pub fn to_compile_error(
    diag: &PhotoDiagnostic,
    project_dir: &std::path::Path,
) -> super::error::CompileError {
    let (file, line) = diag
        .repair_target
        .as_deref()
        .and_then(|t| anchor_for(project_dir, t))
        .unwrap_or_default();
    super::error::CompileError {
        code: diag.code.clone(),
        message: diag.message.clone(),
        file,
        line,
        col: 0,
        suggestion: Some(format!(
            "repair-target:{} severity={:.2} confidence={:.2}",
            diag.repair_target.as_deref().unwrap_or("none"),
            diag.severity,
            diag.confidence
        )),
        source_line: None,
        kind: super::error::ErrorKind::PhotoDefect,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_skin(w: u32, h: u32) -> Image {
        Image::blank(w, h, Rgb::new(200, 150, 115))
    }

    fn varied_skin(w: u32, h: u32) -> Image {
        // In-tolerance checker: both tones read as skin, neighbor
        // diffs far above any finish — passes detail while the empty
        // face band still (honestly) fires the features check.
        let mut img = Image::blank(w, h, Rgb::new(200, 150, 115));
        for y in 0..h {
            for x in 0..w {
                if (x / 4 + y / 4) % 2 == 0 {
                    img.set(x, y, Rgb::new(175, 125, 90));
                }
            }
        }
        img
    }

    #[test]
    fn flat_skin_flags_with_numbers() {
        let d = verify_photo(&flat_skin(320, 240), &PhotoExpect::default());
        assert!(
            d.iter().any(|x| x.code == "PHOTO_FLAT_SKIN"),
            "flat must flag: {:?}",
            d.iter().map(|x| &x.code).collect::<Vec<_>>()
        );
        let flat = d.iter().find(|x| x.code == "PHOTO_FLAT_SKIN").unwrap();
        assert!(flat.severity > 0.9);
        assert_eq!(flat.repair_target.as_deref(), Some("studio.skin.mottle"));
        assert!(flat.evidence.iter().any(|(k, _)| k == "detail"));
    }

    #[test]
    fn varied_skin_passes_flat_but_flags_features() {
        // Tonal variety without any dark features: flat clears,
        // features honestly fires (empty face band is empty).
        let d = verify_photo(&varied_skin(320, 240), &PhotoExpect::default());
        assert!(!d.iter().any(|x| x.code == "PHOTO_FLAT_SKIN"));
        assert!(d.iter().any(|x| x.code == "PHOTO_NO_FEATURES"));
    }

    #[test]
    fn empty_image_is_no_figure() {
        let d = verify_photo(
            &Image::blank(0, 0, Rgb::new(0, 0, 0)),
            &PhotoExpect::default(),
        );
        assert!(d.iter().any(|x| x.code == "PHOTO_NO_FIGURE"));
        assert!(d[0].repair_target.is_none());
    }

    #[test]
    fn anchors_resolve_in_repo() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let (f, l) = anchor_for(&dir, "studio.skin.mottle").expect("mottle anchor");
        assert_eq!(f, "src/engine/scene.rs");
        assert!(l > 0);
        assert!(anchor_for(&dir, "nope.nothing").is_none());
    }
}
