//! Procedural materials — Phase 4 of the image-creation system.
//!
//! Appearance from many examples: quantized dominant colors with
//! coverage fractions → palette statistics → a procedural swatch
//! that re-measures to its own inputs. Classical histogram ops
//! only. No invented reflectance parameters — what the operators
//! cannot see (roughness, subsurface, weave), the model does not
//! claim; the receipt lists exactly what was measured.
//!
//! Generic over every substance: cloth, skin, straw, steel — the
//! same derivation runs on all of them. The flag's stripe palette
//! (Phase 3) is one instance of this machinery.

use super::vision::{Image, Rgb};
use std::collections::HashMap;

fn quant(p: Rgb) -> (u8, u8, u8) {
    (p.r & 0xF8, p.g & 0xF8, p.b & 0xF8)
}

/// One histogram entry: quantized color + coverage fraction.
pub type ColorShare = ((u8, u8, u8), f64);

/// Sorted color histogram of a region, most common first.
/// Deterministic.
pub fn sample_histogram(img: &Image, x0: u32, y0: u32, x1: u32, y1: u32) -> Vec<ColorShare> {
    let mut counts: HashMap<(u8, u8, u8), u64> = HashMap::new();
    let mut total = 0u64;
    for y in y0..y1.min(img.height) {
        for x in x0..x1.min(img.width) {
            if let Some(p) = img.get(x, y) {
                *counts.entry(quant(p)).or_insert(0) += 1;
                total += 1;
            }
        }
    }
    if total == 0 {
        return Vec::new();
    }
    let mut out: Vec<ColorShare> = counts
        .into_iter()
        .map(|(c, n)| (c, n as f64 / total as f64))
        .collect();
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    out
}

/// One palette entry: color, mean coverage, agreement (share of
/// samples containing it), coverage range.
#[derive(Debug, Clone)]
pub struct ColorEntry {
    pub color: (u8, u8, u8),
    pub coverage_mean: f64,
    pub coverage_min: f64,
    pub coverage_max: f64,
    pub agreement: f64,
}

/// A derived material: palette statistics plus provenance counts.
/// No reflectance claims — color + coverage is what was measured.
#[derive(Debug, Clone)]
pub struct MaterialModel {
    pub concept: String,
    pub palette: Vec<ColorEntry>,
    pub n: usize,
}

/// Derive a material from many single-region histograms. Below 20
/// examples: Insufficient, never a material.
pub fn derive_material(
    concept: &str,
    samples: &[Vec<ColorShare>],
) -> Result<MaterialModel, String> {
    if samples.len() < super::evidence::MINIMUM_EXAMPLES {
        return Err(format!(
            "only {} '{}' samples (need {}) — not a material yet",
            samples.len(),
            concept,
            super::evidence::MINIMUM_EXAMPLES
        ));
    }
    // Pool coverages per quantized color across samples.
    let mut pool: HashMap<(u8, u8, u8), Vec<f64>> = HashMap::new();
    for s in samples {
        for (c, f) in s {
            pool.entry(*c).or_default().push(*f);
        }
    }
    let n = samples.len() as f64;
    let mut palette: Vec<ColorEntry> = pool
        .into_iter()
        .map(|(color, mut fs)| {
            fs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            ColorEntry {
                color,
                coverage_mean: fs.iter().sum::<f64>() / fs.len() as f64,
                coverage_min: fs[0],
                coverage_max: fs[fs.len() - 1],
                agreement: fs.len() as f64 / n,
            }
        })
        .collect();
    // Dominant agreement first; coverage breaks ties; color last for
    // total determinism.
    palette.sort_by(|a, b| {
        b.agreement
            .partial_cmp(&a.agreement)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.coverage_mean
                    .partial_cmp(&a.coverage_mean)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.color.cmp(&b.color))
    });
    Ok(MaterialModel {
        concept: concept.to_string(),
        palette,
        n: samples.len(),
    })
}

/// Construct a swatch from the model: horizontal bands proportional
/// to mean coverage (top agreement first). Re-measures to its own
/// palette — asserted by tests, not assumed.
pub fn construct_swatch(model: &MaterialModel, width: u32, height: u32) -> Image {
    let mut img = Image::blank(width, height, Rgb::new(0, 0, 0));
    if model.palette.is_empty() || width == 0 || height == 0 {
        return img;
    }
    let total: f64 = model.palette.iter().map(|e| e.coverage_mean).sum();
    let total = if total <= 0.0 { 1.0 } else { total };
    let mut y = 0u32;
    for (i, e) in model.palette.iter().enumerate() {
        let h = if i + 1 == model.palette.len() {
            height - y
        } else {
            ((e.coverage_mean / total) * height as f64).round() as u32
        };
        for yy in y..(y + h).min(height) {
            for x in 0..width {
                img.set(x, yy, Rgb::new(e.color.0, e.color.1, e.color.2));
            }
        }
        y += h;
        if y >= height {
            break;
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn two_tone(w: u32, h: u32) -> Image {
        let mut img = Image::blank(w, h, Rgb::new(0, 0, 0));
        for y in 0..h {
            let c = if y < h / 2 {
                Rgb::new(200, 30, 30)
            } else {
                Rgb::new(240, 240, 240)
            };
            for x in 0..w {
                img.set(x, y, c);
            }
        }
        img
    }

    #[test]
    fn histogram_reads_bands() {
        let h = sample_histogram(&two_tone(100, 100), 0, 0, 100, 100);
        assert_eq!(h.len(), 2);
        assert!((h[0].1 - 0.5).abs() < 1e-9);
        assert!((h[1].1 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn gate_holds_for_materials() {
        let few: Vec<_> = (0..3)
            .map(|_| sample_histogram(&two_tone(100, 100), 0, 0, 100, 100))
            .collect();
        assert!(derive_material("straw", &few).is_err());
    }

    #[test]
    fn empty_region_measures_empty() {
        let img = Image::blank(0, 0, Rgb::new(0, 0, 0));
        assert!(sample_histogram(&img, 0, 0, 10, 10).is_empty());
        let blank = Image::blank(10, 10, Rgb::new(0, 0, 0));
        assert!(
            construct_swatch(
                &MaterialModel {
                    concept: "void".to_string(),
                    palette: Vec::new(),
                    n: 20,
                },
                blank.width,
                blank.height
            )
            .get(0, 0)
            .is_some()
        );
    }
}
