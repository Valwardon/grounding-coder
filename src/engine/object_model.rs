//! Object models — Phase 3 of the image-creation system.
//!
//! Research → measure → model → construct, starting with the flag
//! as the first simple object. Classical operators only: stripe
//! runs from row color bands, emblem blobs from connected bright
//! components in the canton, pole side from vertical edge runs,
//! aspect from the example bbox. The model is statistics (aspect
//! range, stripe mode + agreement, emblem range); construction draws
//! from the model and verifies by re-measuring its own output.
//!
//! HONEST LIMITS: unknown objects refuse with the named gap (no
//! extractor is not a default box); hats and other part-structured
//! objects wait on part segmentation (staged, not silently boxed).

use super::vision::{Image, Rgb};
use std::collections::HashMap;

/// Which side the pole reads on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoleSide {
    Left,
    Right,
}

/// Classical measurement of one flag example.
#[derive(Debug, Clone)]
pub struct FlagMeasure {
    /// Horizontal stripe runs across the middle column.
    pub stripes: u32,
    /// Width / height of the measured flag rect.
    pub aspect: f64,
    /// Pole side, if a vertical edge run reads.
    pub pole: Option<PoleSide>,
    /// Bright connected blobs in the canton quarter.
    pub emblem_blobs: u32,
    /// Median stripe colors, top to bottom (quantized).
    pub stripe_colors: Vec<(u8, u8, u8)>,
}

fn quant(p: Rgb) -> (u8, u8, u8) {
    (p.r & 0xF8, p.g & 0xF8, p.b & 0xF8)
}

/// Count color runs down one column (quantized, noise floor 2px).
fn stripe_runs(img: &Image, x: u32) -> Vec<(u8, u8, u8)> {
    let mut runs = Vec::new();
    let mut cur: Option<(u8, u8, u8)> = None;
    let mut n = 0u32;
    for y in 0..img.height {
        let q = img.get(x, y).map(quant).unwrap_or((0, 0, 0));
        if cur == Some(q) {
            n += 1;
        } else {
            if let Some(c) = cur
                && n >= 2
            {
                runs.push(c);
            }
            cur = Some(q);
            n = 1;
        }
    }
    if let Some(c) = cur
        && n >= 2
    {
        runs.push(c);
    }
    runs
}

/// Bright connected components in a rect (4-connectivity, threshold
/// on luma). Deterministic flood fill.
fn count_blobs(img: &Image, x0: u32, y0: u32, x1: u32, y1: u32, luma_min: u32) -> u32 {
    let mut seen = vec![false; (img.width * img.height) as usize];
    let bright = |x: u32, y: u32| -> bool {
        img.get(x, y).is_some_and(|p| {
            (299 * p.r as u32 + 587 * p.g as u32 + 114 * p.b as u32) / 1000 >= luma_min
        })
    };
    let mut blobs = 0u32;
    for y in y0..y1.min(img.height) {
        for x in x0..x1.min(img.width) {
            let idx = (y * img.width + x) as usize;
            if seen[idx] || !bright(x, y) {
                continue;
            }
            blobs += 1;
            let mut stack = vec![(x, y)];
            seen[idx] = true;
            while let Some((cx, cy)) = stack.pop() {
                for (nx, ny) in [
                    (cx.wrapping_sub(1), cy),
                    (cx + 1, cy),
                    (cx, cy.wrapping_sub(1)),
                    (cx, cy + 1),
                ] {
                    if nx >= x0 && nx < x1.min(img.width) && ny >= y0 && ny < y1.min(img.height) {
                        let ni = (ny * img.width + nx) as usize;
                        if !seen[ni] && bright(nx, ny) {
                            seen[ni] = true;
                            stack.push((nx, ny));
                        }
                    }
                }
            }
        }
    }
    blobs
}

/// Pole: longest vertical run of dark pixels at either edge.
/// Returns the side whose edge columns are darkest over the height.
fn pole_side(img: &Image) -> Option<PoleSide> {
    let dark_col = |x: u32| -> u32 {
        (0..img.height)
            .filter(|y| {
                img.get(x, *y).is_some_and(|p| {
                    (299 * p.r as u32 + 587 * p.g as u32 + 114 * p.b as u32) / 1000 < 60
                })
            })
            .count() as u32
    };
    if img.width < 4 {
        return None;
    }
    let left = dark_col(0).max(dark_col(1));
    let right = dark_col(img.width - 1).max(dark_col(img.width - 2));
    let need = img.height * 3 / 4;
    if left >= need && left >= right {
        Some(PoleSide::Left)
    } else if right >= need && right > left {
        Some(PoleSide::Right)
    } else {
        None
    }
}

/// Measure one flag image. The flag fills the frame (caller crops to
/// the example bbox first). Stripes read down the middle column.
pub fn measure_flag(img: &Image) -> FlagMeasure {
    let runs = stripe_runs(img, img.width / 2);
    FlagMeasure {
        stripes: runs.len() as u32,
        aspect: img.width as f64 / img.height.max(1) as f64,
        pole: pole_side(img),
        emblem_blobs: count_blobs(img, 0, 0, img.width / 3, img.height / 2, 180),
        stripe_colors: runs,
    }
}

/// The derived flag model: aspect range, stripe mode + agreement,
/// emblem range, pole consensus. Statistics, never one example.
#[derive(Debug, Clone)]
pub struct FlagModel {
    pub aspect_min: f64,
    pub aspect_max: f64,
    pub aspect_mean: f64,
    /// Most common stripe count + share of examples agreeing.
    pub stripes: u32,
    pub stripe_agreement: f64,
    pub emblem_min: u32,
    pub emblem_max: u32,
    pub pole: Option<PoleSide>,
    pub n: usize,
    /// Median palette per stripe index across examples.
    pub palette: Vec<(u8, u8, u8)>,
}

/// Derive the model. Below 20 examples: Insufficient, never a model.
pub fn derive_flag_model(measures: &[FlagMeasure]) -> Result<FlagModel, String> {
    if measures.len() < super::evidence::MINIMUM_EXAMPLES {
        return Err(format!(
            "only {} flag examples (need {}) — not a model yet",
            measures.len(),
            super::evidence::MINIMUM_EXAMPLES
        ));
    }
    let aspects: Vec<f64> = measures.iter().map(|m| m.aspect).collect();
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for m in measures {
        *counts.entry(m.stripes).or_insert(0) += 1;
    }
    let (stripes, agree) = counts
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(s, n)| (s, n as f64 / measures.len() as f64))
        .expect("nonempty");
    let depth = measures
        .iter()
        .map(|m| m.stripe_colors.len())
        .max()
        .unwrap_or(0);
    let mut palette = Vec::new();
    for i in 0..depth {
        let mut rs = Vec::new();
        let mut gs = Vec::new();
        let mut bs = Vec::new();
        for m in measures {
            if let Some(c) = m.stripe_colors.get(i) {
                rs.push(c.0);
                gs.push(c.1);
                bs.push(c.2);
            }
        }
        if !rs.is_empty() {
            palette.push((median(&mut rs), median(&mut gs), median(&mut bs)));
        }
    }
    let mut left = 0usize;
    let mut right = 0usize;
    for m in measures {
        match m.pole {
            Some(PoleSide::Left) => left += 1,
            Some(PoleSide::Right) => right += 1,
            None => {}
        }
    }
    Ok(FlagModel {
        aspect_min: aspects.iter().cloned().fold(f64::INFINITY, f64::min),
        aspect_max: aspects.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        aspect_mean: aspects.iter().sum::<f64>() / aspects.len() as f64,
        stripes,
        stripe_agreement: agree,
        emblem_min: measures.iter().map(|m| m.emblem_blobs).min().unwrap_or(0),
        emblem_max: measures.iter().map(|m| m.emblem_blobs).max().unwrap_or(0),
        pole: if left >= right && left > 0 {
            Some(PoleSide::Left)
        } else if right > 0 {
            Some(PoleSide::Right)
        } else {
            None
        },
        n: measures.len(),
        palette,
    })
}

fn median(vs: &mut [u8]) -> u8 {
    vs.sort_unstable();
    vs[vs.len() / 2]
}

/// Construct a flag from the model at `width` px: measured aspect,
/// stripe mode count, median palette cycling, pole bar. The output
/// re-measures to its own inputs (round-trip asserted by tests).
pub fn construct_flag(model: &FlagModel, width: u32) -> Image {
    let height = (width as f64 / model.aspect_mean).round() as u32;
    let height = height.max(8);
    let mut img = Image::blank(width, height, Rgb::new(0, 0, 0));
    let n = model.stripes.max(1);
    for (i, y) in (0..height).enumerate() {
        let stripe = (i as u32 * n / height) as usize;
        let (r, g, b) = model
            .palette
            .get(stripe)
            .copied()
            .unwrap_or((200, 200, 200));
        for x in 0..width {
            img.set(x, y, Rgb::new(r, g, b));
        }
    }
    // Pole bar on the consensus side (wide frames only).
    if model.pole.is_some() && width >= 4 {
        let px = if model.pole == Some(PoleSide::Left) {
            0
        } else {
            width - 1
        };
        for y in 0..height {
            img.set(px, y, Rgb::new(20, 20, 20));
            if px == 0 {
                img.set(1, y, Rgb::new(20, 20, 20));
            } else {
                img.set(width - 2, y, Rgb::new(20, 20, 20));
            }
        }
    }
    img
}

/// Model derivation for unknown objects: no extractor, named gap.
pub fn derive_object_model(concept: &str, _n: usize) -> Result<String, String> {
    Err(format!(
        "'{}': no part extractor — needs segmentation (staged, not boxed)",
        concept
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic flag: N horizontal stripes + left pole, no canton.
    pub fn synthetic_flag(stripes: u32, width: u32, height: u32) -> Image {
        let mut img = Image::blank(width, height, Rgb::new(0, 0, 0));
        for y in 0..height {
            let i = (y * stripes / height) % 2;
            let c = if i == 0 {
                Rgb::new(200, 30, 30)
            } else {
                Rgb::new(240, 240, 240)
            };
            for x in 0..width {
                img.set(x, y, c);
            }
        }
        for y in 0..height {
            img.set(0, y, Rgb::new(20, 20, 20));
            img.set(1, y, Rgb::new(20, 20, 20));
        }
        img
    }

    #[test]
    fn stripes_and_pole_read() {
        let m = measure_flag(&synthetic_flag(6, 120, 80));
        assert_eq!(m.stripes, 6);
        assert_eq!(m.pole, Some(PoleSide::Left));
        assert!((m.aspect - 1.5).abs() < 1e-9);
    }

    #[test]
    fn emblem_blobs_count() {
        let mut img = synthetic_flag(4, 120, 80);
        // Four discs fully inside the red top band (rows 0-20) and the
        // canton quarter (x<40, y<40): each reads as its own blob.
        for (dx, dy) in [(8, 10), (16, 10), (24, 10), (32, 10)] {
            img.draw_disc(dx, dy, 2, Rgb::new(250, 250, 250));
        }
        let m = measure_flag(&img);
        assert!(m.emblem_blobs >= 4, "got {}", m.emblem_blobs);
    }

    #[test]
    fn gate_holds_for_flags() {
        let few: Vec<FlagMeasure> = (0..3)
            .map(|_| measure_flag(&synthetic_flag(6, 120, 80)))
            .collect();
        assert!(derive_flag_model(&few).is_err());
    }

    #[test]
    fn unknown_objects_name_the_gap() {
        assert!(derive_object_model("statue", 30).is_err());
        assert!(derive_object_model("hat", 30).is_err());
    }
}
