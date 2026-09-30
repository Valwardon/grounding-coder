//! Family-photo restoration: find damage, fill it, log it.
//!
//! Dust specks (isolated pixels far from their neighborhood median)
//! and scratches (long thin runs of extreme luma) are detected with
//! classical operators — no model, no training — then inpainted by
//! [`super::vision::Image::inpaint`]. Clean photos report zero
//! defects; the pipeline never invents damage to fix.
use super::vision::{Image, Rgb};

/// What restoration found and did.
#[derive(Debug, Clone)]
pub struct DefectReport {
    pub specks: u64,
    pub scratch_pixels: u64,
    /// Defect bbox in plate pixels (for the log).
    pub bbox: Option<(u32, u32, u32, u32)>,
    /// Inpaint iterations used.
    pub iterations: u32,
}

fn luma(p: Rgb) -> f64 {
    0.299 * p.r as f64 + 0.587 * p.g as f64 + 0.114 * p.b as f64
}

/// 5x5 neighborhood median excluding the center pixel.
fn neighborhood_median(img: &Image, x: u32, y: u32) -> f64 {
    let mut vs = Vec::with_capacity(24);
    for dy in -2i32..=2 {
        for dx in -2i32..=2 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let sx = (x as i32 + dx).clamp(0, img.width as i32 - 1) as u32;
            let sy = (y as i32 + dy).clamp(0, img.height as i32 - 1) as u32;
            if let Some(p) = img.get(sx, sy) {
                vs.push(luma(p));
            }
        }
    }
    vs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vs.get(vs.len() / 2).copied().unwrap_or(0.0)
}

/// Detect dust specks: pixels whose luma differs from the
/// neighborhood median by more than `thresh`.
pub fn detect_specks(img: &Image, thresh: f64) -> Vec<bool> {
    let mut mask = vec![false; (img.width * img.height) as usize];
    for y in 0..img.height {
        for x in 0..img.width {
            if let Some(p) = img.get(x, y)
                && (luma(p) - neighborhood_median(img, x, y)).abs() > thresh
            {
                mask[(y * img.width + x) as usize] = true;
            }
        }
    }
    mask
}

/// Detect scratches: horizontal/vertical runs of extreme luma
/// (near-black or near-white) at least `min_run` long and at most 3
/// wide. Returns a mask of scratch pixels.
pub fn detect_scratches(img: &Image, min_run: u32) -> Vec<bool> {
    let mut mask = vec![false; (img.width * img.height) as usize];
    let extreme = |x: u32, y: u32| -> bool {
        img.get(x, y).is_some_and(|p| {
            let l = luma(p);
            l < 20.0 || l > 235.0
        })
    };
    // Horizontal runs.
    for y in 0..img.height {
        let mut x = 0u32;
        while x < img.width {
            if extreme(x, y) {
                let mut x1 = x;
                while x1 + 1 < img.width && extreme(x1 + 1, y) {
                    x1 += 1;
                }
                if x1 - x + 1 >= min_run {
                    for xx in x..=x1 {
                        mask[(y * img.width + xx) as usize] = true;
                    }
                }
                x = x1 + 1;
            } else {
                x += 1;
            }
        }
    }
    // Vertical runs.
    for x in 0..img.width {
        let mut y = 0u32;
        while y < img.height {
            if extreme(x, y) {
                let mut y1 = y;
                while y1 + 1 < img.height && extreme(x, y1 + 1) {
                    y1 += 1;
                }
                if y1 - y + 1 >= min_run {
                    for yy in y..=y1 {
                        mask[(yy * img.width + x) as usize] = true;
                    }
                }
                y = y1 + 1;
            } else {
                y += 1;
            }
        }
    }
    mask
}

/// Restore: detect specks + scratches, inpaint once, report.
/// Clean plates return (clone-equivalent, zero report) — the log
/// proves nothing was invented.
pub fn restore_plate(img: &Image, thresh: f64, min_run: u32) -> (Image, DefectReport) {
    let specks = detect_specks(img, thresh);
    let scratches = detect_scratches(img, min_run);
    let mut mask = vec![false; specks.len()];
    let mut specks_n = 0u64;
    let mut scratch_n = 0u64;
    for i in 0..mask.len() {
        if scratches[i] {
            scratch_n += 1;
        } else if specks[i] {
            specks_n += 1;
        }
        mask[i] = specks[i] || scratches[i];
    }
    let bbox = mask_bbox(&mask, img.width);
    // Iterations scale with defect extent: small spots converge fast.
    let iterations = if scratch_n + specks_n == 0 { 0 } else { 400 };
    let out = if iterations == 0 {
        img.clone()
    } else {
        img.inpaint(&mask, iterations)
    };
    (
        out,
        DefectReport {
            specks: specks_n,
            scratch_pixels: scratch_n,
            bbox,
            iterations,
        },
    )
}

fn mask_bbox(mask: &[bool], width: u32) -> Option<(u32, u32, u32, u32)> {
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
    let mut any = false;
    for (i, m) in mask.iter().enumerate() {
        if !m {
            continue;
        }
        any = true;
        let x = (i as u32) % width;
        let y = (i as u32) / width;
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    any.then_some((x0, y0, x1, y1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn damaged_gradient() -> Image {
        // Smooth ramp + 5 white specks + one 20px black scratch.
        let mut img = Image::blank(60, 40, Rgb::new(0, 0, 0));
        for y in 0..40 {
            for x in 0..60 {
                let v = (x as f64 * 200.0 / 59.0 + 20.0).round() as u8;
                img.set(x, y, Rgb::new(v, v, v));
            }
        }
        for (x, y) in [(5, 5), (50, 10), (30, 30), (10, 35), (55, 20)] {
            img.set(x, y, Rgb::new(255, 255, 255));
        }
        for x in 10..30 {
            img.set(x, 20, Rgb::new(0, 0, 0));
        }
        img
    }

    #[test]
    fn specks_and_scratches_found() {
        let img = damaged_gradient();
        let specks = detect_specks(&img, 40.0);
        // 5 planted specks plus the 20 scratch pixels (extreme black
        // against the ramp also trips the outlier test — the report
        // below attributes scratch pixels to scratches first).
        assert_eq!(specks.iter().filter(|m| **m).count(), 25);
        let scratch = detect_scratches(&img, 12);
        // The 20px scratch row, plus the 5 specks if extreme (white
        // 255 > 235: isolated, runs of length 1 < 12 — excluded).
        let n = scratch.iter().filter(|m| **m).count();
        assert!((18..=22).contains(&n), "scratch row, got {}", n);
    }

    #[test]
    fn restore_heals_and_reports() {
        let img = damaged_gradient();
        let (out, report) = restore_plate(&img, 40.0, 12);
        assert_eq!(report.specks, 5);
        assert!(report.scratch_pixels >= 18, "{}", report.scratch_pixels);
        assert!(report.bbox.is_some());
        // Healed scratch pixels track the ramp (Laplace is exact on
        // linear gradients, up to rounding).
        for x in 10..30 {
            let want = (x as f64 * 200.0 / 59.0 + 20.0).round() as u8;
            let got = out.get(x, 20).unwrap().r;
            assert!(
                (got as i32 - want as i32).abs() <= 6,
                "x={} got={} want={}",
                x,
                got,
                want
            );
        }
        // Healed specks land near neighbors, not white.
        assert!(out.get(5, 5).unwrap().r < 200);
    }

    #[test]
    fn clean_photo_reports_zero() {
        let mut img = Image::blank(40, 30, Rgb::new(0, 0, 0));
        for y in 0..30 {
            for x in 0..40 {
                let v = (x as f64 * 150.0 / 39.0 + 30.0).round() as u8;
                img.set(x, y, Rgb::new(v, v, v));
            }
        }
        let (out, report) = restore_plate(&img, 40.0, 12);
        assert_eq!((report.specks, report.scratch_pixels), (0, 0));
        assert!(report.bbox.is_none());
        assert_eq!(report.iterations, 0);
        assert_eq!(out.get(20, 15), img.get(20, 15));
    }
}
