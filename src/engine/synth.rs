//! True synthesis: novel faces from aggregates, never a copy.
//!
//! The montage path reframes a real person's photo — honestly a stock
//! image with a new backdrop, not a generated image. This module is
//! the other road: N face regions align (translation/scale only) and
//! collapse to a per-pixel MEDIAN, then measured per-pixel variation
//! drives seeded detail. Every output pixel is statistics, no donor
//! pixel is copied.
//!
//! Honesty contract, stated upfront:
//! - Alignment is crude (head-height heuristic + resize, no landmark
//!   detector — fine-joint extraction is still unimplemented). Expect
//!   blur. The receipt carries a sharpness proxy so blur is measured,
//!   not hand-waved.
//! - Below [`MIN_FACES`] usable faces the pipeline refuses with the
//!   shortfall — it never averages 2 faces and calls it a person.
//! - Novelty is asserted, not assumed: the log carries the minimum
//!   distance to any donor, and tests pin "not a copy".

use super::vision::{Image, Rgb};

/// Canonical face frame: all donors land here before statistics.
pub const CANON_W: u32 = 96;
pub const CANON_H: u32 = 128;
/// Minimum usable faces for a synthesis. Below this: refusal, never
/// an average of two strangers.
pub const MIN_FACES: usize = 8;
/// Face-region heuristic: top fraction of the person box height.
/// Heads ride at the top of full/half-body boxes; the receipt states
/// this guess so it stays checkable.
pub const FACE_HEIGHT_FRAC: f64 = 0.38;
/// Smallest face region worth aligning (pixels). Below: skipped with
/// the reason, never upscaled noise.
pub const MIN_FACE_PX: u32 = 24;

/// How one synthesis was made.
#[derive(Debug, Clone)]
pub struct SynthLog {
    /// Donor plates that contributed (indices into the input).
    pub donors: Vec<usize>,
    /// Why skipped inputs were skipped.
    pub refused: Vec<String>,
    /// Sharpness proxy: mean brightness gradient magnitude per pixel
    /// (higher = crisper; a median of misaligned faces scores low).
    pub sharpness: f64,
    /// Minimum novelty distance to any donor (0 = exact copy).
    pub min_novelty: f64,
    /// Ops applied, in order.
    pub ops: Vec<String>,
}

/// Face region of a person box (fractions) in plate pixels.
/// The head estimate is the top [`FACE_HEIGHT_FRAC`] of the box,
/// full box width. Returns None with the reason when degenerate.
pub fn extract_face_region(
    plate: &Image,
    bbox: (f64, f64, f64, f64),
    donor_idx: usize,
) -> Result<Image, String> {
    let (pw, ph) = (plate.width as f64, plate.height as f64);
    let x0 = (bbox.0 * pw).clamp(0.0, pw - 1.0) as u32;
    let y0 = (bbox.1 * ph).clamp(0.0, ph - 1.0) as u32;
    let x1 = (bbox.2 * pw).clamp(0.0, pw - 1.0) as u32;
    let y1 = (bbox.3 * ph).clamp(0.0, ph - 1.0) as u32;
    if x1 <= x0 || y1 <= y0 {
        return Err(format!("donor {}: degenerate box — skipping", donor_idx));
    }
    let box_h = y1 - y0;
    let face_h = ((box_h as f64 * FACE_HEIGHT_FRAC) as u32).max(1);
    let fy1 = (y0 + face_h).min(y1);
    let (fw, fh) = (x1 - x0 + 1, fy1 - y0 + 1);
    if fw < MIN_FACE_PX || fh < MIN_FACE_PX {
        return Err(format!(
            "donor {}: face region {}x{} below {}px — skipping",
            donor_idx, fw, fh, MIN_FACE_PX
        ));
    }
    Ok(plate.crop(x0, y0, fw, fh))
}

/// Land a face region on the canonical frame (scale only — no
/// rotation, no warp; the receipt says so).
pub fn align(face: &Image) -> Image {
    face.resize_smooth(CANON_W, CANON_H)
}

/// Per-pixel, per-channel median across aligned donors.
/// Robust to outliers by construction: one bad donor loses the vote.
/// All inputs must share dimensions; empties refuse.
pub fn median_face(aligned: &[Image]) -> Result<Image, String> {
    if aligned.is_empty() {
        return Err("no aligned faces — refusing".to_string());
    }
    let (w, h) = (aligned[0].width, aligned[0].height);
    if aligned.iter().any(|im| im.width != w || im.height != h) {
        return Err("mixed face dimensions — refusing".to_string());
    }
    let mut out = Image::blank(w, h, Rgb::new(0, 0, 0));
    let mut rs = Vec::with_capacity(aligned.len());
    let mut gs = Vec::with_capacity(aligned.len());
    let mut bs = Vec::with_capacity(aligned.len());
    for y in 0..h {
        for x in 0..w {
            rs.clear();
            gs.clear();
            bs.clear();
            for im in aligned {
                if let Some(p) = im.get(x, y) {
                    rs.push(p.r);
                    gs.push(p.g);
                    bs.push(p.b);
                }
            }
            rs.sort_unstable();
            gs.sort_unstable();
            bs.sort_unstable();
            let m = rs.len() / 2;
            out.set(x, y, Rgb::new(rs[m], gs[m], bs[m]));
        }
    }
    Ok(out)
}

/// Mean per-channel absolute difference, 0.0–1.0. Zero means the
/// same pixels — a copy, which synthesis must never be.
pub fn novelty(a: &Image, b: &Image) -> f64 {
    if a.width != b.width || a.height != b.height {
        return 1.0;
    }
    let mut acc = 0u64;
    let mut n = 0u64;
    for y in 0..a.height {
        for x in 0..a.width {
            if let (Some(p), Some(q)) = (a.get(x, y), b.get(x, y)) {
                acc += (p.r as i32 - q.r as i32).unsigned_abs() as u64;
                acc += (p.g as i32 - q.g as i32).unsigned_abs() as u64;
                acc += (p.b as i32 - q.b as i32).unsigned_abs() as u64;
                n += 3;
            }
        }
    }
    if n == 0 {
        return 1.0;
    }
    acc as f64 / n as f64 / 255.0
}

/// Per-pixel brightness deviation across donors (mean abs deviation
/// from the median frame). This is the measured texture budget:
/// detail grafting may spend up to this, never more.
fn deviation_map(aligned: &[Image], median: &Image) -> Vec<f64> {
    let (w, h) = (median.width, median.height);
    let mut dev = vec![0.0; (w * h) as usize];
    if aligned.is_empty() {
        return dev;
    }
    for y in 0..h {
        for x in 0..w {
            let m = median.get(x, y).map(|p| p.brightness()).unwrap_or(0.0);
            let mut acc = 0.0;
            let mut n = 0usize;
            for im in aligned {
                if let Some(p) = im.get(x, y) {
                    acc += (p.brightness() - m).abs();
                    n += 1;
                }
            }
            dev[(y * w + x) as usize] = if n > 0 { acc / n as f64 } else { 0.0 };
        }
    }
    dev
}

/// Seeded deterministic noise source (LCG — same family as the
/// sampler in `refine`, stated here so it stays auditable).
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64) / (u32::MAX as f64)
    }
}

/// Graft measured texture: per-pixel seeded noise bounded by the
/// donors' own deviation there (times `amount`). Flat-agreement
/// regions (low deviation) stay smooth; high-variation regions
/// (eyes, hairline) get grain. Never invents beyond measurement.
pub fn graft_detail(base: &Image, dev: &[f64], seed: u64, amount: f64) -> Image {
    let (w, h) = (base.width, base.height);
    let mut out = base.clone();
    let mut rng = Lcg(seed);
    for y in 0..h {
        for x in 0..w {
            let d = dev.get((y * w + x) as usize).copied().unwrap_or(0.0);
            let span = d * amount * 255.0;
            if span < 0.5 {
                continue;
            }
            if let Some(p) = base.get(x, y) {
                let mut jitter = |c: u8| {
                    let n = (rng.next_f64() * 2.0 - 1.0) * span;
                    (c as f64 + n).round().clamp(0.0, 255.0) as u8
                };
                out.set(x, y, Rgb::new(jitter(p.r), jitter(p.g), jitter(p.b)));
            }
        }
    }
    out
}

/// Sharpness proxy: mean Sobel-ish gradient magnitude of brightness
/// per pixel. Medians of misaligned faces score low — the number is
/// the blur receipt, not a quality claim.
pub fn sharpness(img: &Image) -> f64 {
    if img.width < 3 || img.height < 3 {
        return 0.0;
    }
    let bright = |x: u32, y: u32| img.get(x, y).map(|p| p.brightness()).unwrap_or(0.0);
    let mut acc = 0.0;
    let mut n = 0usize;
    for y in 1..img.height - 1 {
        for x in 1..img.width - 1 {
            let gx = bright(x + 1, y) - bright(x - 1, y);
            let gy = bright(x, y + 1) - bright(x, y - 1);
            acc += (gx * gx + gy * gy).sqrt();
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { acc / n as f64 }
}

/// Full pipeline: plates + person boxes (fractions) → one novel
/// face. Refuses below [`MIN_FACES`] with the shortfall.
pub fn synthesize(
    plates: &[Image],
    boxes: &[(f64, f64, f64, f64)],
    seed: u64,
) -> Result<(Image, SynthLog), String> {
    if plates.len() != boxes.len() {
        return Err(format!(
            "plates/boxes mismatch ({} vs {}) — refusing",
            plates.len(),
            boxes.len()
        ));
    }
    let mut aligned = Vec::new();
    let mut donors = Vec::new();
    let mut refused = Vec::new();
    for (i, (plate, bbox)) in plates.iter().zip(boxes.iter()).enumerate() {
        match extract_face_region(plate, *bbox, i) {
            Ok(face) => {
                aligned.push(align(&face));
                donors.push(i);
            }
            Err(e) => refused.push(e),
        }
    }
    if aligned.len() < MIN_FACES {
        return Err(format!(
            "INSUFFICIENT: {}/{} usable faces — need {} ({} refused)",
            aligned.len(),
            plates.len(),
            MIN_FACES,
            refused.len()
        ));
    }
    let median = median_face(&aligned)?;
    let dev = deviation_map(&aligned, &median);
    let mut img = graft_detail(&median, &dev, seed, 0.6);
    img.grain(seed ^ 0x5EED, 2);
    let sharp = sharpness(&img);
    let mut min_novel: f64 = 1.0;
    for d in &aligned {
        min_novel = min_novel.min(novelty(&img, d));
    }
    let n_donors = donors.len();
    let n_refused = refused.len();
    let log = SynthLog {
        donors,
        refused,
        sharpness: sharp,
        min_novelty: min_novel,
        ops: vec![
            format!(
                "face regions: {} usable of {} (top {:.0}% of box, ≥{}px)",
                n_donors,
                n_donors + n_refused,
                FACE_HEIGHT_FRAC * 100.0,
                MIN_FACE_PX
            ),
            format!(
                "align: resize to {}x{} (scale only, no landmarks)",
                CANON_W, CANON_H
            ),
            "median: per-pixel per-channel (outliers lose the vote)".to_string(),
            "detail: seeded graft bounded by measured deviation (0.6x) + grain(2)".to_string(),
            format!("sharpness {:.4}, min novelty {:.4}", sharp, min_novel),
        ],
    };
    Ok((img, log))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic "photograph": backdrop with a skin disc head on top.
    fn photo(skin: Rgb, bg: Rgb, cx: i32, w: u32, h: u32) -> Image {
        let mut img = Image::blank(w, h, bg);
        img.draw_disc(cx, (h as i32) / 3, h / 6, skin);
        img
    }

    fn person_box() -> (f64, f64, f64, f64) {
        (0.1, 0.1, 0.9, 0.9)
    }

    #[test]
    fn median_resists_outlier_donor() {
        // 8 donors with tone A, 1 with tone B: median keeps A.
        let bg = Rgb::new(60, 80, 120);
        let a = Rgb::new(200, 150, 115);
        let b = Rgb::new(120, 80, 55);
        let mut faces = Vec::new();
        for _ in 0..8 {
            faces.push(align(
                &extract_face_region(&photo(a, bg, 60, 120, 160), person_box(), 0).unwrap(),
            ));
        }
        faces.push(align(
            &extract_face_region(&photo(b, bg, 60, 120, 160), person_box(), 8).unwrap(),
        ));
        let med = median_face(&faces).unwrap();
        let c = med.get(CANON_W / 2, CANON_H / 3).unwrap();
        assert!(
            (c.r as i32 - 200).abs() <= 12,
            "outlier must lose the vote: {:?}",
            c
        );
    }

    #[test]
    fn synthesis_is_deterministic() {
        let bg = Rgb::new(60, 80, 120);
        let mk = |i: u32| {
            photo(
                Rgb::new(
                    195 + (i % 5) as u8,
                    148 + (i % 4) as u8,
                    112 + (i % 3) as u8,
                ),
                bg,
                60,
                120,
                160,
            )
        };
        let plates: Vec<Image> = (0..9).map(mk).collect();
        let boxes = vec![person_box(); 9];
        let (a, _) = synthesize(&plates, &boxes, 42).unwrap();
        let (b, _) = synthesize(&plates, &boxes, 42).unwrap();
        assert_eq!((a.width, a.height), (CANON_W, CANON_H));
        for y in (0..CANON_H).step_by(7) {
            for x in (0..CANON_W).step_by(7) {
                assert_eq!(a.get(x, y), b.get(x, y), "pixel ({},{})", x, y);
            }
        }
    }

    #[test]
    fn synthesis_is_novel_never_a_copy() {
        // Distinct donors: output must differ from every one of them.
        let bg = Rgb::new(60, 80, 120);
        let plates: Vec<Image> = (0..9)
            .map(|i| {
                let mut p = photo(
                    Rgb::new(190 + i * 2, 145 + i, 110 + i),
                    bg,
                    55 + i as i32,
                    120,
                    160,
                );
                p.draw_rect(10 + i as u32 * 3, 100, 20, 12, Rgb::new(48, 30, 17));
                p
            })
            .collect();
        let boxes = vec![person_box(); 9];
        let (img, log) = synthesize(&plates, &boxes, 7).unwrap();
        assert_eq!(log.donors.len(), 9);
        assert!(
            log.min_novelty > 0.0,
            "synthesis must not copy any donor: {}",
            log.min_novelty
        );
        assert_eq!(img.width, CANON_W);
    }

    #[test]
    fn gates_refuse_honestly() {
        assert!(median_face(&[]).is_err());
        // 3 donors < MIN_FACES: shortfall stated, never an average.
        let bg = Rgb::new(60, 80, 120);
        let plates = vec![photo(Rgb::new(200, 150, 115), bg, 60, 120, 160); 3];
        let boxes = vec![person_box(); 3];
        let err = synthesize(&plates, &boxes, 1).expect_err("must refuse");
        assert!(err.contains("INSUFFICIENT"), "wrong refusal: {}", err);
        // Mismatched inputs refuse.
        let err = synthesize(&plates, &[person_box(); 2], 1).expect_err("must refuse");
        assert!(err.contains("mismatch"), "wrong refusal: {}", err);
        // Tiny box skips with the reason.
        let tiny = extract_face_region(&plates[0], (0.0, 0.0, 0.05, 0.05), 0);
        assert!(tiny.is_err());
    }

    #[test]
    fn graft_stays_within_measured_budget() {
        // Flat-agreement donors: deviation ~0, graft must not invent.
        let bg = Rgb::new(60, 80, 120);
        let skin = Rgb::new(200, 150, 115);
        let faces: Vec<Image> = (0..8)
            .map(|_| {
                align(
                    &extract_face_region(&photo(skin, bg, 60, 120, 160), person_box(), 0).unwrap(),
                )
            })
            .collect();
        let med = median_face(&faces).unwrap();
        let dev = deviation_map(&faces, &med);
        let grafted = graft_detail(&med, &dev, 99, 0.6);
        // Center of the agreed disc: identical donors agree almost
        // exactly, so the graft leaves it (nearly) untouched.
        let c0 = med.get(CANON_W / 2, CANON_H / 3).unwrap();
        let c1 = grafted.get(CANON_W / 2, CANON_H / 3).unwrap();
        let drift = (c0.r as i32 - c1.r as i32).abs()
            + (c0.g as i32 - c1.g as i32).abs()
            + (c0.b as i32 - c1.b as i32).abs();
        assert!(drift <= 30, "graft invented beyond budget: {}", drift);
    }
}
