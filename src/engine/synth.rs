//! True synthesis: novel faces from aggregates, never a copy.
//!
//! There is no montage path: N face regions align (translation/scale
//! only) and collapse to a per-pixel MEDIAN, then measured
//! per-pixel variation drives seeded detail. Every output pixel is
//! statistics, no donor pixel is copied.
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
/// Eye-pair level tolerance as a fraction of head-square side (0.20 ≈
/// 11° of head tilt). Measured near-misses at 0.145 locked nothing at
/// 0.12 — tilted heads are faces too, so the gate moved with the
/// evidence logged beside it.
pub const EYE_LEVEL_FRAC: f64 = 0.20;
/// Two dark-blob centers in head-square pixels, left first.
pub type EyePair = ((u32, u32), (u32, u32));
/// A scored eye-pair candidate: score plus the pair.
type ScoredPair = (f64, (u32, u32), (u32, u32));

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

/// One extracted face: pixels plus how it was landed. `anchored`
/// means the head was measured (skin blob), not guessed from box
/// geometry; `eye_aligned` means scale and translation were set from
/// a measured dark pair (eyes — or brows, stated below), not from the
/// blob's extent. The receipt counts all three tiers: unanchored
/// donors blur the median, and the count says so out loud.
#[derive(Debug, Clone)]
pub struct FaceHit {
    pub image: Image,
    pub anchored: bool,
    pub eye_aligned: bool,
}

/// Eye-pair diagnostics for one head square: dark-pixel fraction,
/// dark-blob count, and either the locked pair geometry or which
/// gate rejects the best candidate. Tuning evidence — the receipt
/// for why a donor did or did not lock.
pub fn eye_debug(head: &Image) -> String {
    let (w, h) = (head.width, head.height);
    if w < MIN_FACE_PX || h < MIN_FACE_PX {
        return format!("head {}x{} below gate", w, h);
    }
    let uh = h / 2;
    let upper = head.crop(0, 0, w, uh);
    let mut dark_n = 0usize;
    let mut dark = vec![false; (w * uh) as usize];
    for y in 0..uh {
        for x in 0..w {
            if upper.get(x, y).map(|p| p.brightness()).unwrap_or(1.0) < 0.30 {
                dark[(y * w + x) as usize] = true;
                dark_n += 1;
            }
        }
    }
    let cleaned = Image::morph_close(&Image::morph_open(&dark, w, uh, 1), w, uh, 1);
    let area = (w * uh) as f64;
    let blobs = Image::all_blobs(&cleaned, w, uh, (area * 0.0003) as usize + 4);
    let marks: Vec<(f64, f64, usize)> = blobs
        .iter()
        .filter(|b| b.0 as f64 <= area * 0.05)
        .map(|b| (((b.1 + b.3) / 2) as f64, ((b.2 + b.4) / 2) as f64, b.0))
        .collect();
    if marks.len() < 2 {
        return format!(
            "dark {:.3}, blobs {} (marks {}): need 2+ marks",
            dark_n as f64 / area,
            blobs.len(),
            marks.len()
        );
    }
    let side = w.max(uh) as f64;
    let mut near_miss = String::from("no pair in range");
    let mut best_score = 0.0f64;
    for i in 0..marks.len() {
        for j in (i + 1)..marks.len() {
            let (ax, ay, aa) = marks[i];
            let (bx, by, ba) = marks[j];
            let (l, r) = if ax <= bx {
                ((ax, ay), (bx, by))
            } else {
                ((bx, by), (ax, ay))
            };
            let dx = (r.0 - l.0).abs();
            let dy = (r.1 - l.1).abs();
            let midx = (l.0 + r.0) / 2.0;
            let bigr = aa.max(ba) as f64 / aa.min(ba).max(1) as f64;
            let gate = if dy > side * EYE_LEVEL_FRAC {
                format!("level dy={:.1}", dy)
            } else if dx < side * 0.12 || dx > side * 0.5 {
                format!("separation dx={:.1}", dx)
            } else if bigr > 3.0 {
                format!("size ratio {:.1}", bigr)
            } else if (midx - w as f64 / 2.0).abs() > side * 0.15 {
                format!("off-center midx={:.1}", midx)
            } else {
                String::from("SCORED")
            };
            if gate == "SCORED" {
                best_score = best_score.max(1.0 - dy / (side * EYE_LEVEL_FRAC) + dx / side);
            } else if best_score == 0.0 {
                near_miss = gate;
            }
        }
    }
    format!(
        "dark {:.3}, blobs {} (marks {}), best {}",
        dark_n as f64 / area,
        blobs.len(),
        marks.len(),
        if best_score > 0.0 {
            format!("score {:.2}", best_score)
        } else {
            format!("rejected: {}", near_miss)
        }
    )
}
/// Eye-or-brow pair in head-square pixels: two dark blobs in the top
/// half, roughly level, sanely separated, symmetric about the
/// vertical center. Pupils, lashes, and brows all read dark against
/// skin; the PAIR geometry is the landmark, never any single blob.
/// Separation is anatomically gated: human inter-eye distance runs
/// ~0.3 of head width, and the head square always contains the whole
/// blob it was built around, so implied ratios outside [0.12, 0.50]
/// of the side cannot be eyes — nostrils below, ears and hair masses
/// above. Measured donor ratios confirmed the upper bound (locked
/// junk clustered at 0.55+). Brows satisfy the same geometry as eyes
/// at a nearby height — both pin vertical scale, so either lock
/// aligns; the docs say so.
/// Returns the two blob centers, left first.
pub fn eye_pair(head: &Image) -> Option<EyePair> {
    let (w, h) = (head.width, head.height);
    if w < MIN_FACE_PX || h < MIN_FACE_PX {
        return None;
    }
    let uh = h / 2;
    let upper = head.crop(0, 0, w, uh);
    let mut dark = vec![false; (w * uh) as usize];
    for y in 0..uh {
        for x in 0..w {
            if upper.get(x, y).map(|p| p.brightness()).unwrap_or(1.0) < 0.30 {
                dark[(y * w + x) as usize] = true;
            }
        }
    }
    let cleaned = Image::morph_close(&Image::morph_open(&dark, w, uh, 1), w, uh, 1);
    let area = (w * uh) as f64;
    let blobs = Image::all_blobs(&cleaned, w, uh, (area * 0.0003) as usize + 4);
    // Candidates: compact dark marks, never hair masses.
    let marks: Vec<(f64, f64, usize)> = blobs
        .iter()
        .filter(|b| b.0 as f64 <= area * 0.05)
        .map(|b| (((b.1 + b.3) / 2) as f64, ((b.2 + b.4) / 2) as f64, b.0))
        .collect();
    let side = w.max(uh) as f64;
    let mut best: Option<ScoredPair> = None;
    for i in 0..marks.len() {
        for j in (i + 1)..marks.len() {
            let (ax, ay, aa) = marks[i];
            let (bx, by, ba) = marks[j];
            let (l, r) = if ax <= bx {
                ((ax, ay), (bx, by))
            } else {
                ((bx, by), (ax, ay))
            };
            let dx = (r.0 - l.0).abs();
            let dy = (r.1 - l.1).abs();
            if dy > side * EYE_LEVEL_FRAC || dx < side * 0.12 || dx > side * 0.5 {
                continue;
            }
            let big = aa.max(ba) as f64;
            let small = aa.min(ba).max(1) as f64;
            if big / small > 3.0 {
                continue;
            }
            let midx = (l.0 + r.0) / 2.0;
            if (midx - w as f64 / 2.0).abs() > side * 0.15 {
                continue;
            }
            // Score: level + well-separated + centered + substantial.
            let score = (1.0 - dy / (side * EYE_LEVEL_FRAC)) + dx / side + (aa + ba) as f64 / area;
            let pair = ((l.0 as u32, l.1 as u32), (r.0 as u32, r.1 as u32));
            if best.map(|(s, _, _)| score > s).unwrap_or(true) {
                best = Some((score, pair.0, pair.1));
            }
        }
    }
    best.map(|(_, l, r)| (l, r))
}

/// Head square in plate pixels: the largest skin blob in the upper
/// 60% of the person box becomes the head estimate, expanded to a
/// square (1.8x the longest blob side for hair/chin context). Returns
/// None when no plausible head blob exists (wash, speckle, or bare
/// background — each with its own shape of nothing).
pub fn head_square(plate: &Image, bbox: (f64, f64, f64, f64)) -> Option<(u32, u32, u32)> {
    let (pw, ph) = (plate.width, plate.height);
    let x0 = (bbox.0 * pw as f64).clamp(0.0, pw as f64 - 1.0) as u32;
    let y0 = (bbox.1 * ph as f64).clamp(0.0, ph as f64 - 1.0) as u32;
    let x1 = (bbox.2 * pw as f64).clamp(0.0, pw as f64 - 1.0) as u32;
    let y1 = (bbox.3 * ph as f64).clamp(0.0, ph as f64 - 1.0) as u32;
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    // Head zone: upper 60% of the box. Faces ride high; torsos don't.
    let zone_h = (((y1 - y0 + 1) as f64 * 0.6) as u32).max(1);
    let zy1 = (y0 + zone_h).min(y1).min(ph - 1);
    let zw = x1 - x0 + 1;
    let zh = zy1 - y0 + 1;
    if zw < MIN_FACE_PX || zh < MIN_FACE_PX {
        return None;
    }
    let zone = plate.crop(x0, y0, zw, zh);
    let cleaned = Image::morph_close(&Image::morph_open(&zone.skin_mask(), zw, zh, 2), zw, zh, 2);
    let zone_area = (zw * zh) as f64;
    let blobs = Image::all_blobs(&cleaned, zw, zh, (zone_area * 0.002) as usize);
    let (area, bx0, by0, bx1, by1) = *blobs.first()?;
    // Wash rejection: the "head" cannot be most of the zone.
    if area as f64 > zone_area * 0.5 {
        return None;
    }
    let bw = bx1 - bx0 + 1;
    let bh = by1 - by0 + 1;
    let side = (((bw.max(bh) as f64 * 1.8) as u32).max(MIN_FACE_PX)).min(pw.min(ph));
    // Blob center in plate coordinates drives the square.
    let cx = x0 + (bx0 + bx1) / 2;
    let cy = y0 + (by0 + by1) / 2;
    let half = side / 2;
    let sx0 = cx.saturating_sub(half);
    let sy0 = cy.saturating_sub(half);
    // Clamp the square inside the plate without shrinking it when
    // possible; shrink only when the plate itself is smaller.
    let sx0 = sx0.min(pw.saturating_sub(side));
    let sy0 = sy0.min(ph.saturating_sub(side));
    let (ew, eh) = (pw - sx0, ph - sy0);
    let side = side.min(ew.min(eh));
    if side < MIN_FACE_PX {
        return None;
    }
    Some((sx0, sy0, side))
}

/// Face region of a person box (fractions) in plate pixels.
/// First choice is the measured head square ([`head_square`]); the
/// fallback is the top [`FACE_HEIGHT_FRAC`] of the box, full box
/// width. The fallback is flagged on the hit — it aligns box
/// geometry, not heads, and medians built on it blur.
pub fn extract_face_region(
    plate: &Image,
    bbox: (f64, f64, f64, f64),
    donor_idx: usize,
) -> Result<FaceHit, String> {
    if let Some((sx0, sy0, side)) = head_square(plate, bbox) {
        let head = plate.crop(sx0, sy0, side, side);
        // Eye lock: scale + translation from the measured pair. The
        // square side is 3.4x the inter-eye distance with the pair line
        // at 38% from the top — face proportions, not box geometry.
        if let Some(((lx, ly), (rx, ry))) = eye_pair(&head) {
            let ied = ((rx as f64 - lx as f64).hypot(ry as f64 - ly as f64)).max(1.0);
            let s = ((ied * 3.4) as u32).max(MIN_FACE_PX);
            let midx = sx0 as f64 + (lx as f64 + rx as f64) / 2.0;
            let midy = sy0 as f64 + (ly as f64 + ry as f64) / 2.0;
            let ex0 = (midx - s as f64 / 2.0)
                .round()
                .clamp(0.0, plate.width as f64 - 1.0) as u32;
            let ey0 = (midy - s as f64 * 0.38)
                .round()
                .clamp(0.0, plate.height as f64 - 1.0) as u32;
            let ew = (plate.width - ex0).min(s);
            let eh = (plate.height - ey0).min(s);
            let eside = ew.min(eh);
            if eside >= MIN_FACE_PX {
                return Ok(FaceHit {
                    image: plate.crop(ex0, ey0, eside, eside),
                    anchored: true,
                    eye_aligned: true,
                });
            }
        }
        return Ok(FaceHit {
            image: head,
            anchored: true,
            eye_aligned: false,
        });
    }
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
    Ok(FaceHit {
        image: plate.crop(x0, y0, fw, fh),
        anchored: false,
        eye_aligned: false,
    })
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
    let mut anchored = 0usize;
    let mut eye_locked = 0usize;
    for (i, (plate, bbox)) in plates.iter().zip(boxes.iter()).enumerate() {
        match extract_face_region(plate, *bbox, i) {
            Ok(hit) => {
                anchored += hit.anchored as usize;
                eye_locked += hit.eye_aligned as usize;
                aligned.push(align(&hit.image));
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
                "face regions: {} usable of {} (head-anchored {}/{}, eye-locked {}/{}, rest top-{:.0}%-of-box heuristic, ≥{}px)",
                n_donors,
                n_donors + n_refused,
                anchored,
                n_donors,
                eye_locked,
                n_donors,
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

/// Eye-locked synthesis: same pipeline, but only donors whose scale
/// and translation were set from a measured eye pair contribute.
/// Head-blob and heuristic crops blur the median by construction, so
/// they are refused here (counted, with reasons) instead of averaged
/// in. Below [`MIN_FACES`] eye-locked donors: refusal, never a thin
/// average.
pub fn synthesize_locked(
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
    let mut hits = Vec::new();
    let mut refused = Vec::new();
    for (i, (plate, bbox)) in plates.iter().zip(boxes.iter()).enumerate() {
        match extract_face_region(plate, *bbox, i) {
            Ok(hit) if hit.eye_aligned => hits.push((i, hit)),
            Ok(_) => refused.push(format!("donor {}: no eye lock — excluded (would blur)", i)),
            Err(e) => refused.push(e),
        }
    }
    if hits.len() < MIN_FACES {
        return Err(format!(
            "INSUFFICIENT: {}/{} eye-locked faces — need {} ({} refused/excluded)",
            hits.len(),
            plates.len(),
            MIN_FACES,
            refused.len()
        ));
    }
    let donors: Vec<usize> = hits.iter().map(|(i, _)| *i).collect();
    let aligned: Vec<Image> = hits.iter().map(|(_, h)| align(&h.image)).collect();
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
                "face regions: {} eye-locked of {} (scale+translation from measured eye pairs)",
                n_donors,
                n_donors + n_refused
            ),
            format!(
                "align: resize to {}x{} (eye-line at 38% height, no rotation correction)",
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
                &extract_face_region(&photo(a, bg, 60, 120, 160), person_box(), 0)
                    .unwrap()
                    .image,
            ));
        }
        faces.push(align(
            &extract_face_region(&photo(b, bg, 60, 120, 160), person_box(), 8)
                .unwrap()
                .image,
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
    fn heads_anchor_on_skin_otherwise_heuristic() {
        // Synthetic head (skin disc on flat backdrop): the blob is
        // measured, so the hit is anchored and roughly head-sized.
        let bg = Rgb::new(60, 80, 120);
        let skin = Rgb::new(200, 150, 115);
        let plate = photo(skin, bg, 60, 120, 160);
        let hit = extract_face_region(&plate, person_box(), 0).unwrap();
        assert!(hit.anchored, "measured head must anchor");
        assert!(hit.image.width >= MIN_FACE_PX && hit.image.width == hit.image.height);
        // Blank plate: no blob, heuristic falls below the pixel gate.
        let flat = Image::blank(120, 160, bg);
        assert!(head_square(&flat, person_box()).is_none());
    }

    #[test]
    fn eye_pair_locks_symmetric_dark_marks() {
        // Two dark discs, level and symmetric: the landmark fires with
        // the pair straddling the center.
        let mut head = Image::blank(120, 120, Rgb::new(200, 150, 115));
        head.draw_disc(42, 50, 6, Rgb::new(20, 14, 10));
        head.draw_disc(78, 50, 6, Rgb::new(20, 14, 10));
        let ((lx, ly), (rx, ry)) = eye_pair(&head).expect("pair must lock");
        assert!(lx < rx);
        assert!(((lx + rx) / 2).abs_diff(60) <= 9, "must straddle center");
        assert!(ly.abs_diff(ry) <= 14, "must be level");
        assert!((rx - lx) >= 18 && (rx - lx) <= 72);
        // Featureless skin: no pair, no lock.
        assert!(eye_pair(&Image::blank(120, 120, Rgb::new(200, 150, 115))).is_none());
        // Full synthetic head extracts eye-aligned end to end.
        let bg = Rgb::new(60, 80, 120);
        let mut plate = Image::blank(160, 200, bg);
        plate.draw_disc(80, 60, 30, Rgb::new(200, 150, 115));
        plate.draw_disc(68, 55, 5, Rgb::new(20, 14, 10));
        plate.draw_disc(92, 55, 5, Rgb::new(20, 14, 10));
        let hit = extract_face_region(&plate, (0.1, 0.05, 0.9, 0.95), 0).unwrap();
        assert!(hit.anchored && hit.eye_aligned, "eyes must lock the crop");
    }

    #[test]
    fn locked_synthesis_needs_eye_locks() {
        // Nine donors with measurable eye pairs: locked synthesis runs
        // with all of them.
        let bg = Rgb::new(60, 80, 120);
        let mk_eye = || {
            let mut p = Image::blank(160, 200, bg);
            p.draw_disc(80, 60, 30, Rgb::new(200, 150, 115));
            p.draw_disc(68, 55, 5, Rgb::new(20, 14, 10));
            p.draw_disc(92, 55, 5, Rgb::new(20, 14, 10));
            p
        };
        let plates: Vec<Image> = (0..9).map(|_| mk_eye()).collect();
        let boxes = vec![(0.1, 0.05, 0.9, 0.95); 9];
        let (img, log) = synthesize_locked(&plates, &boxes, 3).unwrap();
        assert_eq!(log.donors.len(), 9);
        assert_eq!((img.width, img.height), (CANON_W, CANON_H));
        // Same heads with the eyes painted over: blobs still anchor,
        // but nothing locks — refusal names the eye gate.
        let mk_plain = || {
            let mut p = Image::blank(160, 200, bg);
            p.draw_disc(80, 60, 30, Rgb::new(200, 150, 115));
            p
        };
        let plains: Vec<Image> = (0..9).map(|_| mk_plain()).collect();
        let err = synthesize_locked(&plains, &boxes, 3).expect_err("must refuse");
        assert!(err.contains("eye-locked"), "wrong refusal: {}", err);
    }

    #[test]
    fn graft_stays_within_measured_budget() {
        // Flat-agreement donors: deviation ~0, graft must not invent.
        let bg = Rgb::new(60, 80, 120);
        let skin = Rgb::new(200, 150, 115);
        let faces: Vec<Image> = (0..8)
            .map(|_| {
                align(
                    &extract_face_region(&photo(skin, bg, 60, 120, 160), person_box(), 0)
                        .unwrap()
                        .image,
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
