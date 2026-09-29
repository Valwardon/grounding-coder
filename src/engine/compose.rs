//! Photographic portraits from sourced plates: segment the subject
//! with classical skin-locus + morphology (no model), composite onto
//! a studio backdrop with feathered edges and a matched finish.
//!
//! Every output pixel is either source-plate pixels or generated
//! backdrop/grade — the [`ComposeLog`] records which, with the
//! subject bbox and every op parameter. A composite never claims to
//! be a photograph of a new person; it is a documented edit of a
//! sourced plate, and the log is the difference.
use super::vision::{Image, Rgb};

/// How one composite was made, step by step.
#[derive(Debug, Clone)]
pub struct ComposeLog {
    /// Source plate file (plus its own provenance manifest).
    pub source: String,
    /// Subject bbox in plate pixels (x0, y0, x1, y1).
    pub subject_bbox: (u32, u32, u32, u32),
    /// Pasted subject rect in output pixels (ox, oy, sw, sh).
    pub pasted: (u32, u32, u32, u32),
    /// Subject area as a fraction of the plate.
    pub subject_fraction: f64,
    /// Ops applied, in order, with parameters.
    pub ops: Vec<String>,
}

/// Vertical gradient studio backdrop.
pub fn studio_backdrop(w: u32, h: u32, top: Rgb, bottom: Rgb) -> Image {
    let mut img = Image::blank(w, h, Rgb::new(0, 0, 0));
    for y in 0..h {
        let t = y as f64 / (h.max(2) - 1) as f64;
        let c = Rgb::new(
            (top.r as f64 * (1.0 - t) + bottom.r as f64 * t).round() as u8,
            (top.g as f64 * (1.0 - t) + bottom.g as f64 * t).round() as u8,
            (top.b as f64 * (1.0 - t) + bottom.b as f64 * t).round() as u8,
        );
        for x in 0..w {
            img.set(x, y, c);
        }
    }
    img
}

/// Build a portrait: find the subject, frame them, finish the photo.
/// Refuses when no plausible subject exists rather than compositing
/// furniture.
///
/// Composition priors, stated: the subject is the largest skin
/// region, occupying 0.1%–50% of the frame with its centroid in the
/// upper three-quarters. A mahogany table dead-center would pass;
/// the log records the bbox so the assumption stays checkable.
/// A found subject: bbox in the searched image, area, part count.
/// A found subject: bbox, area, assembly parts, centroid.
pub(crate) struct Subject {
    pub(crate) x0: u32,
    pub(crate) y0: u32,
    pub(crate) x1: u32,
    pub(crate) y1: u32,
    pub(crate) area: u64,
    pub(crate) parts: usize,
    pub(crate) cx: f64,
    pub(crate) cy: f64,
}

pub(crate) fn find_subject(img: &Image) -> Result<Subject, String> {
    let (w, h) = (img.width, img.height);
    let frame_area = (w * h) as f64;
    let cleaned = Image::morph_close(&Image::morph_open(&img.skin_mask(), w, h, 2), w, h, 2);
    // Subject assembly: the largest blob seeds, nearby fragments
    // (head split from torso by hair, hands split by sleeves) merge
    // back in. Largest-blob-only framing decapitated a sitter in
    // testing — torso composited, head left on the plate.
    let blobs = Image::all_blobs(&cleaned, w, h, (frame_area * 0.0005) as usize);
    let (x0, y0, x1, y1, area, parts) = Image::assemble_subject(&blobs, w, h, 0.06)
        .ok_or_else(|| "no skin region found".to_string())?;
    let area_u = area as u64;
    let fraction = area as f64 / frame_area;
    if !(0.001..=0.5).contains(&fraction) {
        return Err(format!(
            "subject implausible: {:.3} of frame is skin",
            fraction
        ));
    }
    let (cx, cy) = ((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0);
    if cy > h as f64 * 0.75 {
        return Err("skin centroid in lower quarter — likely not a portrait subject".to_string());
    }
    // Coherence: one compact blob, or a merged union that still fits
    // comfortably inside the frame. Speckle fields spanning the whole
    // plate fail the union bound — unless anchored: the seed alone is
    // a large compact mass (a torso cropped by the frame, the classic
    // portrait crop), so fragments joining it are trusted further.
    // The fraction gate above still caps full-frame skin washes.
    let union_area = (x1 - x0 + 1) as f64 * (y1 - y0 + 1) as f64;
    let fill = area as f64 / union_area;
    let seed_fill = {
        let (sa, sx0, sy0, sx1, sy1) = (
            blobs[0].0 as f64,
            blobs[0].1,
            blobs[0].2,
            blobs[0].3,
            blobs[0].4,
        );
        sa / ((sx1 - sx0 + 1) as f64 * (sy1 - sy0 + 1) as f64)
    };
    let anchored = blobs[0].0 as f64 >= frame_area * 0.05 && seed_fill >= 0.40;
    if parts == 1 {
        if fill < 0.30 {
            return Err(format!("subject scattered (compactness {:.2})", fill));
        }
    } else if anchored {
        // The anchor carries the decision, so the union bound is a
        // backstop (literally full-bleed only) and fill does the work:
        // a close-up figure cropped by the frame fills its union
        // sparsely but genuinely; a speckle halo around a seed does
        // not reach 0.15.
        if union_area > frame_area * 0.99 || fill < 0.15 {
            return Err(format!(
                "anchored subject incoherent (union {:.2} of frame, fill {:.2})",
                union_area / frame_area,
                fill
            ));
        }
    } else if union_area > frame_area * 0.80 || fill < 0.15 {
        return Err(format!(
            "assembled subject incoherent ({} parts, fill {:.2})",
            parts, fill
        ));
    }
    Ok(Subject {
        x0,
        y0,
        x1,
        y1,
        area: area_u,
        parts,
        cx,
        cy,
    })
}

pub fn compose_portrait(
    plate: &Image,
    source: &str,
    out_w: u32,
    out_h: u32,
) -> Result<(Image, ComposeLog), String> {
    let (pw, ph) = (plate.width, plate.height);
    // Collage handling: gutter seams split the plate into photo cells
    // and each cell is searched independently. Merging across a seam
    // once composited half of one photo with half of another — the
    // largest passing cell wins and the log names it.
    let seams = plate.find_seams(60.0);
    let mut bands: Vec<(u32, u32)> = Vec::new();
    let mut top = 0u32;
    for (s0, s1) in &seams {
        if *s0 > top + ph / 10 {
            bands.push((top, s0 - 1));
        }
        top = s1 + 1;
    }
    if top < ph.saturating_sub(ph / 10) {
        bands.push((top, ph - 1));
    }
    if bands.is_empty() {
        bands.push((0, ph - 1));
    }
    let mut best: Option<(usize, Subject)> = None;
    let mut cell_notes = Vec::new();
    for (i, (cy0, cy1)) in bands.iter().enumerate() {
        let cell = plate.crop(0, *cy0, pw, cy1 - cy0 + 1);
        match find_subject(&cell) {
            Ok(s) => {
                cell_notes.push(format!(
                    "cell {} rows {}-{}: subject {} parts",
                    i, cy0, cy1, s.parts
                ));
                match &best {
                    Some((_, b)) if b.area >= s.area => {}
                    _ => best = Some((i, s)),
                }
            }
            Err(e) => cell_notes.push(format!("cell {} rows {}-{}: {}", i, cy0, cy1, e)),
        }
    }
    let (cell_idx, subj) = best.ok_or_else(|| {
        format!(
            "no cell holds a subject — refusing ({})",
            cell_notes.join("; ")
        )
    })?;
    let (cell_y0, cell_y1) = bands[cell_idx];
    // Map the winning bbox back to plate coordinates.
    let (x0, y0, x1, y1) = (subj.x0, subj.y0 + cell_y0, subj.x1, subj.y1 + cell_y0);
    let (area, parts, cx, cy) = (subj.area, subj.parts, subj.cx, subj.cy + cell_y0 as f64);
    let frame_area = (pw * ph) as f64;
    let fraction = area as f64 / frame_area;

    // Frame: expand the bbox, clamp to the winning cell (never bleed
    // into the neighboring photo), crop, fit to 88% of output height
    // preserving aspect.
    let mx = ((x1 - x0) as f64 * 0.6) as u32;
    let my = ((y1 - y0) as f64 * 0.6) as u32;
    let fx0 = x0.saturating_sub(mx);
    let fy0 = y0.saturating_sub(my).max(cell_y0);
    let fx1 = (x1 + mx).min(pw - 1);
    let fy1 = (y1 + my).min(cell_y1);
    let crop = plate.crop(fx0, fy0, fx1 - fx0 + 1, fy1 - fy0 + 1);
    // Fit inside (out_w, 88% out_h) preserving aspect: clamping width
    // without rescaling height stretches faces — the test caught a
    // victim (827px squeezed to 640 while height stayed 704).
    let target_h = out_h * 88 / 100;
    let fit =
        (target_h as f64 / crop.height.max(1) as f64).min(out_w as f64 / crop.width.max(1) as f64);
    let sw = ((crop.width as f64 * fit).round() as u32).max(1);
    let sh = ((crop.height as f64 * fit).round() as u32).max(1);
    let subject = crop.resize_smooth(sw, sh);

    let ox = (out_w.saturating_sub(subject.width)) / 2;
    let oy = out_h.saturating_sub(subject.height) / 2 / 2;
    let mut mask = vec![false; (out_w * out_h) as usize];
    for y in oy..(oy + subject.height).min(out_h) {
        for x in ox..(ox + subject.width).min(out_w) {
            mask[(y * out_w + x) as usize] = true;
        }
    }
    // Feather scales with output: 4px at 640 wide, more in HD.
    let feather_r = (out_w / 160).max(2);
    let alpha = Image::feather(&mask, out_w, out_h, feather_r);
    let mut backdrop = studio_backdrop(out_w, out_h, Rgb::new(72, 72, 82), Rgb::new(28, 28, 34));
    // One light: lift the subject toward the backdrop's mean luma so
    // the two halves read as one photo. Logged, bounded ±40.
    let bg_mean = backdrop.mean_brightness();
    let mut subject_only = subject.clone();
    let lift = subject_only.match_luma(bg_mean);
    let mut fg = Image::blank(out_w, out_h, Rgb::new(0, 0, 0));
    fg.overlay(&subject_only, ox, oy);
    let mut out = Image::composite(&fg, &backdrop, &alpha)?;
    // Contact shadow under the subject's feet grounds the cutout.
    let shadow = Image::contact_shadow(
        out_w,
        out_h,
        (
            ox,
            oy + subject.height,
            ox + subject.width,
            oy + subject.height,
        ),
        0.35,
    );
    out.apply_shadow(&shadow);
    out.grain(0xC0FFEE, 4);
    out.vignette(0.22);

    let log = ComposeLog {
        source: source.to_string(),
        subject_bbox: (fx0, fy0, fx1, fy1),
        pasted: (ox, oy, subject.width, subject.height),
        subject_fraction: fraction,
        ops: vec![
            format!(
                "collage check: {} seam(s), composing from cell {} ({})",
                seams.len(),
                cell_idx,
                cell_notes.join("; ")
            ),
            format!("skin_mask Cb[77,127] Cr[133,173] Y>40 on {}x{}", pw, ph),
            "morph_open x2 then morph_close x2".to_string(),
            format!(
                "assembled subject: area={} centroid=({:.0},{:.0}) parts={}",
                area, cx, cy, parts
            ),
            format!(
                "frame bbox ({},{})-({},{}) with 60% margin",
                fx0, fy0, fx1, fy1
            ),
            format!(
                "fit {}x{} keep-aspect → {}x{} at ({},{})",
                crop.width, crop.height, subject.width, subject.height, ox, oy
            ),
            format!(
                "feather r{}, luma lift {:+.1}, contact shadow, grain(0xC0FFEE,4), vignette(0.22)",
                feather_r, lift
            ),
        ],
    };
    Ok((out, log))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skin_patch() -> Image {
        // Skin rect on a blue field: the only skin in the frame.
        let mut img = Image::blank(40, 30, Rgb::new(60, 110, 200));
        img.draw_rect(10, 8, 12, 10, Rgb::new(200, 150, 115));
        img
    }

    #[test]
    fn pipeline_frames_the_subject() {
        let plate = skin_patch();
        let (out, log) = compose_portrait(&plate, "synthetic", 64, 80).expect("pipeline");
        assert_eq!((out.width, out.height), (64, 80));
        // Aspect preserved end to end: a 3:1 wide subject stays ~3:1
        // on the output — the old clamp-then-force path stretched one
        // 827px-wide victim to 640 while keeping height 704.
        let mut wide = Image::blank(60, 30, Rgb::new(60, 110, 200));
        wide.draw_rect(5, 10, 30, 10, Rgb::new(200, 150, 115));
        let (_, wlog) = compose_portrait(&wide, "synthetic", 64, 80).expect("wide pipeline");
        let (_, _, sw, sh) = wlog.pasted;
        let ratio = sw as f64 / sh as f64;
        assert!(
            ratio > 2.0 && ratio < 4.0,
            "aspect kept: {}x{} → {:.2}",
            sw,
            sh,
            ratio
        );
        // Bbox covers the rect (10..22, 8..18) with margin, clamped.
        let (x0, y0, x1, y1) = log.subject_bbox;
        assert!(
            x0 <= 10 && y0 <= 8 && x1 >= 21 && y1 >= 17,
            "{:?}",
            log.subject_bbox
        );
        assert!(
            (log.subject_fraction - 120.0 / 1200.0).abs() < 0.02,
            "{}",
            log.subject_fraction
        );
        assert_eq!(log.ops.len(), 7);
        // Single photo: no seams, one cell.
        assert!(log.ops[0].contains("0 seam(s)"), "{:?}", log.ops);
        // Subject pixels survived at the output center-top.
        let center = out.get(32, 20).expect("pixel");
        assert!(center.r > 100 && center.b < 160, "{:?}", center);
    }

    #[test]
    fn collage_composes_one_photo_not_both() {
        // Two stacked "photos": big skin rect on top, small below,
        // split by a white gutter. The seam must not merge them.
        let mut plate = Image::blank(60, 120, Rgb::new(40, 60, 120));
        plate.draw_rect(15, 10, 30, 30, Rgb::new(200, 150, 115));
        for x in 0..60 {
            plate.set(x, 60, Rgb::new(255, 255, 255));
            plate.set(x, 61, Rgb::new(255, 255, 255));
        }
        plate.draw_rect(20, 80, 12, 12, Rgb::new(200, 150, 115));
        let (out, log) = compose_portrait(&plate, "synthetic", 64, 80).expect("collage");
        assert!(log.ops[0].contains("1 seam(s)"), "{:?}", log.ops);
        assert!(log.ops[0].contains("cell 0"), "{:?}", log.ops);
        // Subject bbox stays in the top cell (rows < 60).
        assert!(log.subject_bbox.3 < 60, "{:?}", log.subject_bbox);
        assert_eq!((out.width, out.height), (64, 80));
    }

    #[test]
    fn anchored_closeup_accepts_wide_union() {
        // Portrait crop arithmetic, exact: seed 2212px + three frags
        // = 4900/10000 (fraction ok), union 95x85 = 80.75% (past the
        // free-floating 80% bar, inside the anchored 95% bar).
        // Under the old rule this exact layout refused.
        let mut plate = Image::blank(100, 100, Rgb::new(40, 60, 120));
        let skin = Rgb::new(200, 150, 115);
        plate.draw_rect(1, 39, 79, 28, skin);
        plate.draw_rect(86, 46, 10, 20, skin);
        plate.draw_rect(22, 9, 48, 24, skin);
        plate.draw_rect(22, 67, 48, 27, skin);
        let (out, log) = compose_portrait(&plate, "synthetic", 64, 80).expect("anchored");
        assert_eq!((out.width, out.height), (64, 80));
        assert!(
            log.subject_fraction > 0.45 && log.subject_fraction <= 0.5,
            "{}",
            log.subject_fraction
        );
    }

    #[test]
    fn no_subject_refuses() {
        let flat = Image::blank(40, 30, Rgb::new(60, 110, 200));
        assert!(compose_portrait(&flat, "synthetic", 64, 80).is_err());
    }

    #[test]
    fn skin_toned_background_refuses_honestly() {
        // The JFK plate's umber background sits inside the skin locus
        // (measured: corners read Cb≈103 Cr≈150) — chroma cannot
        // separate subject from backdrop here, and the pipeline must
        // refuse rather than composite the furniture. Eye-anchored
        // segmentation is the named follow-up; this test pins the
        // refusal until it exists.
        let path = std::path::Path::new("samples/plates/plate-00.bmp");
        let plate = Image::load_bmp(path).expect("committed sample plate must load");
        let err = compose_portrait(&plate, "plate-00", 320, 400).expect_err("must refuse");
        assert!(err.contains("implausible"), "wrong refusal: {}", err);
    }
}
