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
pub fn compose_portrait(
    plate: &Image,
    source: &str,
    out_w: u32,
    out_h: u32,
) -> Result<(Image, ComposeLog), String> {
    let (pw, ph) = (plate.width, plate.height);
    let frame_area = (pw * ph) as f64;
    let cleaned = Image::morph_close(&Image::morph_open(&plate.skin_mask(), pw, ph, 2), pw, ph, 2);
    let blob = Image::largest_blob(&cleaned, pw, ph);
    let (area, cx, cy, x0, y0, x1, y1) = Image::blob_stats(&blob, pw)
        .ok_or_else(|| "no skin region found — refusing".to_string())?;
    let fraction = area as f64 / frame_area;
    if !(0.001..=0.5).contains(&fraction) {
        return Err(format!(
            "subject implausible: {:.3} of frame is skin — refusing",
            fraction
        ));
    }
    if cy > ph as f64 * 0.75 {
        return Err("skin centroid in lower quarter — likely not a portrait subject".to_string());
    }
    // Compactness: a face-and-hands cluster fills much of its bbox;
    // a speckle field scattered across an umber ground does not.
    // Painted portraits routinely fail here — that refusal is the
    // detector telling the truth about its own limits.
    let bbox_area = (x1 - x0 + 1) as f64 * (y1 - y0 + 1) as f64;
    if area as f64 / bbox_area < 0.30 {
        return Err(format!(
            "subject scattered (compactness {:.2}) — no coherent region, refusing",
            area as f64 / bbox_area
        ));
    }

    // Frame: expand the bbox, clamp to the plate, crop, fit to 88% of
    // output height preserving aspect.
    let mx = ((x1 - x0) as f64 * 0.6) as u32;
    let my = ((y1 - y0) as f64 * 0.6) as u32;
    let fx0 = x0.saturating_sub(mx);
    let fy0 = y0.saturating_sub(my);
    let fx1 = (x1 + mx).min(pw - 1);
    let fy1 = (y1 + my).min(ph - 1);
    let crop = plate.crop(fx0, fy0, fx1 - fx0 + 1, fy1 - fy0 + 1);
    // Fit inside (out_w, 88% out_h) preserving aspect: clamping width
    // without rescaling height stretches faces — the test caught a
    // victim (827px squeezed to 640 while height stayed 704).
    let target_h = out_h * 88 / 100;
    let fit =
        (target_h as f64 / crop.height.max(1) as f64).min(out_w as f64 / crop.width.max(1) as f64);
    let sw = ((crop.width as f64 * fit).round() as u32).max(1);
    let sh = ((crop.height as f64 * fit).round() as u32).max(1);
    let subject = crop.resize(sw, sh);

    let mut fg = Image::blank(out_w, out_h, Rgb::new(0, 0, 0));
    let ox = (out_w.saturating_sub(subject.width)) / 2;
    let oy = out_h.saturating_sub(subject.height) / 2 / 2;
    fg.overlay(&subject, ox, oy);
    let mut mask = vec![false; (out_w * out_h) as usize];
    for y in oy..(oy + subject.height).min(out_h) {
        for x in ox..(ox + subject.width).min(out_w) {
            mask[(y * out_w + x) as usize] = true;
        }
    }
    let alpha = Image::feather(&mask, out_w, out_h, 4);
    let backdrop = studio_backdrop(out_w, out_h, Rgb::new(72, 72, 82), Rgb::new(28, 28, 34));
    let mut out = Image::composite(&fg, &backdrop, &alpha)?;
    out.grain(0xC0FFEE, 4);
    out.vignette(0.22);

    let log = ComposeLog {
        source: source.to_string(),
        subject_bbox: (fx0, fy0, fx1, fy1),
        pasted: (ox, oy, subject.width, subject.height),
        subject_fraction: fraction,
        ops: vec![
            format!("skin_mask Cb[77,127] Cr[133,173] Y>40 on {}x{}", pw, ph),
            "morph_open x2 then morph_close x2".to_string(),
            format!("largest blob: area={} centroid=({:.0},{:.0})", area, cx, cy),
            format!(
                "frame bbox ({},{})-({},{}) with 60% margin",
                fx0, fy0, fx1, fy1
            ),
            format!(
                "fit {}x{} keep-aspect → {}x{} at ({},{})",
                crop.width, crop.height, subject.width, subject.height, ox, oy
            ),
            "feather radius 4, studio backdrop, grain(0xC0FFEE,4), vignette(0.22)".to_string(),
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
        assert_eq!(log.ops.len(), 6);
        // Subject pixels survived at the output center-top.
        let center = out.get(32, 20).expect("pixel");
        assert!(center.r > 100 && center.b < 160, "{:?}", center);
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
