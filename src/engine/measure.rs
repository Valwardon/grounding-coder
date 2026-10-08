//! Measuring researched photographs: the knowledge lives in the photo.
//!
//! [`super::scene::scene_from_facts`] draws nothing this module did not
//! first measure. No proportion tables, no anatomy, no pose library —
//! every dimension, color and light direction in the render comes out
//! of a researched reference plate and lands in a JSON file you can
//! read in a diff. The photo is the visual cue; facts are its
//! structured reading, committed as memory.
//!
//! ```text
//! plate.bmp + provenance → measure → VisualFacts (committed JSON)
//! ```
//!
//! Thresholding is Otsu's method — the photo picks its own cut — so
//! there is no tuning constant to carry a subject. A subject must sit
//! inside its ground, touching no frame edge; a region that fills the
//! frame is a scene. Both it and a plate with no separable subject are
//! refused with what to clarify — never guessed (see `AGENTS.md`).
use super::vision::{Image, Rgb};
use serde::{Deserialize, Serialize};

/// Silhouette samples carried from photo to scene, bottom row first.
/// Coarse enough to read in a diff, fine enough to keep a pose's taper.
pub const PROFILE_ROWS: usize = 24;

/// Provenance for one plate: straight from the research bank's
/// `provenance.json`, so a fact always names its source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlateMeta {
    pub title: String,
    #[serde(default)]
    pub query: String,
    #[serde(default)]
    pub source_url: String,
    #[serde(default)]
    pub page_url: String,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub license: String,
    /// Plate file inside the bank directory.
    #[serde(default)]
    pub file: String,
}

/// Colors read from the plate: what surrounds the subject, what the
/// subject itself is made of, and what sits under it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Palette {
    /// Mean color of the subject's own pixels.
    pub subject: [u8; 3],
    /// Median of the plate's border: whatever surrounds the subject.
    pub surround: [u8; 3],
    /// Mean of the plate's bottom band — floor, ground, foreground.
    pub ground: [u8; 3],
}

/// Everything the plate says about where and how wide the subject is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubjectFacts {
    /// Which measurement separated subject from ground: `"border"`
    /// (color distance to the plate's border) or `"dominant"` (color
    /// distance to the plate's dominant color).
    pub method: String,
    /// Bounding box in plate pixels: `x0, y0, x1, y1` inclusive.
    pub bbox: [u32; 4],
    /// Subject centroid as a fraction of plate width and height.
    pub centroid: [f64; 2],
    /// Subject height as a fraction of the plate height.
    pub height_frac: f64,
    /// Subject width as a fraction of the plate width.
    pub width_frac: f64,
    /// Subject pixels / plate pixels — how much of the plate it is.
    pub area_frac: f64,
    /// Silhouette width per bucket as a fraction of the bbox width,
    /// bottom row first. Measurement, never anatomy: buckets the
    /// photograph showed are counted from the subject's own pixels;
    /// buckets it hid (a black dress on a dark ground) carry the
    /// width between measured neighbours and say so in
    /// `profile_covered`, while `row_colors` keeps the color the
    /// plate actually paints there.
    pub profile: Vec<f64>,
    /// Subject pixels on the brighter right half minus the left half,
    /// normalized to -1..1. Pure measurement: +1 = right side lit.
    pub side_balance: f64,
    /// Brighter top half minus bottom half, normalized to -1..1.
    /// Pure measurement: +1 = lit from above.
    pub vertical_balance: f64,
    /// Masked pixels behind these numbers (evidence strength).
    pub pixels: u64,
    /// Every foreground region the plate showed, recorded with bbox,
    /// size and mean color. The subject is the column-clustered
    /// subset of these; anything beside the figure stays on the
    /// record so a reader can see what was measured and left out.
    #[serde(default)]
    pub components: Vec<ComponentFacts>,
    /// Per silhouette bucket: did the photograph actually show the
    /// subject at this height, or is the width carried between
    /// measured neighbours?
    #[serde(default)]
    pub profile_covered: Vec<bool>,
    /// Measured color per silhouette bucket — what the plate paints
    /// at that height, dark dress included.
    #[serde(default)]
    pub row_colors: Vec<[u8; 3]>,
}

/// The committed unit of visual memory: one plate, read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisualFacts {
    pub plate: PlateMeta,
    /// Plate dimensions in pixels, the space `bbox` lives in.
    pub plate_size: [u32; 2],
    pub subject: SubjectFacts,
    pub palette: Palette,
}

/// Median color of the plate's one-pixel border — what surrounds the
/// subject when the subject doesn't touch the edge.
fn border_median(img: &Image) -> Rgb {
    let (w, h) = (img.width, img.height);
    let mut rs: Vec<u8> = Vec::new();
    let mut gs: Vec<u8> = Vec::new();
    let mut bs: Vec<u8> = Vec::new();
    let mut push = |p: Rgb| {
        rs.push(p.r);
        gs.push(p.g);
        bs.push(p.b);
    };
    for x in 0..w {
        if let Some(p) = img.get(x, 0) {
            push(p);
        }
        if let Some(p) = img.get(x, h - 1) {
            push(p);
        }
    }
    for y in 1..h.saturating_sub(1) {
        if let Some(p) = img.get(0, y) {
            push(p);
        }
        if let Some(p) = img.get(w - 1, y) {
            push(p);
        }
    }
    let median = |v: &mut Vec<u8>| -> u8 {
        v.sort_unstable();
        v.get(v.len() / 2).copied().unwrap_or(0)
    };
    Rgb::new(median(&mut rs), median(&mut gs), median(&mut bs))
}

/// Otsu's threshold over a 256-bin histogram: the split maximizing
/// between-class variance. The plate picks its own cut, so no tuning
/// constant can carry a subject through.
fn otsu(hist: &[u64; 256]) -> u8 {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return 0;
    }
    let total = total as f64;
    let mut sum = 0.0;
    for (i, &c) in hist.iter().enumerate() {
        sum += i as f64 * c as f64;
    }
    let (mut sum_b, mut w_b, mut best, mut thr) = (0.0f64, 0.0f64, -1.0f64, 0u8);
    for (t, &count) in hist.iter().enumerate() {
        let w_f = count as f64;
        if w_f == 0.0 {
            continue;
        }
        let w_b_next = w_b + w_f;
        let w_f_next = total - w_b_next;
        if w_f_next <= 0.0 {
            break;
        }
        sum_b += t as f64 * w_f;
        let mean_b = sum_b / w_b_next;
        let mean_f = (sum - sum_b) / w_f_next;
        let between = w_b_next * w_f_next * (mean_b - mean_f) * (mean_b - mean_f);
        if between > best {
            best = between;
            thr = t as u8;
        }
        w_b = w_b_next;
    }
    thr
}

/// Subject mask by color distance to a reference color, split at the
/// plate's own Otsu threshold. Returns `mask` row-major.
fn mask_far_from(img: &Image, reference: Rgb) -> Vec<bool> {
    let mut hist = [0u64; 256];
    let mut bins = vec![0u8; (img.width * img.height) as usize];
    for y in 0..img.height {
        for x in 0..img.width {
            let Some(px) = img.get(x, y) else { continue };
            let d = px.distance2(&reference);
            // RMS per channel, 0..255.
            let bin = ((d / 3) as f64).sqrt().round().clamp(0.0, 255.0) as u8;
            hist[bin as usize] += 1;
            bins[(y * img.width + x) as usize] = bin;
        }
    }
    let thr = otsu(&hist);
    bins.iter().map(|&b| b > thr).collect()
}

/// Mean color of pixel indices.
fn mean_of(img: &Image, pixels: &[usize]) -> [u8; 3] {
    let (mut r, mut g, mut b) = (0u64, 0u64, 0u64);
    let w = img.width;
    for &i in pixels {
        if let Some(p) = img.get(i as u32 % w, i as u32 / w) {
            r += p.r as u64;
            g += p.g as u64;
            b += p.b as u64;
        }
    }
    let n = pixels.len().max(1) as u64;
    [(r / n) as u8, (g / n) as u8, (b / n) as u8]
}

/// One connected foreground region and where it sits on the plate.
#[derive(Debug, Clone)]
struct Component {
    pixels: Vec<usize>,
    bbox: [u32; 4],
}

/// Foreground regions as recorded on the facts file: proof that the
/// subject was chosen from what the plate showed, not assumed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentFacts {
    /// `x0, y0, x1, y1` inclusive, plate pixels.
    pub bbox: [u32; 4],
    pub area: u64,
    pub mean: [u8; 3],
    /// True when this region is part of the read subject (it shares
    /// columns with the figure); false for whatever stood beside it.
    pub in_subject: bool,
}

/// 8-connected components of the mask. Regions smaller than 0.1% of
/// the plate are sensor speckle, not content.
fn components(mask: &[bool], w: u32, h: u32) -> Vec<Component> {
    let total = (w * h) as usize;
    let min_area = total / 1000;
    let mut seen = vec![false; total];
    let mut out = Vec::new();
    for start in 0..total {
        if !mask[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        let mut stack = vec![start];
        let mut pixels = Vec::new();
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        while let Some(i) = stack.pop() {
            pixels.push(i);
            let (x, y) = (i as u32 % w, i as u32 / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let ni = (ny as u32 * w + nx as u32) as usize;
                    if mask[ni] && !seen[ni] {
                        seen[ni] = true;
                        stack.push(ni);
                    }
                }
            }
        }
        if pixels.len() >= min_area {
            out.push(Component {
                pixels,
                bbox: [x0, y0, x1, y1],
            });
        }
    }
    out
}

/// The subject of a plate is often photographed in pieces — a face
/// here, a skirt there, with a black silk dress the plate cannot tell
/// from its own background in between. Pieces that share columns
/// (overlap in x) are stacked views of one thing and are read as one
/// subject; a region beside the figure with no column in common — a
/// coat-of-arms, a distant object — is recorded but not claimed.
fn cluster_columns(comps: &[Component]) -> Vec<usize> {
    let n = comps.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for a in 0..n {
        for b in (a + 1)..n {
            let (ca, cb) = (comps[a].bbox, comps[b].bbox);
            if ca[0] <= cb[2] && cb[0] <= ca[2] {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        match groups.iter_mut().find(|(r, _)| *r == root) {
            Some((_, members)) => members.push(i),
            None => groups.push((root, vec![i])),
        }
    }
    groups
        .into_iter()
        .max_by_key(|(_, members)| {
            members
                .iter()
                .map(|&i| comps[i].pixels.len())
                .sum::<usize>()
        })
        .map(|(_, members)| members)
        .unwrap_or_default()
}

/// Mean color of a pixel row band.
fn mean_band(img: &Image, y0: u32, y1: u32) -> [u8; 3] {
    let (mut r, mut g, mut b, mut n) = (0u64, 0u64, 0u64, 0u64);
    for y in y0..y1.min(img.height) {
        for x in 0..img.width {
            if let Some(p) = img.get(x, y) {
                r += p.r as u64;
                g += p.g as u64;
                b += p.b as u64;
                n += 1;
            }
        }
    }
    let n = n.max(1);
    [(r / n) as u8, (g / n) as u8, (b / n) as u8]
}

/// What one plate says about its subject's silhouette: width, whether
/// the photograph actually showed pixels there, and the color the
/// plate has at that height.
#[derive(Debug, Clone)]
struct Buckets {
    profile: Vec<f64>,
    covered: Vec<bool>,
    row_colors: Vec<[u8; 3]>,
}

/// Mean color of plate pixels in rows `y0..=y1`, columns `xa..=xb`.
fn mean_span(img: &Image, y0: u32, y1: u32, xa: u32, xb: u32) -> [u8; 3] {
    let (mut r, mut g, mut b, mut n) = (0u64, 0u64, 0u64, 0u64);
    for y in y0..=y1.min(img.height.saturating_sub(1)) {
        for x in xa..=xb.min(img.width.saturating_sub(1)) {
            if let Some(p) = img.get(x, y) {
                r += p.r as u64;
                g += p.g as u64;
                b += p.b as u64;
                n += 1;
            }
        }
    }
    let n = n.max(1);
    [(r / n) as u8, (g / n) as u8, (b / n) as u8]
}

/// Silhouette buckets, bottom row first.
///
/// A bucket the photograph showed is measured from the subject's own
/// pixels. A bucket the photograph showed nothing in — a black silk
/// dress the plate cannot tell from its background — keeps its width
/// carried between the measured neighbours and samples the plate's own
/// color across that span, so the render paints there what the photo
/// painted there. `covered` records which is which.
fn bucketize(img: &Image, blob: &[usize], w: u32, bbox: [u32; 4]) -> Buckets {
    let [x0, y0, x1, y1] = bbox;
    let rows = (y1 - y0 + 1) as usize;
    let mut count = vec![0u32; rows];
    let mut row_x0 = vec![i32::MAX; rows];
    let mut row_x1 = vec![i32::MIN; rows];
    let (mut sr, mut sg, mut sb) = (vec![0u64; rows], vec![0u64; rows], vec![0u64; rows]);
    for &i in blob {
        let y = i as u32 / w;
        if y < y0 || y > y1 {
            continue;
        }
        let x = i as u32 % w;
        let r = (y - y0) as usize;
        count[r] += 1;
        row_x0[r] = row_x0[r].min(x as i32);
        row_x1[r] = row_x1[r].max(x as i32);
        if let Some(p) = img.get(x, y) {
            sr[r] += p.r as u64;
            sg[r] += p.g as u64;
            sb[r] += p.b as u64;
        }
    }

    let n = PROFILE_ROWS;
    let bw = (x1 - x0 + 1) as f64;
    let mut profile = vec![0.0f64; n];
    let mut covered = vec![false; n];
    let mut xa = vec![-1i32; n];
    let mut xb = vec![-1i32; n];
    let mut colors = vec![[0u8; 3]; n];
    for i in 0..n {
        let start = rows * i / n;
        let end = (rows * (i + 1) / n).max(start + 1).min(rows);
        let (mut lo, mut hi) = (i32::MAX, i32::MIN);
        let (mut extent, mut extent_n) = (0.0f64, 0u32);
        // Bucket rows run bottom-first: offset `off` maps to row
        // `rows - 1 - off`.
        for off in start..end {
            let r = rows - 1 - off;
            if count[r] > 0 {
                // Silhouette width is the row's horizontal extent,
                // not how densely the mask filled it.
                extent += (row_x1[r] - row_x0[r] + 1) as f64;
                extent_n += 1;
                lo = lo.min(row_x0[r]);
                hi = hi.max(row_x1[r]);
            }
        }
        if extent_n > 0 && lo <= hi {
            covered[i] = true;
            profile[i] = (extent / extent_n as f64 / bw).clamp(0.0, 1.0);
            xa[i] = lo;
            xb[i] = hi;
            // Color of the subject's own pixels in this band.
            let (mut cr, mut cg, mut cb, mut c) = (0u64, 0u64, 0u64, 0u64);
            for off in start..end {
                let r = rows - 1 - off;
                if count[r] > 0 {
                    cr += sr[r];
                    cg += sg[r];
                    cb += sb[r];
                    c += count[r] as u64;
                }
            }
            let c = c.max(1);
            colors[i] = [(cr / c) as u8, (cg / c) as u8, (cb / c) as u8];
        }
    }

    // Carry width, span and coverage across what the plate hid —
    // between measured neighbours only, never past the figure's ends.
    if let (Some(first), Some(last)) = (
        covered.iter().position(|&c| c),
        covered.iter().rposition(|&c| c),
    ) {
        for i in first..=last {
            if covered[i] {
                continue;
            }
            let prev = (0..i).rev().find(|&j| covered[j]).unwrap_or(first);
            let next = ((i + 1)..n).find(|&j| covered[j]).unwrap_or(last);
            let t = if next > prev {
                (i - prev) as f64 / (next - prev) as f64
            } else {
                0.0
            };
            let lerp = |a: f64, b: f64| a + (b - a) * t;
            profile[i] = lerp(profile[prev], profile[next]);
            let l = lerp(xa[prev] as f64, xa[next] as f64).round() as i32;
            let r = lerp(xb[prev] as f64, xb[next] as f64).round() as i32;
            xa[i] = l;
            xb[i] = r;
            let start = rows * i / n;
            let end = (rows * (i + 1) / n).max(start + 1).min(rows);
            let y_lo = y0 + (rows as u32 - 1 - (end as u32 - 1));
            let y_hi = y0 + (rows as u32 - 1 - start as u32);
            colors[i] = mean_span(img, y_lo, y_hi, l.max(0) as u32, r.max(0) as u32);
        }
    }

    Buckets {
        profile,
        covered,
        row_colors: colors,
    }
}

/// Read one plate into committed visual facts.
///
/// Two candidate subject masks are tried in order — color distance to
/// the plate's border, then to its dominant color — and the first one
/// yielding a plausible subject (1%..90% of the plate) that sits
/// inside its ground (touching no frame edge) is measured. The mask is
/// then split into regions; the column-clustered group with the most
/// pixels is the subject, and every other region is kept on the
/// record.
///
/// A plate with no separable subject fails with a reason, and a plate
/// whose region fills or runs off the frame fails as a scene rather
/// than being measured as a figure. Nothing is guessed.
pub fn measure(img: &Image, meta: PlateMeta) -> Result<VisualFacts, String> {
    let w = img.width;
    let h = img.height;
    if w < 16 || h < 16 {
        return Err(format!("plate {w}x{h} too small to measure (need 16+)"));
    }
    let surround_px = border_median(img);
    let dominant = img.dominant_color().map(|(c, _)| c).unwrap_or(surround_px);

    let mut chosen: Option<(Vec<usize>, Vec<Component>, String)> = None;
    let mut saw_frame_filling = false;
    for (reference, method) in [(surround_px, "border"), (dominant, "dominant")] {
        let mask = mask_far_from(img, reference);
        let cleaned = Image::morph_open(&mask, w, h, 1);
        let cleaned = Image::morph_close(&cleaned, w, h, 1);
        let comps = components(&cleaned, w, h);
        let members = cluster_columns(&comps);
        let area: usize = members.iter().map(|&i| comps[i].pixels.len()).sum();
        let frac = area as f64 / (w * h) as f64;
        if (0.01..0.9).contains(&frac) {
            // A subject sits inside its ground: its silhouette must be
            // bounded by the surround on all four sides, clear of the
            // frame by more than the morphology radius (a subject that
            // reaches the frame loses its outermost ring to the open,
            // so `> margin` and not `> 0`).
            const FRAME_MARGIN: u32 = 1;
            let (mut fx0, mut fy0, mut fx1, mut fy1) = (u32::MAX, u32::MAX, 0u32, 0u32);
            for &i in &members {
                let [cx0, cy0, cx1, cy1] = comps[i].bbox;
                fx0 = fx0.min(cx0);
                fy0 = fy0.min(cy0);
                fx1 = fx1.max(cx1);
                fy1 = fy1.max(cy1);
            }
            if fx0 > FRAME_MARGIN
                && fy0 > FRAME_MARGIN
                && fx1 + 1 + FRAME_MARGIN < w
                && fy1 + 1 + FRAME_MARGIN < h
            {
                chosen = Some((members, comps, method.to_string()));
                break;
            }
            saw_frame_filling = true;
        }
    }
    let (members, comps, method) = chosen.ok_or_else(|| {
        if saw_frame_filling {
            "subject runs off the frame: this plate is a scene, not a \
             bounded subject — clarify which region is the subject, \
             then re-run"
                .to_string()
        } else {
            "no separable subject: this plate has no region measurably \
             different from its surroundings — clarify which region is \
             the subject, then re-run"
                .to_string()
        }
    })?;

    // The subject as one body of pixels, plus the whole record of
    // what else was foreground.
    let mut blob: Vec<usize> = Vec::new();
    for &i in &members {
        blob.extend_from_slice(&comps[i].pixels);
    }
    let area = blob.len() as u64;
    if area == 0 {
        return Err("empty subject".into());
    }
    let (mut sx, mut sy) = (0u64, 0u64);
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
    for &i in &blob {
        let (x, y) = (i as u32 % w, i as u32 / w);
        sx += x as u64;
        sy += y as u64;
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let (cx, cy) = (sx as f64 / area as f64, sy as f64 / area as f64);
    let bbox = [x0, y0, x1, y1];
    let buckets = bucketize(img, &blob, w, bbox);

    // Brightness balance across the subject: where the light came from.
    let (mut left, mut left_n, mut right, mut right_n) = (0.0, 0u64, 0.0, 0u64);
    let (mut top, mut top_n, mut bottom, mut bottom_n) = (0.0, 0u64, 0.0, 0u64);
    for &i in &blob {
        let (x, y) = (i as u32 % w, i as u32 / w);
        let Some(p) = img.get(x, y) else { continue };
        let b = p.brightness();
        if (x as f64) < cx {
            left += b;
            left_n += 1;
        } else {
            right += b;
            right_n += 1;
        }
        if (y as f64) < cy {
            top += b;
            top_n += 1;
        } else {
            bottom += b;
            bottom_n += 1;
        }
    }
    let (left, right) = (left / left_n.max(1) as f64, right / right_n.max(1) as f64);
    let (top, bottom) = (top / top_n.max(1) as f64, bottom / bottom_n.max(1) as f64);
    let side_balance = if left + right > 0.0 {
        ((right - left) / (left + right)).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let vertical_balance = if top + bottom > 0.0 {
        ((top - bottom) / (top + bottom)).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    let total = (w * h) as f64;
    let mut record: Vec<ComponentFacts> = comps
        .iter()
        .enumerate()
        .map(|(i, c)| ComponentFacts {
            bbox: c.bbox,
            area: c.pixels.len() as u64,
            mean: mean_of(img, &c.pixels),
            in_subject: members.contains(&i),
        })
        .collect();
    record.sort_by_key(|c| std::cmp::Reverse(c.area));
    let Buckets {
        profile,
        covered,
        row_colors,
    } = buckets;
    Ok(VisualFacts {
        plate: meta,
        plate_size: [w, h],
        subject: SubjectFacts {
            method,
            bbox,
            centroid: [cx / w as f64, cy / h as f64],
            height_frac: (y1 - y0 + 1) as f64 / h as f64,
            width_frac: (x1 - x0 + 1) as f64 / w as f64,
            area_frac: area as f64 / total,
            profile,
            side_balance,
            vertical_balance,
            pixels: area,
            components: record,
            profile_covered: covered,
            row_colors,
        },
        palette: Palette {
            subject: mean_of(img, &blob),
            surround: [surround_px.r, surround_px.g, surround_px.b],
            ground: mean_band(img, (h as f64 * 0.85) as u32, h),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> PlateMeta {
        PlateMeta {
            title: "File:Test.jpg".into(),
            query: "a woman standing".into(),
            source_url: "https://example.test/plate.jpg".into(),
            page_url: "https://example.test/File:Test.jpg".into(),
            author: "Tester".into(),
            license: "CC0".into(),
            file: "plate-0000.bmp".into(),
        }
    }

    /// A subject on a plain ground: measurement must find exactly the
    /// rectangle that was drawn, at the size and place it was drawn.
    fn rect_plate() -> (Image, [u32; 4]) {
        let mut img = Image::blank(64, 64, Rgb::new(200, 200, 200));
        // x 20..43, y 10..53 inclusive.
        img.draw_rect(20, 10, 24, 44, Rgb::new(40, 60, 200));
        (img, [20, 10, 43, 53])
    }

    #[test]
    fn measure_finds_the_subject_that_was_drawn() {
        let (img, expect) = rect_plate();
        let facts = measure(&img, meta()).expect("subject is separable");
        assert_eq!(facts.subject.bbox, expect, "bbox must be the drawn rect");
        assert!(facts.subject.height_frac > 0.6 && facts.subject.height_frac < 0.75);
        assert!(facts.subject.width_frac > 0.3 && facts.subject.width_frac < 0.45);
        // A solid rectangle: every silhouette row is full width.
        for (i, &w) in facts.subject.profile.iter().enumerate() {
            assert!(
                w > 0.9,
                "row {i} of a solid rect measured {w}, expected ~1.0"
            );
        }
        // Palette: subject blue, surround grey.
        let [r, g, b] = facts.palette.subject;
        assert!(b > 180 && r < 80, "subject color {r},{g},{b} not the blue");
        let [sr, sg, sb] = facts.palette.surround;
        assert!(
            sr > 180 && sg > 180 && sb > 180,
            "surround {sr},{sg},{sb} not the grey ground"
        );
    }

    #[test]
    fn measure_carries_the_plate_provenance() {
        let (img, _) = rect_plate();
        let facts = measure(&img, meta()).unwrap();
        assert_eq!(facts.plate.title, "File:Test.jpg");
        assert_eq!(facts.plate.license, "CC0");
        assert_eq!(facts.plate.query, "a woman standing");
        assert_eq!(facts.subject.method, "border");
        assert_eq!(facts.subject.pixels, 24 * 44);
    }

    #[test]
    fn measure_is_deterministic() {
        let (img, _) = rect_plate();
        let a = measure(&img, meta()).unwrap();
        let b = measure(&img, meta()).unwrap();
        assert_eq!(a, b, "same plate, same facts — always");
    }

    /// A subject the photograph splits — lit face, dark middle the
    /// plate cannot tell from its own background, lit skirt — is
    /// still one subject: the pieces share columns, the bbox spans
    /// them, and the hidden stretch admits it instead of pretending
    /// the plate measured it.
    #[test]
    fn a_subject_the_plate_split_into_pieces_is_read_as_one() {
        let mut img = Image::blank(64, 96, Rgb::new(200, 200, 200));
        img.draw_rect(20, 10, 24, 30, Rgb::new(40, 60, 200)); // y 10..39
        img.draw_rect(20, 56, 24, 30, Rgb::new(40, 60, 200)); // y 56..85
        let facts = measure(&img, meta()).expect("subject is separable");
        assert_eq!(
            facts.subject.bbox,
            [20, 10, 43, 85],
            "both pieces belong to one figure"
        );
        let covered = &facts.subject.profile_covered;
        assert_eq!(covered.len(), PROFILE_ROWS);
        assert!(covered[0], "the bottom row was photographed");
        assert!(covered[PROFILE_ROWS - 1], "the top row was photographed");
        let hidden = covered.iter().filter(|&&c| !c).count();
        assert!(
            hidden >= 3,
            "the dark middle must read as unmeasured, got {hidden} hidden buckets"
        );
        assert_eq!(facts.subject.row_colors.len(), PROFILE_ROWS);
        assert!(
            facts.subject.row_colors.iter().any(|&c| c != [0, 0, 0]),
            "every bucket carries the plate's own color"
        );
    }

    /// A region beside the figure — the coat-of-arms this plate is
    /// documented to carry — is measured and recorded, but never
    /// claimed as part of the subject.
    #[test]
    fn an_object_beside_the_figure_is_recorded_not_claimed() {
        let mut img = Image::blank(64, 64, Rgb::new(200, 200, 200));
        img.draw_rect(20, 10, 24, 44, Rgb::new(40, 60, 200)); // the figure
        img.draw_rect(2, 20, 9, 24, Rgb::new(200, 40, 40)); // beside it
        let facts = measure(&img, meta()).expect("both regions separable");
        assert_eq!(
            facts.subject.bbox,
            [20, 10, 43, 53],
            "the subject is the figure alone"
        );
        let rec = &facts.subject.components;
        assert!(rec.len() >= 2, "both regions recorded, got {}", rec.len());
        assert!(
            rec.iter().filter(|c| c.in_subject).count() == 1,
            "one column cluster claims the figure"
        );
        assert!(
            rec.iter().any(|c| !c.in_subject),
            "the beside-object stays on the record"
        );
    }

    #[test]
    fn facts_survive_a_json_round_trip() {
        let (img, _) = rect_plate();
        let facts = measure(&img, meta()).unwrap();
        let json = serde_json::to_string_pretty(&facts).expect("serializes");
        let back: VisualFacts = serde_json::from_str(&json).expect("parses");
        assert_eq!(facts, back, "committed facts must survive disk");
    }

    #[test]
    fn a_featureless_plate_fails_honestly() {
        let img = Image::blank(32, 32, Rgb::new(128, 128, 128));
        let err = measure(&img, meta()).expect_err("nothing to measure");
        assert!(
            err.contains("no separable subject"),
            "unmeasurable plate must say so, not guess: {err}"
        );
    }

    #[test]
    fn tiny_plates_are_refused() {
        let img = Image::blank(8, 8, Rgb::new(0, 0, 0));
        let err = measure(&img, meta()).expect_err("too small");
        assert!(err.contains("too small"), "{err}");
    }

    /// A region that runs to the frame edge is the scene, not a
    /// subject: a landscape or crowded room cannot be lifted back
    /// into one figure.
    #[test]
    fn a_frame_filling_region_is_refused_as_a_scene() {
        let mut img = Image::blank(64, 64, Rgb::new(200, 200, 200));
        img.draw_rect(20, 0, 24, 64, Rgb::new(40, 60, 200)); // top to bottom
        let err = measure(&img, meta()).expect_err("a scene, not a subject");
        assert!(
            err.contains("scene"),
            "a frame-filling region must be refused as a scene: {err}"
        );
    }

    /// Diagnostic: what did the subject mask actually catch?
    /// `cargo test --lib dump_mask -- --ignored --nocapture <plate.bmp>`
    #[test]
    #[ignore = "reads an arbitrary plate from argv"]
    fn dump_mask_ascii() {
        let path = std::env::args()
            .rev()
            .find(|a| a.ends_with(".bmp"))
            .unwrap_or_else(|| "/tmp/bank-sem/plate-0000.bmp".into());
        let img = Image::load_bmp(std::path::Path::new(&path)).expect("plate");
        let surround = border_median(&img);
        let dominant = img.dominant_color().map(|(c, _)| c).unwrap_or(surround);
        for (reference, name) in [(surround, "border"), (dominant, "dominant")] {
            let mask = mask_far_from(&img, reference);
            let cleaned = Image::morph_open(&mask, img.width, img.height, 1);
            let cleaned = Image::morph_close(&cleaned, img.width, img.height, 1);
            let blob = Image::largest_blob(&cleaned, img.width, img.height);
            let frac = blob.len() as f64 / (img.width * img.height) as f64;
            println!(
                "method {name}: ref {:?} full-mask {} px ({:.1}%) · largest component {} px ({:.1}%)",
                [reference.r, reference.g, reference.b],
                cleaned.iter().filter(|&&v| v).count(),
                cleaned.iter().filter(|&&v| v).count() as f64 / (img.width * img.height) as f64
                    * 100.0,
                blob.len(),
                frac * 100.0
            );
            // Per-row mask density: does the figure hold together?
            println!("row density (40 bands, '#'=mask):");
            for cy in 0..40 {
                let y0 = cy as u32 * img.height / 40;
                let y1 = (cy as u32 + 1) * img.height / 40;
                let mut n = 0u32;
                let mut tot = 0u32;
                for y in y0..y1 {
                    for x in 0..img.width {
                        tot += 1;
                        if cleaned[(y * img.width + x) as usize] {
                            n += 1;
                        }
                    }
                }
                let f = n as f64 / tot as f64;
                println!(
                    "  y{y0:3}-{y1:3} {:>5.1}% {}",
                    f * 100.0,
                    "#".repeat((f * 40.0) as usize)
                );
            }
            let set: std::collections::HashSet<usize> = cleaned
                .iter()
                .enumerate()
                .filter(|(_, v)| **v)
                .map(|(i, _)| i)
                .collect();
            for cy in 0..40 {
                let mut line = String::new();
                for cx in 0..60 {
                    let x = (cx as f64 * img.width as f64 / 60.0) as u32;
                    let y = (cy as f64 * img.height as f64 / 40.0) as u32;
                    let i = (y * img.width + x) as usize;
                    let px = img.get(x, y).unwrap_or(Rgb::new(0, 0, 0));
                    let lum = px.brightness();
                    let ch = if set.contains(&i) {
                        '#'
                    } else {
                        b" .:-=+*#%@"[(lum * 10.0).clamp(0.0, 9.0) as usize] as char
                    };
                    line.push(ch);
                }
                println!("{line}");
            }
            println!();
        }
    }
}
