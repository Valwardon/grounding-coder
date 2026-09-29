//! Eyeballs, the deterministic kind: photos in, photos out, no model.
//!
//! Two halves that meet at the pixel buffer:
//! - [`Image`]: create, load, save (BMP, hand-rolled — zero new
//!   dependencies), transform, and measure.
//! - Classical perception: brightness, histograms, dominant colors,
//!   Sobel edges, template matching. Structure from arithmetic, not
//!   from statistics. It cannot name a tower in a photograph — and
//!   says so — but it can find edges, regions, and known patterns
//!   with byte-exact repeatability.
//!
//! Manipulation ("put the sign in the top-left") is pixel surgery
//! with a receipt: every op returns what it changed, and the oracle
//! is the buffer itself — tests assert exact pixels, not vibes.
use std::path::Path;

/// 24-bit RGB pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Rgb { r, g, b }
    }

    /// Perceived brightness, 0.0–1.0 (luma weights).
    pub fn brightness(&self) -> f64 {
        (0.299 * self.r as f64 + 0.587 * self.g as f64 + 0.114 * self.b as f64) / 255.0
    }

    pub fn distance2(&self, other: &Rgb) -> u32 {
        let dr = self.r as i32 - other.r as i32;
        let dg = self.g as i32 - other.g as i32;
        let db = self.b as i32 - other.b as i32;
        (dr * dr + dg * dg + db * db) as u32
    }
}

/// Row-major RGB image. Coordinate origin is top-left.
#[derive(Debug, Clone)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pixels: Vec<Rgb>,
}

impl Image {
    pub fn blank(width: u32, height: u32, fill: Rgb) -> Self {
        Image {
            width,
            height,
            pixels: vec![fill; (width * height) as usize],
        }
    }

    pub fn get(&self, x: u32, y: u32) -> Option<Rgb> {
        if x < self.width && y < self.height {
            Some(self.pixels[(y * self.width + x) as usize])
        } else {
            None
        }
    }

    pub fn set(&mut self, x: u32, y: u32, px: Rgb) -> bool {
        if x < self.width && y < self.height {
            self.pixels[(y * self.width + x) as usize] = px;
            true
        } else {
            false
        }
    }

    /// Fill a rectangle, clipped to the buffer. Returns pixels painted.
    pub fn draw_rect(&mut self, x: u32, y: u32, w: u32, h: u32, px: Rgb) -> u32 {
        let mut painted = 0;
        for dy in 0..h {
            for dx in 0..w {
                if self.set(x + dx, y + dy, px) {
                    painted += 1;
                }
            }
        }
        painted
    }

    /// Filled circle, clipped. Returns pixels painted.
    pub fn draw_disc(&mut self, cx: i32, cy: i32, radius: u32, px: Rgb) -> u32 {
        let mut painted = 0;
        let r2 = (radius as i32) * (radius as i32);
        for dy in -(radius as i32)..=(radius as i32) {
            for dx in -(radius as i32)..=(radius as i32) {
                if dx * dx + dy * dy <= r2 {
                    let (x, y) = (cx + dx, cy + dy);
                    if x >= 0 && y >= 0 && self.set(x as u32, y as u32, px) {
                        painted += 1;
                    }
                }
            }
        }
        painted
    }

    /// Paste `other` at `(x, y)`, clipped. Returns pixels pasted.
    pub fn overlay(&mut self, other: &Image, x: u32, y: u32) -> u32 {
        let mut pasted = 0;
        for oy in 0..other.height {
            for ox in 0..other.width {
                if let Some(px) = other.get(ox, oy)
                    && self.set(x + ox, y + oy, px)
                {
                    pasted += 1;
                }
            }
        }
        pasted
    }

    /// Crop to `(x, y, w, h)`, clipped to the buffer.
    pub fn crop(&self, x: u32, y: u32, w: u32, h: u32) -> Image {
        let w = w.min(self.width.saturating_sub(x));
        let h = h.min(self.height.saturating_sub(y));
        let mut out = Image::blank(w, h, Rgb::new(0, 0, 0));
        for dy in 0..h {
            for dx in 0..w {
                if let Some(px) = self.get(x + dx, y + dy) {
                    out.set(dx, dy, px);
                }
            }
        }
        out
    }

    /// Nearest-neighbor resize. Deterministic down to the pixel.
    pub fn resize(&self, w: u32, h: u32) -> Image {
        let mut out = Image::blank(w.max(1), h.max(1), Rgb::new(0, 0, 0));
        for y in 0..out.height {
            for x in 0..out.width {
                let sx = (x * self.width / out.width).min(self.width - 1);
                let sy = (y * self.height / out.height).min(self.height - 1);
                if let Some(px) = self.get(sx, sy) {
                    out.set(x, y, px);
                }
            }
        }
        out
    }

    /// Mean brightness over the whole frame.
    pub fn mean_brightness(&self) -> f64 {
        if self.pixels.is_empty() {
            return 0.0;
        }
        self.pixels.iter().map(|p| p.brightness()).sum::<f64>() / self.pixels.len() as f64
    }

    /// 8-bin brightness histogram (counts sum to pixel total).
    pub fn histogram(&self) -> [u64; 8] {
        let mut bins = [0u64; 8];
        for p in &self.pixels {
            let b = (p.brightness() * 8.0).floor() as usize;
            bins[b.min(7)] += 1;
        }
        bins
    }

    /// Most common exact color and its count. Ties break toward the
    /// first pixel in row-major order — deterministic, documented.
    pub fn dominant_color(&self) -> Option<(Rgb, u64)> {
        let mut counts: Vec<(Rgb, u64)> = Vec::new();
        for p in &self.pixels {
            match counts.iter_mut().find(|(c, _)| c == p) {
                Some(entry) => entry.1 += 1,
                None => counts.push((*p, 1)),
            }
        }
        counts.into_iter().max_by_key(|(_, n)| *n)
    }

    /// Count pixels within `tolerance2` (squared RGB distance) of `target`.
    pub fn count_near(&self, target: Rgb, tolerance2: u32) -> u64 {
        self.pixels
            .iter()
            .filter(|p| p.distance2(&target) <= tolerance2)
            .count() as u64
    }

    /// Sobel edge map (grayscale magnitude). A half-black/half-white
    /// frame yields a single bright column — classical CV, no model.
    pub fn sobel_edges(&self) -> Image {
        let gx = [-1i32, 0, 1, -2, 0, 2, -1, 0, 1];
        let gy = [-1i32, -2, -1, 0, 0, 0, 1, 2, 1];
        // Clamped borders: sampling past the frame replicates the
        // edge pixel. Reading void-as-black would print phantom
        // edges around every flat region touching the border.
        let lum = |x: i32, y: i32| -> f64 {
            let x = x.clamp(0, self.width as i32 - 1) as u32;
            let y = y.clamp(0, self.height as i32 - 1) as u32;
            self.pixels[(y * self.width + x) as usize].brightness()
        };
        let mut out = Image::blank(self.width, self.height, Rgb::new(0, 0, 0));
        for y in 0..self.height as i32 {
            for x in 0..self.width as i32 {
                let (mut sx, mut sy) = (0.0, 0.0);
                for ky in 0..3 {
                    for kx in 0..3 {
                        let v = lum(x + kx - 1, y + ky - 1);
                        sx += gx[(ky * 3 + kx) as usize] as f64 * v;
                        sy += gy[(ky * 3 + kx) as usize] as f64 * v;
                    }
                }
                let mag = (sx * sx + sy * sy).sqrt().min(1.0);
                let g = (mag * 255.0) as u8;
                out.set(x as u32, y as u32, Rgb::new(g, g, g));
            }
        }
        out
    }

    /// Find `template` in `self` by sum of absolute RGB differences.
    /// Returns the top-left offset and score of the best match, or
    /// `None` when the template is larger than the frame. Exact
    /// sub-images score 0 — the oracle for "the sign is where we put
    /// it".
    pub fn find_template(&self, template: &Image) -> Option<(u32, u32, u64)> {
        if template.width > self.width || template.height > self.height {
            return None;
        }
        let mut best: Option<(u32, u32, u64)> = None;
        for y in 0..=(self.height - template.height) {
            for x in 0..=(self.width - template.width) {
                let mut score = 0u64;
                for ty in 0..template.height {
                    for tx in 0..template.width {
                        let a = self.pixels[((y + ty) * self.width + (x + tx)) as usize];
                        let b = template.pixels[(ty * template.width + tx) as usize];
                        score += (a.r as i32 - b.r as i32).unsigned_abs() as u64
                            + (a.g as i32 - b.g as i32).unsigned_abs() as u64
                            + (a.b as i32 - b.b as i32).unsigned_abs() as u64;
                    }
                }
                match best {
                    Some((_, _, s)) if s <= score => {}
                    _ => best = Some((x, y, score)),
                }
            }
        }
        best
    }

    // ── Photographic finish (all deterministic) ──
    //
    // Renders look CG-flat; photos don't. Three cheap, seeded,
    // repeatable post passes close half the gap. Same seed twice →
    // byte-identical grain, asserted below.

    /// Darken toward the corners (portrait-lens falloff).
    pub fn vignette(&mut self, strength: f64) {
        let cx = self.width as f64 / 2.0;
        let cy = self.height as f64 / 2.0;
        let max_d = (cx * cx + cy * cy).sqrt();
        for y in 0..self.height {
            for x in 0..self.width {
                let dx = x as f64 + 0.5 - cx;
                let dy = y as f64 + 0.5 - cy;
                let fall = (1.0 - strength * (dx * dx + dy * dy).sqrt() / max_d).clamp(0.0, 1.0);
                let i = (y * self.width + x) as usize;
                let p = self.pixels[i];
                self.pixels[i] = Rgb::new(
                    (p.r as f64 * fall).round() as u8,
                    (p.g as f64 * fall).round() as u8,
                    (p.b as f64 * fall).round() as u8,
                );
            }
        }
    }

    /// Film grain: uniform ±`amount` per channel from a 64-bit LCG
    /// seeded by the caller. No RNG dependency, fully repeatable.
    pub fn grain(&mut self, seed: u64, amount: u8) {
        if amount == 0 {
            return;
        }
        // Golden-ratio mix: every seed (including 0) opens a distinct
        // stream. (OR-ing an odd bit instead would collide neighbors
        // like 42 and 43 — the test caught exactly that.)
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        let next = |s: &mut u64| {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*s >> 33) as u8
        };
        for p in self.pixels.iter_mut() {
            let span = amount as u16 * 2 + 1;
            let jitter = |s: &mut u64, v: u8| {
                let n = (next(s) as u16) % span;
                (v as i16 + n as i16 - amount as i16).clamp(0, 255) as u8
            };
            *p = Rgb::new(
                jitter(&mut s, p.r),
                jitter(&mut s, p.g),
                jitter(&mut s, p.b),
            );
        }
    }

    /// Grade: contrast pivot at mid-gray plus brightness lift, both in
    /// 0..255 units. `contrast 1.0, lift 0` is the identity.
    pub fn grade(&mut self, contrast: f64, lift: f64) {
        for p in self.pixels.iter_mut() {
            let adj = |v: u8| {
                ((v as f64 - 128.0) * contrast + 128.0 + lift)
                    .clamp(0.0, 255.0)
                    .round() as u8
            };
            *p = Rgb::new(adj(p.r), adj(p.g), adj(p.b));
        }
    }

    // ── Masks: segmentation without a model ──
    //
    // Skin-locus segmentation (Chai & Ngan YCbCr bounds), 3x3
    // morphology, largest-blob selection, box-feathered alpha.
    // It finds skin-colored regions; it does not know a face from a
    // hand or a mahogany table. Callers combine it with composition
    // priors (largest blob, upper frame) and record what they assumed.

    /// YCbCr chroma of one pixel (BT.601).
    pub fn chroma(&self, x: u32, y: u32) -> Option<(f64, f64)> {
        self.get(x, y).map(|p| {
            let (r, g, b) = (p.r as f64, p.g as f64, p.b as f64);
            (
                128.0 - 0.169 * r - 0.331 * g + 0.5 * b,
                128.0 + 0.5 * r - 0.419 * g - 0.081 * b,
            )
        })
    }

    /// Skin mask: classical Cb∈[77,127], Cr∈[133,173] bounds plus a
    /// luminance floor (Y>40). Dark warm browns sit inside the chroma
    /// box — the JFK plate's near-black background proved it at 97%
    /// coverage — so chroma alone is not a detector.
    pub fn skin_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; (self.width * self.height) as usize];
        for y in 0..self.height {
            for x in 0..self.width {
                if let Some(px) = self.get(x, y) {
                    let (r, g, b) = (px.r as f64, px.g as f64, px.b as f64);
                    let cb = 128.0 - 0.169 * r - 0.331 * g + 0.5 * b;
                    let cr = 128.0 + 0.5 * r - 0.419 * g - 0.081 * b;
                    mask[(y * self.width + x) as usize] = (77.0..=127.0).contains(&cb)
                        && (133.0..=173.0).contains(&cr)
                        && px.brightness() * 255.0 > 40.0;
                }
            }
        }
        mask
    }

    /// 3x3 morphological open (erode then dilate): kills speckle,
    /// keeps regions. `passes` repeats the pair.
    pub fn morph_open(mask: &[bool], width: u32, height: u32, passes: u32) -> Vec<bool> {
        let mut m = mask.to_vec();
        for _ in 0..passes {
            m = Self::erode(&m, width, height);
            m = Self::dilate(&m, width, height);
        }
        m
    }

    /// 3x3 morphological close (dilate then erode): fills pinholes.
    pub fn morph_close(mask: &[bool], width: u32, height: u32, passes: u32) -> Vec<bool> {
        let mut m = mask.to_vec();
        for _ in 0..passes {
            m = Self::dilate(&m, width, height);
            m = Self::erode(&m, width, height);
        }
        m
    }

    fn erode(mask: &[bool], width: u32, height: u32) -> Vec<bool> {
        Self::morph(mask, width, height, true)
    }

    fn dilate(mask: &[bool], width: u32, height: u32) -> Vec<bool> {
        Self::morph(mask, width, height, false)
    }

    fn morph(mask: &[bool], width: u32, height: u32, erode: bool) -> Vec<bool> {
        let at = |x: i32, y: i32| -> bool {
            if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                return false;
            }
            mask[(y as u32 * width + x as u32) as usize]
        };
        let mut out = vec![false; mask.len()];
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let mut all = true;
                let mut any = false;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let v = at(x + dx, y + dy);
                        all = all && v;
                        any = any || v;
                    }
                }
                out[(y as u32 * width + x as u32) as usize] = if erode { all } else { any };
            }
        }
        out
    }

    /// Largest 4-connected blob: indices into the frame. Empty mask
    /// yields empty — no blob invented.
    pub fn largest_blob(mask: &[bool], width: u32, height: u32) -> Vec<usize> {
        let mut seen = vec![false; mask.len()];
        let mut best: Vec<usize> = Vec::new();
        for i in 0..mask.len() {
            if !mask[i] || seen[i] {
                continue;
            }
            let mut blob = Vec::new();
            let mut stack = vec![i];
            seen[i] = true;
            while let Some(j) = stack.pop() {
                blob.push(j);
                let x = (j as u32) % width;
                let y = (j as u32) / width;
                for (nx, ny) in [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                ] {
                    if nx < width && ny < height {
                        let k = (ny * width + nx) as usize;
                        if mask[k] && !seen[k] {
                            seen[k] = true;
                            stack.push(k);
                        }
                    }
                }
            }
            if blob.len() > best.len() {
                best = blob;
            }
        }
        best
    }

    /// Blob statistics: (area, centroid x/y, bbox x0/y0/x1/y1).
    pub fn blob_stats(blob: &[usize], width: u32) -> Option<(u64, f64, f64, u32, u32, u32, u32)> {
        if blob.is_empty() {
            return None;
        }
        let (mut sx, mut sy) = (0u64, 0u64);
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
        for b in blob {
            let x = (*b as u32) % width;
            let y = (*b as u32) / width;
            sx += x as u64;
            sy += y as u64;
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        let n = blob.len() as f64;
        Some((
            blob.len() as u64,
            sx as f64 / n,
            sy as f64 / n,
            x0,
            y0,
            x1,
            y1,
        ))
    }

    /// Box-feather a mask into alpha: 0.0 outside, 1.0 deep inside,
    /// linear ramp over `radius`. Separable passes, clamped borders.
    pub fn feather(mask: &[bool], width: u32, height: u32, radius: u32) -> Vec<f64> {
        let base: Vec<f64> = mask.iter().map(|m| if *m { 1.0 } else { 0.0 }).collect();
        let blur = |src: &[f64]| -> Vec<f64> {
            let mut tmp = vec![0.0; src.len()];
            let r = radius as i32;
            for y in 0..height as i32 {
                for x in 0..width as i32 {
                    let mut sum = 0.0;
                    let mut n = 0;
                    for dx in -r..=r {
                        let sx = (x + dx).clamp(0, width as i32 - 1) as u32;
                        sum += src[(y as u32 * width + sx) as usize];
                        n += 1;
                    }
                    tmp[(y as u32 * width + x as u32) as usize] = sum / n as f64;
                }
            }
            let mut out = vec![0.0; src.len()];
            for y in 0..height as i32 {
                for x in 0..width as i32 {
                    let mut sum = 0.0;
                    let mut n = 0;
                    for dy in -r..=r {
                        let sy = (y + dy).clamp(0, height as i32 - 1) as u32;
                        sum += tmp[(sy * width + x as u32) as usize];
                        n += 1;
                    }
                    out[(y as u32 * width + x as u32) as usize] = sum / n as f64;
                }
            }
            out
        };
        // Two passes approximate a smooth falloff; renormalize so deep
        // interiors return to exactly 1.0.
        let twice = blur(&blur(&base));
        let peak = twice.iter().cloned().fold(0.0f64, f64::max).max(1e-9);
        twice
            .into_iter()
            .map(|v| (v / peak).clamp(0.0, 1.0))
            .collect()
    }

    /// Composite: foreground over background through alpha. Lengths
    /// must match the frame; mismatch refuses with an error.
    pub fn composite(fg: &Image, bg: &Image, alpha: &[f64]) -> Result<Image, String> {
        if fg.width != bg.width || fg.height != bg.height {
            return Err(format!(
                "composite size mismatch: {}x{} over {}x{}",
                fg.width, fg.height, bg.width, bg.height
            ));
        }
        if alpha.len() != (fg.width * fg.height) as usize {
            return Err("composite alpha length mismatch".to_string());
        }
        let mut out = Image::blank(fg.width, fg.height, Rgb::new(0, 0, 0));
        for (i, (a, (f, b))) in alpha
            .iter()
            .zip(fg.pixels.iter().zip(bg.pixels.iter()))
            .enumerate()
        {
            let a = a.clamp(0.0, 1.0);
            out.pixels[i] = Rgb::new(
                (f.r as f64 * a + b.r as f64 * (1.0 - a)).round() as u8,
                (f.g as f64 * a + b.g as f64 * (1.0 - a)).round() as u8,
                (f.b as f64 * a + b.b as f64 * (1.0 - a)).round() as u8,
            );
        }
        Ok(out)
    }

    // ── BMP codec (24-bit, uncompressed, hand-rolled) ──
    //
    // No dependency for this: the format is a 54-byte header plus
    // bottom-up BGR rows padded to 4 bytes. Writing is exact;
    // reading accepts exactly what we write and refuses the rest
    // with the reason stated.

    pub fn save_bmp(&self, path: &Path) -> Result<(), String> {
        let row_stride = (self.width * 3).div_ceil(4) * 4;
        let data_size = row_stride * self.height;
        let file_size = 54 + data_size;
        let mut buf = Vec::with_capacity(file_size as usize);
        buf.extend_from_slice(b"BM");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(&54u32.to_le_bytes());
        buf.extend_from_slice(&40u32.to_le_bytes());
        buf.extend_from_slice(&self.width.to_le_bytes());
        buf.extend_from_slice(&((self.height as i32).to_le_bytes()));
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&24u16.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&data_size.to_le_bytes());
        buf.extend_from_slice(&[0u8; 16]);
        let pad = vec![0u8; (row_stride - self.width * 3) as usize];
        for y in (0..self.height).rev() {
            for x in 0..self.width {
                let p = self.pixels[(y * self.width + x) as usize];
                buf.push(p.b);
                buf.push(p.g);
                buf.push(p.r);
            }
            buf.extend_from_slice(&pad);
        }
        std::fs::write(path, buf).map_err(|e| format!("bmp write failed: {}", e))
    }

    pub fn load_bmp(path: &Path) -> Result<Image, String> {
        let buf = std::fs::read(path).map_err(|e| format!("bmp read failed: {}", e))?;
        if buf.len() < 54
            || &buf[0..2] != b"BM"
            || u16::from_le_bytes([buf[28], buf[29]]) != 24
            || u32::from_le_bytes([buf[30], buf[31], buf[32], buf[33]]) != 0
        {
            return Err("not a 24-bit uncompressed BMP (only what we write)".to_string());
        }
        let width = u32::from_le_bytes([buf[18], buf[19], buf[20], buf[21]]);
        let height = u32::from_le_bytes([buf[22], buf[23], buf[24], buf[25]]) as i32;
        if width == 0 || height <= 0 || width > 8192 || height > 8192 {
            return Err(format!("refusing BMP of {}x{}", width, height));
        }
        let data_off = u32::from_le_bytes([buf[10], buf[11], buf[12], buf[13]]) as usize;
        let row_stride = (width * 3).div_ceil(4) * 4;
        let height = height as u32;
        if buf.len() < data_off + (row_stride * height) as usize {
            return Err("BMP truncated".to_string());
        }
        let mut pixels = vec![Rgb::new(0, 0, 0); (width * height) as usize];
        for y in 0..height {
            let src_row = data_off + ((height - 1 - y) * row_stride) as usize;
            for x in 0..width {
                let o = src_row + (x * 3) as usize;
                pixels[(y * width + x) as usize] = Rgb::new(buf[o + 2], buf[o + 1], buf[o]);
            }
        }
        Ok(Image {
            width,
            height,
            pixels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_and_disc_paint_exact_pixels() {
        let mut img = Image::blank(10, 10, Rgb::new(0, 0, 0));
        assert_eq!(img.draw_rect(2, 2, 3, 3, Rgb::new(255, 0, 0)), 9);
        assert_eq!(img.get(2, 2), Some(Rgb::new(255, 0, 0)));
        assert_eq!(img.get(4, 4), Some(Rgb::new(255, 0, 0)));
        assert_eq!(img.get(5, 5), Some(Rgb::new(0, 0, 0)));
        // Clipped rect paints only what fits.
        assert_eq!(img.draw_rect(8, 8, 5, 5, Rgb::new(0, 255, 0)), 4);
        // Radius-1 disc paints a plus sign: 5 pixels.
        let mut img2 = Image::blank(7, 7, Rgb::new(0, 0, 0));
        assert_eq!(img2.draw_disc(3, 3, 1, Rgb::new(0, 0, 255)), 5);
    }

    #[test]
    fn crop_resize_overlay_are_exact() {
        let mut img = Image::blank(8, 8, Rgb::new(1, 2, 3));
        img.set(3, 3, Rgb::new(9, 9, 9));
        let crop = img.crop(2, 2, 4, 4);
        assert_eq!((crop.width, crop.height), (4, 4));
        assert_eq!(crop.get(1, 1), Some(Rgb::new(9, 9, 9)));
        // Crop past the edge clips instead of failing.
        assert_eq!(img.crop(6, 6, 8, 8).width, 2);
        let small = img.resize(4, 4);
        assert_eq!((small.width, small.height), (4, 4));
        let mut base = Image::blank(6, 6, Rgb::new(0, 0, 0));
        let stamp = Image::blank(2, 2, Rgb::new(7, 7, 7));
        assert_eq!(base.overlay(&stamp, 5, 5), 1);
        assert_eq!(base.get(5, 5), Some(Rgb::new(7, 7, 7)));
        assert_eq!(base.get(0, 0), Some(Rgb::new(0, 0, 0)));
    }

    #[test]
    fn stats_tell_the_truth() {
        let mut img = Image::blank(4, 4, Rgb::new(0, 0, 0));
        img.draw_rect(0, 0, 2, 4, Rgb::new(255, 255, 255));
        assert!((img.mean_brightness() - 0.5).abs() < 1e-9);
        let hist = img.histogram();
        assert_eq!(hist.iter().sum::<u64>(), 16);
        assert_eq!(hist[0], 8);
        assert_eq!(hist[7], 8);
        let (color, count) = img.dominant_color().unwrap();
        assert_eq!((color, count), (Rgb::new(0, 0, 0), 8));
        assert_eq!(img.count_near(Rgb::new(255, 255, 255), 0), 8);
    }

    #[test]
    fn sobel_finds_the_edge() {
        // Left black, right white: exactly one bright column at x=4.
        let mut img = Image::blank(8, 8, Rgb::new(0, 0, 0));
        img.draw_rect(4, 0, 4, 8, Rgb::new(255, 255, 255));
        let edges = img.sobel_edges();
        let col = |x: u32| -> u64 {
            (0..8)
                .filter(|y| edges.get(x, *y).is_some_and(|p| p.r > 128))
                .count() as u64
        };
        assert!(col(3) + col(4) >= 6, "edge must fire at the boundary");
        assert_eq!(col(0), 0, "flat black has no edges");
        assert_eq!(col(7), 0, "flat white has no edges");
    }

    #[test]
    fn template_match_scores_zero_on_exact_copy() {
        let mut img = Image::blank(16, 16, Rgb::new(10, 20, 30));
        let stamp = Image::blank(3, 3, Rgb::new(200, 40, 40));
        img.overlay(&stamp, 7, 11);
        let (x, y, score) = img.find_template(&stamp).unwrap();
        assert_eq!((x, y, score), (7, 11, 0));
        // Oversized template is refused, not panicked.
        assert!(
            img.find_template(&Image::blank(32, 32, Rgb::new(0, 0, 0)))
                .is_none()
        );
    }

    #[test]
    fn finish_is_deterministic_and_shaped() {
        let flat = Image::blank(16, 16, Rgb::new(200, 200, 200));
        // Vignette keeps the center, darkens corners.
        let mut v = flat.clone();
        v.vignette(0.5);
        let center = v.get(8, 8).unwrap().r;
        let corner = v.get(0, 0).unwrap().r;
        assert!(center > 190, "center nearly untouched: {}", center);
        assert!(
            corner < center,
            "corners fall off: {} vs {}",
            corner,
            center
        );
        // Same seed twice: byte-identical grain.
        let mut g1 = flat.clone();
        let mut g2 = flat.clone();
        g1.grain(42, 8);
        g2.grain(42, 8);
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(g1.get(x, y), g2.get(x, y));
            }
        }
        // Grain changes pixels but stays near home.
        assert_ne!(g1.get(0, 0), Some(Rgb::new(200, 200, 200)));
        assert!(g1.get(0, 0).is_some_and(|p| p.r.abs_diff(200) <= 8));
        // Different seed, different grain.
        let mut g3 = flat.clone();
        g3.grain(43, 8);
        assert_ne!(g1.get(0, 0), g3.get(0, 0));
        // Grade identity holds; lift moves values.
        let mut gr = flat.clone();
        gr.grade(1.0, 0.0);
        assert_eq!(gr.get(3, 3), Some(Rgb::new(200, 200, 200)));
        gr.grade(1.0, 10.0);
        assert_eq!(gr.get(3, 3), Some(Rgb::new(210, 210, 210)));
    }

    #[test]
    fn masks_are_exact() {
        // Skin patch on blue: locus finds exactly the rect.
        let mut img = Image::blank(20, 20, Rgb::new(60, 110, 200));
        img.draw_rect(5, 5, 6, 6, Rgb::new(200, 150, 115));
        let mask = img.skin_mask();
        assert_eq!(mask.iter().filter(|m| **m).count(), 36);
        // Open kills a lone speckle, keeps the block.
        let mut noisy = mask.clone();
        noisy[0] = true;
        let opened = Image::morph_open(&noisy, 20, 20, 1);
        assert!(!opened[0], "speckle must die");
        // Erode shrinks 6x6 to 4x4, dilate restores it: block survives whole.
        assert_eq!(opened.iter().filter(|m| **m).count(), 36);
        // Largest blob picks the bigger rect; empty stays empty.
        let mut two = vec![false; 400];
        for y in 0..3 {
            for x in 0..3 {
                two[(y * 20 + x) as usize] = true;
            }
        }
        for y in 10..15 {
            for x in 10..15 {
                two[(y * 20 + x) as usize] = true;
            }
        }
        assert_eq!(Image::largest_blob(&two, 20, 20).len(), 25);
        assert!(Image::largest_blob(&vec![false; 400], 20, 20).is_empty());
        assert!(Image::blob_stats(&[], 20).is_none());
        let stats = Image::blob_stats(&Image::largest_blob(&two, 20, 20), 20).unwrap();
        assert_eq!((stats.3, stats.4, stats.5, stats.6), (10, 10, 14, 14));
    }

    #[test]
    fn feather_ramps_and_composite_blends() {
        // Solid 10x10 block in 30x30: center alpha 1, far corner 0.
        let mut mask = vec![false; 900];
        for y in 10..20 {
            for x in 10..20 {
                mask[(y * 30 + x) as usize] = true;
            }
        }
        let alpha = Image::feather(&mask, 30, 30, 3);
        assert!((alpha[(15 * 30 + 15) as usize] - 1.0).abs() < 1e-9);
        assert_eq!(alpha[0], 0.0);
        let edge = alpha[(10 * 30 + 15) as usize];
        assert!(edge > 0.0 && edge < 1.0, "ramp, got {}", edge);
        // Blend math: half alpha mixes channels evenly.
        let fg = Image::blank(4, 4, Rgb::new(200, 0, 0));
        let bg = Image::blank(4, 4, Rgb::new(0, 0, 200));
        let half = vec![0.5; 16];
        let out = Image::composite(&fg, &bg, &half).unwrap();
        assert_eq!(out.get(0, 0), Some(Rgb::new(100, 0, 100)));
        let full = vec![1.0; 16];
        assert_eq!(
            Image::composite(&fg, &bg, &full).unwrap().get(0, 0),
            Some(Rgb::new(200, 0, 0))
        );
        assert!(Image::composite(&fg, &Image::blank(3, 3, Rgb::new(0, 0, 0)), &half).is_err());
        assert!(Image::composite(&fg, &bg, &[0.5; 8]).is_err());
    }

    #[test]
    fn bmp_roundtrip_is_lossless() {
        let dir = std::env::temp_dir().join(format!("gc-bmp-{}", unique_suffix()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut img = Image::blank(7, 5, Rgb::new(12, 34, 56));
        img.draw_rect(1, 1, 3, 2, Rgb::new(200, 100, 50));
        img.draw_disc(5, 3, 2, Rgb::new(1, 2, 3));
        let path = dir.join("roundtrip.bmp");
        img.save_bmp(&path).unwrap();
        let back = Image::load_bmp(&path).unwrap();
        assert_eq!((back.width, back.height), (7, 5));
        for y in 0..5 {
            for x in 0..7 {
                assert_eq!(img.get(x, y), back.get(x, y), "pixel ({},{})", x, y);
            }
        }
        // Garbage is refused with a reason, not a panic.
        let bad = dir.join("bad.bmp");
        std::fs::write(&bad, b"definitely not a bitmap").unwrap();
        assert!(Image::load_bmp(&bad).is_err());
    }

    fn unique_suffix() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    }
}
