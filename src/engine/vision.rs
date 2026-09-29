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
