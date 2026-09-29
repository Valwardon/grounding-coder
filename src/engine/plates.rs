//! Photographic plates from the open web, with complete provenance.
//!
//! The curiosity loop can source its own base photos: search
//! Wikimedia Commons, read each file's machine-readable metadata,
//! and ingest only what carries complete provenance — source URL,
//! author, and a redistribution license from the allowlist below.
//! Anything with missing or unlisted provenance is refused with the
//! reason stated, exactly like an EditPlan without evidence.
//!
//! Provenance rides with every plate and lands in the knowledge
//! store beside the pixels, so any future use can answer: where did
//! this come from, who made it, under what terms.
use super::vision::{Image, Rgb};

/// Licenses the loop accepts, matched against Commons'
/// `LicenseShortName` (lowercased, trimmed). Public domain and
/// attribution licenses only.
const LICENSE_ALLOWLIST: &[&str] = &[
    "public domain",
    "cc0",
    "cc-by",
    "cc-by-sa",
    "cc-by-sa-4.0",
    "cc-by-4.0",
    "cc-by-3.0",
    "cc-by-sa-3.0",
    "cc-by-2.0",
    "cc-by-sa-2.0",
];

/// Complete provenance for one ingested plate. No field optional:
/// a plate that cannot answer all three is not ingested.
#[derive(Debug, Clone)]
pub struct PlateProvenance {
    /// Direct file URL the bytes came from.
    pub source_url: String,
    /// File description page (human-readable provenance).
    pub page_url: String,
    /// Author string from the file metadata.
    pub author: String,
    /// License short name as listed by Commons.
    pub license: String,
}

#[derive(Debug, Clone)]
pub struct SourcedPlate {
    pub image: Image,
    pub provenance: PlateProvenance,
}

/// Check one Commons `imageinfo` record. Returns provenance on
/// success, or the exact missing/unlisted field on refusal.
pub fn require_provenance(
    title: &str,
    page_url: &str,
    file_url: &str,
    author: &str,
    license: &str,
) -> Result<PlateProvenance, String> {
    // Strip HTML tags Commons embeds in author/artist strings.
    let author = strip_tags(author).trim().to_string();
    let license = license.trim().to_string();
    if file_url.trim().is_empty() {
        return Err(format!("{:?}: no file URL, cannot fetch bytes", title));
    }
    if author.is_empty() || author == "Unknown author" {
        return Err(format!(
            "{:?}: no author recorded, provenance incomplete",
            title
        ));
    }
    let listed = LICENSE_ALLOWLIST
        .iter()
        .any(|allow| license.to_lowercase() == *allow);
    if !listed {
        return Err(format!(
            "{:?}: license {:?} not in the provenance allowlist",
            title, license
        ));
    }
    Ok(PlateProvenance {
        source_url: file_url.to_string(),
        page_url: page_url.to_string(),
        author,
        license,
    })
}

fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut inside = false;
    for c in s.chars() {
        match c {
            '<' => inside = true,
            '>' => inside = false,
            _ if !inside => out.push(c),
            _ => {}
        }
    }
    out
}

/// Decode fetched bytes (JPEG/PNG via the `image` crate) into the
/// engine buffer. Alpha is flattened onto black; everything else
/// about the pixels is preserved exactly.
pub fn decode_plate(bytes: &[u8]) -> Result<Image, String> {
    let dyn_img =
        image::load_from_memory(bytes).map_err(|e| format!("plate decode failed: {}", e))?;
    let rgb = dyn_img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    if w == 0 || h == 0 || w > 8192 || h > 8192 {
        return Err(format!("refusing plate of {}x{}", w, h));
    }
    let mut img = Image::blank(w, h, Rgb::new(0, 0, 0));
    for (x, y, p) in rgb.enumerate_pixels() {
        img.set(x, y, Rgb::new(p[0], p[1], p[2]));
    }
    Ok(img)
}

/// Commons API search for files, newest first. Returns raw
/// `(title, page_url, file_url, author, license)` tuples; the caller
/// passes each through [`require_provenance`].
pub async fn search_commons(
    query: &str,
    limit: u32,
) -> Result<Vec<(String, String, String, String, String)>, String> {
    let url = format!(
        "https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search\
         &gsrsearch=filetype:bitmap%20{}&gsrnamespace=6&gsrlimit={}&prop=imageinfo\
         &iiprop=url%7Cuser%7Cextmetadata&iiurlwidth=1280",
        percent_encode(query),
        limit.min(20)
    );
    let value: serde_json::Value = crate::http::get_json(&url, None, None).await?;
    let mut out = Vec::new();
    let pages = value
        .get("query")
        .and_then(|q| q.get("pages"))
        .and_then(|p| p.as_object());
    let pages = match pages {
        Some(p) => p,
        None => return Ok(out),
    };
    for (_, page) in pages {
        let title = page
            .get("title")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        let info = page
            .get("imageinfo")
            .and_then(|i| i.as_array())
            .and_then(|a| a.first());
        let info = match info {
            Some(i) => i,
            None => continue,
        };
        let str_field = |key: &str| {
            info.get(key)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        let meta = |key: &str| {
            info.get("extmetadata")
                .and_then(|m| m.get(key))
                .and_then(|m| m.get("value"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        // Prefer the full-resolution URL; fall back to the thumbnail.
        let file_url = str_field("url");
        let page_url = format!(
            "https://commons.wikimedia.org/wiki/{}",
            title.replace(' ', "_")
        );
        out.push((
            title,
            page_url,
            file_url,
            meta("Artist"),
            meta("LicenseShortName"),
        ));
    }
    Ok(out)
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
            out.push(b as char);
        } else if b == b' ' {
            out.push_str("%20");
        } else {
            out.push_str(&format!("%{:02X}", b));
        }
    }
    out
}

/// Source up to `limit` plates for a query: search, require
/// provenance, fetch bytes, decode. Each step can refuse with its
/// reason; refusals are reported, never silent, and never fatal to
/// the plates that did verify.
pub async fn source_plates(query: &str, limit: u32) -> (Vec<SourcedPlate>, Vec<String>) {
    let mut plates = Vec::new();
    let mut refused = Vec::new();
    let candidates = match search_commons(query, limit).await {
        Ok(c) => c,
        Err(e) => {
            refused.push(format!("search failed: {}", e));
            return (plates, refused);
        }
    };
    for (title, page_url, file_url, author, license) in candidates {
        if plates.len() >= limit as usize {
            break;
        }
        let provenance = match require_provenance(&title, &page_url, &file_url, &author, &license) {
            Ok(p) => p,
            Err(e) => {
                refused.push(e);
                continue;
            }
        };
        let bytes = match crate::http::get_bytes(&provenance.source_url).await {
            Ok(b) => b,
            Err(e) => {
                refused.push(format!("{:?}: fetch failed: {}", title, e));
                continue;
            }
        };
        match decode_plate(&bytes) {
            Ok(image) => plates.push(SourcedPlate { image, provenance }),
            Err(e) => refused.push(format!("{:?}: {}", title, e)),
        }
    }
    (plates, refused)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_requires_all_three_fields() {
        // Complete record passes.
        let p = require_provenance(
            "File:Example.jpg",
            "https://commons.wikimedia.org/wiki/File:Example.jpg",
            "https://upload.wikimedia.org/example.jpg",
            "Jane Doe",
            "CC-BY-SA-4.0",
        )
        .expect("complete record");
        assert_eq!(p.author, "Jane Doe");
        // Missing author refuses.
        assert!(require_provenance("F", "P", "http://x/y.jpg", "", "CC0").is_err());
        assert!(require_provenance("F", "P", "http://x/y.jpg", "Unknown author", "CC0").is_err());
        // Unlisted license refuses — press-agency terms fail here.
        assert!(
            require_provenance(
                "File:Star.jpg",
                "P",
                "http://x/y.jpg",
                "Photo Agency",
                "Copyrighted, all rights reserved"
            )
            .is_err()
        );
        // HTML-wrapped authors are stripped, not rejected.
        let p = require_provenance(
            "F",
            "P",
            "http://x/y.jpg",
            "<a href=\"x\">Jane</a>",
            "Public domain",
        )
        .expect("stripped author");
        assert_eq!(p.author, "Jane");
        // Case-insensitive license match.
        assert!(require_provenance("F", "P", "http://x/y.jpg", "A. Uthor", "cc-by-4.0").is_ok());
    }

    #[test]
    fn decode_roundtrip_preserves_pixels() {
        // Build pixels with the engine, encode via image crate, decode back.
        let mut img = Image::blank(8, 6, Rgb::new(12, 34, 56));
        img.draw_rect(1, 1, 3, 2, Rgb::new(200, 100, 50));
        let mut buf = Vec::new();
        {
            let mut rgb = image::RgbImage::new(8, 6);
            for y in 0..6 {
                for x in 0..8 {
                    let p = img.get(x, y).unwrap();
                    rgb.put_pixel(x, y, image::Rgb([p.r, p.g, p.b]));
                }
            }
            let mut cursor = std::io::Cursor::new(&mut buf);
            image::write_buffer_with_format(
                &mut cursor,
                &rgb,
                8,
                6,
                image::ColorType::Rgb8,
                image::ImageFormat::Png,
            )
            .expect("encode");
        }
        let back = decode_plate(&buf).expect("decode");
        assert_eq!((back.width, back.height), (8, 6));
        for y in 0..6 {
            for x in 0..8 {
                assert_eq!(img.get(x, y), back.get(x, y), "pixel ({},{})", x, y);
            }
        }
        assert!(decode_plate(b"not an image").is_err());
    }
}
