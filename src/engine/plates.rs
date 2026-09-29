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
    /// How the license was determined ("Commons file metadata",
    /// "page states cc0"). Auditable evidence, not a bare claim.
    pub basis: String,
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
    // Substring match: metadata concatenates ("Unknown authorUnknown
    // author, scan by X") and exact equality misses the junk.
    if author.is_empty() || author.to_lowercase().contains("unknown author") {
        return Err(format!(
            "{:?}: no author recorded, provenance incomplete",
            title
        ));
    }
    // Commons writes "CC BY-SA 3.0", pages write "cc-by-sa-3.0" —
    // separators are noise, the token is the signal.
    let norm = |s: &str| {
        s.to_lowercase()
            .chars()
            .filter(|c| ![' ', '-', '_'].contains(c))
            .collect::<String>()
    };
    let license_norm = norm(&license);
    let listed = LICENSE_ALLOWLIST
        .iter()
        .any(|allow| norm(allow) == license_norm)
        || SITE_LICENSES
            .iter()
            .any(|(_, name)| norm(name) == license_norm);
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
         &iiprop=url%7Cuser%7Cextmetadata&iiurlwidth=2560",
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
        // Prefer the 1280px thumbnail: separate throttle bucket from
        // full-res originals, and already at sample scale. Fall back
        // to the original when no thumbnail exists.
        let thumb = str_field("thumburl");
        let file_url = if thumb.is_empty() {
            str_field("url")
        } else {
            thumb
        };
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
        // Politeness delay between file fetches: burst traffic earns
        // 429s (measured), and shared infrastructure deserves better.
        // Skipped before the first fetch of the run.
        if !plates.is_empty() || !refused.is_empty() {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
        let bytes = match crate::http::get_bytes(&provenance.source_url).await {
            Ok(b) => b,
            Err(e) => {
                refused.push(format!("{:?}: fetch failed: {}", title, e));
                continue;
            }
        };
        match decode_plate(&bytes) {
            Ok(image) => plates.push(SourcedPlate {
                image,
                provenance,
                basis: "Commons file metadata (LicenseShortName)".to_string(),
            }),
            Err(e) => refused.push(format!("{:?}: {}", title, e)),
        }
    }
    (plates, refused)
}

/// Site licenses recorded verbatim when the hosting page states
/// them. Same standing as Commons short names: the evidence is the
/// page's own words, quoted in the refusal-or-accept log.
const SITE_LICENSES: &[(&str, &str)] = &[
    ("unsplash.com/license", "Unsplash License"),
    ("pexels.com/license", "Pexels License"),
    ("pixabay.com/service/license", "Pixabay License"),
];

/// One image-search hit: direct file URL plus the page that hosts it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebHit {
    pub title: String,
    pub file_url: String,
    pub page_url: String,
}

/// DuckDuckGo image search: fetch the results page, extract the `vqd`
/// token, call the `i.js` JSON endpoint. No key, same stack as the
/// word oracle. Token flow or JSON shape changes refuse as
/// search-failed — never parse garbage into candidates.
pub async fn ddg_images(query: &str, limit: u32) -> Result<Vec<WebHit>, String> {
    let search_url = format!(
        "https://duckduckgo.com/?q={}&iar=images&iax=images&ia=images",
        percent_encode(query)
    );
    let (status, body) = crate::http::get_text(&search_url)
        .await
        .map_err(|e| format!("image search fetch failed: {}", e))?;
    if !(200..300).contains(&status) {
        return Err(format!("image search HTTP {}", status));
    }
    let vqd = extract_vqd(&body)
        .ok_or_else(|| "image search refused: no vqd token in results page".to_string())?;
    let api_url = format!(
        "https://duckduckgo.com/i.js?l=us-en&o=json&q={}&vqd={}&f=,,,,,&p=1",
        percent_encode(query),
        vqd
    );
    let value: serde_json::Value = crate::http::get_json(&api_url, None, Some("application/json"))
        .await
        .map_err(|e| format!("image search api failed: {}", e))?;
    Ok(parse_image_hits(&value, limit))
}

fn extract_vqd(html: &str) -> Option<String> {
    for marker in ["vqd='", "vqd=\"", "vqd="] {
        if let Some(pos) = html.find(marker) {
            let after = &html[pos + marker.len()..];
            let end = after
                .find(['\'', '"', '&', ' ', ';'])
                .unwrap_or(after.len());
            let token = &after[..end];
            if token.len() >= 8 {
                return Some(token.trim_matches('\'').trim_matches('"').to_string());
            }
        }
    }
    None
}

fn parse_image_hits(value: &serde_json::Value, limit: u32) -> Vec<WebHit> {
    let mut out = Vec::new();
    let results = value.get("results").and_then(|r| r.as_array());
    let results = match results {
        Some(r) => r,
        None => return out,
    };
    for r in results {
        if out.len() >= limit as usize {
            break;
        }
        let file_url = r.get("image").and_then(|v| v.as_str()).unwrap_or("");
        let page_url = r.get("url").and_then(|v| v.as_str()).unwrap_or("");
        let title = r
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if (file_url.starts_with("https://") || file_url.starts_with("http://"))
            && (page_url.starts_with("https://") || page_url.starts_with("http://"))
        {
            out.push(WebHit {
                title,
                file_url: file_url.to_string(),
                page_url: page_url.to_string(),
            });
        }
    }
    out
}

/// Page evidence for one hit: author plus license, both read off the
/// hosting page. Author sources, strongest first: JSON-LD `author.name`,
/// `<meta name="author">`, OpenGraph `article:author`. License
/// sources: Creative Commons strings, public-domain statements, then
/// known site-license pages. Anything unfound refuses downstream —
/// this function reports what it saw, never invents.
#[derive(Debug, Clone)]
pub struct PageEvidence {
    pub author: String,
    pub license: String,
    pub basis: String,
}

pub fn page_evidence(page_url: &str, html: &str) -> PageEvidence {
    let lower = html.to_lowercase();
    let author = jsonld_author(html)
        .or_else(|| meta_content(html, "author"))
        .or_else(|| meta_property(html, "article:author"))
        .unwrap_or_default();
    let mut license = String::new();
    let mut basis = String::new();
    for token in [
        "cc0",
        "public domain",
        "cc-by-sa",
        "cc-by",
        "creative commons",
    ] {
        if lower.contains(token) {
            license = token.to_string();
            basis = format!("page states {:?}", token);
            break;
        }
    }
    if license.is_empty() {
        for (marker, name) in SITE_LICENSES {
            if lower.contains(marker) {
                license = name.to_string();
                basis = format!("page references {}", marker);
                break;
            }
        }
    }
    // Creative-Commons umbrella alone is not a grant: require the
    // specific variant. A bare mention without deed terms refuses.
    if license == "creative commons" {
        license.clear();
        basis = "bare Creative Commons mention without variant terms".to_string();
    }
    let _ = page_url;
    PageEvidence {
        author,
        license,
        basis,
    }
}

fn meta_content(html: &str, name: &str) -> Option<String> {
    // <meta name="author" content="...">, attributes in either order.
    let lower = html.to_lowercase();
    let mut rest = lower.as_str();
    let want = format!("name=\"{}\"", name);
    let want2 = format!("name='{}'", name);
    loop {
        let pos = rest.find("<meta")?;
        let tag_end = rest[pos..].find('>')?;
        let tag = &rest[pos..pos + tag_end];
        if tag.contains(&want) || tag.contains(&want2) {
            let orig = &html[html.len() - rest.len() + pos..];
            let orig_end = orig.find('>').unwrap_or(orig.len());
            let orig_tag = &orig[..orig_end];
            return attr_value(orig_tag, "content");
        }
        rest = &rest[pos + 5..];
    }
}

fn meta_property(html: &str, prop: &str) -> Option<String> {
    let lower = html.to_lowercase();
    let mut rest = lower.as_str();
    let want = format!("property=\"{}\"", prop);
    loop {
        let pos = rest.find("<meta")?;
        let tag_end = rest[pos..].find('>')?;
        let tag = &rest[pos..pos + tag_end];
        if tag.contains(&want) {
            let orig = &html[html.len() - rest.len() + pos..];
            let orig_end = orig.find('>').unwrap_or(orig.len());
            return attr_value(&orig[..orig_end], "content");
        }
        rest = &rest[pos + 5..];
    }
}

fn attr_value(tag: &str, attr: &str) -> Option<String> {
    for quote in ['"', '\''] {
        for key in [format!("{}={}", attr, quote)] {
            if let Some(pos) = tag.find(&key) {
                let after = &tag[pos + key.len()..];
                let end = after.find(quote).unwrap_or(after.len());
                let v = after[..end].trim().to_string();
                if !v.is_empty() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn jsonld_author(html: &str) -> Option<String> {
    // "author": {"name": "..."} or "author": "…" inside ld+json blocks.
    let mut rest = html;
    loop {
        let pos = rest.find("\"author\"")?;
        let after = &rest[pos + 8..];
        let after = after.trim_start_matches([' ', ':', '\n', '\r', '\t']);
        if after.starts_with('{') {
            if let Some(npos) = after.find("\"name\"") {
                let vstart = &after[npos + 6..];
                let vstart = vstart.trim_start_matches([' ', ':', '"', '\'', '\n', '\r', '\t']);
                let end = vstart.find(['"', '\'', '}']).unwrap_or(0);
                let name = strip_tags(&vstart[..end]).trim().to_string();
                if !name.is_empty() {
                    return Some(name);
                }
            }
            rest = after.strip_prefix('{').unwrap_or(after);
        } else if after.starts_with('"') || after.starts_with('\'') {
            let q = after.chars().next().unwrap();
            let end = after[1..].find(q).unwrap_or(0);
            let name = strip_tags(&after[1..1 + end]).trim().to_string();
            if !name.is_empty() {
                return Some(name);
            }
            rest = &after[1..];
        } else {
            rest = after;
        }
        if rest.len() < 12 {
            break;
        }
    }
    None
}

/// Source plates from general web image search: DDG hits → hosting
/// page evidence → provenance requirements → fetch → decode. The same
/// completeness rule as Commons applies — most of the web refuses,
/// and the refusals say exactly what was missing.
pub async fn source_plates_web(query: &str, limit: u32) -> (Vec<SourcedPlate>, Vec<String>) {
    let mut plates = Vec::new();
    let mut refused = Vec::new();
    let hits = match ddg_images(query, limit * 3).await {
        Ok(h) => h,
        Err(e) => {
            refused.push(format!("web image search failed: {}", e));
            return (plates, refused);
        }
    };
    for hit in hits {
        if plates.len() >= limit as usize {
            break;
        }
        let label = if hit.title.is_empty() {
            hit.page_url.clone()
        } else {
            hit.title.clone()
        };
        let html = match crate::http::get_text(&hit.page_url).await {
            Ok((st, body)) if (200..300).contains(&st) => body,
            _ => {
                refused.push(format!("{:?}: hosting page unreadable", label));
                continue;
            }
        };
        let ev = page_evidence(&hit.page_url, &html);
        if !ev.basis.is_empty() && ev.license.is_empty() {
            refused.push(format!("{:?}: {}", label, ev.basis));
            continue;
        }
        let provenance = match require_provenance(
            &label,
            &hit.page_url,
            &hit.file_url,
            &ev.author,
            &ev.license,
        ) {
            Ok(p) => p,
            Err(e) => {
                refused.push(format!("{} ({})", e, ev.basis));
                continue;
            }
        };
        // Prefer the page's own og:image when the hit URL is a
        // thumbnail proxy; fall back to the hit URL itself.
        let mut file_urls = vec![provenance.source_url.clone()];
        if let Some(og) = meta_property(&html, "og:image")
            && (og.starts_with("https://") || og.starts_with("http://"))
        {
            file_urls.insert(0, og);
        }
        let mut decoded = None;
        let mut fetch_err = String::new();
        if !plates.is_empty() || !refused.is_empty() {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
        for url in &file_urls {
            match crate::http::get_bytes(url).await {
                Ok(bytes) => match decode_plate(&bytes) {
                    Ok(img) => {
                        decoded = Some(img);
                        break;
                    }
                    Err(e) => fetch_err = e,
                },
                Err(e) => fetch_err = e,
            }
        }
        match decoded {
            Some(image) => plates.push(SourcedPlate {
                image,
                provenance,
                basis: ev.basis.clone(),
            }),
            None => refused.push(format!("{:?}: undecodable: {}", label, fetch_err)),
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

    #[test]
    fn vqd_and_hits_parse_canned_responses() {
        let html =
            r#"<html><head></head><body><script>vqd='4-1234567890abcdef';</script></body></html>"#;
        assert_eq!(extract_vqd(html).as_deref(), Some("4-1234567890abcdef"));
        assert!(extract_vqd("<html>nothing here</html>").is_none());
        let json: serde_json::Value = serde_json::from_str(
            r#"{"results": [
                {"title": "A portrait", "image": "https://cdn.example.org/a.jpg", "url": "https://example.org/page-a"},
                {"title": "Bad scheme", "image": "data:image/gif;base64,xx", "url": "https://example.org/page-b"},
                {"title": "No file", "url": "https://example.org/page-c"}
            ]}"#,
        )
        .unwrap();
        let hits = parse_image_hits(&json, 10);
        assert_eq!(hits.len(), 1, "{:?}", hits);
        assert_eq!(hits[0].file_url, "https://cdn.example.org/a.jpg");
        assert!(parse_image_hits(&serde_json::json!({}), 10).is_empty());
    }

    #[test]
    fn page_evidence_reads_author_and_license() {
        // Meta author plus a CC deed badge.
        let html = r#"<html><head><meta name="author" content="Jane Doe"></head>
            <body><p>Released under <a href="https://creativecommons.org/publicdomain/zero/1.0/">CC0</a> terms.</p></body></html>"#;
        let ev = page_evidence("https://example.org/p", html);
        assert_eq!(ev.author, "Jane Doe");
        assert_eq!(ev.license, "cc0");
        // JSON-LD author plus a site license reference.
        let html2 = r#"<html><head><script type="application/ld+json">{"@type":"ImageObject","author":{"name":"John Smith"}}</script></head>
            <body>See https://unsplash.com/license for terms.</body></html>"#;
        let ev2 = page_evidence("https://unsplash.com/photos/x", html2);
        assert_eq!(ev2.author, "John Smith");
        assert_eq!(ev2.license, "Unsplash License");
        // Bare "creative commons" without a variant is not a grant.
        let ev3 = page_evidence(
            "https://example.org/q",
            "<html><body>creative commons stuff</body></html>",
        );
        assert!(ev3.license.is_empty(), "{:?}", ev3);
        assert!(!ev3.basis.is_empty());
        // Nothing present: empty evidence, downstream refuses.
        let ev4 = page_evidence(
            "https://example.org/r",
            "<html><body>buy this photo $99</body></html>",
        );
        assert!(ev4.author.is_empty() && ev4.license.is_empty());
        // And the provenance gate accepts the site-license token.
        assert!(
            require_provenance(
                "T",
                "https://unsplash.com/photos/x",
                "https://images.unsplash.com/y.jpg",
                "John Smith",
                "Unsplash License"
            )
            .is_ok()
        );
    }
}
