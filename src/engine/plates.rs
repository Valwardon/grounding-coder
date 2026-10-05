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

/// Collection engine: many plates per request, politely and only once.
///
/// Per-request training needs dozens of plates, but the old loop
/// fetched them one at a time with a 2s sleep between every two —
/// 100 plates meant 200s+ of waiting. This module fetches with
/// bounded concurrency while keeping every politeness promise:
/// - at most [`FETCH_CONCURRENCY`] fetches in flight at once;
/// - at least [`POLITENESS_SECS`] between two fetches from the same
///   host (reservations are atomic, so concurrent workers queue
///   instead of stampeding);
/// - results come back in CANDIDATE order, never completion order,
///   so selection stays deterministic run to run;
/// - fetched bytes land in a shared disk cache (`.grounding/`),
///   so re-runs and retries resume instead of re-downloading.
pub const FETCH_CONCURRENCY: usize = 4;
pub const POLITENESS_SECS: u64 = 2;
/// One indexed fetch outcome: candidate index plus bytes or reason.
pub type BytesFetch = (usize, Result<Vec<u8>, String>);
/// One indexed page outcome: candidate index plus status/body or reason.
pub type PageFetch = (usize, Result<(u16, String), String>);

/// Host part of a URL (`https://a.b/c` → `a.b`). Best-effort parse
/// for politeness bucketing — unknown shapes bucket together.
pub fn host_of(url: &str) -> String {
    let after_scheme = url.split("://").nth(1).unwrap_or(url);
    after_scheme
        .split('/')
        .next()
        .unwrap_or(after_scheme)
        .to_lowercase()
}

/// Cache key for a fetched URL: hex SHA-256, so cache files carry
/// no information about what was fetched.
pub fn cache_key(url: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(url.as_bytes());
    format!("{:x}", h.finalize())
}

/// Shared byte cache dir (runtime state, gitignored like the rest
/// of `.grounding/`).
fn byte_cache_dir() -> std::path::PathBuf {
    std::path::Path::new(".grounding/plate-bytes").to_path_buf()
}

/// Read-through byte cache: cached bytes win, otherwise fetch,
/// store, and return. Corrupt-or-missing cache reads fall through
/// to a fresh fetch — the cache never fails a fetch, it only
/// avoids repeats.
pub async fn fetch_bytes_cached(url: &str) -> Result<Vec<u8>, String> {
    let dir = byte_cache_dir();
    let path = dir.join(cache_key(url));
    if let Ok(bytes) = std::fs::read(&path)
        && !bytes.is_empty()
    {
        return Ok(bytes);
    }
    let bytes = crate::http::get_bytes(url).await?;
    if bytes.is_empty() {
        return Err("empty response — refusing to cache nothing".to_string());
    }
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(&path, &bytes);
    Ok(bytes)
}

/// Per-host politeness reservations. Cloneable across workers;
/// the reservation table is the shared choke point.
#[derive(Debug, Clone)]
pub struct Politeness {
    last: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, std::time::Instant>>>,
    min_interval: std::time::Duration,
}

impl Politeness {
    pub fn new() -> Self {
        Politeness {
            last: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            min_interval: std::time::Duration::from_secs(POLITENESS_SECS),
        }
    }

    /// Wait until this URL's host is due, then reserve the slot.
    /// The reservation is atomic under one lock acquisition and the
    /// lock is never held across the sleep.
    pub async fn wait(&self, url: &str) {
        let host = host_of(url);
        let now = std::time::Instant::now();
        let delay = {
            let mut map = self.last.lock().unwrap_or_else(|e| e.into_inner());
            let wait_until = map
                .get(&host)
                .map(|t| *t + self.min_interval)
                .unwrap_or(now);
            let delay = wait_until.saturating_duration_since(now);
            map.insert(host, if delay.is_zero() { now } else { wait_until });
            delay
        };
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
    }
}

impl Default for Politeness {
    fn default() -> Self {
        Self::new()
    }
}

/// Fetch many page texts with bounded concurrency and shared
/// politeness, in candidate order. Hosting pages are NOT disk-cached:
/// license evidence must be read fresh every run, never trusted from
/// a stale copy.
pub async fn fetch_many_text(urls: &[String]) -> Vec<PageFetch> {
    use futures_util::stream::{self, StreamExt};
    let polite = Politeness::new();
    let mut out: Vec<PageFetch> = stream::iter(urls.iter().enumerate())
        .map(|(i, url)| {
            let polite = polite.clone();
            let url = url.clone();
            async move {
                polite.wait(&url).await;
                let r = crate::http::get_text(&url)
                    .await
                    .map_err(|e| format!("page fetch failed: {}", e));
                (i, r)
            }
        })
        .buffer_unordered(FETCH_CONCURRENCY.max(1))
        .collect()
        .await;
    out.sort_by_key(|(i, _)| *i);
    out
}

/// Fetch many URLs with bounded concurrency, shared politeness, and
/// the read-through cache. Returns `(candidate index, bytes)` pairs
/// sorted by index — completion order never leaks into selection.
pub async fn fetch_many(urls: &[String]) -> Vec<BytesFetch> {
    use futures_util::stream::{self, StreamExt};
    let polite = Politeness::new();
    let mut out: Vec<BytesFetch> = stream::iter(urls.iter().enumerate())
        .map(|(i, url)| {
            let polite = polite.clone();
            let url = url.clone();
            async move {
                polite.wait(&url).await;
                (i, fetch_bytes_cached(&url).await)
            }
        })
        .buffer_unordered(FETCH_CONCURRENCY.max(1))
        .collect()
        .await;
    out.sort_by_key(|(i, _)| *i);
    out
}

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
    /// Source title (Commons file title / page title). Travels into
    /// evidence records so every banked plate stays auditable.
    pub title: String,
}

/// Age/safety review: title words that refuse a candidate BEFORE
/// fetch. No minors, no nudity, no sexual content — in ANY source
/// path, without exception. Keyword matching is crude by design:
/// over-refusal is the safe direction, and every refusal names its
/// reason.
const UNSAFE_TITLE_WORDS: &[&str] = &[
    "naked",
    "nude",
    "nudity",
    "boy",
    "girl",
    "child",
    "children",
    "kid",
    "kids",
    "teen",
    "teenager",
    "baby",
    "infant",
    "toddler",
    "minor",
    "schoolboy",
    "schoolgirl",
    "erotic",
    "porn",
    "sexual",
    "sexy",
    "fetish",
    "bdsm",
];

/// Refuse unsafe titles with the reason. Pure function of the title.
pub fn review_title(title: &str) -> Result<(), String> {
    let lower = title.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    if let Some(hit) = UNSAFE_TITLE_WORDS.iter().find(|w| words.contains(w)) {
        return Err(format!(
            "{:?}: refused by age/safety review ({})",
            title, hit
        ));
    }
    Ok(())
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
    // Any CC-BY / CC-BY-SA version qualifies (1.0 through 4.0 and
    // beyond — the grant is the family, not the version number).
    // NonCommercial and NoDerivatives variants never qualify.
    let cc_free = license_norm.starts_with("ccby")
        && !license_norm.contains("nc")
        && !license_norm.contains("nd");
    let listed = cc_free
        || LICENSE_ALLOWLIST
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
        limit.min(50)
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
/// the plates that did verify. File fetches run through the shared
/// collection engine (bounded concurrency, per-host politeness,
/// disk cache) in candidate order.
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
    // Pass 1 (cheap, sequential): title review + provenance. Only
    // verified candidates spend network.
    let mut vetted: Vec<(String, PlateProvenance, String)> = Vec::new();
    for (title, page_url, file_url, author, license) in candidates {
        if vetted.len() >= limit as usize {
            break;
        }
        if let Err(e) = review_title(&title) {
            refused.push(e);
            continue;
        }
        match require_provenance(&title, &page_url, &file_url, &author, &license) {
            Ok(p) => vetted.push((
                title,
                p,
                "Commons file metadata (LicenseShortName)".to_string(),
            )),
            Err(e) => refused.push(e),
        }
    }
    // Pass 2 (bulk): concurrent polite cached fetches, order kept.
    let urls: Vec<String> = vetted
        .iter()
        .map(|(_, p, _)| p.source_url.clone())
        .collect();
    let mut fetched = fetch_many(&urls).await;
    // Pass 3 (in order): decode wins and losses both reported.
    for ((title, provenance, basis), (_, bytes)) in vetted.into_iter().zip(fetched.drain(..)) {
        if plates.len() >= limit as usize {
            break;
        }
        match bytes {
            Err(e) => refused.push(format!("{:?}: fetch failed: {}", title, e)),
            Ok(b) => match decode_plate(&b) {
                Ok(image) => plates.push(SourcedPlate {
                    image,
                    provenance,
                    basis,
                    title: title.clone(),
                }),
                Err(e) => refused.push(format!("{:?}: {}", title, e)),
            },
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
    // Pass 1 (cheap, sequential): title review only. Pages and
    // files are bulk-fetched below.
    let mut vetted_hits: Vec<(String, WebHit)> = Vec::new();
    for hit in hits {
        if plates.len() >= limit as usize {
            break;
        }
        let label = if hit.title.is_empty() {
            hit.page_url.clone()
        } else {
            hit.title.clone()
        };
        if let Err(e) = review_title(&label) {
            refused.push(e);
            continue;
        }
        vetted_hits.push((label, hit));
    }
    // Pass 2 (bulk): hosting pages concurrently (fresh every run —
    // license evidence is never cached), order kept.
    let page_urls: Vec<String> = vetted_hits
        .iter()
        .map(|(_, h)| h.page_url.clone())
        .collect();
    let pages = fetch_many_text(&page_urls).await;
    // Pass 3 (cheap, sequential): page evidence + provenance. No
    // cap here — fetch/decode failures below backfill from later
    // hits, exactly like the old sequential loop.
    let mut vetted: Vec<(String, PlateProvenance, String, Vec<String>)> = Vec::new();
    for ((label, hit), (_, page)) in vetted_hits.into_iter().zip(pages) {
        let html = match page {
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
        vetted.push((label, provenance, ev.basis.clone(), file_urls));
    }
    // Pass 4 (bulk): first-choice files concurrently, order kept.
    let firsts: Vec<String> = vetted
        .iter()
        .map(|(_, _, _, urls)| urls[0].clone())
        .collect();
    let first_bytes = fetch_many(&firsts).await;
    // Pass 5 (bulk): second choices only for first-choice failures.
    let mut need_second = Vec::new();
    let mut second_for: Vec<usize> = Vec::new();
    for (k, (_, r)) in first_bytes.iter().enumerate() {
        if r.is_err() && vetted[k].3.len() > 1 {
            second_for.push(k);
            need_second.push(vetted[k].3[1].clone());
        }
    }
    let second_bytes = fetch_many(&need_second).await;
    let mut second_by_k: std::collections::HashMap<usize, Result<Vec<u8>, String>> =
        std::collections::HashMap::new();
    for (k, (_, r)) in second_for.into_iter().zip(second_bytes) {
        second_by_k.insert(k, r);
    }
    // Pass 6 (in order): decode wins and losses both reported.
    for (k, (label, provenance, basis, _)) in vetted.into_iter().enumerate() {
        if plates.len() >= limit as usize {
            break;
        }
        let mut decoded = None;
        let mut fetch_err = String::new();
        let attempts: Vec<Result<Vec<u8>, String>> = vec![
            first_bytes.get(k).map(|(_, r)| match r {
                Ok(b) => Ok(b.clone()),
                Err(e) => Err(e.clone()),
            }),
            second_by_k.remove(&k),
        ]
        .into_iter()
        .flatten()
        .collect();
        for bytes in attempts {
            match bytes {
                Ok(b) => match decode_plate(&b) {
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
                basis,
                title: label.clone(),
            }),
            None => refused.push(format!("{:?}: undecodable: {}", label, fetch_err)),
        }
    }
    (plates, refused)
}

/// Open Images (Google) as a people source: millions of CC-licensed
/// photographs with machine-readable person boxes. No segmentation
/// needed — the dataset states where people stand. Adult rule: any
/// image carrying a Girl or Boy box is excluded, no exceptions; the
/// remaining Woman/Person boxes are composable strangers, never
/// named individuals (the metadata carries no identities by design).
pub mod openimages {
    use super::{SourcedPlate, decode_plate, require_provenance};

    const BASE: &str = "https://storage.googleapis.com/openimages/2018_04/validation";
    const IMAGES_FILE: &str = "validation-images-with-rotation.csv";
    const BBOX_FILE: &str = "validation-annotations-bbox.csv";

    /// Boxable label IDs (V4–V7 stable).
    pub const PERSON: &str = "/m/01g317";
    pub const WOMAN: &str = "/m/03bt1vf";
    pub const GIRL: &str = "/m/05r655";
    pub const BOY: &str = "/m/01bl7v";

    /// One annotated person: the plate plus its box as fractions.
    #[derive(Debug, Clone)]
    pub struct OpenPlate {
        pub plate: SourcedPlate,
        pub bbox: (f64, f64, f64, f64),
        pub label: String,
    }

    /// Query words to target/excluded label sets. "man" has no
    /// boxable ID: Person-labeled images with no Woman/Girl/Boy
    /// labels anywhere. A stated heuristic, not a guarantee.
    pub fn labels_for(query: &str) -> (Vec<&'static str>, Vec<&'static str>) {
        let q = query.to_lowercase();
        if q.contains("woman") || q.contains("female") || q.contains("lady") {
            (vec![WOMAN], vec![GIRL, BOY])
        } else if q.contains("man") || q.contains("male") || q.contains("gentleman") {
            (vec![PERSON], vec![WOMAN, GIRL, BOY])
        } else {
            (vec![PERSON, WOMAN], vec![GIRL, BOY])
        }
    }

    /// Fetch a metadata file into the cache dir (once; reuse after).
    async fn cached_file(
        cache_dir: &std::path::Path,
        name: &str,
    ) -> Result<std::path::PathBuf, String> {
        let path = cache_dir.join(name);
        if path.exists() {
            return Ok(path);
        }
        std::fs::create_dir_all(cache_dir)
            .map_err(|e| format!("plate cache unavailable: {}", e))?;
        let url = format!("{}/{}", BASE, name);
        let bytes = crate::http::get_bytes(&url)
            .await
            .map_err(|e| format!("metadata fetch failed: {}", e))?;
        std::fs::write(&path, bytes).map_err(|e| format!("metadata cache write failed: {}", e))?;
        Ok(path)
    }

    #[derive(Debug, Clone)]
    pub struct ImageMeta {
        pub url: String,
        pub page: String,
        pub license: String,
        pub author: String,
        pub title: String,
        pub rotation: f64,
    }

    /// Parse one images-TSV row. Front fields (ID..Profile) and back
    /// fields (Size, MD5, Thumb, Rotation) never contain commas;
    /// Author and Title do, unquoted. The Author display name is
    /// recovered whole by joining the middle; Title is dropped (the
    /// landing page carries it) rather than guessed.
    pub fn parse_image_row(line: &str) -> Option<(String, ImageMeta)> {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 12 {
            return None;
        }
        let n = parts.len();
        let rotation: f64 = parts[n - 1].trim().parse().unwrap_or(0.0);
        let id = parts[0].trim().to_string();
        let url = parts[2].trim().to_string();
        let page = parts[3].trim().to_string();
        let license = parts[4].trim().to_string();
        let author = parts[6..n - 4].join(",").trim().to_string();
        if id.is_empty() || id == "ImageID" || url.is_empty() {
            return None;
        }
        Some((
            id,
            ImageMeta {
                url,
                page,
                license,
                author,
                title: String::new(),
                rotation,
            },
        ))
    }

    #[derive(Debug, Clone)]
    pub struct BoxRow {
        pub image: String,
        pub label: String,
        pub x0: f64,
        pub y0: f64,
        pub x1: f64,
        pub y1: f64,
        pub clean: bool,
    }

    /// Parse one bbox-TSV row: ImageID,Source,LabelName,Confidence,
    /// XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,
    /// IsDepiction,IsInside. Clean = no occlusion/truncation/group/
    /// depiction flags.
    pub fn parse_box_row(line: &str) -> Option<BoxRow> {
        let p: Vec<&str> = line.split(',').collect();
        if p.len() < 13 || p[0] == "ImageID" {
            return None;
        }
        let num = |i: usize| p.get(i).and_then(|v| v.trim().parse::<f64>().ok());
        let flag = |i: usize| p.get(i).map(|v| v.trim() == "1").unwrap_or(false);
        Some(BoxRow {
            image: p[0].trim().to_string(),
            label: p[2].trim().to_string(),
            x0: num(4)?,
            y0: num(6)?,
            x1: num(5)?,
            y1: num(7)?,
            clean: !(flag(8) || flag(9) || flag(10) || flag(11)),
        })
    }

    /// Creative-Commons license URLs to allowlist tokens. Anything
    /// else (NC/ND variants, unknown) refuses downstream.
    pub fn license_token(url: &str) -> Option<String> {
        let u = url.to_lowercase();
        if u.contains("/licenses/by-sa/") {
            Some("CC-BY-SA".to_string())
        } else if u.contains("/licenses/by/") {
            Some("CC-BY".to_string())
        } else if u.contains("publicdomain/zero") || u.contains("/publicdomain/") {
            Some("CC0".to_string())
        } else {
            None
        }
    }

    /// Search cached metadata: target boxes minus excluded images,
    /// joined to image rows, largest clean box per image first.
    /// Streams both files; early-exits at `limit` plates.
    pub async fn search_openimages(
        cache_dir: &std::path::Path,
        query: &str,
        limit: u32,
    ) -> (Vec<OpenPlate>, Vec<String>) {
        let mut refused = Vec::new();
        let images_path = match cached_file(cache_dir, IMAGES_FILE).await {
            Ok(p) => p,
            Err(e) => {
                refused.push(e);
                return (Vec::new(), refused);
            }
        };
        let bbox_path = match cached_file(cache_dir, BBOX_FILE).await {
            Ok(p) => p,
            Err(e) => {
                refused.push(e);
                return (Vec::new(), refused);
            }
        };
        let (targets, excluded) = labels_for(query);
        // Pass 1 (boxes): target rows per image + excluded image set.
        let mut hits: std::collections::HashMap<String, Vec<BoxRow>> =
            std::collections::HashMap::new();
        let mut banned: std::collections::HashSet<String> = std::collections::HashSet::new();
        let bbox_text = match std::fs::read_to_string(&bbox_path) {
            Ok(t) => t,
            Err(e) => {
                refused.push(format!("bbox cache unreadable: {}", e));
                return (Vec::new(), refused);
            }
        };
        for line in bbox_text.lines() {
            let Some(row) = parse_box_row(line) else {
                continue;
            };
            if excluded.contains(&row.label.as_str()) {
                banned.insert(row.image.clone());
            }
            if targets.contains(&row.label.as_str()) {
                hits.entry(row.image.clone()).or_default().push(row);
            }
        }
        if hits.is_empty() {
            refused.push(format!("no {:?} boxes in metadata", query));
            return (Vec::new(), refused);
        }
        // Pass 2 (images): join metadata for surviving images.
        let images_text = match std::fs::read_to_string(&images_path) {
            Ok(t) => t,
            Err(e) => {
                refused.push(format!("image cache unreadable: {}", e));
                return (Vec::new(), refused);
            }
        };
        let mut metas: std::collections::HashMap<String, ImageMeta> =
            std::collections::HashMap::new();
        for line in images_text.lines() {
            if let Some((id, meta)) = parse_image_row(line)
                && hits.contains_key(&id)
            {
                metas.insert(id, meta);
            }
        }
        // Assemble: biggest clean box wins per image; anything else
        // takes the biggest box available. Banned images drop out.
        // Rounds preserve the old sequential selection order exactly
        // (sorted IDs, first `limit` plates): each round vets the next
        // chunk, bulk-fetches it concurrently, and decodes in order —
        // fetch failures backfill from later IDs like before.
        let mut plates = Vec::new();
        let mut order: Vec<String> = hits.keys().cloned().collect();
        order.sort();
        let mut cursor = 0usize;
        let want = limit as usize;
        while plates.len() < want && cursor < order.len() {
            let mut chunk: Vec<(String, BoxRow, ImageMeta, super::PlateProvenance)> = Vec::new();
            while chunk.len() + plates.len() < want && cursor < order.len() {
                let id = &order[cursor];
                cursor += 1;
                if banned.contains(id) {
                    continue;
                }
                let Some(meta) = metas.get(id) else {
                    refused.push(format!("{}: no image row", id));
                    continue;
                };
                if meta.rotation.abs() > 0.01 {
                    refused.push(format!("{}: rotated, boxes would misalign", id));
                    continue;
                }
                let rows = &hits[id];
                let pick = rows
                    .iter()
                    .filter(|r| r.clean)
                    .max_by(|a, b| {
                        let area = |r: &BoxRow| (r.x1 - r.x0) * (r.y1 - r.y0);
                        area(a)
                            .partial_cmp(&area(b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .or_else(|| {
                        rows.iter().max_by(|a, b| {
                            let area = |r: &BoxRow| (r.x1 - r.x0) * (r.y1 - r.y0);
                            area(a)
                                .partial_cmp(&area(b))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                    });
                let Some(row) = pick else { continue };
                let Some(token) = license_token(&meta.license) else {
                    refused.push(format!(
                        "{}: license {:?} not allowlisted",
                        id, meta.license
                    ));
                    continue;
                };
                match require_provenance(id, &meta.page, &meta.url, &meta.author, &token) {
                    Ok(p) => chunk.push((id.clone(), (*row).clone(), (*meta).clone(), p)),
                    Err(e) => refused.push(e),
                }
            }
            if chunk.is_empty() {
                continue;
            }
            let urls: Vec<String> = chunk
                .iter()
                .map(|(_, _, _, p)| p.source_url.clone())
                .collect();
            let fetched = super::fetch_many(&urls).await;
            for ((id, row, meta, provenance), (_, bytes)) in chunk.into_iter().zip(fetched) {
                if plates.len() >= want {
                    break;
                }
                match bytes {
                    Err(e) => refused.push(format!("{}: fetch failed: {}", id, e)),
                    Ok(b) => match decode_plate(&b) {
                        Ok(image) => plates.push(OpenPlate {
                            plate: SourcedPlate {
                                image,
                                provenance,
                                basis: format!(
                                    "Open Images V4 box {} by {}",
                                    row.label, meta.author
                                ),
                                title: id.clone(),
                            },
                            bbox: (row.x0, row.y0, row.x1, row.y1),
                            label: row.label.clone(),
                        }),
                        Err(e) => refused.push(format!("{}: {}", id, e)),
                    },
                }
            }
        }
        (plates, refused)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn image_rows_parse_around_commas() {
            // Author carries a comma; the middle-join recovers it whole.
            let line = "abc123,validation,https://farm1/x.jpg,https://flickr.com/p/1,https://creativecommons.org/licenses/by/2.0/,https://flickr.com/people/u,Doe, Jane,4405052,QUJD,https://thumb/z.jpg,0.0";
            let (id, meta) = parse_image_row(line).expect("parses");
            assert_eq!(id, "abc123");
            assert_eq!(meta.author, "Doe, Jane");
            assert_eq!(meta.license, "https://creativecommons.org/licenses/by/2.0/");
            assert_eq!(license_token(&meta.license).as_deref(), Some("CC-BY"));
            assert!(parse_image_row("ImageID,Subset,URL").is_none());
            // Real header shape (12 clean fields) parses.
            let clean = "abc123,validation,https://farm1/x.jpg,https://flickr.com/p/1,https://creativecommons.org/licenses/by/2.0/,https://flickr.com/people/u,Jane,Beach portrait,4405052,QUJD,https://thumb/z.jpg,0.0";
            let (_, meta) = parse_image_row(clean).expect("parses");
            // Middle joined whole: author and title inseparable raw.
            assert_eq!(meta.author, "Jane,Beach portrait");
        }

        #[test]
        fn box_rows_parse_flags() {
            let clean = "img1,freeform,/m/03bt1vf,1,0.1,0.5,0.2,0.8,0,0,0,0,0";
            let row = parse_box_row(clean).expect("parses");
            assert_eq!((row.label.as_str(), row.clean), ("/m/03bt1vf", true));
            assert!((row.x0, row.y0, row.x1, row.y1) == (0.1, 0.2, 0.5, 0.8));
            let dirty = "img1,freeform,/m/01g317,1,0.1,0.5,0.2,0.8,1,0,0,0,0";
            assert!(!parse_box_row(dirty).expect("parses").clean);
            assert!(parse_box_row("ImageID,Source,LabelName").is_none());
        }

        #[test]
        fn queries_map_to_adult_labels() {
            assert_eq!(labels_for("woman"), (vec![WOMAN], vec![GIRL, BOY]));
            let (t, e) = labels_for("man");
            assert_eq!(t, vec![PERSON]);
            assert!(e.contains(&WOMAN) && e.contains(&GIRL) && e.contains(&BOY));
            assert_eq!(
                labels_for("portrait"),
                (vec![PERSON, WOMAN], vec![GIRL, BOY])
            );
        }

        #[test]
        fn licenses_gate_cleanly() {
            assert_eq!(
                license_token("https://creativecommons.org/licenses/by-sa/2.0/").as_deref(),
                Some("CC-BY-SA")
            );
            assert!(license_token("https://creativecommons.org/licenses/by-nc/2.0/").is_none());
            assert!(license_token("").is_none());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_buckets_hosts() {
        assert_eq!(
            host_of("https://upload.wikimedia.org/x.jpg"),
            "upload.wikimedia.org"
        );
        assert_eq!(host_of("http://a.b/c?d=e"), "a.b");
        assert_eq!(host_of("https://EXAMPLE.org/"), "example.org");
        assert_eq!(host_of("notaurl"), "notaurl");
    }

    #[test]
    fn engine_cache_keys_are_opaque_and_stable() {
        let a = cache_key("https://example.org/a.jpg");
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(a, cache_key("https://example.org/a.jpg"));
        assert_ne!(a, cache_key("https://example.org/b.jpg"));
    }

    #[tokio::test]
    async fn engine_empty_fetch_is_empty() {
        // No URLs, no network: the bulk path must be a no-op.
        let out = fetch_many(&[]).await;
        assert!(out.is_empty());
        let out = fetch_many_text(&[]).await;
        assert!(out.is_empty());
    }

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
        // Any CC-BY/SA version qualifies; NC/ND never do.
        assert!(require_provenance("F", "P", "http://x/y.jpg", "A. Uthor", "CC BY-SA 2.5").is_ok());
        assert!(
            require_provenance("F", "P", "http://x/y.jpg", "A. Uthor", "CC BY-NC 2.0").is_err()
        );
        assert!(
            require_provenance("F", "P", "http://x/y.jpg", "A. Uthor", "CC BY-ND 4.0").is_err()
        );
    }

    #[test]
    fn safety_review_refuses_minors_and_nudity() {
        assert!(review_title("File:Man standing on a mountain.jpg").is_ok());
        assert!(review_title("File:Portrait of a woman.jpg").is_ok());
        // Whole-word matching: "childhood" is not the word "child"
        // (titles can still hide what words don't say — the review
        // is one gate, not a guarantee).
        assert!(review_title("File:Childhood street games.jpg").is_ok());
        for bad in [
            "File:A man posing naked.jpg",
            "File:Young boy smiling.jpg",
            "File:Girl with a flag.jpg",
            "File:Teenager portrait.jpg",
            "File:Erotic sculpture detail.jpg",
        ] {
            assert!(review_title(bad).is_err(), "must refuse {}", bad);
        }
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
