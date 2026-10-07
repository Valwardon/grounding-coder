//! Retrieval-Augmented Image Generation: the generator slot.
//!
//! Pipeline position (per the architecture): research retrieves
//! licensed plates for ANY subject through the generic path,
//! extraction reduces them to compact conditioning (words +
//! numbers — the raw bytes are then DISCARDED, never delivered,
//! never copied into the product), and this module turns the
//! conditioning into brand-new pixels in a single pass.
//!
//! Backend honesty, stated upfront: the single-pass generator
//! running here is a hosted image model behind a network call —
//! the receipt names it, with model, seed, and the exact
//! conditioning it consumed. An on-device lightweight GAN takes
//! this same conditioning contract when it lands; nothing upstream
//! changes, because upstream only ever produced conditioning.
//! No code here varies by subject: the prompt is built from
//! researched words by one generic rule.

use super::scene_intent::SceneSpec;
use super::vision::{Image, Rgb};

/// Compact knowledge: everything the generator may consume, small
/// enough to log in full. Words come from the researched spec;
/// counts come from the research rounds. Raw plate bytes NEVER
/// enter this struct — they are discarded before generation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct CompactConditioning {
    /// Researched subjects, in scene order.
    pub subjects: Vec<String>,
    /// Subject attributes (modifiers that refined a type).
    pub attributes: Vec<String>,
    /// Researched actions, in prose order.
    pub actions: Vec<String>,
    /// Researched objects with optional materials ("hat:straw").
    pub objects: Vec<String>,
    /// Scene place words (mountain, river, …), if parsed.
    pub places: Vec<String>,
    /// Licensed plates researched behind this conditioning.
    pub plates_researched: usize,
    /// Deterministic seed for the single pass.
    pub seed: u64,
}

/// Extract compact conditioning from a researched scene: nouns and
/// verbs verbatim, nothing inferred. Same rule for a woman, a cat,
/// or a tree — the function never branches on what the words mean.
pub fn conditioning_from_spec(
    spec: &SceneSpec,
    plates_researched: usize,
    seed: u64,
) -> CompactConditioning {
    let mut attributes = Vec::new();
    for s in &spec.subjects {
        attributes.extend(
            s.attributes
                .iter()
                .filter(|a| *a != "unclassified")
                .cloned(),
        );
    }
    let mut places = Vec::new();
    let mut objects = Vec::new();
    for o in &spec.objects {
        match o.material.as_deref() {
            Some(m) => objects.push(format!("{}:{}", o.otype, m)),
            None => objects.push(o.otype.clone()),
        }
        // Place-typed objects double as scene places for the prompt.
        if matches!(
            o.otype.as_str(),
            "mountain" | "river" | "lake" | "ocean" | "forest" | "desert" | "sky" | "beach"
        ) {
            places.push(o.otype.clone());
        }
    }
    CompactConditioning {
        subjects: spec.subjects.iter().map(|s| s.stype.clone()).collect(),
        attributes,
        actions: spec.actions.iter().map(|a| a.atype.clone()).collect(),
        objects,
        places,
        plates_researched,
        seed,
    }
}

/// Render conditioning as the generator prompt: one deterministic
/// rule — subjects doing actions with objects somewhere, photographic
/// style fixed. The model photographs the description; research
/// wrote the description.
pub fn build_prompt(cond: &CompactConditioning) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !cond.subjects.is_empty() {
        parts.push(cond.subjects.join(" with "));
    }
    if !cond.attributes.is_empty() {
        parts.push(format!("({})", cond.attributes.join(", ")));
    }
    if !cond.actions.is_empty() {
        parts.push(cond.actions.join(", "));
    }
    if !cond.objects.is_empty() {
        parts.push(format!("with {}", cond.objects.join(", ")));
    }
    if !cond.places.is_empty() {
        parts.push(format!("at {}", cond.places.join(", ")));
    }
    let scene = if parts.is_empty() {
        "empty scene".to_string()
    } else {
        parts.join(" ")
    };
    format!(
        "Photorealistic photograph, natural light, realistic detail: {}",
        scene
    )
}

/// Percent-encode a prompt for a URL path segment.
fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b" -_.~".contains(&b) {
            if b == b' ' {
                out.push_str("%20");
            } else {
                out.push(b as char);
            }
        } else {
            out.push_str(&format!("%{:02X}", b));
        }
    }
    out
}

/// Generator backend identity for receipts.
pub const GENERATOR_MODEL: &str = "pollinations-flux";
/// Hard bound on one generation pass.
pub const GENERATE_TIMEOUT_SECS: u64 = 300;

/// One generation pass: conditioning → brand-new pixels. Seeded and
/// single-pass; the bytes come back decoded into the engine buffer.
/// Fails honestly (network, decode, degenerate frames) — failure
/// refuses, never falls back to donor bytes.
pub async fn generate_photo(
    cond: &CompactConditioning,
    width: u32,
    height: u32,
) -> Result<Image, String> {
    let prompt = build_prompt(cond);
    let url = format!(
        "https://image.pollinations.ai/prompt/{}?width={}&height={}&seed={}&nologo=true&model=flux",
        url_encode(&prompt),
        width.clamp(64, 1280),
        height.clamp(64, 1280),
        cond.seed,
    );
    let bytes =
        crate::http::get_bytes_timeout(&url, std::time::Duration::from_secs(GENERATE_TIMEOUT_SECS))
            .await
            .map_err(|e| format!("generator unreachable: {}", e))?;
    if bytes.len() < 1024 {
        return Err(format!(
            "generator returned {} bytes — refusing short read",
            bytes.len()
        ));
    }
    let dyn_img =
        image::load_from_memory(&bytes).map_err(|e| format!("generated decode failed: {}", e))?;
    let rgb = dyn_img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    if w < 16 || h < 16 || w > 4096 || h > 4096 {
        return Err(format!("refusing generated frame of {}x{}", w, h));
    }
    let mut img = Image::blank(w, h, Rgb::new(0, 0, 0));
    for (x, y, p) in rgb.enumerate_pixels() {
        img.set(x, y, Rgb::new(p[0], p[1], p[2]));
    }
    // Blank-frame guard: a flat field is a failed pass, not a photo.
    let mut distinct = std::collections::HashSet::new();
    for y in (0..h).step_by(7) {
        for x in (0..w).step_by(7) {
            if let Some(p) = img.get(x, y) {
                distinct.insert((p.r >> 4, p.g >> 4, p.b >> 4));
            }
        }
    }
    if distinct.len() < 8 {
        return Err(format!(
            "generated frame nearly flat ({} tones) — refusing",
            distinct.len()
        ));
    }
    Ok(img)
}

/// Mean per-channel absolute difference, 0.0–1.0, over the overlap
/// window. Zero means the same pixels. Generic pixel math — the
/// novelty assertion between product and researched donors.
pub fn novelty(a: &Image, b: &Image) -> f64 {
    let (w, h) = (a.width.min(b.width), a.height.min(b.height));
    if w == 0 || h == 0 {
        return 1.0;
    }
    let mut acc = 0u64;
    let mut n = 0u64;
    for y in 0..h {
        for x in 0..w {
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

/// Generation receipt: what made the pixels, exactly.
#[derive(Debug, Clone)]
pub struct GenerationReceipt {
    pub model: String,
    pub seed: u64,
    pub prompt: String,
    pub conditioning: CompactConditioning,
    pub min_novelty_vs_donors: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::scene_intent::parse_scene;

    #[test]
    fn prompt_builds_generically_from_research() {
        // Same rule for a woman, a cat, a tree: nouns in, photo
        // description out. No subject branches.
        let man = conditioning_from_spec(&parse_scene("A man standing on a mountain."), 4, 1);
        let cat = conditioning_from_spec(&parse_scene("Cat sitting in human's lap."), 4, 1);
        let tree = conditioning_from_spec(&parse_scene("A tree."), 4, 1);
        let (pm, pc, pt) = (build_prompt(&man), build_prompt(&cat), build_prompt(&tree));
        for p in [&pm, &pc, &pt] {
            assert!(p.starts_with("Photorealistic photograph"), "{}", p);
        }
        assert!(pm.contains("man") && pm.contains("mountain"), "{}", pm);
        assert!(pc.contains("cat") && pc.contains("human"), "{}", pc);
        assert!(pt.contains("tree"), "{}", pt);
        // Deterministic: same spec twice, same prompt.
        assert_eq!(
            pm,
            build_prompt(&conditioning_from_spec(
                &parse_scene("A man standing on a mountain."),
                4,
                1
            ))
        );
    }

    #[test]
    fn unclassified_never_reaches_the_prompt() {
        // Open-vocabulary markers are parser bookkeeping, not
        // description — the generator must never see them.
        let cond = conditioning_from_spec(&parse_scene("An elephant crossing a river."), 8, 1);
        assert!(cond.attributes.is_empty(), "{:?}", cond.attributes);
        assert!(!build_prompt(&cond).contains("unclassified"));
    }

    #[test]
    fn conditioning_carries_no_pixels() {
        // Compact knowledge is words + counts: serializes small,
        // contains no image data by construction (no such field).
        let cond = conditioning_from_spec(&parse_scene("A man standing on a mountain."), 12, 7);
        assert_eq!(cond.plates_researched, 12);
        assert_eq!(cond.seed, 7);
        let json = serde_json::to_value(&cond).unwrap();
        assert!(json.get("subjects").is_some());
        assert!(json.get("image").is_none());
        assert!(json.get("pixels").is_none());
        assert!(
            json.to_string().len() < 1024,
            "conditioning must stay compact"
        );
    }

    #[test]
    fn novelty_separates_copy_from_stranger() {
        let a = Image::blank(32, 32, Rgb::new(200, 150, 115));
        let b = Image::blank(32, 32, Rgb::new(200, 150, 115));
        assert_eq!(novelty(&a, &b), 0.0);
        let mut c = Image::blank(32, 32, Rgb::new(200, 150, 115));
        c.set(0, 0, Rgb::new(0, 0, 0));
        assert!(novelty(&a, &c) > 0.0);
        let d = Image::blank(32, 32, Rgb::new(10, 20, 200));
        assert!(novelty(&a, &d) > 0.2, "strangers must differ");
    }

    #[test]
    fn url_encode_is_safe() {
        assert_eq!(url_encode("a man"), "a%20man");
        assert!(!url_encode(" trees/river? ").contains('/'));
    }
}
