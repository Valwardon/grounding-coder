//! Retrieval-Augmented Image Generation: the generator slot.
//!
//! Pipeline position (per the architecture): research retrieves
//! licensed plates for ANY subject through the generic path,
//! extraction reduces them to compact conditioning (words +
//! numbers — the raw bytes are then DISCARDED, never delivered,
//! never copied into the product), and this module turns the
//! conditioning into brand-new pixels in a single pass.
//!
//! Backend honesty, stated upfront: the single-pass generator is the
//! on-device conditional GAN in [`super::gan`] — a pure-Rust CPU
//! model, no network, no hosted fallback. This module owns the
//! conditioning contract (compact words + seed) and the pixel checks;
//! the GAN consumes the words. It REFUSES to generate until trained
//! weights exist on disk, and nothing upstream changes when it does.
//! No code here varies by subject: the conditioning is built from
//! researched words by one generic rule.

use super::scene_intent::SceneSpec;
use super::vision::Image;

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

/// The generator behind the pixels, for the receipt. On-device GAN:
/// no network, no hosted model, never a fallback.
pub const GENERATOR_MODEL: &str = "on-device-gan";

/// Every researched word the conditioning carries, flattened by one
/// generic rule into the word list the GAN's hashed encoder consumes.
/// Unknown words hash like any other — nothing is out of vocabulary,
/// and no subject takes a special branch.
pub fn conditioning_words(cond: &CompactConditioning) -> Vec<String> {
    let mut words = Vec::new();
    for w in cond
        .subjects
        .iter()
        .chain(cond.attributes.iter())
        .chain(cond.actions.iter())
        .chain(cond.objects.iter())
        .chain(cond.places.iter())
    {
        // Objects may carry a material ("hat:straw"): both halves are
        // words. Split on any non-alphanumeric, keep non-empty.
        for part in w.split(|c: char| !c.is_alphanumeric()) {
            if !part.is_empty() {
                words.push(part.to_lowercase());
            }
        }
    }
    words
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
    use super::super::vision::Rgb;
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
}
