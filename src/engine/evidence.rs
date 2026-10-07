//! Evidence store — the research phase, as data structures.
//!
//! Spec §2–§4: every scene requirement owns its research queries,
//! its collected examples, and (later) its derived model. Photos
//! become measurements at ingestion: bbox, segmentation summary,
//! dimensions, and a feature vector — never bare image files.
//! Counts enforce the 20/100/500 budgets: below minimum the store
//! reports `Insufficient` with the shortfall, never a model.
//!
//! Collection itself is a thin runner over query strings
//! (`collect_examples` takes any fetch function — the live path
//! feeds it `plates::source_plates`; tests feed it synthetics).
//! Keypoints stay `None` with the reason until fine-joint
//! extraction exists (`body_measure` reports it missing).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// What kind of requirement this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Category {
    Subject,
    Action,
    Object,
    Material,
    Place,
}

/// A named relationship to another requirement or actor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub rel: String,
    pub target: String,
}

/// One scene requirement with everything research found so far.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualRequirement {
    pub id: String,
    pub category: Category,
    pub concept: String,
    pub attributes: Vec<String>,
    pub relationships: Vec<Relationship>,
    pub research_queries: Vec<String>,
    pub examples: Vec<VisualExample>,
    /// Model label once derived (e.g. "ranges-v1"). None until then —
    /// examples are not a model.
    pub model: Option<String>,
}

/// One researched visual example: measurements, not pixels.
/// The image itself lives beside the store (plate files); this
/// record is what the model builders read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualExample {
    /// Local plate file the measurements came from.
    pub source: String,
    /// Source title (Commons file title / page title) — audit trail.
    pub title: String,
    /// Human-readable description page.
    pub page_url: String,
    /// License short name — provenance travels with the data.
    pub license: String,
    /// How the license was determined.
    pub basis: String,
    /// Bounding box (x0, y0, x1, y1). Full frame when no detector
    /// box exists — stated, not hidden.
    pub bbox: [u32; 4],
    /// Segmentation summary ("skin-blob:0.23") or "none".
    pub segmentation: Option<String>,
    /// Keypoints: None until fine-joint extraction exists, with the
    /// reason inline. No silent nulls.
    pub keypoints: Option<String>,
    /// Pixel dimensions.
    pub dimensions: (u32, u32),
    /// Feature vector: coarse bands, aspect, tones, brightness.
    pub features: HashMap<String, f64>,
}

/// How far collection has come.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sufficiency {
    /// Below minimum — not usable, shortfall stated.
    Insufficient { n: usize, need: usize },
    /// Minimum met, preferred not yet — usable with care.
    Collecting { n: usize },
    /// Preferred met — a model may be derived.
    Sufficient { n: usize },
}

/// The evidence store: requirements plus enforced budgets.
#[derive(Debug, Clone)]
pub struct EvidenceStore {
    pub requirements: Vec<VisualRequirement>,
    pub minimum: usize,
    pub preferred: usize,
    pub maximum: usize,
}

/// Spec §3 budgets.
pub const MINIMUM_EXAMPLES: usize = 20;
pub const PREFERRED_EXAMPLES: usize = 100;
pub const MAXIMUM_EXAMPLES: usize = 500;

impl EvidenceStore {
    pub fn new() -> Self {
        EvidenceStore {
            requirements: Vec::new(),
            minimum: MINIMUM_EXAMPLES,
            preferred: PREFERRED_EXAMPLES,
            maximum: MAXIMUM_EXAMPLES,
        }
    }

    /// Small budgets for tests (same gates, fast runs).
    pub fn with_budgets(minimum: usize, preferred: usize, maximum: usize) -> Self {
        EvidenceStore {
            requirements: Vec::new(),
            minimum,
            preferred,
            maximum,
        }
    }

    /// Derive requirements from a parsed scene + its research plan.
    /// Generic over every concept and material — nothing here names
    /// any particular object, action, or substance.
    pub fn require_from_spec(
        &mut self,
        spec: &super::scene_intent::SceneSpec,
        plan: &[super::scene_intent::ResearchQuery],
    ) {
        let queries_for = |needle: &str| -> Vec<String> {
            plan.iter()
                .filter(|q| {
                    q.requirement.contains(needle) || q.queries.iter().any(|s| s.contains(needle))
                })
                .flat_map(|q| q.queries.clone())
                .collect::<Vec<_>>()
        };
        for s in &spec.subjects {
            let label = if s.attributes.is_empty() {
                s.stype.clone()
            } else {
                format!("{} {}", s.attributes.join(" "), s.stype)
            };
            self.requirements.push(VisualRequirement {
                id: format!("subject:{}", label),
                category: Category::Subject,
                concept: s.stype.clone(),
                attributes: s.attributes.clone(),
                relationships: Vec::new(),
                research_queries: queries_for(&label),
                examples: Vec::new(),
                model: None,
            });
        }
        for a in &spec.actions {
            let mut rels = Vec::new();
            if let Some(t) = &a.target {
                rels.push(Relationship {
                    rel: "target".to_string(),
                    target: t.clone(),
                });
            }
            if let Some(o) = &a.object {
                rels.push(Relationship {
                    rel: "object".to_string(),
                    target: o.clone(),
                });
            }
            rels.push(Relationship {
                rel: "actor".to_string(),
                target: a.actor.clone(),
            });
            self.requirements.push(VisualRequirement {
                id: format!("action:{}", a.atype),
                category: Category::Action,
                concept: a.atype.clone(),
                attributes: if a.ambiguous {
                    vec!["ambiguous".to_string()]
                } else {
                    Vec::new()
                },
                relationships: rels,
                research_queries: queries_for(&a.atype),
                examples: Vec::new(),
                model: None,
            });
        }
        for o in &spec.objects {
            let mut rels = Vec::new();
            if let Some(w) = &o.worn_by {
                rels.push(Relationship {
                    rel: "worn_by".to_string(),
                    target: w.clone(),
                });
            }
            self.requirements.push(VisualRequirement {
                id: format!("object:{}", o.id),
                category: if o.otype == "mountain" {
                    Category::Place
                } else {
                    Category::Object
                },
                concept: o.otype.clone(),
                attributes: o.attributes.clone(),
                relationships: rels,
                research_queries: queries_for(&o.id),
                examples: Vec::new(),
                model: None,
            });
            if let Some(m) = &o.material {
                self.requirements.push(VisualRequirement {
                    id: format!("material:{}-on-{}", m, o.id),
                    category: Category::Material,
                    concept: m.clone(),
                    attributes: Vec::new(),
                    relationships: vec![Relationship {
                        rel: "material_on".to_string(),
                        target: o.id.clone(),
                    }],
                    research_queries: queries_for(m),
                    examples: Vec::new(),
                    model: None,
                });
            }
        }
    }

    pub fn get(&self, id: &str) -> Option<&VisualRequirement> {
        self.requirements.iter().find(|r| r.id == id)
    }

    /// Bank one example. Unknown ids and over-maximum stores refuse.
    pub fn add_example(&mut self, id: &str, ex: VisualExample) -> Result<(), String> {
        let req = self
            .requirements
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or_else(|| format!("unknown requirement '{}' — refusing", id))?;
        if req.examples.len() >= self.maximum {
            return Err(format!(
                "requirement '{}' at maximum {} — refusing",
                id, self.maximum
            ));
        }
        req.examples.push(ex);
        Ok(())
    }

    /// Sufficiency of one requirement's collection.
    pub fn sufficiency(&self, id: &str) -> Result<Sufficiency, String> {
        let req = self
            .get(id)
            .ok_or_else(|| format!("unknown requirement '{}'", id))?;
        let n = req.examples.len();
        Ok(if n < self.minimum {
            Sufficiency::Insufficient {
                n,
                need: self.minimum,
            }
        } else if n < self.preferred {
            Sufficiency::Collecting { n }
        } else {
            Sufficiency::Sufficient { n }
        })
    }
}

impl Default for EvidenceStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature extraction at ingestion: skin-mask fractions in head /
/// torso / legs thirds-bands, frame aspect, mean brightness, and
/// median skin tone — straight off the engine's own vision buffer.
/// Pure and deterministic. These features rank researched plates for
/// delivery; they never feed a renderer.
pub fn features_from_plate(img: &super::vision::Image) -> HashMap<String, f64> {
    let (w, h) = (img.width, img.height);
    let mask = img.skin_mask();
    let band_frac = |y0: u32, y1: u32| -> f64 {
        if w == 0 || y1 <= y0 {
            return 0.0;
        }
        let mut n = 0u64;
        let mut total = 0u64;
        for y in y0..y1.min(h) {
            for x in 0..w {
                total += 1;
                if mask[(y * w + x) as usize] {
                    n += 1;
                }
            }
        }
        if total == 0 {
            0.0
        } else {
            n as f64 / total as f64
        }
    };
    let mut f = HashMap::new();
    f.insert("skin_head".to_string(), band_frac(0, h / 3));
    f.insert("skin_torso".to_string(), band_frac(h / 3, 2 * h / 3));
    f.insert("skin_legs".to_string(), band_frac(2 * h / 3, h));
    f.insert(
        "aspect".to_string(),
        if h == 0 { 0.0 } else { w as f64 / h as f64 },
    );
    f.insert("brightness".to_string(), img.mean_brightness());
    // Median tone over masked pixels only.
    let mut rs = Vec::new();
    let mut gs = Vec::new();
    let mut bs = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if !mask[(y * w + x) as usize] {
                continue;
            }
            if let Some(p) = img.get(x, y) {
                rs.push(p.r);
                gs.push(p.g);
                bs.push(p.b);
            }
        }
    }
    if !rs.is_empty() {
        rs.sort_unstable();
        gs.sort_unstable();
        bs.sort_unstable();
        let mid = rs.len() / 2;
        f.insert("tone_r".to_string(), rs[mid] as f64);
        f.insert("tone_g".to_string(), gs[mid] as f64);
        f.insert("tone_b".to_string(), bs[mid] as f64);
    }
    f
}

/// Build a VisualExample record for a sourced plate. Bbox is the
/// full frame (no detector box) — stated in the record. Keypoints
/// are None with the reason: fine joints are still unextractable.
pub fn example_from_plate(
    source: &str,
    title: &str,
    page_url: &str,
    license: &str,
    basis: &str,
    img: &super::vision::Image,
) -> VisualExample {
    let features = features_from_plate(img);
    let skin_total = features.get("skin_head").copied().unwrap_or(0.0)
        + features.get("skin_torso").copied().unwrap_or(0.0)
        + features.get("skin_legs").copied().unwrap_or(0.0);
    VisualExample {
        source: source.to_string(),
        title: title.to_string(),
        page_url: page_url.to_string(),
        license: license.to_string(),
        basis: basis.to_string(),
        bbox: [0, 0, img.width, img.height],
        segmentation: Some(format!("skin-blob:{:.2}", skin_total)),
        keypoints: Some("unavailable: fine-joint extraction not implemented".to_string()),
        dimensions: (img.width, img.height),
        features,
    }
}

/// Collection runner: walk the queries in order, fetching until
/// `want` examples bank or queries run out. Returns what banked plus
/// the trail (every query tried, hits and misses). The fetch function
/// is the only I/O — tests inject synthetics, the live path injects
/// `plates::source_plates`.
pub fn collect_examples<F>(
    queries: &[String],
    want: usize,
    mut fetch: F,
) -> (Vec<VisualExample>, Vec<String>)
where
    F: FnMut(&str, usize) -> (Vec<VisualExample>, Vec<String>),
{
    let mut banked = Vec::new();
    let mut trail = Vec::new();
    for q in queries {
        if banked.len() >= want {
            break;
        }
        let need = want - banked.len();
        let (mut got, refused) = fetch(q, need);
        trail.push(format!("query '{}': {} hit(s)", q, got.len()));
        for r in refused {
            trail.push(format!("  refused: {}", r));
        }
        banked.append(&mut got);
    }
    banked.truncate(want);
    (banked, trail)
}

/// Persist the store beside its plates: one JSON object per line.
pub fn save_jsonl(store: &EvidenceStore, path: &std::path::Path) -> Result<(), String> {
    let mut out = String::new();
    for r in &store.requirements {
        out.push_str(&serde_json::to_string(r).map_err(|e| format!("serialize: {}", e))?);
        out.push('\n');
    }
    std::fs::write(path, out).map_err(|e| format!("write: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_match_spec() {
        assert_eq!(MINIMUM_EXAMPLES, 20);
        assert_eq!(PREFERRED_EXAMPLES, 100);
        assert_eq!(MAXIMUM_EXAMPLES, 500);
    }
}
