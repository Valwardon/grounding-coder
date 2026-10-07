//! The chat picture pipeline: the bot's own hands, end to end.
//!
//! When chat receives a picture request (human subject + poseable
//! action), it comes HERE — not to code tasks. The pipeline parses
//! the scene, researches its own sources (people from adult-filtered
//! OpenImages, places/objects/materials from Commons — the user
//! never names a source), measures the plates as structural
//! knowledge only (skin tone, framing, palette — never pixels),
//! and renders a fresh image from those numbers with the rest
//! receipted as defaults. No montage path exists.
//!
//! No step is skippable and no step is hand-driven: this is the
//! function the app's chat box calls.

use super::evidence::{self, EvidenceStore};
use super::vision::Rgb;

/// True when the prompt is a picture request: a human subject doing
/// something poseable, or an unclassified (researched, never guessed)
/// subject doing anything parsed. Everything else stays on the code
/// path.
pub fn is_picture_request(prompt: &str) -> bool {
    let spec = super::scene_intent::parse_scene(prompt);
    let human = spec.subjects.iter().any(|s| {
        matches!(
            s.stype.as_str(),
            "man" | "woman" | "human" | "person" | "people"
        )
    });
    let poseable = [
        "salute",
        "wave",
        "stand",
        "peace_sign",
        "sit",
        "walk",
        "point",
        "raise",
    ];
    if human
        && spec
            .actions
            .iter()
            .any(|a| poseable.contains(&a.atype.as_str()))
    {
        return true;
    }
    // Open vocabulary: a researched subject (human words excluded —
    // those took the branch above) with a parsed action is a picture
    // of that thing doing that thing. "A tower out of glass" stays
    // out: no action parsed.
    let generic = spec.subjects.iter().any(|s| {
        !matches!(
            s.stype.as_str(),
            "man" | "woman" | "human" | "person" | "people"
        )
    });
    generic && !spec.actions.is_empty()
}

/// Chat-style outcome: lines the app would show as the reply, plus
/// the files behind them.
#[derive(Debug)]
pub struct PictureOutcome {
    pub reply: Vec<String>,
    pub image_path: std::path::PathBuf,
    pub receipt_path: std::path::PathBuf,
}

/// Source the bot picks per requirement category. People come from
/// adult-filtered OpenImages (boxes, no minors by construction);
/// everything else from Commons text search. Deterministic table —
/// the user never chooses.
fn source_for(category: super::evidence::Category) -> &'static str {
    match category {
        super::evidence::Category::Subject => "openimages",
        super::evidence::Category::Action => "commons",
        super::evidence::Category::Object => "commons",
        super::evidence::Category::Material => "commons",
        super::evidence::Category::Place => "commons",
    }
}

/// Median skin tone across a requirement's examples (tone features
/// banked at ingestion). Tiered by collection state: Sufficient
/// collections give a researched palette; Collecting ones (at/above
/// minimum, below preferred) give a ROUGH palette that carries its
/// own warning. Below minimum: nothing — defaults stay honestly
/// defaulted. Returns (tone, note, n, sufficient).
pub fn derive_skin_palette(
    store: &EvidenceStore,
    req_id: &str,
) -> Option<(Rgb, String, usize, bool)> {
    let sufficient = matches!(
        store.sufficiency(req_id),
        Ok(evidence::Sufficiency::Sufficient { .. })
    );
    let collecting = matches!(
        store.sufficiency(req_id),
        Ok(evidence::Sufficiency::Collecting { .. })
    );
    if !sufficient && !collecting {
        return None;
    }
    let req = store.get(req_id)?;
    let mut rs = Vec::new();
    let mut gs = Vec::new();
    let mut bs = Vec::new();
    for ex in &req.examples {
        if let (Some(r), Some(g), Some(b)) = (
            ex.features.get("tone_r"),
            ex.features.get("tone_g"),
            ex.features.get("tone_b"),
        ) {
            rs.push(*r as u8);
            gs.push(*g as u8);
            bs.push(*b as u8);
        }
    }
    if rs.is_empty() {
        return None;
    }
    rs.sort_unstable();
    gs.sort_unstable();
    bs.sort_unstable();
    let mid = rs.len() / 2;
    let note = if sufficient {
        format!("researched: {} photo tones", req.examples.len())
    } else {
        format!(
            "rough: {} photo tones (below preferred {})",
            req.examples.len(),
            store.preferred
        )
    };
    Some((
        Rgb::new(rs[mid], gs[mid], bs[mid]),
        note,
        req.examples.len(),
        sufficient,
    ))
}

/// Rough flag model from a flag requirement's banked plates: reload
/// each example's saved BMP, measure stripes classically, derive
/// with the rough gate (3+). Anything below that is not a model —
/// the cloth stays honestly gray.
pub fn derive_flag_for(
    store: &EvidenceStore,
    req_id: &str,
) -> Option<(super::object_model::FlagModel, String)> {
    let req = store.get(req_id)?;
    let mut measures = Vec::new();
    for ex in &req.examples {
        // Unreadable plates skip (logged nowhere here — the evidence
        // record already vouches the file existed at banking); the
        // rough gate judges what remains.
        if let Ok(img) = super::vision::Image::load_bmp(std::path::Path::new(&ex.source)) {
            measures.push(super::object_model::measure_flag(&img));
        }
    }
    match super::object_model::rough_flag_model(&measures) {
        Ok(m) => {
            let note = format!(
                "researched: {} stripes, agreement {:.2} over {} examples",
                m.stripes, m.stripe_agreement, m.n
            );
            Some((m, note))
        }
        Err(_) => None,
    }
}

/// Run the whole picture loop for a chat prompt. `per_req` bounds
/// examples per requirement; `out_dir` receives plates plus the
/// finished image and receipt. Plates downscale past `max_dim`
/// (evidence features don't need full resolution; the repo stays
/// lean like every other collection path).
pub async fn picture_from_prompt(
    prompt: &str,
    out_dir: &std::path::Path,
    per_req: u32,
) -> Result<PictureOutcome, String> {
    picture_from_prompt_dim(prompt, out_dir, per_req, 640).await
}

fn downscale(img: &super::vision::Image, max_dim: u32) -> super::vision::Image {
    let longest = img.width.max(img.height);
    if longest <= max_dim.max(16) {
        return img.clone();
    }
    let cap = max_dim.max(16);
    if img.width >= img.height {
        img.resize_smooth(cap, img.height * cap / img.width)
    } else {
        img.resize_smooth(img.width * cap / img.height, cap)
    }
}

async fn picture_from_prompt_dim(
    prompt: &str,
    out_dir: &std::path::Path,
    per_req: u32,
    max_dim: u32,
) -> Result<PictureOutcome, String> {
    std::fs::create_dir_all(out_dir).map_err(|e| format!("cannot create dir: {}", e))?;
    // Fresh-only: per-request training, narrow scope. Research what
    // this prompt needs (subject + pose + backdrop) as structural
    // measurements, then render fresh pixels — never a montage.
    match super::imagine::imagine(prompt, 640, 800).await {
        Ok((img, ilog)) => {
            let image_path = out_dir.join("picture.bmp");
            img.save_bmp(&image_path)
                .map_err(|e| format!("cannot save picture: {}", e))?;
            let receipt_path = out_dir.join("picture.json");
            let mut reply = vec![
                "I see a picture request — researching structure for this prompt now.".to_string(),
            ];
            for l in &ilog {
                reply.push(format!("research: {}", l));
            }
            reply.push(
                "path: fresh construction (researched measurements, rendered pixels)".to_string(),
            );
            reply.push(format!("Done — picture at {}", image_path.display()));
            let body = serde_json::json!({
                "prompt": prompt,
                "path": "fresh-construction",
                "reply": reply,
                "log": ilog,
            });
            std::fs::write(
                &receipt_path,
                serde_json::to_string_pretty(&body).unwrap_or_default(),
            )
            .map_err(|e| format!("cannot save receipt: {}", e))?;
            Ok(PictureOutcome {
                reply,
                image_path,
                receipt_path,
            })
        }
        Err(ilog) => {
            // Fresh construction refused — keep its trail for the
            // receipt, then fall through to evidence + construction
            // with per-requirement models (honestly labeled).
            let mut reply = vec![format!(
                "Fresh construction refused ({} step(s)) — falling back to evidence + construction.",
                ilog.len()
            )];
            for l in &ilog {
                reply.push(format!("research: {}", l));
            }
            picture_fallback_evidence(prompt, out_dir, per_req, max_dim, reply).await
        }
    }
}

async fn picture_fallback_evidence(
    prompt: &str,
    out_dir: &std::path::Path,
    per_req: u32,
    max_dim: u32,
    mut reply: Vec<String>,
) -> Result<PictureOutcome, String> {
    use super::scene_intent::plan_research;
    let spec = super::scene_intent::parse_scene(prompt);
    let plan = plan_research(&spec);
    let mut store = EvidenceStore::new();
    store.require_from_spec(&spec, &plan);
    reply.push(format!(
        "I see {} — {} requirement(s). Researching each one now.",
        spec.subjects
            .first()
            .map(|s| s.stype.clone())
            .unwrap_or_else(|| "a scene".to_string()),
        store.requirements.len()
    ));

    std::fs::create_dir_all(out_dir).map_err(|e| format!("cannot create dir: {}", e))?;
    let per = per_req.clamp(1, 100) as usize;
    let ids: Vec<String> = store.requirements.iter().map(|r| r.id.clone()).collect();
    let mut n = 0u32;
    for id in &ids {
        let req = store.get(id).ok_or("requirement vanished")?.clone();
        let src = source_for(req.category);
        reply.push(format!("{}: looking on {} …", id, src));
        let mut have = 0usize;
        for q in &req.research_queries {
            if have >= per {
                break;
            }
            let need = (per - have) as u32;
            // OpenImages is a people-only index: only human subject
            // requirements sweep it. A cat requirement answered with
            // Woman boxes would be a false candidate set — Commons
            // serves every other subject through the same gates.
            let people_subject = matches!(req.category, super::evidence::Category::Subject)
                && matches!(
                    req.concept.as_str(),
                    "man" | "woman" | "human" | "person" | "people"
                );
            if src == "openimages" && people_subject {
                let cache = out_dir.join(".oicache");
                let (found, refused) =
                    super::plates::openimages::search_openimages(&cache, q, need).await;
                for r in &refused {
                    reply.push(format!("  {}: {}", id, r));
                }
                for hit in found {
                    if have >= per {
                        break;
                    }
                    let file = format!("plate-{:02}.bmp", n);
                    n += 1;
                    let path = out_dir.join(&file);
                    let small = downscale(&hit.plate.image, max_dim);
                    if small.save_bmp(&path).is_err() {
                        reply.push(format!("  {}: cannot save {}", id, file));
                        continue;
                    }
                    let (w, h) = (small.width, small.height);
                    let ex = evidence::VisualExample {
                        source: path.to_string_lossy().to_string(),
                        title: hit.plate.title.clone(),
                        page_url: hit.plate.provenance.page_url.clone(),
                        license: "openimages".to_string(),
                        basis: "Open Images adult-filtered boxes".to_string(),
                        bbox: [
                            (hit.bbox.0 * w as f64) as u32,
                            (hit.bbox.1 * h as f64) as u32,
                            (hit.bbox.2 * w as f64) as u32,
                            (hit.bbox.3 * h as f64) as u32,
                        ],
                        segmentation: None,
                        keypoints: Some(
                            "unavailable: fine-joint extraction not implemented".to_string(),
                        ),
                        dimensions: (w, h),
                        features: evidence::features_from_plate(&small),
                    };
                    if store.add_example(id, ex).is_err() {
                        break;
                    }
                    have += 1;
                }
            } else {
                let (sourced, refused) = super::plates::source_plates(q, need).await;
                for r in &refused {
                    reply.push(format!("  {}: {}", id, r));
                }
                for plate in sourced {
                    if have >= per {
                        break;
                    }
                    let file = format!("plate-{:02}.bmp", n);
                    n += 1;
                    let path = out_dir.join(&file);
                    let small = downscale(&plate.image, max_dim);
                    if small.save_bmp(&path).is_err() {
                        reply.push(format!("  {}: cannot save {}", id, file));
                        continue;
                    }
                    let ex = evidence::example_from_plate(
                        &path.to_string_lossy(),
                        &plate.title,
                        &plate.provenance.page_url,
                        &plate.provenance.license,
                        &plate.basis,
                        &small,
                    );
                    if store.add_example(id, ex).is_err() {
                        break;
                    }
                    have += 1;
                }
            }
        }
        match store.sufficiency(id) {
            Ok(evidence::Sufficiency::Insufficient { n, need }) => reply.push(format!(
                "{}: {} example(s) — not enough to model (need {}), constructing with defaults",
                id, n, need
            )),
            Ok(evidence::Sufficiency::Collecting { n }) => {
                reply.push(format!("{}: {} examples — collecting", id, n))
            }
            Ok(evidence::Sufficiency::Sufficient { n }) => {
                reply.push(format!("{}: {} examples — modeling", id, n))
            }
            Err(e) => reply.push(format!("{}: {}", id, e)),
        }
    }
    evidence::save_jsonl(&store, &out_dir.join("evidence.jsonl"))
        .map_err(|e| format!("cannot save evidence: {}", e))?;

    // Researched inputs apply when the collections support them:
    // skin palettes tiered (researched at Sufficient, rough at
    // Collecting), flag models rough-gated (3+ measured plates).
    // Everything else stays defaulted with the receipt saying so.
    let mut skin: Option<(Rgb, String)> = None;
    for r in &store.requirements {
        if r.category == super::evidence::Category::Subject
            && let Some((c, src, _, _)) = derive_skin_palette(&store, &r.id)
        {
            skin = Some((c, src));
            reply.push(format!("{}: skin palette researched", r.id));
            break;
        }
    }
    let mut flag: Option<super::object_model::FlagModel> = None;
    for r in &store.requirements {
        if r.category == super::evidence::Category::Object
            && r.concept.contains("flag")
            && let Some((m, note)) = derive_flag_for(&store, &r.id)
        {
            reply.push(format!("{}: {}", r.id, note));
            flag = Some(m);
            break;
        }
    }
    let creation = super::studio::create_image_with(prompt, skin, flag)
        .map_err(|e| format!("cannot construct: {}", e))?;
    let image_path = out_dir.join("picture.bmp");
    creation
        .image
        .save_bmp(&image_path)
        .map_err(|e| format!("cannot save picture: {}", e))?;
    let receipt_path = out_dir.join("picture.json");
    let body = serde_json::json!({
        "prompt": prompt,
        "reply": reply,
        "requirements": creation.receipt.iter().map(|r| {
            serde_json::json!({"requirement": r.requirement, "researched": r.researched, "note": r.note})
        }).collect::<Vec<_>>(),
    });
    std::fs::write(
        &receipt_path,
        serde_json::to_string_pretty(&body).unwrap_or_default(),
    )
    .map_err(|e| format!("cannot save receipt: {}", e))?;
    let mut out_reply = reply;
    out_reply.push(format!("Done — picture at {}", image_path.display()));
    Ok(PictureOutcome {
        reply: out_reply,
        image_path,
        receipt_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_sends_pictures_to_pictures() {
        assert!(is_picture_request("a man standing on a mountain"));
        assert!(is_picture_request(
            "Make a man saluting, waving an American flag."
        ));
        assert!(is_picture_request("Human woman giving peace sign."));
        assert!(!is_picture_request("Create a Counter struct."));
        assert!(!is_picture_request("Build the android apk."));
        assert!(!is_picture_request("!!! ... ???"));
        assert!(!is_picture_request("A tower out of glass."));
        // Human but no poseable action: not a picture request.
        assert!(!is_picture_request("A man."));
    }

    #[test]
    fn routing_sends_generic_subjects_to_pictures() {
        // Open vocabulary: an unclassified subject with a parsed
        // action is a picture request — research resolves the noun.
        assert!(is_picture_request("An elephant crossing a river."));
        // Still out: action-less scenes and code tasks.
        assert!(!is_picture_request("An elephant."));
        assert!(!is_picture_request("A tower out of glass."));
        // The sitter precedes the seat: a picture request.
        assert!(is_picture_request("Cat sitting in human's lap."));
    }

    #[test]
    fn subjects_come_from_openimages() {
        assert_eq!(source_for(super::evidence::Category::Subject), "openimages");
        assert_eq!(source_for(super::evidence::Category::Place), "commons");
        assert_eq!(source_for(super::evidence::Category::Material), "commons");
    }

    #[test]
    fn skin_tiers_gate_honestly() {
        // Budgets: minimum 2, preferred 3. Below minimum nothing;
        // at minimum a ROUGH palette with its warning; at preferred
        // a researched one.
        let mut store = EvidenceStore::with_budgets(2, 3, 4);
        store.requirements.push(super::evidence::VisualRequirement {
            id: "subject:man".to_string(),
            category: super::evidence::Category::Subject,
            concept: "man".to_string(),
            attributes: Vec::new(),
            relationships: Vec::new(),
            research_queries: Vec::new(),
            examples: Vec::new(),
            model: None,
        });
        assert!(derive_skin_palette(&store, "subject:man").is_none());
        let tones = |r: f64, g: f64, b: f64| {
            let mut f = std::collections::HashMap::new();
            f.insert("tone_r".to_string(), r);
            f.insert("tone_g".to_string(), g);
            f.insert("tone_b".to_string(), b);
            f.insert("skin_head".to_string(), 0.2);
            f.insert("skin_torso".to_string(), 0.2);
            f.insert("skin_legs".to_string(), 0.1);
            f.insert("aspect".to_string(), 0.5);
            f.insert("brightness".to_string(), 0.5);
            super::evidence::VisualExample {
                source: "test".to_string(),
                title: "test".to_string(),
                page_url: "test".to_string(),
                license: "test".to_string(),
                basis: "test".to_string(),
                bbox: [0, 0, 8, 8],
                segmentation: None,
                keypoints: None,
                dimensions: (8, 8),
                features: f,
            }
        };
        store
            .add_example("subject:man", tones(200.0, 150.0, 115.0))
            .unwrap();
        store
            .add_example("subject:man", tones(205.0, 155.0, 120.0))
            .unwrap();
        let (c, note, n, sufficient) =
            derive_skin_palette(&store, "subject:man").expect("rough palette");
        assert_eq!((n, sufficient), (2, false));
        assert!(note.contains("rough"), "must carry its warning: {}", note);
        assert!((c.r as i32 - 200).abs() <= 8);
        store
            .add_example("subject:man", tones(195.0, 145.0, 110.0))
            .unwrap();
        let (_, note, n, sufficient) =
            derive_skin_palette(&store, "subject:man").expect("researched palette");
        assert_eq!((n, sufficient), (3, true));
        assert!(note.contains("researched"), "{}", note);
    }
}
