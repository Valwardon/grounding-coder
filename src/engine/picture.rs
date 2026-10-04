//! The chat picture pipeline: the bot's own hands, end to end.
//!
//! When chat receives a picture request (human subject + poseable
//! action), it comes HERE — not to code tasks. The pipeline parses
//! the scene, chooses its own sources (people from adult-filtered
//! OpenImages, places/objects/materials from Commons — the user
//! never names a source), collects evidence with every gate
//! enforced, derives whatever models sufficiency allows, and
//! constructs with the rest receipted as defaults.
//!
//! No step is skippable and no step is hand-driven: this is the
//! function the app's chat box calls.

use super::evidence::{self, EvidenceStore};
use super::vision::Rgb;

/// True when the prompt is a picture request: a human subject doing
/// something poseable. Everything else stays on the code path.
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
    human
        && spec
            .actions
            .iter()
            .any(|a| poseable.contains(&a.atype.as_str()))
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
/// banked at ingestion). Requires a Sufficient collection — fewer
/// than preferred examples is not a researched palette.
pub fn derive_skin_model(store: &EvidenceStore, req_id: &str) -> Option<(Rgb, String, usize)> {
    if !matches!(
        store.sufficiency(req_id),
        Ok(evidence::Sufficiency::Sufficient { .. })
    ) {
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
    Some((
        Rgb::new(rs[mid], gs[mid], bs[mid]),
        format!("researched: {} photo tones", req.examples.len()),
        req.examples.len(),
    ))
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
    // Photo-first: per-request training, narrow scope. Try to composite
    // real people over real places (grounded pixels, never noise) before
    // spending budget on statistical evidence + procedural fallback.
    // Only what this prompt needs: subject + pose + backdrop.
    match super::imagine::imagine(prompt, 640, 800).await {
        Ok((img, ilog)) => {
            let image_path = out_dir.join("picture.bmp");
            img.save_bmp(&image_path)
                .map_err(|e| format!("cannot save picture: {}", e))?;
            let receipt_path = out_dir.join("picture.json");
            let mut reply = vec![
                "I see a picture request — researching photos for this prompt now.".to_string(),
            ];
            for l in &ilog {
                reply.push(format!("research: {}", l));
            }
            reply.push("path: photographic montage (real people, real place)".to_string());
            reply.push(format!("Done — picture at {}", image_path.display()));
            let body = serde_json::json!({
                "prompt": prompt,
                "path": "photographic-montage",
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
            // Montage refused — keep its trail for the receipt, then fall
            // through to evidence + procedural maquette (honestly labeled).
            let mut reply = vec![format!(
                "Photo montage refused ({} step(s)) — falling back to evidence + construction.",
                ilog.len()
            )];
            for l in &ilog {
                reply.push(format!("montage: {}", l));
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
            if src == "openimages" {
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

    // Researched skin applies when (and only when) a subject
    // collection went Sufficient; everything else stays defaulted
    // with the receipt saying so.
    let mut skin: Option<(Rgb, String)> = None;
    for r in &store.requirements {
        if r.category == super::evidence::Category::Subject
            && let Some((c, src, _)) = derive_skin_model(&store, &r.id)
        {
            skin = Some((c, src));
            reply.push(format!("{}: skin palette researched", r.id));
            break;
        }
    }
    let creation = super::studio::create_image_with(prompt, skin)
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
    fn subjects_come_from_openimages() {
        assert_eq!(source_for(super::evidence::Category::Subject), "openimages");
        assert_eq!(source_for(super::evidence::Category::Place), "commons");
        assert_eq!(source_for(super::evidence::Category::Material), "commons");
    }

    #[test]
    fn skin_needs_sufficient_collection() {
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
        assert!(derive_skin_model(&store, "subject:man").is_none());
    }
}
