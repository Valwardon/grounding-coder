//! The chat picture pipeline: the bot's own hands, end to end.
//!
//! When chat receives a picture request (human subject + poseable
//! action), it comes HERE — not to code tasks. The pipeline
//! researches licensed photographs for the prompt and synthesizes
//! one novel figure from their median. The product is brand-new
//! pixels with donor provenance — never a donor copy, never a
//! montage, never a rendered mesh.
//!
//! No step is skippable and no step is hand-driven: this is the
//! function the app's chat box calls.

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

/// Run the whole picture loop for a chat prompt. `per_req` bounds
/// examples per requirement (currently unused — research depth is
/// fixed by the imagine rounds); `out_dir` receives the delivered
/// photograph and its receipt.
pub async fn picture_from_prompt(
    prompt: &str,
    out_dir: &std::path::Path,
    per_req: u32,
) -> Result<PictureOutcome, String> {
    let _ = per_req;
    std::fs::create_dir_all(out_dir).map_err(|e| format!("cannot create dir: {}", e))?;
    // Synthesis-only: research what this prompt needs, synthesize
    // one novel figure from the licensed donors' median.
    match super::imagine::imagine(prompt, 640, 800).await {
        Ok((photo, ilog)) => {
            let image_path = out_dir.join("picture.bmp");
            photo
                .image
                .save_bmp(&image_path)
                .map_err(|e| format!("cannot save picture: {}", e))?;
            let receipt_path = out_dir.join("picture.json");
            let mut reply = vec![
                "I see a picture request — researching licensed photos for this prompt now."
                    .to_string(),
            ];
            for l in &ilog {
                reply.push(format!("research: {}", l));
            }
            reply.push(
                "path: synthesized photograph (brand-new pixels from donor statistics)".to_string(),
            );
            reply.push(format!("Done — picture at {}", image_path.display()));
            let body = serde_json::json!({
                "prompt": prompt,
                "path": "synthesized-photograph",
                "donors": photo.donors,
                "donor_provenance": photo.donor_provenance,
                "sharpness": photo.sharpness,
                "min_novelty": photo.min_novelty,
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
            // Research refused — its trail is the receipt. Nothing is
            // fabricated to fill the gap.
            let mut reply = vec![format!(
                "No licensed photograph found ({} research step(s)) — refusing, not rendering.",
                ilog.len()
            )];
            for l in &ilog {
                reply.push(format!("research: {}", l));
            }
            Err(reply.join("\n"))
        }
    }
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
}
