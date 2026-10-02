//! Studio person suite: the end product, receipted.
//!
//! `create_image` on the walkthrough prompt must succeed with a
//! labeled maquette (every requirement defaulted, none researched),
//! strict mode must refuse listing the missing, and non-humans must
//! refuse outright. The committed BMP is the best construction with
//! current knowledge — the receipt, not the pixels, says so.
use grounding_coder::engine::studio::{CreationError, create_image, create_image_strict};
use grounding_coder::engine::vision::Image;

#[test]
fn walkthrough_creates_receipted_maquette() {
    let c = create_image("A man saluting, waving an American flag, wearing a straw hat.")
        .expect("walkthrough must create");
    // Real pixels, studio size.
    assert_eq!((c.image.width, c.image.height), (640, 480));
    assert!(c.image.mean_brightness() > 0.2 && c.image.mean_brightness() < 0.9);
    // Skin pixels exist (the figure reads through the finish).
    let skin = c.image.count_near(
        grounding_coder::engine::vision::Rgb::new(200, 150, 115),
        3000,
    );
    assert!(skin > 500, "figure must read, got {}", skin);
    // Composition audit from the renderer's own pixel counts.
    let total: u64 = c.pixel_counts.iter().map(|(_, n)| n).sum();
    let bg: u64 = c
        .pixel_counts
        .iter()
        .filter(|(m, _)| *m == "sky" || *m == "sweep")
        .map(|(_, n)| *n)
        .sum();
    let fill = 1.0 - bg as f64 / total.max(1) as f64;
    assert!(fill > 0.12, "tableau must fill the frame: {:.2}", fill);
    // A face, not a ball: eyes, mouth, hair, and garment all render
    // in their own materials (thresholds ~50% under live counts).
    use grounding_coder::engine::vision::Rgb;
    let eyes = c.image.count_near(Rgb::new(22, 13, 9), 2000);
    assert!(eyes > 50, "eyes must read, got {}", eyes);
    let mouth = c.image.count_near(Rgb::new(88, 28, 22), 2000);
    assert!(mouth > 80, "mouth must read, got {}", mouth);
    let hair = c.image.count_near(Rgb::new(48, 30, 17), 2000);
    assert!(hair > 200, "hair must read, got {}", hair);
    let shorts = c.image.count_near(Rgb::new(38, 38, 44), 2000);
    assert!(shorts > 200, "garment must read, got {}", shorts);
    // Nostrils render on both sides of the nose (symmetric dots).
    let (mut nl, mut nr) = (0u32, 0u32);
    for y in 0..c.image.height {
        for x in 0..c.image.width {
            if let Some(p) = c.image.get(x, y) {
                let d = (p.r as i32 - 28).abs() + (p.g as i32 - 16).abs() + (p.b as i32 - 13).abs();
                if d < 60 {
                    if x < c.image.width / 2 {
                        nl += 1;
                    } else {
                        nr += 1;
                    }
                }
            }
        }
    }
    assert!(
        nl > 0 && nr > 0,
        "nostrils must read both sides: {} {}",
        nl,
        nr
    );
    // Receipt: subject rides the researched measured rig; everything
    // else is still defaulted.
    assert!(!c.receipt.is_empty());
    let subjects: Vec<_> = c
        .receipt
        .iter()
        .filter(|r| r.requirement.starts_with("subject:"))
        .collect();
    assert!(
        !subjects.is_empty(),
        "subject status missing: {:?}",
        c.receipt.iter().map(|r| &r.requirement).collect::<Vec<_>>()
    );
    for s in &subjects {
        assert!(
            s.researched,
            "subject must be researched (measured rig): {:?}",
            s.requirement
        );
        assert!(
            s.note.contains("proportions researched (hm08 rig)"),
            "subject note must cite measured rig, got {:?}",
            s.note
        );
    }
    for r in c
        .receipt
        .iter()
        .filter(|r| !r.requirement.starts_with("subject:") && !r.requirement.starts_with("open:"))
    {
        assert!(
            !r.researched,
            "only subject is researched, got {:?} researched",
            r.requirement
        );
    }
}

#[test]
fn strict_refuses_listing_missing() {
    match create_image_strict("A man saluting, waving an American flag.") {
        Err(CreationError::MissingEvidence(ids)) => {
            assert!(!ids.is_empty());
            assert!(ids.iter().any(|i| i.contains("subject")));
        }
        other => panic!(
            "strict must refuse with evidence list, got {:?}",
            other.is_ok()
        ),
    }
}

#[test]
fn non_humans_refuse() {
    assert!(matches!(
        create_image("A tower out of glass."),
        Err(CreationError::UnsupportedPrompt(_))
    ));
}

#[test]
fn humans_without_actions_refuse_pose() {
    assert!(matches!(
        create_image("A man."),
        Err(CreationError::MissingPose(_))
    ));
}

#[test]
fn studio_person_commits_sample() {
    let c = create_image("A man saluting, waving an American flag, wearing a straw hat.")
        .expect("walkthrough must create");
    let stem = format!("samples/studio-person-{}x{}", c.image.width, c.image.height);
    let bmp = std::path::PathBuf::from(format!("{}.bmp", stem));
    let json = std::path::PathBuf::from(format!("{}.json", stem));
    c.image.save_bmp(&bmp).expect("person must save");
    let body = serde_json::json!({
        "kind": "constructed-studio-maquette, not a photograph",
        "prompt": c.prompt,
        "requirements": c.receipt.iter().map(|r| {
            serde_json::json!({"requirement": r.requirement, "researched": r.researched, "note": r.note})
        }).collect::<Vec<_>>(),
    });
    std::fs::write(&json, serde_json::to_string_pretty(&body).unwrap()).expect("receipt must save");
    let bytes = std::fs::read(&bmp).unwrap();
    assert_eq!(&bytes[0..2], b"BM");
    let back = Image::load_bmp(&bmp).expect("person must read back");
    assert_eq!((back.width, back.height), (c.image.width, c.image.height));
}
