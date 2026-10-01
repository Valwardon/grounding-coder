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
