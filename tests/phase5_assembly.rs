//! Phase 5 assembly suite: relationships become placements.
//!
//! The full walkthrough scene assembles on the scaffold: straw hat
//! above the head, flag pole at the free hand, salute verified at
//! the head. Everything unplaceable stays open with its reason.
use grounding_coder::engine::assembly::assemble;
use grounding_coder::engine::scene_intent::parse_scene;
use grounding_coder::engine::skeleton::{
    BodyProportions, Joint, Pose, forward_kinematics, render_diagram,
};
use grounding_coder::engine::vision::{Image, Rgb};

fn full_scene() -> grounding_coder::engine::scene_intent::SceneSpec {
    parse_scene("A man saluting, waving an American flag, wearing a straw hat.")
}

#[test]
fn hat_anchors_above_head() {
    let spec = full_scene();
    let props = BodyProportions::adult_male();
    let pos = forward_kinematics(&props, &Pose::salute(true)).unwrap();
    let asm = assemble(&spec, &pos, &props);
    let hat = asm
        .attachments
        .iter()
        .find(|a| a.object_id.contains("hat"))
        .expect("hat attachment");
    assert_eq!(hat.joint, Joint::Head);
    assert!(
        hat.anchor.y > pos[&Joint::Head].y,
        "crown above head: {:?}",
        hat.anchor
    );
    assert!(hat.note.contains("worn by"));
}

#[test]
fn flag_goes_to_free_hand() {
    let spec = full_scene();
    let props = BodyProportions::adult_male();
    let pos = forward_kinematics(&props, &Pose::salute(true)).unwrap();
    let asm = assemble(&spec, &pos, &props);
    let flag = asm
        .attachments
        .iter()
        .find(|a| a.object_id == "american_flag")
        .expect("flag attachment");
    // Salute takes the right hand; the flag goes left.
    assert_eq!(flag.joint, Joint::HandL);
    assert_eq!(
        (flag.anchor.x, flag.anchor.y),
        (pos[&Joint::HandL].x, pos[&Joint::HandL].y)
    );
}

#[test]
fn flag_defaults_right_without_salute() {
    let spec = parse_scene("A man waving an American flag.");
    let props = BodyProportions::adult_male();
    let pos = forward_kinematics(&props, &Pose::wave(true)).unwrap();
    let asm = assemble(&spec, &pos, &props);
    let flag = asm
        .attachments
        .iter()
        .find(|a| a.object_id == "american_flag")
        .expect("flag attachment");
    assert_eq!(flag.joint, Joint::HandR);
}

#[test]
fn unworn_hat_stays_open() {
    // No wear verb, no wearer: the hat is honestly unplaced.
    let spec = parse_scene("A man saluting, with a hat made of straw.");
    let props = BodyProportions::adult_male();
    let pos = forward_kinematics(&props, &Pose::salute(true)).unwrap();
    let asm = assemble(&spec, &pos, &props);
    assert!(asm.attachments.iter().all(|a| !a.object_id.contains("hat")));
    assert!(asm.unresolved.iter().any(|u| u.contains("hat")));
}

#[test]
fn missed_salute_pose_reports() {
    // Standing pose cannot salute: the hand misses the head and the
    // assembly says so instead of blessing the placement.
    let spec = full_scene();
    let props = BodyProportions::adult_male();
    let pos = forward_kinematics(&props, &Pose::standing()).unwrap();
    let asm = assemble(&spec, &pos, &props);
    assert!(asm.unresolved.iter().any(|u| u.contains("misses")));
}

#[test]
fn assembly_commits_diagram() {
    let spec = full_scene();
    let props = BodyProportions::adult_male();
    let pos = forward_kinematics(&props, &Pose::salute(true)).unwrap();
    let asm = assemble(&spec, &pos, &props);
    assert!(
        asm.unresolved.is_empty(),
        "full scene places clean: {:?}",
        asm.unresolved
    );
    // Diagram the assembly: scaffold plus gold attachment markers.
    let mut img = render_diagram(&pos, 320, 240);
    let sx: f64 = 320.0 / 2.4;
    let sy: f64 = 240.0 / 2.2;
    let s = sx.min(sy);
    let cx: f64 = 320.0 / 2.0;
    let base: f64 = 240.0 - 12.0;
    for a in &asm.attachments {
        let x = (cx + a.anchor.x * s) as i32;
        let y = (base - a.anchor.y * s) as i32;
        img.draw_disc(x, y, 6, Rgb::new(255, 200, 40));
    }
    let stem = format!(
        "samples/assembly-salute-flag-hat-{}x{}",
        img.width, img.height
    );
    let bmp = std::path::PathBuf::from(format!("{}.bmp", stem));
    let json = std::path::PathBuf::from(format!("{}.json", stem));
    img.save_bmp(&bmp).expect("diagram must save");
    let body = serde_json::json!({
        "kind": "assembly-diagram, not a photograph",
        "attachments": asm.attachments.iter().map(|a| &a.object_id).collect::<Vec<_>>(),
    });
    std::fs::write(&json, serde_json::to_string_pretty(&body).unwrap()).expect("receipt must save");
    let bytes = std::fs::read(&bmp).unwrap();
    assert_eq!(&bytes[0..2], b"BM");
    let back = Image::load_bmp(&bmp).expect("diagram must read back");
    assert_eq!((back.width, back.height), (img.width, img.height));
}
