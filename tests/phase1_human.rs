//! Phase 1 human suite: articulated skeleton, honest measurements.
//!
//! Tests the spec's Phase 1 order — man/woman/standing/sitting/
//! walking, then raise/wave/salute/point — as joint math, not
//! pictures. Poses validate against ROM; impossible poses refuse;
//! measurements aggregate with a 20-example gate; the only render is
//! an annotated measurement diagram (instrumentation, receipted).
use grounding_coder::engine::body_measure::{self, PersonBox};
use grounding_coder::engine::skeleton::{
    BodyProportions, Joint, JointAngles, Pose, V3, forward_kinematics, render_diagram,
};
use grounding_coder::engine::vision::{Image, Rgb};
use std::collections::HashMap;

fn male() -> BodyProportions {
    BodyProportions::adult_male()
}

/// Deterministic synthetic plate: neutral backdrop, one skin block.
fn synthetic_plate() -> Image {
    let mut img = Image::blank(100, 150, Rgb::new(40, 60, 90));
    img.draw_rect(30, 10, 40, 30, Rgb::new(200, 150, 115));
    img
}

#[test]
fn tree_matches_spec() {
    use Joint::*;
    assert_eq!(Spine.parent(), Some(Pelvis));
    assert_eq!(Chest.parent(), Some(Spine));
    assert_eq!(Neck.parent(), Some(Chest));
    assert_eq!(Head.parent(), Some(Neck));
    assert_eq!(ElbowL.parent(), Some(ShoulderL));
    assert_eq!(WristL.parent(), Some(ElbowL));
    assert_eq!(HandL.parent(), Some(WristL));
    assert_eq!(ElbowR.parent(), Some(ShoulderR));
    assert_eq!(KneeL.parent(), Some(HipL));
    assert_eq!(AnkleL.parent(), Some(KneeL));
    assert_eq!(FootL.parent(), Some(AnkleL));
    assert_eq!(KneeR.parent(), Some(HipR));
    assert_eq!(Pelvis.parent(), None);
    assert_eq!(Joint::all().len(), 21);
}

#[test]
fn standing_is_symmetric_and_full_height() {
    let pos = forward_kinematics(&male(), &Pose::standing()).unwrap();
    // Left/right mirror: x flips, y/z match.
    for (l, r) in [
        (Joint::ShoulderL, Joint::ShoulderR),
        (Joint::ElbowL, Joint::ElbowR),
        (Joint::HandL, Joint::HandR),
        (Joint::HipL, Joint::HipR),
        (Joint::KneeL, Joint::KneeR),
        (Joint::FootL, Joint::FootR),
    ] {
        assert!((pos[&l].x + pos[&r].x).abs() < 1e-9, "{:?}/{:?}", l, r);
        assert!((pos[&l].y - pos[&r].y).abs() < 1e-9, "{:?}/{:?}", l, r);
        assert!((pos[&l].z - pos[&r].z).abs() < 1e-9, "{:?}/{:?}", l, r);
    }
    // Feet on the ground, head within 2% of stature.
    assert!(
        pos[&Joint::FootL].y.abs() < 0.03,
        "foot {:?}",
        pos[&Joint::FootL]
    );
    let head_top = pos[&Joint::Head].y + male().stature_m / 7.5 / 2.0;
    assert!(
        (head_top - male().stature_m).abs() < 0.02 * male().stature_m,
        "head top {}",
        head_top
    );
    // Fingertips land mid-thigh: hand below hip, above knee.
    assert!(pos[&Joint::HandL].y < pos[&Joint::HipL].y);
    assert!(pos[&Joint::HandL].y > pos[&Joint::KneeL].y);
}

#[test]
fn man_and_woman_proportions_differ() {
    let m = forward_kinematics(&BodyProportions::adult_male(), &Pose::standing()).unwrap();
    let f = forward_kinematics(&BodyProportions::adult_female(), &Pose::standing()).unwrap();
    let top = |pos: &HashMap<Joint, V3>, s: f64| pos[&Joint::Head].y + s / 7.5 / 2.0;
    assert!(top(&f, 1.62) < top(&m, 1.75));
    let hip_w = |pos: &HashMap<Joint, V3>| (pos[&Joint::HipR].x - pos[&Joint::HipL].x).abs();
    assert!(hip_w(&f) > hip_w(&m), "female pelvis reads wider");
}

#[test]
fn sitting_makes_horizontal_thighs() {
    let pos = forward_kinematics(&male(), &Pose::sitting()).unwrap();
    // Thigh horizontal: knee at hip height, forward of the hip.
    assert!(
        (pos[&Joint::KneeL].y - pos[&Joint::HipL].y).abs() < 0.05,
        "knee {:?}",
        pos[&Joint::KneeL]
    );
    assert!(pos[&Joint::KneeL].z > pos[&Joint::HipL].z + 0.2);
    // Shin vertical: ankle below the knee.
    assert!(pos[&Joint::AnkleL].y < pos[&Joint::KneeL].y - 0.2);
}

#[test]
fn walking_swings_limbs_in_opposition() {
    // Peak swing phases: limbs oppose left/right and across the cycle.
    let a = Pose::walking(std::f64::consts::FRAC_PI_2);
    let b = Pose::walking(3.0 * std::f64::consts::FRAC_PI_2);
    assert!((a.get(Joint::HipL).flex - 40.0).abs() < 1e-9);
    assert!((b.get(Joint::HipL).flex - 0.0).abs() < 1e-9);
    assert!((a.get(Joint::HipL).flex - a.get(Joint::HipR).flex).abs() > 30.0);
    // Both phases validate and move the feet differently.
    let pa = forward_kinematics(&male(), &a).unwrap();
    let pb = forward_kinematics(&male(), &b).unwrap();
    assert!(
        (pa[&Joint::FootL].z - pb[&Joint::FootL].z).abs() > 0.05,
        "feet must travel"
    );
}

#[test]
fn salute_reaches_the_head() {
    let pos = forward_kinematics(&male(), &Pose::salute(true)).unwrap();
    let hand = pos[&Joint::HandR];
    let head = pos[&Joint::Head];
    let d = ((hand.x - head.x).powi(2) + (hand.y - head.y).powi(2)).sqrt();
    assert!(d < 0.45, "hand must reach the head, got {:.2}m", d);
}

#[test]
fn wave_clears_the_head() {
    let pos = forward_kinematics(&male(), &Pose::wave(true)).unwrap();
    assert!(
        pos[&Joint::HandR].y > pos[&Joint::Head].y,
        "hand {:?} head {:?}",
        pos[&Joint::HandR],
        pos[&Joint::Head]
    );
}

#[test]
fn point_reaches_forward() {
    let pos = forward_kinematics(&male(), &Pose::point(true)).unwrap();
    assert!(
        pos[&Joint::HandR].z > pos[&Joint::ShoulderR].z + 0.35,
        "hand {:?}",
        pos[&Joint::HandR]
    );
}

#[test]
fn raise_hand_is_highest_point() {
    let pos = forward_kinematics(&male(), &Pose::raise_hand(true)).unwrap();
    let top = pos[&Joint::Head].y + male().stature_m / 7.5 / 2.0;
    assert!(pos[&Joint::HandR].y > top, "hand {:?}", pos[&Joint::HandR]);
}

#[test]
fn impossible_pose_refuses() {
    let mut p = Pose::rest();
    p.set(Joint::ElbowR, JointAngles::flex(200.0));
    let err = forward_kinematics(&male(), &p).unwrap_err();
    assert!(err.contains("outside"), "got {}", err);
    let mut p = Pose::rest();
    p.set(Joint::KneeL, JointAngles::flex(-10.0));
    assert!(forward_kinematics(&male(), &p).is_err());
}

#[test]
fn measurement_gate_holds_offline() {
    let pb = PersonBox {
        x0: 0,
        y0: 0,
        x1: 100,
        y1: 150,
    };
    // Three plates: numbers, never a model.
    let few: Vec<_> = (0..3)
        .map(|_| body_measure::measure_plate(&synthetic_plate(), &pb))
        .collect();
    let model = body_measure::aggregate(&few);
    assert!(body_measure::verified(&model).is_err());
    assert!(!body_measure::fine_joints_available());
    assert_eq!(body_measure::supported_detail(), "coarse-bands");
}

#[test]
fn diagram_commits_measurement_photo() {
    // Instrumentation, receipted: a standing measurement diagram.
    let pos = forward_kinematics(&male(), &Pose::standing()).unwrap();
    let img = render_diagram(&pos, 320, 240);
    let white = img.count_near(Rgb::new(220, 225, 235), 2000);
    assert!(white > 500, "bones must draw, got {}", white);
    let red = img.count_near(Rgb::new(255, 90, 90), 2000);
    assert!(red > 500, "joints must draw, got {}", red);
    let bmp = std::path::Path::new("samples/skeleton-measurement-standing-320x240.bmp");
    let json = std::path::Path::new("samples/skeleton-measurement-standing-320x240.json");
    img.save_bmp(bmp).expect("diagram must save");
    let body = serde_json::json!({
        "kind": "measurement-diagram, not a portrait",
        "pose": "standing",
        "stature_m": male().stature_m,
        "joints": Joint::all().len(),
        "rom": "all validated",
    });
    std::fs::write(json, serde_json::to_string_pretty(&body).unwrap()).expect("receipt must save");
    let bytes = std::fs::read(bmp).unwrap();
    assert_eq!(&bytes[0..2], b"BM");
    let back = Image::load_bmp(bmp).expect("diagram must read back");
    assert_eq!((back.width, back.height), (320, 240));
}
