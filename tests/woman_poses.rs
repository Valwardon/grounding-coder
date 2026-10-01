//! Woman-poses integration suite (female oracle morph).
//!
//! Offline / deterministic: measured rig from the vendored female
//! morph, FK agreement by construction, three posed variants, and a
//! committed standing-maquette sample with receipt.

use grounding_coder::engine::body_oracle::load_oracle_body;
use grounding_coder::engine::skeleton::{Joint, Pose, forward_kinematics};
use grounding_coder::engine::studio::create_image;
use grounding_coder::engine::vision::{Image, Rgb};

fn oracle_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/oracle")
}

fn female_body() -> grounding_coder::engine::body_oracle::OracleBody {
    load_oracle_body(&oracle_dir(), Some("female-young.target")).expect("female morph loads")
}

fn stature_of(body: &grounding_coder::engine::body_oracle::OracleBody) -> f64 {
    let m = grounding_coder::engine::body_oracle::measure_body(body);
    m.stature_m
}

#[test]
fn rig_reproduces_measured_positions() {
    use grounding_coder::engine::body_oracle::{rig_for, rig_to_proportions};
    let body = female_body();
    let rig = rig_for(&body).expect("vendored rig fits vendored mesh");
    // Every joint has a measured rest position from a real joint
    // group — no canon fallbacks anywhere in this rig.
    assert_eq!(rig.joints.len(), Joint::all().len());
    for j in Joint::all() {
        assert!(rig.joints.contains_key(&j), "rig missing {:?}", j);
    }
    // FK rest reproduces the measured positions exactly.
    let props = rig_to_proportions(&rig, stature_of(&body));
    let rest = forward_kinematics(&props, &Pose::rest()).expect("rest validates");
    for (j, want) in &rig.joints {
        let p = rest.get(j).unwrap_or_else(|| panic!("FK missing {:?}", j));
        let d =
            ((p.x - want[0]).powi(2) + (p.y - want[1]).powi(2) + (p.z - want[2]).powi(2)).sqrt();
        assert!(
            d < 1e-9,
            "FK rest must match measured {:?}: got {:?} want {:?} (d={})",
            j,
            (p.x, p.y, p.z),
            want,
            d
        );
    }
}

fn check_common(
    posed: &[[f64; 3]],
    fk: &std::collections::HashMap<Joint, grounding_coder::engine::skeleton::V3>,
    rest_min: f64,
    label: &str,
) {
    assert!(!posed.is_empty(), "{}: posed verts empty", label);
    for (i, v) in posed.iter().enumerate() {
        assert!(
            v[0].is_finite() && v[1].is_finite() && v[2].is_finite(),
            "{}: vert {} not finite: {:?}",
            label,
            i,
            v
        );
    }
    let min_y = posed.iter().map(|v| v[1]).fold(f64::INFINITY, f64::min);
    // Feet stay planted in every named pose (none lifts a leg):
    // mesh minimum tracks the rest minimum. Joints are not surface
    // points, so absolute floors would compare the wrong things.
    assert!(
        (min_y - rest_min).abs() < 0.02,
        "{}: mesh min-y {} drifted from rest {}",
        label,
        min_y,
        rest_min
    );
    let head_y = fk[&Joint::Head].y;
    assert!(
        head_y > fk[&Joint::ShoulderL].y,
        "{}: head {} must clear left shoulder {}",
        label,
        head_y,
        fk[&Joint::ShoulderL].y
    );
    assert!(
        head_y > fk[&Joint::ShoulderR].y,
        "{}: head {} must clear right shoulder {}",
        label,
        head_y,
        fk[&Joint::ShoulderR].y
    );
}

#[test]
fn same_woman_three_poses() {
    use grounding_coder::engine::body_oracle::{measured_proportions, pose_body, rig_for};
    let body = female_body();
    let props = measured_proportions(&body).expect("female rig measures");
    let rig = rig_for(&body).expect("female rig loads");

    // Standing.
    let pose = Pose::standing();
    let posed = pose_body(&body, &rig, &props, &pose).expect("standing poses");
    let fk = forward_kinematics(&props, &pose).expect("standing FK");
    let rest_min = posed.iter().map(|v| v[1]).fold(f64::INFINITY, f64::min);
    check_common(&posed, &fk, rest_min, "standing");
    assert!(
        fk[&Joint::FootL].y < 0.15 && fk[&Joint::FootR].y < 0.15,
        "standing feet near ground: L={} R={}",
        fk[&Joint::FootL].y,
        fk[&Joint::FootR].y
    );

    // Wave (right): right hand clears the head.
    let pose = Pose::wave(true);
    let posed = pose_body(&body, &rig, &props, &pose).expect("wave poses");
    let fk = forward_kinematics(&props, &pose).expect("wave FK");
    check_common(&posed, &fk, rest_min, "wave");
    assert!(
        fk[&Joint::HandR].y > fk[&Joint::Head].y,
        "wave: right hand {} must clear head {}",
        fk[&Joint::HandR].y,
        fk[&Joint::Head].y
    );

    // Salute (right): hand reaches toward the head.
    let pose = Pose::salute(true);
    let posed = pose_body(&body, &rig, &props, &pose).expect("salute poses");
    let fk = forward_kinematics(&props, &pose).expect("salute FK");
    check_common(&posed, &fk, rest_min, "salute");
    let (hand, head) = (fk[&Joint::HandR], fk[&Joint::Head]);
    let d =
        ((hand.x - head.x).powi(2) + (hand.y - head.y).powi(2) + (hand.z - head.z).powi(2)).sqrt();
    assert!(d < 0.5, "salute: hand-to-head distance {} must be < 0.5", d);
}

#[test]
fn woman_standing_renders_and_commits() {
    let c = create_image("a woman standing").expect("woman standing must create");
    assert_eq!((c.image.width, c.image.height), (640, 480));
    let stem = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("samples/studio-woman-standing-640x480");
    let bmp = stem.with_extension("bmp");
    let json = stem.with_extension("json");
    c.image.save_bmp(&bmp).expect("woman standing must save");
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
    let back = Image::load_bmp(&bmp).expect("woman standing must read back");
    assert_eq!((back.width, back.height), (c.image.width, c.image.height));
    let skin = c.image.count_near(Rgb::new(200, 150, 115), 3000);
    assert!(skin > 500, "figure must read, got {}", skin);
}
