//! Phase 3 objects suite: measure → model → construct, honestly.
//!
//! Synthetic striped flags stand in for researched examples (same
//! classical operators run on both). The round trip is the proof:
//! a flag constructed from the model re-measures to its inputs.
use grounding_coder::engine::object_model::{
    PoleSide, construct_flag, derive_flag_model, derive_object_model, measure_flag,
};
use grounding_coder::engine::vision::{Image, Rgb};

fn synthetic_flag(stripes: u32, width: u32, height: u32) -> Image {
    let mut img = Image::blank(width, height, Rgb::new(0, 0, 0));
    for y in 0..height {
        let i = (y * stripes / height) % 2;
        let c = if i == 0 {
            Rgb::new(200, 30, 30)
        } else {
            Rgb::new(240, 240, 240)
        };
        for x in 0..width {
            img.set(x, y, c);
        }
    }
    for y in 0..height {
        img.set(0, y, Rgb::new(20, 20, 20));
        img.set(1, y, Rgb::new(20, 20, 20));
    }
    img
}

fn twenty_measures() -> Vec<grounding_coder::engine::object_model::FlagMeasure> {
    (0..20)
        .map(|i| measure_flag(&synthetic_flag(6, 120 + i, 80)))
        .collect()
}

#[test]
fn model_derives_statistics_not_anecdotes() {
    let model = derive_flag_model(&twenty_measures()).expect("20 examples");
    assert_eq!(model.n, 20);
    assert_eq!(model.stripes, 6);
    assert!((model.stripe_agreement - 1.0).abs() < 1e-9);
    assert_eq!(model.pole, Some(PoleSide::Left));
    assert!(model.aspect_min <= model.aspect_mean && model.aspect_mean <= model.aspect_max);
    // Palette: alternating red/white medians per stripe index.
    assert!(model.palette.len() >= 6);
    assert!(
        model.palette[0].0 > 150,
        "red first: {:?}",
        model.palette[0]
    );
}

#[test]
fn construction_round_trips() {
    let model = derive_flag_model(&twenty_measures()).expect("model");
    let built = construct_flag(&model, 180);
    let re = measure_flag(&built);
    assert_eq!(re.stripes, 6, "constructed stripes must re-measure");
    assert!((re.aspect - built.width as f64 / built.height as f64).abs() < 1e-9);
    assert_eq!(re.pole, Some(PoleSide::Left));
}

#[test]
fn construction_commits_sample_photo() {
    let model = derive_flag_model(&twenty_measures()).expect("model");
    let img = construct_flag(&model, 320);
    let stem = format!("samples/flag-constructed-{}x{}", img.width, img.height);
    let bmp = std::path::PathBuf::from(format!("{}.bmp", stem));
    let json = std::path::PathBuf::from(format!("{}.json", stem));
    img.save_bmp(&bmp).expect("flag must save");
    let body = serde_json::json!({
        "kind": "constructed-object, not a photograph",
        "concept": "flag",
        "stripes": model.stripes,
        "agreement": model.stripe_agreement,
        "examples": model.n,
    });
    std::fs::write(&json, serde_json::to_string_pretty(&body).unwrap()).expect("receipt must save");
    let bytes = std::fs::read(&bmp).unwrap();
    assert_eq!(&bytes[0..2], b"BM");
    let back = Image::load_bmp(&bmp).expect("flag must read back");
    assert_eq!((back.width, back.height), (img.width, img.height));
}

#[test]
fn disagreement_shows_in_agreement() {
    let mut ms = twenty_measures();
    for _ in 0..10 {
        ms.push(measure_flag(&synthetic_flag(4, 120, 80)));
    }
    let model = derive_flag_model(&ms).expect("model");
    assert_eq!(model.stripes, 6);
    assert!(model.stripe_agreement < 1.0 && model.stripe_agreement > 0.5);
}

#[test]
fn hats_wait_on_segmentation() {
    // Part-structured objects refuse with the named gap — staged,
    // never silently boxed.
    let err = derive_object_model("hat", 30).unwrap_err();
    assert!(err.contains("segmentation"), "got {}", err);
}
