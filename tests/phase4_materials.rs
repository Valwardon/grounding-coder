//! Phase 4 materials suite: appearance from many examples.
//!
//! Synthetic two-tone swatches stand in for researched material
//! photos (same histogram operators run on both). The round trip is
//! the proof: a swatch constructed from the model re-measures to
//! its palette and coverages.
use grounding_coder::engine::material::{
    ColorShare, construct_swatch, derive_material, sample_histogram,
};
use grounding_coder::engine::vision::{Image, Rgb};

fn two_tone(w: u32, h: u32) -> Image {
    split_tone(w, h, 2)
}

fn split_tone(w: u32, h: u32, parts: u32) -> Image {
    let mut img = Image::blank(w, h, Rgb::new(0, 0, 0));
    for y in 0..h {
        let c = if y < h / parts {
            Rgb::new(200, 30, 30)
        } else {
            Rgb::new(240, 240, 240)
        };
        for x in 0..w {
            img.set(x, y, c);
        }
    }
    img
}

fn twenty_samples() -> Vec<Vec<ColorShare>> {
    (0..20)
        .map(|i| sample_histogram(&two_tone(100 + i, 100), 0, 0, 100 + i, 100))
        .collect()
}

#[test]
fn straw_derives_like_any_material() {
    // The concept label rides along; derivation never branches on it.
    for concept in ["straw", "steel", "cloth"] {
        let model = derive_material(concept, &twenty_samples()).expect("20 samples");
        assert_eq!(model.concept, concept);
        assert_eq!(model.n, 20);
        assert_eq!(model.palette.len(), 2);
        assert!((model.palette[0].agreement - 1.0).abs() < 1e-9);
        assert!((model.palette[0].coverage_mean - 0.5).abs() < 0.02);
    }
}

#[test]
fn swatch_round_trips() {
    let model = derive_material("straw", &twenty_samples()).expect("model");
    let swatch = construct_swatch(&model, 200, 100);
    let re = sample_histogram(&swatch, 0, 0, 200, 100);
    assert_eq!(re.len(), 2);
    assert_eq!(re[0].0, model.palette[0].color);
    assert_eq!(re[1].0, model.palette[1].color);
    assert!((re[0].1 - model.palette[0].coverage_mean).abs() < 0.03);
}

#[test]
fn swatch_commits_sample_photo() {
    let model = derive_material("straw", &twenty_samples()).expect("model");
    let img = construct_swatch(&model, 320, 200);
    let stem = format!("samples/material-swatch-{}x{}", img.width, img.height);
    let bmp = std::path::PathBuf::from(format!("{}.bmp", stem));
    let json = std::path::PathBuf::from(format!("{}.json", stem));
    img.save_bmp(&bmp).expect("swatch must save");
    let body = serde_json::json!({
        "kind": "constructed-material, not a photograph",
        "concept": model.concept,
        "entries": model.palette.len(),
        "examples": model.n,
    });
    std::fs::write(&json, serde_json::to_string_pretty(&body).unwrap()).expect("receipt must save");
    let bytes = std::fs::read(&bmp).unwrap();
    assert_eq!(&bytes[0..2], b"BM");
    let back = Image::load_bmp(&bmp).expect("swatch must read back");
    assert_eq!((back.width, back.height), (img.width, img.height));
}

#[test]
fn disagreement_shows_in_coverage_range() {
    let mut samples = twenty_samples();
    for _ in 0..10 {
        samples.push(sample_histogram(&split_tone(100, 90, 3), 0, 0, 100, 90));
    }
    let model = derive_material("cloth", &samples).expect("model");
    assert!(model.palette[0].coverage_max - model.palette[0].coverage_min > 0.05);
}

#[test]
fn gate_and_empty_refuse() {
    let few: Vec<_> = (0..3)
        .map(|_| sample_histogram(&two_tone(50, 50), 0, 0, 50, 50))
        .collect();
    assert!(derive_material("steel", &few).is_err());
}
