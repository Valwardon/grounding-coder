//! Truth probe (dev tool): build FaceTruth from a bank of real
//! plates and print what actually got measured.
//! `cargo run --example truth_probe -- <bank-dir>`
//!
//! This is the verification the unit tests cannot give: synthetic
//! discs prove mechanics, but only real photographs prove the
//! extractors measure anatomy. Prints per-field measured values
//! with provenance plus the gap list (the seeker's work orders).
use grounding_coder::engine::{imagine, synth, truth, vision::Image};

const USAGE: &str = "usage: truth_probe <bank-dir>";

fn main() {
    let bank = std::env::args().nth(1).expect(USAGE);
    let bank_path = std::path::Path::new(&bank);
    let manifest =
        std::fs::read_to_string(bank_path.join("provenance.json")).expect("read manifest");
    let entries: Vec<serde_json::Value> = serde_json::from_str(&manifest).expect("parse manifest");

    let mut crops = Vec::new();
    let mut pairs = Vec::new();
    let mut eye_locked = 0u32;
    let mut skipped = 0u32;
    for e in &entries {
        let file = e.get("file").and_then(|v| v.as_str()).unwrap_or("");
        let author = e.get("author").and_then(|v| v.as_str()).unwrap_or("");
        let bbox = e.get("bbox").and_then(|v| v.as_array());
        let (x0, y0, x1, y1) = match bbox {
            Some(b) if b.len() == 4 => (
                b[0].as_f64().unwrap_or(0.0),
                b[1].as_f64().unwrap_or(0.0),
                b[2].as_f64().unwrap_or(1.0),
                b[3].as_f64().unwrap_or(1.0),
            ),
            _ => {
                skipped += 1;
                continue;
            }
        };
        let area = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
        if area > 0.9 {
            skipped += 1;
            continue;
        }
        let worst = x0.min(y0).min(1.0 - x1);
        if worst < 0.03 {
            skipped += 1;
            continue;
        }
        if imagine::looks_like_person_name(author) {
            skipped += 1;
            continue;
        }
        let img = match Image::load_bmp(&bank_path.join(file)) {
            Ok(img) => img,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        // Head square first (measured), so the pair (if any) lands in
        // head-square coordinates like the truth model expects.
        let (head, pair) = match synth::head_square(&img, (x0, y0, x1, y1)) {
            Some((sx0, sy0, side)) => {
                let h = img.crop(sx0, sy0, side, side);
                let pair = synth::eye_pair(&h);
                (h, pair)
            }
            None => continue,
        };
        if pair.is_some() {
            eye_locked += 1;
        }
        crops.push(head);
        pairs.push(pair);
    }
    println!(
        "truth: {} head crops ({} eye-locked), {} skipped",
        crops.len(),
        eye_locked,
        skipped
    );
    let face = truth::face_truth(&crops, &pairs);
    println!("truth: measured fields {}/5", face.measured_fields());
    match &face.skin {
        truth::Truth::Measured { value, provenance } => println!(
            "truth: skin median=({},{},{}) spread=({},{},{}) via {} over {} donors",
            value.median.r,
            value.median.g,
            value.median.b,
            value.spread.0,
            value.spread.1,
            value.spread.2,
            provenance.method,
            provenance.donors.len()
        ),
        truth::Truth::Insufficient { gap } => println!("truth: skin GAP: {}", gap),
    }
    match &face.proportions {
        truth::Truth::Measured { value, provenance } => println!(
            "truth: eye_spacing med={:.3} range=[{:.3},{:.3}], eyeline med={:.3} via {} over {} donors",
            value.eye_spacing.0,
            value.eye_spacing.1,
            value.eye_spacing.2,
            value.head_aspect.0,
            provenance.method,
            provenance.donors.len()
        ),
        truth::Truth::Insufficient { gap } => println!("truth: proportions GAP: {}", gap),
    }
    match &face.landmarks {
        truth::Truth::Measured { value, provenance } => println!(
            "truth: landmarks L=({:.3},{:.3}) R=({:.3},{:.3}) side={}px via {} donor {:?}",
            value.left_eye.0,
            value.left_eye.1,
            value.right_eye.0,
            value.right_eye.1,
            value.head_side_px,
            provenance.method,
            provenance.donors
        ),
        truth::Truth::Insufficient { gap } => println!("truth: landmarks GAP: {}", gap),
    }
    println!("truth: gaps ({}):", face.gaps().len());
    for g in face.gaps() {
        println!("truth:   - {}", g);
    }
}
