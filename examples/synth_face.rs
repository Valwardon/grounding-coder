//! Synthesize a novel face from banked Open Images plates:
//! `cargo run --example synth_face -- <bank-dir> <out-bmp> [seed]`
//!
//! The bank dir holds `plate-*.bmp` plus the `provenance.json` manifest
//! written by `gc plate --source openimages` (fractional bboxes). This
//! example applies the sane-box filter (no full-frame boxes, no
//! edge-cropped boxes, no name-titled plates — same rules as the
//! imagine path), aligns the survivors, and collapses them to a
//! per-pixel median with measured detail. Every output pixel is
//! statistics; no donor pixel is copied. A `<out>.json` receipt lands
//! beside the photo.
use grounding_coder::engine::{imagine, synth, vision::Image};

const USAGE: &str = "usage: synth_face <bank-dir> <out-bmp> [seed]";

fn main() {
    let mut args = std::env::args().skip(1);
    let bank = args.next().expect(USAGE);
    let out = args.next().expect(USAGE);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0xFACE);

    let bank_path = std::path::Path::new(&bank);
    let manifest =
        std::fs::read_to_string(bank_path.join("provenance.json")).expect("read manifest");
    let entries: Vec<serde_json::Value> = serde_json::from_str(&manifest).expect("parse manifest");

    let mut plates = Vec::new();
    let mut boxes = Vec::new();
    let mut kept: Vec<String> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
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
                skipped.push(format!("{}: no bbox", file));
                continue;
            }
        };
        let area = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
        if area > 0.9 {
            skipped.push(format!("{}: full-frame box ({:.3})", file, area));
            continue;
        }
        let worst = x0.min(y0).min(1.0 - x1);
        if worst < 0.03 {
            skipped.push(format!("{}: edge-cropped ({:.3} margin)", file, worst));
            continue;
        }
        if imagine::looks_like_person_name(author) {
            skipped.push(format!("{}: name-titled ({:?})", file, author));
            continue;
        }
        match Image::load_bmp(&bank_path.join(file)) {
            Ok(img) => {
                plates.push(img);
                boxes.push((x0, y0, x1, y1));
                kept.push(file.to_string());
            }
            Err(e) => skipped.push(format!("{}: unreadable ({})", file, e)),
        }
    }
    println!(
        "synth: {} candidate(s), {} skipped",
        kept.len(),
        skipped.len()
    );
    for s in &skipped {
        println!("synth: skipped {}", s);
    }
    match synth::synthesize(&plates, &boxes, seed) {
        Ok((img, log)) => {
            for op in &log.ops {
                println!("synth: {}", op);
            }
            println!(
                "synth: sharpness {:.4}, min novelty {:.4}",
                log.sharpness, log.min_novelty
            );
            img.save_bmp(std::path::Path::new(&out))
                .expect("save photo");
            let receipt = serde_json::json!({
                "bank": bank,
                "seed": seed,
                "donors": kept,
                "donor_count": log.donors.len(),
                "skipped": skipped,
                "sharpness": log.sharpness,
                "min_novelty": log.min_novelty,
                "ops": log.ops,
                "size": [img.width, img.height],
            });
            let json_path = format!("{}.json", out);
            std::fs::write(&json_path, serde_json::to_string_pretty(&receipt).unwrap())
                .expect("save receipt");
            println!("synth: wrote {} + {}", out, json_path);
        }
        Err(e) => {
            eprintln!("SYNTH REFUSED: {}", e);
            std::process::exit(2);
        }
    }
}
