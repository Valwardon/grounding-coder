//! At-run training demo: banked plates → aligned faces → trained
//! denoiser → seeded sample. `cargo run --example diffuse_train --
//! <bank-dir> <out-bmp> [seed]`
//!
//! The whole doctrine in one command: research scope is the bank,
//! training data is face regions aligned by `synth` (downscaled to
//! keep the toy trainer honest about its scale), the adapter is
//! hash-pinned, and sampling is fixed-seed DDIM. A `<out>.json`
//! receipt lands beside the photo with dataset hashes, training
//! losses, adapter hash, and seed — everything needed to reproduce
//! the exact bytes. Append `--epochs N` to override the training
//! budget (receipted, never silent).
use grounding_coder::engine::{diffuse, imagine, synth, vision::Image};

const USAGE: &str = "usage: diffuse_train <bank-dir> <out-bmp> [seed] [--epochs N]";
const TRAIN_SIZE: u32 = 32;
const EPOCHS: u32 = 15;
const STEPS: usize = 8;

fn main() {
    let mut args = std::env::args().skip(1);
    let bank = args.next().expect(USAGE);
    let out = args.next().expect(USAGE);
    let mut seed: u64 = 0xD1FF;
    let mut epochs = EPOCHS;
    let rest: Vec<String> = args.collect();
    let mut k = 0;
    while k < rest.len() {
        if rest[k] == "--epochs" {
            epochs = rest
                .get(k + 1)
                .and_then(|v| v.parse().ok())
                .unwrap_or(EPOCHS);
            k += 2;
            continue;
        }
        let a = &rest[k];
        if let Some(v) = a
            .strip_prefix("0x")
            .and_then(|h| u64::from_str_radix(h, 16).ok())
        {
            seed = v;
        } else if let Ok(v) = a.parse::<u64>() {
            seed = v;
        }
        k += 1;
    }

    let bank_path = std::path::Path::new(&bank);
    let manifest =
        std::fs::read_to_string(bank_path.join("provenance.json")).expect("read manifest");
    let entries: Vec<serde_json::Value> = serde_json::from_str(&manifest).expect("parse manifest");

    // Same sane-box filter as the synth runner: no full-frame boxes,
    // no edge crops, no named subjects.
    let mut plates = Vec::new();
    let mut boxes = Vec::new();
    let mut kept: Vec<String> = Vec::new();
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
        match Image::load_bmp(&bank_path.join(file)) {
            Ok(img) => {
                plates.push(img);
                boxes.push((x0, y0, x1, y1));
                kept.push(file.to_string());
            }
            Err(_) => skipped += 1,
        }
    }
    println!("diffuse: {} candidate(s), {} skipped", kept.len(), skipped);

    // Training data: aligned face regions at toy scale. Small on
    // purpose — the denoiser is 67 parameters and the receipt states
    // the resolution it actually trained at.
    let mut data = Vec::new();
    let mut donors = Vec::new();
    for (k, (plate, bbox)) in plates.iter().zip(boxes.iter()).enumerate() {
        match synth::extract_face_region(plate, *bbox, k) {
            Ok(hit) => {
                data.push(synth::align(&hit.image).resize_smooth(TRAIN_SIZE, TRAIN_SIZE));
                donors.push(kept[k].clone());
            }
            Err(e) => println!("diffuse: no face {}: {}", kept[k], e),
        }
    }
    println!(
        "diffuse: {} training faces at {}x{}",
        data.len(),
        TRAIN_SIZE,
        TRAIN_SIZE
    );

    let dataset_hashes: Vec<String> = data.iter().map(diffuse::plate_hash).collect();
    let (den, tlog) = diffuse::train(&data, seed, epochs, STEPS).unwrap_or_else(|e| {
        eprintln!("TRAIN REFUSED: {}", e);
        std::process::exit(2);
    });
    println!(
        "diffuse: loss {:.4} -> {:.4} over {} epochs, adapter {}",
        tlog.first_loss,
        tlog.final_loss,
        tlog.epochs,
        &tlog.adapter_hash[..16]
    );

    let img = diffuse::sample(&den, seed, TRAIN_SIZE, TRAIN_SIZE, STEPS);
    img.save_bmp(std::path::Path::new(&out))
        .expect("save photo");
    let receipt = serde_json::json!({
        "bank": bank,
        "seed": seed,
        "train_size": [TRAIN_SIZE, TRAIN_SIZE],
        "epochs": tlog.epochs,
        "steps": tlog.steps,
        "donors": donors,
        "donor_hashes": dataset_hashes,
        "first_loss": tlog.first_loss,
        "final_loss": tlog.final_loss,
        "adapter_hash": tlog.adapter_hash,
        "size": [img.width, img.height],
    });
    let json_path = format!("{}.json", out);
    std::fs::write(&json_path, serde_json::to_string_pretty(&receipt).unwrap())
        .expect("save receipt");
    println!("diffuse: wrote {} + {}", out, json_path);
}
