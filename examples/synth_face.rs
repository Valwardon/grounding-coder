//! Synthesize a novel face from banked Open Images plates:
//! `cargo run --example synth_face -- <bank-dir> <out-bmp> [seed] [--locked]`
//!
//! The bank dir holds `plate-*.bmp` plus the `provenance.json` manifest
//! written by `gc plate --source openimages` (fractional bboxes). This
//! example applies the sane-box filter (no full-frame boxes, no
//! edge-cropped boxes, no name-titled plates — same rules as the
//! imagine path), aligns the survivors, and collapses them to a
//! per-pixel median with measured detail. Every output pixel is
//! statistics; no donor pixel is copied. A `<out>.json` receipt lands
//! beside the photo.
//!
//! `--locked`: eye-locked synthesis only — donors without a measured
//! eye pair are excluded instead of averaged in (needs 8+ locks).
use grounding_coder::engine::{imagine, synth, vision::Image};

const USAGE: &str = "usage: synth_face <bank-dir> <out-bmp> [seed] [--locked]";

fn main() {
    let mut args = std::env::args().skip(1);
    let bank = args.next().expect(USAGE);
    let out = args.next().expect(USAGE);
    let mut seed: u64 = 0xFACE;
    let mut locked = false;
    let mut debug_eyes = false;
    for a in args {
        if a == "--locked" {
            locked = true;
        } else if a == "--debug-eyes" {
            debug_eyes = true;
        } else if let Some(v) = a
            .strip_prefix("0x")
            .and_then(|h| u64::from_str_radix(h, 16).ok())
        {
            seed = v;
        } else if let Ok(v) = a.parse::<u64>() {
            seed = v;
        }
    }

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
        if imagine::looks_like_person_name(author) {
            skipped.push(format!("{}: name-titled ({:?})", file, author));
            continue;
        }
        let img = match Image::load_bmp(&bank_path.join(file)) {
            Ok(img) => img,
            Err(e) => {
                skipped.push(format!("{}: unreadable ({})", file, e));
                continue;
            }
        };
        // Margin rule is montage framing policy; for synthesis what
        // matters is the FACE being whole. Edge-cropped boxes get a
        // second chance iff eye-locked with the head square fully
        // interior (≥2% margins) — a complete face is usable no matter
        // what the box does at the frame edge.
        let worst = x0.min(y0).min(1.0 - x1);
        if worst < 0.03 {
            let complete = match synth::head_square(&img, (x0, y0, x1, y1)) {
                Some((sx0, sy0, side)) => {
                    let m = 0.02 * img.width.max(img.height) as f64;
                    let head = img.crop(sx0, sy0, side, side);
                    synth::eye_pair(&head).is_some()
                        && sx0 as f64 >= m
                        && sy0 as f64 >= m
                        && (sx0 + side) as f64 <= img.width as f64 - m
                        && (sy0 + side) as f64 <= img.height as f64 - m
                }
                None => false,
            };
            if !complete {
                skipped.push(format!(
                    "{}: edge-cropped ({:.3} margin, face incomplete)",
                    file, worst
                ));
                continue;
            }
            kept.push(format!("{} (margin-exempt: face complete)", file));
        } else {
            kept.push(file.to_string());
        }
        plates.push(img);
        boxes.push((x0, y0, x1, y1));
    }
    println!(
        "synth: {} candidate(s), {} skipped",
        kept.len(),
        skipped.len()
    );
    for s in &skipped {
        println!("synth: skipped {}", s);
    }
    if debug_eyes {
        // Per-donor workup with the exact pipeline pieces synthesis
        // uses: extraction tier + aligned sharpness. If donors are
        // sharp and the median is mush, the fault is alignment; if
        // donors are mush, the fault is content.
        let mut sharps = Vec::new();
        for (k, (img, bx)) in plates.iter().zip(boxes.iter()).enumerate() {
            let diag = match synth::head_square(img, *bx) {
                Some((sx0, sy0, side)) => {
                    let head = img.crop(sx0, sy0, side, side);
                    format!("head {}x{} {}", side, side, synth::eye_debug(&head))
                }
                None => "no head square".to_string(),
            };
            let workup = match synth::extract_face_region(img, *bx, k) {
                Ok(hit) => {
                    let a = synth::align(&hit.image);
                    let s = synth::sharpness(&a);
                    sharps.push(s);
                    format!(
                        "tier={} sharp={:.4}",
                        if hit.eye_aligned {
                            "eye"
                        } else if hit.anchored {
                            "head"
                        } else {
                            "heur"
                        },
                        s
                    )
                }
                Err(e) => format!("unusable ({})", e),
            };
            println!("synth: eyes {}: {} | {}", kept[k], diag, workup);
        }
        sharps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if !sharps.is_empty() {
            println!(
                "synth: donor sharpness min={:.4} median={:.4} max={:.4} (median output was 0.0947)",
                sharps[0],
                sharps[sharps.len() / 2],
                sharps[sharps.len() - 1]
            );
        }
        // Locked-median probe: median of eye-tier donors ONLY, no
        // gate, nothing saved — a diagnostic, not a synthesis. If
        // this number beats the all-donor median, alignment is the
        // fault and eye-locked banking is the fix; if flat, the fault
        // is deeper (rotation, brow/eye mixing).
        let mut locked_aligned = Vec::new();
        for (img, bx) in plates.iter().zip(boxes.iter()) {
            if let Ok(hit) = synth::extract_face_region(img, *bx, 0)
                && hit.eye_aligned
            {
                locked_aligned.push(synth::align(&hit.image));
            }
        }
        if let Ok(med) = synth::median_face(&locked_aligned) {
            println!(
                "synth: locked-median probe: N={} sharpness={:.4} (diagnostic only)",
                locked_aligned.len(),
                synth::sharpness(&med)
            );
        } else {
            println!("synth: locked-median probe: no locked donors");
        }
        return;
    }
    println!(
        "synth: mode {}",
        if locked { "eye-locked" } else { "all-usable" }
    );
    let result = if locked {
        synth::synthesize_locked(&plates, &boxes, seed)
    } else {
        synth::synthesize(&plates, &boxes, seed)
    };
    match result {
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
                "locked": locked,
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
