//! Photo gate: every build proves the loop holds end to end —
//! messy prose routes to the right action, conditioning builds
//! deterministically from researched specs, novelty separates
//! copies from strangers, and scenes render as worlds.
//!
//! Three questions, three answers, all offline and deterministic
//! (no generation pass runs in tests — the generator is the one
//! network step, proven live, not here):
//!   1. Does it understand the action? Visual verbs (`imagine`,
//!      `render`, `draw`) reach the create frame with a confidence
//!      receipt — Ask with curiosity, never a vacuous Block or a
//!      blind Execute.
//!   2. Is conditioning compact and deterministic? The same spec
//!      builds the same prompt twice; conditioning serializes
//!      small with no pixel fields.
//!   3. What about scene? The demo world renders sky/tower/ground
//!      with sky up top and ground below (pixel-sampled, not assumed).
//!
//! The scene photo commits under `samples/` next to its JSON receipt —
//! the photo IS the test output, not a promise of one.
use grounding_coder::engine::{
    generate::{build_prompt, conditioning_from_spec, novelty},
    scene::{self, demo_scene},
    scene_intent::parse_scene,
    understand::{self, Disposition},
    vision::{Image, Rgb},
};
use std::collections::HashMap;

const W: u32 = 320;
const H: u32 = 240;

fn disp(prose: &str) -> (String, Disposition, f64) {
    let u = understand::understand(prose, None);
    let d = understand::disposition(u.confidence, &u.frame, prose);
    (u.frame, d, u.confidence)
}

fn receipt_map(receipt: &[(String, u64)]) -> HashMap<&str, u64> {
    receipt.iter().map(|(n, c)| (n.as_str(), *c)).collect()
}

fn write_photo(
    stem: &str,
    img: &Image,
    receipt: &[(String, u64)],
) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = std::path::Path::new("samples");
    let bmp = dir.join(format!("{}-{}x{}.bmp", stem, W, H));
    let json = dir.join(format!("{}-{}x{}.json", stem, W, H));
    img.save_bmp(&bmp).expect("photo must save");
    let body = serde_json::json!({
        "width": W,
        "height": H,
        "receipt": receipt_map(receipt),
    });
    std::fs::write(&json, serde_json::to_string_pretty(&body).unwrap()).expect("receipt must save");
    (bmp, json)
}

fn check_bmp_file(bmp: &std::path::Path) -> Image {
    let bytes = std::fs::read(bmp).expect("photo must exist");
    assert!(bytes.len() > 100_000, "photo too small: {}", bytes.len());
    assert_eq!(&bytes[0..2], b"BM", "not a BMP file");
    let back = Image::load_bmp(bmp).expect("photo must read back");
    assert_eq!((back.width, back.height), (W, H));
    back
}

// --- 1. Does it understand the action? ---

#[test]
fn photo_prose_routes_imagine_to_create_ask() {
    // A portrait request with no name and no quoted content is
    // parseable but unactionable: Ask with curiosity, never Block.
    let (frame, d, conf) = disp("Imagine a portrait of a waving man.");
    assert_eq!(frame, "create");
    assert_eq!(d, Disposition::Ask, "conf {}", conf);
    let cs = understand::curiosities("Imagine a portrait of a waving man.", None);
    assert!(!cs.is_empty(), "photo prose must raise curiosities");
}

#[test]
fn photo_prose_routes_render_without_blocking() {
    // `render` is a visual creation verb: the demo-scene request
    // reaches a known frame and asks for what's missing — it never
    // Blocks vacantly and never Executes blind.
    let (frame, d, _) = disp("Render the demo scene.");
    assert_eq!(frame, "create");
    assert_eq!(d, Disposition::Ask);
}

#[test]
fn photo_prose_routes_draw_waver_to_create_ask() {
    let (frame, d, _) = disp("Draw a waving man.");
    assert_eq!(frame, "create");
    assert_eq!(d, Disposition::Ask);
}

// --- 2. Is conditioning compact and deterministic? ---

#[test]
fn conditioning_builds_same_prompt_twice() {
    // Same researched spec twice → same prompt, compact JSON, no
    // pixel fields anywhere in the contract.
    let spec = parse_scene("A man standing on a mountain.");
    let a = conditioning_from_spec(&spec, 9, 42);
    let b = conditioning_from_spec(&spec, 9, 42);
    assert_eq!(build_prompt(&a), build_prompt(&b));
    assert!(build_prompt(&a).contains("man"));
    let json = serde_json::to_value(&a).unwrap();
    assert!(json.get("image").is_none());
    assert!(json.to_string().len() < 1024);
}

#[test]
fn novelty_separates_copy_from_stranger_offline() {
    // The gate the live path enforces: identical frames read 0.0,
    // strangers read far above it.
    let a = Image::blank(32, 32, Rgb::new(200, 150, 115));
    let b = Image::blank(32, 32, Rgb::new(200, 150, 115));
    assert_eq!(novelty(&a, &b), 0.0);
    let c = Image::blank(32, 32, Rgb::new(10, 20, 200));
    assert!(novelty(&a, &c) > 0.2);
}

// --- 3. What about scene? ---

#[test]
fn scene_renders_world_receipt_and_commits_photo() {
    let (img, receipt) = scene::render(&demo_scene(W, H));
    let m = receipt_map(&receipt);
    assert!(
        m.get("sky").copied().unwrap_or(0) > 15_000,
        "sky must read, got {:?}",
        receipt
    );
    assert!(
        m.get("ground").copied().unwrap_or(0) > 25_000,
        "ground must read, got {:?}",
        receipt
    );
    assert!(
        m.get("tower").copied().unwrap_or(0) > 1_500,
        "tower must read, got {:?}",
        receipt
    );
    // Sky up top (blue dominates), ground below (green dominates) —
    // sampled from the pixels, not assumed from the receipt.
    let top = img.get(0, 0).expect("top-left pixel");
    assert!(
        top.b > 150 && top.b > top.r,
        "top must be sky, got {:?}",
        top
    );
    let bottom = img.get(0, H - 1).expect("bottom-left pixel");
    assert!(
        bottom.g > bottom.r && bottom.g >= bottom.b,
        "bottom must be ground, got {:?}",
        bottom
    );

    let (bmp, json) = write_photo("photo-gate-scene", &img, &receipt);
    check_bmp_file(&bmp);
    assert!(json.exists());
}

#[test]
fn photos_are_byte_identical_reruns() {
    // Determinism is the whole claim: same scene twice, same bytes.
    let (a, _) = scene::render(&demo_scene(W, H));
    let (b, _) = scene::render(&demo_scene(W, H));
    let pa = std::path::Path::new("/tmp/photo-gate-det-a.bmp");
    let pb = std::path::Path::new("/tmp/photo-gate-det-b.bmp");
    a.save_bmp(pa).unwrap();
    b.save_bmp(pb).unwrap();
    assert_eq!(std::fs::read(pa).unwrap(), std::fs::read(pb).unwrap());
    let _ = std::fs::remove_file(pa);
    let _ = std::fs::remove_file(pb);
}
