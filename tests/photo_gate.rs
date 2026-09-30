//! Photo gate: every build proves, with real pixels, that the loop
//! holds end to end — messy prose routes to the right action, humans
//! render as humans, and scenes render as worlds.
//!
//! Three questions, three answers, all offline and deterministic:
//!   1. Does it understand the action? Visual verbs (`imagine`,
//!      `render`, `draw`) reach the create frame with a confidence
//!      receipt — Ask with curiosity, never a vacuous Block or a
//!      blind Execute.
//!   2. Does it create humans? The genome waver renders skin, flag,
//!      and noodle pixels (receipt-measured, thresholds pinned from
//!      live 320x240 renders with ~50% margin).
//!   3. What about scene? The demo world renders sky/tower/ground
//!      with sky up top and ground below (pixel-sampled, not assumed).
//!
//! Both photos commit under `samples/` next to their JSON receipts —
//! the photo IS the test output, not a promise of one.
use grounding_coder::engine::{
    figure, human,
    scene::{self, demo_scene},
    understand::{self, Disposition},
    vision::Image,
};
use std::collections::HashMap;

const W: u32 = 320;
const H: u32 = 240;

fn test_genome() -> human::HumanGenome {
    human::HumanGenome::with_params(35, 0.5, 0.55, 0.5, 0.95, 0.8, 0.5, 0.6, 0.5, 1.0)
        .expect("test genome validates")
}

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

// --- 2. Does it create humans? ---

#[test]
fn figure_renders_human_pixels_and_commits_photo() {
    let (img, receipt) = scene::render(&figure::waver_scene(W, H, &test_genome()));
    let m = receipt_map(&receipt);
    let skin = m.get("skin").copied().unwrap_or(0) + m.get("eye-white").copied().unwrap_or(0);
    assert!(skin > 600, "face must read, got {:?}", receipt);
    let red = m.get("flag-red").copied().unwrap_or(0)
        + m.get("leaf").copied().unwrap_or(0)
        + m.get("mouth-open").copied().unwrap_or(0);
    assert!(red > 500, "flag must read, got {:?}", receipt);
    let white =
        m.get("flag-white").copied().unwrap_or(0) + m.get("eye-white").copied().unwrap_or(0);
    assert!(white > 250, "contrast must read, got {:?}", receipt);
    let noodle: u64 = m.get("noodle").copied().unwrap_or(0);
    assert!(noodle > 150, "noodle hat must read, got {:?}", receipt);

    // Same finish as `gc figure` (fixed seed: byte-identical every run).
    let mut finished = img;
    finished.grade(1.08, 4.0);
    finished.vignette(0.25);
    finished.grain(0xF16E, 4);
    let (bmp, json) = write_photo("photo-gate-figure", &finished, &receipt);
    let back = check_bmp_file(&bmp);
    assert!(back.mean_brightness() > 0.1 && back.mean_brightness() < 0.9);
    assert!(json.exists());
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
