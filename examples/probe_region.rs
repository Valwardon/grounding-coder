use grounding_coder::engine::measure;
use grounding_coder::engine::vision::{Image, Rgb};

/// Region-local separation probe: for a user-supplied box, re-derive the
/// ground from the box's own border ring and ask whether the biggest
/// separable component there is a substantial, bounded figure.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("plate.bmp");
    let box_: Vec<u32> = args
        .get(2)
        .expect("x0,y0,x1,y1")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let img = Image::load_bmp(std::path::Path::new(&path)).expect("plate");
    let (w, h) = (img.width, img.height);
    let [bx0, by0, bx1, by1] = [box_[0], box_[1], box_[2], box_[3]];
    let region = measure::ground_stats_in(&img, bx0, by0, bx1, by1);
    let surround = measure::border_median(&img);
    let dominant = img.dominant_color().map(|(c, _)| c).unwrap_or(surround);
    println!(
        "plate {}x{} box {bx0},{by0},{bx1},{by1} — region ground med {:?} sigma {:?}",
        w, h, region.median, region.sigma
    );
    for (name, center) in [
        ("region-border", region.median),
        ("dominant", measure::opponent(dominant)),
    ] {
        let mask = measure::mask_far_from(&img, &region, center);
        let cleaned = Image::morph_open(&mask, w, h, 1);
        let cleaned = Image::morph_close(&cleaned, w, h, 1);
        let blob = Image::largest_blob(&cleaned, w, h);
        let frac = blob.len() as f64 / (w * h) as f64;
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        for &i in &blob {
            let (x, y) = (i as u32 % w, i as u32 / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        let edges = [x0 <= 1, y0 <= 1, x1 + 2 >= w, y1 + 2 >= h]
            .into_iter()
            .filter(|&t| t)
            .count();
        println!(
            "  {name}: blob {}px ({:.1}%) bbox {x0},{y0},{x1},{y1} · hfrac {:.2} wfrac {:.2} · edges {edges}",
            blob.len(),
            frac * 100.0,
            (y1 - y0 + 1) as f64 / h as f64,
            (x1 - x0 + 1) as f64 / w as f64,
        );
    }
    let _ = Rgb::new(0, 0, 0);
}
