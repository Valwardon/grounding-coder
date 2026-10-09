use grounding_coder::engine::measure;
use grounding_coder::engine::vision::Image;

/// Whole-frame subject discovery: report exactly what the measure
/// pipeline would read as the subject cluster before its gates rule.
/// A refused plate prints its candidate bbox here, so the answer to
/// "which region is the subject" is measured, not guessed.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("plate.bmp");
    let img = Image::load_bmp(std::path::Path::new(&path)).expect("plate");
    let (w, h) = (img.width, img.height);
    let surround = measure::border_median(&img);
    let dominant = img.dominant_color().map(|(c, _)| c).unwrap_or(surround);
    let ground = measure::ground_stats(&img);
    println!(
        "plate {}x{} — ground med {:?} sigma {:?}",
        w, h, ground.median, ground.sigma
    );
    for (name, center) in [
        ("border", ground.median),
        ("dominant", measure::opponent(dominant)),
    ] {
        let mask = measure::mask_far_from(&img, &ground, center);
        let opened = Image::morph_open(&mask, w, h, 1);
        let cleaned = Image::morph_close(&opened, w, h, 1);
        let comps = measure::components(&cleaned, w, h);
        let members = measure::cluster_columns(&comps);
        let area: usize = members.iter().map(|&i| comps[i].pixels.len()).sum();
        let frac = area as f64 / (w * h) as f64;
        let (mut fx0, mut fy0, mut fx1, mut fy1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        for &i in &members {
            let [x0, y0, x1, y1] = comps[i].bbox;
            fx0 = fx0.min(x0);
            fy0 = fy0.min(y0);
            fx1 = fx1.max(x1);
            fy1 = fy1.max(y1);
        }
        let edges = [fx0 <= 1, fy0 <= 1, fx1 + 2 >= w, fy1 + 2 >= h]
            .into_iter()
            .filter(|&t| t)
            .count();
        let candidate = if (0.01..0.9).contains(&frac) {
            "candidate"
        } else {
            "none"
        };
        println!(
            "{name:>8}: {area}px ({:.1}% {candidate}) bbox {fx0},{fy0},{fx1},{fy1} · \
             hfrac {:.2} wfrac {:.2} · edges {edges}",
            frac * 100.0,
            (fy1 - fy0 + 1) as f64 / h as f64,
            (fx1 - fx0 + 1) as f64 / w as f64,
        );
        for &i in &members {
            let [x0, y0, x1, y1] = comps[i].bbox;
            println!(
                "   comp {}px bbox {x0},{y0},{x1},{y1}",
                comps[i].pixels.len()
            );
        }
    }
}
