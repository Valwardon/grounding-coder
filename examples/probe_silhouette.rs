use grounding_coder::engine::measure;
use grounding_coder::engine::scene::{render_labels, scene_from_facts};

fn main() {
    let facts_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/visual/a-woman-standing.json".into());
    let json = std::fs::read_to_string(&facts_path).expect("facts");
    let facts: measure::VisualFacts = serde_json::from_str(&json).expect("parse");
    let (img, labels, table) = render_labels(&scene_from_facts(&facts, 320, 240));
    let subs: Vec<usize> = table
        .iter()
        .enumerate()
        .filter(|(_, n)| n.starts_with("subject"))
        .map(|(i, _)| i)
        .collect();
    println!("subject materials: {} table {}", subs.len(), table.len());
    println!("silhouette per band (row x-range of ANY subject material):");
    for cy in 0..24 {
        let y0 = cy * img.height / 24;
        let y1 = (cy + 1) * img.height / 24;
        let mut xs: Vec<usize> = Vec::new();
        for y in y0..y1 {
            for x in 0..img.width {
                let l = labels[(y * img.width + x) as usize] as usize;
                if subs.contains(&l) {
                    xs.push(x as usize);
                }
            }
        }
        let s = if xs.is_empty() {
            "EMPTY".to_string()
        } else {
            let lo = xs.iter().min().unwrap();
            let hi = xs.iter().max().unwrap();
            format!("x {lo:3}..{hi:3}  width {}", hi - lo + 1)
        };
        println!("  y{y0:3}-{y1:3}  {s}");
    }
}
