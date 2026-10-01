//! Evidence-store suite: requirements from specs, enforced budgets,
//! collection trails — all offline with synthetic fetches. Live
//! collection runs through `gc research` (same runner, plate source).
use grounding_coder::engine::evidence::{
    Category, EvidenceStore, Sufficiency, VisualExample, collect_examples, example_from_plate,
    features_from_plate,
};
use grounding_coder::engine::scene_intent::{parse_scene, plan_research};
use grounding_coder::engine::vision::{Image, Rgb};
use std::collections::HashMap;

fn synth_fetch(query: &str, need: usize) -> (Vec<VisualExample>, Vec<String>) {
    if query.contains("nothing") {
        return (Vec::new(), vec!["no hits".to_string()]);
    }
    let mut out = Vec::new();
    for i in 0..need.min(2) {
        out.push(VisualExample {
            source: format!("synthetic:{}:{}", query, i),
            license: "synthetic".to_string(),
            basis: "test".to_string(),
            bbox: [0, 0, 100, 150],
            segmentation: Some("skin-blob:0.10".to_string()),
            keypoints: Some("unavailable: fine-joint extraction not implemented".to_string()),
            dimensions: (100, 150),
            features: HashMap::new(),
        });
    }
    (out, Vec::new())
}

#[test]
fn requirements_derive_from_full_scene() {
    let spec =
        parse_scene("Make a man saluting, waving an American flag, with a hat made of straw.");
    let plan = plan_research(&spec);
    let mut store = EvidenceStore::with_budgets(2, 3, 4);
    store.require_from_spec(&spec, &plan);
    let ids: Vec<&str> = store.requirements.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.iter().any(|i| i.contains("subject")), "{:?}", ids);
    assert!(ids.iter().any(|i| i.contains("salute")), "{:?}", ids);
    assert!(ids.iter().any(|i| i.contains("wave")), "{:?}", ids);
    assert!(ids.iter().any(|i| i.contains("american_flag")), "{:?}", ids);
    assert!(ids.iter().any(|i| i.contains("hat")), "{:?}", ids);
    // Straw, like any material, becomes its own requirement —
    // related to the hat, never an object.
    let mat = store
        .requirements
        .iter()
        .find(|r| r.id.contains("straw"))
        .expect("straw requirement");
    assert_eq!(mat.category, Category::Material);
    assert!(mat.relationships.iter().any(|rel| rel.rel == "material_on"));
    assert!(mat.research_queries.iter().any(|q| q.contains("straw")));
    // Wave carries its object relationship; salute its actor.
    let wave = store
        .requirements
        .iter()
        .find(|r| r.id == "action:wave")
        .expect("wave requirement");
    assert!(wave.relationships.iter().any(|rel| rel.rel == "object"));
    // Nothing derives a model at this stage.
    assert!(store.requirements.iter().all(|r| r.model.is_none()));
}

#[test]
fn budgets_gate_collection() {
    let mut store = EvidenceStore::with_budgets(2, 3, 4);
    store
        .requirements
        .push(grounding_coder::engine::evidence::VisualRequirement {
            id: "subject:man".to_string(),
            category: Category::Subject,
            concept: "man".to_string(),
            attributes: Vec::new(),
            relationships: Vec::new(),
            research_queries: vec!["human man photograph".to_string()],
            examples: Vec::new(),
            model: None,
        });
    let mk = || VisualExample {
        source: "synthetic".to_string(),
        license: "synthetic".to_string(),
        basis: "test".to_string(),
        bbox: [0, 0, 10, 10],
        segmentation: None,
        keypoints: None,
        dimensions: (10, 10),
        features: HashMap::new(),
    };
    assert_eq!(
        store.sufficiency("subject:man"),
        Ok(Sufficiency::Insufficient { n: 0, need: 2 })
    );
    store.add_example("subject:man", mk()).unwrap();
    assert_eq!(
        store.sufficiency("subject:man"),
        Ok(Sufficiency::Insufficient { n: 1, need: 2 })
    );
    store.add_example("subject:man", mk()).unwrap();
    assert_eq!(
        store.sufficiency("subject:man"),
        Ok(Sufficiency::Collecting { n: 2 })
    );
    store.add_example("subject:man", mk()).unwrap();
    assert_eq!(
        store.sufficiency("subject:man"),
        Ok(Sufficiency::Sufficient { n: 3 })
    );
    store.add_example("subject:man", mk()).unwrap();
    assert!(store.add_example("subject:man", mk()).is_err());
    assert!(store.add_example("nope", mk()).is_err());
    assert!(store.sufficiency("nope").is_err());
}

#[test]
fn collector_walks_queries_until_want() {
    let (got, trail) = collect_examples(&["a".to_string(), "b".to_string()], 3, synth_fetch);
    assert_eq!(got.len(), 3);
    assert!(trail.iter().any(|t| t.contains("'a'")));
    assert!(trail.iter().any(|t| t.contains("'b'")));
    // Misses land in the trail, honestly.
    let (got, trail) = collect_examples(&["nothing here".to_string()], 3, synth_fetch);
    assert!(got.is_empty());
    assert!(trail.iter().any(|t| t.contains("refused")));
}

#[test]
fn features_come_from_classifiers() {
    let mut img = Image::blank(100, 150, Rgb::new(40, 60, 90));
    img.draw_rect(30, 10, 40, 30, Rgb::new(200, 150, 115));
    let f = features_from_plate(&img);
    assert!(f.get("skin_head").copied().unwrap_or(0.0) > 0.2);
    assert!(f.contains_key("aspect"));
    assert!(f.contains_key("brightness"));
    assert!(f.contains_key("tone_r"));
    let ex = example_from_plate("synthetic", "synthetic", "test", &img);
    assert_eq!(ex.bbox, [0, 0, 100, 150]);
    assert!(ex.segmentation.is_some());
    assert!(ex.keypoints.is_some_and(|k| k.contains("unavailable")));
}

#[test]
fn default_budgets_are_spec_values() {
    let store = EvidenceStore::new();
    assert_eq!(
        (store.minimum, store.preferred, store.maximum),
        (20, 100, 500)
    );
}
