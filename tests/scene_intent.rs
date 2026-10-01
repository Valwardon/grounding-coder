//! Scene-intent prompt suite: known prompts, expected interpretations.
//!
//! The first milestone is the representation, not a renderer: this
//! suite measures whether explicit rules recover entities, actions,
//! materials, and relationships — and whether research queries and
//! capability verdicts come out honest. A fix that breaks a row here
//! is a regression, not progress.
use grounding_coder::engine::scene_intent::{
    describe_capability, match_capabilities, parse_scene, plan_research,
};

#[test]
fn man_on_mountain_parses_fully() {
    let spec = parse_scene("A man standing on a mountain.");
    assert_eq!(spec.subjects.len(), 1);
    assert_eq!(spec.subjects[0].stype, "man");
    assert!(spec.subjects[0].attributes.is_empty());
    assert_eq!(spec.actions.len(), 1);
    assert_eq!(spec.actions[0].atype, "stand");
    assert_eq!(spec.actions[0].actor, "man");
    assert_eq!(spec.actions[0].target.as_deref(), Some("mountain"));
    assert_eq!(spec.objects.len(), 1);
    assert_eq!(spec.objects[0].otype, "mountain");
    assert!(
        (spec.confidence - 1.0).abs() < 1e-9,
        "conf {}",
        spec.confidence
    );

    let plan = plan_research(&spec);
    assert!(
        plan.iter()
            .any(|q| q.queries.iter().any(|s| s.contains("mountain"))
                && q.capability == "photographic-place"),
        "mountain photo query missing: {:?}",
        plan.iter().map(|q| &q.queries).collect::<Vec<_>>()
    );
    // Subject + place are photographic (supported); the standing pose
    // needs a rig that does not exist (honestly missing).
    let verdicts = match_capabilities(&plan);
    assert!(
        verdicts
            .iter()
            .any(|v| v.supported && v.requirement.contains("subject")),
        "subject should be supported: {:?}",
        verdicts
            .iter()
            .map(|v| (&v.requirement, v.supported))
            .collect::<Vec<_>>()
    );
    assert!(
        verdicts
            .iter()
            .any(|v| !v.supported && v.missing.as_deref() == Some("articulated-pose")),
        "stand pose must name its missing capability"
    );
}

#[test]
fn woman_peace_sign_attributes_and_queries() {
    let spec = parse_scene("Human woman giving peace sign.");
    assert_eq!(spec.subjects.len(), 1);
    assert_eq!(spec.subjects[0].stype, "human");
    assert_eq!(spec.subjects[0].attributes, vec!["woman".to_string()]);
    assert_eq!(spec.actions.len(), 1);
    assert_eq!(spec.actions[0].atype, "peace_sign");
    assert!(
        (spec.confidence - 1.0).abs() < 1e-9,
        "conf {}",
        spec.confidence
    );

    let plan = plan_research(&spec);
    assert!(
        plan.iter()
            .any(|q| q.queries.iter().any(|s| s.contains("peace sign"))),
        "peace-sign query missing"
    );
    assert!(
        plan.iter().any(|q| q
            .queries
            .iter()
            .any(|s| s.contains("woman") && s.contains("photograph"))),
        "woman photo query missing"
    );
}

#[test]
fn macaroni_is_material_never_object() {
    let spec = parse_scene("A hat made of macaroni.");
    assert!(
        spec.objects
            .iter()
            .all(|o| o.otype != "macaroni" && o.otype != "pasta"),
        "macaroni must never be a scene object: {:?}",
        spec.objects
    );
    let hat = spec
        .objects
        .iter()
        .find(|o| o.otype == "hat")
        .expect("hat object");
    assert_eq!(hat.material.as_deref(), Some("macaroni"));
    assert!(
        spec.unresolved.iter().any(|u| u.contains("wearer")),
        "open wearer relationship missing: {:?}",
        spec.unresolved
    );
    assert!(
        (spec.confidence - 1.0).abs() < 1e-9,
        "conf {}",
        spec.confidence
    );

    let plan = plan_research(&spec);
    assert!(
        plan.iter()
            .any(|q| q.queries.iter().any(|s| s.contains("silhouette"))),
        "hat silhouette query missing"
    );
    assert!(
        plan.iter().any(|q| q
            .queries
            .iter()
            .any(|s| s.contains("macaroni") && s.contains("geometry"))),
        "macaroni geometry query missing"
    );
    // Photographic assets exist: everything here is supported.
    assert!(
        match_capabilities(&plan).iter().all(|v| v.supported),
        "material requirements should all be supported"
    );
}

#[test]
fn any_material_binds_generically() {
    // Straw, glass, steel: identical treatment, no special cases.
    for (prose, material) in [
        ("A hat made of straw.", "straw"),
        ("A tower out of glass.", "glass"),
        ("A bowl made from steel.", "steel"),
    ] {
        let spec = parse_scene(prose);
        assert!(
            spec.objects.iter().all(|o| o.otype != material),
            "{}: {:?}",
            prose,
            spec.objects
        );
        let obj = spec.objects.first().expect("base object");
        assert_eq!(obj.material.as_deref(), Some(material), "{}", prose);
        let plan = plan_research(&spec);
        assert!(
            plan.iter().any(|q| q
                .queries
                .iter()
                .any(|s| s.contains(material) && s.contains("geometry"))),
            "{}: no material query",
            prose
        );
        assert!(
            (spec.confidence - 1.0).abs() < 1e-9,
            "{}: conf {}",
            prose,
            spec.confidence
        );
    }
}

#[test]
fn waving_flag_disambiguates_by_object() {
    let spec = parse_scene("Man waving an American flag.");
    assert_eq!(spec.actions.len(), 1);
    let wave = &spec.actions[0];
    assert_eq!(wave.atype, "wave");
    assert!(!wave.ambiguous, "named flag must resolve the reading");
    assert_eq!(wave.object.as_deref(), Some("american_flag"));
    let flag = spec
        .objects
        .iter()
        .find(|o| o.otype == "american_flag")
        .expect("flag object");
    assert!(flag.attributes.contains(&"american".to_string()));
    assert!(flag.state.contains(&"waving".to_string()));

    let plan = plan_research(&spec);
    assert!(
        plan.iter()
            .any(|q| q.queries.iter().any(|s| s.contains("flag design"))),
        "flag design query missing"
    );
    let verdicts = match_capabilities(&plan);
    assert!(
        verdicts
            .iter()
            .any(|v| !v.supported && v.missing.as_deref() == Some("photo-cloth")),
        "cloth must name its missing capability"
    );
}

#[test]
fn bare_waving_preserves_ambiguity() {
    let spec = parse_scene("A man waving.");
    assert_eq!(spec.actions.len(), 1);
    assert!(spec.actions[0].ambiguous);
    assert_eq!(spec.actions[0].object, None);
    assert!(
        spec.unresolved.iter().any(|u| u.contains("hand or flag")),
        "ambiguity must be stated: {:?}",
        spec.unresolved
    );
}

#[test]
fn salute_names_missing_joints_no_block_figure() {
    let spec = parse_scene("Man saluting.");
    assert_eq!(spec.actions.len(), 1);
    assert_eq!(spec.actions[0].atype, "salute");
    assert_eq!(spec.actions[0].target.as_deref(), Some("head"));

    let plan = plan_research(&spec);
    let verdicts = match_capabilities(&plan);
    let salute = verdicts
        .iter()
        .find(|v| v.requirement.contains("salute"))
        .expect("salute verdict");
    assert!(!salute.supported);
    assert_eq!(salute.missing.as_deref(), Some("articulated-pose"));
    // No verdict may ever bless procedural humans — that path is
    // deleted, and the registry says so out loud.
    assert!(
        verdicts
            .iter()
            .all(|v| v.missing.as_deref() != Some("procedural-human-geometry") || !v.supported),
        "procedural geometry must never read as supported"
    );
    assert!(
        describe_capability("procedural-human-geometry")
            .unwrap_or("")
            .contains("Minecraft"),
        "registry must state why the path is gone"
    );
}

#[test]
fn synonyms_and_morphology_normalize() {
    let spec = parse_scene("A gentleman salutes.");
    assert_eq!(spec.subjects.len(), 1);
    assert_eq!(spec.subjects[0].stype, "man");
    assert_eq!(spec.actions.len(), 1);
    assert_eq!(spec.actions[0].atype, "salute");
    assert!(
        (spec.confidence - 1.0).abs() < 1e-9,
        "conf {}",
        spec.confidence
    );
}

#[test]
fn wearing_binds_worn_by() {
    let spec = parse_scene("A woman wearing a hat.");
    let hat = spec
        .objects
        .iter()
        .find(|o| o.otype == "hat")
        .expect("hat object");
    assert_eq!(hat.worn_by.as_deref(), Some("woman"));
    assert!(
        spec.unresolved.iter().all(|u| !u.contains("wearer")),
        "bound wearer must not stay open: {:?}",
        spec.unresolved
    );
    assert!(
        (spec.confidence - 1.0).abs() < 1e-9,
        "conf {}",
        spec.confidence
    );
}

#[test]
fn stand_without_place_stays_open() {
    let spec = parse_scene("A man stands.");
    assert_eq!(spec.actions.len(), 1);
    assert_eq!(spec.actions[0].target, None);
    assert!(
        spec.unresolved.iter().any(|u| u.contains("no place named")),
        "open place must be stated: {:?}",
        spec.unresolved
    );
}

#[test]
fn gibberish_parses_to_nothing() {
    let spec = parse_scene("!!! ... ???");
    assert!(spec.subjects.is_empty());
    assert!(spec.actions.is_empty());
    assert!(spec.objects.is_empty());
    assert_eq!(spec.confidence, 0.0);
}

#[test]
fn spec_serializes_to_doc_shape() {
    // The representation from the design: subjects, actions, objects
    // as JSON with type/actor/target/object/worn_by/material slots.
    let spec = parse_scene("Man waving an American flag.");
    let v = serde_json::to_value(&spec).expect("spec must serialize");
    assert!(v.get("subjects").is_some());
    assert!(v.get("actions").is_some());
    assert!(v.get("objects").is_some());
    assert!(v.get("unresolved").is_some());
    assert!(v.get("confidence").is_some());
    assert_eq!(v["actions"][0]["object"], "american_flag");
    assert_eq!(v["actions"][0]["ambiguous"], false);
}
