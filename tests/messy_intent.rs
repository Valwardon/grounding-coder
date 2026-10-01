//! Messy-intent battery: everyday spellings, kind synonyms, typo'd
//! kinds, verb inflections, and material relations (`X made of Y`).
//!
//! Each case asserts the frame AND the disposition — a parse without a
//! decision proves nothing. Run with the rest of the suite; a fix that
//! breaks a row here or in `intent_benchmark` is a regression.
use grounding_coder::engine::understand::{self, Disposition};

fn disp(prose: &str) -> (String, Disposition, f64) {
    let u = understand::understand(prose, None);
    let d = understand::disposition(u.confidence, &u.frame, prose);
    (u.frame, d, u.confidence)
}

#[test]
fn kind_synonym_webpage_executes() {
    // Everyday spelling: webpage == page.
    let u = understand::understand("Create a webpage called Home with \"welcome back\".", None);
    assert_eq!(u.frame, "create");
    let def = u
        .intent
        .define
        .into_iter()
        .flatten()
        .next()
        .expect("define");
    assert_eq!(def.kind, "page");
    assert_eq!(def.name, "Home");
    let (_, d, conf) = disp("Create a webpage called Home with \"welcome back\".");
    assert_eq!(d, Disposition::Execute, "conf {}", conf);
}

#[test]
fn kind_synonym_homepage_executes() {
    let (_, d, conf) = disp("Make a homepage called Home with \"hello world\".");
    assert_eq!(d, Disposition::Execute, "conf {}", conf);
}

#[test]
fn plural_kind_parses() {
    // `pages` strips to `page` for the kind slot; the name still
    // resolves from the adjacent noun.
    let u = understand::understand("Create a Counter pages.", None);
    assert_eq!(u.frame, "create");
    let def = u
        .intent
        .define
        .into_iter()
        .flatten()
        .next()
        .expect("define");
    assert_eq!(def.kind, "page");
    assert_eq!(def.name, "Counter");
}

#[test]
fn typo_kind_forgiven_name_kept() {
    // `stuct` forgives to struct; `Counter` passes through verbatim.
    let u = understand::understand("Create a Counter stuct.", None);
    assert_eq!(u.frame, "create");
    let def = u
        .intent
        .define
        .into_iter()
        .flatten()
        .next()
        .expect("define");
    assert_eq!(def.kind, "struct");
    assert_eq!(def.name, "Counter");
    // Parsable but fieldless structs ask, never execute blind.
    let (_, d, _) = disp("Create a Counter stuct.");
    assert_eq!(d, Disposition::Ask);
}

#[test]
fn verb_inflection_builds() {
    // `Building` is an action, not a name.
    let u = understand::understand("Building the android apk.", None);
    assert_eq!(u.frame, "build");
    assert_eq!(u.intent.platform, "android".to_string());
}

#[test]
fn visual_verb_creates_with_receipt() {
    // `Imagine` routes to creation; a bare portrait with no
    // name/content asks with curiosity instead of blocking vacantly.
    let (frame, d, conf) = disp("Imagine a portrait of a woman.");
    assert_eq!(frame, "create");
    assert_eq!(d, Disposition::Ask, "conf {}", conf);
    let cs = understand::curiosities("Imagine a portrait of a woman.", None);
    assert!(
        cs.iter()
            .any(|c| c.unknown == "portrait" || c.unknown == "woman"),
        "expected curiosity, got {:?}",
        cs.iter().map(|c| &c.unknown).collect::<Vec<_>>()
    );
}

#[test]
fn material_relation_banks_research_plan() {
    // The walkthrough example: Steps A+B parse the material edge,
    // Steps C+D become explicit research questions. Macaroni here is
    // one material among many — the machinery is fully generic.
    let u = understand::understand("A hat made of macaroni.", None);
    assert!(
        u.intent
            .constraints
            .iter()
            .any(|c| c == "material:macaroni"),
        "constraints {:?}",
        u.intent.constraints
    );
    assert!(
        u.intent.references.iter().any(|r| r == "hat"),
        "references {:?}",
        u.intent.references
    );
    assert!(
        u.intent.references.iter().any(|r| r == "macaroni"),
        "references {:?}",
        u.intent.references
    );
    // Research plan: base shape, material geometry, placement rules.
    assert!(
        u.intent
            .unknown_requirements
            .iter()
            .any(|q| q.contains("hat") && q.contains("shape")),
        "plan missing {:?}",
        u.intent.unknown_requirements
    );
    assert!(
        u.intent
            .unknown_requirements
            .iter()
            .any(|q| q.contains("macaroni") && q.contains("geometry")),
        "plan missing {:?}",
        u.intent.unknown_requirements
    );
    // Asks with curiosity — never a vacuous block, never a blind execute.
    let (_, d, _) = disp("A hat made of macaroni.");
    assert_eq!(d, Disposition::Ask);
    let cs = understand::curiosities("A hat made of macaroni.", None);
    assert!(
        cs.iter()
            .any(|c| c.unknown == "hat" || c.unknown == "macaroni"),
        "no curiosity {:?}",
        cs.iter().map(|c| &c.unknown).collect::<Vec<_>>()
    );
}

#[test]
fn material_out_of_variant() {
    let u = understand::understand("A tower out of glass.", None);
    assert!(
        u.intent.constraints.iter().any(|c| c == "material:glass"),
        "constraints {:?}",
        u.intent.constraints
    );
}

#[test]
fn any_material_parses_identically() {
    // No material is special: straw, steel, glass all bank the same
    // constraint/reference/plan shape as the walkthrough example.
    for (prose, tag) in [
        ("A hat made of straw.", "material:straw"),
        ("A hat made of steel.", "material:steel"),
        ("A bowl made from wood.", "material:wood"),
    ] {
        let u = understand::understand(prose, None);
        assert!(
            u.intent.constraints.iter().any(|c| c == tag),
            "{}: {:?}",
            prose,
            u.intent.constraints
        );
        assert!(
            u.intent
                .unknown_requirements
                .iter()
                .any(|q| q.contains("geometry")),
            "{}: {:?}",
            prose,
            u.intent.unknown_requirements
        );
        let (_, d, _) = disp(prose);
        assert_eq!(d, Disposition::Ask, "{}", prose);
    }
}

#[test]
fn destructive_still_blocks_with_material() {
    // Destructive beats informative: even a material-rich request refuses.
    let (_, d, _) = disp("Delete the hat made of macaroni.");
    assert_eq!(d, Disposition::Block);
}
