//! Intent benchmark: what the parser understands, case by case.
//!
//! Each case asserts the frame AND the disposition (execute / ask /
//! block) — a parse without a decision proves nothing. Run this suite
//! against every change to `understand.rs`; a fix that breaks a row
//! here is a regression, not progress.
use grounding_coder::engine::understand::{self, Disposition};

fn disp(prose: &str) -> (String, Disposition, f64) {
    let u = understand::understand(prose, None);
    let d = understand::disposition(u.confidence, &u.frame, prose);
    (u.frame, d, u.confidence)
}

#[test]
fn straightforward_bare_struct_asks_for_fields() {
    // A struct with no fields cannot build — the calibrated answer is
    // a question naming what's missing, not a doomed execution.
    let (frame, d, conf) = disp("Create a Counter struct.");
    assert_eq!(frame, "create");
    assert_eq!(d, Disposition::Ask, "conf {}", conf);
}

#[test]
fn straightforward_page_with_content_executes() {
    // Quoted content completes the page: verb + kind + name + content.
    let (frame, d, conf) = disp("Create a page called Home with \"welcome back\".");
    assert_eq!(frame, "create");
    assert_eq!(d, Disposition::Execute);
    assert!(conf >= 0.75, "conf {}", conf);
}

#[test]
fn typo_tolerance_executes() {
    // "buld" forgiven, name passes through verbatim (never corrected).
    let u = understand::understand("Buld a settings page.", None);
    assert_eq!(u.frame, "create");
    let def = u
        .intent
        .define
        .into_iter()
        .flatten()
        .next()
        .expect("define");
    assert_eq!(def.name, "Settings");
}

#[test]
fn head_noun_governs_kind() {
    // "settings page" is a page about Settings — the LAST kind word
    // (head noun) governs, not the first. A first-match rule would
    // misfile this as a config.
    let u = understand::understand("Buld a settings page.", None);
    let def = u
        .intent
        .define
        .into_iter()
        .flatten()
        .next()
        .expect("define");
    assert_eq!(def.kind, "page");
    assert_eq!(def.name, "Settings");
    // Same rule, single-kind phrase.
    let u = understand::understand("Create a counter struct.", None);
    let def = u
        .intent
        .define
        .into_iter()
        .flatten()
        .next()
        .expect("define");
    assert_eq!(def.kind, "struct");
    assert_eq!(def.name, "Counter");
}

#[test]
fn typo_verb_never_a_name() {
    // "bild" sits exactly where a name would go, but it is a typo'd
    // action — the adjacent scan must skip it like any verb, leaving
    // the request nameless (a question) instead of a "Bild" page.
    let u = understand::understand("Build a bild page.", None);
    let names: Vec<String> = u
        .intent
        .define
        .into_iter()
        .flatten()
        .map(|d| d.name)
        .collect();
    assert!(
        !names.iter().any(|n| n == "Bild"),
        "typo'd verb became a name: {:?}",
        names
    );
    let (_, d, _) = disp("Build a bild page.");
    assert_ne!(d, Disposition::Execute);
}

#[test]
fn destructive_blocks_when_fully_specified() {
    // Disposition checks destructive words BEFORE confidence: even a
    // complete request (verb + kind + quoted name) never executes.
    let (frame, d, conf) = disp("Delete the file called Cleanup.");
    assert_eq!(d, Disposition::Block, "frame {} conf {}", frame, conf);
}

#[test]
fn missing_information_asks() {
    // A dashboard with no content and no name: parseable, unactionable.
    let (_frame, d, conf) = disp("Make a dashboard.");
    assert!(conf < 0.75, "conf {}", conf);
    assert_eq!(d, Disposition::Ask);
}

#[test]
fn multi_step_asks_for_staging() {
    let (frame, d, _) = disp("Create a config and then add tests.");
    assert_eq!(d, Disposition::Ask, "frame was {}", frame);
}

#[test]
fn conflicting_requirements_ask() {
    let (frame, d, _) = disp("Make it read-only but allow editing.");
    assert_eq!(d, Disposition::Ask, "frame was {}", frame);
}

#[test]
fn unknown_concepts_ask_with_curiosity() {
    let (frame, d, conf) = disp("Add a flux capacitor.");
    assert_eq!(d, Disposition::Ask, "frame {} conf {}", frame, conf);
    let cs = understand::curiosities("Add a flux capacitor.", None);
    assert!(
        cs.iter()
            .any(|c| c.unknown == "flux" || c.unknown == "capacitor"),
        "no curiosity about the unknown: {:?}",
        cs.iter().map(|c| &c.unknown).collect::<Vec<_>>()
    );
}

#[test]
fn context_dependent_names_history() {
    // "the previous screen" is answerable only from verified history.
    assert!(understand::wants_context(
        "Give it the same layout as the previous screen."
    ));
    assert!(!understand::wants_context("Build the apk."));
}

#[test]
fn safety_boundary_blocks() {
    // Destructive ambiguity never executes, however confident.
    let (frame, d, _) = disp("Remove the old files");
    assert_eq!(d, Disposition::Block, "frame was {}", frame);
    let (frame, d, _) = disp("delete everything now");
    assert_eq!(d, Disposition::Block, "frame was {}", frame);
}

#[test]
fn gibberish_blocks_without_questions() {
    // Vacuous input: nothing to ask about, nothing to do.
    let (_, d, conf) = disp("!!! ... ???");
    assert_eq!(d, Disposition::Block);
    assert_eq!(conf, 0.0);
}

#[test]
fn function_contracts_parse_to_cases() {
    // given/returns quotes become synthesis cases; page quotes stay content.
    let u = understand::understand(
        "Create a function called Shout given \"hi\" returns \"HI\".",
        None,
    );
    assert_eq!(u.frame, "create");
    let def = u
        .intent
        .define
        .into_iter()
        .flatten()
        .next()
        .expect("define");
    assert_eq!(def.kind, "function");
    assert_eq!(def.cases.len(), 1);
    assert_eq!(def.cases[0].input, "\"hi\"");
    assert_eq!(def.cases[0].expected, "\"HI\"");
    assert_eq!(
        def.signature.as_deref(),
        Some("fn Shout(input: &str) -> String")
    );
    // No case keywords: quotes are not cases.
    let u = understand::understand("Create a page called Home with \"welcome back\".", None);
    let def = u
        .intent
        .define
        .into_iter()
        .flatten()
        .next()
        .expect("define");
    assert!(def.cases.is_empty());
}
