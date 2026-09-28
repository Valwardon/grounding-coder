//! Drive the engine as a library:
//! `cargo run --example build_page -- <project-dir> <intent-json> [budget]`
//! `cargo run --example build_page -- <project-dir> --prose "<prompt>" [budget]`
//! `cargo run --example build_page -- <project-dir> --understand "<prompt>"`
//! prints the outcome; the project dir holds whatever the engine proved
//! and committed.
//!
//! `--prose` resolves the prompt through the deterministic understander
//! (lexicon + frames, no model) and runs the engine on the result when
//! confidence clears the threshold. Successful prose runs are recorded
//! into the project's translation history (verified outcomes only).
//!
//! `--understand` runs no model and touches nothing: the deterministic
//! lexical parser prints the parsed intent, its confidence receipt, and
//! any precedent — comprehension you can audit line by line.
use grounding_coder::engine::CodeBot;

const USAGE: &str = "usage: build_page <project-dir> (<intent-json> | --prose \"<prompt>\" | --understand \"<prompt>\") [budget]";

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let project = args.next().expect(USAGE);
    let second = args.next().expect(USAGE);
    if second == "--understand" {
        let prompt = args.next().expect(USAGE);
        show_understanding(&project, &prompt);
        return;
    }
    let (intent_json, budget, prose) = if second == "--prose" {
        let prompt = args.next().expect(USAGE);
        let budget: u32 = args.next().and_then(|b| b.parse().ok()).unwrap_or(5);
        (resolve_prose(&project, &prompt).await, budget, Some(prompt))
    } else {
        let budget: u32 = args.next().and_then(|b| b.parse().ok()).unwrap_or(5);
        let intent_json = std::fs::read_to_string(&second).expect("read intent");
        (intent_json, budget, None)
    };
    let mut bot = CodeBot::new(&project, budget);
    // GITHUB_TOKEN env wires the publish actor when the intent asks for it.
    if let Ok(token) = std::env::var("GITHUB_TOKEN") {
        bot.set_github_token(Some(token));
    }
    // GROUNDING_ALLOW_REPLICATION=1 opts out of clean-room authorship so
    // `files[]` manifests (owned templates only) are honored. Default is
    // clean-room: replication refused before any fetch.
    if std::env::var("GROUNDING_ALLOW_REPLICATION").is_ok_and(|v| v == "1") {
        bot.set_clean_room(false);
    }
    match bot.run_task(&intent_json).await {
        Ok(outcome) => {
            println!("{}", outcome);
            // History learns only from verified outcomes: record the
            // prose→intent pair when the engine proved something.
            if let Some(prompt) = prose {
                let rendered = format!("{}", outcome);
                if (rendered.contains("SUCCESS") || rendered.contains("PARTIAL"))
                    && let Ok(intent) = serde_json::from_str::<
                        grounding_coder::engine::tasks::StructuredIntent,
                    >(&intent_json)
                {
                    grounding_coder::engine::understand::save_history(
                        std::path::Path::new(&project),
                        &prompt,
                        &intent,
                    );
                }
            }
        }
        Err(e) => {
            eprintln!("ENGINE FAILED: {}", e);
            std::process::exit(1);
        }
    }
}

/// Deterministic comprehension without any model: parse the prose,
/// print the intent, the confidence receipt, and any precedent.
/// Read-only — never touches the project beyond reading history.
fn show_understanding(project: &str, prompt: &str) {
    use grounding_coder::engine::understand;
    let understood = understand::understand(prompt, Some(std::path::Path::new(project)));
    println!("frame: {}", understood.frame);
    println!("confidence: {:.2}", understood.confidence);
    match &understood.similar {
        Some(hit) => println!("precedent: {:.2} {}", hit.score, hit.goal),
        None => println!("precedent: none"),
    }
    match serde_json::to_string_pretty(&understood.intent) {
        Ok(json) => println!("{}", json),
        Err(e) => {
            eprintln!("UNDERSTAND FAILED to serialize: {}", e);
            std::process::exit(1);
        }
    }
    // Curiosity: what the parse could not place, with hypotheses.
    // Read-only like everything else here.
    let curios = understand::curiosities(prompt, Some(std::path::Path::new(project)));
    for c in curios.iter().take(3) {
        if c.suggestions.is_empty() {
            println!("curious: what is {:?}?", c.unknown);
        } else {
            println!(
                "curious: {:?} — did you mean {}?",
                c.unknown,
                c.suggestions.join(", ")
            );
        }
        if let Some(note) = &c.history_note {
            println!("curious: {}", note);
        }
    }
    if understand::clause_count(prompt) > 1 {
        println!("curious: that looks like multiple requests — try one at a time");
    }
}

/// Resolve prose into an intent with no model anywhere in the path.
/// Confidence at or above [`LOCAL_CONFIDENCE`] with a known frame
/// executes; anything vaguer refuses with its receipt (run
/// `--understand` to see exactly which slots are missing) instead of
/// running blind. Routing is printed to stderr so the outcome on
/// stdout stays machine-readable.
async fn resolve_prose(project: &str, prompt: &str) -> String {
    use grounding_coder::engine::understand;
    let understood = understand::understand(prompt, Some(std::path::Path::new(project)));
    if understood.confidence >= LOCAL_CONFIDENCE && understood.frame != "unknown" {
        eprintln!(
            "UNDERSTOOD (frame={}, confidence={:.2})",
            understood.frame, understood.confidence
        );
        return serde_json::to_string(&understood.intent).unwrap_or_else(|e| {
            eprintln!("UNDERSTAND FAILED to serialize intent: {}", e);
            std::process::exit(1);
        });
    }
    let mut msg = format!(
        "UNDERSTOOD confidence {:.2} frame={} — too thin to act on; missing: {:?}. No model fallback exists by design.",
        understood.confidence, understood.frame, understood.intent.unknown_requirements,
    );
    let curios = grounding_coder::engine::understand::curiosities(
        prompt,
        Some(std::path::Path::new(project)),
    );
    if let Some(c) = curios.first() {
        msg.push_str(&format!(
            " Curious: what is {:?}?{}",
            c.unknown,
            if c.suggestions.is_empty() {
                String::new()
            } else {
                format!(" Did you mean {}?", c.suggestions.join(", "))
            }
        ));
    }
    eprintln!("{}", msg);
    std::process::exit(2);
}

/// Minimum deterministic confidence to act. Below this the parse is
/// too thin to act on blind (missing verb, kind, or name).
const LOCAL_CONFIDENCE: f64 = 0.75;
