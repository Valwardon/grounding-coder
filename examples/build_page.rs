//! Drive the engine as a library:
//! `cargo run --example build_page -- <project-dir> <intent-json> [budget]`
//! `cargo run --example build_page -- <project-dir> --prose "<prompt>" [budget]`
//! `cargo run --example build_page -- <project-dir> --understand "<prompt>"`
//! prints the outcome; the project dir holds whatever the engine proved
//! and committed.
//!
//! `--prose` wires the translator: an external model turns the prompt
//! into a structured intent (prose in, metadata out — never code), and
//! the deterministic engine takes over from there. Needs a key via
//! `OPENROUTER_API_KEY` or the config file; without one it says so and
//! stops instead of guessing. Successful prose runs are recorded into
//! the project's translation history (verified outcomes only).
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
}

/// Resolve prose into an intent, model-free when possible.
/// The deterministic parser runs first: confidence at or above
/// [`LOCAL_CONFIDENCE`] with a known frame executes with no model
/// involved at all. Anything vaguer falls back to the external
/// translator (prose in, metadata out — never code). Routing is
/// printed to stderr so the outcome on stdout stays machine-readable.
async fn resolve_prose(project: &str, prompt: &str) -> String {
    use grounding_coder::engine::understand;
    let understood = understand::understand(prompt, Some(std::path::Path::new(project)));
    if understood.confidence >= LOCAL_CONFIDENCE && understood.frame != "unknown" {
        eprintln!(
            "UNDERSTOOD locally (frame={}, confidence={:.2}) — no model involved",
            understood.frame, understood.confidence
        );
        return serde_json::to_string(&understood.intent).unwrap_or_else(|e| {
            eprintln!("UNDERSTAND FAILED to serialize intent: {}", e);
            std::process::exit(1);
        });
    }
    eprintln!(
        "UNDERSTOOD confidence {:.2} — falling back to model translator",
        understood.confidence
    );
    translate_prose(prompt).await
}

/// Minimum deterministic confidence to skip the model. Below this the
/// parse is too thin to act on blind (missing verb, kind, or name),
/// so a model takes the attempt — or fails honestly without a key.
const LOCAL_CONFIDENCE: f64 = 0.75;

/// Translate prose into an intent through the external model, then
/// serialize for the engine. The model output is validated and
/// normalized by the translator (metadata only — planner rejects any
/// `code` downstream anyway). No key means no translation, stated
/// plainly.
async fn translate_prose(prompt: &str) -> String {
    use grounding_coder::llm::{LlmClient, has_api_key, load_config_default};
    let mut config = load_config_default();
    if let Ok(key) = std::env::var("OPENROUTER_API_KEY")
        && !key.trim().is_empty()
    {
        config.openrouter_key = Some(key);
    }
    if !has_api_key(&config) {
        eprintln!(
            "TRANSLATOR UNAVAILABLE: set OPENROUTER_API_KEY or save a key in Settings; prose cannot become an intent without it."
        );
        std::process::exit(2);
    }
    let client = LlmClient::new(config);
    match client.translate(prompt).await {
        Ok(intent) => serde_json::to_string(&intent).unwrap_or_else(|e| {
            eprintln!("TRANSLATOR FAILED to serialize intent: {}", e);
            std::process::exit(1);
        }),
        Err(e) => {
            eprintln!("TRANSLATOR FAILED: {}", e);
            std::process::exit(1);
        }
    }
}
