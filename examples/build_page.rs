//! Drive the engine as a library:
//! `cargo run --example build_page -- <project-dir> <intent-json> [budget]`
//! `cargo run --example build_page -- <project-dir> --prose "<prompt>" [budget]`
//! prints the outcome; the project dir holds whatever the engine proved
//! and committed.
//!
//! `--prose` wires the translator: an external model turns the prompt
//! into a structured intent (prose in, metadata out — never code), and
//! the deterministic engine takes over from there. Needs a key via
//! `OPENROUTER_API_KEY` or the config file; without one it says so and
//! stops instead of guessing.
use grounding_coder::engine::CodeBot;

const USAGE: &str =
    "usage: build_page <project-dir> (<intent-json> | --prose \"<prompt>\") [budget]";

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let project = args.next().expect(USAGE);
    let second = args.next().expect(USAGE);
    let (intent_json, budget) = if second == "--prose" {
        let prompt = args.next().expect(USAGE);
        let budget: u32 = args.next().and_then(|b| b.parse().ok()).unwrap_or(5);
        (translate_prose(&prompt).await, budget)
    } else {
        let budget: u32 = args.next().and_then(|b| b.parse().ok()).unwrap_or(5);
        let intent_json = std::fs::read_to_string(&second).expect("read intent");
        (intent_json, budget)
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
        Ok(outcome) => println!("{}", outcome),
        Err(e) => {
            eprintln!("ENGINE FAILED: {}", e);
            std::process::exit(1);
        }
    }
}

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
