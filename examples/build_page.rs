//! Drive the engine as a library: `cargo run --example build_page --
//! <project-dir> <intent-json> [budget]` — prints the outcome; the project
//! dir holds whatever the engine proved and committed.
use grounding_coder::engine::CodeBot;

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let project = args
        .next()
        .expect("usage: build_page <project-dir> <intent-json> [budget]");
    let intent_path = args
        .next()
        .expect("usage: build_page <project-dir> <intent-json> [budget]");
    let budget: u32 = args.next().and_then(|b| b.parse().ok()).unwrap_or(5);
    let intent_json = std::fs::read_to_string(&intent_path).expect("read intent");
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
