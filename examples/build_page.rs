//! Drive the engine as a library: `cargo run --example build_page --
//! <project-dir> <intent-json>` — prints the outcome; the project dir
//! holds whatever the engine proved and committed.
use grounding_coder::engine::CodeBot;

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let project = args
        .next()
        .expect("usage: build_page <project-dir> <intent-json>");
    let intent_path = args
        .next()
        .expect("usage: build_page <project-dir> <intent-json>");
    let intent_json = std::fs::read_to_string(&intent_path).expect("read intent");
    let mut bot = CodeBot::new(&project, 5);
    match bot.run_task(&intent_json).await {
        Ok(outcome) => println!("{}", outcome),
        Err(e) => {
            eprintln!("ENGINE FAILED: {}", e);
            std::process::exit(1);
        }
    }
}
