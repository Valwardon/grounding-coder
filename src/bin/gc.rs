use clap::{Parser, Subcommand};
use grounding_coder::{CodeBot, config};

#[derive(Parser)]
#[command(name = "gc", about = "Grounding Coder — deterministic coding agent")]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Chat with the bot: takes natural language, produces verified code
    Chat {
        /// The task request in natural language
        prompt: String,
        /// Target project directory
        #[arg(short, long, default_value = ".")]
        project: String,
        /// Max correction attempts (RetryBudget)
        #[arg(short, long, default_value_t = 5)]
        budget: u32,
    },
    /// Run the bot as a background daemon
    Daemon {
        #[arg(short, long, default_value = ".")]
        project: String,
    },
    /// Show the symbol table for the project
    Symbols {
        #[arg(short, long, default_value = ".")]
        project: String,
    },
    /// Show recipes in the error-correction log
    Recipes,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    rt.block_on(async move {
        match cli.cmd {
            Commands::Chat {
                prompt,
                project,
                budget,
            } => {
                // No model anywhere in this path: the deterministic
                // understander parses, the engine proves. Below-threshold
                // parses refuse with their receipt.
                use grounding_coder::engine::understand;
                let understood =
                    understand::understand(&prompt, Some(std::path::Path::new(&project)));
                if understood.confidence < 0.75 || understood.frame == "unknown" {
                    eprintln!(
                        "UNDERSTOOD confidence {:.2} frame={} — too thin: {:?}",
                        understood.confidence,
                        understood.frame,
                        understood.intent.unknown_requirements,
                    );
                    std::process::exit(2);
                }
                let config = config::load_config_default();
                let intent_json =
                    serde_json::to_string(&understood.intent).expect("intent should serialize");
                let mut bot = CodeBot::new(&project, budget);
                bot.set_progress_listener(std::sync::Arc::new(|ev| eprintln!("[progress] {}", ev)));
                bot.set_github_token(config.github_key.clone());
                match bot.run_task(&intent_json).await {
                    Ok(result) => println!("{}", result),
                    Err(e) => {
                        eprintln!("FAILED: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            Commands::Daemon { project } => {
                println!("Starting grounding-coder daemon in {}...", project);
                let bot = CodeBot::new(&project, 5);
                bot.run_daemon().await;
            }
            Commands::Symbols { project } => {
                let bot = CodeBot::new(&project, 5);
                for (label, symbol) in bot.symbols() {
                    println!("  {} [{}] at {}", label, symbol.kind, symbol.location());
                }
            }
            Commands::Recipes => {
                let bot = CodeBot::new(".", 5);
                for recipe in bot.recipes() {
                    println!("  {}", recipe);
                }
            }
        }
    });
}
