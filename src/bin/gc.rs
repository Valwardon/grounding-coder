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
    /// Render the demo scene (person, tower, ground, sky) to a BMP photo
    Render {
        /// Output file path
        #[arg(short, long, default_value = "photo.bmp")]
        out: String,
        /// Frame width in pixels
        #[arg(long, default_value_t = 320)]
        width: u32,
        /// Frame height in pixels
        #[arg(long, default_value_t = 240)]
        height: u32,
        /// Portrait framing: camera in close on the person
        #[arg(long)]
        closeup: bool,
        /// Group framing: three studied individuals, three-point light
        #[arg(long)]
        group: bool,
    },
    /// Source photographic plates: search Commons, require complete
    /// provenance, fetch and decode. Refusals print with reasons.
    Plate {
        /// What to look for
        query: String,
        /// Max plates to ingest
        #[arg(short, long, default_value_t = 3)]
        limit: u32,
        /// Directory for BMP files plus a provenance manifest
        #[arg(short, long, default_value = "plates")]
        out: String,
        /// Where to search: commons (curated metadata) or web (image search)
        #[arg(long, default_value = "commons")]
        source: String,
        /// Downscale longer edge past this many pixels (repo stays lean)
        #[arg(long, default_value_t = 1280)]
        max_dim: u32,
    },
    /// Dream: run the idle learning loop — investigate open questions
    /// and promote only what verifies. Bounded by budget, then sleeps.
    Dream {
        /// Target project directory (holds .grounding/knowledge.jsonl)
        #[arg(short, long, default_value = ".")]
        project: String,
        /// Max investigations this pass
        #[arg(short, long, default_value_t = 5)]
        budget: u32,
        /// Look up unknowns on verified web sources (without it,
        /// hypotheses are held for a later pass)
        #[arg(long)]
        research: bool,
    },
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
                // understander parses, the disposition engine routes,
                // the engine proves. Anything but Execute refuses with
                // its receipt.
                use grounding_coder::engine::understand;
                use grounding_coder::engine::understand::Disposition;
                let understood =
                    understand::understand(&prompt, Some(std::path::Path::new(&project)));
                if understand::disposition(understood.confidence, &understood.frame, &prompt)
                    != Disposition::Execute
                {
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
            Commands::Render {
                out,
                width,
                height,
                closeup,
                group,
            } => {
                use grounding_coder::engine::{scene, vision::Image};
                let width = width.clamp(16, 1920);
                let height = height.clamp(16, 1920);
                let people = closeup || group;
                let (mut img, receipt) = scene::render(&if group {
                    scene::group_scene(width, height)
                } else if closeup {
                    scene::portrait_scene(width, height)
                } else {
                    scene::demo_scene(width, height)
                });
                // Photo finish on people: grade, vignette, seeded grain.
                // Deterministic — same flags twice, byte-identical file.
                if people {
                    img.grade(1.12, 6.0);
                    img.vignette(0.30);
                    img.grain(0xC10C, 5);
                }
                let path = std::path::Path::new(&out);
                match img.save_bmp(path) {
                    Ok(()) => {
                        println!("rendered {} ({}x{})", out, width, height);
                        for (name, count) in &receipt {
                            println!("  {:16} {} px", name, count);
                        }
                        // Read-back proves the file is a photo, not a promise.
                        match Image::load_bmp(path) {
                            Ok(back) => println!(
                                "verified: read back {}x{}, brightness {:.2}",
                                back.width,
                                back.height,
                                back.mean_brightness()
                            ),
                            Err(e) => {
                                eprintln!("READBACK FAILED: {}", e);
                                std::process::exit(1);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("RENDER FAILED: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            Commands::Plate {
                query,
                limit,
                out,
                max_dim,
                source,
            } => {
                use grounding_coder::engine::plates;
                let dir = std::path::Path::new(&out);
                if std::fs::create_dir_all(dir).is_err() {
                    eprintln!("PLATE FAILED: cannot create {}", out);
                    std::process::exit(1);
                }
                let (sourced, refused) = if source == "web" {
                    plates::source_plates_web(&query, limit).await
                } else {
                    plates::source_plates(&query, limit).await
                };
                for r in &refused {
                    println!("refused: {}", r);
                }
                let mut manifest = Vec::new();
                for (i, plate) in sourced.iter().enumerate() {
                    let file = format!("plate-{:02}.bmp", i);
                    let path = dir.join(&file);
                    let img = {
                        let longest = plate.image.width.max(plate.image.height);
                        if longest > max_dim.max(16) {
                            let cap = max_dim.max(16);
                            if plate.image.width >= plate.image.height {
                                plate
                                    .image
                                    .resize(cap, plate.image.height * cap / plate.image.width)
                            } else {
                                plate
                                    .image
                                    .resize(plate.image.width * cap / plate.image.height, cap)
                            }
                        } else {
                            plate.image.clone()
                        }
                    };
                    match img.save_bmp(&path) {
                        Ok(()) => {
                            println!(
                                "plate: {} ({}x{}, {}, {})",
                                file,
                                img.width,
                                img.height,
                                plate.provenance.author,
                                plate.provenance.license
                            );
                            manifest.push(serde_json::json!({
                                "file": file,
                                "source_url": plate.provenance.source_url,
                                "page_url": plate.provenance.page_url,
                                "author": plate.provenance.author,
                                "license": plate.provenance.license,
                                "basis": plate.basis,
                            }));
                        }
                        Err(e) => println!("refused: {}: {}", file, e),
                    }
                }
                let manifest_path = dir.join("provenance.json");
                if std::fs::write(
                    &manifest_path,
                    serde_json::to_string_pretty(&manifest).unwrap_or_default(),
                )
                .is_err()
                {
                    eprintln!("PLATE FAILED: cannot write manifest");
                    std::process::exit(1);
                }
                println!(
                    "plates: {} ingested, {} refused — manifest at {}",
                    manifest.len(),
                    refused.len(),
                    manifest_path.display()
                );
            }
            Commands::Dream {
                project,
                budget,
                research,
            } => {
                use grounding_coder::engine::knowledge;
                let outcomes =
                    knowledge::dream_loop(std::path::Path::new(&project), budget, research).await;
                if outcomes.is_empty() {
                    println!("dream: nothing open — sleeping");
                }
                for o in &outcomes {
                    println!(
                        "dream: {:?} {:?} -> {:?} ({})",
                        o.concept, o.from, o.to, o.note
                    );
                }
                let store = knowledge::KnowledgeStore::open(std::path::Path::new(&project));
                let (verified, open) = store.all().iter().fold((0, 0), |(v, o), i| {
                    use grounding_coder::engine::knowledge::KnowledgeState as S;
                    match i.state {
                        S::Verified | S::Generalized => (v + 1, o),
                        S::Question | S::Unknown | S::Hypothesis | S::Experiment | S::Evidence => {
                            (v, o + 1)
                        }
                        S::Rejected => (v, o),
                    }
                });
                println!("knowledge: {} verified, {} open", verified, open);
            }
        }
    });
}
