use clap::{Parser, Subcommand};
use grounding_coder::{CodeBot, config};

#[derive(Parser)]
#[command(name = "gc", about = "Grounding Coder — deterministic coding agent")]
struct Cli {
    /// Verbose: debug logging plus per-command wall time on stderr.
    #[arg(long, global = true)]
    verbose: bool,
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
    /// Render the demo scene (tower, ground, sky) to a BMP photo.
    /// People come from researched photographs, never from meshes.
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
        /// Where to search: commons (curated metadata), web (image
        /// search), or openimages (CC people boxes, adult-filtered)
        #[arg(long, default_value = "commons")]
        source: String,
        /// Downscale longer edge past this many pixels (repo stays lean)
        #[arg(long, default_value_t = 1920)]
        max_dim: u32,
    },
    /// Compose a photographic portrait from a sourced plate: segment
    /// the subject, frame them on a studio backdrop, finish the photo.
    /// The op log prints; refusals (no subject) exit nonzero.
    Compose {
        /// Source plate BMP (plus its provenance manifest beside it)
        plate: String,
        /// Output BMP path
        #[arg(short, long, default_value = "portrait.bmp")]
        out: String,
        /// Output width
        #[arg(long, default_value_t = 960)]
        width: u32,
        /// Output height
        #[arg(long, default_value_t = 1200)]
        height: u32,
    },
    /// Imagine a photo from prose: parse the brief, research
    /// references, study them, build fresh pixels. One command, full
    /// receipts at every step.
    Imagine {
        /// What to picture ("man holding peace sign")
        prompt: String,
        /// Output BMP path
        #[arg(short, long, default_value = "imagined.bmp")]
        out: String,
        /// Output width
        #[arg(long, default_value_t = 640)]
        width: u32,
        /// Output height
        #[arg(long, default_value_t = 800)]
        height: u32,
    },
    /// Restore a family photo: detect dust specks and scratches,
    /// inpaint them, report every defect. Clean photos report zero.
    Restore {
        /// Source plate BMP
        plate: String,
        /// Output BMP path
        #[arg(short, long, default_value = "restored.bmp")]
        out: String,
        /// Outlier threshold vs neighborhood median (luma units)
        #[arg(long, default_value_t = 40.0)]
        thresh: f64,
    },
    /// Render a genome figure: laughing man, big nose, noodle hat,
    /// waving Canadian flag. Every pixel procedural — the creation
    /// test, runnable.
    Figure {
        /// Output BMP path
        #[arg(short, long, default_value = "figure.bmp")]
        out: String,
        /// Output width
        #[arg(long, default_value_t = 640)]
        width: u32,
        /// Output height
        #[arg(long, default_value_t = 480)]
        height: u32,
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
    let cli = Cli::parse();
    if cli.verbose {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::init();
    }
    let start = std::time::Instant::now();
    let verbose = cli.verbose;
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
            Commands::Render { out, width, height } => {
                use grounding_coder::engine::{scene, vision::Image};
                let width = width.clamp(16, 1920);
                let height = height.clamp(16, 1920);
                let (img, receipt) = scene::render(&scene::demo_scene(width, height));
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
                if source == "openimages" {
                    use grounding_coder::engine::plates::openimages;
                    let cache = dir.join(".oicache");
                    let (found, refused) =
                        openimages::search_openimages(&cache, &query, limit).await;
                    for r in &refused {
                        println!("refused: {}", r);
                    }
                    let mut manifest = Vec::new();
                    for (i, hit) in found.iter().enumerate() {
                        let file = format!("plate-{:02}.bmp", i);
                        let path = dir.join(&file);
                        // Same max_dim discipline as the other sources:
                        // full-res originals bloat the repo for no gain.
                        let img = {
                            let longest = hit.plate.image.width.max(hit.plate.image.height);
                            if longest > max_dim.max(16) {
                                let cap = max_dim.max(16);
                                if hit.plate.image.width >= hit.plate.image.height {
                                    hit.plate.image.resize_smooth(
                                        cap,
                                        hit.plate.image.height * cap / hit.plate.image.width,
                                    )
                                } else {
                                    hit.plate.image.resize_smooth(
                                        hit.plate.image.width * cap / hit.plate.image.height,
                                        cap,
                                    )
                                }
                            } else {
                                hit.plate.image.clone()
                            }
                        };
                        match img.save_bmp(&path) {
                            Ok(()) => {
                                println!(
                                    "plate: {} ({}x{}, box {:?}, {}, {})",
                                    file,
                                    img.width,
                                    img.height,
                                    hit.bbox,
                                    hit.plate.provenance.author,
                                    hit.plate.provenance.license
                                );
                                manifest.push(serde_json::json!({
                                    "file": file,
                                    "source_url": hit.plate.provenance.source_url,
                                    "page_url": hit.plate.provenance.page_url,
                                    "author": hit.plate.provenance.author,
                                    "license": hit.plate.provenance.license,
                                    "basis": hit.plate.basis,
                                    "bbox": hit.bbox,
                                    "label": hit.label,
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
                    return;
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
            Commands::Compose {
                plate,
                out,
                width,
                height,
            } => {
                use grounding_coder::engine::{compose, vision::Image};
                let plate_img = match Image::load_bmp(std::path::Path::new(&plate)) {
                    Ok(img) => img,
                    Err(e) => {
                        eprintln!("COMPOSE FAILED: cannot load plate: {}", e);
                        std::process::exit(1);
                    }
                };
                match compose::compose_portrait(
                    &plate_img,
                    &plate,
                    width.clamp(16, 1920),
                    height.clamp(16, 1920),
                ) {
                    Ok((img, log)) => {
                        println!(
                            "subject: bbox {:?}, {:.3} of frame",
                            log.subject_bbox, log.subject_fraction
                        );
                        for op in &log.ops {
                            println!("  op: {}", op);
                        }
                        match img.save_bmp(std::path::Path::new(&out)) {
                            Ok(()) => println!("composed {} ({}x{})", out, img.width, img.height),
                            Err(e) => {
                                eprintln!("COMPOSE FAILED: {}", e);
                                std::process::exit(1);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("COMPOSE REFUSED: {}", e);
                        std::process::exit(2);
                    }
                }
            }
            Commands::Imagine {
                prompt,
                out,
                width,
                height,
            } => {
                use grounding_coder::engine::imagine;
                match imagine::imagine(&prompt, width.clamp(16, 1920), height.clamp(16, 1920)).await
                {
                    Ok((img, log)) => {
                        for line in &log {
                            println!("imagine: {}", line);
                        }
                        match img.save_bmp(std::path::Path::new(&out)) {
                            Ok(()) => println!("imagined {} ({}x{})", out, img.width, img.height),
                            Err(e) => {
                                eprintln!("IMAGINE FAILED: {}", e);
                                std::process::exit(1);
                            }
                        }
                    }
                    Err(log) => {
                        for line in &log {
                            println!("imagine: {}", line);
                        }
                        eprintln!("IMAGINE REFUSED: no complete photographic subject");
                        std::process::exit(2);
                    }
                }
            }
            Commands::Restore { plate, out, thresh } => {
                use grounding_coder::engine::{restore, vision::Image};
                let plate_img = match Image::load_bmp(std::path::Path::new(&plate)) {
                    Ok(img) => img,
                    Err(e) => {
                        eprintln!("RESTORE FAILED: cannot load plate: {}", e);
                        std::process::exit(1);
                    }
                };
                let (img, report) = restore::restore_plate(&plate_img, thresh, 12);
                println!(
                    "restore: {} specks, {} scratch pixels, {} iterations",
                    report.specks, report.scratch_pixels, report.iterations
                );
                match &report.bbox {
                    Some(b) => println!("restore: defect bbox {:?}", b),
                    None => println!("restore: no defects — photo untouched"),
                }
                match img.save_bmp(std::path::Path::new(&out)) {
                    Ok(()) => println!("restored {} ({}x{})", out, img.width, img.height),
                    Err(e) => {
                        eprintln!("RESTORE FAILED: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            Commands::Figure { out, width, height } => {
                use grounding_coder::engine::{figure, human};
                let genome = human::HumanGenome::with_params(
                    35, 0.5, 0.55, 0.5, 0.95, 0.8, 0.5, 0.6, 0.5, 1.0,
                )
                .expect("test genome validates");
                for line in genome.describe() {
                    println!("genome: {}", line);
                }
                let (mut img, receipt) = grounding_coder::engine::scene::render(
                    &figure::waver_scene(width.clamp(16, 1920), height.clamp(16, 1920), &genome),
                );
                for (name, count) in &receipt {
                    println!("  {:16} {} px", name, count);
                }
                img.grade(1.08, 4.0);
                img.vignette(0.25);
                img.grain(0xF16E, 4);
                match img.save_bmp(std::path::Path::new(&out)) {
                    Ok(()) => println!("figured {} ({}x{})", out, img.width, img.height),
                    Err(e) => {
                        eprintln!("FIGURE FAILED: {}", e);
                        std::process::exit(1);
                    }
                }
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
    if verbose {
        eprintln!("[verbose] wall time: {:.2}s", start.elapsed().as_secs_f64());
    }
}
