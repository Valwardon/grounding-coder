use clap::{Parser, Subcommand};
use grounding_coder::{CodeBot, config};

/// Today as `YYYY-MM-DD`, the format the decision and question logs use.
fn date_today() -> String {
    let days = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|s| s.as_secs() / 86400)
        .unwrap_or(0) as i64;
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Days since the Unix epoch to civil date, Howard Hinnant's algorithm
/// (`civil_from_days` shifts by 719468 days internally).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Parse `x0,y0,x1,y1` from an optional CLI flag. Malformed input is a
/// hard CLI error; `None` is an absent flag.
fn parse_box(s: Option<&str>, flag: &str) -> Option<[u32; 4]> {
    match s {
        None => None,
        Some(s) => {
            let parts: Vec<u32> = s
                .split(',')
                .map(|p| {
                    p.trim().parse().unwrap_or_else(|e| {
                        eprintln!("{flag} expects x0,y0,x1,y1 (got {s:?}): {e}");
                        std::process::exit(1);
                    })
                })
                .collect();
            if parts.len() != 4 {
                eprintln!("{flag} expects x0,y0,x1,y1");
                std::process::exit(1);
            }
            Some([parts[0], parts[1], parts[2], parts[3]])
        }
    }
}

/// Parse `aspect=value` CLI answers, ignoring malformed entries.
fn parse_answers(list: &[String]) -> Vec<(String, String)> {
    list.iter()
        .filter_map(|a| {
            a.split_once('=')
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        })
        .collect()
}

/// Fold answered code aspects into a structured intent as facts the
/// planner reads. The answer text is the user's own — never invented.
fn apply_code_answers(
    intent: &mut grounding_coder::engine::tasks::StructuredIntent,
    answers: &[(String, String)],
) {
    for (aspect, value) in answers {
        match aspect.as_str() {
            "language" => {
                if value.as_str() == "no preference" {
                    intent.constraints.push(format!("{aspect}: {value}"));
                } else {
                    intent.language = Some(value.clone());
                }
            }
            "platform" => intent.platform = value.clone(),
            _ => intent.constraints.push(format!("{aspect}: {value}")),
        }
    }
}

/// Write a prose clarification round to the question journal: the
/// durable record of what the request left open (and what closed it).
fn store_prose_round(
    c: &grounding_coder::engine::clarify::Clarify,
    path: &str,
    answers: &[(String, String)],
) {
    use grounding_coder::engine::clarify::Domain;
    if let Some(parent) = std::path::Path::new(path).parent()
        && !parent.as_os_str().is_empty()
    {
        let _ = std::fs::create_dir_all(parent);
    }
    let aspects: Vec<serde_json::Value> = c
        .ambiguities
        .iter()
        .map(|a| {
            serde_json::json!({
                "aspect": a.aspect,
                "question": a.question,
                "options": a.options,
            })
        })
        .collect();
    let domain = match c.domain {
        Domain::Scene => "scene",
        Domain::Code => "code",
        Domain::Unknown => "unknown",
    };
    let record = serde_json::json!({
        "date": date_today(),
        "kind": "prose",
        "request": c.request,
        "domain": domain,
        "aspects": aspects,
        "answers": answers
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<std::collections::HashMap<_, _>>(),
        "resolved": c.resolved,
    });
    let mut line = serde_json::to_string(&record).unwrap_or_default();
    line.push('\n');
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));
}

/// Measure one plate and write its facts file. Shared by single-plate,
/// sweep, and replay so all three measure identically.
fn measure_plate(
    bank: &str,
    entries: &[serde_json::Value],
    idx: u32,
    out_path: &str,
    region: Option<[u32; 4]>,
) -> Result<grounding_coder::engine::measure::VisualFacts, String> {
    use grounding_coder::engine::measure;
    use grounding_coder::engine::vision::Image;
    let entry = entries
        .get(idx as usize)
        .ok_or_else(|| format!("plate {idx} not in provenance ({} entries)", entries.len()))?;
    let meta: measure::PlateMeta = serde_json::from_value(entry.clone())
        .map_err(|e| format!("plate metadata unreadable: {e}"))?;
    if meta.file.is_empty() {
        return Err(format!("plate {idx} has no file: nothing to measure"));
    }
    let img = Image::load_bmp(&std::path::Path::new(bank).join(&meta.file))
        .map_err(|e| format!("plate unreadable: {}", e))?;
    let facts = measure::measure_with_region(&img, meta, region)?;
    if let Some(parent) = std::path::Path::new(out_path)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        let _ = std::fs::create_dir_all(parent);
    }
    let json =
        serde_json::to_string_pretty(&facts).map_err(|e| format!("facts serialize failed: {e}"))?;
    std::fs::write(out_path, format!("{json}\n"))
        .map_err(|e| format!("facts write failed {out_path}: {e}"))?;
    Ok(facts)
}

/// The question journal: every open question about a stimulus is a
/// durable fact. These helpers read, upsert, and write it.
fn question_records(path: &str) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .map(|t| {
            t.lines()
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect()
        })
        .unwrap_or_default()
}

fn write_question_records(path: &str, records: &[serde_json::Value]) {
    if let Some(parent) = std::path::Path::new(path).parent()
        && !parent.as_os_str().is_empty()
    {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut body = records
        .iter()
        .map(|r| serde_json::to_string(r).unwrap_or_default())
        .collect::<Vec<String>>()
        .join("\n");
    if !body.is_empty() {
        body.push('\n');
    }
    let _ = std::fs::write(path, body);
}

/// The one open question a measure refusal carries.
fn measure_question(plate: u32, title: &str, question: String) -> serde_json::Value {
    serde_json::json!({
        "date": date_today(),
        "kind": "measure",
        "plate": plate,
        "title": title,
        "aspect": "subject-region",
        "question": question,
        "answer": null,
        "resolved": false,
    })
}

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
        /// `aspect=value`, repeatable: the answers from a past
        /// `gc clarify` round. When given, a code request must be
        /// fully resolved — refusing to guess rather than plan on
        /// half-answers. Answer fields fold into the structured
        /// intent the planner reads.
        #[arg(long = "clarify-answers")]
        clarify_answers: Vec<String>,
        /// Question-memory file: record the resolved round
        #[arg(long = "clarify-store")]
        clarify_store: Option<String>,
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
    /// With --facts, render the committed measurements of a plate
    /// instead: the subject comes back where the photograph had it.
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
        /// Visual facts JSON from `gc measure` — render those
        /// measurements instead of the demo scene
        #[arg(long)]
        facts: Option<String>,
    },
    /// Measure a researched plate into committed visual facts: the
    /// subject's silhouette, palette and light direction, read out of
    /// the photograph and written as reviewable JSON. No tables, no
    /// per-subject constants — a significance rule against the plate's
    /// own ground picks the cut the plate implies.
    Measure {
        /// Research bank holding provenance.json and plate BMPs
        #[arg(long)]
        bank: String,
        /// Which plate in provenance.json to measure. Omit to sweep the
        /// whole bank in one pass (refusals go into --clarify-out).
        #[arg(long)]
        index: Option<u32>,
        /// Output facts JSON path (single-plate mode)
        #[arg(short, long)]
        out: Option<String>,
        /// Output directory for committed facts (sweep mode)
        #[arg(long)]
        out_dir: Option<String>,
        /// Answer to the previous clarification: which region is the
        /// subject, as `x0,y0,x1,y1`. Attention narrows to that box.
        #[arg(long)]
        hint: Option<String>,
        /// Question-memory file: every refusal and every subject with
        /// an open question is recorded here so the run continues
        /// instead of dying.
        #[arg(long)]
        clarify_out: Option<String>,
    },
    /// Sense what a request leaves open and retire answers
    Clarify {
        /// The request in natural language
        #[arg(default_value = "")]
        prompt: String,
        /// `aspect=value`, repeatable: answers that retire questions
        #[arg(long = "answer")]
        answer: Vec<String>,
        /// Question-memory file: record the round (durable, reviewable)
        #[arg(long = "store")]
        store: Option<String>,
        /// Replay the question-memory file: apply the answers, mark
        /// each measure record resolved, and re-measure the answered
        /// plates (needs --bank and --out-dir).
        #[arg(long)]
        replay: Option<String>,
        /// Research bank (for replay)
        #[arg(long)]
        bank: Option<String>,
        /// Output directory for re-measured facts (for replay)
        #[arg(long)]
        out_dir: Option<String>,
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
    /// Research a picture prompt: parse scene requirements, collect
    /// visual examples per requirement into an evidence store.
    /// Below-minimum collections report INSUFFICIENT, never a model.
    Research {
        /// What to picture ("a man standing on a mountain")
        prompt: String,
        /// Max examples per requirement
        #[arg(short, long, default_value_t = 3)]
        limit: u32,
        /// Directory for plates plus evidence.jsonl
        #[arg(short, long, default_value = "evidence")]
        out: String,
        /// Downscale longer edge past this many pixels (lean evidence)
        #[arg(long, default_value_t = 640)]
        max_dim: u32,
    },
    /// Imagine a photo from prose: research the intent, extract
    /// conditioning, discard raw bytes, generate brand-new pixels
    /// in a single pass. Receipt names model, seed, and novelty.
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
    /// Train the on-device conditional GAN on a provenance-gated
    /// research bank. Weights + receipt land in the weights dir;
    /// picture generation refuses until they exist.
    Train {
        /// Bank dir holding provenance.json and plate BMPs
        bank: String,
        /// Where to write generator weights + training receipt
        #[arg(short, long, default_value = ".grounding/gan")]
        weights: String,
        /// Alternating training steps
        #[arg(long, default_value_t = 200)]
        steps: u64,
    },
    /// Gather a semantic research bank from many prompts at once:
    /// research each query across keyless sources and keep every
    /// plate that has complete provenance AND a real title. Titles
    /// are the GAN's conditioning — a titleless bank trains an
    /// unconditional generator, so a missing title is refused.
    Gather {
        /// Output bank dir (provenance.json + plate BMPs)
        #[arg(short, long, default_value = "bank")]
        out: String,
        /// Comma-separated research queries ("a woman standing, a cat sitting")
        #[arg(long)]
        queries: String,
        /// Max plates kept per query
        #[arg(short = 'n', long, default_value_t = 8)]
        per_query: u32,
        /// Comma-separated keyless sources: commons, web
        #[arg(long, default_value = "commons,web")]
        sources: String,
        /// Downscale longer edge past this many pixels (repo stays lean)
        #[arg(long, default_value_t = 256)]
        max_dim: u32,
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

/// Search project sources for a term: relative .rs paths
/// containing the word, sorted, capped at 5. Ground truth of what
/// the codebase already knows — a codebase hit is reliable.
fn search_codebase(project: &str, query: &str) -> Vec<String> {
    search_files(project, query, &["rs"])
}

/// Search project docs (.md/.toml) for a term. Same contract.
fn search_docs(project: &str, query: &str) -> Vec<String> {
    search_files(project, query, &["md", "toml"])
}

fn search_files(project: &str, query: &str, exts: &[&str]) -> Vec<String> {
    let root = std::path::Path::new(project);
    let q = query.to_lowercase();
    let mut hits = Vec::new();
    let walker = walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            !matches!(
                e.file_name().to_string_lossy().as_ref(),
                ".git" | "target" | "build" | ".gradle" | ".grounding" | "node_modules"
            )
        })
        .flatten();
    for entry in walker {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if !path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| exts.contains(&e))
        {
            continue;
        }
        if let Ok(text) = std::fs::read_to_string(path)
            && text.to_lowercase().contains(&q)
        {
            hits.push(
                path.strip_prefix(root)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_string(),
            );
            if hits.len() >= 5 {
                break;
            }
        }
    }
    hits.sort();
    hits
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
                clarify_answers,
                clarify_store,
            } => {
                // Self-questions first: the bot answers what it is,
                // can do, and cannot do from its capability table —
                // never generated, never guessed.
                if grounding_coder::engine::self_model::is_self_question(&prompt) {
                    print!(
                        "{}",
                        grounding_coder::engine::self_model::answer_self(
                            &prompt,
                            std::path::Path::new(&project)
                        )
                    );
                    return;
                }
                if grounding_coder::engine::picture::is_picture_request(&prompt) {
                    use grounding_coder::engine::picture::picture_from_prompt;
                    let stamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    let out = std::path::PathBuf::from(format!("picture-{}", stamp));
                    match picture_from_prompt(&prompt, &out, 20).await {
                        Ok(outcome) => {
                            for line in &outcome.reply {
                                println!("{}", line);
                            }
                        }
                        Err(e) => {
                            eprintln!("PICTURE FAILED: {}", e);
                            std::process::exit(1);
                        }
                    }
                    return;
                }
                // No model anywhere in this path: the deterministic
                // understander parses, the disposition engine routes,
                // the engine proves. Anything but Execute refuses with
                // its receipt.
                use grounding_coder::engine::understand;
                use grounding_coder::engine::understand::Disposition;
                let mut understood =
                    understand::understand(&prompt, Some(std::path::Path::new(&project)));

                // The coding loop closes here: answers from a past
                // `gc clarify` round become facts the planner reads.
                // A code request engaged with --clarify-answers owes a
                // full answer — never plan on half of one.
                let code_answers = parse_answers(&clarify_answers);
                if !code_answers.is_empty() {
                    use grounding_coder::engine::clarify;
                    use grounding_coder::engine::clarify::Domain;
                    let c = clarify::resolve(&prompt, &code_answers);
                    if matches!(c.domain, Domain::Code) {
                        if !c.resolved {
                            for line in clarify::log_lines(&c) {
                                eprintln!("{}", line);
                            }
                            eprintln!(
                                "REFUSED: a code request owes every answer — resolve the open \
                                 aspects above and re-run with --clarify-answers."
                            );
                            std::process::exit(2);
                        }
                        apply_code_answers(&mut understood.intent, &code_answers);
                        if let Some(path) = &clarify_store {
                            store_prose_round(&c, path, &code_answers);
                            println!("resolved round recorded → {}", path);
                        }
                        let mut folded: Vec<String> = Vec::new();
                        if let Some(lang) = &understood.intent.language {
                            folded.push(format!("language={lang}"));
                        }
                        if !understood.intent.platform.is_empty() {
                            folded.push(format!(
                                "platform={}",
                                understood.intent.platform
                            ));
                        }
                        if !folded.is_empty() {
                            println!("facts folded: {}", folded.join(", "));
                        }
                    }
                }

                if understand::disposition(understood.confidence, &understood.frame, &prompt)
                    != Disposition::Execute
                {
                    // Unknown Resolution: investigate before refusing.
                    // Per-unknown records, typed routes, bounded budget,
                    // trail-kept attempts, structured report on
                    // exhaustion. Exit 2 still signals "did not act".
                    use grounding_coder::engine::research::ResearchOracle;
                    use grounding_coder::engine::unknown::{
                        decide, Decision, Evidence, ResearchAttempt, UnknownRecord, UnknownStatus,
                    };
                    let tasks = understand::research_tasks(&prompt);
                    let trail_path = std::path::Path::new(&project)
                        .join(".grounding")
                        .join("research.jsonl");
                    let trail = grounding_coder::engine::unknown::load_trail(&trail_path);
                    let mut oracle = ResearchOracle::new(6);
                    let mut web_left = 6u32;
                    for task in tasks.iter().take(3) {
                        let mut record = UnknownRecord::open(&task.unknown);
                        // Trail = attempted routes already walked.
                        if let Some((summary, source)) = trail.get(&task.unknown) {
                            record.record_attempt(ResearchAttempt {
                                method: "trail".to_string(),
                                query: task.unknown.clone(),
                                found: true,
                                reliable: false,
                                remaining: "prior trail hit, unverified".to_string(),
                            });
                            record.record_evidence(Evidence {
                                source: format!("trail:{}", source),
                                content: summary.clone(),
                                reliable: false,
                            });
                        }
                        loop {
                            match decide(&record, web_left, false) {
                                Decision::Proceed { evidence } => {
                                    println!(
                                        "RESEARCHED {}: {} [{}]",
                                        task.unknown, evidence.content, evidence.source
                                    );
                                    for f in grounding_coder::engine::unknown::followups(
                                        &record,
                                        &evidence,
                                    ) {
                                        println!("  next question: {}", f);
                                        record.followups.push(f);
                                    }
                                    break;
                                }
                                Decision::Continue { method, query } => {
                                    match method {
                                        "web" => {
                                            web_left = web_left.saturating_sub(1);
                                            match oracle.research_word(&query).await {
                                                Some(def) => {
                                                    let reliable = def
                                                        .source_url
                                                        .contains("wikipedia.org");
                                                    record.record_attempt(ResearchAttempt {
                                                        method: "web".to_string(),
                                                        query: query.clone(),
                                                        found: true,
                                                        reliable,
                                                        remaining: if reliable {
                                                            "answered".to_string()
                                                        } else {
                                                            "unverified snippet".to_string()
                                                        },
                                                    });
                                                    record.record_evidence(Evidence {
                                                        source: def.source_url.clone(),
                                                        content: def.summary.clone(),
                                                        reliable,
                                                    });
                                                }
                                                None => {
                                                    record.record_attempt(ResearchAttempt {
                                                        method: "web".to_string(),
                                                        query: query.clone(),
                                                        found: false,
                                                        reliable: false,
                                                        remaining: "no usable result"
                                                            .to_string(),
                                                    });
                                                }
                                            }
                                        }
                                        "codebase" => {
                                            let hits =
                                                search_codebase(&project, &query);
                                            if hits.is_empty() {
                                                record.record_attempt(ResearchAttempt {
                                                    method: "codebase".to_string(),
                                                    query: query.clone(),
                                                    found: false,
                                                    reliable: false,
                                                    remaining: "not in sources".to_string(),
                                                });
                                            } else {
                                                println!(
                                                    "RESEARCHED {}: in project sources: {}",
                                                    task.unknown,
                                                    hits.join(", ")
                                                );
                                                record.record_attempt(ResearchAttempt {
                                                    method: "codebase".to_string(),
                                                    query: query.clone(),
                                                    found: true,
                                                    reliable: true,
                                                    remaining: "answered".to_string(),
                                                });
                                                record.record_evidence(Evidence {
                                                    source: "codebase".to_string(),
                                                    content: format!(
                                                        "{} appears in {}",
                                                        query,
                                                        hits.join(", ")
                                                    ),
                                                    reliable: true,
                                                });
                                            }
                                        }
                                        "docs" => {
                                            let hits = search_docs(&project, &query);
                                            if hits.is_empty() {
                                                record.record_attempt(ResearchAttempt {
                                                    method: "docs".to_string(),
                                                    query: query.clone(),
                                                    found: false,
                                                    reliable: false,
                                                    remaining: "not in docs".to_string(),
                                                });
                                            } else {
                                                println!(
                                                    "RESEARCHED {}: in project docs: {}",
                                                    task.unknown,
                                                    hits.join(", ")
                                                );
                                                record.record_attempt(ResearchAttempt {
                                                    method: "docs".to_string(),
                                                    query: query.clone(),
                                                    found: true,
                                                    reliable: true,
                                                    remaining: "answered".to_string(),
                                                });
                                                record.record_evidence(Evidence {
                                                    source: "docs".to_string(),
                                                    content: format!(
                                                        "{} documented in {}",
                                                        query,
                                                        hits.join(", ")
                                                    ),
                                                    reliable: true,
                                                });
                                            }
                                        }
                                        _ => {
                                            // "user" with nobody home, or
                                            // anything unrecognized: the
                                            // route itself is exhausted.
                                            record.record_attempt(ResearchAttempt {
                                                method: method.to_string(),
                                                query: query.clone(),
                                                found: false,
                                                reliable: false,
                                                remaining: "route unavailable".to_string(),
                                            });
                                        }
                                    }
                                }
                                Decision::Blocked { report } => {
                                    println!("STATUS: BLOCKED");
                                    println!("OBJECTIVE: {}", report.objective);
                                    for e in &report.established {
                                        println!("ESTABLISHED: {}", e);
                                    }
                                    for i in &report.investigated {
                                        println!("INVESTIGATED: {}", i);
                                    }
                                    for u in &report.unresolved {
                                        println!("UNRESOLVED: {}", u);
                                    }
                                    println!("NEXT POSSIBLE STEP: {}", report.next_step);
                                    println!("NO CHANGES APPLIED.");
                                    break;
                                }
                            }
                        }
                        // Persist the record (status + attempts): the
                        // next turn never re-walks these routes.
                        record.status = match decide(&record, 0, false) {
                            Decision::Proceed { .. } => UnknownStatus::Resolved,
                            Decision::Blocked { .. } => {
                                if matches!(
                                    record.status,
                                    UnknownStatus::Resolved
                                ) {
                                    UnknownStatus::Resolved
                                } else {
                                    UnknownStatus::Exhausted
                                }
                            }
                            Decision::Continue { .. } => UnknownStatus::Exhausted,
                        };
                        let dir = std::path::Path::new(&project).join(".grounding");
                        let _ = std::fs::create_dir_all(&dir);
                        let row = serde_json::json!({
                            "term": record.question,
                            "kind": format!("{:?}", record.kind),
                            "status": format!("{:?}", record.status),
                            "summary": record.known_facts.first().map(|e| e.content.clone()).unwrap_or_default(),
                            "source": record.known_facts.first().map(|e| e.source.clone()).unwrap_or_default(),
                            "attempts": record.attempted_routes.iter().map(|a| format!("{}:{}", a.method, a.query)).collect::<Vec<_>>(),
                        });
                        let mut trail_text =
                            std::fs::read_to_string(&trail_path).unwrap_or_default();
                        trail_text.push_str(&row.to_string());
                        trail_text.push('\n');
                        let _ = std::fs::write(&trail_path, trail_text);
                    }
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
                facts,
            } => {
                use grounding_coder::engine::{scene, vision::Image};
                let width = width.clamp(16, 1920);
                let height = height.clamp(16, 1920);
                let scene = match &facts {
                    Some(path) => {
                        let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
                            eprintln!("FACTS UNREADABLE {path}: {e}");
                            std::process::exit(1);
                        });
                        let facts: grounding_coder::engine::measure::VisualFacts =
                            serde_json::from_str(&text).unwrap_or_else(|e| {
                                eprintln!("FACTS INVALID {path}: {e}");
                                std::process::exit(1);
                            });
                        println!(
                            "rendering measurements of \"{}\" ({})",
                            facts.plate.title, facts.plate.query
                        );
                        println!(
                            "  measured bbox {:?} · {}×{} plate · method {}",
                            facts.subject.bbox,
                            facts.plate_size[0],
                            facts.plate_size[1],
                            facts.subject.method
                        );
                        scene::scene_from_facts(&facts, width, height)
                    }
                    None => scene::demo_scene(width, height),
                };
                let (img, receipt) = scene::render(&scene);
                let path = std::path::Path::new(&out);
                let save = if out.to_lowercase().ends_with(".png") {
                    img.save_png(path)
                } else {
                    img.save_bmp(path)
                };
                match save {
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
            Commands::Measure {
                bank,
                index,
                out,
                out_dir,
                hint,
                clarify_out,
            } => {
                let bank_dir = std::path::Path::new(&bank);
                let prov_path = bank_dir.join("provenance.json");
                let text = std::fs::read_to_string(&prov_path).unwrap_or_else(|e| {
                    eprintln!("NO PROVENANCE {}: {}", prov_path.display(), e);
                    std::process::exit(1);
                });
                let entries: Vec<serde_json::Value> =
                    serde_json::from_str(&text).unwrap_or_else(|e| {
                        eprintln!("PROVENANCE INVALID {}: {}", prov_path.display(), e);
                        std::process::exit(1);
                    });
                let region = parse_box(hint.as_deref(), "--hint");

                fn title_of(e: &serde_json::Value) -> String {
                    e.get("title")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string()
                }

                // One plate: measure, and if it refuses, remember the
                // question instead of dying with it.
                let handle_one = |entries: &[serde_json::Value],
                                  idx: u32,
                                  out_path: &str,
                                  qpath: Option<&str>,
                                  recs: Option<&mut Vec<serde_json::Value>>,
                                  exit_on_refuse: bool|
                 -> Result<(), ()> {
                    let title = title_of(
                        entries
                            .get(idx as usize)
                            .unwrap_or(&serde_json::Value::Null),
                    );
                    println!(
                        "measuring \"{}\" (plate {idx})",
                        if title.is_empty() { "(untitled)" } else { &title }
                    );
                    match measure_plate(&bank, entries, idx, out_path, region) {
                        Ok(facts) => {
                            let s = &facts.subject;
                            println!(
                                "  subject   bbox {:?} · {:.1}% of plate",
                                s.bbox,
                                s.area_frac * 100.0
                            );
                            println!("  method    {}", s.method);
                            println!("committed → {}", out_path);
                            if let (Some(path), Some(recs)) = (qpath, recs)
                                && let Some(rec) = recs.iter_mut().find(|r| {
                                    r["kind"] == "measure"
                                        && r["plate"] == serde_json::json!(idx)
                                        && r["resolved"] == serde_json::json!(false)
                                })
                            {
                                // A plate that measures plainly heals
                                // its own old open question.
                                rec["resolved"] = serde_json::json!(true);
                                rec["resolved_on"] = serde_json::json!(date_today());
                                rec["slug"] = serde_json::json!("measured without an answer");
                                println!("  healed a stale question → {}", path);
                            }
                            Ok(())
                        }
                        Err(e) => {
                            eprintln!("MEASURE FAILED: {}", e);
                            if let Some(path) = qpath {
                                let fresh = measure_question(idx, &title, e);
                                match recs {
                                    Some(recs) => {
                                        if let Some(old) = recs.iter_mut().find(|r| {
                                            r["kind"] == "measure"
                                                && r["plate"] == serde_json::json!(idx)
                                        }) {
                                            *old = fresh;
                                        } else {
                                            recs.push(fresh);
                                        }
                                    }
                                    None => {
                                        let mut all = question_records(path);
                                        if let Some(old) = all.iter_mut().find(|r| {
                                            r["kind"] == "measure"
                                                && r["plate"] == serde_json::json!(idx)
                                        }) {
                                            *old = fresh;
                                        } else {
                                            all.push(fresh);
                                        }
                                        write_question_records(path, &all);
                                    }
                                }
                                println!("question recorded → {}", path);
                            }
                            if exit_on_refuse {
                                std::process::exit(2);
                            }
                            Err(())
                        }
                    }
                };

                if let Some(idx) = index {
                    let Some(out_path) = out else {
                        eprintln!("--out required for a single-plate measure");
                        std::process::exit(1);
                    };
                    let _ = handle_one(
                        &entries,
                        idx,
                        &out_path,
                        clarify_out.as_deref(),
                        None,
                        true,
                    );
                } else {
                    // Sweep the whole bank: committed plates write their
                    // facts, refusals become open questions.
                    let Some(dir) = out_dir else {
                        eprintln!("--out-dir required when sweeping the bank");
                        std::process::exit(1);
                    };
                    let Some(qpath) = clarify_out.as_deref() else {
                        eprintln!("--clarify-out required when sweeping the bank");
                        std::process::exit(1);
                    };
                    let _ = std::fs::create_dir_all(&dir);
                    let mut recs = question_records(qpath);
                    let (mut committed, mut refused) = (0, 0);
                    for idx in 0..entries.len() as u32 {
                        let out_path = format!("{dir}/plate-{idx:04}.json");
                        match handle_one(
                            &entries,
                            idx,
                            &out_path,
                            Some(qpath),
                            Some(&mut recs),
                            false,
                        ) {
                            Ok(()) => committed += 1,
                            Err(()) => refused += 1,
                        }
                    }
                    write_question_records(qpath, &recs);
                    println!(
                        "sweep: {committed} committed, {refused} refused → questions in {}",
                        qpath
                    );
                }
            }
            Commands::Clarify {
                prompt,
                answer,
                store,
                replay,
                bank,
                out_dir,
            } => {
                use grounding_coder::engine::clarify;
                let answers: Vec<(String, String)> = answer
                    .iter()
                    .filter_map(|a| {
                        a.split_once('=')
                            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                    })
                    .collect();

                // Replay: read the journal, apply the answers, mark each
                // measure question resolved, and re-measure the answered
                // plates into the out-dir. Memory, acted on.
                if let Some(replay_path) = replay {
                    let Some(bank) = bank else {
                        eprintln!("--bank required with --replay");
                        std::process::exit(1);
                    };
                    let Some(dir) = out_dir else {
                        eprintln!("--out-dir required with --replay");
                        std::process::exit(1);
                    };
                    let prov_path = std::path::Path::new(&bank).join("provenance.json");
                    let text = std::fs::read_to_string(&prov_path).unwrap_or_else(|e| {
                        eprintln!("NO PROVENANCE {}: {}", prov_path.display(), e);
                        std::process::exit(1);
                    });
                    let entries: Vec<serde_json::Value> =
                        serde_json::from_str(&text).unwrap_or_else(|e| {
                            eprintln!("PROVENANCE INVALID {}: {}", prov_path.display(), e);
                            std::process::exit(1);
                        });
                    let mut recs = question_records(&replay_path);
                    let mut acted = 0;
                    for rec in recs.iter_mut() {
                        if rec["kind"] != "measure" || rec["resolved"] == serde_json::json!(true)
                        {
                            continue;
                        }
                        let Some(aspect) = rec["aspect"].as_str().map(|a| a.to_string()) else {
                            continue;
                        };
                        let Some(plate) = rec["plate"].as_u64() else { continue };
                        // Answers may name a precise plate (`3:subject-region=…`)
                        // or a general aspect (`subject-region=…`); the precise
                        // one wins for that plate.
                        let specific = answers.iter().find(|(k, _)| {
                            *k == format!("{plate}:{aspect}")
                        });
                        let general = answers.iter().find(|(k, _)| k == &aspect);
                        let Some(v) = specific.or(general).map(|(_, v)| v.clone()) else {
                            continue;
                        };
                        rec["answer"] = serde_json::json!(v);
                        rec["resolved"] = serde_json::json!(true);
                        rec["resolved_on"] = serde_json::json!(date_today());
                        let out_path = format!("{dir}/plate-{plate:04}.json");
                        match measure_plate(
                            &bank,
                            &entries,
                            plate as u32,
                            &out_path,
                            parse_box(Some(&v), "--answer"),
                        ) {
                            Ok(_) => {
                                rec["slug"] =
                                    serde_json::json!(format!("committed → {out_path}"));
                                println!("plate {plate}: answered, measured → {}", out_path);
                            }
                            Err(e) => {
                                rec["slug"] = serde_json::json!(format!("still refuses: {e}"));
                                eprintln!("plate {plate}: answered but still refuses: {}", e);
                            }
                        }
                        acted += 1;
                    }
                    write_question_records(&replay_path, &recs);
                    println!(
                        "replay: {acted} questions acted on ({} records in {replay_path})",
                        recs.len()
                    );
                    return;
                }

                let c = clarify::resolve(&prompt, &answers);
                if let Some(path) = store {
                    if let Some(parent) = std::path::Path::new(&path).parent()
                        && !parent.as_os_str().is_empty()
                    {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let aspects: Vec<serde_json::Value> = c
                        .ambiguities
                        .iter()
                        .map(|a| {
                            serde_json::json!({
                                "aspect": a.aspect,
                                "question": a.question,
                                "options": a.options,
                            })
                        })
                        .collect();
                    let domain = match c.domain {
                        clarify::Domain::Scene => "scene",
                        clarify::Domain::Code => "code",
                        clarify::Domain::Unknown => "unknown",
                    };
                    let record = serde_json::json!({
                        "date": date_today(),
                        "kind": "prose",
                        "request": prompt,
                        "domain": domain,
                        "aspects": aspects,
                        "answers": answers
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect::<std::collections::HashMap<_, _>>(),
                        "resolved": c.resolved,
                    });
                    let mut line = serde_json::to_string(&record).unwrap_or_default();
                    line.push('\n');
                    let _ = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&path)
                        .map(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));
                    println!("round recorded → {}", path);
                }
                for line in clarify::log_lines(&c) {
                    println!("{line}");
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
                                    "title": hit.plate.title,
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
                                "title": plate.title,
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
            Commands::Research {
                prompt,
                limit,
                out,
                max_dim,
            } => {
                use grounding_coder::engine::{evidence, plates, scene_intent};
                let spec = scene_intent::parse_scene(&prompt);
                let plan = scene_intent::plan_research(&spec);
                let mut store = evidence::EvidenceStore::new();
                store.require_from_spec(&spec, &plan);
                println!("requirements: {}", store.requirements.len());
                for r in &store.requirements {
                    println!("  {} [{} queries]", r.id, r.research_queries.len());
                }
                let dir = std::path::Path::new(&out);
                if std::fs::create_dir_all(dir).is_err() {
                    eprintln!("RESEARCH FAILED: cannot create {}", out);
                    std::process::exit(1);
                }
                let per = limit.clamp(1, 100) as usize;
                let mut n = 0u32;
                let ids: Vec<String> = store.requirements.iter().map(|r| r.id.clone()).collect();
                for id in &ids {
                    let queries = store
                        .get(id)
                        .map(|r| r.research_queries.clone())
                        .unwrap_or_default();
                    for q in &queries {
                        let have = store.get(id).map(|r| r.examples.len()).unwrap_or(0);
                        if have >= per {
                            break;
                        }
                        let (sourced, refused) =
                            plates::source_plates(q, (per - have) as u32).await;
                        for r in &refused {
                            println!("refused: {}: {}", id, r);
                        }
                        for plate in sourced {
                            let file = format!("plate-{:02}.bmp", n);
                            n += 1;
                            let path = dir.join(&file);
                            let img = {
                                let longest = plate.image.width.max(plate.image.height);
                                if longest > max_dim.max(16) {
                                    let cap = max_dim.max(16);
                                    if plate.image.width >= plate.image.height {
                                        plate.image.resize(
                                            cap,
                                            plate.image.height * cap / plate.image.width,
                                        )
                                    } else {
                                        plate.image.resize(
                                            plate.image.width * cap / plate.image.height,
                                            cap,
                                        )
                                    }
                                } else {
                                    plate.image.clone()
                                }
                            };
                            if img.save_bmp(&path).is_err() {
                                println!("refused: {}: cannot save {}", id, file);
                                continue;
                            }
                            let ex = evidence::example_from_plate(
                                &path.to_string_lossy(),
                                &plate.title,
                                &plate.provenance.page_url,
                                &plate.provenance.license,
                                &plate.basis,
                                &img,
                            );
                            if store.add_example(id, ex).is_err() {
                                break;
                            }
                            println!("example: {} <- {} ({})", id, file, plate.provenance.license);
                        }
                    }
                    match store.sufficiency(id) {
                        Ok(evidence::Sufficiency::Insufficient { n, need }) => {
                            println!("{}: INSUFFICIENT ({}/{})", id, n, need)
                        }
                        Ok(evidence::Sufficiency::Collecting { n }) => {
                            println!("{}: collecting ({})", id, n)
                        }
                        Ok(evidence::Sufficiency::Sufficient { n }) => {
                            println!("{}: SUFFICIENT ({})", id, n)
                        }
                        Err(e) => println!("{}: {}", id, e),
                    }
                }
                match evidence::save_jsonl(&store, &dir.join("evidence.jsonl")) {
                    Ok(()) => println!(
                        "evidence: {} requirements at {}/evidence.jsonl",
                        store.requirements.len(),
                        out
                    ),
                    Err(e) => {
                        eprintln!("RESEARCH FAILED: {}", e);
                        std::process::exit(1);
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
                    Ok((photo, log)) => {
                        for line in &log {
                            println!("imagine: {}", line);
                        }
                        println!(
                            "imagine: generated via {} (seed {}, novelty {:.4})",
                            photo.model, photo.seed, photo.min_novelty_vs_donors
                        );
                        let save = if out.to_lowercase().ends_with(".png") {
                            photo.image.save_png(std::path::Path::new(&out))
                        } else {
                            photo.image.save_bmp(std::path::Path::new(&out))
                        };
                        match save {
                            Ok(()) => println!(
                                "imagined {} ({}x{})",
                                out, photo.image.width, photo.image.height
                            ),
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
                        eprintln!("IMAGINE REFUSED: no licensed photograph found (see log)");
                        std::process::exit(2);
                    }
                }
            }
            Commands::Train {
                bank,
                weights,
                steps,
            } => {
                use grounding_coder::engine::gan;
                match gan::train(
                    std::path::Path::new(&bank),
                    std::path::Path::new(&weights),
                    steps,
                ) {
                    Ok(r) => {
                        println!(
                            "train: {} step(s) over {} example(s), {} refused",
                            r.steps,
                            r.examples,
                            r.refused.len()
                        );
                        for (d, g) in &r.loss_trail {
                            println!("train: d_loss {:.5} g_loss {:.5}", d, g);
                        }
                        println!("train: weights + receipt written to {}", weights);
                    }
                    Err(e) => {
                        eprintln!("TRAIN FAILED: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            Commands::Gather {
                out,
                queries,
                per_query,
                sources,
                max_dim,
            } => {
                use grounding_coder::engine::plates;
                let dir = std::path::Path::new(&out);
                if std::fs::create_dir_all(dir).is_err() {
                    eprintln!("GATHER FAILED: cannot create {}", out);
                    std::process::exit(1);
                }
                let want: Vec<String> = sources
                    .split(',')
                    .map(|s| s.trim().to_lowercase())
                    .filter(|s| !s.is_empty())
                    .collect();
                let queries: Vec<String> = queries
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                let mut manifest = Vec::new();
                let mut seen = std::collections::HashSet::new();
                let mut kept = 0usize;
                let mut refused = 0usize;
                for q in &queries {
                    let mut found = Vec::new();
                    if want.iter().any(|s| s == "commons") {
                        let (mut c, r) = plates::source_plates(q, per_query).await;
                        for x in &r {
                            println!("refused [{}]: {}", q, x);
                            refused += 1;
                        }
                        found.append(&mut c);
                    }
                    if want.iter().any(|s| s == "web") {
                        let (mut w, r) = plates::source_plates_web(q, per_query).await;
                        for x in &r {
                            println!("refused [{}]: {}", q, x);
                            refused += 1;
                        }
                        found.append(&mut w);
                    }
                    for plate in found.into_iter().take(per_query as usize) {
                        if !seen.insert(plate.provenance.source_url.clone()) {
                            continue;
                        }
                        if plate.title.trim().is_empty() {
                            refused += 1;
                            continue;
                        }
                        let cap = max_dim.max(16);
                        let img = if plate.image.width.max(plate.image.height) > cap {
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
                        };
                        let file = format!("plate-{:04}.bmp", kept);
                        let path = dir.join(&file);
                        match img.save_bmp(&path) {
                            Ok(()) => {
                                println!(
                                    "plate: {} <- {:?} ({}x{}, {}, {})",
                                    file,
                                    q,
                                    img.width,
                                    img.height,
                                    plate.provenance.author,
                                    plate.provenance.license
                                );
                                manifest.push(serde_json::json!({
                                    "file": file,
                                    "title": plate.title,
                                    "query": q,
                                    "source_url": plate.provenance.source_url,
                                    "page_url": plate.provenance.page_url,
                                    "author": plate.provenance.author,
                                    "license": plate.provenance.license,
                                    "basis": plate.basis,
                                }));
                                kept += 1;
                            }
                            Err(e) => {
                                println!("refused [{}]: {}: {}", q, file, e);
                                refused += 1;
                            }
                        }
                    }
                }
                let manifest_path = dir.join("provenance.json");
                if std::fs::write(
                    &manifest_path,
                    serde_json::to_string_pretty(&manifest).unwrap_or_default(),
                )
                .is_err()
                {
                    eprintln!("GATHER FAILED: cannot write manifest");
                    std::process::exit(1);
                }
                println!(
                    "gather: {} plate(s) kept, {} refused — manifest at {}",
                    kept,
                    refused,
                    manifest_path.display()
                );
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
