pub mod arena;
pub mod budget;
pub mod corrector;
pub mod error;
pub mod lang;
pub mod plan;
pub mod recipes;
pub mod research;
pub mod symbols;
pub mod synthesize;
pub mod tasks;
pub mod verifier;
pub mod writer;

pub use crate::oracle::{
    CodePattern, CodeSymbolInfo, KnowledgeAdapter, KnowledgeOracle, KnowledgeResult, VerifiedFact,
};
pub use arena::{CodeArena, CodeSymbol, SymbolEdge, SymbolId, SymbolKind, SymbolRelation};
pub use budget::RetryBudget;
pub use corrector::CorrectionPipeline;
pub use error::{CompileError, ErrorClassifier, ErrorKind};
pub use plan::{EditPlan, Evidence, SourceEdit, TaskState};
pub use recipes::RecipeLog;
pub use research::ResearchOracle;
pub use symbols::{CodeDef, SymbolTable};
pub use synthesize::{ContractCase, SynthRequest, Synthesizer};
pub use tasks::{
    EditIntent, FieldDef, IntentAction, IntentDefinition, IntentTest, MethodDef, SectionDef,
    StructuredIntent, SubTask, TaskDecomposer, TaskKind, TestCase,
};
pub use verifier::CodeVerifier;
pub use writer::{CodeWriter, FileSnapshot};

use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::Arc;

/// The deterministic coding engine — zero LLM, zero guessing.
/// The LLM (if used) is an unverified layer that only translates
/// natural language into structured intents. This engine takes
/// over from there: it resolves symbols, writes code, verifies
/// with the compiler, and self-corrects via recipe lookup.
pub struct CodeBot {
    project_dir: PathBuf,
    pub arena: Arc<RwLock<CodeArena>>,
    symbol_table: SymbolTable,
    budget: RetryBudget,
    recipes: RecipeLog,
    corrector: CorrectionPipeline,
    verifier: CodeVerifier,
    decomposer: TaskDecomposer,
    writer: CodeWriter,
    /// Research oracle — fetches verified definitions from docs.rs, Android SDK docs, etc.
    /// The bot's "curiosity harvester" repurposed: it can discover new symbols
    /// from authoritative sources, verify them with the compiler, and cache them.
    researcher: ResearchOracle,
    /// Live progress sink (chat UI, CLI). `None` = silent.
    progress: Option<ProgressCallback>,
    /// GitHub token for the publish actor. `None` = publishing disabled;
    /// the engine never invents credentials.
    github_token: Option<String>,
}

/// Result of a completed coding task.
#[derive(Debug, Clone)]
pub struct TaskResult {
    pub success: bool,
    pub changes: Vec<String>,
    pub errors_fixed: u32,
    pub budget_used: u32,
    pub recipes_learned: u32,
    pub message: String,
}

impl std::fmt::Display for TaskResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Result: {}",
            if self.success { "SUCCESS" } else { "FAILURE" }
        )?;
        writeln!(f, "Message: {}", self.message)?;
        writeln!(f, "Changes: {} files", self.changes.len())?;
        writeln!(f, "Errors fixed: {}", self.errors_fixed)?;
        writeln!(f, "Budget used: {}", self.budget_used)?;
        write!(f, "Recipes learned: {}", self.recipes_learned)?;
        std::fmt::Result::Ok(())
    }
}

/// The outcome of a task execution.
#[derive(Debug, Clone)]
pub enum AgentOutcome {
    /// Task completed successfully
    Success(TaskResult),
    /// Changes applied and syntax-verified, but no full type/test oracle
    /// ran on this host (e.g. Android without cargo). Bytes placed with
    /// evidence — honestly labeled, never a claimed full verification.
    Partial { result: TaskResult, caveat: String },
    /// Task is blocked - no safe fix available
    Blocked {
        reason: BlockReason,
        diagnostics: Vec<CompileError>,
    },
    /// Task failed with a reason
    Failed { reason: String },
}

/// Reasons a task can be blocked.
#[derive(Debug, Clone)]
pub enum BlockReason {
    UnknownSymbol,
    NoVerifiedPattern,
    UnsupportedAction,
    NoSafeFix,
    VerificationToolUnavailable,
    StaleEdit,
}

impl std::fmt::Display for BlockReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            BlockReason::UnknownSymbol => "UnknownSymbol",
            BlockReason::NoVerifiedPattern => "NoVerifiedPattern",
            BlockReason::UnsupportedAction => "UnsupportedAction",
            BlockReason::NoSafeFix => "NoSafeFix",
            BlockReason::VerificationToolUnavailable => "VerificationToolUnavailable",
            BlockReason::StaleEdit => "StaleEdit",
        };
        write!(f, "{}", s)
    }
}

impl std::fmt::Display for AgentOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentOutcome::Success(r) => write!(f, "{}", r),
            AgentOutcome::Partial { result, caveat } => {
                writeln!(f, "Result: PARTIAL")?;
                write!(f, "{}", result)?;
                write!(f, "\nCaveat: {}", caveat)
            }
            AgentOutcome::Blocked {
                reason,
                diagnostics,
            } => {
                writeln!(f, "Result: BLOCKED ({})", reason)?;
                for d in diagnostics {
                    writeln!(f, "  {} {}:{} {}", d.code, d.file, d.line, d.message)?;
                }
                Ok(())
            }
            AgentOutcome::Failed { reason } => write!(f, "Result: FAILURE\nMessage: {}", reason),
        }
    }
}

/// Inject a page definition when the intent asks for a webpage whose
/// target file does not exist yet. Without this, valid-schema model output
/// with stub actions dies on the read — the exact reported PLAN_ERROR.
/// No-op when a page define exists, the target isn't `.html`, or the file
/// already exists (edit flows stay untouched).
fn ensure_page_create(intent: &mut StructuredIntent, project_dir: &std::path::Path) {
    if intent.define.iter().flatten().any(|d| d.kind == "page") {
        return;
    }
    let target = match intent.file.clone() {
        Some(f) => f.trim_start_matches("./").to_string(),
        None => return,
    };
    if !(target.ends_with(".html") || target.ends_with(".htm")) {
        return;
    }
    if project_dir.join(&target).exists() {
        return;
    }
    let goal = intent.goal.clone();
    let lower = goal.to_lowercase();
    const TRIGGERS: &[&str] = &[
        "html",
        "website",
        "webpage",
        "homepage",
        "landing page",
        "web site",
        "web page",
        "site",
        "page",
    ];
    if !TRIGGERS.iter().any(|t| lower.contains(t)) {
        return;
    }
    let title = page_title_from_goal(&goal);
    intent.define.push(Some(IntentDefinition {
        name: "homepage".to_string(),
        kind: "page".to_string(),
        references: Vec::new(),
        signature: None,
        cases: Vec::new(),
        fields: Vec::new(),
        methods: Vec::new(),
        title: Some(title),
        sections: vec![SectionDef {
            heading: "Welcome".to_string(),
            body: goal.chars().take(200).collect(),
        }],
        footer: Some(String::new()),
    }));
    // Stub tasks cannot build a missing page — the definition owns it now.
    intent.actions.clear();
    intent.imports.clear();
    intent.test.clear();
    log::info!("Page create injected for {}", target);
}

/// Title from "for X" / "about X" by words (no byte arithmetic across
/// case mappings — see the translator for why that matters).
fn page_title_from_goal(goal: &str) -> String {
    let words: Vec<&str> = goal.split_whitespace().collect();
    for (i, w) in words.iter().enumerate() {
        if w.eq_ignore_ascii_case("for") || w.eq_ignore_ascii_case("about") {
            let tail: Vec<&str> = words
                .iter()
                .skip(i + 1)
                .filter(|w| !["a", "an", "the"].contains(&w.to_lowercase().as_str()))
                .take(6)
                .cloned()
                .collect();
            if !tail.is_empty() {
                let title = tail
                    .join(" ")
                    .trim_end_matches(['.', ',', '!', '?'])
                    .to_string();
                return capitalize_page_title(&title);
            }
        }
    }
    capitalize_page_title(
        &words
            .iter()
            .filter(|w| {
                ![
                    "build", "create", "make", "write", "generate", "a", "an", "the", "simple",
                    "new",
                ]
                .contains(&w.to_lowercase().as_str())
            })
            .take(6)
            .cloned()
            .collect::<Vec<_>>()
            .join(" "),
    )
}

fn capitalize_page_title(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => "Homepage".to_string(),
    }
}

/// One step of the run, pushed live to whoever is watching (chat UI,
/// CLI stderr). This is the feedback loop: a silent worker is
/// indistinguishable from a dead one, so the engine narrates itself.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// A task started: id, description, kind.
    Task { id: u64, desc: String, kind: String },
    /// A pipeline stage within a task: planned/applied/verify/repair/...
    Stage {
        id: u64,
        stage: &'static str,
        detail: String,
    },
}

/// Push-based progress sink. `None` (default) = silent, for tests.
pub type ProgressCallback = std::sync::Arc<dyn Fn(ProgressEvent) + Send + Sync>;

impl std::fmt::Display for ProgressEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProgressEvent::Task { id, desc, kind } => {
                write!(f, "▸ task {} [{}] {}", id, kind, desc)
            }
            ProgressEvent::Stage { id, stage, detail } => {
                if detail.is_empty() {
                    write!(f, "  [{}] {}", id, stage)
                } else {
                    write!(f, "  [{}] {} — {}", id, stage, detail)
                }
            }
        }
    }
}

/// A terminal honest refusal with one diagnostic.
fn blocked_outcome(reason: BlockReason, code: &str, message: String) -> AgentOutcome {
    AgentOutcome::Blocked {
        reason,
        diagnostics: vec![CompileError {
            code: code.to_string(),
            message,
            file: String::new(),
            line: 0,
            col: 0,
            suggestion: None,
            source_line: None,
            kind: crate::engine::error::ErrorKind::Other,
        }],
    }
}

impl CodeBot {
    pub fn new(project_dir: &str, max_retries: u32) -> Self {
        let project_path = PathBuf::from(project_dir);
        let arena = Arc::new(RwLock::new(CodeArena::new()));
        let symbol_table = SymbolTable::new();
        let budget = RetryBudget::new(max_retries as f64);
        let recipes = RecipeLog::new();
        let corrector = CorrectionPipeline::new(recipes.clone());
        let verifier = CodeVerifier::new(project_path.clone());
        let decomposer = TaskDecomposer::new();
        let writer = CodeWriter::new(project_path.clone());
        let researcher = ResearchOracle::new(20);

        let mut bot = CodeBot {
            project_dir: project_path,
            arena,
            symbol_table,
            budget,
            recipes,
            corrector,
            verifier,
            decomposer,
            writer,
            researcher,
            progress: None,
            github_token: None,
        };

        // Bootstrap: scan the project and build the code symbol graph.
        bot.bootstrap();
        bot
    }

    /// Attach a GitHub token for the publish actor. Without it, publish
    /// requests report honestly instead of failing obscurely.
    pub fn set_github_token(&mut self, token: Option<String>) {
        self.github_token = token.filter(|t| !t.trim().is_empty());
    }

    /// Attach a live progress listener (chat UI, CLI stderr).
    pub fn set_progress_listener(&mut self, cb: ProgressCallback) {
        self.progress = Some(cb);
    }

    fn emit(&self, ev: ProgressEvent) {
        if let Some(cb) = &self.progress {
            cb(ev);
        }
    }

    /// Scan the project's source files and populate the CodeArena
    /// with symbol nodes + edges. This is the bot's "perfect knowledge"
    /// — it reads the actual codebase, not an LLM's hallucination.
    fn bootstrap(&mut self) {
        let sources = self.writer.scan_project(&self.project_dir);
        let mut arena = self.arena.write();
        for symbol in sources {
            arena.insert(symbol);
        }
    }

    /// Run a structured intent through the deterministic pipeline.
    ///
    /// This is the core loop, mirroring grounded's:
    ///   GapDetector → CuriosityBudget → DefinitionResolver → VerificationLoop
    ///
    /// Here it maps to:
    ///   TaskDecomposer → RetryBudget → SymbolBinder → CodeVerifier → CorrectionPipeline
    ///
    /// The critical invariant: every task follows the transactional pattern:
    ///   snapshot → apply → verify → commit OR rollback
    pub async fn run_task(&mut self, intent_json: &str) -> Result<AgentOutcome, String> {
        // Step 0: Parse intent and research unknown symbols
        let mut intent: StructuredIntent = serde_json::from_str(intent_json)
            .map_err(|e| format!("Failed to parse intent: {}", e))?;

        // A webpage request whose target does not exist yet cannot be served
        // by stub tasks (they fail the read). Inject a page definition from
        // the goal so the page family creates it — content slots carry the
        // goal's own words, never invented copy. Existing files are untouched.
        ensure_page_create(&mut intent, &self.project_dir);

        // Collect all referenced symbols that the bot doesn't know yet
        let mut unknown_symbols = Vec::new();
        for sym in &intent.references {
            if !self.symbol_table.is_known(sym) {
                unknown_symbols.push(sym.clone());
            }
        }
        for d in intent.define.iter().flatten() {
            if !self.symbol_table.is_known(&d.name) {
                unknown_symbols.push(d.name.clone());
            }
            for r in &d.references {
                if !self.symbol_table.is_known(r) {
                    unknown_symbols.push(r.clone());
                }
            }
        }
        for sym in unknown_symbols {
            let lang = intent
                .language
                .clone()
                .unwrap_or_else(|| "rust".to_string());
            if let Some(found) = self.researcher.research(&sym, &lang).await {
                log::info!("Resolving unknown symbol {} via {}", sym, found.source_url);
            } else {
                log::warn!(
                    "Could not research symbol: {} — no verified source found",
                    sym
                );
            }
        }

        // Step 0b: registry-verified external crates (Rust only). For every
        // non-std import root, ask crates.io. A hit pins name → version in
        // the symbol table (evidence, never a guess) and later synthesizes
        // an EnsureDep task for Cargo.toml. No is_known skip: bundled
        // knowledge must not suppress the manifest dep, or the import
        // would resolve nowhere at check time. Bounded: 8 lookups per run.
        if intent.language.as_deref().unwrap_or("rust") == "rust" {
            let mut roots = Vec::new();
            for imp in &intent.imports {
                let seg = imp.split([':', '.']).next().unwrap_or("").to_lowercase();
                if seg.is_empty()
                    || ["std", "core", "alloc", "crate", "self", "super"].contains(&seg.as_str())
                    || roots.contains(&seg)
                {
                    continue;
                }
                roots.push(seg);
            }
            for root in roots.into_iter().take(8) {
                match crate::engine::research::verify_crate(&root).await {
                    Some(version) => {
                        log::info!("Verified external crate {} v{}", root, version);
                        self.emit(ProgressEvent::Stage {
                            id: 0,
                            stage: "crate-verified",
                            detail: format!("{} v{}", root, version),
                        });
                        self.symbol_table.index_crate(&root, &version);
                    }
                    None => log::warn!("No registry entry for {}", root),
                }
            }
        }

        // Step 1: TaskDecomposer breaks the intent into sub-tasks.
        // Re-serialize: repairs above (page injection) mutate the intent,
        // and the decomposer must see the repaired shape, not the raw input.
        let intent_json = serde_json::to_string(&intent)
            .map_err(|e| format!("Failed to re-serialize intent: {}", e))?;
        let tasks = self.decomposer.decompose(&intent_json, &self.arena.read());

        let mut changes = Vec::new();
        let mut errors_fixed = 0u32;
        let mut budget_used = 0u32;
        let mut recipes_learned = 0u32;
        // Tasks verified syntax-only (no toolchain) land here: applied
        // with evidence, honestly labeled instead of claimed verified.
        let mut partial_notes: Vec<String> = Vec::new();
        let partial_caveat =
            "syntax-only verification (no toolchain on this host): types and behavior unproven"
                .to_string();

        // Research tasks were already attempted in Step 0 — they touch no
        // files, so filter them here and let concrete edits proceed.
        let mut active = Vec::new();
        for task in tasks {
            log::info!(
                "TASK\n  {}\n  kind={} id={}",
                task.description,
                task.kind,
                task.id
            );
            log::info!(
                "GROUNDING\n  required={:?}\n  targets={:?}\n  payload={}",
                task.required_symbols,
                task.target_symbols,
                task.payload
            );
            if matches!(
                task.kind,
                crate::engine::tasks::TaskKind::ResearchRequirements
                    | crate::engine::tasks::TaskKind::ResolveSymbol
                    | crate::engine::tasks::TaskKind::CreateProjectManifest
            ) {
                log::info!(
                    "RESULT\n  SKIPPED research task {}\n  reason=already_attempted_in_step_0",
                    task.id
                );
                self.emit(ProgressEvent::Stage {
                    id: task.id,
                    stage: "skipped",
                    detail: "research already attempted".to_string(),
                });
                continue;
            }
            self.emit(ProgressEvent::Task {
                id: task.id,
                desc: task.description.clone(),
                kind: task.kind.to_string(),
            });
            active.push(task);
        }

        // Verified external crates become EnsureDep tasks FIRST, so the
        // manifest lands before any code that needs it.
        if intent.language.as_deref().unwrap_or("rust") == "rust"
            && self.project_dir.join("Cargo.toml").exists()
        {
            let mut dep_roots = Vec::new();
            for imp in &intent.imports {
                let seg = imp.split([':', '.']).next().unwrap_or("").to_lowercase();
                if seg.is_empty()
                    || ["std", "core", "alloc", "crate", "self", "super"].contains(&seg.as_str())
                    || dep_roots.contains(&seg)
                {
                    continue;
                }
                dep_roots.push(seg);
            }
            let mut deps = Vec::new();
            for root in dep_roots {
                if let Some(version) = self.symbol_table.crate_version(&root) {
                    let mut t = SubTask::new(
                        crate::engine::tasks::TaskKind::EnsureDep,
                        format!("Ensure dep {} v{}", root, version),
                        serde_json::json!({"crate": root, "version": version}),
                        "external_crate".to_string(),
                    );
                    t.target_symbols.push("Cargo.toml".to_string());
                    self.emit(ProgressEvent::Task {
                        id: t.id,
                        desc: t.description.clone(),
                        kind: t.kind.to_string(),
                    });
                    deps.push(t);
                }
            }
            deps.extend(active);
            active = deps;
        }

        // Phase A: prove EVERYTHING plannable before touching disk, and
        // collect the file list. Any planning failure blocks with nothing
        // applied — atomicity by construction.
        //
        // Plans hold byte offsets against pristine files, so they are used
        // here only for the file list. Phase C re-plans each task fresh
        // against current disk state (planning is deterministic).
        let mut all_files = self.writer.project_source_files();
        for task in &active {
            let files: Vec<PathBuf> =
                if task.kind == crate::engine::tasks::TaskKind::SynthesizeFunction {
                    match self.writer.plan_synthesis(task) {
                        Ok(plans) => plans
                            .iter()
                            .flat_map(|p| p.edits.iter().map(|e| e.file.clone()))
                            .collect(),
                        Err(e) => {
                            self.emit(ProgressEvent::Stage {
                                id: task.id,
                                stage: "plan-blocked",
                                detail: e.clone(),
                            });
                            return Ok(blocked_outcome(
                                BlockReason::NoVerifiedPattern,
                                "SYNTH_ERROR",
                                e,
                            ));
                        }
                    }
                } else {
                    match self.writer.plan(task, &self.symbol_table) {
                        Ok(plan) => plan.edits.iter().map(|e| e.file.clone()).collect(),
                        Err(e) => {
                            self.emit(ProgressEvent::Stage {
                                id: task.id,
                                stage: "plan-blocked",
                                detail: e.clone(),
                            });
                            return Ok(blocked_outcome(
                                BlockReason::NoVerifiedPattern,
                                "PLAN_ERROR",
                                e,
                            ));
                        }
                    }
                };
            all_files.extend(files);
        }

        // Phase B: one global snapshot over every planned file plus all
        // project sources (corrections can't escape the transaction).
        let global = match self.writer.snapshot_many(&all_files) {
            Ok(s) => s,
            Err(e) => {
                return Ok(AgentOutcome::Failed {
                    reason: format!("Failed to create snapshot: {}", e),
                });
            }
        };

        // Phase C: execute in order, re-planning each task fresh so byte
        // offsets always match current disk state. ANY terminal failure
        // rolls back the whole run — a multi-file feature either lands
        // complete or not at all.
        for task in active {
            // Synthesis tasks try each candidate body in order —
            // first fully-verified candidate commits, exhaustion blocks.
            if task.kind == crate::engine::tasks::TaskKind::SynthesizeFunction {
                let plans = match self.writer.plan_synthesis(&task) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = self.writer.rollback(&global);
                        return Ok(blocked_outcome(
                            BlockReason::NoVerifiedPattern,
                            "SYNTH_ERROR",
                            e,
                        ));
                    }
                };
                match self.run_synthesis_plans(&plans).await {
                    Ok(AgentOutcome::Success(r)) => {
                        changes.extend(r.changes.clone());
                        errors_fixed += r.errors_fixed;
                        budget_used += r.budget_used;
                        recipes_learned += r.recipes_learned;
                        continue;
                    }
                    Ok(AgentOutcome::Partial { result: r, caveat }) => {
                        changes.extend(r.changes.clone());
                        errors_fixed += r.errors_fixed;
                        budget_used += r.budget_used;
                        recipes_learned += r.recipes_learned;
                        partial_notes.push(caveat);
                        continue;
                    }
                    Ok(blocked_or_failed) => {
                        let _ = self.writer.rollback(&global);
                        return Ok(blocked_or_failed);
                    }
                    Err(e) => {
                        let _ = self.writer.rollback(&global);
                        return Ok(blocked_outcome(
                            BlockReason::NoVerifiedPattern,
                            "SYNTH_ERROR",
                            e,
                        ));
                    }
                }
            }
            // Re-plan against current disk (offsets from Phase A are stale
            // once earlier tasks have applied).
            let plan = match self.writer.plan(&task, &self.symbol_table) {
                Ok(p) => p,
                Err(e) => {
                    let _ = self.writer.rollback(&global);
                    self.emit(ProgressEvent::Stage {
                        id: task.id,
                        stage: "plan-blocked",
                        detail: e.clone(),
                    });
                    return Ok(blocked_outcome(
                        BlockReason::NoVerifiedPattern,
                        "PLAN_ERROR",
                        e,
                    ));
                }
            };
            self.emit(ProgressEvent::Stage {
                id: task.id,
                stage: "planned",
                detail: format!(
                    "{} edits · {}",
                    plan.edits.len(),
                    plan.edits
                        .iter()
                        .map(|e| e.file.to_string_lossy().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            });

            // 3. Apply: apply ONLY the precise edits from the plan
            let files = match self.writer.apply_plan(&plan) {
                Ok(f) => f,
                Err(e) => {
                    // Rollback everything on failure
                    let _ = self.writer.rollback(&global);
                    self.emit(ProgressEvent::Stage {
                        id: task.id,
                        stage: "apply-failed",
                        detail: e.clone(),
                    });
                    return Ok(AgentOutcome::Failed {
                        reason: format!("Failed to apply plan: {}", e),
                    });
                }
            };
            self.emit(ProgressEvent::Stage {
                id: task.id,
                stage: "applied",
                detail: files.join(", "),
            });

            // 4. Verify: run the compiler/tests
            let verdict = self.verifier.verify().await;
            log::info!(
                "VERIFICATION\n  clean={}\n  errors={}\n  warnings={}",
                verdict.is_clean(),
                verdict.errors.len(),
                verdict.warnings.len()
            );
            self.emit(ProgressEvent::Stage {
                id: task.id,
                stage: "verify",
                detail: if verdict.is_clean() {
                    "clean".to_string()
                } else {
                    format!("{} errors", verdict.errors.len())
                },
            });

            if verdict.is_clean() {
                // 5a. Commit: changes are clean, commit the transaction.
                // Syntax-only verdicts commit too but mark the run Partial.
                self.writer.commit(&global)?;
                changes.extend(files.clone());
                log::info!("RESULT\n  SUCCESS\n  files={:?}", files);
                self.emit(ProgressEvent::Stage {
                    id: task.id,
                    stage: if verdict.full {
                        "committed"
                    } else {
                        "committed-syntax-only"
                    },
                    detail: String::new(),
                });
                if !verdict.full {
                    partial_notes.push(partial_caveat.clone());
                }
                continue;
            }

            // 5b. Errors detected → CorrectionPipeline
            //     Bounded fix → re-verify loop. Each attempt consumes budget;
            //     when budget runs out the bot yields honestly instead of guessing.
            let mut verdict = verdict;
            let mut attempts = 0u32;
            loop {
                let mut any_fixed = false;
                for error in &verdict.errors {
                    self.emit(ProgressEvent::Stage {
                        id: task.id,
                        stage: "repair",
                        detail: format!("{} {}:{}", error.code, error.file, error.line),
                    });
                    let correction =
                        self.corrector
                            .try_correct(error, &self.arena.read(), &self.writer);

                    if correction.fixed {
                        errors_fixed += 1;
                        recipes_learned += correction.new_recipes;
                        changes.extend(correction.files_changed);
                        any_fixed = true;
                    } else {
                        // No recipe found and can't synthesize fix — the bot
                        // KNOWS it doesn't know. Roll back the whole run so
                        // no unproven bytes remain, then report honestly
                        // instead of guessing.
                        let _ = self.writer.rollback(&global);
                        self.emit(ProgressEvent::Stage {
                            id: task.id,
                            stage: "repair-blocked",
                            detail: error.code.clone(),
                        });
                        return Ok(AgentOutcome::Blocked {
                            reason: BlockReason::NoSafeFix,
                            diagnostics: vec![error.clone()],
                        });
                    }
                }

                attempts += 1;
                budget_used += 1;
                if !self.budget.consume(1.0, 0.0, 0.0) {
                    // Rollback before returning - budget exhausted
                    let _ = self.writer.rollback(&global);
                    self.emit(ProgressEvent::Stage {
                        id: task.id,
                        stage: "budget-exhausted",
                        detail: format!("{} attempts", attempts),
                    });
                    return Ok(AgentOutcome::Blocked {
                        reason: BlockReason::NoSafeFix,
                        diagnostics: vec![CompileError {
                            code: "BUDGET_EXHAUSTED".to_string(),
                            message: format!(
                                "Retry budget exhausted ({:.1} remaining) after {} attempt(s). \
                                 Yielding rather than guessing further.",
                                self.budget.remaining(),
                                attempts
                            ),
                            file: String::new(),
                            line: 0,
                            col: 0,
                            suggestion: None,
                            source_line: None,
                            kind: crate::engine::error::ErrorKind::Other,
                        }],
                    });
                }

                if !any_fixed {
                    // Rollback - fixes applied but verification is not clean
                    let _ = self.writer.rollback(&global);
                    return Ok(AgentOutcome::Failed {
                        reason: format!(
                            "Fixes applied but verification is not clean after {} attempt(s).",
                            attempts
                        ),
                    });
                }

                // Re-verify
                verdict = self.verifier.verify().await;
                if verdict.is_clean() {
                    log::info!("Verification passed after {} attempt(s)", attempts);
                    if !verdict.full {
                        partial_notes.push(partial_caveat.clone());
                    }
                    break;
                }
            }
        }

        let mut result = TaskResult {
            success: true,
            changes,
            errors_fixed,
            budget_used,
            recipes_learned,
            message: format!(
                "Done. Fixed {} errors, learned {} recipes, used {}/{} attempts.",
                errors_fixed, recipes_learned, budget_used, self.budget.total
            ),
        };
        // GitHub publish is delivery, not proof: it appends to the message
        // and never flips the outcome. A failed publish reports plainly.
        if let Some(note) = self.maybe_publish(&intent).await {
            result.message.push_str(&note);
        }
        if partial_notes.is_empty() {
            Ok(AgentOutcome::Success(result))
        } else {
            partial_notes.sort();
            partial_notes.dedup();
            Ok(AgentOutcome::Partial {
                result,
                caveat: partial_notes.join("; "),
            })
        }
    }

    /// Publish to GitHub when the intent asks for it and a token is set.
    /// Returns a message suffix (success URL or honest failure), or None
    /// when publishing was never requested. Reads only; never rewrites
    /// local files.
    async fn maybe_publish(&self, intent: &StructuredIntent) -> Option<String> {
        let token = self.github_token.clone()?;
        let haystack = format!(
            "{} {}",
            intent.goal,
            intent
                .actions
                .iter()
                .map(|a| match a {
                    crate::engine::tasks::IntentAction::Action { action, .. } => action.clone(),
                    crate::engine::tasks::IntentAction::Research { topic, .. } => topic.clone(),
                })
                .collect::<Vec<_>>()
                .join(" ")
        )
        .to_lowercase();
        let wants_repo = haystack.contains("repositor")
            || haystack.contains("github")
            || haystack.contains("publish")
            || haystack.contains("upload");
        if !wants_repo {
            return None;
        }
        // Repo name: explicit `repo:<name>` wins, then "called/named X",
        // else the project dir name.
        let repo_name = haystack
            .split("repo:")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .map(|s| s.to_string())
            .or_else(|| {
                let words: Vec<&str> = haystack.split_whitespace().collect();
                let stop = [
                    "then",
                    "and",
                    "to",
                    "for",
                    "with",
                    "on",
                    "a",
                    "an",
                    "the",
                    "called",
                    "named",
                    "repository",
                    "repo",
                ];
                words.iter().enumerate().find_map(|(i, w)| {
                    if *w != "called" && *w != "named" {
                        return None;
                    }
                    let mut name = Vec::new();
                    for extra in words.iter().skip(i + 1).take(3) {
                        if stop.contains(extra) {
                            break;
                        }
                        name.push(*extra);
                    }
                    if name.is_empty() {
                        None
                    } else {
                        Some(name.join(" "))
                    }
                })
            })
            .unwrap_or_else(|| {
                self.project_dir
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "grounding-project".to_string())
            });
        self.emit(ProgressEvent::Stage {
            id: 0,
            stage: "github-publish",
            detail: repo_name.clone(),
        });
        match crate::github::token_owner(&token).await {
            Ok(owner) => {
                match crate::github::publish(&token, &owner, &repo_name, &self.project_dir).await {
                    Ok((url, skipped)) => {
                        let mut note = format!("\nGitHub: {}", url);
                        if !skipped.is_empty() {
                            note.push_str(&format!(" ({} files skipped)", skipped.len()));
                        }
                        Some(note)
                    }
                    Err(e) => Some(format!("\nGitHub publish failed: {}", e)),
                }
            }
            Err(e) => Some(format!("\nGitHub publish failed: {}", e)),
        }
    }

    /// Try each synthesis candidate: snapshot → apply → verify →
    /// commit on first fully-clean candidate, rollback otherwise.
    /// Behavior is judged by the contract test, not just compilation.
    /// Plans are prebuilt by the caller (Phase A); terminal failure here
    /// leaves nothing applied — the caller rolls back the global snapshot.
    async fn run_synthesis_plans(&self, plans: &[EditPlan]) -> Result<AgentOutcome, String> {
        let mut last_errors = Vec::new();
        for (i, plan) in plans.iter().enumerate() {
            log::info!("SYNTH candidate {}/{}", i + 1, plans.len());
            self.emit(ProgressEvent::Stage {
                id: plan.task_id,
                stage: "synth-candidate",
                detail: format!("{}/{}", i + 1, plans.len()),
            });
            let snapshot = self.writer.snapshot(plan)?;
            match self.writer.apply_plan(plan) {
                Ok(files) => {
                    let verdict = self.verifier.verify().await;
                    log::info!(
                        "VERIFICATION candidate {}\n  clean={}\n  errors={}",
                        i + 1,
                        verdict.is_clean(),
                        verdict.errors.len()
                    );
                    self.emit(ProgressEvent::Stage {
                        id: plan.task_id,
                        stage: "verify",
                        detail: if verdict.is_clean() {
                            format!("candidate {} clean", i + 1)
                        } else {
                            format!("candidate {}: {} errors", i + 1, verdict.errors.len())
                        },
                    });
                    if verdict.is_clean() {
                        self.writer.commit(&snapshot)?;
                        log::info!("RESULT\n  SUCCESS candidate {}", i + 1);
                        self.emit(ProgressEvent::Stage {
                            id: plan.task_id,
                            stage: "committed",
                            detail: format!("candidate {}/{}", i + 1, plans.len()),
                        });
                        let result = TaskResult {
                            success: true,
                            changes: files,
                            errors_fixed: 0,
                            budget_used: i as u32,
                            recipes_learned: 0,
                            message: format!(
                                "Synthesized and verified candidate {}/{}.",
                                i + 1,
                                plans.len()
                            ),
                        };
                        if verdict.full {
                            return Ok(AgentOutcome::Success(result));
                        }
                        return Ok(AgentOutcome::Partial {
                            result,
                            caveat: "syntax-only verification (no toolchain on this host): types and behavior unproven".to_string(),
                        });
                    }
                    last_errors = verdict.errors;
                    let _ = self.writer.rollback(&snapshot);
                }
                Err(e) => {
                    let _ = self.writer.rollback(&snapshot);
                    last_errors = vec![CompileError {
                        code: "APPLY_ERROR".to_string(),
                        message: e,
                        file: String::new(),
                        line: 0,
                        col: 0,
                        suggestion: None,
                        source_line: None,
                        kind: crate::engine::error::ErrorKind::Other,
                    }];
                }
            }
        }
        Ok(AgentOutcome::Blocked {
            reason: BlockReason::NoSafeFix,
            diagnostics: last_errors,
        })
    }

    /// Run as a persistent background daemon.
    pub async fn run_daemon(&self) {
        log::info!(
            "Grounding Coder daemon started in {}",
            self.project_dir.display()
        );
        // The daemon keeps running, waiting for structured intents.
        // The LLM (if used) feeds intents here via IPC/API.
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            log::info!("Daemon alive. Arena: {} symbols.", self.arena.read().len());
        }
    }

    pub fn symbols(&self) -> Vec<(String, CodeSymbol)> {
        self.arena.read().all_symbols()
    }

    pub fn recipes(&self) -> Vec<String> {
        self.recipes.list()
    }
}
