pub mod arena;
pub mod budget;
pub mod corrector;
pub mod error;
pub mod plan;
pub mod recipes;
pub mod research;
pub mod symbols;
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
pub use tasks::{
    EditIntent, IntentAction, IntentDefinition, IntentTest, StructuredIntent, SubTask, TaskDecomposer, TaskKind,
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
    /// Task is blocked - no safe fix available
    Blocked {
        reason: BlockReason,
        diagnostics: Vec<CompileError>,
    },
    /// Task failed with a reason
    Failed {
        reason: String,
    },
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
        };

        // Bootstrap: scan the project and build the code symbol graph.
        bot.bootstrap();
        bot
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
        let intent: StructuredIntent = serde_json::from_str(intent_json)
            .map_err(|e| format!("Failed to parse intent: {}", e))?;

        // Collect all referenced symbols that the bot doesn't know yet
        let mut unknown_symbols = Vec::new();
        for sym in &intent.references {
            if !self.symbol_table.is_known(sym) {
                unknown_symbols.push(sym.clone());
            }
        }
        for sym in &intent.define {
            if let Some(d) = sym {
                if !self.symbol_table.is_known(&d.qname) {
                    unknown_symbols.push(d.qname.clone());
                }
            }
        }
        for sym in unknown_symbols {
            if let Some(source_url) = self.researcher.research(&sym) {
                log::info!("Resolving unknown symbol {} via {}", sym, source_url);
            } else {
                log::warn!(
                    "Could not research symbol: {} — no verified source found",
                    sym
                );
            }
        }

        // Step 1: TaskDecomposer breaks the intent into sub-tasks
        let tasks = self.decomposer.decompose(intent_json, &self.arena.read());

        let mut changes = Vec::new();
        let mut errors_fixed = 0u32;
        let mut budget_used = 0u32;
        let mut recipes_learned = 0u32;

        for task in tasks {
            log::info!("Task: {} ({})", task.description, task.kind);

            // === TRANSACTIONAL LOOP ===
            // 1. Plan: generate an EditPlan with precise edits
            let plan = match self.writer.plan(&task, &self.symbol_table) {
                Ok(p) => p,
                Err(e) => {
                    return Ok(AgentOutcome::Blocked {
                        reason: BlockReason::NoVerifiedPattern,
                        diagnostics: vec![CompileError {
                            code: "PLAN_ERROR".to_string(),
                            message: e,
                            file: String::new(),
                            line: 0,
                            col: 0,
                            suggestion: None,
                            source_line: None,
                            kind: crate::engine::error::ErrorKind::Other,
                        }],
                    });
                }
            };

            // 2. Snapshot: capture current state before applying edits
            let snapshot = match self.writer.snapshot(&plan) {
                Ok(s) => s,
                Err(e) => {
                    return Ok(AgentOutcome::Failed {
                        reason: format!("Failed to create snapshot: {}", e),
                    });
                }
            };

            // 3. Apply: apply ONLY the precise edits from the plan
            let files = match self.writer.apply_plan(&plan) {
                Ok(f) => f,
                Err(e) => {
                    // Rollback on failure
                    let _ = self.writer.rollback(&snapshot);
                    return Ok(AgentOutcome::Failed {
                        reason: format!("Failed to apply plan: {}", e),
                    });
                }
            };

            // 4. Verify: run the compiler/tests
            let verdict = self.verifier.verify().await;

            if verdict.is_clean() {
                // 5a. Commit: changes are clean, commit the transaction
                self.writer.commit(&snapshot)?;
                changes.extend(files);
                log::info!("Verification passed - committed");
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
                        // KNOWS it doesn't know. It reports this honestly
                        // instead of guessing. (grounded's "honesty principle")
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
                    let _ = self.writer.rollback(&snapshot);
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
                    let _ = self.writer.rollback(&snapshot);
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
                    break;
                }
            }
        }

        Ok(AgentOutcome::Success(TaskResult {
            success: true,
            changes,
            errors_fixed,
            budget_used,
            recipes_learned,
            message: format!(
                "Done. Fixed {} errors, learned {} recipes, used {}/{} attempts.",
                errors_fixed, recipes_learned, budget_used, self.budget.total
            ),
        }))
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
