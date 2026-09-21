pub mod arena;
pub mod budget;
pub mod corrector;
pub mod error;
pub mod recipes;
pub mod research;
pub mod symbols;
pub mod tasks;
pub mod verifier;
pub mod writer;

pub use arena::{CodeArena, CodeSymbol, SymbolEdge, SymbolId, SymbolKind, SymbolRelation};
pub use budget::RetryBudget;
pub use corrector::CorrectionPipeline;
pub use error::{CompileError, ErrorClassifier, ErrorKind};
pub use recipes::RecipeLog;
pub use research::ResearchOracle;
pub use symbols::{SymbolTable, CodeDef};
pub use tasks::{SubTask, StructuredIntent, IntentAction, IntentDefinition, IntentTest, TaskDecomposer, TaskKind};
pub use verifier::CodeVerifier;
pub use writer::CodeWriter;
pub use oracle::{KnowledgeOracle, KnowledgeAdapter, KnowledgeResult, CodeSymbolInfo, VerifiedFact, CodePattern};

use std::path::PathBuf;
use std::sync::Arc;
use parking_lot::RwLock;

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
        writeln!(f, "Result: {}", if self.success { "SUCCESS" } else { "FAILURE" })?;
        writeln!(f, "Message: {}", self.message)?;
        writeln!(f, "Changes: {} files", self.changes.len())?;
        writeln!(f, "Errors fixed: {}", self.errors_fixed)?;
        writeln!(f, "Budget used: {}", self.budget_used)?;
        write!(f, "Recipes learned: {}", self.recipes_learned)?;
        std::fmt::Result::Ok(())
    }
}

impl CodeBot {
    pub fn new(project_dir: &str, max_retries: u32) -> Self {
        let project_path = PathBuf::from(project_dir);
        let arena = Arc::new(RwLock::new(CodeArena::new()));
        let symbol_table = SymbolTable::new();
        let budget = RetryBudget::new(max_retries as f64);
        let recipes = RecipeLog::new();
        let corrector = CorrectionPipeline::new(budget.clone(), recipes.clone());
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
    pub async fn run_task(&mut self, intent_json: &str) -> Result<TaskResult, String> {
        // Step 0: Parse intent and research unknown symbols
        // The LLM translator outputs structured intent. Before we decompose
        // into tasks, we research any unknown symbols via the ResearchOracle —
        // fetching from verified sources (docs.rs, Android SDK docs).
        let intent: StructuredIntent = serde_json::from_str(intent_json)
            .map_err(|e| format!("Failed to parse intent: {}", e))?;

        // Collect all referenced symbols that the bot doesn't know yet
        let mut unknown_symbols: Vec<String> = Vec::new();
        for ref_name in &intent.references {
            let sym = ref_name.to_lowercase();
            if !self.symbol_table.is_known(&sym) && self.arena.read().lookup(ref_name).is_none() {
                unknown_symbols.push(ref_name.clone());
            }
        }
        if let Some(def) = &intent.define {
            for ref_name in &def.references {
                let sym = ref_name.to_lowercase();
                if !self.symbol_table.is_known(&sym) && self.arena.read().lookup(ref_name).is_none() {
                    unknown_symbols.push(ref_name.clone());
                }
            }
        }

        // Research unknown symbols from verified sources
        for sym in &unknown_symbols {
            let lang = if intent.language.to_lowercase().contains("rust") { "rust" } else { &intent.language };
            if let Some(def) = self.researcher.research(sym, lang).await {
                let qname = def.qname.clone();
                let source_url = def.source_url.clone();
                let code_def = def.into_code_def();
                self.symbol_table.index(&qname, code_def);
                log::info!("Researched symbol: {} from {}", sym, source_url);
            } else {
                log::warn!("Could not research symbol: {} — no verified source found", sym);
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

            // Step 2: CodeWriter proposes code using known patterns
            // (NEVER generates from scratch — always composes from
            //  patterns found in the codebase or API reference)
            let draft = self.writer.write(&task, &self.symbol_table);

            // Step 3: Apply draft to the codebase
            let files = self.writer.apply(&task, &draft, &self.project_dir);

            // Step 4: CodeVerifier — the oracle (grounded's VerificationLoop)
            // Runs cargo check / cargo test / clippy
            let verdict = self.verifier.verify().await;

            if verdict.is_clean() {
                changes.extend(files);
                log::info!("Verification passed");
                continue;
            }

            // Step 5: Errors detected → CorrectionPipeline (grounded's SelfHealingPipeline)
            // 5 phases: classify → recipe lookup → apply → re-verify → record
            for error in &verdict.errors {
                let original_budget = self.budget.remaining();
                let correction = self.corrector.try_correct(error, &self.arena.read(), &self.writer);

                if correction.fixed {
                    errors_fixed += 1;
                    budget_used += 1;
                    self.budget.consume(1.0, 0.0, 0.0); // simple unit cost
                    recipes_learned += correction.new_recipes;
                    changes.extend(correction.files_changed);
                } else {
                    // No recipe found and can't synthesize fix — the bot
                    // KNOWS it doesn't know. It reports this honestly
                    // instead of guessing. (grounded's "honesty principle")
                    return Err(format!(
                        "Cannot resolve error {}: no recipe found. \
                         This is a structural limitation, not a guess. \
                         Budget remaining: {:.1}",
                        error.code, original_budget
                    ));
                }
            }
        }

        Ok(TaskResult {
            success: true,
            changes,
            errors_fixed,
            budget_used,
            recipes_learned,
            message: format!(
                "Done. Fixed {} errors, learned {} recipes, used {}/{} attempts.",
                errors_fixed, recipes_learned, budget_used, self.budget.total
            ),
        })
    }

    /// Run as a persistent background daemon.
    pub async fn run_daemon(&self) {
        log::info!("Grounding Coder daemon started in {}", self.project_dir.display());
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
