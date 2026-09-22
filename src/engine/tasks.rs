use serde::{Deserialize, Serialize};

/// Priority weight for error magnitude (Eₚ) in the priority formula.
/// Adapted from grounded's GoalFormationEngine ALPHA.
const ALPHA: f64 = 0.35;
/// Priority weight for novelty level (N) in the priority formula.
/// Adapted from grounded's GoalFormationEngine BETA.
const BETA: f64 = 0.25;
/// Priority weight for unresolved symbol ratio (U) — adapts γ (drive deprivation).
/// High unresolved ratio → high priority to focus on resolution.
const GAMMA: f64 = 0.25;

/// Ring buffer size for tracking per-symbol error history.
/// Adapted from grounded's ERROR_HISTORY_SIZE.
const ERROR_HISTORY_SIZE: usize = 8;

/// Minimum consecutive high-error ticks before a symbol is prioritized.
/// Adapted from grounded's ERROR_PERSISTENCE_THRESHOLD.
const UNRESOLVED_PERSISTENCE: u8 = 3;

/// Requirements for the target runtime/platform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeRequirements {
    /// Minimum SDK version (Android)
    pub min_sdk: Option<u32>,
    /// Target SDK version (Android)
    pub target_sdk: Option<u32>,
    /// Android package identifier
    pub package: Option<String>,
    /// API level constraints
    pub api_constraints: Vec<String>,
}

/// Types of sub-tasks the bot can form from a structured intent.
/// These map directly to grounded's `GoalReason` variants but are specific
/// to code: instead of "understand_node", we get "resolve_symbol", "write_fn", etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskKind {
    /// Resolve an unknown symbol (grounded's "understand_" goal)
    ResolveSymbol,
    /// Write a new function (grounded's new goal from gap detection)
    WriteFunction,
    /// Add an import/use statement
    AddImport,
    /// Add a field to a struct/class
    AddField,
    /// Wire up an event handler / callback
    WireHandler,
    /// Add a test for existing code
    AddTest,
    /// Create a new file
    CreateFile,
    /// Research unknown requirements
    ResearchRequirements,
    /// Create project manifest
    CreateProjectManifest,
}

impl TaskKind {
    pub fn label(&self) -> &'static str {
        match self {
            TaskKind::ResolveSymbol => "resolve_symbol",
            TaskKind::WriteFunction => "write_function",
            TaskKind::AddImport => "add_import",
            TaskKind::AddField => "add_field",
            TaskKind::WireHandler => "wire_handler",
            TaskKind::AddTest => "add_test",
            TaskKind::CreateFile => "create_file",
            TaskKind::ResearchRequirements => "research_requirements",
            TaskKind::CreateProjectManifest => "create_project_manifest",
        }
    }

    pub fn as_str(&self) -> &'static str {
        self.label()
    }
}

impl std::fmt::Display for TaskKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Specific edit operations that can be performed.
///
/// This replaces the old `payload: serde_json::Value` approach with
/// explicit, typed edit operations that the CodeWriter can execute
/// deterministically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EditIntent {
    AddImport {
        file: String,
        import: String,
    },

    InsertAfterSymbol {
        file: String,
        symbol: String,
        code_pattern: String,
    },

    ReplaceRange {
        file: String,
        start: usize,
        end: usize,
        replacement: String,
    },

    AddDefinition {
        file: String,
        name: String,
        kind: String,
        signature: String,
    },
}

/// A single code symbol's unresolved error history.
/// Repurposes grounded's `ErrorHistory` ring buffer.
#[derive(Debug, Clone)]
pub struct SymbolErrorHistory {
    /// Ring buffer of recent unresolved counts (how many attempts failed for this symbol).
    pub counts: [u32; ERROR_HISTORY_SIZE],
    /// Write index into the ring buffer.
    pub write_idx: usize,
    /// How many samples have been recorded.
    pub count: usize,
    /// Consecutive ticks this symbol has been unresolved.
    pub consecutive_unresolved: u8,
}

impl Default for SymbolErrorHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolErrorHistory {
    pub fn new() -> Self {
        SymbolErrorHistory {
            counts: [0; ERROR_HISTORY_SIZE],
            write_idx: 0,
            count: 0,
            consecutive_unresolved: 0,
        }
    }

    /// Record a new unresolved count (number of unknown references in this symbol).
    pub fn record(&mut self, unresolved_count: u32) {
        self.counts[self.write_idx] = unresolved_count;
        self.write_idx = (self.write_idx + 1) % ERROR_HISTORY_SIZE;
        if self.count < ERROR_HISTORY_SIZE {
            self.count += 1;
        }
        if unresolved_count > 0 {
            self.consecutive_unresolved += 1;
        } else {
            self.consecutive_unresolved = 0;
        }
    }

    /// EMA of recent unresolved counts.
    pub fn ema_unresolved(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mut sum = 0.0;
        let n = self.count.min(ERROR_HISTORY_SIZE);
        for i in 0..n {
            let idx = (self.write_idx + ERROR_HISTORY_SIZE - 1 - i) % ERROR_HISTORY_SIZE;
            let weight = 0.7_f64.powi(i as i32);
            sum += (self.counts[idx] as f64) * weight;
        }
        sum / n as f64
    }

    /// True if this symbol has been persistently unresolved.
    pub fn persistently_unresolved(&self) -> bool {
        self.consecutive_unresolved >= UNRESOLVED_PERSISTENCE
    }

    /// Rising trend detection.
    pub fn rising_trend(&self) -> bool {
        if self.count < 2 {
            return false;
        }
        let ema_now = self.ema_unresolved();
        let mut sum = 0.0;
        let n = self.count.min(ERROR_HISTORY_SIZE);
        for i in 0..n {
            let idx = (self.write_idx + ERROR_HISTORY_SIZE - 1 - i) % ERROR_HISTORY_SIZE;
            let weight = 0.7_f64.powi(i as i32);
            sum += (self.counts[idx] as f64) * weight;
        }
        let ema_prev = sum / n as f64;
        ema_now > ema_prev
    }
}

/// A single code symbol's unresolved error history.
/// Repurposes grounded's `ErrorHistory` ring buffer.
#[derive(Debug, Clone)]
pub struct TrackedSymbol {
    pub id: SymbolId,
    pub history: SymbolErrorHistory,
    pub task_formed: bool,
}

/// Requirements for the target runtime/platform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentDefinition {
    /// Name of the new symbol (e.g., "fun vibrateButton")
    pub name: String,
    /// Type of symbol: "function", "struct", "enum", etc.
    pub kind: String,
    /// Symbols this definition references (needs resolution)
    pub references: Vec<String>,
    /// Optional type signature (e.g., "fn vibrate(duration_ms: Long)")
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentTest {
    /// Test function name
    pub name: String,
    /// What to test (assertion descriptions)
    pub assertions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredIntent {
    /// Platform (android, ios, web, desktop)
    pub platform: String,
    /// Architecture (arm64, x86_64, etc.)
    pub architecture: String,
    /// Runtime information (sdk version, etc.)
    pub runtime: String,
    /// Device capabilities required
    pub capabilities: Vec<String>,
    /// Problem domains
    pub domains: Vec<String>,
    /// Constraints (performance, size, etc.)
    pub constraints: Vec<String>,
    /// Dependencies (external crates/libs)
    pub dependencies: Vec<String>,
    /// Requirements that the LLM is unsure about
    pub unknown_requirements: Vec<String>,
    /// Goal of the intent
    pub goal: String,
    /// Actions to perform
    pub actions: Vec<IntentAction>,
    /// Definitions to create/add
    pub define: Vec<Option<IntentDefinition>>,
    /// Tests to run/add
    pub test: Vec<IntentTest>,
    /// References to existing symbols
    pub references: Vec<String>,
}

/// A single code symbol's unresolved error history.
/// Repurposes grounded's `ErrorHistory` ring buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentAction {
    /// Perform an action (e.g., "vibrate", "navigate")
    Action {
        /// The action to perform
        action: String,
        /// Parameters for the action
        params: Vec<String>,
        /// References needed for this action
        references: Vec<String>,
    },
    /// Declare an intent to research something
    Research {
        /// What to research
        topic: String,
        /// Why we need to research it
        reason: String,
    },
}

impl Default for TaskDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskDecomposer {
    pub fn new() -> Self {
        TaskDecomposer {
            error_histories: Vec::with_capacity(64),
            task_count: 0,
            tick: 0,
            budget_remaining: 1.0,
            budget_total: 1.0,
        }
    }

    /// Decompose a structured intent into sub-tasks.
    ///
    /// This is the core of grounded's goal formation:
    /// Take a high-level intent and break it down into
    /// actionable sub-goals that can be pursued independently.
    ///
    /// The LLM gives us a structured intent. We decompose it
    /// into tasks that our executor can handle.
    pub fn decompose(&mut self, intent_json: &str, arena: &CodeArena) -> Vec<SubTask> {
        let intent: StructuredIntent = serde_json::from_str(intent_json)
            .expect("Failed to parse intent JSON");

        let mut tasks = Vec::new();

        // Handle unknown requirements - create research tasks
        for req in &intent.unknown_requirements {
            tasks.push(self.form_research_task(&req));
        }

        // Handle definitions - create definition tasks
        for def_opt in &intent.define {
            if let Some(def) = def_opt {
                tasks.extend(self.form_definition_tasks(def, &intent));
            }
        }

        // Handle tests - create test tasks
        for test in &intent.test {
            if let Some(task) = self.form_test_task(&test, &intent) {
                tasks.push(task);
            }
        }

        // Handle actions - create action tasks
        for action in &intent.actions {
            if let Some(task) = self.form_action_task(&action, &intent) {
                tasks.push(task);
            }
        }

        // Apply budget pressure adjustments
        tasks = self.check_budget_pressure(tasks);

        tasks
    }

    /// Form a task for researching an unknown requirement.
    fn form_research_task(&mut self, requirement: &str) -> SubTask {
        let description = format!("Research: {}", requirement);
        let priority = 0.3;
        let mut task = SubTask::new(
            TaskKind::ResearchRequirements,
            description,
            serde_json::json!({}),
            "unknown_requirement".to_string()
        ).with_priority(priority)
         .with_deadline(self.tick + 1000);

        task
    }

    /// Form tasks for a definition (could be multiple sub-tasks).
    fn form_definition_tasks(
        &mut self,
        def: &IntentDefinition,
        intent: &StructuredIntent,
    ) -> Vec<SubTask> {
        let mut tasks = Vec::new();

        // Add the definition itself
        let mut def_task = SubTask::new(
            TaskKind::AddDefinition,
            format!("Add {} {}", def.kind, def.name),
            serde_json::json!({}),
            "intent_definition".to_string()
        ).with_priority(0.7)
         .with_deadline(self.tick + 1000);

        // Add required symbols as needed
        def_task.required_symbols = def.references.clone();

        tasks.push(def_task);

        // If this definition needs imports, add import tasks
        if let Some(sig) = &def.signature {
            // Extract potential imports from signature (simplified)
            // In practice, this would use proper parsing
            for reference in &def.references {
                if !reference.contains('.') {
                    // Likely a local reference, skip
                    continue;
                }
                
                let parts: Vec<&str> = reference.split('.').collect();
                if parts.len() >= 2 {
                    let import_path = parts[..parts.len()-1].join(".");
                    let import_task = SubTask::new(
                        TaskKind::AddImport,
                        format!("Add import for {}", reference),
                        serde_json::json!({}),
                        "import_from_definition".to_string()
                    ).with_priority(0.6)
                     .with_deadline(self.tick + 1000);
                     
                    tasks.push(import_task);
                }
            }
        }

        tasks
    }

    /// Form a task for adding a test.
    fn form_test_task(
        &mut self,
        test: &IntentTest,
        _intent: &StructuredIntent,
    ) -> Option<SubTask> {
        let description = format!("Add test: {}", test.name);
        let mut task = SubTask::new(
            TaskKind::AddTest,
            description,
            serde_json::json!({}),
            "intent_test".to_string()
        ).with_priority(0.6)
         .with_deadline(self.tick + 1000);

        Some(task)
    }

    /// Form a task from an intent action.
    fn form_action_task(
        &mut self,
        action: &IntentAction,
        intent: &StructuredIntent,
    ) -> Option<SubTask> {
        match action {
            IntentAction::Action { action, params, references } => {
                let action_lower = action.to_lowercase();

                let (kind, description, edit) =
                    if action_lower.contains("button") || action_lower.contains("vibrate") {
                        (
                            TaskKind::WriteFunction,
                            format!("Handle action: {}", action),
                            EditIntent::InsertAfterSymbol {
                                file: intent.file.clone().unwrap_or_else(|| "src/main.rs".to_string()),
                                symbol: "onCreate".to_string(),
                                code_pattern: format!("
        // Handle {} action
        {}", action, params.join(", "))
                            },
                        )
                    } else if action_lower.contains("test") || action_lower.contains("assert") {
                        (
                            TaskKind::AddTest,
                            format!("Test action: {}", action),
                            EditIntent::AddDefinition {
                                file: intent.file.clone().unwrap_or_else(|| "src/main.rs".to_string()),
                                name: format!("test_{}", action.to_lowercase().replace(" ", "_")),
                                kind: "function".to_string(),
                                signature: format!("fn {}()", action.to_lowercase().replace(" ", "_")),
                            },
                        )
                    } else {
                        (
                            TaskKind::WriteFunction,
                            format!("Execute action: {}", action),
                            EditIntent::InsertAfterSymbol {
                                file: intent.file.clone().unwrap_or_else(|| "src/main.rs".to_string()),
                                symbol: "main".to_string(),
                                code_pattern: format!("
        // Action: {}
        {}", action, params.join(", "))
                            },
                        )
                    };

                let priority = 0.6;
                let mut task = SubTask::new(
                    kind,
                    description,
                    serde_json::json!({}),
                    "intent_action".to_string()
                ).with_priority(priority)
                 .with_deadline(self.tick + 1000);

                // Extract any referenced symbols from params
                if let Some(refs) = params.get(0) {
                    if !refs.is_empty() {
                        task.required_symbols.push(refs.clone());
                    }
                }

                Some(task)
            },
            IntentAction::Research { topic, reason } => {
                let description = format!("Research: {}", topic);
                let mut task = SubTask::new(
                    TaskKind::ResearchRequirements,
                    description,
                    serde_json::json!({}),
                    "intent_research".to_string()
                ).with_priority(0.3)
                 .with_deadline(self.tick + 1000);
                 
                Some(task)
            }
        }
    }

    /// Detect if the ratio of unresolved symbols is rising.
    /// This is a simplified version — grounded's version uses a ring buffer
    /// per node. Here we track a simple EMA across intents.
    fn detect_rising_trend(&mut self, total: u32, unresolved: u32) -> bool {
        let _ratio = unresolved as f64 / total.max(1) as f64;

        // Track in error histories vector (repurposing the pattern)
        if self.error_histories.is_empty() {
            self.error_histories.resize(
                1,
                TrackedSymbol {
                    id: SymbolId(0),
                    history: SymbolErrorHistory::new(),
                    task_formed: false,
                },
            );
        }

        self.error_histories[0].history.record(unresolved);
        self.error_histories[0].history.rising_trend()
    }

    /// Check for budget pressure and form urgent tasks.
    pub fn check_budget_pressure(&mut self, remaining_tasks: Vec<SubTask>) -> Vec<SubTask> {
        if self.budget_total > 0.0 && self.budget_remaining / self.budget_total < 0.3 {
            // Less than 30% budget left — elevate all pending tasks
            let mut elevated = remaining_tasks;
            for task in &mut elevated {
                task.priority = (task.priority * 1.5).clamp(0.05, 1.0);
                task.source = format!("budget_pressure: {}", task.source);
            }
            elevated
        } else {
            remaining_tasks
        }
    }

    /// Tick the decomposer forward (for deadline tracking).
    pub fn tick(&mut self) {
        self.tick += 1;
    }

    /// Current tick count.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    pub fn task_count(&self) -> u64 {
        self.task_count
    }
}

// --- Builder-style extensions on SubTask ---

impl SubTask {
    pub fn with_priority(mut self, priority: f64) -> Self {
        self.priority = priority.clamp(0.05, 1.0);
        self
    }

    pub fn with_deadline(mut self, deadline: u64) -> Self {
        self.deadline = deadline;
        self
    }
}

/// Parse an intent JSON string into a structured intent.
/// Returns None if parsing fails (and logs the error).
pub fn parse_intent(json: &str) -> Option<StructuredIntent> {
    match serde_json::from_str::<StructuredIntent>(json) {
        Ok(intent) => Some(intent),
        Err(e) => {
            log::error!("Failed to parse intent: {}", e);
            None
        }
    }
}

/// Extract all referenced symbols from an intent.
/// Used for pre-scan validation before writing any code.
pub fn extract_symbols(intent: &StructuredIntent) -> Vec<String> {
    let mut symbols = intent.references.clone();
    if let Some(def) = &intent.define {
        for def_opt in def {
            if let Some(d) = def_opt {
                symbols.extend(d.references.clone());
            }
        }
    }
    symbols
}
