use super::arena::{CodeArena, SymbolId};
use serde::{Deserialize, Serialize};

/// Priority weight for error magnitude (Eₚ) in the priority formula.
/// Adapted from grounded's GoalFormationEngine ALPHA.
#[allow(dead_code)]
const ALPHA: f64 = 0.35;
/// Priority weight for novelty level (N) in the priority formula.
/// Adapted from grounded's GoalFormationEngine BETA.
#[allow(dead_code)]
const BETA: f64 = 0.25;
/// Priority weight for unresolved symbol ratio (U) — adapts γ (drive deprivation).
/// High unresolved ratio → high priority to focus on resolution.
#[allow(dead_code)]
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
    /// Synthesize a function body from contract cases (Phase 1)
    SynthesizeFunction,
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
    /// Add a verified definition (no LLM code allowed)
    AddDefinition,
    /// Verify the project as-is and repair what the oracles flag.
    /// Emitted when an intent carries no edits (e.g. "fix this repo").
    VerifyOnly,
    /// Ensure a registry-verified external crate is in Cargo.toml.
    /// Built by the engine itself after Step-0 crate verification —
    /// the decomposer never emits this.
    EnsureDep,
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
            TaskKind::AddDefinition => "add_definition",
            TaskKind::VerifyOnly => "verify_only",
            TaskKind::EnsureDep => "ensure_dep",
            TaskKind::SynthesizeFunction => "synthesize_function",
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

/// One contract case: input/output literals as pure data.
/// The engine embeds these into a test template it controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    #[serde(default)]
    pub input: String,
    #[serde(default)]
    pub expected: String,
}

/// A struct field as metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDef {
    #[serde(default)]
    pub name: String,
    #[serde(rename = "type", default)]
    pub ty: String,
}

/// One page section as metadata: heading + body, both plain text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionDef {
    #[serde(default)]
    pub heading: String,
    #[serde(default)]
    pub body: String,
}

/// A method as metadata. `op` names a verified operation; the engine
/// rejects anything outside its vocabulary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodDef {
    #[serde(default)]
    pub name: String,
    /// `none` (associated fn), `ref` (&self), `mut` (&mut self).
    #[serde(rename = "self", default)]
    pub self_kind: String,
    /// Params as `"name: Type"` strings.
    #[serde(default, deserialize_with = "de_vec_default")]
    pub params: Vec<String>,
    #[serde(default)]
    pub ret: Option<String>,
    #[serde(default)]
    pub op: String,
    #[serde(default)]
    pub field: Option<String>,
    #[serde(default)]
    pub amount: Option<String>,
}

/// Requirements for the target runtime/platform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentDefinition {
    /// Name of the new symbol (e.g., "fun vibrateButton")
    #[serde(default)]
    pub name: String,
    /// Type of symbol: "function", "struct", "enum", etc.
    #[serde(default)]
    pub kind: String,
    /// Symbols this definition references (needs resolution)
    #[serde(default, deserialize_with = "de_vec_default")]
    pub references: Vec<String>,
    /// Optional type signature (e.g., "fn vibrate(duration_ms: Long)")
    #[serde(default)]
    pub signature: Option<String>,
    /// Contract cases (input/output literals). Presence routes the
    /// definition to the synthesizer instead of stub planning.
    #[serde(default, deserialize_with = "de_vec_default")]
    pub cases: Vec<TestCase>,
    /// Struct fields (kind == "struct").
    #[serde(default, deserialize_with = "de_vec_default")]
    pub fields: Vec<FieldDef>,
    /// Method specs (kind == "struct").
    #[serde(default, deserialize_with = "de_vec_default")]
    pub methods: Vec<MethodDef>,
    /// Page title (kind == "page").
    #[serde(default)]
    pub title: Option<String>,
    /// Page sections (kind == "page").
    #[serde(default, deserialize_with = "de_vec_default")]
    pub sections: Vec<SectionDef>,
    /// Page footer (kind == "page").
    #[serde(default)]
    pub footer: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentTest {
    /// Test function name
    #[serde(default)]
    pub name: String,
    /// What to test (assertion descriptions)
    #[serde(default, deserialize_with = "de_vec_default")]
    pub assertions: Vec<String>,
}

fn default_confidence() -> f64 {
    1.0
}

use serde::Deserializer;

/// Deserialize `null` as default (empty vec) — LLMs emit null for empty arrays.
fn de_vec_default<'de, D, T>(d: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    let o: Option<Vec<T>> = Option::deserialize(d)?;
    Ok(o.unwrap_or_default())
}

fn de_vec_opt_default<'de, D, T>(d: D) -> Result<Vec<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    let o: Option<Vec<Option<T>>> = Option::deserialize(d)?;
    Ok(o.unwrap_or_default())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredIntent {
    /// Platform (android, ios, web, desktop)
    #[serde(default)]
    pub platform: String,
    /// Architecture (arm64, x86_64, etc.)
    #[serde(default)]
    pub architecture: String,
    /// Runtime information (sdk version, etc.) — free-form for forward compat
    #[serde(default)]
    pub runtime: String,
    /// Device capabilities required
    #[serde(default, deserialize_with = "de_vec_default")]
    pub capabilities: Vec<String>,
    /// Problem domains
    #[serde(default, deserialize_with = "de_vec_default")]
    pub domains: Vec<String>,
    /// Constraints (performance, size, etc.)
    #[serde(default, deserialize_with = "de_vec_default")]
    pub constraints: Vec<String>,
    /// Dependencies (external crates/libs)
    #[serde(default, deserialize_with = "de_vec_default")]
    pub dependencies: Vec<String>,
    /// Requirements that the LLM is unsure about
    #[serde(default, deserialize_with = "de_vec_default")]
    pub unknown_requirements: Vec<String>,
    /// Goal of the intent
    #[serde(default)]
    pub goal: String,
    /// Target file path (relative to project root)
    #[serde(default)]
    pub file: Option<String>,
    /// Language target
    #[serde(default)]
    pub language: Option<String>,
    /// Imports to ensure exist
    #[serde(default, deserialize_with = "de_vec_default")]
    pub imports: Vec<String>,
    /// Confidence 0.0-1.0
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    /// Actions to perform
    #[serde(default, deserialize_with = "de_vec_default")]
    pub actions: Vec<IntentAction>,
    /// Definitions to create/add
    #[serde(default, deserialize_with = "de_vec_opt_default")]
    pub define: Vec<Option<IntentDefinition>>,
    /// Tests to run/add
    #[serde(default, deserialize_with = "de_vec_default")]
    pub test: Vec<IntentTest>,
    /// References to existing symbols
    #[serde(default, deserialize_with = "de_vec_default")]
    pub references: Vec<String>,
}

/// A concrete sub-task decomposed from a structured intent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubTask {
    pub id: u64,
    pub kind: TaskKind,
    pub description: String,
    pub target_symbols: Vec<String>,
    pub required_symbols: Vec<String>,
    pub priority: f64,
    pub source: String,
    pub payload: serde_json::Value,
    pub deadline: u64,
}

impl SubTask {
    pub fn new(
        kind: TaskKind,
        description: String,
        payload: serde_json::Value,
        source: String,
    ) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self {
            id: COUNTER.fetch_add(1, Ordering::Relaxed),
            kind,
            description,
            target_symbols: Vec::new(),
            required_symbols: Vec::new(),
            priority: 0.5,
            source,
            payload,
            deadline: 0,
        }
    }
}

/// The TaskDecomposer — deterministic goal formation for code.
pub struct TaskDecomposer {
    #[allow(dead_code)]
    error_histories: Vec<TrackedSymbol>,
    task_count: u64,
    tick: u64,
    budget_remaining: f64,
    budget_total: f64,
}

/// A single code symbol's unresolved error history.
/// Repurposes grounded's `ErrorHistory` ring buffer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentAction {
    /// Perform an action (e.g., "vibrate", "navigate")
    Action {
        /// The action to perform
        #[serde(default)]
        action: String,
        /// Parameters for the action
        #[serde(default, deserialize_with = "de_vec_default")]
        params: Vec<String>,
        /// References needed for this action
        #[serde(default, deserialize_with = "de_vec_default")]
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
    pub fn decompose(&mut self, intent_json: &str, _arena: &CodeArena) -> Vec<SubTask> {
        self.tick += 1;
        let intent: StructuredIntent = match serde_json::from_str(intent_json) {
            Ok(i) => i,
            Err(e) => {
                log::error!("Failed to parse intent JSON: {}", e);
                return Vec::new();
            }
        };

        let mut tasks = Vec::new();

        // Handle unknown requirements - create research tasks
        for req in &intent.unknown_requirements {
            tasks.push(self.form_research_task(req));
        }

        // Handle explicit imports - create import tasks
        for imp in intent.imports.iter().chain(
            intent
                .references
                .iter()
                .filter(|r| r.contains("::") || r.contains('.')),
        ) {
            let mut t = SubTask::new(
                TaskKind::AddImport,
                format!("Add import {}", imp),
                serde_json::json!({"import": imp, "file": intent.file.clone().unwrap_or_else(|| "src/lib.rs".to_string())}),
                "intent_import".to_string(),
            )
            .with_priority(0.6)
            .with_deadline(self.tick + 1000);
            if let Some(f) = intent.file.clone() {
                t.target_symbols.push(f);
            }
            tasks.push(t);
        }

        // Handle definitions - create definition tasks
        for def in intent.define.iter().flatten() {
            tasks.extend(self.form_definition_tasks(def, &intent));
        }

        // Handle tests - create test tasks
        for test in &intent.test {
            if let Some(task) = self.form_test_task(test, &intent) {
                tasks.push(task);
            }
        }

        // Handle actions - create action tasks
        for action in &intent.actions {
            if let Some(task) = self.form_action_task(action, &intent) {
                tasks.push(task);
            }
        }

        // Nothing to build but verification was asked for (or is the
        // only sensible reading): run the oracles and repair loop.
        if tasks.is_empty() {
            tasks.push(
                SubTask::new(
                    TaskKind::VerifyOnly,
                    "Verify project and repair flagged errors".to_string(),
                    serde_json::json!({}),
                    "verify_only".to_string(),
                )
                .with_priority(0.5)
                .with_deadline(self.tick + 1000),
            );
        }

        self.task_count += tasks.len() as u64;
        // Apply budget pressure adjustments
        tasks = self.check_budget_pressure(tasks);

        tasks
    }

    /// Form a task for researching an unknown requirement.
    fn form_research_task(&mut self, requirement: &str) -> SubTask {
        let description = format!("Research: {}", requirement);
        let priority = 0.3;
        SubTask::new(
            TaskKind::ResearchRequirements,
            description,
            serde_json::json!({}),
            "unknown_requirement".to_string(),
        )
        .with_priority(priority)
        .with_deadline(self.tick + 1000)
    }

    /// Form tasks for a definition (could be multiple sub-tasks).
    fn form_definition_tasks(
        &mut self,
        def: &IntentDefinition,
        intent: &StructuredIntent,
    ) -> Vec<SubTask> {
        let mut tasks = Vec::new();
        let target_file = intent
            .file
            .clone()
            .unwrap_or_else(|| "src/lib.rs".to_string());

        // Contract present → synthesize a real body; otherwise plan a stub.
        // NO code from LLM either way, only metadata + literal values.
        // Pages (content slots) always synthesize — structure is fixed.
        let has_contract = !def.cases.is_empty() || def.kind == "page";
        let kind = if has_contract {
            TaskKind::SynthesizeFunction
        } else {
            TaskKind::AddDefinition
        };
        let mut def_task = SubTask::new(
            kind,
            format!("Add {} {}", def.kind, def.name),
            serde_json::json!({
                "file": target_file,
                "name": def.name,
                "kind": def.kind,
                "signature": def.signature.clone().unwrap_or_default(),
                "references": def.references,
                "cases": def.cases.iter().map(|c| serde_json::json!({"input": c.input, "expected": c.expected})).collect::<Vec<_>>(),
                "fields": def.fields.iter().map(|f| serde_json::json!({"name": f.name, "type": f.ty})).collect::<Vec<_>>(),
                "title": def.title,
                "sections": def.sections.iter().map(|s| serde_json::json!({"heading": s.heading, "body": s.body})).collect::<Vec<_>>(),
                "footer": def.footer,
                "methods": def.methods.iter().map(|m| serde_json::json!({
                    "name": m.name, "self": m.self_kind, "params": m.params,
                    "ret": m.ret, "op": m.op, "field": m.field, "amount": m.amount,
                })).collect::<Vec<_>>(),
            }),
            "intent_definition".to_string(),
        )
        .with_priority(0.7)
        .with_deadline(self.tick + 1000);

        // Add required symbols as needed
        def_task.required_symbols = def.references.clone();
        def_task.target_symbols.push(target_file.clone());

        tasks.push(def_task);

        // If this definition needs imports, add import tasks. Every task
        // carries its target file — untargeted tasks die in planning
        // ("No source file found") on projects without a src/*.rs fallback.
        for reference in &def.references {
            if !reference.contains("::") && !reference.contains('.') {
                continue;
            }
            let mut import_task = SubTask::new(
                TaskKind::AddImport,
                format!("Add import for {}", reference),
                serde_json::json!({"import": reference, "file": target_file}),
                "import_from_definition".to_string(),
            )
            .with_priority(0.6)
            .with_deadline(self.tick + 1000);
            import_task.target_symbols.push(target_file.clone());
            tasks.push(import_task);
        }

        tasks
    }

    /// Form a task for adding a test.
    fn form_test_task(&mut self, test: &IntentTest, intent: &StructuredIntent) -> Option<SubTask> {
        let description = format!("Add test: {}", test.name);
        let target_file = intent
            .file
            .clone()
            .unwrap_or_else(|| "src/lib.rs".to_string());
        let mut task = SubTask::new(
            TaskKind::AddTest,
            description,
            serde_json::json!({
                "file": target_file,
                "test": test.name,
                "assertions": test.assertions,
            }),
            "intent_test".to_string(),
        )
        .with_priority(0.6)
        .with_deadline(self.tick + 1000);
        task.target_symbols.push(target_file);

        Some(task)
    }

    /// Form a task from an intent action.
    fn form_action_task(
        &mut self,
        action: &IntentAction,
        intent: &StructuredIntent,
    ) -> Option<SubTask> {
        match action {
            IntentAction::Action {
                action,
                params,
                references,
            } => {
                let action_lower = action.to_lowercase();
                // GitHub delivery belongs to the publish actor (post-phase),
                // never to stub functions. Drop these so no junk lands.
                if action_lower.contains("github")
                    || action_lower.contains("repositor")
                    || action_lower.contains("publish")
                    || action_lower.contains("upload")
                {
                    log::info!(
                        "Skipping GitHub-flavored action (actor owns it): {}",
                        action
                    );
                    return None;
                }
                let target_file = intent
                    .file
                    .clone()
                    .unwrap_or_else(|| "src/main.rs".to_string());

                let (kind, description) =
                    if action_lower.contains("button") || action_lower.contains("vibrate") {
                        (
                            TaskKind::WriteFunction,
                            format!("Handle action: {}", action),
                        )
                    } else if action_lower.contains("test") || action_lower.contains("assert") {
                        (TaskKind::AddTest, format!("Test action: {}", action))
                    } else {
                        (
                            TaskKind::WriteFunction,
                            format!("Execute action: {}", action),
                        )
                    };

                let priority = 0.6;
                let mut task = SubTask::new(
                    kind,
                    description,
                    serde_json::json!({
                        "file": target_file,
                        "action": action,
                        "params": params,
                        "references": references,
                    }),
                    "intent_action".to_string(),
                )
                .with_priority(priority)
                .with_deadline(self.tick + 1000);

                task.target_symbols.push(target_file);
                // Extract any referenced symbols from params
                for r in references {
                    task.required_symbols.push(r.clone());
                }
                if let Some(refs) = params.first()
                    && !refs.is_empty()
                {
                    task.required_symbols.push(refs.clone());
                }

                Some(task)
            }
            IntentAction::Research { topic, reason: _ } => {
                let description = format!("Research: {}", topic);
                let task = SubTask::new(
                    TaskKind::ResearchRequirements,
                    description,
                    serde_json::json!({"topic": topic}),
                    "intent_research".to_string(),
                )
                .with_priority(0.3)
                .with_deadline(self.tick + 1000);

                Some(task)
            }
        }
    }

    /// Detect if the ratio of unresolved symbols is rising.
    /// This is a simplified version — grounded's version uses a ring buffer
    /// per node. Here we track a simple EMA across intents.
    #[allow(dead_code)]
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
    for d in intent.define.iter().flatten() {
        symbols.extend(d.references.clone());
    }
    symbols.extend(intent.imports.clone());
    symbols
}
