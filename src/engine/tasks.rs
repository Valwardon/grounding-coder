use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
use crate::engine::arena::{CodeArena, SymbolId, SymbolRelation};

/// Priority weight for error magnitude (Eₚ) in the priority formula.
/// Adapted from grounded's GoalFormationEngine ALPHA.
const ALPHA: f64 = 0.35;
/// Priority weight for novelty level (N) in the priority formula.
/// Adapted from grounded's GoalFormationEngine BETA.
const BETA: f64 = 0.25;
/// Priority weight for unresolved symbol ratio (U) — adapts γ (drive deprivation).
/// High unresolved ratio → high priority to focus on resolution.
const GAMMA: f64 = 0.25;
/// Priority weight for budget pressure (B) — adapts δ (systemic inefficiency).
/// Low remaining budget → high priority to complete quickly.
const DELTA: f64 = 0.15;

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
            self.consecutive_unresolved = (self.consecutive_unresolved + 1).min(255);
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

    /// Rising trend: last 3 unresolved counts increasing.
    pub fn rising_trend(&self) -> bool {
        if self.count < 4 {
            return false;
        }
        let i0 = (self.write_idx + ERROR_HISTORY_SIZE - 1) % ERROR_HISTORY_SIZE;
        let i1 = (self.write_idx + ERROR_HISTORY_SIZE - 2) % ERROR_HISTORY_SIZE;
        let i2 = (self.write_idx + ERROR_HISTORY_SIZE - 3) % ERROR_HISTORY_SIZE;
        (self.counts[i2] as f64) < (self.counts[i1] as f64) && (self.counts[i1] as f64) < (self.counts[i0] as f64)
    }
}

/// A tracked symbol that the error history applies to.
#[derive(Debug, Clone)]
struct TrackedSymbol {
    id: SymbolId,
    history: SymbolErrorHistory,
    /// Whether we've already formed a "resolve" task for this symbol.
    task_formed: bool,
}

/// A concrete sub-task decomposed from a structured intent.
/// This is the code equivalent of grounded's `GoalNode` — but instead of
/// a semantic goal like "understand_node_42", it's a actionable code task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubTask {
    /// Auto-incremented task ID (like grounded's node ID)
    pub id: u64,
    /// What kind of task this is
    pub kind: TaskKind,
    /// Human-readable description
    pub description: String,
    /// The symbol(s) this task operates on
    pub target_symbols: Vec<String>,
    /// Required imports/references needed
    pub required_symbols: Vec<String>,
    /// Priority: clamped [0.05, 1.0] — higher = more urgent
    pub priority: f64,
    /// Source of this task (intent field, error pattern, etc.)
    pub source: String,
    /// Structured data for the CodeWriter to consume
    pub payload: serde_json::Value,
    /// Deadlines: tick + N (like grounded's deadline_tick)
    pub deadline: u64,
}

impl SubTask {
    pub fn new(kind: TaskKind, description: String, payload: serde_json::Value, source: String) -> Self {
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

/// Reasons a sub-task was formed (mirrors grounded's GoalReason).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskReason {
    /// Symbol persistently unresolved (grounded: PersistentPredictionError)
    PersistentUnresolved,
    /// Unresolved count rising (grounded: RisingErrorTrend)
    RisingUnresolved,
    /// High fraction of unresolved symbols in the intent (grounded: DriveDeprivation)
    ResolutionNeeded,
    /// Direct field in structured intent (grounded: none — explicit goal)
    ExplicitIntent,
    /// Budget pressure — need to finish before retries exhaust (grounded: SystemicInefficiency)
    BudgetPressure,
}

impl TaskReason {
    pub fn description(&self) -> &'static str {
        match self {
            TaskReason::PersistentUnresolved => "Symbol persistently unresolved",
            TaskReason::RisingUnresolved => "Rising unresolved count on symbol",
            TaskReason::ResolutionNeeded => "High unresolved symbol ratio in intent",
            TaskReason::ExplicitIntent => "Explicit field in structured intent",
            TaskReason::BudgetPressure => "Budget pressure — urgent completion needed",
        }
    }
}

/// The TaskDecomposer — grounded's `GoalFormationEngine` repurposed for code.
///
/// In grounded, the GFE monitors prediction errors on semantic graph nodes,
/// tracks per-node error history, and forms "understand_<node>" goals when
/// a node has persistent high prediction error.
///
/// Here, the TaskDecomposer takes a structured intent JSON (produced by the
/// unverified LLM layer), scans it for code symbols that reference things
/// the bot doesn't know (unresolved), and forms concrete sub-tasks:
///   - "resolve_symbol: android.widget.Button" (form a resolve task)
///   - "write_function: fun vibrateButton(button: Button)" (form a write task)
///   - "add_import: import android.os.Vibrator" (form an import task)
///
/// Pure deterministic — uses the same error history ring buffer, EMA trend
/// detection, and priority formula as grounded's GFE.
pub struct TaskDecomposer {
    /// Per-symbol error history (indexed by SymbolId.0).
    error_histories: Vec<TrackedSymbol>,
    /// Total sub-tasks formed (monotonic counter).
    task_count: u64,
    /// Tick counter for deadline assignment.
    tick: u64,
    /// Current budget remaining (fraction of total). Used for budget pressure tasks.
    budget_remaining: f64,
    /// Total budget (for calculating pressure).
    budget_total: f64,
}

/// Structured intent format — what the LLM translator outputs (unverified layer).
/// This is the ONLY thing the LLM produces. Everything else is deterministic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredIntent {
    /// High-level description of what the human wants.
    pub goal: String,
    /// Target file path (relative to project root).
    pub file: String,
    /// Language target: "kotlin", "rust", etc.
    pub language: String,
    /// Actions to perform.
    pub actions: Vec<IntentAction>,
    /// Symbols referenced that need to be resolved.
    pub references: Vec<String>,
    /// Optional: new symbol to define.
    pub define: Option<IntentDefinition>,
    /// Optional: imports to add.
    pub imports: Vec<String>,
    /// Optional: test to write.
    pub test: Option<IntentTest>,
    /// Project platform: "android", "desktop", "web"
    pub platform: String,
    /// Project architecture: "native", "cross-platform", "hybrid"
    pub architecture: String,
    /// Target runtime SDK (Android)
    pub runtime: Option<RuntimeRequirements>,
    /// Capability requirements (network, filesystem, background execution)
    pub capabilities: Vec<String>,
    /// Domains this code operates in
    pub domains: Vec<String>,
    /// Explicit constraints (security, performance, etc.)
    pub constraints: Vec<String>,
    /// Dependencies (crates, libraries, packages)
    pub dependencies: Vec<String>,
    /// Requirements hypothesis from LLM - what we don't know yet
    pub unknown_requirements: Vec<String>,
    /// Confidence level in the intent completeness (0.0-1.0)
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentAction {
    /// The action type (e.g., "add_button", "wire_vibrator", "call_function")
    pub action: String,
    /// Parameters for this action (e.g., { "text": "Click me", "on_click": "vibrate" })
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentDefinition {
    /// Name of the new symbol (e.g., "fun vibrateButton")
    pub name: String,
    /// Type of symbol: "function", "struct", "enum", etc.
    pub kind: String,
    /// Full code/text for this definition
    pub code: String,
    /// Symbols this definition references (needs resolution)
    pub references: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentTest {
    /// Test function name
    pub name: String,
    /// What to test (assertion descriptions)
    pub assertions: Vec<String>,
    /// Code for the test
    pub code: String,
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

    pub fn with_budget(total: f64) -> Self {
        TaskDecomposer {
            error_histories: Vec::with_capacity(64),
            task_count: 0,
            tick: 0,
            budget_remaining: total,
            budget_total: total,
        }
    }

    /// Set the current budget remaining (for budget-pressure tasks).
    pub fn set_budget_remaining(&mut self, remaining: f64) {
        self.budget_remaining = remaining;
    }

    /// The core decompose method — grounded's `tick()` repurposed.
    ///
    /// Takes a structured intent JSON string and breaks it into concrete
    /// sub-tasks. This is deterministic: same intent → same tasks (given
    /// the same error history state).
    ///
    /// Algorithm (mirrors grounded's GFE tick):
    ///   1. Parse the structured intent JSON
    ///   2. Scan references for unknown symbols → track error history
    ///   3. Check each reference for persistent unresolvability → form resolve task
    ///   4. Check for rising unresolved trend → form investigate task
    ///   5. Form explicit tasks from intent actions/imports/definitions
    ///   6. Check budget pressure → elevate remaining tasks
    pub fn decompose(&mut self, intent_json: &str, arena: &CodeArena) -> Vec<SubTask> {
        self.tick += 1;

        let intent: StructuredIntent = match serde_json::from_str(intent_json) {
            Ok(i) => i,
            Err(e) => {
                log::error!("Failed to parse intent JSON: {}", e);
                return Vec::new();
            }
        };

        let mut tasks = Vec::new();
        let mut resolve_needed = Vec::new();
        let mut total_refs = 0u32;
        let mut unresolved_refs = 0u32;

        // 1. Scan all referenced symbols for resolution status
        for ref_name in &intent.references {
            total_refs += 1;
            if arena.lookup(ref_name).is_none() {
                unresolved_refs += 1;
                resolve_needed.push(ref_name.clone());
            }
        }

        // Also check references in the definition
        if let Some(def) = &intent.define {
            for ref_name in &def.references {
                total_refs += 1;
                if arena.lookup(ref_name).is_none() {
                    unresolved_refs += 1;
                    resolve_needed.push(ref_name.clone());
                }
            }
        }

        // 2. Update error histories for unresolved references
        for ref_name in &resolve_needed {
            let _sym_id = arena.lookup(ref_name);
            let ema = unresolved_refs as f64 / total_refs.max(1) as f64;
            let persistence = ema;

            // Form a resolve task for this specific symbol
            let task = self.form_resolve_task(ref_name, ema, persistence, intent.goal.clone());
            tasks.push(task);
        }

        // 3. Check for rising unresolved trend across the intent
        if self.detect_rising_trend(total_refs, unresolved_refs) {
            let priority = (ALPHA * (unresolved_refs as f64 / total_refs.max(1) as f64)
                + BETA * (unresolved_refs as f64 / total_refs.max(1) as f64))
                .clamp(0.05, 1.0);
            tasks.push(SubTask::new(
                TaskKind::ResolveSymbol,
                format!("Investigate rising unresolved trend in intent: {}", intent.goal),
                serde_json::json!({
                    "references": resolve_needed,
                    "total": total_refs,
                    "unresolved": unresolved_refs
                }),
                "trend_detection".to_string(),
            ).with_priority(priority)
             .with_deadline(self.tick + 2000));
        }

        // 4. Form explicit tasks from intent actions
        for action in &intent.actions {
            if let Some(task) = self.form_action_task(action, &intent) {
                tasks.push(task);
            }
        }

        // 5. Form import tasks
        for import_path in &intent.imports {
            let priority = GAMMA * (unresolved_refs as f64 / total_refs.max(1) as f64);
            tasks.push(SubTask::new(
                TaskKind::AddImport,
                format!("Add import: {}", import_path),
                serde_json::json!({ "import": import_path }),
                "intent_imports".to_string(),
            ).with_priority(priority.clamp(0.05, 1.0))
             .with_deadline(self.tick + 1500));
        }

        // 6. Form definition (write function) tasks
        if let Some(def) = &intent.define {
            let priority = ALPHA * 0.8 + GAMMA * (unresolved_refs as f64 / total_refs.max(1) as f64);
            let mut task = SubTask::new(
                TaskKind::WriteFunction,
                format!("Define {} ({})", def.name, def.kind),
                serde_json::json!({
                    "name": def.name,
                    "kind": def.kind,
                    "code": def.code,
                    "file": intent.file
                }),
                "intent_define".to_string(),
            ).with_priority(priority.clamp(0.05, 1.0))
             .with_deadline(self.tick + 1000);
            task.target_symbols = vec![def.name.clone()];
            task.required_symbols = def.references.clone();
            tasks.push(task);
        }

    // 7. Form research requirements tasks
    if !intent.unknown_requirements.is_empty() {
        let priority = BETA * (unresolved_refs as f64 / total_refs.max(1) as f64);
        let mut task = SubTask::new(
            TaskKind::ResearchRequirements,
            format!("Research requirements: {}", intent.goal),
            serde_json::json!({
                "goal": intent.goal,
                "unknown_requirements": intent.unknown_requirements,
                "domains": intent.domains,
                "platform": intent.platform,
                "architecture": intent.architecture,
                "capabilities": intent.capabilities,
                "constraints": intent.constraints,
                "dependencies": intent.dependencies,
                "runtime": intent.runtime
            }),
            "intent_requirements".to_string(),
        ).with_priority(priority.clamp(0.05, 1.0))
         .with_deadline(self.tick + 3000);
        tasks.push(task);
    }

    // 8. Form create project manifest tasks
    if intent.platform == "android" || intent.architecture == "native" {
        let priority = GAMMA * 0.8;
        let mut task = SubTask::new(
            TaskKind::CreateProjectManifest,
            format!("Create project manifest for {} {}", intent.platform, intent.architecture),
            serde_json::json!({
                "platform": intent.platform,
                "architecture": intent.architecture,
                "runtime": intent.runtime,
                "capabilities": intent.capabilities,
                "domains": intent.domains,
                "constraints": intent.constraints,
                "dependencies": intent.dependencies,
                "file": "Cargo.toml"
            }),
            "intent_manifest".to_string(),
        ).with_priority(priority.clamp(0.05, 1.0))
         .with_deadline(self.tick + 2000);
        tasks.push(task);
    }

    // 9. Sort by priority (descending) — like grounded's priority-based selection
    tasks.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));

        tasks
    }

    /// Form a resolve_task for an unresolved symbol.
    /// Mirrors grounded's "understand_<node>" goal formation.
    fn form_resolve_task(
        &mut self,
        symbol: &str,
        ema_error: f64,
        _persistence: f64,
        intent_goal: String,
    ) -> SubTask {
        let priority = (ALPHA * ema_error + BETA * 0.5).clamp(0.05, 1.0);
        let mut task = SubTask::new(
            TaskKind::ResolveSymbol,
            format!("Resolve unknown symbol: {}", symbol),
            serde_json::json!({ "symbol": symbol }),
            format!("intent: {}", intent_goal),
        ).with_priority(priority)
         .with_deadline(self.tick + 5000);
        task.target_symbols = vec![symbol.to_string()];
        task
    }

    /// Form a task from an intent action.
    /// Maps intent action strings (e.g., "add_button") to concrete sub-tasks.
    fn form_action_task(&mut self, action: &IntentAction, intent: &StructuredIntent) -> Option<SubTask> {
        let action_lower = action.action.to_lowercase();

        let (kind, description, payload) = if action_lower.contains("button") || action_lower.contains("vibrate") {
            (
                TaskKind::WriteFunction,
                format!("Handle action: {}", action.action),
                serde_json::json!({
                    "action": action.action,
                    "params": action.params,
                    "file": intent.file,
                    "language": intent.language
                }),
            )
        } else if action_lower.contains("test") || action_lower.contains("assert") {
            (
                TaskKind::AddTest,
                format!("Test action: {}", action.action),
                serde_json::json!({
                    "action": action.action,
                    "params": action.params,
                    "file": intent.file
                }),
            )
        } else {
            (
                TaskKind::WriteFunction,
                format!("Execute action: {}", action.action),
                serde_json::json!({
                    "action": action.action,
                    "params": action.params,
                    "file": intent.file,
                    "language": intent.language
                }),
            )
        };

        let priority = 0.6;
        let mut task = SubTask::new(kind, description, payload, "intent_action".to_string())
            .with_priority(priority)
            .with_deadline(self.tick + 1000);

        // Extract any referenced symbols from params
        if let Some(refs) = action.params.get("references") {
            if let Some(arr) = refs.as_array() {
                for r in arr {
                    if let Some(s) = r.as_str() {
                        task.required_symbols.push(s.to_string());
                    }
                }
            }
        }

        Some(task)
    }

    /// Detect if the ratio of unresolved symbols is rising.
    /// This is a simplified version — grounded's version uses a ring buffer
    /// per node. Here we track a simple EMA across intents.
    fn detect_rising_trend(&mut self, total: u32, unresolved: u32) -> bool {
        let _ratio = unresolved as f64 / total.max(1) as f64;

        // Track in error histories vector (repurposing the pattern)
        if self.error_histories.is_empty() {
            self.error_histories.resize(1, TrackedSymbol {
                id: SymbolId(0),
                history: SymbolErrorHistory::new(),
                task_formed: false,
            });
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
        symbols.extend(def.references.clone());
    }
    symbols
}
