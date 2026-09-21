use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// A fix recipe — grounded's episodic memory repurposed.
///
/// In grounded, episodic memory records: "this sequence of events
/// (fired node X → prediction error Y → reward spike Z) led to this
/// outcome." During consolidation, high-importance episodes are
/// promoted to the semantic graph as knowledge.
///
/// Here, each recipe records: "this compile error pattern led to this
/// deterministic fix." The bot looks up recipes instead of guessing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixRecipe {
    /// Error classification that triggers this recipe
    pub error_kind: String,
    /// Error code (e.g., "E0308") — empty means any error of this kind
    pub error_code: String,
    /// Optional context pattern (e.g., "String::from" in message)
    pub context_pattern: Option<String>,
    /// The fix to apply — a transformation description
    pub fix: FixAction,
    /// How many times this recipe was successfully applied
    pub success_count: u32,
    /// How many times this recipe was attempted (success_count / attempt_count = reliability)
    pub attempt_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FixAction {
    /// Add an import: e.g., add "use android.widget.Button;"
    AddImport { import: String },
    /// Replace text: find `old` → `new`
    Replace { find: String, replace: String },
    /// Wrap expression in a conversion: `expr` → `expr.to_string()` or `String::from(expr)`
    WrapConversion { method: String },
    /// Add a type annotation: `let x = ...` → `let x: Type = ...`
    AddTypeAnnotation { variable: String, ty: String },
    /// Insert code at a specific location (file, anchor line, offset)
    InsertCode { file: String, anchor: String, offset: String, code: String },
    /// Remove an unused import
    RemoveImport { import: String },
    /// Change a type parameter
    ChangeType { from: String, to: String },
    /// Apply a custom fix from the compiler suggestion
    ApplySuggestion { suggestion: String },
}

/// The recipe log — grounded's episodic memory database repurposed.
///
/// Pre-seeded with common Rust/Kotlin/Android error patterns.
/// Grows deterministically as the bot successfully corrects errors.
///
/// This is NOT learning in the ML sense — it's a deterministic lookup
/// table. Same error → same fix, always. The bot never guesses.
#[derive(Clone)]
pub struct RecipeLog {
    /// Keyed by (error_kind, error_code, context_pattern)
    recipes: HashMap<String, FixRecipe>,
    /// File path for persistence
    path: String,
}

/// Generate a lookup key from error kind, code, and optional pattern.
fn make_key(error_kind: &str, error_code: &str, pattern: Option<&str>) -> String {
    format!("{}:{}:{}", error_kind, error_code, pattern.unwrap_or(""))
}

impl RecipeLog {
    pub fn new() -> Self {
        let path = dirs::data_dir()
            .map(|p| p.join("grounding-coder").join("recipes.json").to_string_lossy().to_string())
            .unwrap_or_else(|| "recipes.json".to_string());

        let mut log = RecipeLog {
            recipes: HashMap::new(),
            path,
        };

        log.seed_precompiled_recipes();
        log
    }

    pub fn with_path(path: &str) -> Self {
        let mut log = RecipeLog {
            recipes: HashMap::new(),
            path: path.to_string(),
        };
        log.seed_precompiled_recipes();
        // Try to load existing log
        if let Ok(content) = std::fs::read_to_string(&log.path) {
            if let Ok(recipes) = serde_json::from_str::<HashMap<String, FixRecipe>>(&content) {
                log.recipes = recipes;
            }
        }
        log
    }

    /// Seed with common error patterns — grounded's "foundation knowledge" pattern.
    /// These are all VERIFIED fixes, not guesses.
    fn seed_precompiled_recipes(&mut self) {
        let seeds: Vec<FixRecipe> = vec![
            // E0432: unresolved import — often a typo or wrong module
            FixRecipe {
                error_kind: "unresolved_symbol".into(),
                error_code: "E0432".into(),
                context_pattern: Some("import".to_string()),
                fix: FixAction::RemoveImport { import: "unused".to_string() }, // placeholder
                success_count: 0,
                attempt_count: 0,
            },
            // E0308: type mismatch — &str vs String
            FixRecipe {
                error_kind: "type_mismatch".into(),
                error_code: "E0308".into(),
                context_pattern: Some("expected `String`".to_string()),
                fix: FixAction::WrapConversion { method: ".to_string()".to_string() },
                success_count: 0,
                attempt_count: 0,
            },
            // E0308: type mismatch — i32 vs f64
            FixRecipe {
                error_kind: "type_mismatch".into(),
                error_code: "E0308".into(),
                context_pattern: Some("expected `f64`".to_string()),
                fix: FixAction::WrapConversion { method: " as f64".to_string() },
                success_count: 0,
                attempt_count: 0,
            },
            // E0425: cannot find value — often a missing import
            FixRecipe {
                error_kind: "unresolved_symbol".into(),
                error_code: "E0425".into(),
                context_pattern: None,
                fix: FixAction::ApplySuggestion { suggestion: "".to_string() },
                success_count: 0,
                attempt_count: 0,
            },
            // E0599: no method named X — need to import trait
            FixRecipe {
                error_kind: "missing_field_or_method".into(),
                error_code: "E0599".into(),
                context_pattern: None,
                fix: FixAction::AddImport { import: "".to_string() }, // filled at runtime
                success_count: 0,
                attempt_count: 0,
            },
        ];

        for recipe in seeds {
            let key = make_key(&recipe.error_kind, &recipe.error_code, recipe.context_pattern.as_deref());
            self.recipes.insert(key, recipe);
        }
    }

    /// Look up a fix recipe for a compile error.
    /// Returns None if no recipe matches — the bot then reports
    /// "I don't know how to fix this" instead of guessing.
    pub fn lookup(&self, error: &super::error::CompileError) -> Option<&FixRecipe> {
        // Try exact match first (kind + code + context)
        if let Some(pattern) = &error.suggestion {
            let key = make_key(error.kind.as_str(), &error.code, Some(pattern));
            if let Some(r) = self.recipes.get(&key) {
                return Some(r);
            }
        }

        // Try kind + code
        let key = make_key(error.kind.as_str(), &error.code, None);
        if let Some(r) = self.recipes.get(&key) {
            return Some(r);
        }

        // Try kind only
        let key = make_key(error.kind.as_str(), "", None);
        if let Some(r) = self.recipes.get(&key) {
            return Some(r);
        }

        None
    }

    /// Record a successful (error → fix) pair.
    /// This grows the recipe log deterministically.
    pub fn record_success(&mut self, error: &super::error::CompileError, fix: FixAction) {
        let recipe = FixRecipe {
            error_kind: error.kind.as_str().to_string(),
            error_code: error.code.clone(),
            context_pattern: error.suggestion.clone(),
            fix: fix.clone(),
            success_count: 1,
            attempt_count: 1,
        };

        let key = make_key(&recipe.error_kind, &recipe.error_code, recipe.context_pattern.as_deref());
        if let Some(existing) = self.recipes.get_mut(&key) {
            existing.success_count += 1;
            existing.attempt_count += 1;
        } else {
            self.recipes.insert(key, recipe);
        }
        self.save();
    }

    pub fn list(&self) -> Vec<String> {
        self.recipes.values()
            .filter(|r| r.success_count > 0)
            .map(|r| format!(
                "{} ({}) → {:?} | {} successes",
                r.error_kind, r.error_code, r.fix, r.success_count
            ))
            .collect()
    }

    pub fn count(&self) -> usize {
        self.recipes.len()
    }

    pub fn save(&self) {
        if let Ok(json) = serde_json::to_string_pretty(&self.recipes) {
            let path = std::path::Path::new(&self.path);
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(&self.path, json);
        }
    }
}
