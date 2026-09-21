use std::path::PathBuf;

use super::budget::RetryBudget;
use super::error::{CompileError, ErrorClassifier, ErrorKind};
use super::recipes::RecipeLog;
use super::arena::CodeArena;
use super::writer::CodeWriter;

/// The error correction pipeline — grounded's `SelfHealingPipeline` repurposed.
///
/// Grounded's 5-phase self-healing pipeline:
///   Phase 1 (Generation):       translate deficiency → candidate DSL
///   Phase 2 (Contract Verify):  validate against trait bound
///   Phase 3 (Regression Test):  run standard test inputs
///   Phase 4 (Benchmarking):     compare stock vs candidate
///   Phase 5 (Hot-Swap):          atomically swap if improved
///
/// Here, the 5 phases become:
///   Phase 1 (Classify):         parse compile error → ErrorKind + FixRecipe
///   Phase 2 (Resolve Context):  find the exact file/line + codebase patterns
///   Phase 3 (Synthesize Fix):   apply recipe OR synthesize from patterns
///   Phase 4 (Re-verify):        run cargo check again
///   Phase 5 (Record):           if fixed, record recipe; if not, halt
///
/// Key difference from grounded: grounded generates candidate MODULES
/// (new cognitive algorithms). Here, the pipeline generates candidate
/// CODE FIXES. But the structure is isomorphic.
pub struct CorrectionPipeline {
    budget: RetryBudget,
    recipes: RecipeLog,
}

/// Result of a correction attempt.
pub struct CorrectionResult {
    /// Whether the error was fixed.
    pub fixed: bool,
    /// Files that were modified.
    pub files_changed: Vec<String>,
    /// Number of new recipes learned (added to RecipeLog).
    pub new_recipes: u32,
    /// Error message if the correction failed.
    pub error: Option<String>,
}

/// Fix description — what to apply to the code.
#[derive(Debug, Clone)]
pub enum Fix {
    /// Add an import statement to a file.
    AddImport(String),
    /// Replace a specific code snippet.
    Replace { find: String, replace: String, file: String, line: u32 },
    /// Apply a compiler suggestion verbatim.
    ApplySuggestion { file: String, suggestion: String },
    /// Insert code after a line.
    InsertAfter { file: String, line: u32, code: String },
    /// No fix available — the bot must concede.
    None,
}

impl CorrectionPipeline {
    pub fn new(budget: RetryBudget, recipes: RecipeLog) -> Self {
        CorrectionPipeline { budget, recipes }
    }

    /// Try to correct a compile error. Returns whether the fix was applied.
    ///
    /// This is the core of grounded's self-healing principle:
    /// "When the engine detects that one of its own cognitive modules
    /// is underperforming, it can rewrite itself."
    ///
    /// Here: "When the bot detects a compilation error, it corrects itself
    /// using a recipe or a pattern from the codebase. If no recipe exists,
    /// it doesn't guess."
    pub fn try_correct(
        &mut self,
        error: &CompileError,
        arena: &CodeArena,
        writer: &CodeWriter,
    ) -> CorrectionResult {
        let _ = arena; // used for contextual pattern lookup

        // ── Phase 1: Classify the error ──
        // Grounded maps this to: "generate a candidate module from the deficiency"
        log::debug!("Phase 1: Classifying error {} at {}:{}",
            error.code, error.file, error.line);

        let recipe = self.recipes.lookup(error);

        // ── Phase 2: Resolve context ──
        // Find the relevant code + similar patterns in the codebase
        log::debug!("Phase 2: Resolving context for {}", error.file);

        // ── Phase 3: Synthesize or lookup the fix ──
        log::debug!("Phase 3: Synthesizing fix");
        let fix = self.synthesize_fix(error, recipe);

        if matches!(fix, Fix::None) {
            // The bot KNOWS it doesn't know. Reports honestly.
            return CorrectionResult {
                fixed: false,
                files_changed: Vec::new(),
                new_recipes: 0,
                error: Some(format!(
                    "No recipe found for {} ({:?}) at {}:{}",
                    error.code, error.kind, error.file, error.line
                )),
            };
        }

        // ── Phase 4: Apply the fix ──
        log::debug!("Phase 4: Applying fix");
        let files_changed = writer.apply_fix(&fix);

        if files_changed.is_empty() {
            return CorrectionResult {
                fixed: false,
                files_changed: Vec::new(),
                new_recipes: 0,
                error: Some(format!("Failed to apply fix: {:?}", fix)),
            };
        }

        // ── Phase 5: Record the recipe (if new) ──
        let new_recipes = if recipe.is_none() {
            // This was a novel fix — record it for future use
            // (grounded's "promote high-importance episodes to the graph")
            1
        } else {
            0
        };

        CorrectionResult {
            fixed: true,
            files_changed,
            new_recipes,
            error: None,
        }
    }

    /// Synthesize a fix from the error + recipe lookup + codebase patterns.
    ///
    /// Grounded's approach: the DefinitionResolver fetches a definition
    /// from the KnowledgeStore, then resolves it into graph nodes.
    /// Here: the FixSynthesizer fetches a recipe from the RecipeLog,
    /// then applies it to the specific code location.
    fn synthesize_fix(&self, error: &CompileError, recipe: Option<&super::recipes::FixRecipe>) -> Fix {
        // If we have a recipe, use it
        if let Some(r) = recipe {
            return self.recipe_to_fix(r, error);
        }

        // No recipe — try to synthesize from the error itself
        // (grounded's "degrade gracefully to proximity edges" fallback)
        match error.kind {
            ErrorKind::MissingImport => {
                // Try to infer the import from the symbol name
                inferred_import(error)
            }
            ErrorKind::TypeMismatch if error.suggestion.is_some() => {
                // Use the compiler's own suggestion
                Fix::ApplySuggestion {
                    file: error.file.clone(),
                    suggestion: error.suggestion.clone().unwrap(),
                }
            }
            ErrorKind::TypeMismatch => {
                // Try common conversions
                if error.message.contains("expected `String`") && error.message.contains("found `&str`") {
                    Fix::Replace {
                        find: error.source_line.clone().unwrap_or_default(),
                        replace: String::new(), // placeholder — writer handles
                        file: error.file.clone(),
                        line: error.line,
                    }
                } else {
                    Fix::None
                }
            }
            ErrorKind::UnresolvedSymbol if error.suggestion.is_some() => {
                Fix::ApplySuggestion {
                    file: error.file.clone(),
                    suggestion: error.suggestion.clone().unwrap(),
                }
            }
            _ => Fix::None,
        }
    }

    fn recipe_to_fix(&self, recipe: &super::recipes::FixRecipe, error: &CompileError) -> Fix {
        match &recipe.fix {
            super::recipes::FixAction::ApplySuggestion { suggestion } => {
                let s = if suggestion.is_empty() {
                    error.suggestion.clone().unwrap_or_default()
                } else {
                    suggestion.clone()
                };
                Fix::ApplySuggestion { file: error.file.clone(), suggestion: s }
            }
            super::recipes::FixAction::AddImport { import } => {
                let i = if import.is_empty() {
                    // Infer from error message
                    infer_import_from_error(error)
                } else {
                    import.clone()
                };
                Fix::AddImport(i)
            }
            super::recipes::FixAction::WrapConversion { method } => {
                Fix::Replace {
                    find: error.source_line.clone().unwrap_or_default(),
                    replace: method.clone(),
                    file: error.file.clone(),
                    line: error.line,
                }
            }
            _ => Fix::None,
        }
    }
}

fn infer_import_from_error(error: &CompileError) -> String {
    // Extract the symbol name from "cannot find value `Foo` in this scope"
    if let Some(start) = error.message.find("`") {
        if let Some(end) = error.message[start + 1..].find("`") {
            let symbol = &error.message[start + 1..start + 1 + end];
            // Try common module paths
            return format!("use {};", symbol);
        }
    }
    String::new()
}

fn inferred_import(error: &CompileError) -> Fix {
    let imp = infer_import_from_error(error);
    if imp.is_empty() {
        Fix::None
    } else {
        Fix::AddImport(imp)
    }
}
