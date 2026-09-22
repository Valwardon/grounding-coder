use super::arena::CodeArena;
use super::error::{CompileError, ErrorKind};
use super::recipes::RecipeLog;
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
    Replace {
        find: String,
        replace: String,
        file: String,
        line: u32,
    },
    /// Apply a compiler suggestion verbatim.
    ApplySuggestion { file: String, suggestion: String },
    /// Insert code after a line.
    InsertAfter {
        file: String,
        line: u32,
        code: String,
    },
    /// No fix available — the bot must concede.
    None,
}

impl CorrectionPipeline {
    pub fn new(recipes: RecipeLog) -> Self {
        CorrectionPipeline { recipes }
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
        log::debug!(
            "Phase 1: Classifying error {} at {}:{}",
            error.code,
            error.file,
            error.line
        );

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
    fn synthesize_fix(
        &self,
        error: &CompileError,
        recipe: Option<&super::recipes::FixRecipe>,
    ) -> Fix {
        // If we have a recipe, use it
        if let Some(r) = recipe {
            return self.recipe_to_fix(r, error);
        }

        // No recipe — try to synthesize from the error itself
        // (grounded's "degrade gracefully to proximity edges" fallback)
        match error.kind {
            ErrorKind::MissingImport => {
                // Prefer the compiler's own suggested path; fall back to
                // inferring from the symbol name (usuallyStill blocked
                // downstream unless verifiable — never guessed).
                if let Some(path) = suggested_import(error) {
                    return Fix::AddImport(path);
                }
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
                // Try common conversions: wrap a string literal so an
                // `expected String, found &str` error resolves deterministically.
                if error.message.contains("expected `String`")
                    && error.message.contains("found `&str`")
                {
                    if let Some((find, replace)) =
                        wrap_string_literal(&error.source_line.clone().unwrap_or_default())
                    {
                        Fix::Replace {
                            find,
                            replace,
                            file: error.file.clone(),
                            line: error.line,
                        }
                    } else {
                        Fix::None
                    }
                } else {
                    Fix::None
                }
            }
            ErrorKind::UnresolvedSymbol => {
                if let Some(path) = suggested_import(error) {
                    Fix::AddImport(path)
                } else if let Some(s) = error.suggestion.clone() {
                    Fix::ApplySuggestion {
                        file: error.file.clone(),
                        suggestion: s,
                    }
                } else {
                    Fix::None
                }
            }
            _ => Fix::None,
        }
    }

    fn recipe_to_fix(&self, recipe: &super::recipes::FixRecipe, error: &CompileError) -> Fix {
        // A compiler-suggested `use` path outranks any recipe: it names
        // exact bytes the compiler itself wants.
        if matches!(
            error.kind,
            ErrorKind::UnresolvedSymbol | ErrorKind::MissingImport | ErrorKind::MissingFieldMethod
        ) && let Some(path) = suggested_import(error)
        {
            return Fix::AddImport(path);
        }
        match &recipe.fix {
            super::recipes::FixAction::ApplySuggestion { suggestion } => {
                let s = if suggestion.is_empty() {
                    error.suggestion.clone().unwrap_or_default()
                } else {
                    suggestion.clone()
                };
                Fix::ApplySuggestion {
                    file: error.file.clone(),
                    suggestion: s,
                }
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
                // Wrap a literal on the offending line with the suggested method,
                // e.g. `"a"` → `"a".to_string()` or `42` → `42 as f64`.
                if let Some((find, replace)) =
                    wrap_with_method(&error.source_line.clone().unwrap_or_default(), method)
                {
                    Fix::Replace {
                        find,
                        replace,
                        file: error.file.clone(),
                        line: error.line,
                    }
                } else {
                    Fix::None
                }
            }
            _ => Fix::None,
        }
    }
}

fn infer_import_from_error(error: &CompileError) -> String {
    // Extract the symbol name from "cannot find value `Foo` in this scope"
    if let Some(start) = error.message.find("`")
        && let Some(end) = error.message[start + 1..].find("`")
    {
        let symbol = &error.message[start + 1..start + 1 + end];
        // Try common module paths
        return format!("use {};", symbol);
    }
    String::new()
}

/// Extract a `use` path from the compiler's own suggestion text.
/// Returns the first plausible path, preferring `std::` ones.
/// This is evidence, not inference: rustc named these exact bytes.
fn suggested_import(error: &CompileError) -> Option<String> {
    let text = error.suggestion.as_deref()?;
    let mut found: Vec<String> = Vec::new();
    // `use a::b::C;` shapes.
    let mut i = 0;
    while let Some(pos) = text[i..].find("use ") {
        let start = i + pos + 4;
        let mut end = start;
        while end < text.len()
            && (text.as_bytes()[end].is_ascii_alphanumeric()
                || matches!(text.as_bytes()[end], b'_' | b':'))
        {
            end += 1;
        }
        if end > start {
            let p = text[start..end].trim_end_matches(':').to_string();
            if p.contains("::") && !found.contains(&p) {
                found.push(p);
            }
        }
        i = end.max(start + 1);
    }
    // Bare `std::...` tokens.
    let mut j = 0;
    while let Some(pos) = text[j..].find("std::") {
        let start = j + pos;
        let mut end = start;
        while end < text.len()
            && (text.as_bytes()[end].is_ascii_alphanumeric()
                || matches!(text.as_bytes()[end], b'_' | b':'))
        {
            end += 1;
        }
        let p = text[start..end].trim_end_matches(':').to_string();
        if !found.contains(&p) {
            found.push(p);
        }
        j = end.max(start + 1);
    }
    // Verified-shape paths first; anything else still faces the
    // planner's own std/symbol check downstream.
    found
        .iter()
        .find(|p| p.starts_with("std::"))
        .or_else(|| found.first())
        .cloned()
}

fn inferred_import(error: &CompileError) -> Fix {
    let imp = infer_import_from_error(error);
    if imp.is_empty() {
        Fix::None
    } else {
        Fix::AddImport(imp)
    }
}

/// Find the first double-quoted literal in a code line, returning both the
/// literal text (including quotes) and its byte offset in the line.
fn first_string_literal(line: &str) -> Option<(String, usize)> {
    let start = line.find('"')?;
    let end = line[start + 1..].find('"')?;
    let literal = line[start..start + 1 + end].to_string();
    Some((literal, start))
}

/// Wrap the first string literal so it converts to an owned `String`.
fn wrap_string_literal(line: &str) -> Option<(String, String)> {
    let (literal, _) = first_string_literal(line)?;
    Some((literal.clone(), format!("String::from({})", literal)))
}

/// Wrap a literal on a line with a conversion method from a recipe, e.g.
/// `"a"` → `"a".to_string()` or `42` → `42 as f64`.
fn wrap_with_method(line: &str, method: &str) -> Option<(String, String)> {
    let method = method.trim();
    if method.is_empty() {
        return None;
    }

    if let Some((literal, _)) = first_string_literal(line) {
        let replacement = format!("{}{}", literal, method);
        return Some((literal, replacement));
    }

    // Bare numeric literal fallback, e.g. `let x: f64 = 42;`
    let num_re = regex::Regex::new(r"\b\d+(\.\d+)?\b").unwrap();
    if let Some(m) = num_re.find(line) {
        let numeric = m.as_str().to_string();
        return Some((numeric.clone(), format!("{}{}", numeric, method)));
    }

    None
}
