/// A structured compiler error extracted from `cargo check` / `cargo clippy` output.
///
/// This is the deterministic oracle — the compiler never lies.
/// Every error has a code, location, message, and optional suggestion.
/// The bot uses this to look up a fix recipe (never guesses).
#[derive(Debug, Clone)]
pub struct CompileError {
    /// Error code, e.g. "E0308", "E0425", "warning:unused_import"
    pub code: String,
    /// Human-readable error message
    pub message: String,
    /// Source file path
    pub file: String,
    /// Line number (1-indexed)
    pub line: u32,
    /// Column number
    pub col: u32,
    /// The suggested fix from the compiler (if any)
    pub suggestion: Option<String>,
    /// The source line that triggered the error
    pub source_line: Option<String>,
    /// Classification — grounded's semantic-category approach repurposed
    pub kind: ErrorKind,
}

/// Error categories — enables deterministic recipe lookup.
///
/// Just as grounded's CCG parser classifies tokens into semantic
/// categories (Entity, Action, Property, Relation), the error
/// classifier categorizes compiler errors so the CorrectionPipeline
/// knows which recipe to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// Type mismatch (E0308) — wrong type where another was expected
    TypeMismatch,
    /// Unresolved symbol / import (E0425, E0432, E0433)
    UnresolvedSymbol,
    /// Wrong number of arguments (E0589, E0061)
    WrongArity,
    /// Missing import
    MissingImport,
    /// Lifetime/borrow error (E0507, E0597)
    BorrowError,
    /// Missing method/field (E0599, E0609)
    MissingFieldMethod,
    /// Duplicate definition
    DuplicateDefinition,
    /// Trait bound not satisfied (E0271, E0277)
    TraitBound,
    /// Generic type parameter error
    GenericError,
    /// Other / unclassified
    Other,
}

impl ErrorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorKind::TypeMismatch => "type_mismatch",
            ErrorKind::UnresolvedSymbol => "unresolved_symbol",
            ErrorKind::WrongArity => "wrong_arity",
            ErrorKind::MissingImport => "missing_import",
            ErrorKind::BorrowError => "borrow_error",
            ErrorKind::MissingFieldMethod => "missing_field_or_method",
            ErrorKind::DuplicateDefinition => "duplicate_definition",
            ErrorKind::TraitBound => "trait_bound",
            ErrorKind::GenericError => "generic_error",
            ErrorKind::Other => "other",
        }
    }
}

/// Deterministic compiler error parser — grounded's structural error
/// detection repurposed for code.
///
/// Parses `rustc`/`cargo check` output format:
/// ```text
/// error[E0308]: mismatched types
///   --> src/main.rs:10:15
///    |
/// 10 |     let x: String = 42;
///    |              ^^   --   expected `String`,
///    |                      found `i32`
///    |                      expected due to this
/// help: try `String::from(42)` or `42.to_string()`
/// ```
pub struct ErrorClassifier;

impl ErrorClassifier {
    /// Parse all errors from a cargo/rustc compiler output string.
    pub fn parse(output: &str) -> Vec<CompileError> {
        let mut errors = Vec::new();
        let lines: Vec<&str> = output.lines().collect();

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();

            // Match: error[E0308]: message  OR  warning[code]: message
            if let Some(cap) = parse_error_header(line) {
                let (code, message, _is_warning) = cap;
                let kind = classify_error(&code, &message);

                // Look for the --> location on the next line
                let mut file = String::new();
                let mut line_no = 0u32;
                let mut col = 0u32;
                if i + 1 < lines.len()
                    && let Some(loc) = parse_location(lines[i + 1])
                {
                    file = loc.0;
                    line_no = loc.1;
                    col = loc.2;
                }

                // Look for source line (line with |)
                let mut source_line = None;
                if line_no > 0 && i + 3 < lines.len() {
                    source_line = Some(lines[i + 3].to_string());
                }

                // Look for help/suggestion, including rustc's proposed
                // code insertions (`1 + use std::collections::HashMap;`).
                // The insertion line is first-class evidence: it is the
                // compiler telling us the exact bytes it wants.
                let mut suggestion = None;
                let mut j = i + 1;
                while j < lines.len() && j < i + 20 {
                    let clean = lines[j].trim();
                    if clean.starts_with("help:") || clean.starts_with("note:") {
                        suggestion = Some(
                            clean
                                .trim_start_matches("help:")
                                .trim_start_matches("note:")
                                .trim()
                                .to_string(),
                        );
                    } else if let Some(inserted) = parse_suggestion_insert(lines[j]) {
                        suggestion = Some(match suggestion {
                            Some(prev) => format!("{} | suggested: {}", prev, inserted),
                            None => format!("suggested: {}", inserted),
                        });
                    }
                    if clean.starts_with("error[") || clean.starts_with("warning[") {
                        break;
                    }
                    j += 1;
                }

                errors.push(CompileError {
                    code,
                    message,
                    file,
                    line: line_no,
                    col,
                    suggestion,
                    source_line,
                    kind,
                });
                i = j;
                continue;
            }

            // Match: error: message (no error code). rustc puts the
            // location on a following `--> file:line:col` line — look
            // ahead for it so diagnostics stay actionable. Without a
            // location the repair loop cannot anchor a fix and blocks.
            if line.starts_with("error:") {
                let message = line.trim_start_matches("error:").trim().to_string();
                let kind = classify_error("", &message);
                let (mut file, mut line_no, mut col) = (String::new(), 0u32, 0u32);
                let mut j = i + 1;
                while j < lines.len() && j < i + 6 {
                    if let Some(loc) = parse_location(lines[j]) {
                        file = loc.0;
                        line_no = loc.1;
                        col = loc.2;
                        break;
                    }
                    // A new diagnostic starts: stop looking.
                    let t = lines[j].trim();
                    if t.starts_with("error") || t.starts_with("warning") {
                        break;
                    }
                    j += 1;
                }
                errors.push(CompileError {
                    code: "error".to_string(),
                    message,
                    file,
                    line: line_no,
                    col,
                    suggestion: None,
                    source_line: None,
                    kind,
                });
                i += 1;
                continue;
            }

            i += 1;
        }

        errors
    }

    /// Parse a single error string (non-batch mode).
    pub fn parse_single(line: &str) -> Option<CompileError> {
        if let Some((code, message, _)) = parse_error_header(line) {
            let kind = classify_error(&code, &message);
            Some(CompileError {
                code,
                message,
                file: String::new(),
                line: 0,
                col: 0,
                suggestion: None,
                source_line: None,
                kind,
            })
        } else {
            None
        }
    }
}

/// Parse `error[E0308]: mismatched types` or `warning[unused]: ...` header.
fn parse_error_header(line: &str) -> Option<(String, String, bool)> {
    // error[E0308]: message (the code inside brackets already has its E)
    if let Some(rest) = line.strip_prefix("error[")
        && let Some(close) = rest.find(']')
    {
        let code = rest[..close].to_string();
        let msg = rest[close + 1..].trim_start_matches(":").trim().to_string();
        return Some((code, msg, false));
    }
    if let Some(rest) = line.strip_prefix("warning[")
        && let Some(close) = rest.find(']')
    {
        let code = rest[..close].to_string();
        let msg = rest[close + 1..].trim_start_matches(":").trim().to_string();
        return Some((code, msg, true));
    }
    None
}

/// Parse rustc's proposed code insertion: `12 + use std::foo::Bar;`.
/// Returns the inserted code verbatim.
fn parse_suggestion_insert(line: &str) -> Option<String> {
    let trimmed = line.trim_start_matches('|').trim();
    let mut parts = trimmed.splitn(2, '+');
    let num = parts.next()?.trim();
    if num.is_empty() || !num.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let code = parts.next()?.trim();
    if code.is_empty() {
        return None;
    }
    Some(code.to_string())
}

/// Parse `--> src/main.rs:10:15`
fn parse_location(line: &str) -> Option<(String, u32, u32)> {
    let trimmed = line.trim();
    let prefix = "--> ";
    if let Some(rest) = trimmed.strip_prefix(prefix) {
        // Format: file:line:col
        let parts: Vec<&str> = rest.split(':').collect();
        if parts.len() >= 3
            && let (Ok(line_num), Ok(col_num)) = (
                parts[parts.len() - 2].parse::<u32>(),
                parts[parts.len() - 1].parse::<u32>(),
            )
        {
            let file = parts[..parts.len() - 2].join(":");
            return Some((file, line_num, col_num));
        }
    }
    None
}

/// Classify a compiler error into a deterministic category.
/// This mirrors grounded's semantic category classification in
/// the CCG parser's `classify()` method.
fn classify_error(code: &str, message: &str) -> ErrorKind {
    match code {
        c if c.starts_with("E0308") || message.contains("mismatched types") => {
            ErrorKind::TypeMismatch
        }
        c if c.starts_with("E0425") || c.starts_with("E0432") || c.starts_with("E0433") => {
            ErrorKind::UnresolvedSymbol
        }
        c if c.starts_with("E0589") || c.starts_with("E0061") => ErrorKind::WrongArity,
        c if c.starts_with("E0507") || c.starts_with("E0597") => ErrorKind::BorrowError,
        c if c.starts_with("E0599") || c.starts_with("E0609") => ErrorKind::MissingFieldMethod,
        c if c.starts_with("E0428") => ErrorKind::DuplicateDefinition,
        c if c.starts_with("E0271") || c.starts_with("E0277") => ErrorKind::TraitBound,
        c if c.starts_with("E0107") || c.starts_with("E0282") || c.starts_with("E0283") => {
            ErrorKind::GenericError
        }
        _ => {
            if message.contains("cannot find") {
                ErrorKind::UnresolvedSymbol
            } else if message.contains("expected") && message.contains("found") {
                ErrorKind::TypeMismatch
            } else if message.contains("import") {
                ErrorKind::MissingImport
            } else if message.contains("borrow") {
                ErrorKind::BorrowError
            } else {
                ErrorKind::Other
            }
        }
    }
}
