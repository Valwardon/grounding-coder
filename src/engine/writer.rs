use std::fs;
use std::path::{Path, PathBuf};

use super::arena::{CodeArena, SymbolId, SymbolKind};
use super::corrector::Fix;
use super::symbols::SymbolTable;

/// A code file that can be read, modified, and written back.
#[derive(Debug, Clone)]
pub struct CodeFile {
    pub path: PathBuf,
    pub content: String,
    pub lines: Vec<String>,
}

/// The code writer — grounded's "motor system" repurposed.
///
/// In grounded, the motor system closes the perception-action loop:
/// `MotorCommand nodes fire → generate RenderCommand → renderer executes
/// → sends back actual_hash → prediction error → reward/novelty spike`.
///
/// Here, the writer closes the code-generation loop:
/// `Task → resolve symbols (GapDetector) → compose code from
/// verified patterns → verifier confirms`.
///
/// The writer NEVER generates code from scratch. It composes from
/// patterns found in the CodebaseIndex and APIReference — grounded's
/// KnowledgeStore principle. This is the determinism guarantee:
/// the bot only produces code that references symbols it has verified exist.
pub struct CodeWriter {
    project_dir: PathBuf,
}

impl CodeWriter {
    pub fn new(project_dir: PathBuf) -> Self {
        CodeWriter { project_dir }
    }

    /// Scan the project for source files and extract code symbols.
    /// This builds the CodebaseIndex (grounded's runtime cache pattern).
    pub fn scan_project(&self, project_dir: &Path) -> Vec<super::arena::CodeSymbol> {
        let mut symbols = Vec::new();

        for entry in walkdir::WalkDir::new(project_dir)
            .into_iter()
            .filter_entry(|e| !Self::should_ignore(e))
            .flatten()
        {
            let path = entry.path();
            if path.extension().map_or(false, |ext| {
                matches!(ext.to_str(), Some("rs") | Some("kt") | Some("java"))
            }) {
                if let Ok(content) = fs::read_to_string(path) {
                    self.extract_symbols(path, &content, &mut symbols);
                }
            }
        }

        symbols
    }

    fn should_ignore(entry: &walkdir::DirEntry) -> bool {
        let name = entry.file_name().to_string_lossy().to_string();
        [".git", "target", "build", ".gradle", ".idea", "node_modules"]
            .contains(&name.as_str())
    }

    /// Extract code symbols from a source file.
    fn extract_symbols(&self, path: &Path, content: &str, symbols: &mut Vec<super::arena::CodeSymbol>) {
        let rel_path = path.strip_prefix(&self.project_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        for (i, line) in content.lines().enumerate() {
            let line_num = i as u32 + 1;
            let trimmed = line.trim();

            if let Some(name) = extract_function_name(trimmed) {
                symbols.push(super::arena::CodeSymbol {
                    id: SymbolId::fresh(),
                    qname: format!("{}::{}", rel_path, name),
                    kind: SymbolKind::Function,
                    file_path: rel_path.clone(),
                    line: line_num,
                    col: 0,
                    signature: trimmed.to_string(),
                    source: Some(trimmed.to_string()),
                    edges: Vec::new(),
                });
            }

            let kw = trimmed.split_whitespace().next().unwrap_or("");
            if let Some(name) = extract_type_name(trimmed, &["struct", "class", "enum", "interface", "trait"]) {
                let kind = match kw {
                    "struct" => SymbolKind::Struct,
                    "class" | "interface" => SymbolKind::Struct,
                    "enum" => SymbolKind::Enum,
                    "trait" => SymbolKind::Trait,
                    _ => SymbolKind::Struct,
                };
                symbols.push(super::arena::CodeSymbol {
                    id: SymbolId::fresh(),
                    qname: format!("{}::{}", rel_path, name),
                    kind,
                    file_path: rel_path.clone(),
                    line: line_num,
                    col: 0,
                    signature: trimmed.to_string(),
                    source: Some(trimmed.to_string()),
                    edges: Vec::new(),
                });
            }

            if let Some(name) = extract_import(trimmed) {
                symbols.push(super::arena::CodeSymbol {
                    id: SymbolId::fresh(),
                    qname: name.clone(),
                    kind: SymbolKind::Import,
                    file_path: rel_path.clone(),
                    line: line_num,
                    col: 0,
                    signature: format!("use {}", name),
                    source: Some(trimmed.to_string()),
                    edges: Vec::new(),
                });
            }
        }
    }

    /// Write code for a task — composed from verified patterns.
    ///
    /// This is grounded's "render AST compilation" repurposed.
    /// The writer only emits code built from:
    ///   1. API reference examples (verified against Android SDK)
    ///   2. Codebase patterns (verified by compilation)
    /// It NEVER hallucinates a function signature or import path.
    pub fn write(&self, task: &super::tasks::SubTask, table: &SymbolTable) -> String {
        if task.target_symbols.is_empty() {
            return "// ERROR: no target symbols in task\n".to_string();
        }
        let target = &task.target_symbols[0];
        let def = table.fetch(target);

        if def.is_none() {
            // grounded's honesty principle: "when it doesn't know
            // something, it knows it doesn't know — that's a structural error"
            return format!(
                "// ERROR: unknown symbol '{}'. Cannot generate code.\n\
                 // The bot knows what it doesn't know and refuses to guess.\n",
                target
            );
        }

        let def = def.unwrap();

        let mut code = String::new();

        // Compose from verified examples
        if !def.examples.is_empty() {
            code.push_str(&def.examples[0]);
            code.push('\n');
        } else {
            code.push_str(&format!("// {} {}\n", def.kind, def.qname));
        }

        code
    }

    /// Apply a fix to the codebase.
    pub fn apply_fix(&self, fix: &Fix) -> Vec<String> {
        let mut changed = Vec::new();

        match fix {
            Fix::AddImport(imp) => {
                if let Some(file) = self.find_source_file() {
                    if self.add_import(&file, imp) {
                        changed.push(file.to_string_lossy().to_string());
                    }
                }
            }
            Fix::Replace { find, replace, file, line: _ } => {
                if let Ok(content) = fs::read_to_string(file) {
                    let new_content = content.replacen(find, replace, 1);
                    if new_content != content {
                        if fs::write(file, new_content).is_ok() {
                            changed.push(file.clone());
                        }
                    }
                }
            }
            Fix::ApplySuggestion { file, suggestion } => {
                self.apply_suggestion(file, suggestion, &mut changed);
            }
            Fix::InsertAfter { file, line, code } => {
                if let Ok(content) = fs::read_to_string(file) {
                    let mut lines: Vec<&str> = content.lines().collect();
                    if *line as usize <= lines.len() {
                        lines.insert(*line as usize, code);
                        if fs::write(file, lines.join("\n")).is_ok() {
                            changed.push(file.clone());
                        }
                    }
                }
            }
            Fix::None => {}
        }

        changed
    }

    fn apply_suggestion(&self, file: &str, suggestion: &str, changed: &mut Vec<String>) {
        if let Ok(content) = fs::read_to_string(file) {
            // rustc suggestions look like: help: try `String::from(x)` or `x.to_string()`
            let code = suggestion.trim();
            let new_content = content.replacen(code, code, 1);
            if new_content != content {
                if fs::write(file, new_content).is_ok() {
                    changed.push(file.to_string());
                }
            }
        }
    }

    /// Apply generated code to the target file.
    pub fn apply(
        &self,
        task: &super::tasks::SubTask,
        draft: &str,
        project_dir: &Path,
    ) -> Vec<String> {
        let mut changed = Vec::new();

        let target_file = match &task.target_symbols.first() {
            Some(t) if t.contains("main.rs") => project_dir.join("src/main.rs"),
            Some(t) if t.contains("lib.rs") => project_dir.join("src/lib.rs"),
            Some(t) => project_dir.join(format!("src/{}.rs", t.to_lowercase())),
            None => project_dir.join("src/generated.rs"),
        };

        if let Some(parent) = target_file.parent() {
            let _ = fs::create_dir_all(parent);
        }

        if fs::write(&target_file, draft).is_ok() {
            changed.push(target_file.to_string_lossy().to_string());
        }

        changed
    }

    fn find_source_file(&self) -> Option<PathBuf> {
        let src = self.project_dir.join("src");
        if src.exists() {
            if let Ok(entries) = fs::read_dir(&src) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().map_or(false, |e| e == "rs") {
                        return Some(path);
                    }
                }
            }
        }
        None
    }

    fn add_import(&self, file: &Path, import: &str) -> bool {
        if let Ok(content) = fs::read_to_string(file) {
            if content.contains(import) {
                return false;
            }

            let new_content = if content.contains("use ") {
                content.replacen("use ", &format!("use {}\nuse ", import), 1)
            } else {
                format!("use {};\n{}", import, content)
            };

            fs::write(file, new_content).is_ok()
        } else {
            false
        }
    }
}

/// Extract function name from `fn foo(...)` or `fun foo(...)`
fn extract_function_name(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if trimmed.starts_with("fn ") || trimmed.starts_with("fun ") {
        let after = trimmed.split_whitespace().nth(1)?;
        let name = after.split('(').next()?;
        if !name.is_empty() {
            return Some(name.to_string());
        }
    }
    None
}

/// Extract type name from `struct Foo`, `class Foo`, `enum Foo`, etc.
fn extract_type_name(line: &str, keywords: &[&str]) -> Option<String> {
    let trimmed = line.trim_start();
    for kw in keywords {
        if trimmed.starts_with(&format!("{} ", kw)) || trimmed.starts_with(&format!("{}\n", kw)) {
            let after = trimmed[kw.len()..].trim_start();
            let name = after.split_whitespace().next()?.split('<').next()?;
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    None
}

/// Extract import from `use Foo::Bar;` or `import foo.bar.Baz`
fn extract_import(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.starts_with("use ") {
        trimmed
            .strip_prefix("use ")
            .and_then(|s| s.strip_suffix(';'))
            .map(|s| s.trim().to_string())
    } else if trimmed.starts_with("import ") {
        trimmed
            .strip_prefix("import ")
            .and_then(|s| s.strip_suffix(';'))
            .map(|s| s.trim().to_string())
    } else {
        None
    }
}
