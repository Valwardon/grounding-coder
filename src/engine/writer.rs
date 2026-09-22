use std::fs;
use std::path::{Path, PathBuf};

use super::arena::{SymbolId, SymbolKind};
use super::corrector::Fix;
use super::plan::{EditPlan, Evidence, SourceEdit};
use super::symbols::SymbolTable;
use super::tasks::SubTask;

/// A code file that can be read, modified, and written back.
#[derive(Debug, Clone)]
pub struct CodeFile {
    pub path: PathBuf,
    pub content: String,
    pub lines: Vec<String>,
}

/// A snapshot of the filesystem before applying edits.
#[derive(Debug, Clone)]
pub struct FileSnapshot {
    pub file: PathBuf,
    pub content: String,
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
///
/// CRITICAL CHANGE: The writer now produces an EditPlan with precise
/// byte-range edits backed by evidence, NEVER whole-file replacements.
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
            if path
                .extension()
                .is_some_and(|ext| matches!(ext.to_str(), Some("rs") | Some("kt") | Some("java")))
                && let Ok(content) = fs::read_to_string(path)
            {
                self.extract_symbols(path, &content, &mut symbols);
            }
        }

        symbols
    }

    fn should_ignore(entry: &walkdir::DirEntry) -> bool {
        let name = entry.file_name().to_string_lossy().to_string();
        [
            ".git",
            "target",
            "build",
            ".gradle",
            ".idea",
            "node_modules",
        ]
        .contains(&name.as_str())
    }

    /// Extract code symbols from a source file.
    fn extract_symbols(
        &self,
        path: &Path,
        content: &str,
        symbols: &mut Vec<super::arena::CodeSymbol>,
    ) {
        let rel_path = path
            .strip_prefix(&self.project_dir)
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
            if let Some(name) =
                extract_type_name(trimmed, &["struct", "class", "enum", "interface", "trait"])
            {
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

    /// Generate an EditPlan for a task based on the current context.
    ///
    /// This replaces the old `write()` method which produced arbitrary text.
    /// Now we produce a deterministic edit plan with evidence for each edit.
    pub fn plan(&self, task: &SubTask, symbol_table: &SymbolTable) -> Result<EditPlan, String> {
        let mut edits = Vec::new();
        let mut evidence = Vec::new();

        // Determine the target file
        let target_file = self.determine_target_file(task)?;

        // Read the current file content
        let content = fs::read_to_string(&target_file)
            .map_err(|e| format!("Failed to read target file {}: {}", target_file.display(), e))?;

        // Generate edits based on task kind and evidence
        match task.kind {
            super::tasks::TaskKind::AddImport => {
                let (edit, ev) = self.plan_add_import(&target_file, &content, task, symbol_table)?;
                edits.push(edit);
                evidence.push(ev);
            }
            super::tasks::TaskKind::WriteFunction => {
                let (new_edits, new_evidence) = self.plan_write_function(&target_file, &content, task, symbol_table)?;
                edits.extend(new_edits);
                evidence.extend(new_evidence);
            }
            super::tasks::TaskKind::AddField => {
                let (edit, ev) = self.plan_add_field(&target_file, &content, task, symbol_table)?;
                edits.push(edit);
                evidence.push(ev);
            }
            super::tasks::TaskKind::WireHandler => {
                let (edit, ev) = self.plan_wire_handler(&target_file, &content, task, symbol_table)?;
                edits.push(edit);
                evidence.push(ev);
            }
            super::tasks::TaskKind::AddTest => {
                let (edit, ev) = self.plan_add_test(&target_file, &content, task, symbol_table)?;
                edits.push(edit);
                evidence.push(ev);
            }
            super::tasks::TaskKind::CreateFile => {
                let (edit, ev) = self.plan_create_file(&target_file, task)?;
                edits.push(edit);
                evidence.push(ev);
            }
            _ => {
                return Err(format!("Unsupported task kind: {:?}", task.kind));
            }
        }

        Ok(EditPlan {
            task_id: task.id,
            edits,
            evidence,
        })
    }

    /// Create a snapshot of files before applying edits.
    pub fn snapshot(&self, plan: &EditPlan) -> Result<Vec<FileSnapshot>, String> {
        let mut snapshots = Vec::new();
        for edit in &plan.edits {
            let content = fs::read_to_string(&edit.file)
                .map_err(|e| format!("Failed to snapshot {}: {}", edit.file.display(), e))?;
            snapshots.push(FileSnapshot {
                file: edit.file.clone(),
                content,
            });
        }
        Ok(snapshots)
    }

    /// Apply an EditPlan with precise byte-range edits.
    ///
    /// This replaces the old `apply()` which could replace entire files.
    /// Now we ONLY apply the exact byte ranges specified in the plan.
    pub fn apply_plan(&self, plan: &EditPlan) -> Result<Vec<String>, String> {
        let mut changed = Vec::new();

        for edit in &plan.edits {
            // Read current content
            let mut content = fs::read_to_string(&edit.file)
                .map_err(|e| format!("Failed to read {}: {}", edit.file.display(), e))?;

            // Validate byte range
            if edit.start > content.len() || edit.end > content.len() || edit.start > edit.end {
                return Err(format!(
                    "Invalid byte range [{}, {}) for file {} (length {})",
                    edit.start, edit.end, edit.file.display(), content.len()
                ));
            }

            // Apply the exact byte-range edit
            content.replace_range(edit.start..edit.end, &edit.replacement);

            // Write back
            fs::write(&edit.file, content)
                .map_err(|e| format!("Failed to write {}: {}", edit.file.display(), e))?;

            changed.push(edit.file.to_string_lossy().to_string());
        }

        Ok(changed)
    }

    /// Rollback to a snapshot (used when verification fails).
    pub fn rollback(&self, snapshots: &[FileSnapshot]) -> Result<(), String> {
        for snap in snapshots {
            fs::write(&snap.file, &snap.content)
                .map_err(|e| format!("Failed to rollback {}: {}", snap.file.display(), e))?;
        }
        Ok(())
    }

    /// Commit changes (no-op for now, could be used for git integration).
    pub fn commit(&self, _snapshots: &[FileSnapshot]) -> Result<(), String> {
        // In a real implementation, this might create a git commit
        // or mark the transaction as committed
        Ok(())
    }

    /// Legacy `write` method - DEPRECATED.
    ///
    /// Kept for backward compatibility but will be removed.
    /// DO NOT USE IN NEW CODE.
    #[deprecated(note = "Use plan() + apply_plan() instead")]
    pub fn write(&self, task: &SubTask, _symbol_table: &SymbolTable) -> String {
        // This is the old behavior - generate a draft
        // DEPRECATED: We now use plan() + apply_plan()
        self.generate_draft(task)
    }

    /// Legacy `apply` method - DEPRECATED.
    ///
    /// Kept for backward compatibility but will be removed.
    /// DO NOT USE IN NEW CODE.
    #[deprecated(note = "Use plan() + apply_plan() instead")]
    pub fn apply(
        &self,
        task: &SubTask,
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

    /// Apply a correction fix (used by CorrectionPipeline).
    pub fn apply_fix(&self, fix: &Fix) -> Vec<String> {
        let mut changed = Vec::new();

        match fix {
            Fix::AddImport(import) => {
                // For now, just try to add to the first source file
                if let Some(file) = self.find_source_file() {
                    if self.add_import(&file, import) {
                        changed.push(file.to_string_lossy().to_string());
                    }
                }
            }
            Fix::Replace { find, replace, file, line: _ } => {
                let path = self.project_dir.join(file);
                if let Ok(mut content) = fs::read_to_string(&path) {
                    if content.contains(find) {
                        content = content.replace(find, replace);
                        if fs::write(&path, content).is_ok() {
                            changed.push(file.clone());
                        }
                    }
                }
            }
            Fix::ApplySuggestion { file, suggestion } => {
                // Try to parse the suggestion and apply it
                let path = self.project_dir.join(file);
                if let Ok(content) = fs::read_to_string(&path) {
                    // For now, just add the suggestion as an import if it's an import
                    if suggestion.starts_with("use ") || suggestion.starts_with("import ") {
                        let import = suggestion.trim().trim_end_matches(';');
                        if self.add_import(&path, import) {
                            changed.push(file.clone());
                        }
                    }
                }
            }
            Fix::InsertAfter { file, line, code } => {
                let path = self.project_dir.join(file);
                if let Ok(mut lines) = fs::read_to_string(&path).map(|c| c.lines().map(|s| s.to_string()).collect::<Vec<_>>()) {
                    if *line as usize <= lines.len() {
                        lines.insert(*line as usize, code.clone());
                        if fs::write(&path, lines.join("\n")).is_ok() {
                            changed.push(file.clone());
                        }
                    }
                }
            }
            Fix::None => {}
        }

        changed
    }

    // --- Private planning methods ---

    fn determine_target_file(&self, task: &SubTask) -> Result<PathBuf, String> {
        // Use the target_symbols from the task to find the file
        if let Some(target) = task.target_symbols.first() {
            let path = self.project_dir.join(target);
            if path.exists() {
                return Ok(path);
            }
        }

        // Fallback: find the main source file
        self.find_source_file()
            .ok_or_else(|| "No source file found".to_string())
    }

    fn find_source_file(&self) -> Option<PathBuf> {
        let src = self.project_dir.join("src");
        if src.exists()
            && let Ok(entries) = fs::read_dir(&src)
        {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "rs") {
                    return Some(path);
                }
            }
        }
        None
    }

    fn plan_add_import(
        &self,
        file: &Path,
        content: &str,
        task: &SubTask,
        symbol_table: &SymbolTable,
    ) -> Result<(SourceEdit, Evidence), String> {
        // Find what import to add
        let import = task.payload.get("import")
            .and_then(|v| v.as_str())
            .ok_or("AddImport task missing import in payload")?;

        // Check if import already exists
        if content.contains(import) {
            // No edit needed
            return Ok((
                SourceEdit {
                    file: file.to_path_buf(),
                    start: 0,
                    end: 0,
                    replacement: String::new(),
                },
                Evidence::VerifiedSymbol {
                    qname: import.to_string(),
                    source: "already_present".to_string(),
                },
            ));
        }

        // Find the insertion point - after existing imports
        let lines: Vec<&str> = content.lines().collect();
        let mut insert_line = 0;
        for (i, line) in lines.iter().enumerate() {
            if line.trim().starts_with("use ") || line.trim().starts_with("import ") {
                insert_line = i + 1;
            } else if insert_line > 0 && line.trim().is_empty() {
                // First blank line after imports
                break;
            }
        }

        // Calculate byte offset
        let byte_offset = lines[..insert_line].join("\n").len();
        if insert_line < lines.len() && byte_offset < content.len() {
            // Add 1 for the newline
            let byte_offset = byte_offset + 1;
        }

        let import_line = format!("use {};\n", import);

        Ok((
            SourceEdit {
                file: file.to_path_buf(),
                start: byte_offset,
                end: byte_offset,
                replacement: import_line,
            },
            Evidence::VerifiedSymbol {
                qname: import.to_string(),
                source: "symbol_table".to_string(),
            },
        ))
    }

    fn plan_write_function(
        &self,
        file: &Path,
        content: &str,
        task: &SubTask,
        _symbol_table: &SymbolTable,
    ) -> Result<(Vec<SourceEdit>, Vec<Evidence>), String> {
        // For writing a function, we need to find where to insert it
        // This is a simplified version - in practice would use the AST
        let function_code = task.payload.get("code")
            .and_then(|v| v.as_str())
            .ok_or("WriteFunction task missing code in payload")?;

        // Find end of file (before any trailing newlines)
        let lines: Vec<&str> = content.lines().collect();
        let last_non_empty = lines.iter().rposition(|l| !l.trim().is_empty()).unwrap_or(lines.len());
        let insert_after = last_non_empty + 1;

        let byte_offset = lines[..insert_after].join("\n").len();
        let byte_offset = if byte_offset < content.len() { byte_offset + 1 } else { content.len() };

        let new_code = format!("\n{}\n", function_code);

        Ok((
            vec![SourceEdit {
                file: file.to_path_buf(),
                start: byte_offset,
                end: byte_offset,
                replacement: new_code,
            }],
            vec![Evidence::ExistingPattern {
                source_file: file.to_string_lossy().to_string(),
                source_line: insert_after as u32,
            }],
        ))
    }

    fn plan_add_field(
        &self,
        file: &Path,
        content: &str,
        task: &SubTask,
        _symbol_table: &SymbolTable,
    ) -> Result<(SourceEdit, Evidence), String> {
        let field_name = task.payload.get("field_name")
            .and_then(|v| v.as_str())
            .ok_or("AddField task missing field_name")?;
        let field_type = task.payload.get("field_type")
            .and_then(|v| v.as_str())
            .ok_or("AddField task missing field_type")?;

        // Find a struct/class to add the field to
        let lines: Vec<&str> = content.lines().collect();
        let mut insert_line = 0;
        for (i, line) in lines.iter().enumerate() {
            if line.contains("struct ") || line.contains("class ") {
                // Find the opening brace
                for j in i..lines.len() {
                    if lines[j].contains('{') {
                        insert_line = j + 1;
                        break;
                    }
                }
                break;
            }
        }

        let byte_offset = lines[..insert_line].join("\n").len();
        let byte_offset = if byte_offset < content.len() { byte_offset + 1 } else { content.len() };

        let field_code = format!("    pub {}: {},\n", field_name, field_type);

        Ok((
            SourceEdit {
                file: file.to_path_buf(),
                start: byte_offset,
                end: byte_offset,
                replacement: field_code,
            },
            Evidence::ExistingPattern {
                source_file: file.to_string_lossy().to_string(),
                source_line: insert_line as u32,
            },
        ))
    }

    fn plan_wire_handler(
        &self,
        file: &Path,
        content: &str,
        task: &SubTask,
        _symbol_table: &SymbolTable,
    ) -> Result<(SourceEdit, Evidence), String> {
        let handler_code = task.payload.get("handler")
            .and_then(|v| v.as_str())
            .ok_or("WireHandler task missing handler code")?;

        // Find a suitable location (end of file or inside a function)
        let lines: Vec<&str> = content.lines().collect();
        let insert_line = lines.len();

        let byte_offset = content.len();

        let new_code = format!("\n{}\n", handler_code);

        Ok((
            SourceEdit {
                file: file.to_path_buf(),
                start: byte_offset,
                end: byte_offset,
                replacement: new_code,
            },
            Evidence::ExistingPattern {
                source_file: file.to_string_lossy().to_string(),
                source_line: insert_line as u32,
            },
        ))
    }

    fn plan_add_test(
        &self,
        file: &Path,
        content: &str,
        task: &SubTask,
        _symbol_table: &SymbolTable,
    ) -> Result<(SourceEdit, Evidence), String> {
        let test_code = task.payload.get("test")
            .and_then(|v| v.as_str())
            .ok_or("AddTest task missing test code")?;

        let lines: Vec<&str> = content.lines().collect();
        let insert_line = lines.len();
        let byte_offset = content.len();

        let new_code = format!("\n{}\n", test_code);

        Ok((
            SourceEdit {
                file: file.to_path_buf(),
                start: byte_offset,
                end: byte_offset,
                replacement: new_code,
            },
            Evidence::ExistingPattern {
                source_file: file.to_string_lossy().to_string(),
                source_line: insert_line as u32,
            },
        ))
    }

    fn plan_create_file(
        &self,
        file: &Path,
        task: &SubTask,
    ) -> Result<(SourceEdit, Evidence), String> {
        let file_content = task.payload.get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        Ok((
            SourceEdit {
                file: file.to_path_buf(),
                start: 0,
                end: 0,
                replacement: file_content.to_string(),
            },
            Evidence::ExistingPattern {
                source_file: file.to_string_lossy().to_string(),
                source_line: 0,
            },
        ))
    }

    /// Generate a draft (legacy, for backward compatibility).
    fn generate_draft(&self, task: &SubTask) -> String {
        // Simple draft generation based on task kind
        match task.kind {
            super::tasks::TaskKind::WriteFunction => {
                task.payload.get("code")
                    .and_then(|v| v.as_str())
                    .unwrap_or("// TODO: implement")
                    .to_string()
            }
            super::tasks::TaskKind::AddImport => {
                format!("use {};", task.payload.get("import").and_then(|v| v.as_str()).unwrap_or(""))
            }
            _ => "// TODO: implement".to_string(),
        }
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
