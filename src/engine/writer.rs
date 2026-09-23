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
/// `existed == false` means the file is created by the plan — rollback
/// deletes it, so failed transactions leave no orphan files behind.
#[derive(Debug, Clone)]
pub struct FileSnapshot {
    pub file: PathBuf,
    pub content: String,
    pub existed: bool,
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
    extra: Vec<super::lang::LanguageSpec>,
}

impl CodeWriter {
    pub fn new(project_dir: PathBuf) -> Self {
        let extra = super::lang::load_extra(&project_dir);
        CodeWriter { project_dir, extra }
    }

    /// Scan the project for source files and extract code symbols.
    /// This builds the CodebaseIndex (grounded's runtime cache pattern).
    /// Extensions come from the language registry — any backend language
    /// is indexed, not just Rust-family ones.
    /// Hard caps (depth/files/symbols): on Android an unbounded walk over
    /// a huge tree means OOM death with no message. Big trees get a
    /// partial index and keep running.
    pub fn scan_project(&self, project_dir: &Path) -> Vec<super::arena::CodeSymbol> {
        let mut symbols = Vec::new();

        for (files_seen, entry) in walkdir::WalkDir::new(project_dir)
            .max_depth(8)
            .into_iter()
            .filter_entry(|e| !Self::should_ignore(e))
            .flatten()
            .enumerate()
        {
            if files_seen >= 2000 || symbols.len() >= 20000 {
                log::warn!("scan cap hit in {}", project_dir.display());
                break;
            }
            let path = entry.path();
            let known = path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|ext| {
                    super::lang::KNOWN_EXTENSIONS.contains(&ext)
                        || self
                            .extra
                            .iter()
                            .flat_map(|s| s.extensions.iter())
                            .any(|e| e == ext)
                });
            if known && let Ok(content) = fs::read_to_string(path) {
                self.extract_symbols(path, &content, &mut symbols);
            }
        }

        symbols
    }

    fn should_ignore(entry: &walkdir::DirEntry) -> bool {
        let name = entry.file_name().to_string_lossy().to_string();
        // Hidden dirs (.cache, .config, .dbus…) explode scans on-device
        // and never hold project sources.
        if name.starts_with('.') {
            return true;
        }
        ["target", "build", ".gradle", ".idea", "node_modules"].contains(&name.as_str())
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

        // Read the current file content. Creation-tolerant kinds start
        // from empty when the file doesn't exist yet (snapshot/rollback
        // cover the creation); all other kinds fail honestly.
        let content = match fs::read_to_string(&target_file) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => match task.kind {
                super::tasks::TaskKind::AddImport | super::tasks::TaskKind::CreateFile => {
                    String::new()
                }
                _ => {
                    return Err(format!(
                        "Failed to read target file {}: {}",
                        target_file.display(),
                        e
                    ));
                }
            },
            Err(e) => {
                return Err(format!(
                    "Failed to read target file {}: {}",
                    target_file.display(),
                    e
                ));
            }
        };

        // Generate edits based on task kind and evidence.
        // Planner-only contract: NO LLM-provided code is ever used.
        // If payload contains `code`, reject it to enforce the boundary.
        if task.payload.get("code").is_some() {
            return Err("LLM code in payload rejected: planner-only boundary".to_string());
        }
        match task.kind {
            super::tasks::TaskKind::SynthesizeFunction => {
                return Err(
                    "SynthesizeFunction must go through plan_synthesis() candidate loop"
                        .to_string(),
                );
            }
            super::tasks::TaskKind::VerifyOnly => {
                // No edits by design: the value is the verify+repair loop
                // the caller runs after apply.
                return Ok(EditPlan {
                    task_id: task.id,
                    edits: Vec::new(),
                    evidence: Vec::new(),
                });
            }
            super::tasks::TaskKind::AddImport => {
                let (edit, ev) =
                    self.plan_add_import(&target_file, &content, task, symbol_table)?;
                edits.push(edit);
                evidence.push(ev);
            }
            super::tasks::TaskKind::WriteFunction => {
                let (new_edits, new_evidence) =
                    self.plan_write_function(&target_file, &content, task, symbol_table)?;
                edits.extend(new_edits);
                evidence.extend(new_evidence);
            }
            super::tasks::TaskKind::AddDefinition => {
                let (new_edits, new_evidence) =
                    self.plan_add_definition(&target_file, &content, task)?;
                edits.extend(new_edits);
                evidence.extend(new_evidence);
            }
            super::tasks::TaskKind::AddField => {
                let (edit, ev) = self.plan_add_field(&target_file, &content, task, symbol_table)?;
                edits.push(edit);
                evidence.push(ev);
            }
            super::tasks::TaskKind::WireHandler => {
                let (edit, ev) =
                    self.plan_wire_handler(&target_file, &content, task, symbol_table)?;
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
            super::tasks::TaskKind::ResolveSymbol
            | super::tasks::TaskKind::ResearchRequirements
            | super::tasks::TaskKind::CreateProjectManifest => {
                return Err(format!(
                    "Task kind {:?} requires research/resolution first — BLOCKED",
                    task.kind
                ));
            }
        }

        Ok(EditPlan {
            task_id: task.id,
            edits,
            evidence,
        })
    }

    /// Build one EditPlan per synthesis candidate (body + contract test).
    /// The caller tries each in snapshot/apply/verify/rollback order —
    /// first fully-clean candidate wins, exhaustion means BLOCKED.
    pub fn plan_synthesis(&self, task: &SubTask) -> Result<Vec<EditPlan>, String> {
        if task.payload.get("code").is_some() {
            return Err("LLM code in payload rejected: planner-only boundary".to_string());
        }
        let target_file = self.determine_target_file(task)?;
        // Missing target = new module: synthesize from empty base.
        let content = match fs::read_to_string(&target_file) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => {
                return Err(format!(
                    "Failed to read target file {}: {}",
                    target_file.display(),
                    e
                ));
            }
        };

        let name = task
            .payload
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or("Synthesize task missing name")?
            .to_string();
        let sig_raw = task
            .payload
            .get("signature")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let kind = task
            .payload
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("function");
        // Structs carry fields + method specs as metadata, pages carry
        // content slots; functions parse the signature string. Either way:
        // metadata in, never code.
        let struct_def = if kind == "struct" {
            Some(parse_struct_def(task)?)
        } else {
            None
        };
        let page_def = if kind == "page" {
            if !content.trim().is_empty() {
                return Err(
                    "Page target already has content — no merge semantics, BLOCKED".to_string(),
                );
            }
            Some(parse_page_def(task)?)
        } else {
            None
        };
        let (params, ret) = if struct_def.is_some() || page_def.is_some() {
            (Vec::new(), String::new())
        } else {
            parse_fn_signature(&name, sig_raw)?
        };
        let cases: Vec<super::synthesize::ContractCase> = task
            .payload
            .get("cases")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|c| {
                        Some(super::synthesize::ContractCase {
                            input: c.get("input")?.as_str()?.to_string(),
                            expected: c.get("expected")?.as_str()?.to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let synth = super::synthesize::Synthesizer::new();
        let req = super::synthesize::SynthRequest {
            fn_name: name.clone(),
            params,
            ret,
            cases,
            struct_def,
            page_def,
        };
        let candidates = synth.synthesize(&req)?;
        let contract_test = synth.contract_test(&req);
        let item_exists = if req.struct_def.is_some() {
            content.contains(&format!("struct {}", req.fn_name))
        } else {
            content.contains(&format!("fn {}", req.fn_name))
        };

        // New or undeclared top-level module → wire `mod` into the parent
        // crate root inside the SAME plan (one atomic transaction).
        let wire_edit = self.mod_wire_edit(&target_file)?;
        let wire_evidence = wire_edit.as_ref().map(|e| Evidence::ExistingPattern {
            source_file: e.file.to_string_lossy().to_string(),
            source_line: 0,
        });

        Ok(candidates
            .into_iter()
            .map(|c| {
                // Single insert at EOF (body + contract) so offsets stay valid
                // within one apply pass.
                let replacement = if item_exists {
                    contract_test.clone()
                } else {
                    format!("\n{}\n{}", c.body, contract_test)
                };
                let mut evidence = c.evidence;
                evidence.push(Evidence::ExistingPattern {
                    source_file: target_file.to_string_lossy().to_string(),
                    source_line: content.lines().count() as u32,
                });
                let mut edits = vec![SourceEdit {
                    file: target_file.clone(),
                    start: content.len(),
                    end: content.len(),
                    expected_old: String::new(),
                    replacement,
                }];
                if let Some(w) = &wire_edit {
                    edits.push(w.clone());
                }
                if let Some(ev) = &wire_evidence {
                    evidence.push(ev.clone());
                }
                EditPlan {
                    task_id: task.id,
                    edits,
                    evidence,
                }
            })
            .collect())
    }

    /// If `target` is a top-level `src/<m>.rs` module not declared by the
    /// crate root, build the `mod <m>;` insert edit. `None` when no wiring
    /// is needed. Errors when no crate root exists (module unreachable).
    fn mod_wire_edit(&self, target: &Path) -> Result<Option<SourceEdit>, String> {
        let rel = target.strip_prefix(&self.project_dir).unwrap_or(target);
        let mut comps = rel.components();
        let (Some(first), Some(second), None) = (comps.next(), comps.next(), comps.next()) else {
            return Ok(None);
        };
        use std::path::Component;
        let (Component::Normal(dir), Component::Normal(file)) = (first, second) else {
            return Ok(None);
        };
        if dir != "src" {
            return Ok(None);
        }
        let file = file.to_string_lossy().to_string();
        let module = file.strip_suffix(".rs").ok_or_else(|| {
            format!(
                "Synthesis target {} is not a Rust file — BLOCKED",
                rel.display()
            )
        })?;
        if !is_ident(module) || matches!(module, "lib" | "main" | "mod") {
            return Ok(None);
        }
        let root = ["src/lib.rs", "src/main.rs"]
            .iter()
            .map(|r| self.project_dir.join(r))
            .find(|p| p.exists())
            .ok_or_else(|| format!("No crate root for module {} — BLOCKED", rel.display()))?;
        let root_content = fs::read_to_string(&root)
            .map_err(|e| format!("Failed to read crate root {}: {}", root.display(), e))?;
        let want = format!("mod {};", module);
        for line in root_content.lines() {
            if line.trim() == want || line.trim().starts_with(&format!("pub {}", want)) {
                return Ok(None);
            }
        }
        // Insert after the last `mod` line, else after leading
        // attributes/comments/blanks so `#![...]` stays first.
        let lines: Vec<&str> = root_content.lines().collect();
        let mut insert_line = 0;
        let mut seen_mod = false;
        for (i, line) in lines.iter().enumerate() {
            let t = line.trim();
            if t.starts_with("mod ") || t.starts_with("pub mod ") {
                insert_line = i + 1;
                seen_mod = true;
            } else if !seen_mod
                && (t.is_empty()
                    || t.starts_with("//")
                    || t.starts_with("#!")
                    || t.starts_with("#["))
            {
                insert_line = i + 1;
            }
        }
        // Byte offset of the start of `insert_line`.
        let off: usize = lines[..insert_line.min(lines.len())]
            .iter()
            .map(|l| l.len() + 1)
            .sum::<usize>()
            .min(root_content.len());
        // If appending at EOF without a trailing newline, lead with one.
        let needs_lead =
            !root_content.is_empty() && !root_content.ends_with('\n') && off == root_content.len();
        let replacement = if needs_lead {
            format!("\n{}\n", want)
        } else {
            format!("{}\n", want)
        };
        Ok(Some(SourceEdit {
            file: root,
            start: off,
            end: off,
            expected_old: String::new(),
            replacement,
        }))
    }

    /// Snapshot an explicit file list (deduped), covering planned files
    /// plus every project source so corrections can't escape the transaction.
    pub fn snapshot_many(&self, files: &[PathBuf]) -> Result<Vec<FileSnapshot>, String> {
        let mut seen = std::collections::HashSet::new();
        let mut snapshots = Vec::new();
        for file in files {
            if !seen.insert(file.clone()) {
                continue;
            }
            match fs::read_to_string(file) {
                Ok(content) => snapshots.push(FileSnapshot {
                    file: file.clone(),
                    content,
                    existed: true,
                }),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    snapshots.push(FileSnapshot {
                        file: file.clone(),
                        content: String::new(),
                        existed: false,
                    })
                }
                Err(e) => {
                    return Err(format!("Failed to snapshot {}: {}", file.display(), e));
                }
            }
        }
        Ok(snapshots)
    }

    /// Every source under the project — transaction coverage for files
    /// the correction pipeline may touch outside the plan. Capped like
    /// every other walk: snapshot-all must never OOM the transaction.
    pub fn project_source_files(&self) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let src = self.project_dir.join("src");
        for entry in walkdir::WalkDir::new(&src)
            .max_depth(8)
            .into_iter()
            .filter_entry(|e| !Self::should_ignore(e))
            .flatten()
        {
            if out.len() >= 2000 {
                break;
            }
            let path = entry.path().to_path_buf();
            if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
        out
    }

    /// Create a snapshot of files before applying edits.
    /// Missing files snapshot as `existed == false` — the plan creates them
    /// and rollback deletes them.
    pub fn snapshot(&self, plan: &EditPlan) -> Result<Vec<FileSnapshot>, String> {
        let mut snapshots = Vec::new();
        for edit in &plan.edits {
            match fs::read_to_string(&edit.file) {
                Ok(content) => snapshots.push(FileSnapshot {
                    file: edit.file.clone(),
                    content,
                    existed: true,
                }),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    snapshots.push(FileSnapshot {
                        file: edit.file.clone(),
                        content: String::new(),
                        existed: false,
                    })
                }
                Err(e) => {
                    return Err(format!("Failed to snapshot {}: {}", edit.file.display(), e));
                }
            }
        }
        Ok(snapshots)
    }

    /// Apply an EditPlan with precise byte-range edits.
    ///
    /// Hardened contract:
    ///
    /// - byte range must be valid AND on char boundaries
    /// - current bytes at [start,end) must equal `expected_old` (STALE check)
    /// - no-op inserts (empty replacement at same offset) are skipped
    ///
    /// This replaces the old `apply()` which could replace entire files.
    pub fn apply_plan(&self, plan: &EditPlan) -> Result<Vec<String>, String> {
        // Observability: log the full plan before touching disk.
        log::info!("TASK {}\n{}", plan.task_id, plan);
        let mut changed = Vec::new();

        for edit in &plan.edits {
            // Read current content (missing file = empty base for creation).
            let mut content = match fs::read_to_string(&edit.file) {
                Ok(c) => c,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
                Err(e) => {
                    return Err(format!("Failed to read {}: {}", edit.file.display(), e));
                }
            };

            // Validate byte range
            if edit.start > content.len() || edit.end > content.len() || edit.start > edit.end {
                return Err(format!(
                    "Invalid byte range [{}, {}) for file {} (length {})",
                    edit.start,
                    edit.end,
                    edit.file.display(),
                    content.len()
                ));
            }
            if !content.is_char_boundary(edit.start) || !content.is_char_boundary(edit.end) {
                return Err(format!(
                    "Non-char-boundary range [{}, {}) for {}",
                    edit.start,
                    edit.end,
                    edit.file.display()
                ));
            }

            let current_old = &content[edit.start..edit.end];
            if current_old != edit.expected_old {
                return Err(format!(
                    "STALE edit for {} [{}, {}): expected {:?}, found {:?} — BLOCKED",
                    edit.file.display(),
                    edit.start,
                    edit.end,
                    truncate_str(&edit.expected_old, 120),
                    truncate_str(current_old, 120),
                ));
            }

            if edit.start == edit.end && edit.replacement.is_empty() {
                // No-op — already present.
                continue;
            }

            // Apply the exact byte-range edit
            content.replace_range(edit.start..edit.end, &edit.replacement);

            // Write back (create parent dirs for new files).
            if let Some(parent) = edit.file.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create dir {}: {}", parent.display(), e))?;
            }
            fs::write(&edit.file, content)
                .map_err(|e| format!("Failed to write {}: {}", edit.file.display(), e))?;

            changed.push(edit.file.to_string_lossy().to_string());
        }

        Ok(changed)
    }

    /// Rollback to a snapshot (used when verification fails).
    /// Files created by the plan (`existed == false`) are deleted.
    pub fn rollback(&self, snapshots: &[FileSnapshot]) -> Result<(), String> {
        for snap in snapshots {
            if snap.existed {
                fs::write(&snap.file, &snap.content)
                    .map_err(|e| format!("Failed to rollback {}: {}", snap.file.display(), e))?;
            } else if snap.file.exists() {
                fs::remove_file(&snap.file).map_err(|e| {
                    format!("Failed to remove created {}: {}", snap.file.display(), e)
                })?;
            }
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
    pub fn apply(&self, task: &SubTask, draft: &str, project_dir: &Path) -> Vec<String> {
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
    /// Hardened: all fixes go through verified EditPlan path.
    /// - AddImport: allowed via plan path
    /// - Replace/InsertAfter/ApplySuggestion with arbitrary code: BLOCKED unless
    ///   it is a pure import insertion. No heuristic string replace.
    pub fn apply_fix(&self, fix: &Fix) -> Vec<String> {
        match fix {
            Fix::AddImport(import) => {
                let clean = import
                    .trim()
                    .trim_start_matches("use ")
                    .trim_end_matches(';')
                    .trim()
                    .to_string();
                if let Some(file) = self.find_source_file()
                    && let Ok(content) = fs::read_to_string(&file)
                {
                    // Build a minimal import task and route through plan/apply.
                    let task = SubTask {
                        id: 0,
                        kind: super::tasks::TaskKind::AddImport,
                        description: format!("Fix import {}", clean),
                        target_symbols: vec![
                            file.strip_prefix(&self.project_dir)
                                .unwrap_or(&file)
                                .to_string_lossy()
                                .to_string(),
                        ],
                        required_symbols: vec![],
                        priority: 0.9,
                        source: "correction".to_string(),
                        payload: serde_json::json!({"import": clean}),
                        deadline: 0,
                    };
                    // Use a dummy symbol table that knows std + current content.
                    let st = SymbolTable::new();
                    if let Ok((edit, _)) = self.plan_add_import(&file, &content, &task, &st) {
                        // Already present: desired state holds (idempotent
                        // success — repeat errors must not read as failure).
                        if edit.replacement.is_empty() {
                            return vec![file.to_string_lossy().to_string()];
                        }
                        let plan = EditPlan {
                            task_id: 0,
                            edits: vec![edit],
                            evidence: vec![Evidence::CompilerSuggestion {
                                code: "E0432/E0433".to_string(),
                                file: file.to_string_lossy().to_string(),
                                line: 0,
                                col: 0,
                            }],
                        };
                        if let Ok(ch) = self.apply_plan(&plan) {
                            return ch;
                        }
                    }
                }
                Vec::new()
            }
            Fix::Replace { .. } | Fix::InsertAfter { .. } | Fix::ApplySuggestion { .. } => {
                // Heuristic code replacement is disabled — must come from
                // a compiler span + known recipe via EditPlan.
                // For now, only import-shaped suggestions are allowed.
                if let Fix::ApplySuggestion { file, suggestion } = fix {
                    let s = suggestion.trim();
                    if s.starts_with("use ") || s.starts_with("import ") {
                        let clean = s
                            .trim_start_matches("use ")
                            .trim_start_matches("import ")
                            .trim_end_matches(';')
                            .trim();
                        return self.apply_fix(&Fix::AddImport(clean.to_string()));
                    }
                    let _ = file;
                }
                log::warn!("Blocked heuristic fix: {:?}", fix);
                Vec::new()
            }
            Fix::None => Vec::new(),
        }
    }

    // --- Private planning methods ---

    fn determine_target_file(&self, task: &SubTask) -> Result<PathBuf, String> {
        // Use the target_symbols from the task to find the file.
        // Missing relative paths without `..` are returned as-is: synthesis
        // creates them (snapshot/rollback cover creation), while other
        // planners fail honestly when the file cannot be read.
        if let Some(target) = task.target_symbols.first() {
            let rel = Path::new(target);
            let safe = !rel.is_absolute()
                && !rel
                    .components()
                    .any(|c| matches!(c, std::path::Component::ParentDir));
            if safe {
                return Ok(self.project_dir.join(rel));
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

    fn byte_offset_for_line(content: &str, insert_line: usize) -> usize {
        if insert_line == 0 {
            return 0;
        }
        let lines: Vec<&str> = content.lines().collect();
        if insert_line >= lines.len() {
            return content.len();
        }
        let mut off = lines[..insert_line].join("\n").len();
        if off < content.len() {
            off += 1; // newline
        }
        off
    }

    fn plan_add_import(
        &self,
        file: &Path,
        content: &str,
        task: &SubTask,
        symbol_table: &SymbolTable,
    ) -> Result<(SourceEdit, Evidence), String> {
        use super::lang::LanguageBackend;
        let backend = super::lang::backend_for_file(file, &self.extra);
        // Find what import to add (bare path — strip any language syntax
        // the caller left in, since the backend renders it).
        let raw = task
            .payload
            .get("import")
            .and_then(|v| v.as_str())
            .ok_or("AddImport task missing import in payload")?;
        let import = normalize_import_path(raw);

        // Check if import already exists
        if content.contains(import.as_str()) {
            // No edit needed
            return Ok((
                SourceEdit {
                    file: file.to_path_buf(),
                    start: 0,
                    end: 0,
                    expected_old: String::new(),
                    replacement: String::new(),
                },
                Evidence::VerifiedSymbol {
                    qname: import.to_string(),
                    source: "already_present".to_string(),
                },
            ));
        }

        // Require evidence: known in the symbol table OR std for the
        // file's own language backend. Anything else blocks.
        if !symbol_table.is_known(&import) && !backend.is_known_std(&import) {
            return Err(format!(
                "Import {} not verified ({} backend) — BLOCKED",
                import,
                backend.language()
            ));
        }

        // Find the insertion point - after existing imports, never
        // before the backend's header lines (e.g. Go `package`).
        let lines: Vec<&str> = content.lines().collect();
        let mut insert_line = backend.skip_lines();
        for (i, line) in lines.iter().enumerate() {
            if backend.is_import_line(line) {
                insert_line = (i + 1).max(insert_line);
            } else if insert_line > backend.skip_lines() && line.trim().is_empty() {
                break;
            }
        }

        let byte_offset = Self::byte_offset_for_line(content, insert_line);
        let import_line = format!("{}\n", backend.import_line(&import));

        Ok((
            SourceEdit {
                file: file.to_path_buf(),
                start: byte_offset,
                end: byte_offset,
                expected_old: String::new(),
                replacement: import_line,
            },
            Evidence::VerifiedSymbol {
                qname: import.to_string(),
                source: backend.language().to_string(),
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
        // Deterministic stub only — NEVER LLM code.
        // Payload provides metadata: action name or function name.
        let action_name = task
            .payload
            .get("action")
            .and_then(|v| v.as_str())
            .or_else(|| task.payload.get("name").and_then(|v| v.as_str()))
            .unwrap_or("task");
        let fn_name = sanitize_fn_name(action_name);

        // Avoid duplicate definitions.
        if content.contains(&format!("fn {}", fn_name)) {
            return Ok((
                vec![SourceEdit {
                    file: file.to_path_buf(),
                    start: 0,
                    end: 0,
                    expected_old: String::new(),
                    replacement: String::new(),
                }],
                vec![Evidence::ExistingPattern {
                    source_file: file.to_string_lossy().to_string(),
                    source_line: 0,
                }],
            ));
        }

        let lines: Vec<&str> = content.lines().collect();
        let last_non_empty = lines
            .iter()
            .rposition(|l| !l.trim().is_empty())
            .map(|i| i + 1)
            .unwrap_or(lines.len());

        let byte_offset = Self::byte_offset_for_line(content, last_non_empty);
        let new_code = format!(
            "\n// grounded: {} — deterministic stub\nfn {}() {{}}\n",
            task.description.replace('\n', " "),
            fn_name
        );

        Ok((
            vec![SourceEdit {
                file: file.to_path_buf(),
                start: byte_offset,
                end: byte_offset,
                expected_old: String::new(),
                replacement: new_code,
            }],
            vec![Evidence::ExistingPattern {
                source_file: file.to_string_lossy().to_string(),
                source_line: last_non_empty as u32,
            }],
        ))
    }

    fn plan_add_definition(
        &self,
        file: &Path,
        content: &str,
        task: &SubTask,
    ) -> Result<(Vec<SourceEdit>, Vec<Evidence>), String> {
        let name = task
            .payload
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or("AddDefinition missing name")?;
        let kind = task
            .payload
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("function");
        let signature = task
            .payload
            .get("signature")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if content.contains(name) {
            return Ok((
                vec![SourceEdit {
                    file: file.to_path_buf(),
                    start: 0,
                    end: 0,
                    expected_old: String::new(),
                    replacement: String::new(),
                }],
                vec![Evidence::ExistingPattern {
                    source_file: file.to_string_lossy().to_string(),
                    source_line: 0,
                }],
            ));
        }

        let lines: Vec<&str> = content.lines().collect();
        let insert_at = lines.len();
        let byte_offset = content.len();
        let stub = match kind {
            "struct" => format!("\n#[derive(Debug, Clone)]\npub struct {} {{\n}}\n", name),
            "enum" => format!("\n#[derive(Debug, Clone)]\npub enum {} {{\n}}\n", name),
            _ => {
                if signature.is_empty() {
                    format!(
                        "\n// grounded definition: {}\nfn {}() {{}}\n",
                        name,
                        sanitize_fn_name(name)
                    )
                } else {
                    format!("\n// grounded definition: {}\n{} {{}}\n", name, signature)
                }
            }
        };

        Ok((
            vec![SourceEdit {
                file: file.to_path_buf(),
                start: byte_offset,
                end: byte_offset,
                expected_old: String::new(),
                replacement: stub,
            }],
            vec![Evidence::ExistingPattern {
                source_file: file.to_string_lossy().to_string(),
                source_line: insert_at as u32,
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
        let field_name = task
            .payload
            .get("field_name")
            .and_then(|v| v.as_str())
            .or_else(|| task.payload.get("name").and_then(|v| v.as_str()))
            .ok_or("AddField task missing field_name")?;
        let field_type = task
            .payload
            .get("field_type")
            .and_then(|v| v.as_str())
            .or_else(|| task.payload.get("signature").and_then(|v| v.as_str()))
            .unwrap_or("String");

        // Find a struct/class to add the field to
        let lines: Vec<&str> = content.lines().collect();
        let mut insert_line = 0;
        for (i, line) in lines.iter().enumerate() {
            if line.contains("struct ") || line.contains("class ") {
                for (j, l2) in lines.iter().enumerate().skip(i) {
                    if l2.contains('{') {
                        insert_line = j + 1;
                        break;
                    }
                }
                break;
            }
        }
        if insert_line == 0 {
            return Err("AddField: no struct/class found — BLOCKED".to_string());
        }

        let byte_offset = Self::byte_offset_for_line(content, insert_line);
        let field_code = format!(
            "    pub {}: {},\n",
            sanitize_fn_name(field_name),
            field_type
        );

        Ok((
            SourceEdit {
                file: file.to_path_buf(),
                start: byte_offset,
                end: byte_offset,
                expected_old: String::new(),
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
        // Deterministic stub only — no LLM handler code.
        let action = task
            .payload
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("handler");
        let fn_name = sanitize_fn_name(action);

        if content.contains(&format!("fn {}", fn_name)) {
            return Ok((
                SourceEdit {
                    file: file.to_path_buf(),
                    start: 0,
                    end: 0,
                    expected_old: String::new(),
                    replacement: String::new(),
                },
                Evidence::ExistingPattern {
                    source_file: file.to_string_lossy().to_string(),
                    source_line: 0,
                },
            ));
        }

        let byte_offset = content.len();
        let new_code = format!("\n// grounded handler: {}\nfn {}() {{}}\n", action, fn_name);

        Ok((
            SourceEdit {
                file: file.to_path_buf(),
                start: byte_offset,
                end: byte_offset,
                expected_old: String::new(),
                replacement: new_code,
            },
            Evidence::ExistingPattern {
                source_file: file.to_string_lossy().to_string(),
                source_line: content.lines().count() as u32,
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
        let test_name = task
            .payload
            .get("test")
            .and_then(|v| v.as_str())
            .or_else(|| task.payload.get("name").and_then(|v| v.as_str()))
            .unwrap_or("grounded_test");
        let fn_name = sanitize_fn_name(test_name);

        if content.contains(&format!("fn {}", fn_name)) {
            return Ok((
                SourceEdit {
                    file: file.to_path_buf(),
                    start: 0,
                    end: 0,
                    expected_old: String::new(),
                    replacement: String::new(),
                },
                Evidence::ExistingPattern {
                    source_file: file.to_string_lossy().to_string(),
                    source_line: 0,
                },
            ));
        }

        let byte_offset = content.len();
        let new_code = format!(
            "\n#[test]\nfn {}() {{\n    // grounded deterministic test stub\n    assert!(true);\n}}\n",
            fn_name
        );

        Ok((
            SourceEdit {
                file: file.to_path_buf(),
                start: byte_offset,
                end: byte_offset,
                expected_old: String::new(),
                replacement: new_code,
            },
            Evidence::ExistingPattern {
                source_file: file.to_string_lossy().to_string(),
                source_line: content.lines().count() as u32,
            },
        ))
    }

    fn plan_create_file(
        &self,
        file: &Path,
        task: &SubTask,
    ) -> Result<(SourceEdit, Evidence), String> {
        // Only allow empty or deterministic header — never LLM file content.
        if task.payload.get("content").is_some() {
            return Err("CreateFile with LLM content rejected — BLOCKED".to_string());
        }
        let header = "// grounded-coder managed file\n";

        Ok((
            SourceEdit {
                file: file.to_path_buf(),
                start: 0,
                end: 0,
                expected_old: String::new(),
                replacement: header.to_string(),
            },
            Evidence::ExistingPattern {
                source_file: file.to_string_lossy().to_string(),
                source_line: 0,
            },
        ))
    }

    /// Generate a draft (legacy, for backward compatibility).
    fn generate_draft(&self, task: &SubTask) -> String {
        // Planner-only: never emit LLM code.
        match task.kind {
            super::tasks::TaskKind::AddImport => {
                format!(
                    "use {};",
                    task.payload
                        .get("import")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                )
            }
            _ => "// grounded stub — use plan() instead".to_string(),
        }
    }

    #[allow(dead_code)]
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

/// Parse page metadata from a task payload into a `PageDef`.
/// All slots are plain text; the template escapes them on render.
fn parse_page_def(task: &SubTask) -> Result<super::synthesize::PageDef, String> {
    let title = task
        .payload
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let sections: Vec<(String, String)> = task
        .payload
        .get("sections")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| {
                    Some((
                        s.get("heading")?.as_str()?.to_string(),
                        s.get("body")?.as_str()?.to_string(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default();
    let footer = task
        .payload
        .get("footer")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    Ok(super::synthesize::PageDef {
        title,
        sections,
        footer,
    })
}

/// Parse struct metadata from a task payload into a `StructDef`.
/// Expected payload shape (all metadata, never code):
/// `fields: [{name, type}]`, `methods: [{name, self ("none"|"ref"|"mut"),
/// params: ["n: T"], ret?, op ("new"|"add_assign"|"get"),
/// field?, amount?}]`.
fn parse_struct_def(task: &SubTask) -> Result<super::synthesize::StructDef, String> {
    let fields: Vec<(String, String)> = task
        .payload
        .get("fields")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|f| {
                    Some((
                        f.get("name")?.as_str()?.to_string(),
                        f.get("type")?.as_str()?.to_string(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default();
    if fields.is_empty() {
        return Err("Struct synthesis needs fields metadata — BLOCKED".to_string());
    }
    let methods: Vec<super::synthesize::MethodSpec> = task
        .payload
        .get("methods")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|m| {
                    Some(super::synthesize::MethodSpec {
                        name: m.get("name")?.as_str()?.to_string(),
                        self_kind: m
                            .get("self")
                            .and_then(|v| v.as_str())
                            .unwrap_or("ref")
                            .to_string(),
                        params: m
                            .get("params")
                            .and_then(|v| v.as_array())
                            .map(|ps| {
                                ps.iter()
                                    .filter_map(|p| {
                                        let s = p.as_str()?;
                                        let mut kv = s.splitn(2, ':');
                                        Some((
                                            kv.next()?.trim().to_string(),
                                            kv.next()?.trim().to_string(),
                                        ))
                                    })
                                    .collect()
                            })
                            .unwrap_or_default(),
                        ret: m.get("ret").and_then(|v| v.as_str()).map(|s| s.to_string()),
                        amount_param: m
                            .get("amount")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        field: m
                            .get("field")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        op: m.get("op")?.as_str()?.to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    if methods.is_empty() {
        return Err("Struct synthesis needs methods metadata — BLOCKED".to_string());
    }
    Ok(super::synthesize::StructDef { fields, methods })
}

/// Parse `fn name(a: T, ...) -> R` (leading `fn name` optional when it
/// matches `expected_name`). Returns ((param, type).., return type).
fn parse_fn_signature(
    expected_name: &str,
    sig: &str,
) -> Result<(Vec<(String, String)>, String), String> {
    let s = sig.trim();
    let s = s.strip_prefix("pub ").unwrap_or(s).trim();
    let s = s.strip_prefix("fn ").unwrap_or(s).trim();
    let open = s.find('(').ok_or("Signature missing (params) — BLOCKED")?;
    let name = s[..open].trim();
    if !name.is_empty() && name != expected_name {
        return Err(format!(
            "Signature name {} != task name {} — BLOCKED",
            name, expected_name
        ));
    }
    let rest = &s[open + 1..];
    let close = rest.find(')').ok_or("Signature missing ) — BLOCKED")?;
    let params_raw = rest[..close].trim();
    let after = rest[close + 1..].trim();
    let ret = after
        .strip_prefix("->")
        .map(|r| r.trim().to_string())
        .filter(|r| !r.is_empty())
        .ok_or("Signature missing -> Return — BLOCKED")?;

    let mut params = Vec::new();
    if !params_raw.is_empty() {
        for p in split_top_level_commas(params_raw) {
            let mut kv = p.splitn(2, ':');
            let n = kv
                .next()
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .ok_or("Bad param (missing name) — BLOCKED")?;
            let t = kv
                .next()
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .ok_or("Bad param (missing type) — BLOCKED")?;
            params.push((n, t));
        }
    }
    Ok((params, ret))
}

fn split_top_level_commas(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '<' | '(' | '[' => {
                depth += 1;
                cur.push(c);
            }
            '>' | ')' | ']' => {
                depth = depth.saturating_sub(1);
                cur.push(c);
            }
            ',' if depth == 0 => {
                out.push(cur.clone());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

/// Strip language import syntax down to the bare path:
/// `use a::b;` → `a::b`, `import os` → `os`, `#include <x.h>` → `x.h`.
fn normalize_import_path(raw: &str) -> String {
    let mut s = raw.trim().to_string();
    for prefix in ["use ", "import ", "from ", "#include"] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest.trim().to_string();
            break;
        }
    }
    // `from X import Y` keeps X.
    if let Some((from, _)) = s.split_once(" import ") {
        s = from.trim().to_string();
    }
    s = s.trim_end_matches(';').trim().to_string();
    s.trim_matches(|c| c == '<' || c == '>' || c == '"' || c == '\'')
        .trim()
        .to_string()
}

fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .enumerate()
            .all(|(i, c)| c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
}

fn sanitize_fn_name(s: &str) -> String {
    let mut out: String = s
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    while out.contains("__") {
        out = out.replace("__", "_");
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        return "grounded_task".to_string();
    }
    if out.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        format!("fn_{}", out)
    } else {
        out
    }
}

fn truncate_str(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        // Floor to a char boundary (see plan.rs): raw byte slicing panics
        // on multibyte text and panics abort the mobile process.
        let mut m = n;
        while !s.is_char_boundary(m) {
            m -= 1;
        }
        format!("{}…", &s[..m])
    }
}

/// Extract function name from `fn foo(...)` or `fun foo(...)`
/// Extract function name from `fn foo`, `fun foo`, `def foo`, `func foo`.
fn extract_function_name(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    for kw in ["fn ", "fun ", "def ", "func "] {
        if trimmed.starts_with(kw) {
            let after = trimmed.split_whitespace().nth(1)?;
            let name = after.split('(').next()?.trim_end_matches(':');
            if !name.is_empty() && name != "_" {
                return Some(name.to_string());
            }
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
