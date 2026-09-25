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
    /// Hash-verified staged bytes for ReplicateFile tasks, keyed by task
    /// id. Filled by `stage_replica` (async fetch lives in run_task);
    /// planning only ever sees bytes that passed the hash gate.
    staging: std::collections::HashMap<u64, Vec<u8>>,
}

impl CodeWriter {
    pub fn new(project_dir: PathBuf) -> Self {
        let extra = super::lang::load_extra(&project_dir);
        CodeWriter {
            project_dir,
            extra,
            staging: std::collections::HashMap::new(),
        }
    }

    /// Project dir accessor for cross-module planning helpers.
    pub fn project_dir(&self) -> &Path {
        &self.project_dir
    }

    /// R-derive-clone: E0277 `` `X` is not satisfied `` for a `Clone`
    /// bound means a struct lacks the derive the migrated code now needs
    /// (0.5 reference-clones masked it). Find `struct X` (same file first,
    /// then project sources) and add `Clone` to its derive list — or fail
    /// honestly when the shape isn't exactly that.
    pub fn migrate_derive_clone(
        project_dir: &Path,
        error: &super::error::CompileError,
    ) -> Option<SourceEdit> {
        if error.code != "E0277" || !error.message.contains("Clone") {
            return None;
        }
        // Type name between backticks: "`TradeEntry: Clone`".
        let msg = &error.message;
        let start = msg.find('`')? + 1;
        let rest = &msg[start..];
        let end = rest.find([':', '`', ' '])?;
        let ty = rest[..end].trim().to_string();
        if ty.is_empty() || !is_ident(&ty) {
            return None;
        }
        // Same file first, then a bounded project search.
        let here = project_dir.join(&error.file);
        let mut candidates = vec![here];
        let mut stack = vec![project_dir.to_path_buf()];
        let mut seen_dirs = 0;
        while let Some(current) = stack.pop() {
            if seen_dirs > 60 {
                break;
            }
            seen_dirs += 1;
            let Ok(entries) = std::fs::read_dir(&current) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    if name.starts_with('.')
                        || ["target", "build", ".gradle", "node_modules"].contains(&name.as_str())
                    {
                        continue;
                    }
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "rs")
                    && let Ok(content) = std::fs::read_to_string(&path)
                    && content.contains(&format!("struct {}", ty))
                    && !candidates.contains(&path)
                {
                    candidates.push(path);
                }
            }
        }
        for path in candidates {
            if let Some(edit) = add_clone_derive(&path, &ty) {
                return Some(edit);
            }
        }
        None
    }

    /// Fetch a URL and gate it on the expected SHA-256. Supports
    /// `https://` (bundled-roots HTTPS) and `file://` (local template
    /// trees — same hash discipline, offline-testable).
    pub async fn fetch_gated(url: &str, sha256: &str) -> Result<Vec<u8>, String> {
        let bytes: Vec<u8> = if let Some(path) = url.strip_prefix("file://") {
            std::fs::read(path).map_err(|e| format!("file:// read failed: {}", e))?
        } else if url.starts_with("https://") {
            crate::http::get_bytes(url)
                .await
                .map_err(|e| format!("fetch failed: {}", e))?
        } else {
            return Err(format!("Unsupported URL scheme (need https/file): {}", url));
        };
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hex = format!("{:x}", hasher.finalize());
        if !hex.eq_ignore_ascii_case(sha256) {
            return Err(format!(
                "SHA-256 mismatch (want {}, got {}) — refusing bytes",
                sha256, hex
            ));
        }
        Ok(bytes)
    }

    /// Stage verified bytes for a replicate task.
    pub fn stage_replica(&mut self, task_id: u64, bytes: Vec<u8>) {
        self.staging.insert(task_id, bytes);
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
        // File-less kinds first: they must never fail on file resolution
        // (a verify-only run on a bare dir has no source file to find).
        if task.kind == super::tasks::TaskKind::VerifyOnly {
            // No edits by design: the value is the verify+repair loop
            // the caller runs after apply.
            return Ok(EditPlan {
                task_id: task.id,
                edits: Vec::new(),
                evidence: Vec::new(),
            });
        }
        if task.kind == super::tasks::TaskKind::SynthesizeFunction {
            return Err(
                "SynthesizeFunction must go through plan_synthesis() candidate loop".to_string(),
            );
        }
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
                super::tasks::TaskKind::AddImport
                | super::tasks::TaskKind::CreateFile
                | super::tasks::TaskKind::ReplicateFile => String::new(),
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
            super::tasks::TaskKind::SynthesizeFunction
            | super::tasks::TaskKind::VerifyOnly
            | super::tasks::TaskKind::Build => {
                // Handled before file resolution above (synthesis loop /
                // verify-only / build oracle); this arm exists so future
                // refactors fail honestly instead of panicking.
                return Err("Internal routing error — BLOCKED".to_string());
            }
            super::tasks::TaskKind::EnsureDep => {
                let (edit, ev) = self.plan_ensure_dep(task)?;
                edits.push(edit);
                evidence.push(ev);
            }
            super::tasks::TaskKind::ReplicateFile => {
                let (edit, ev) = self.plan_replicate(task)?;
                edits.push(edit);
                evidence.push(ev);
            }
            super::tasks::TaskKind::ReplaceExact => {
                let (edit, ev) = self.plan_replace_exact(&target_file, &content, task)?;
                edits.push(edit);
                evidence.push(ev);
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

        let lang = if target_file
            .extension()
            .is_some_and(|e| e == "kt" || e == "java")
        {
            "kotlin"
        } else {
            "rust"
        }
        .to_string();
        let synth = super::synthesize::Synthesizer::new();
        let req = super::synthesize::SynthRequest {
            fn_name: name.clone(),
            lang,
            params,
            ret,
            cases,
            struct_def,
            page_def,
        };
        let candidates = synth.synthesize(&req)?;
        let contract_test = synth.contract_test(&req);
        let item_exists = if req.struct_def.is_some() {
            if req.lang == "kotlin" {
                content.contains(&format!("class {}", req.fn_name))
            } else {
                content.contains(&format!("struct {}", req.fn_name))
            }
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
        // Rust modules only — other languages have no `mod` wiring.
        if !target.extension().is_some_and(|e| e == "rs") {
            return Ok(None);
        }
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
    /// - multi-edit plans apply bottom-up (descending offsets) so every
    ///   range stays valid against pristine coordinates; all planners
    ///   compute offsets against the same pre-apply content
    ///
    /// This replaces the old `apply()` which could replace entire files.
    pub fn apply_plan(&self, plan: &EditPlan) -> Result<Vec<String>, String> {
        // Observability: log the full plan before touching disk.
        log::info!("TASK {}\n{}", plan.task_id, plan);
        let mut changed = Vec::new();
        let mut ordered: Vec<&SourceEdit> = plan.edits.iter().collect();
        ordered.sort_by_key(|e| std::cmp::Reverse(e.start));

        for edit in ordered {
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

            // No-op skip applies only when the file already exists:
            // creating a (possibly empty) missing file must proceed.
            let existed = edit.file.exists();
            if edit.start == edit.end && edit.replacement.is_empty() && existed {
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
            let target = target.strip_prefix("./").unwrap_or(target);
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

    /// Ensure `name = "version"` sits under `[dependencies]` in Cargo.toml.
    /// Evidence is the crates.io registry entry the engine verified in
    /// Step 0 — the compiler re-verifies by resolving the dep at check.
    fn plan_ensure_dep(&self, task: &SubTask) -> Result<(SourceEdit, Evidence), String> {
        let name = task
            .payload
            .get("crate")
            .and_then(|v| v.as_str())
            .ok_or("EnsureDep task missing crate in payload")?;
        let version = task
            .payload
            .get("version")
            .and_then(|v| v.as_str())
            .ok_or("EnsureDep task missing version in payload")?;
        let manifest = self.project_dir.join("Cargo.toml");
        let content = fs::read_to_string(&manifest)
            .map_err(|e| format!("No Cargo.toml ({}); cannot add dep {}", e, name))?;

        // Already declared (any version constraint counts — bumping
        // versions unprompted would be guessing at the user's intent).
        let declared = content.lines().any(|l| {
            let t = l.trim();
            t.starts_with(&format!("{} ", name))
                || t.starts_with(&format!("{}=", name))
                || t == name
        });
        if declared {
            return Ok((
                SourceEdit {
                    file: manifest,
                    start: 0,
                    end: 0,
                    expected_old: String::new(),
                    replacement: String::new(),
                },
                Evidence::VerifiedSymbol {
                    qname: name.to_string(),
                    source: "already_present".to_string(),
                },
            ));
        }

        let lines: Vec<&str> = content.lines().collect();
        let section = lines
            .iter()
            .position(|l| l.trim() == "[dependencies]")
            .ok_or("Cargo.toml has no [dependencies] — BLOCKED")?;
        let byte_offset = Self::byte_offset_for_line(&content, section + 1);
        Ok((
            SourceEdit {
                file: manifest,
                start: byte_offset,
                end: byte_offset,
                expected_old: String::new(),
                replacement: format!("{} = \"{}\"\n", name, version),
            },
            Evidence::VerifiedSymbol {
                qname: format!("crates.io/{}@{}", name, version),
                source: "crates.io".to_string(),
            },
        ))
    }

    /// Plan a byte-exact replication from staged (hash-verified) bytes.
    /// Missing file → create. Existing identical file → no-op. Existing
    /// different file → full replace ONLY when the staged bytes are the
    /// verified payload (expected_old carries current content, so any
    /// concurrent change trips the stale check at apply).
    fn plan_replicate(&self, task: &SubTask) -> Result<(SourceEdit, Evidence), String> {
        let file = task
            .payload
            .get("file")
            .and_then(|v| v.as_str())
            .ok_or("Replicate task missing file in payload")?;
        let sha = task
            .payload
            .get("sha256")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let staged = self
            .staging
            .get(&task.id)
            .ok_or_else(|| format!("No staged bytes for {} — fetch must precede planning", file))?;
        let target = self.project_dir.join(file);
        let replacement = String::from_utf8(staged.clone())
            .map_err(|_| format!("Staged bytes for {} are not UTF-8 — BLOCKED", file))?;
        let (start, end, expected_old) = match fs::read_to_string(&target) {
            Ok(current) => {
                if current == replacement {
                    // Identical: empty edit, skipped at apply.
                    (0, 0, String::new())
                } else {
                    (0, current.len(), current)
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Missing: create (even when empty — apply distinguishes
                // create from no-op by file existence).
                (0, 0, String::new())
            }
            Err(e) => {
                return Err(format!("Failed to read {}: {}", target.display(), e));
            }
        };
        // Identical existing content collapses to the no-op shape.
        let replacement = if start == 0 && end == 0 && expected_old.is_empty() && target.exists() {
            String::new()
        } else {
            replacement
        };
        Ok((
            SourceEdit {
                file: target,
                start,
                end,
                expected_old,
                replacement,
            },
            Evidence::VerifiedSymbol {
                qname: format!("replicate:{}", file),
                source: format!("sha256:{}", sha),
            },
        ))
    }

    /// Plan one exact replacement. The file's own bytes are the evidence:
    /// exactly one occurrence must exist, else BLOCKED (zero = nothing to
    /// rename; many = ambiguous).
    fn plan_replace_exact(
        &self,
        file: &Path,
        content: &str,
        task: &SubTask,
    ) -> Result<(SourceEdit, Evidence), String> {
        let find = task
            .payload
            .get("find")
            .and_then(|v| v.as_str())
            .ok_or("Replace task missing find in payload")?;
        let replace = task
            .payload
            .get("replace")
            .and_then(|v| v.as_str())
            .ok_or("Replace task missing replace in payload")?;
        if find.is_empty() {
            return Err("Empty find string — BLOCKED".to_string());
        }
        let matches: Vec<_> = content.match_indices(find).collect();
        if matches.is_empty() {
            return Err(format!("`{}` not found — BLOCKED", truncate_str(find, 80)));
        }
        if matches.len() > 1 {
            return Err(format!(
                "`{}` matches {} times — ambiguous, BLOCKED",
                truncate_str(find, 80),
                matches.len()
            ));
        }
        let (start, _) = matches[0];
        let end = start + find.len();
        Ok((
            SourceEdit {
                file: file.to_path_buf(),
                start,
                end,
                expected_old: find.to_string(),
                replacement: replace.to_string(),
            },
            Evidence::ExistingPattern {
                source_file: file.to_string_lossy().to_string(),
                source_line: (content[..start].lines().count().max(1)) as u32,
            },
        ))
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

        // Evidence names its source: backend stdlib vs registry crate.
        let source = if backend.is_known_std(&import) {
            backend.language().to_string()
        } else {
            "crates.io".to_string()
        };
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
                source,
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

/// Try the Dioxus 0.5→0.7 migration transforms for one diagnostic.
/// Each returns anchored byte-range edits built SOLELY from the file's own
/// bytes plus fixed templates — no model content anywhere. The caller
/// applies through the stale-checked path and the compiler judges next.
pub fn plan_migration_fix(
    project_dir: &Path,
    error: &super::error::CompileError,
) -> Option<(Vec<SourceEdit>, Evidence)> {
    if error.file.is_empty() || error.line == 0 {
        return None;
    }
    let path = project_dir.join(&error.file);
    let content = fs::read_to_string(&path).ok()?;
    let ev = Evidence::CompilerSuggestion {
        code: error.code.clone(),
        file: error.file.clone(),
        line: error.line,
        col: error.col,
    };
    if let Some(edit) = migrate_rsx_let(&path, &content, error.line) {
        return Some((vec![edit], ev));
    }
    if let Some(edits) = migrate_resource_read(&path, &content, error) {
        return Some((edits, ev));
    }
    if let Some(edit) = migrate_await_spawn(&path, &content, error.line) {
        return Some((vec![edit], ev));
    }
    if let Some(edit) = CodeWriter::migrate_derive_clone(project_dir, error) {
        return Some((vec![edit], ev));
    }
    if let Some(edit) = migrate_mut_binding(&path, &content, error) {
        return Some((vec![edit], ev));
    }
    if let Some(edit) = migrate_fnmut_param(&path, &content, error) {
        return Some((vec![edit], ev));
    }
    if let Some(edit) = migrate_fnmut_call(&path, &content, error) {
        return Some((vec![edit], ev));
    }
    if let Some(edit) = migrate_into_to_string(&path, &content, error) {
        return Some((vec![edit], ev));
    }
    None
}

/// R-mut: E0596 ``cannot borrow `X` as mutable``. Three structural
/// shapes, each the missing half of a 0.7 Shared borrow chain:
/// (a) "not declared as mutable" where `X` is a hook binding
/// (`use_signal` & co. return a handle whose `set` needs `mut`): the
/// single `let X = …` line becomes `let mut X`.
/// (b) "not declared as mutable" where `X` is an `impl Fn` parameter
/// called from inside a `move` handler (`oninput: move |e| oninput(…)`):
/// the 0.7 handler traits are `FnMut`, so the parameter relaxes to
/// `mut X: impl FnMut`.
/// (c) "captured variable in a `Fn` closure" where `X` is a plain local
/// `let` binding (`expanded` captured by `move || expanded.set(..)`
/// passed to a (possibly already relaxed) callee): the binding itself
/// becomes `let mut X` so the `move` capture carries mutability.
/// Anything else (zero/many lets, non-`Fn` params) refuses.
fn migrate_mut_binding(
    path: &Path,
    content: &str,
    error: &super::error::CompileError,
) -> Option<SourceEdit> {
    if error.code != "E0596" {
        return None;
    }
    // Flavor (c) shares the E0596 code with a different message; the
    // hook-binding shapes below only apply to the declared-mutable
    // flavor. Case (c) is handled after the `let` scan.
    let declared_flavor = error.message.contains("not declared as mutable");
    let capture_flavor = error
        .message
        .contains("captured variable in a `Fn` closure");
    if !declared_flavor && !capture_flavor {
        return None;
    }
    let msg = &error.message;
    let start = msg.find('`')? + 1;
    let name: String = msg[start..].chars().take_while(|c| *c != '`').collect();
    if !is_ident(&name) {
        return None;
    }
    let lines: Vec<&str> = content.lines().collect();
    let starts = line_starts(content);
    let mut found = Vec::new();
    for (k, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        let Some(rest) = t.strip_prefix("let ") else {
            continue;
        };
        if rest == name
            || rest.starts_with(&format!("{} ", name))
            || rest.starts_with(&format!("{}:", name))
            || rest.starts_with(&format!("{}=", name))
        {
            // Skip `let mut X` (already mutable — nothing to do honestly).
            if rest.starts_with(&format!("mut {}", name)) {
                continue;
            }
            found.push(k);
        }
    }
    if std::env::var("GROUNDING_DEBUG_MIGRATE").is_ok() {
        eprintln!(
            "[mut] {}:{} name={} lets={}",
            error.file,
            error.line,
            name,
            found.len()
        );
    }
    if found.len() > 1 {
        return None;
    }
    // Case (c): Fn-closure flavor over a plain local binding — the
    // binding itself needs `mut` (the callee side is handled, or will
    // be, by the FnMut recipes across rounds).
    if capture_flavor && !declared_flavor {
        if found.len() != 1 {
            return None;
        }
        let k = found[0];
        let line = lines[k];
        let pos = line.find(&format!("let {}", name))?;
        let ls = *starts.get(k)?;
        let le = starts.get(k + 1).copied().unwrap_or(content.len());
        let old = content.get(ls..le)?.to_string();
        let mut fixed = line.to_string();
        fixed.replace_range(pos + 3..pos + 3, " mut");
        return Some(SourceEdit {
            file: path.to_path_buf(),
            start: ls,
            end: le,
            expected_old: old,
            replacement: fixed + "\n",
        });
    }
    if found.is_empty() {
        // Case (b): X may be an `impl Fn` parameter of the enclosing fn.
        if error.line == 0 {
            return None;
        }
        let err_idx = (error.line as usize).saturating_sub(1);
        if err_idx >= lines.len() {
            return None;
        }
        let mut fn_idx = None;
        for k in (err_idx.saturating_sub(150)..=err_idx.min(lines.len().saturating_sub(1))).rev() {
            if fn_def_name(lines[k]).is_some() {
                fn_idx = Some(k);
                break;
            }
        }
        // relax_param_in_fn only touches `impl Fn(` params, so plain
        // parameters (bool, &str) safely refuse here.
        let out = relax_param_in_fn(path, content, &lines, fn_idx?, &name);
        if std::env::var("GROUNDING_DEBUG_MIGRATE").is_ok() {
            eprintln!(
                "[mut] {}:{} name={} lets=0 fn_idx={:?} -> {}",
                error.file,
                error.line,
                name,
                fn_idx,
                if out.is_some() { "edit" } else { "none" }
            );
        }
        return out;
    }
    let k = found[0];
    let line = lines[k];
    let pos = line.find(&format!("let {}", name))?;
    let ls = *starts.get(k)?;
    let le = starts.get(k + 1).copied().unwrap_or(content.len());
    let old = content.get(ls..le)?.to_string();
    let mut fixed = line.to_string();
    fixed.replace_range(pos + 3..pos + 3, " mut");
    Some(SourceEdit {
        file: path.to_path_buf(),
        start: ls,
        end: le,
        expected_old: old,
        replacement: fixed + "\n",
    })
}

/// R-fnmut: E0596 ``cannot borrow `V` as mutable, as it is a captured
/// variable in a `Fn` closure``. In 0.5, `use_state` handles allowed
/// `set` through `&self`; in 0.7 `Writable::set` takes `&mut self`, so
/// the handle must be `mut` AND the capturing closure must be `FnMut`.
/// Two structural shapes, tried in order:
///
/// Case A — `V` is a parameter of the enclosing local fn (`ontoggle:
/// impl Fn() + 'static` called inside the component's own `move |_|`
/// handler): relax that parameter to `impl FnMut()`.
/// Case B — `V` is a local binding captured by a `move ||` passed as
/// argument N to a local fn (`sec(cond, "T", move || v.set(..), …)`):
/// relax that callee's Nth parameter from `impl Fn` to `impl FnMut`.
/// Anything else (external callees, non-`impl Fn` params) refuses; the
/// compiler judges every relaxation next round.
fn migrate_fnmut_param(
    path: &Path,
    content: &str,
    error: &super::error::CompileError,
) -> Option<SourceEdit> {
    if error.code != "E0596"
        || !error
            .message
            .contains("captured variable in a `Fn` closure")
    {
        return None;
    }
    let msg = &error.message;
    let start = msg.find('`')? + 1;
    let name: String = msg[start..].chars().take_while(|c| *c != '`').collect();
    if !is_ident(&name) {
        return None;
    }
    let lines: Vec<&str> = content.lines().collect();
    if error.line == 0 {
        return None;
    }
    let err_idx = (error.line as usize).saturating_sub(1);
    if err_idx >= lines.len() {
        return None;
    }
    // Case A: V is a parameter of the enclosing fn.
    let mut fn_idx = None;
    for k in (err_idx.saturating_sub(150)..=err_idx.min(lines.len().saturating_sub(1))).rev() {
        if fn_def_name(lines[k]).is_some() {
            fn_idx = Some(k);
            break;
        }
    }
    let fn_idx = fn_idx?;
    if let Some(edit) = relax_param_in_fn(path, content, &lines, fn_idx, &name) {
        return Some(edit);
    }
    // Case B: V is a local binding; find the enclosing `move` closure
    // (cap 40 lines up — the closure may start mid-line as a call
    // argument), then its call and the argument position.
    let mut close_idx = None;
    for k in (err_idx.saturating_sub(40)..=err_idx.min(lines.len().saturating_sub(1))).rev() {
        if has_move_closure(lines[k]) {
            close_idx = Some(k);
            break;
        }
    }
    let close_idx = close_idx?;
    let (callee, arg_idx) = enclosing_call(&lines, close_idx)?;
    // Callee must be defined exactly once in this file — and must not be
    // the enclosing fn itself (that shape means the call was misread;
    // Case A already covers genuine parameter relaxations).
    if let Some(enclosing_name) = fn_def_name(lines[fn_idx])
        && callee == enclosing_name
    {
        return None;
    }
    let mut defs = Vec::new();
    for (k, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        if t.starts_with(&format!("fn {}(", callee)) || t.starts_with(&format!("fn {}<", callee)) {
            defs.push(k);
        }
    }
    if defs.len() != 1 {
        return None;
    }
    relax_nth_param(path, content, &lines, defs[0], arg_idx)
}

/// Binding name of one parameter: leading `mut ` skipped, so
/// `mut ontoggle: impl FnMut()` yields `ontoggle` (a naive ident scan
/// would return `mut` and miss every already-relaxed signature).
fn param_binding(p: &str) -> String {
    let t = p.trim().strip_prefix("mut ").unwrap_or(p.trim());
    t.chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
}

/// Relax parameter `name` of the fn starting at `fn_idx` from
/// `impl Fn` to `impl FnMut`. Returns None unless the parameter exists
/// with exactly that shape.
fn relax_param_in_fn(
    path: &Path,
    content: &str,
    lines: &[&str],
    fn_idx: usize,
    name: &str,
) -> Option<SourceEdit> {
    let sig = fn_sig(content, lines, fn_idx)?;
    for param in sig.params.iter() {
        if param_binding(param) != name {
            continue;
        }
        let fixed = relax_fn_param(param)?;
        let fixed_params: Vec<String> = sig
            .params
            .iter()
            .map(|p| {
                if param_binding(p) == name {
                    fixed.clone()
                } else {
                    p.to_string()
                }
            })
            .collect();
        return splice_fn_sig(path, content, lines, &sig, &fixed_params);
    }
    None
}

/// Relax the Nth parameter of the fn at `fn_idx` from `impl Fn` to
/// `impl FnMut`. Positional: for closures passed as arguments.
fn relax_nth_param(
    path: &Path,
    content: &str,
    lines: &[&str],
    fn_idx: usize,
    arg_idx: usize,
) -> Option<SourceEdit> {
    let sig = fn_sig(content, lines, fn_idx)?;
    let param = sig.params.get(arg_idx)?;
    let fixed = relax_fn_param(param)?;
    let fixed_params: Vec<String> = sig
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if i == arg_idx {
                fixed.clone()
            } else {
                p.to_string()
            }
        })
        .collect();
    splice_fn_sig(path, content, lines, &sig, &fixed_params)
}

/// Relax one parameter from `impl Fn(…)` to `mut …: impl FnMut(…)`.
/// Both halves are required in 0.7: `Writable::set` and `FnMut` calls
/// borrow mutably, and a `move` closure can only carry a mutable
/// capture from a `mut` binding (the compiler suggests exactly this).
/// Refuses anything already `FnMut`/`mut` or not shaped `impl Fn(`.
fn relax_fn_param(p: &str) -> Option<String> {
    if !p.contains("impl Fn(") || p.contains("FnMut") {
        return None;
    }
    let mut q = p.replacen("impl Fn(", "impl FnMut(", 1);
    if !q.trim_start().starts_with("mut ") {
        let ws = q.len() - q.trim_start().len();
        q.insert_str(ws, "mut ");
    }
    Some(q)
}

/// A fn signature located in the file: line span, top-level params,
/// and absolute byte offsets of the head end (`fn f(`), the matching
/// close paren, and its end. All scanning is string-aware over the
/// ORIGINAL bytes (a `","` or paren inside a literal never counts), so
/// the splice below cannot unbalance the file.
struct FnSig {
    ps: usize,
    pe: usize,
    params: Vec<String>,
    head_end: usize,
    close_end: usize,
}

fn fn_sig(content: &str, lines: &[&str], fn_idx: usize) -> Option<FnSig> {
    let starts = line_starts(content);
    // End line: first line where parens balance (strings stripped).
    let mut depth = 0i32;
    let mut started = false;
    let mut pe = None;
    for (k, line) in lines.iter().enumerate().skip(fn_idx).take(30) {
        let s = strip_line_strings(line);
        for c in s.chars() {
            if c == '(' {
                depth += 1;
                started = true;
            } else if c == ')' {
                depth -= 1;
            }
        }
        if started && depth == 0 {
            pe = Some(k);
            break;
        }
    }
    let pe = pe?;
    let ls = *starts.get(fn_idx)?;
    let le = starts.get(pe + 1).copied().unwrap_or(content.len());
    let span = content.get(ls..le)?;
    // String-aware scan over original bytes: head end, matching close.
    let mut in_str = false;
    let mut esc = false;
    let mut d = 0i32;
    let mut head_end = None;
    let mut close_end = None;
    for (i, c) in span.char_indices() {
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        if c == '"' {
            in_str = true;
        } else if c == '(' {
            if head_end.is_none() {
                head_end = Some(ls + i + c.len_utf8());
            }
            d += 1;
        } else if c == ')' {
            d -= 1;
            if head_end.is_some() && d == 0 {
                close_end = Some(ls + i + c.len_utf8());
                break;
            }
        }
    }
    let (head_end, close_end) = (head_end?, close_end?);
    // Split the inner text on top-level commas (string- and bracket-aware).
    let inner = content.get(head_end..close_end - 1)?;
    let mut params = Vec::new();
    let mut cur = String::new();
    let mut in_s = false;
    let mut es = false;
    let mut dd = 0i32;
    for c in inner.chars() {
        if in_s {
            cur.push(c);
            if es {
                es = false;
            } else if c == '\\' {
                es = true;
            } else if c == '"' {
                in_s = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_s = true;
                cur.push(c);
            }
            '(' | '<' | '[' => {
                dd += 1;
                cur.push(c);
            }
            ')' | '>' | ']' => {
                dd -= 1;
                cur.push(c);
            }
            ',' if dd == 0 => {
                params.push(cur.trim().to_string());
                cur = String::new();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        params.push(cur.trim().to_string());
    }
    Some(FnSig {
        ps: fn_idx,
        pe,
        params,
        head_end,
        close_end,
    })
}

/// Replace the fn's parameter list with rebuilt params (one per line,
/// original indentation kept). The head (`fn f(`), the matching close
/// paren, and everything after it are preserved byte-exact — the splice
/// provably preserves bracket balance. Byte-anchored; the compiler
/// judges the semantics.
fn splice_fn_sig(
    path: &Path,
    content: &str,
    lines: &[&str],
    sig: &FnSig,
    params: &[String],
) -> Option<SourceEdit> {
    let starts = line_starts(content);
    let ls = *starts.get(sig.ps)?;
    let le = starts.get(sig.pe + 1).copied().unwrap_or(content.len());
    let old = content.get(ls..le)?.to_string();
    let head = content.get(ls..sig.head_end)?;
    let tail = content.get(sig.close_end..le)?;
    let indent: String = lines[sig.ps]
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect();
    let mut out = String::new();
    out.push_str(head);
    out.push('\n');
    for (i, p) in params.iter().enumerate() {
        out.push_str(&indent);
        out.push_str("    ");
        out.push_str(p);
        if i + 1 < params.len() {
            out.push(',');
        }
        out.push('\n');
    }
    // Re-emit the matching closer the scanner stopped after, then the
    // untouched tail (`-> Ret {`, `where …`). Balance preserved.
    out.push_str(&indent);
    out.push(')');
    out.push_str(tail.trim_start());
    if !out.ends_with('\n') {
        out.push('\n');
    }
    Some(SourceEdit {
        file: path.to_path_buf(),
        start: ls,
        end: le,
        expected_old: old,
        replacement: out,
    })
}

/// Name defined by a fn-definition line, if any. Handles qualifiers
/// (`pub`, `pub(crate)`, `async`, `unsafe`, `extern`) — a bare
/// `starts_with("fn ")` misses `pub fn` items and misattributes the
/// enclosing scope (the bot once blamed `sec` for `Settings`' body).
fn fn_def_name(line: &str) -> Option<String> {
    let mut t = line.trim_start();
    if let Some(rest) = t.strip_prefix("pub") {
        // `pub`, `pub(crate)`, `pub(super)`, …
        if rest.starts_with('(') {
            let end = rest.find(')')?;
            t = rest[end + 1..].trim_start();
        } else if rest.starts_with([' ', '\t']) {
            t = rest.trim_start();
        } else {
            return None;
        }
    }
    t = t.strip_prefix("async ").unwrap_or(t);
    t = t.strip_prefix("unsafe ").unwrap_or(t);
    t = t.strip_prefix("extern ").unwrap_or(t);
    // `extern "C" fn` form.
    if t.starts_with('"') {
        let end = t[1..].find('"')?;
        t = t[1 + end + 1..].trim_start();
    }
    let rest = t.strip_prefix("fn ")?;
    if !rest.contains('(') {
        return None;
    }
    let name: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() {
        return None;
    }
    Some(name)
}

/// True when the line holds a `move ||` / `move |…|` closure opener
/// (`move` as a standalone keyword — `remove |x|` must not match).
fn has_move_closure(line: &str) -> bool {
    let bytes: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i + 4 < bytes.len() {
        if bytes[i..].starts_with(&['m', 'o', 'v', 'e'])
            && (i == 0 || (!bytes[i - 1].is_ascii_alphanumeric() && bytes[i - 1] != '_'))
            && (bytes[i + 4] == ' ' || bytes[i + 4] == '|' || bytes[i + 4] == '\t')
        {
            return true;
        }
        i += 1;
    }
    false
}

/// Enclosing call of the closure starting at `close_idx`: scan upward
/// (cap 25 lines) for the topmost line that leaves parens unbalanced
/// through the closure start and holds a `name(` call. Returns the
/// callee name plus the closure's 0-based argument index (top-level
/// commas before `move`, strings stripped).
fn enclosing_call(lines: &[&str], close_idx: usize) -> Option<(String, usize)> {
    let lo = close_idx.saturating_sub(25);
    // Closure start offset within its line: first standalone `move`
    // (byte index — `move` is ASCII so char/byte offsets coincide here).
    let cline = lines.get(close_idx)?;
    let bytes = cline.as_bytes();
    let mut move_pos = None;
    for i in 0..bytes.len().saturating_sub(4) {
        if &bytes[i..i + 4] == b"move"
            && (i == 0 || (!bytes[i - 1].is_ascii_alphanumeric() && bytes[i - 1] != b'_'))
            && (bytes[i + 4] == b' ' || bytes[i + 4] == b'|' || bytes[i + 4] == b'\t')
        {
            // Byte index == char index only if the prefix is ASCII.
            if cline[..i].is_ascii() {
                move_pos = Some(i);
                break;
            }
            return None;
        }
    }
    let move_pos = move_pos?;
    // Byte prefix for arg counting: lines[lo..close] + line head.
    // All three bracket kinds nest: `&[(a, b), (c, d)]` holds commas
    // at paren-depth 1 that belong to the bracket, not the call.
    for s in (lo..=close_idx).rev() {
        let region: String = lines[s..close_idx].join("\n") + "\n" + &cline[..move_pos];
        let stripped = strip_line_strings(&region);
        let depth: i32 = stripped
            .chars()
            .map(|c| match c {
                '(' | '[' | '{' => 1,
                ')' | ']' | '}' => -1,
                _ => 0,
            })
            .sum();
        if depth <= 0 {
            continue;
        }
        // Topmost unclosed line holding a call: take the FIRST ident(
        // on line s... but the call may start earlier; s is our window
        // start only when s == lo. Prefer: find call on the earliest
        // line of the region that opens an unbalanced paren.
        let sline = strip_line_strings(lines[s]);
        let mut callee = None;
        let mut idx = 0;
        let chars: Vec<char> = sline.chars().collect();
        while idx < chars.len() {
            if chars[idx] == '(' {
                let mut j = idx;
                while j > 0
                    && (chars[j - 1].is_ascii_alphanumeric()
                        || chars[j - 1] == '_'
                        || chars[j - 1] == ':')
                {
                    j -= 1;
                }
                // Skip method calls (`x.f(`) and paths keep last ident.
                let raw: String = chars[j..idx].iter().collect();
                let last_seg = raw.rsplit("::").next().unwrap_or("");
                let last_seg = last_seg.rsplit('.').next().unwrap_or("");
                if last_seg.contains('.') || last_seg.is_empty() {
                    // method call — not our callee; keep scanning
                } else if is_ident(last_seg) && !last_seg.starts_with('.') {
                    // Heuristic: first free-function call on the line.
                    if raw.contains('.') {
                        // method call like `a.b(`: skip
                    } else {
                        callee = Some(last_seg.to_string());
                        break;
                    }
                }
                idx += 1;
            } else {
                idx += 1;
            }
        }
        if callee.is_none() {
            continue;
        }
        // Argument index: top-level commas between the call paren and
        // `move`, at depth 1 relative to the call. The head line is
        // truncated at `move` FIRST (a trailing `, vec![]` after the
        // closure must never count), then stripped, then counted.
        let head = &cline[..move_pos];
        let text = if s == close_idx {
            let shead = strip_line_strings(head);
            let call_start = shead.find(&format!("{}(", callee.clone().unwrap()))?;
            shead[call_start..].to_string()
        } else {
            let call_start = sline.find(&format!("{}(", callee.clone().unwrap()))?;
            let seg = &sline[call_start..];
            // Multi-line args: extend with following lines up to close_idx.
            let mut text = seg.to_string();
            for (k, line) in lines.iter().enumerate().take(close_idx + 1).skip(s + 1) {
                if k == close_idx {
                    text.push_str(head);
                } else {
                    text.push('\n');
                    text.push_str(line);
                }
            }
            text
        };
        let tstripped = strip_line_strings(&text);
        let mut d = 0i32;
        let mut commas = 0usize;
        let mut seen_open = false;
        for c in tstripped.chars() {
            if c == '(' || c == '[' || c == '{' {
                d += 1;
                if c == '(' {
                    seen_open = true;
                }
            } else if c == ')' || c == ']' || c == '}' {
                d -= 1;
            } else if c == ',' && seen_open && d == 1 {
                commas += 1;
            }
        }
        return Some((callee.unwrap(), commas));
    }
    None
}

/// R-fnmut-call: E0277 ``expected an `Fn()` closure, found `impl
/// FnMut() + 'static``` at a call `callee(a, b, f, …)` where one bare
/// argument `f` is an `impl FnMut` parameter of the enclosing fn (it
/// flowed down from an already-relaxed helper like `sec`). The callee's
/// same-index parameter relaxes `impl Fn` → `mut …: impl FnMut`, pushing
/// the relaxation upstream one level per round until the chain compiles.
/// Anything else (no such argument, external callee) refuses.
fn migrate_fnmut_call(
    path: &Path,
    content: &str,
    error: &super::error::CompileError,
) -> Option<SourceEdit> {
    if error.code != "E0277"
        || !error.message.contains("expected an `Fn")
        || !error.message.contains("FnMut")
        || error.line == 0
    {
        return None;
    }
    let lines: Vec<&str> = content.lines().collect();
    let idx = (error.line as usize).checked_sub(1)?;
    let line = lines.get(idx)?;
    let sline = strip_line_strings(line);
    // Outermost call on the line: first bare `ident(`. Method calls
    // (`x.f(`), paths (`a::f(`), and macros (`rsx!(` — the walk stops
    // at `!`, leaving a name not followed by `(`) never match.
    let (callee, paren_at) = {
        let chars: Vec<char> = sline.chars().collect();
        let mut found = None;
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '(' {
                let mut j = i;
                while j > 0 && (chars[j - 1].is_ascii_alphanumeric() || chars[j - 1] == '_') {
                    j -= 1;
                }
                let name: String = chars[j..i].iter().collect();
                let rooted = j == 0 || (chars[j - 1] != '.' && chars[j - 1] != ':');
                if rooted && is_ident(&name) {
                    found = Some((name, i));
                    break;
                }
            }
            i += 1;
        }
        found?
    };
    // Top-level arguments of that call (may run past the line end for
    // multi-line calls — cap 10 lines). Starts AFTER the opening paren
    // (char-indexed: `paren_at` counts chars, and the paren itself is
    // ASCII, so skipping it cannot split a boundary).
    let schars: Vec<char> = sline.chars().collect();
    let mut text: String = schars.get(paren_at + 1..).unwrap_or(&[]).iter().collect();
    for line in lines.iter().take((idx + 10).min(lines.len())).skip(idx + 1) {
        text.push('\n');
        text.push_str(&strip_line_strings(line));
    }
    let mut args: Vec<String> = Vec::new();
    let mut cur = String::new();
    // Depth starts at 1: `text` begins right after the call's opening
    // paren, whose match ends the argument list (the final argument is
    // pushed there — a trailing `cur` after the loop means the call ran
    // past the 10-line window and is unusable).
    let mut d = 1i32;
    let mut closed = false;
    for c in text.chars() {
        match c {
            '(' | '[' | '{' => {
                d += 1;
                cur.push(c);
            }
            ')' | ']' | '}' => {
                d -= 1;
                if d == 0 {
                    if !cur.trim().is_empty() {
                        args.push(cur.trim().to_string());
                    }
                    closed = true;
                    break;
                }
                cur.push(c);
            }
            ',' if d == 1 => {
                args.push(cur.trim().to_string());
                cur = String::new();
            }
            _ => cur.push(c),
        }
    }
    if !closed {
        return None;
    }
    // Enclosing fn and its signature (qualifier-aware).
    let err_cap = idx.min(lines.len().saturating_sub(1));
    let mut fn_idx = None;
    for k in (err_cap.saturating_sub(150)..=err_cap).rev() {
        if fn_def_name(lines[k]).is_some() {
            fn_idx = Some(k);
            break;
        }
    }
    let fn_idx = fn_idx?;
    if fn_def_name(lines[fn_idx]).as_deref() == Some(callee.as_str()) {
        return None; // recursive call — not a relaxation chain
    }
    let sig = fn_sig(content, &lines, fn_idx)?;
    for (ai, arg) in args.iter().enumerate() {
        let arg = arg.trim().trim_start_matches('&');
        if !is_ident(arg) {
            continue;
        }
        // The argument must be an `impl FnMut` parameter of the
        // enclosing fn — proof the value already carries FnMut-ness.
        let is_fnmut_param = sig
            .params
            .iter()
            .any(|p| param_binding(p) == arg && p.contains("FnMut"));
        if !is_fnmut_param {
            continue;
        }
        // Callee: local, single definition; relax its same-index param.
        let mut defs = Vec::new();
        for (k, line) in lines.iter().enumerate() {
            let t = line.trim_start();
            if t.starts_with(&format!("fn {}(", callee))
                || t.starts_with(&format!("fn {}<", callee))
            {
                defs.push(k);
            }
        }
        if defs.len() != 1 {
            continue;
        }
        if let Some(edit) = relax_nth_param(path, content, &lines, defs[0], ai) {
            return Some(edit);
        }
    }
    None
}

/// R-string: E0283 on a line holding a string-literal `.into()`. The
/// common cause is an ambiguous `Into` (a third-party `Into<SafeString>`
/// collides with the blanket `Into<String>`); knock-on inference
/// failures on the same line share the code and the cure. Component
/// props take `String`, so rewrite every string-literal `.into()` on
/// the flagged line to `.to_string()`. Gated on code + line shape only
/// (the parser keeps just the header line, so sub-notes are invisible).
/// One line-span edit; the compiler judges whether `String` was right —
/// a wrong guess surfaces as a fresh error, never silence.
fn migrate_into_to_string(
    path: &Path,
    content: &str,
    error: &super::error::CompileError,
) -> Option<SourceEdit> {
    if error.code != "E0283" {
        return None;
    }
    if error.line == 0 {
        return None;
    }
    let lines: Vec<&str> = content.lines().collect();
    let idx = (error.line as usize).checked_sub(1)?;
    let line = lines.get(idx)?.to_string();
    // Walk the line: clean `"lit"` immediately followed by `.into()`.
    // A literal is clean when it holds no backslash or braces (so
    // `format!("…")` strings and interpolated shapes copy through
    // verbatim — never rewritten, never aborting the scan).
    let bytes: Vec<char> = line.chars().collect();
    let mut i = 0;
    let mut fixed = String::with_capacity(line.len() + 16);
    let mut n = 0u32;
    while i < bytes.len() {
        if bytes[i] != '"' {
            fixed.push(bytes[i]);
            i += 1;
            continue;
        }
        let mut j = i + 1;
        let mut lit = String::new();
        let mut clean = true;
        while j < bytes.len() && bytes[j] != '"' {
            if bytes[j] == '\\' || bytes[j] == '{' || bytes[j] == '}' {
                clean = false;
            }
            lit.push(bytes[j]);
            j += 1;
        }
        if j >= bytes.len() {
            return None; // unmatched quote: refuse, never half-rewrite
        }
        let after: String = bytes[j + 1..].iter().collect();
        if clean && after.starts_with(".into()") {
            fixed.push_str(&format!("\"{}\".to_string()", lit));
            i = j + 1 + ".into()".len();
            n += 1;
        } else {
            // Copy the whole literal through untouched.
            fixed.push('"');
            fixed.push_str(&lit);
            fixed.push('"');
            i = j + 1;
        }
    }
    if n == 0 {
        return None;
    }
    let starts = line_starts(content);
    let ls = *starts.get(idx)?;
    let le = starts.get(idx + 1).copied().unwrap_or(content.len());
    let old = content.get(ls..le)?.to_string();
    Some(SourceEdit {
        file: path.to_path_buf(),
        start: ls,
        end: le,
        expected_old: old,
        replacement: fixed + "\n",
    })
}

/// Byte offset where each line starts (line 0 → 0). `\n`-based; every
/// produced offset is a char boundary.
fn line_starts(content: &str) -> Vec<usize> {
    let mut starts = vec![0];
    for (i, c) in content.char_indices() {
        if c == '\n' {
            starts.push(i + 1);
        }
    }
    starts
}

/// Strip double-quoted spans from a line so brace counting never trips on
/// `"{"` inside strings. Naive but sound for this purpose: an unmatched
/// quote degrades to counting everything (a missed transform, never a
/// wrong one — the compiler still judges).
fn strip_line_strings(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_str = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_str {
            if c == '\\' {
                chars.next();
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        if c == '"' {
            in_str = true;
            continue;
        }
        // Line comments end string scanning for brace purposes.
        if c == '/' && chars.peek() == Some(&'/') {
            break;
        }
        out.push(c);
    }
    out
}

fn count_char(s: &str, target: char) -> usize {
    s.chars().filter(|c| *c == target).count()
}

/// Backward brace match: enclosing `{` opener strictly above `from_idx`.
/// The diagnostic line itself is excluded: a trailing `{` there (e.g. the
/// `if … {` of a multi-line `let`) is not an enclosing block.
fn block_open(lines: &[&str], from_idx: usize) -> Option<usize> {
    let mut depth = 0i32;
    for i in (0..from_idx.min(lines.len())).rev() {
        let s = strip_line_strings(lines[i]);
        // Walk right-to-left: `}` deepens, `{` may open.
        let mut chars: Vec<char> = s.chars().collect();
        while let Some(c) = chars.pop() {
            if c == '}' {
                depth += 1;
            } else if c == '{' {
                if depth == 0 {
                    return Some(i);
                }
                depth -= 1;
            }
        }
    }
    None
}

/// Forward match from an opener line to its closing line.
fn block_close(lines: &[&str], open_idx: usize) -> Option<usize> {
    let open_line = strip_line_strings(lines.get(open_idx)?);
    let first = open_line.find('{')?;
    let mut depth = 1i32;
    let rest = &open_line[first + 1..];
    depth += count_char(rest, '{') as i32 - count_char(rest, '}') as i32;
    if depth <= 0 {
        return Some(open_idx);
    }
    for (k, line) in lines.iter().enumerate().skip(open_idx + 1) {
        let s = strip_line_strings(line);
        depth += count_char(&s, '{') as i32 - count_char(&s, '}') as i32;
        if depth <= 0 {
            return Some(k);
        }
    }
    None
}

/// R-let: `let` bindings directly inside an rsx `for` body or element
/// body fail 0.7 parsing. Wrap: `{ lets… rsx! { element } }`.
/// Triggers on the diagnostic line; validates lets-then-one-element shape.
fn migrate_rsx_let(path: &Path, content: &str, line_1based: u32) -> Option<SourceEdit> {
    if line_1based == 0 {
        return None;
    }
    let lines: Vec<&str> = content.lines().collect();
    let idx = (line_1based as usize).checked_sub(1)?;
    // The diagnostic may point mid-statement (a continuation line of the
    // `let`). Resolve to the statement's opening `let` line: nearest
    // `let ` at most 3 lines above, with no blank lines, closers, or
    // unrelated statements between.
    let mut start_idx = None;
    for back in 0..=3 {
        let k = idx.checked_sub(back)?;
        let t = lines.get(k)?.trim();
        if t.is_empty() || (t.starts_with('}') && !t.contains('{')) {
            break;
        }
        if t.starts_with("let ") {
            start_idx = Some(k);
            break;
        }
    }
    let idx = start_idx?;
    let open_idx = block_open(&lines, idx)?;
    let open_trimmed = lines[open_idx].trim();
    // Only rewrite inside rsx! territory: scan upward for the nearest
    // `rsx!` vs item boundary. Plain-Rust `if` blocks with lets must
    // never be touched (wrapping them in rsx! would corrupt logic).
    if !in_rsx_context(&lines, open_idx) {
        return None;
    }
    // `if COND {` blocks with lets fail identically to for-bodies, and
    // wrap the same way (the `} else {` line stays put as the close edge).
    let is_if = open_trimmed.starts_with("if ") && !open_trimmed.contains("let ");
    let is_for = open_trimmed.starts_with("for ");
    let is_element = !is_for && !is_if && {
        let first_tok: String = open_trimmed
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        !first_tok.is_empty()
            && !matches!(
                first_tok.as_str(),
                "fn" | "if"
                    | "else"
                    | "match"
                    | "while"
                    | "loop"
                    | "unsafe"
                    | "impl"
                    | "mod"
                    | "struct"
                    | "enum"
                    | "trait"
                    | "pub"
                    | "use"
                    | "const"
                    | "static"
                    | "extern"
                    | "return"
                    | "break"
                    | "continue"
                    | "async"
                    | "move"
                    | "in"
                    | "where"
                    | "let"
                    | "mut"
                    | "ref"
                    | "self"
                    | "Self"
                    | "true"
                    | "false"
            )
            && open_trimmed.contains('{')
            && !open_trimmed.contains("=>")
    };
    if !is_for && !is_if && !is_element {
        return None;
    }
    let close_idx = block_close(&lines, open_idx)?;
    if close_idx <= idx + 1 {
        return None;
    }
    // Partition inner into statements: leading `let` statements (which may
    // span lines via braces AND via parens/method chains like
    // `.map(|t| …)`), then the element. Once a `let` opens, lines belong
    // to it until `;` at brace depth 0 — line starts mean nothing mid
    // statement.
    let mut let_end = idx;
    let mut depth = 0i32;
    let mut saw_let = false;
    let mut in_let_stmt = false;
    let mut k = idx;
    while k < close_idx {
        let t = lines[k].trim();
        if t.is_empty() || t.starts_with("//") {
            k += 1;
            continue;
        }
        if !in_let_stmt {
            // Between statements: only a `let` continues the run.
            if depth != 0 {
                break; // malformed; element validation below decides
            }
            if t.starts_with("let ") {
                saw_let = true;
                in_let_stmt = true;
            } else {
                break; // element begins
            }
        }
        let s = strip_line_strings(t);
        depth += count_char(&s, '{') as i32 - count_char(&s, '}') as i32;
        if depth < 0 {
            break;
        }
        if in_let_stmt && depth == 0 && t.contains(';') {
            let_end = k;
            in_let_stmt = false;
        }
        k += 1;
    }
    if !saw_let {
        return None;
    }
    // Element chunk: everything after the lets — non-blank, balanced,
    // and opening like an rsx child (element, macro, or control flow).
    // Anything else (a stray statement, an item) refuses here instead of
    // manufacturing a broken wrap the compiler must then reject.
    let elem_text = lines[let_end + 1..close_idx].join("\n");
    if elem_text.trim().is_empty() {
        return None;
    }
    let elem_first = elem_text
        .lines()
        .map(|l| l.trim())
        .find(|l| !l.is_empty() && !l.starts_with("//"));
    match elem_first {
        Some(f)
            if f.starts_with("let ")
                || f.starts_with("fn ")
                || f.starts_with("pub ")
                || f.starts_with("use ")
                || f.starts_with("struct ")
                || f.starts_with("enum ")
                || f.starts_with("impl ")
                || f.starts_with("mod ")
                || f.starts_with("const ")
                || f.starts_with("static ")
                || f.starts_with("return ") =>
        {
            return None;
        }
        None => return None,
        _ => {}
    }
    let elem_stripped = strip_line_strings(&elem_text);
    if count_char(&elem_stripped, '{') != count_char(&elem_stripped, '}') {
        return None;
    }
    let starts = line_starts(content);
    let range_start = *starts.get(open_idx + 1)?;
    let range_end = *starts.get(close_idx)?;
    let expected_old = content.get(range_start..range_end)?.to_string();
    let indent: String = lines[open_idx]
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect();
    let lets: String = lines[idx..=let_end]
        .iter()
        .map(|l| format!("{}{}", indent, l.trim_start()))
        .collect::<Vec<_>>()
        .join("\n");
    // The element may already be an `rsx!` block (nested rsx! is illegal),
    // in which case wrap only the lets around it verbatim.
    let replacement = if elem_text.trim_start().starts_with("rsx!") {
        format!(
            "\n{}    {{\n{}\n{}{}}}\n{}",
            indent, lets, elem_text, indent, indent
        )
    } else {
        format!(
            "\n{}    {{\n{}\n{}    rsx! {{\n{}\n{}    }}\n{}}}\n{}",
            indent, lets, indent, elem_text, indent, indent, indent
        )
    };
    Some(SourceEdit {
        file: path.to_path_buf(),
        start: range_start,
        end: range_end,
        expected_old,
        replacement,
    })
}

/// True when the block at `open_idx` sits inside an `rsx!` invocation:
/// nearest `rsx!` token above wins over item boundaries. Guards plain-Rust
/// blocks (whose lets must never gain an `rsx!` wrapper) from migration.
fn in_rsx_context(lines: &[&str], open_idx: usize) -> bool {
    for (checked, i) in (0..open_idx.min(lines.len())).rev().enumerate() {
        if checked > 80 {
            break;
        }
        let t = lines[i].trim();
        if t.contains("rsx!") {
            return true;
        }
        if t.starts_with("fn ")
            || t.starts_with("pub ")
            || t.starts_with("impl ")
            || t.starts_with("mod ")
            || t.starts_with("struct ")
            || t.starts_with("enum ")
            || t.starts_with("trait ")
            || t.starts_with("const ")
            || t.starts_with("static ")
        {
            return false;
        }
    }
    false
}

/// Closing line of the `match` starting at `match_idx` (brace match).
/// `None` when unbalanced — the caller refuses rather than guessing.
fn match_span_end(lines: &[&str], match_idx: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut opened = false;
    for (k, line) in lines.iter().enumerate().skip(match_idx) {
        let s = strip_line_strings(line.trim());
        for c in s.chars() {
            if c == '{' {
                depth += 1;
                opened = true;
            } else if c == '}' {
                depth -= 1;
                if opened && depth == 0 {
                    return Some(k);
                }
            }
        }
    }
    None
}

/// R-resource: 0.5 `match X()` on a `Resource` → 0.7 `match &*X.read()`,
/// plus `Some(ref name)` → `Some(name)` arm normalization in the same
/// match span (the scrutinee is already borrowed — `ref` would double it).
/// Returns every edit; all apply atomically or not at all.
fn migrate_resource_read(
    path: &Path,
    content: &str,
    error: &super::error::CompileError,
) -> Option<Vec<SourceEdit>> {
    if error.code != "E0618" || error.line == 0 {
        return None;
    }
    if !error.message.contains("Resource") {
        return None;
    }
    let lines: Vec<&str> = content.lines().collect();
    let idx = (error.line as usize).checked_sub(1)?;
    let line = lines.get(idx)?.to_string();
    // `match tokens() {` — callee must be a bare identifier call.
    let mpos = line.find("match")?;
    let after = line[mpos + "match".len()..].trim_start();
    let paren = after.find('(')?;
    let callee = after[..paren].trim();
    if !is_ident(callee) {
        return None;
    }
    let rest = after[paren..].trim_start_matches('(');
    if !rest.trim_start().starts_with(')') {
        return None;
    }
    let starts = line_starts(content);
    let line_start = *starts.get(idx)?;
    // Span the whole line including its newline when present, so the
    // replacement preserves file shape exactly.
    let end = starts.get(idx + 1).copied().unwrap_or(content.len());
    let expected_old = content.get(line_start..end)?.to_string();
    let new_line = line.replacen(&format!("{}()", callee), &format!("&*{}.read()", callee), 1);
    let mut edits = vec![SourceEdit {
        file: path.to_path_buf(),
        start: line_start,
        end,
        expected_old,
        replacement: new_line,
    }];
    // The scrutinee is now `&*…`: a `Some(ref name)` arm pattern would
    // double-borrow (`&&Vec`), so normalize arm patterns in the match span.
    // Only the exact `Some(ref IDENT)` shape is touched. Bound names are
    // collected: their `.clone()` calls below must become `.to_vec()`,
    // because the binding changed from owned Vec to borrowed &Vec and
    // `Clone for &T` would clone the reference instead of the vector.
    // One edit per line max: overlapping ranges would stale-check each
    // other at apply time.
    let mut bound: Vec<String> = Vec::new();
    let mut touched: Vec<usize> = Vec::new();
    if let Some(span_end) = match_span_end(&lines, idx) {
        for (k, span_line) in lines.iter().enumerate().skip(idx + 1).take(span_end - idx) {
            let t = span_line.trim();
            if !(t.starts_with("Some(ref ") && t.contains("=>")) {
                continue;
            }
            // Binding ident between `Some(ref ` and the next delimiter.
            let after = &t["Some(ref ".len()..];
            let name: String = after
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if name.is_empty() {
                continue;
            }
            bound.push(name.clone());
            touched.push(k);
            let fixed = t.replacen("Some(ref ", "Some(", 1);
            let ls = *starts.get(k)?;
            let le = starts.get(k + 1).copied().unwrap_or(content.len());
            let old = content.get(ls..le)?.to_string();
            edits.push(SourceEdit {
                file: path.to_path_buf(),
                start: ls,
                end: le,
                expected_old: old,
                replacement: fixed + "\n",
            });
        }
        // Carry-through: `<binding>.clone()` → `<binding>.to_vec()` for
        // normalized bindings anywhere in the span. Other `.clone()` calls
        // (autoderef on owned fields) are provably unaffected — untouched.
        for (k, span_line) in lines.iter().enumerate().skip(idx + 1).take(span_end - idx) {
            if touched.contains(&k) {
                continue;
            }
            for name in &bound {
                let needle = format!("{}.clone()", name);
                if !span_line.contains(&needle) {
                    continue;
                }
                let fixed_line = span_line.replacen(&needle, &format!("{}.to_vec()", name), 1);
                let ls = *starts.get(k)?;
                let le = starts.get(k + 1).copied().unwrap_or(content.len());
                let old = content.get(ls..le)?.to_string();
                edits.push(SourceEdit {
                    file: path.to_path_buf(),
                    start: ls,
                    end: le,
                    expected_old: old,
                    replacement: fixed_line,
                });
                touched.push(k);
                break;
            }
        }
    }
    Some(edits)
}

/// Add `Clone` to `struct X`'s derive list in the file, or insert a fresh
/// `#[derive(Clone)]` when no derive attribute exists. Exactly one
/// `struct X` must exist, else refuse (zero = elsewhere, many = unclear).
fn add_clone_derive(path: &Path, ty: &str) -> Option<SourceEdit> {
    let content = fs::read_to_string(path).ok()?;
    let lines: Vec<&str> = content.lines().collect();
    let mut defs = Vec::new();
    for (k, line) in lines.iter().enumerate() {
        let t = line.trim();
        // `struct X`, `pub struct X`, `pub(crate) struct X` + optional `<T>`.
        let rest = t
            .strip_prefix("pub(crate) struct ")
            .or_else(|| t.strip_prefix("pub struct "))
            .or_else(|| t.strip_prefix("struct "))
            .unwrap_or("");
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if name == ty {
            defs.push(k);
        }
    }
    if defs.len() != 1 {
        return None;
    }
    let def_idx = defs[0];
    let starts = line_starts(&content);
    // Walk upward past attributes: extend an existing derive or insert one.
    let mut k = def_idx;
    let mut derive_idx = None;
    while k > 0 {
        k -= 1;
        let t = lines[k].trim();
        if t.is_empty() || t.starts_with("//") {
            continue;
        }
        if t.starts_with("#[derive(") && t.ends_with(")]") {
            derive_idx = Some(k);
        }
        break;
    }
    if let Some(d) = derive_idx {
        let line = lines[d];
        if line.contains("Clone") {
            return None; // already derived — nothing to do honestly
        }
        // `#[derive(A, B)]\n` → `#[derive(A, B, Clone)]\n`, same span.
        let trimmed = line.trim();
        let inner = trimmed.strip_prefix("#[derive(")?.strip_suffix(")]")?;
        let indent: String = line
            .chars()
            .take_while(|c| *c == ' ' || *c == '\t')
            .collect();
        let ls = *starts.get(d)?;
        let le = starts.get(d + 1).copied().unwrap_or(content.len());
        let old = content.get(ls..le)?.to_string();
        Some(SourceEdit {
            file: path.to_path_buf(),
            start: ls,
            end: le,
            expected_old: old,
            replacement: format!("{}#[derive({}, Clone)]\n", indent, inner),
        })
    } else {
        // Fresh attribute above the struct.
        let ls = *starts.get(def_idx)?;
        let indent: String = lines[def_idx]
            .chars()
            .take_while(|c| *c == ' ' || *c == '\t')
            .collect();
        Some(SourceEdit {
            file: path.to_path_buf(),
            start: ls,
            end: ls,
            expected_old: String::new(),
            replacement: format!("{}#[derive(Clone)]\n", indent),
        })
    }
}

/// R-await: `.await` inside a sync `move ||` closure → wrap the closure
/// body in `spawn(async move { … })`.
fn migrate_await_spawn(path: &Path, content: &str, line_1based: u32) -> Option<SourceEdit> {
    if line_1based == 0 {
        return None;
    }
    let lines: Vec<&str> = content.lines().collect();
    let idx = (line_1based as usize).checked_sub(1)?;
    if !lines.get(idx)?.contains(".await") {
        return None;
    }
    // Nearest `move … {` opener above, without crossing item/lone-close
    // boundaries (those mean we already left the closure).
    let mut open_idx = None;
    for i in (0..=idx.min(lines.len().saturating_sub(1))).rev() {
        let t = lines[i].trim();
        if t.starts_with("fn ")
            || t.starts_with("pub ")
            || t.starts_with("#[")
            || (t.starts_with('}') && !t.contains('{'))
        {
            break;
        }
        if t.contains("move") && t.contains('{') && !t.contains("async") {
            open_idx = Some(i);
            break;
        }
    }
    let open_idx = open_idx?;
    let close_idx = block_close(&lines, open_idx)?;
    if close_idx <= idx {
        return None;
    }
    // Refuse if already inside async/spawn.
    if lines[open_idx..=close_idx]
        .iter()
        .any(|l| l.contains("spawn(") || l.contains("async move"))
    {
        return None;
    }
    let starts = line_starts(content);
    let range_start = *starts.get(open_idx + 1)?;
    let range_end = *starts.get(close_idx)?;
    let expected_old = content.get(range_start..range_end)?.to_string();
    let indent: String = lines[open_idx]
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect();
    let body = &expected_old;
    let replacement = format!(
        "\n{}    spawn(async move {{{}\n{}    }});\n{}",
        indent, body, indent, indent
    );
    Some(SourceEdit {
        file: path.to_path_buf(),
        start: range_start,
        end: range_end,
        expected_old,
        replacement,
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::error::{CompileError, ErrorKind};

    fn err(code: &str, file: &str, line: u32, message: &str) -> CompileError {
        CompileError {
            code: code.to_string(),
            message: message.to_string(),
            file: file.to_string(),
            line,
            col: 0,
            suggestion: None,
            source_line: None,
            kind: ErrorKind::Other,
        }
    }

    const FOR_LET: &str = "fn c() -> Element {\n    rsx! {\n        div {\n            nav { class: \"bottom-nav\",\n                for (i, label) in items.iter().enumerate() {\n                    let active = i as u8 == tab();\n                    button {\n                        span { \"{label}\" }\n                    }\n                }\n            }\n";

    #[test]
    fn rsx_let_wraps_for_body() {
        let path = Path::new("t.rs");
        let edit = migrate_rsx_let(path, FOR_LET, 6).expect("must match");
        // Applying the edit must yield compiling 0.7 shape: the outer
        // fixture rsx! plus exactly one wrapped layer (total 2).
        let mut content = FOR_LET.to_string();
        content.replace_range(edit.start..edit.end, &edit.replacement);
        assert!(content.contains("let active"));
        assert_eq!(content.matches("rsx! {").count(), 2);
    }

    #[test]
    fn rsx_let_rejects_non_shapes() {
        let path = Path::new("t.rs");
        // Plain fn body lets are not rsx shapes.
        let src = "fn f() {\n    let x = 1;\n    x + 1\n}\n";
        assert!(migrate_rsx_let(path, src, 2).is_none());
        // Bare div with no rsx! above: refused (plain-Rust guard).
        let src2 = "            div {\n                let a = 1;\n                span { \"x\" }\n                span { \"y\" }\n            }\n";
        assert!(migrate_rsx_let(path, src2, 2).is_none());
        // Same shape under rsx!: allowed (compiles).
        let src3 = "fn c() -> Element {\n    rsx! {\n        div {\n            div {\n                let a = 1;\n                span { \"x\" }\n            }\n        }\n    }\n}\n";
        assert!(migrate_rsx_let(path, src3, 5).is_some());
    }

    const MATCH_RES: &str = "            div {\n                match tokens() {\n                    Some(ref list) => rsx! { div { \"{list.len()}\" } },\n                    None => rsx! { div { \"loading\" } },\n                }\n            }\n";

    #[test]
    fn resource_read_rewrites_match() {
        let path = Path::new("t.rs");
        let e = err(
            "E0618",
            "t.rs",
            2,
            "expected function, found `Resource<Vec<T>>`",
        );
        let edits = migrate_resource_read(path, MATCH_RES, &e).expect("must match");
        // Match line + arm normalization, applied bottom-up.
        assert_eq!(edits.len(), 2);
        let mut ordered = edits;
        ordered.sort_by_key(|e| std::cmp::Reverse(e.start));
        let mut content = MATCH_RES.to_string();
        for edit in &ordered {
            content.replace_range(edit.start..edit.end, &edit.replacement);
        }
        assert!(content.contains("match &*tokens.read() {"));
        assert!(content.contains("Some(list) =>"));
        assert!(!content.contains("Some(ref "));
    }

    #[test]
    fn resource_read_rejects_bare_calls() {
        let path = Path::new("t.rs");
        let e = err(
            "E0618",
            "t.rs",
            1,
            "expected function, found `Resource<Vec<T>>`",
        );
        assert!(migrate_resource_read(path, "foo(bar())\n", &e).is_none());
    }

    const MUT_SRC: &str = "fn c() -> Element {\n    let tab = use_signal(|| 0u8);\n    rsx! { div { \"{tab}\" } }\n}\n";

    #[test]
    fn mut_binding_adds_mut_once() {
        let path = Path::new("t.rs");
        let e = err(
            "E0596",
            "t.rs",
            3,
            "cannot borrow `tab` as mutable, as it is not declared as mutable",
        );
        let edit = migrate_mut_binding(path, MUT_SRC, &e).expect("must match");
        let mut content = MUT_SRC.to_string();
        content.replace_range(edit.start..edit.end, &edit.replacement);
        assert!(content.contains("let mut tab = use_signal"));
        // Wrong code, missing binding, ambiguous bindings: all refuse.
        assert!(
            migrate_mut_binding(
                path,
                MUT_SRC,
                &err("E0000", "t.rs", 3, "cannot borrow `tab` as mutable")
            )
            .is_none()
        );
        assert!(
            migrate_mut_binding(
                path,
                "fn f() {\n    tab.set(1);\n}\n",
                &err(
                    "E0596",
                    "t.rs",
                    2,
                    "cannot borrow `tab` as mutable, as it is not declared as mutable"
                )
            )
            .is_none()
        );
        let two = "fn f() {\n    let tab = a();\n    let tab = b();\n}\n";
        assert!(
            migrate_mut_binding(
                path,
                two,
                &err(
                    "E0596",
                    "t.rs",
                    2,
                    "cannot borrow `tab` as mutable, as it is not declared as mutable"
                )
            )
            .is_none()
        );
    }

    const MUT_PARAM_SRC: &str = "fn field(label: &str, oninput: impl Fn(String) + 'static) -> Element {\n    rsx! {\n        div {\n            input {\n                oninput: move |e| oninput(e.value()),\n            }\n        }\n    }\n}\n";

    #[test]
    fn mut_binding_relaxes_fn_param() {
        let path = Path::new("t.rs");
        let e = err(
            "E0596",
            "t.rs",
            5,
            "cannot borrow `oninput` as mutable, as it is not declared as mutable",
        );
        let edit = migrate_mut_binding(path, MUT_PARAM_SRC, &e).expect("must match");
        let mut content = MUT_PARAM_SRC.to_string();
        content.replace_range(edit.start..edit.end, &edit.replacement);
        assert!(content.contains("mut oninput: impl FnMut(String) + 'static"));
        assert_eq!(content.matches('(').count(), content.matches(')').count());
        // Plain (non-Fn) parameters refuse.
        let plain = "fn f(flag: bool) -> Element {\n    rsx! { div { \"{flag}\" } }\n}\n";
        assert!(
            migrate_mut_binding(
                path,
                plain,
                &err(
                    "E0596",
                    "t.rs",
                    2,
                    "cannot borrow `flag` as mutable, as it is not declared as mutable"
                )
            )
            .is_none()
        );
    }

    const MUT_CAPTURE_SRC: &str = "fn c() -> Element {\n    let expanded = use_signal(|| 0u8);\n    rsx! {\n        div {\n            {sec(true, \"T\", move || expanded.set(3), vec![])}\n        }\n    }\n}\n";

    #[test]
    fn mut_binding_muts_captured_local() {
        let path = Path::new("t.rs");
        let e = err(
            "E0596",
            "t.rs",
            5,
            "cannot borrow `expanded` as mutable, as it is a captured variable in a `Fn` closure",
        );
        let edit = migrate_mut_binding(path, MUT_CAPTURE_SRC, &e).expect("must match");
        let mut content = MUT_CAPTURE_SRC.to_string();
        content.replace_range(edit.start..edit.end, &edit.replacement);
        assert!(content.contains("let mut expanded = use_signal"));
        // A captured parameter is not a let: refuse (FnMut recipes own it).
        let param_src = "fn s(flag: bool, ontoggle: impl Fn() + 'static) -> Element {\n    rsx! {\n        div { onclick: move |_| ontoggle(), }\n    }\n}\n";
        assert!(
            migrate_mut_binding(
                path,
                param_src,
                &err(
                    "E0596",
                    "t.rs",
                    3,
                    "cannot borrow `ontoggle` as mutable, as it is a captured variable in a `Fn` closure"
                )
            )
            .is_none()
        );
    }

    const INTO_SRC: &str = "                        SummaryCard { label: \"Balance\".into(), value: format!(\"{:.3} SOL\", x), color: \"var(--accent)\".into() }\n";

    #[test]
    fn into_to_string_rewrites_literal_intos() {
        let path = Path::new("t.rs");
        let e = err(
            "E0283",
            "t.rs",
            1,
            "multiple `impl`s satisfying `&str: Into<_>` found",
        );
        let edit = migrate_into_to_string(path, INTO_SRC, &e).expect("must match");
        let mut content = INTO_SRC.to_string();
        content.replace_range(edit.start..edit.end, &edit.replacement);
        assert!(content.contains("label: \"Balance\".to_string()"));
        assert!(content.contains("color: \"var(--accent)\".to_string()"));
        assert!(!content.contains(".into()"));
        // format!() output untouched (String::into is unambiguous).
        assert!(content.contains("format!(\"{:.3} SOL\", x)"));
        // Knock-on inference flavors share the code and the cure: any
        // E0283 on a literal-.into() line fires (the parser keeps only
        // the header line, so sub-notes are invisible by design).
        assert!(
            migrate_into_to_string(
                path,
                INTO_SRC,
                &err("E0283", "t.rs", 1, "type annotations needed")
            )
            .is_some()
        );
        // Other codes and lines without literal .into(): refuse.
        assert!(
            migrate_into_to_string(
                path,
                INTO_SRC,
                &err("E0000", "t.rs", 1, "multiple `impl`s satisfying")
            )
            .is_none()
        );
        assert!(
            migrate_into_to_string(
                path,
                "    let x = y.into();\n",
                &err(
                    "E0283",
                    "t.rs",
                    1,
                    "multiple `impl`s satisfying `u32: Into<_>` found"
                )
            )
            .is_none()
        );
    }

    const FNMUT_A: &str = "fn section(expanded: bool, ontoggle: impl Fn() + 'static) -> Element {\n    rsx! {\n        div { onclick: move |_| ontoggle(), }\n    }\n}\n";

    #[test]
    fn fnmut_relaxes_enclosing_param() {
        let path = Path::new("t.rs");
        let e = err(
            "E0596",
            "t.rs",
            3,
            "cannot borrow `ontoggle` as mutable, as it is a captured variable in a `Fn` closure",
        );
        let edit = migrate_fnmut_param(path, FNMUT_A, &e).expect("must match");
        let mut content = FNMUT_A.to_string();
        content.replace_range(edit.start..edit.end, &edit.replacement);
        assert!(content.contains("mut ontoggle: impl FnMut() + 'static"));
        assert!(!content.contains("impl Fn()"));
        // The splice provably preserves paren balance.
        assert_eq!(content.matches('(').count(), content.matches(')').count());
    }

    const FNMUT_B: &str = "fn sec(flag: bool, title: &str, ontoggle: impl Fn() + 'static) -> Element {\n    rsx! {}\n}\n\nfn c() -> Element {\n    let expanded = use_signal(|| 0u8);\n    rsx! {\n        div {\n            {sec(expanded() == 3, \"Trading\", move || expanded.set(3), vec![])}\n        }\n    }\n}\n";

    #[test]
    fn fnmut_relaxes_callee_param_positionally() {
        let path = Path::new("t.rs");
        let e = err(
            "E0596",
            "t.rs",
            9,
            "cannot borrow `expanded` as mutable, as it is a captured variable in a `Fn` closure",
        );
        let edit = migrate_fnmut_param(path, FNMUT_B, &e).expect("must match");
        let mut content = FNMUT_B.to_string();
        content.replace_range(edit.start..edit.end, &edit.replacement);
        assert!(content.contains("mut ontoggle: impl FnMut() + 'static"));
        // Other params untouched.
        assert!(content.contains("flag: bool"));
        assert!(content.contains("title: &str"));
        assert_eq!(content.matches('(').count(), content.matches(')').count());
    }

    const FNMUT_C: &str = "fn sec(\n    expanded: bool,\n    title: &str,\n    ontoggle: impl Fn() + 'static,\n    children: Vec<Element>,\n) -> Element {\n    section(expanded, title, ontoggle)\n}\n\n#[component]\npub fn Settings() -> Element {\n    let mut expanded = use_signal(|| 0u8);\n    rsx! {\n        div {\n            {sec(expanded() == 1, \"RPC\", move || expanded.set(1), vec![])}\n        }\n    }\n}\n";

    #[test]
    fn fnmut_relaxes_multiline_callee() {
        let path = Path::new("t.rs");
        let e = err(
            "E0596",
            "t.rs",
            15,
            "cannot borrow `expanded` as mutable, as it is a captured variable in a `Fn` closure",
        );
        let edit = migrate_fnmut_param(path, FNMUT_C, &e).expect("must match");
        let mut content = FNMUT_C.to_string();
        content.replace_range(edit.start..edit.end, &edit.replacement);
        assert!(content.contains("mut ontoggle: impl FnMut() + 'static"));
        assert_eq!(content.matches('(').count(), content.matches(')').count());
    }

    const FNMUT_D: &str = "fn dropdown(label: &str, value: &str, options: &[(&str, &str)], onchange: impl Fn(String) + 'static) -> Element {\n    rsx! {}\n}\n\n#[component]\npub fn Settings() -> Element {\n    let mut settings = use_signal(|| 0u8);\n    rsx! {\n        div {\n            {dropdown(\"Landing Mode\", &\"normal\", &[(\"normal\",\"Normal\"),(\"zeroslot\",\"ZeroSlot\")], move |v| settings.set(v))}\n        }\n    }\n}\n";

    #[test]
    fn fnmut_relaxes_dropdown_param() {
        let path = Path::new("t.rs");
        let e = err(
            "E0596",
            "t.rs",
            10,
            "cannot borrow `settings` as mutable, as it is a captured variable in a `Fn` closure",
        );
        let edit = migrate_fnmut_param(path, FNMUT_D, &e).expect("must match");
        let mut content = FNMUT_D.to_string();
        content.replace_range(edit.start..edit.end, &edit.replacement);
        assert!(content.contains("mut onchange: impl FnMut(String) + 'static"));
        assert_eq!(content.matches('(').count(), content.matches(')').count());
    }

    const FNMUT_E: &str = "fn section(expanded: bool, title: &str, ontoggle: impl Fn() + 'static, children: Vec<Element>) -> Element {\n    rsx! {}\n}\n\nfn sec(\n    expanded: bool,\n    title: &str,\n    mut ontoggle: impl FnMut() + 'static,\n    children: Vec<Element>,\n) -> Element {\n    section(expanded, title, ontoggle, rsx! { \"x\" })\n}\n";

    #[test]
    fn fnmut_call_relaxes_upstream_callee() {
        let path = Path::new("t.rs");
        let e = err(
            "E0277",
            "t.rs",
            11,
            "expected an `Fn()` closure, found `impl FnMut() + 'static`",
        );
        let edit = migrate_fnmut_call(path, FNMUT_E, &e).expect("must match");
        let mut content = FNMUT_E.to_string();
        content.replace_range(edit.start..edit.end, &edit.replacement);
        assert!(content.contains("mut ontoggle: impl FnMut() + 'static"));
        // section got exactly one relaxation (sec's own stays).
        assert_eq!(content.matches("impl FnMut()").count(), 2);
        assert_eq!(content.matches('(').count(), content.matches(')').count());
        // Other E0277 flavors and non-call lines: refuse.
        assert!(
            migrate_fnmut_call(path, FNMUT_E, &err("E0277", "t.rs", 12, "expected an `Fn`"))
                .is_none()
        );
        assert!(migrate_fnmut_call(path, "fn f() {}\n", &e).is_none());
    }

    #[test]
    fn fnmut_debug_call() {
        let lines: Vec<&str> = FNMUT_E.lines().collect();
        eprintln!("line11={:?}", lines.get(10));
        eprintln!("fnname3={:?}", fn_def_name(lines[3]));
        eprintln!("fnname4={:?}", fn_def_name(lines[4]));
    }

    #[test]
    fn fnmut_refuses_unguessable_shapes() {
        let path = Path::new("t.rs");
        let e = |line| {
            err(
                "E0596",
                "t.rs",
                line,
                "cannot borrow `v` as mutable, as it is a captured variable in a `Fn` closure",
            )
        };
        // No enclosing fn at all.
        assert!(migrate_fnmut_param(path, "    v.set(1);\n", &e(1)).is_none());
        // Enclosing fn is not local (callee undefined).
        let ext = "fn c() -> Element {\n    rsx! {\n        div {\n            {ext_lib(true, move || v.set(1))}\n        }\n    }\n}\n";
        assert!(migrate_fnmut_param(path, ext, &e(4)).is_none());
        // Wrong code.
        assert!(migrate_fnmut_param(path, FNMUT_A, &err("E0000", "t.rs", 3, "x")).is_none());
    }

    const AWAIT_CLOSURE: &str = "    let stop = move || {\n        engine::stop();\n        status.set(engine::snapshot().await);\n    };\n";

    #[test]
    fn await_wraps_in_spawn() {
        let path = Path::new("t.rs");
        let edit = migrate_await_spawn(path, AWAIT_CLOSURE, 3).expect("must match");
        let mut content = AWAIT_CLOSURE.to_string();
        content.replace_range(edit.start..edit.end, &edit.replacement);
        assert!(content.contains("spawn(async move {"));
        assert!(content.contains(".await"));
    }

    #[test]
    fn await_refuses_async_contexts() {
        let path = Path::new("t.rs");
        let src = "    spawn(async move {\n        foo().await;\n    });\n";
        assert!(migrate_await_spawn(path, src, 2).is_none());
    }
}

#[cfg(test)]
mod migration_extra_tests {
    use super::*;
    use std::path::Path;

    const PREFIXED: &str = "fn c() -> Element {\n    rsx! {\n        div {\n                for token in list {\n                    let m = if a {\n                        b()\n                    } else {\n                        c()\n                    };\n                    rsx! {\n                        div { \"{m}\" }\n                    }\n                }\n";

    #[test]
    fn rsx_let_skips_second_macro_layer() {
        let path = Path::new("t.rs");
        // `let m` moved to line 5 under the fixture's fn/rsx!/div prefix.
        let edit = migrate_rsx_let(path, PREFIXED, 5).expect("must match");
        let mut content = PREFIXED.to_string();
        content.replace_range(edit.start..edit.end, &edit.replacement);
        // Outer fixture rsx! + element's own: no third layer added.
        assert_eq!(content.matches("rsx!").count(), 2);
        assert!(content.contains("let m = if a {"));
    }
}

#[cfg(test)]
mod migration_if_tests {
    use super::*;
    use std::path::Path;

    const IF_LET: &str = "fn c() -> Element {\n    rsx! {\n        div {\n            if ok {\n                let v = compute();\n                span { \"{v}\" }\n            }\n        }\n    }\n}\n";

    #[test]
    fn rsx_let_wraps_if_body() {
        let path = Path::new("t.rs");
        // `let v` is line 5 (1-based).
        let edit = migrate_rsx_let(path, IF_LET, 5).expect("must match");
        let mut content = IF_LET.to_string();
        content.replace_range(edit.start..edit.end, &edit.replacement);
        assert!(content.contains("rsx! {"));
        // Balanced overall.
        assert_eq!(content.matches('{').count(), content.matches('}').count());
    }

    const PLAIN_IF: &str = "fn f(x: bool) -> i32 {\n    if x {\n        let y = 1;\n        y + 1\n    } else {\n        0\n    }\n}\n";

    #[test]
    fn rsx_let_refuses_plain_rust() {
        let path = Path::new("t.rs");
        // `let y` is line 3 and the block is plain Rust: must refuse.
        assert!(migrate_rsx_let(path, PLAIN_IF, 3).is_none());
    }
}
