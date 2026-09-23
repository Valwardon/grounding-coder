use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// A deterministic edit plan — the bot NEVER writes whole files.
///
/// Each plan describes a precise set of source edits backed by evidence.
/// The plan is the "contract" between the LLM's intent and the actual
/// filesystem changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditPlan {
    /// The task this plan belongs to
    pub task_id: u64,
    /// The precise edits to apply
    pub edits: Vec<SourceEdit>,
    /// The evidence justifying each edit
    pub evidence: Vec<Evidence>,
}

/// A single source file modification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceEdit {
    /// The file to modify
    pub file: PathBuf,
    /// Byte offset where the replacement starts
    pub start: usize,
    /// Byte offset where the replacement ends (exclusive)
    pub end: usize,
    /// Text we expect to find at [start,end) — verified before apply.
    /// Empty for pure inserts.
    #[serde(default)]
    pub expected_old: String,
    /// The replacement text
    pub replacement: String,
}

impl std::fmt::Display for SourceEdit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "EDIT")?;
        writeln!(f, "  file: {}", self.file.display())?;
        writeln!(f, "  range: [{}, {})", self.start, self.end)?;
        writeln!(f, "  expected_old: {:?}", truncate(&self.expected_old, 160))?;
        writeln!(f, "  replacement: {:?}", truncate(&self.replacement, 320))
    }
}

impl std::fmt::Display for EditPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "EditPlan task={} edits={}",
            self.task_id,
            self.edits.len()
        )?;
        for e in &self.edits {
            write!(f, "{}", e)?;
        }
        writeln!(f, "EVIDENCE")?;
        for ev in &self.evidence {
            writeln!(f, "  {:?}", ev)?;
        }
        Ok(())
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        // Floor to a char boundary: slicing multibyte text at a raw byte
        // offset panics, and on Android a panic aborts the process.
        let mut m = n;
        while !s.is_char_boundary(m) {
            m -= 1;
        }
        format!("{}…", &s[..m])
    }
}

/// Evidence that justifies an edit.
///
/// The bot may only modify bytes that are backed by one of these:
/// - An existing pattern found in the codebase
/// - A compiler suggestion (with file/line/col)
/// - A verified symbol definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Evidence {
    /// The edit matches an existing pattern in the codebase
    ExistingPattern {
        source_file: String,
        source_line: u32,
    },
    /// The edit follows a compiler suggestion
    CompilerSuggestion {
        code: String,
        file: String,
        line: u32,
        col: u32,
    },
    /// The edit uses a verified symbol definition
    VerifiedSymbol { qname: String, source: String },
}

/// The state machine for a task execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// The task is being planned
    Planning,
    /// The task is being resolved (symbol lookup)
    Resolving,
    /// The task is being edited
    Editing,
    /// The task is being verified
    Verifying,
    /// The task is being repaired (if verification failed)
    Repairing,
    /// The task completed successfully
    Complete,
    /// The task is blocked (no safe fix available)
    Blocked,
}

impl std::fmt::Display for TaskState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            TaskState::Planning => "planning",
            TaskState::Resolving => "resolving",
            TaskState::Editing => "editing",
            TaskState::Verifying => "verifying",
            TaskState::Repairing => "repairing",
            TaskState::Complete => "complete",
            TaskState::Blocked => "blocked",
        };
        write!(f, "{}", label)
    }
}
