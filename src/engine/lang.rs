//! Language backends — the polyglot plug-in point.
//!
//! The `EditPlan` protocol (file + byte range + evidence) is language
//! agnostic. What differs per language is three things, all behind the
//! [`LanguageBackend`] trait:
//!   1. edit conventions (how imports look, how to spot them),
//!   2. the verification oracle (compiler / interpreter / test runner),
//!   3. the known-std inventory (what the planner may import unprompted).
//!
//! Rust and Kotlin/Android verification keeps living in `verifier.rs`
//! (mature paths); Python and C verify here. Adding a language means one
//! of two things, neither touching the engine:
//!   1. a hand-tuned struct below (for rich multi-stage oracles), or
//!   2. pure data — a [`LanguageSpec`] in the registry, or a project-local
//!      `.grounding.toml` (`[language]` table) for anything else on earth.
//!
//! Unknown languages with no spec refuse honestly instead of guessing.

use super::error::ErrorKind;
use super::verifier::VerificationResult;
use crate::engine::error::CompileError;
use serde::Deserialize;
use std::path::Path;
use std::process::Command;

/// One language's edit conventions plus its verification oracle.
pub trait LanguageBackend {
    fn language(&self) -> &str;
    fn extensions(&self) -> &'static [&'static str];
    /// Render an import line for `path` (no trailing newline handling —
    /// callers add it).
    fn import_line(&self, path: &str) -> String;
    /// True when `line` is already an import in this language.
    fn is_import_line(&self, line: &str) -> bool;
    /// True for stdlib paths the planner may use as evidence on its own.
    fn is_known_std(&self, symbol: &str) -> bool;
    /// Leading lines imports must go after (Go `package` clause). Default 0.
    fn skip_lines(&self) -> usize {
        0
    }
    /// Run the language oracle over the project. The default refuses
    /// honestly — a backend without an oracle must not report clean.
    fn verify(&self, _project_dir: &Path) -> VerificationResult {
        VerificationResult {
            clean: false,
            errors: vec![CompileError {
                code: "VERIFICATION_UNSUPPORTED".to_string(),
                message: format!(
                    "No verification oracle for language {} on this host",
                    self.language()
                ),
                file: String::new(),
                line: 0,
                col: 0,
                suggestion: None,
                source_line: None,
                kind: ErrorKind::Other,
            }],
            warnings: Vec::new(),
            stdout: String::new(),
            stderr: String::new(),
        }
    }
}

// --- Rust ---

pub struct RustBackend;

impl LanguageBackend for RustBackend {
    fn language(&self) -> &str {
        "rust"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["rs"]
    }

    fn import_line(&self, path: &str) -> String {
        format!("use {};", path.trim().trim_end_matches(';'))
    }

    fn is_import_line(&self, line: &str) -> bool {
        line.trim().starts_with("use ")
    }

    fn is_known_std(&self, symbol: &str) -> bool {
        symbol.starts_with("std::") || symbol.starts_with("core::") || symbol.starts_with("alloc::")
    }
}

// --- Python ---

pub struct PythonBackend;

/// Python stdlib top-level modules the planner may import on its own.
const PY_STDLIB: &[&str] = &[
    "os",
    "sys",
    "json",
    "re",
    "math",
    "pathlib",
    "typing",
    "collections",
    "dataclasses",
    "datetime",
    "time",
    "hashlib",
    "subprocess",
    "argparse",
    "logging",
    "unittest",
    "functools",
    "itertools",
    "enum",
    "abc",
    "io",
    "csv",
    "sqlite3",
    "http",
    "urllib",
    "shutil",
    "glob",
    "tempfile",
    "threading",
    "queue",
    "copy",
    "pprint",
    "statistics",
];

impl LanguageBackend for PythonBackend {
    fn language(&self) -> &str {
        "python"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["py"]
    }

    fn import_line(&self, path: &str) -> String {
        let p = path.trim();
        if let Some((from, name)) = p.split_once("::") {
            format!("from {} import {}", from, name)
        } else {
            format!("import {}", p)
        }
    }

    fn is_import_line(&self, line: &str) -> bool {
        let t = line.trim();
        t.starts_with("import ") || t.starts_with("from ")
    }

    fn is_known_std(&self, symbol: &str) -> bool {
        let base = symbol.split('.').next().unwrap_or(symbol);
        PY_STDLIB.contains(&base)
    }

    fn verify(&self, project_dir: &Path) -> VerificationResult {
        let mut errors = Vec::new();
        let mut stdout = String::new();
        let mut stderr = String::new();

        if !command_available("python3") {
            return VerificationResult {
                clean: false,
                errors: vec![CompileError {
                    code: "PYTHON_MISSING".to_string(),
                    message: "python3 is not available on PATH — cannot verify code".to_string(),
                    file: String::new(),
                    line: 0,
                    col: 0,
                    suggestion: None,
                    source_line: None,
                    kind: ErrorKind::Other,
                }],
                warnings: Vec::new(),
                stdout,
                stderr: "python3 not found".to_string(),
            };
        }

        // Guardrail 1: compile every module (syntax oracle).
        let files = collect_with(project_dir, &["py"]);
        if files.is_empty() {
            return VerificationResult {
                clean: true,
                errors,
                warnings: Vec::new(),
                stdout,
                stderr,
            };
        }
        for f in &files {
            let out = Command::new("python3")
                .arg("-m")
                .arg("py_compile")
                .arg(f)
                .current_dir(project_dir)
                .output();
            match out {
                Ok(o) => {
                    stdout.push_str(&String::from_utf8_lossy(&o.stdout));
                    let err = String::from_utf8_lossy(&o.stderr).to_string();
                    stderr.push_str(&err);
                    if !o.status.success() {
                        errors.extend(parse_py_compile(&err, f));
                    }
                }
                Err(e) => errors.push(CompileError {
                    code: "PYTHON_ERROR".to_string(),
                    message: format!("Failed to run py_compile: {}", e),
                    file: f.clone(),
                    line: 0,
                    col: 0,
                    suggestion: None,
                    source_line: None,
                    kind: ErrorKind::Other,
                }),
            }
        }
        if !errors.is_empty() {
            return VerificationResult {
                clean: false,
                errors,
                warnings: Vec::new(),
                stdout,
                stderr,
            };
        }

        // Guardrail 2: pytest when tests exist. Missing runner with tests
        // present is reported loudly — never a silent clean.
        let has_tests = files
            .iter()
            .any(|f| f.contains("test_") || f.contains("_test") || f.ends_with("tests.py"));
        if has_tests {
            if !command_available("pytest") && !has_pytest_module() {
                errors.push(CompileError {
                    code: "PYTEST_MISSING".to_string(),
                    message:
                        "Test files exist but pytest is not installed — cannot verify behavior"
                            .to_string(),
                    file: String::new(),
                    line: 0,
                    col: 0,
                    suggestion: None,
                    source_line: None,
                    kind: ErrorKind::Other,
                });
                return VerificationResult {
                    clean: false,
                    errors,
                    warnings: Vec::new(),
                    stdout,
                    stderr,
                };
            }
            let out = Command::new("python3")
                .arg("-m")
                .arg("pytest")
                .arg("-q")
                .current_dir(project_dir)
                .output();
            match out {
                Ok(o) => {
                    stdout.push_str(&String::from_utf8_lossy(&o.stdout));
                    stderr.push_str(&String::from_utf8_lossy(&o.stderr));
                    if !o.status.success() {
                        errors.push(CompileError {
                            code: "TEST_FAILURE".to_string(),
                            message: "pytest reported failures".to_string(),
                            file: String::new(),
                            line: 0,
                            col: 0,
                            suggestion: None,
                            source_line: None,
                            kind: ErrorKind::Other,
                        });
                    }
                }
                Err(e) => errors.push(CompileError {
                    code: "PYTEST_ERROR".to_string(),
                    message: format!("Failed to run pytest: {}", e),
                    file: String::new(),
                    line: 0,
                    col: 0,
                    suggestion: None,
                    source_line: None,
                    kind: ErrorKind::Other,
                }),
            }
        }

        VerificationResult {
            clean: errors.is_empty(),
            errors,
            warnings: Vec::new(),
            stdout,
            stderr,
        }
    }
}

/// Parse `py_compile` stderr into structured errors.
/// ```text
///   File "main.py", line 3
///     def broken(
///                ^
/// SyntaxError: '(' was never closed
/// ```
fn parse_py_compile(stderr: &str, fallback_file: &str) -> Vec<CompileError> {
    let mut errors = Vec::new();
    let mut file = fallback_file.to_string();
    let mut line = 0u32;
    for raw in stderr.lines() {
        let l = raw.trim();
        if let Some(rest) = l.strip_prefix("File \"")
            && let Some(end) = rest.find("\", line ")
        {
            file = rest[..end].to_string();
            line = rest[end + "\", line ".len()..].trim().parse().unwrap_or(0);
        } else if l.starts_with("SyntaxError")
            || l.starts_with("IndentationError")
            || l.starts_with("TabError")
        {
            errors.push(CompileError {
                code: "PY_SYNTAX".to_string(),
                message: l.to_string(),
                file: file.clone(),
                line,
                col: 0,
                suggestion: None,
                source_line: None,
                kind: ErrorKind::Other,
            });
        }
    }
    if errors.is_empty() && !stderr.trim().is_empty() {
        errors.push(CompileError {
            code: "PY_COMPILE".to_string(),
            message: stderr
                .trim()
                .lines()
                .last()
                .unwrap_or("py_compile failed")
                .to_string(),
            file: file.clone(),
            line,
            col: 0,
            suggestion: None,
            source_line: None,
            kind: ErrorKind::Other,
        });
    }
    errors
}

// --- C ---

pub struct CBackend;

/// C stdlib headers the planner may include on its own.
const C_STDLIB: &[&str] = &[
    "stdio.h",
    "stdlib.h",
    "string.h",
    "stdint.h",
    "stdbool.h",
    "stddef.h",
    "math.h",
    "assert.h",
    "ctype.h",
    "limits.h",
    "time.h",
    "errno.h",
    "unistd.h",
    "fcntl.h",
];

impl LanguageBackend for CBackend {
    fn language(&self) -> &str {
        "c"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["c", "h"]
    }

    fn import_line(&self, path: &str) -> String {
        let p = path.trim().trim_start_matches("#include").trim();
        let p = p.trim_matches(|c| c == '<' || c == '>' || c == '"' || c == ' ');
        if C_STDLIB.contains(&p) {
            format!("#include <{}>", p)
        } else {
            format!("#include \"{}\"", p)
        }
    }

    fn is_import_line(&self, line: &str) -> bool {
        line.trim().starts_with("#include")
    }

    fn is_known_std(&self, symbol: &str) -> bool {
        let p = symbol.trim().trim_start_matches("#include").trim();
        let p = p.trim_matches(|c| c == '<' || c == '>' || c == '"' || c == ' ');
        C_STDLIB.contains(&p)
    }

    fn verify(&self, project_dir: &Path) -> VerificationResult {
        let mut errors = Vec::new();
        let mut stdout = String::new();
        let mut stderr = String::new();

        if !command_available("gcc") {
            return VerificationResult {
                clean: false,
                errors: vec![CompileError {
                    code: "GCC_MISSING".to_string(),
                    message: "gcc is not available on PATH — cannot verify code".to_string(),
                    file: String::new(),
                    line: 0,
                    col: 0,
                    suggestion: None,
                    source_line: None,
                    kind: ErrorKind::Other,
                }],
                warnings: Vec::new(),
                stdout,
                stderr: "gcc not found".to_string(),
            };
        }

        let files = collect_with(project_dir, &["c"]);
        for f in &files {
            let out = Command::new("gcc")
                .arg("-fsyntax-only")
                .arg("-Wall")
                .arg(f)
                .current_dir(project_dir)
                .output();
            match out {
                Ok(o) => {
                    stdout.push_str(&String::from_utf8_lossy(&o.stdout));
                    let err = String::from_utf8_lossy(&o.stderr).to_string();
                    stderr.push_str(&err);
                    if !o.status.success() {
                        errors.extend(parse_gcc(&err, f));
                    }
                }
                Err(e) => errors.push(CompileError {
                    code: "GCC_ERROR".to_string(),
                    message: format!("Failed to run gcc: {}", e),
                    file: f.clone(),
                    line: 0,
                    col: 0,
                    suggestion: None,
                    source_line: None,
                    kind: ErrorKind::Other,
                }),
            }
        }

        VerificationResult {
            clean: errors.is_empty(),
            errors,
            warnings: Vec::new(),
            stdout,
            stderr,
        }
    }
}

/// Parse `file:line:col: error: message` from gcc stderr.
fn parse_gcc(stderr: &str, fallback_file: &str) -> Vec<CompileError> {
    let mut errors = Vec::new();
    for raw in stderr.lines() {
        let parts: Vec<&str> = raw.splitn(4, ':').collect();
        if parts.len() == 4 && parts[2].trim().parse::<u32>().is_ok() {
            let kind = parts[1].trim();
            if kind.contains("error") {
                errors.push(CompileError {
                    code: "GCC_ERROR".to_string(),
                    message: parts[3].trim().to_string(),
                    file: parts[0].trim().to_string(),
                    line: parts[1].trim().parse().unwrap_or(0),
                    col: parts[2].trim().parse().unwrap_or(0),
                    suggestion: None,
                    source_line: None,
                    kind: ErrorKind::Other,
                });
            }
        }
    }
    if errors.is_empty() && !stderr.trim().is_empty() {
        errors.push(CompileError {
            code: "GCC_ERROR".to_string(),
            message: stderr
                .trim()
                .lines()
                .last()
                .unwrap_or("gcc failed")
                .to_string(),
            file: fallback_file.to_string(),
            line: 0,
            col: 0,
            suggestion: None,
            source_line: None,
            kind: ErrorKind::Other,
        });
    }
    errors
}

// --- Kotlin ---

pub struct KotlinBackend;

/// JVM / Android / Kotlin-stdlib prefixes the planner may import on its own.
const KOTLIN_STDLIB: &[&str] = &[
    "kotlin.",
    "kotlinx.",
    "java.",
    "javax.",
    "android.",
    "androidx.",
    "org.junit.",
    "org.jetbrains.",
];

impl LanguageBackend for KotlinBackend {
    fn language(&self) -> &str {
        "kotlin"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["kt", "java"]
    }

    fn import_line(&self, path: &str) -> String {
        format!("import {}", path.trim().trim_end_matches(';'))
    }

    fn is_import_line(&self, line: &str) -> bool {
        line.trim().starts_with("import ")
    }

    fn is_known_std(&self, symbol: &str) -> bool {
        KOTLIN_STDLIB.iter().any(|p| symbol.starts_with(p))
    }
    // No local kotlinc here: Gradle-backed Android projects keep verifying
    // through the Android oracle path in `verifier.rs`.
}

// --- HTML ---

/// HTML has no compiler, so the oracle is a strict tag-balance +
/// required-elements check over `html.parser` (Python stdlib).
/// There are no imports in this language: `is_known_std` is always
/// false, so import tasks block honestly.
pub struct HtmlBackend;

impl LanguageBackend for HtmlBackend {
    fn language(&self) -> &str {
        "html"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["html", "htm"]
    }

    fn import_line(&self, _path: &str) -> String {
        String::new()
    }

    fn is_import_line(&self, _line: &str) -> bool {
        false
    }

    fn is_known_std(&self, _symbol: &str) -> bool {
        false
    }

    fn verify(&self, project_dir: &Path) -> VerificationResult {
        if !command_available("python3") {
            return VerificationResult {
                clean: false,
                errors: vec![CompileError {
                    code: "PYTHON_MISSING".to_string(),
                    message: "python3 is not available — cannot verify HTML".to_string(),
                    file: String::new(),
                    line: 0,
                    col: 0,
                    suggestion: None,
                    source_line: None,
                    kind: ErrorKind::Other,
                }],
                warnings: Vec::new(),
                stdout: String::new(),
                stderr: "python3 not found".to_string(),
            };
        }
        // argv: [script, project_dir] — the script walks *.html/*.htm,
        // fails on unbalanced tags and on pages missing html/head/title/body.
        let script = r#"
import os, sys
from html.parser import HTMLParser
VOID = {"area","base","br","col","embed","hr","img","input","link","meta","param","source","track","wbr"}
errors = []
root = sys.argv[1]
seen = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in (".git","target","build","node_modules")]
    for fn in sorted(filenames):
        if not (fn.endswith(".html") or fn.endswith(".htm")):
            continue
        path = os.path.join(dirpath, fn)
        rel = os.path.relpath(path, root)
        seen.append(rel)
        stack = []
        title = []
        in_title = []
        ok = {"html": False, "head": False, "body": False}
        class P(HTMLParser):
            def handle_starttag(self, tag, attrs):
                t = tag.lower()
                if t in ("html","head","body"):
                    ok[t] = True
                if t == "title":
                    in_title.append(True)
                if t not in VOID:
                    stack.append((t, self.getpos()[0]))
            def handle_startendtag(self, tag, attrs):
                t = tag.lower()
                if t in ("html","head","body"):
                    ok[t] = True
            def handle_endtag(self, tag):
                t = tag.lower()
                if t == "title" and in_title:
                    in_title.pop()
                if t in VOID:
                    return
                if not stack:
                    errors.append("%s:%s: stray </%s> with empty stack" % (rel, self.getpos()[0], t))
                    return
                top, ln = stack.pop()
                if top != t:
                    errors.append("%s:%s: </%s> closes <%s> opened at line %s" % (rel, self.getpos()[0], t, top, ln))
            def handle_data(self, data):
                if in_title:
                    title.append(data)
        try:
            with open(path, encoding="utf-8") as f:
                P().feed(f.read())
        except Exception as e:
            errors.append("%s:0: unreadable: %s" % (rel, e))
            continue
        for t, ln in stack:
            errors.append("%s:%s: unclosed <%s>" % (rel, ln, t))
        for req in ("html","head","body"):
            if not ok[req]:
                errors.append("%s:0: missing <%s>" % (rel, req))
        if not "".join(title).strip():
            errors.append("%s:0: empty <title>" % rel)
if not seen:
    print("NO_HTML_FILES")
    sys.exit(2)
for e in errors:
    print(e)
sys.exit(1 if errors else 0)
"#;
        let out = Command::new("python3")
            .arg("-c")
            .arg(script)
            .arg(project_dir)
            .current_dir(project_dir)
            .output();
        match out {
            Ok(o) => {
                let stdout = String::from_utf8_lossy(&o.stdout).to_string();
                let stderr = String::from_utf8_lossy(&o.stderr).to_string();
                // Lines are "rel:line: msg".
                let mut structured = Vec::new();
                for line in stdout.lines().chain(stderr.lines()) {
                    let l = line.trim();
                    if l.is_empty() || l == "NO_HTML_FILES" {
                        continue;
                    }
                    let mut p = l.splitn(3, ':');
                    let (f, ln, m) = (
                        p.next().unwrap_or("").to_string(),
                        p.next().and_then(|x| x.parse().ok()).unwrap_or(0),
                        p.next().unwrap_or("").trim().to_string(),
                    );
                    structured.push(CompileError {
                        code: "HTML_STRUCTURE".to_string(),
                        message: m,
                        file: f,
                        line: ln,
                        col: 0,
                        suggestion: None,
                        source_line: None,
                        kind: ErrorKind::Other,
                    });
                }
                VerificationResult {
                    clean: o.status.success() && structured.is_empty(),
                    errors: structured,
                    warnings: Vec::new(),
                    stdout,
                    stderr,
                }
            }
            Err(e) => VerificationResult {
                clean: false,
                errors: vec![CompileError {
                    code: "HTML_SPAWN_ERROR".to_string(),
                    message: format!("Failed to run HTML check: {}", e),
                    file: String::new(),
                    line: 0,
                    col: 0,
                    suggestion: None,
                    source_line: None,
                    kind: ErrorKind::Other,
                }],
                warnings: Vec::new(),
                stdout: String::new(),
                stderr: e.to_string(),
            },
        }
    }
}

// --- Data-driven languages ---

/// A whole language backend as pure data. This is the adaptability
/// mechanism: anything codable can be taught to the engine with one of
/// these — in the built-in [`registry`] or a project `.grounding.toml`
/// `[language]` table — without touching engine code.
///
/// Example `.grounding.toml`:
/// ```toml
/// [language]
/// name = "zz"
/// extensions = ["zz"]
/// import_template = "need {p};"
/// import_prefixes = ["need "]
/// verify_cmd = ["zzc", "--check"]
/// ```
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LanguageSpec {
    /// Lowercase id, e.g. `"go"`. Doubles as the verify-missing code prefix.
    #[serde(default)]
    pub name: String,
    #[serde(default, deserialize_with = "de_str_list")]
    pub extensions: Vec<String>,
    /// `{p}` is the import path, e.g. `"import {p};"`.
    #[serde(default)]
    pub import_template: String,
    /// Line prefixes that mark existing imports, e.g. `["import "]`.
    #[serde(default, deserialize_with = "de_str_list")]
    pub import_prefixes: Vec<String>,
    /// Prefixes the planner may import unprompted, e.g. `["fmt", "os"]`.
    #[serde(default, deserialize_with = "de_str_list")]
    pub std_prefixes: Vec<String>,
    /// Exact paths the planner may import unprompted.
    #[serde(default, deserialize_with = "de_str_list")]
    pub std_exact: Vec<String>,
    /// Leading lines imports must go after (e.g. Go's `package` clause = 1).
    #[serde(default)]
    pub skip_lines: usize,
    /// Oracle command run in the project dir. Empty = honestly unsupported.
    #[serde(default, deserialize_with = "de_str_list")]
    pub verify_cmd: Vec<String>,
    /// Test command run after a clean verify. Empty = no test stage.
    #[serde(default, deserialize_with = "de_str_list")]
    pub test_cmd: Vec<String>,
}

fn de_str_list<'de, D>(d: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let o: Option<Vec<String>> = Option::deserialize(d)?;
    Ok(o.unwrap_or_default())
}

/// A backend built entirely from a [`LanguageSpec`].
pub struct GenericBackend<'a> {
    pub spec: &'a LanguageSpec,
}

impl LanguageBackend for GenericBackend<'_> {
    fn language(&self) -> &str {
        &self.spec.name
    }

    fn skip_lines(&self) -> usize {
        self.spec.skip_lines
    }

    fn extensions(&self) -> &'static [&'static str] {
        &[]
    }

    fn import_line(&self, path: &str) -> String {
        self.spec.import_template.replace("{p}", path.trim())
    }

    fn is_import_line(&self, line: &str) -> bool {
        let t = line.trim();
        self.spec.import_prefixes.iter().any(|p| t.starts_with(p))
    }

    fn is_known_std(&self, symbol: &str) -> bool {
        self.spec.std_exact.iter().any(|e| e == symbol)
            || self.spec.std_prefixes.iter().any(|p| {
                symbol == p
                    || symbol.starts_with(&format!("{}.", p))
                    || symbol.starts_with(&format!("{}::", p))
                    || symbol.starts_with(&format!("{}/", p))
            })
    }

    fn verify(&self, project_dir: &Path) -> VerificationResult {
        if self.spec.verify_cmd.is_empty() {
            return VerificationResult {
                clean: false,
                errors: vec![CompileError {
                    code: "VERIFICATION_UNSUPPORTED".to_string(),
                    message: format!(
                        "Language {} has no verify command — add one to be checkable",
                        self.spec.name
                    ),
                    file: String::new(),
                    line: 0,
                    col: 0,
                    suggestion: None,
                    source_line: None,
                    kind: ErrorKind::Other,
                }],
                warnings: Vec::new(),
                stdout: String::new(),
                stderr: String::new(),
            };
        }
        let tool = self.spec.verify_cmd[0].clone();
        if !command_available(&tool) {
            return VerificationResult {
                clean: false,
                errors: vec![CompileError {
                    code: format!("{}_MISSING", self.spec.name.to_uppercase()),
                    message: format!(
                        "{} is not available on PATH — cannot verify {} code",
                        tool, self.spec.name
                    ),
                    file: String::new(),
                    line: 0,
                    col: 0,
                    suggestion: None,
                    source_line: None,
                    kind: ErrorKind::Other,
                }],
                warnings: Vec::new(),
                stdout: String::new(),
                stderr: format!("{} not found", tool),
            };
        }
        let mut errors = Vec::new();
        let mut stdout = String::new();
        let mut stderr = String::new();
        let out = Command::new(&self.spec.verify_cmd[0])
            .args(&self.spec.verify_cmd[1..])
            .current_dir(project_dir)
            .output();
        match out {
            Ok(o) => {
                stdout.push_str(&String::from_utf8_lossy(&o.stdout));
                let err = String::from_utf8_lossy(&o.stderr).to_string();
                stderr.push_str(&err);
                if !o.status.success() {
                    errors.extend(parse_generic_errors(&stdout, &err));
                }
            }
            Err(e) => errors.push(CompileError {
                code: "VERIFY_SPAWN_ERROR".to_string(),
                message: format!("Failed to run {}: {}", tool, e),
                file: String::new(),
                line: 0,
                col: 0,
                suggestion: None,
                source_line: None,
                kind: ErrorKind::Other,
            }),
        }
        if !errors.is_empty() {
            return VerificationResult {
                clean: false,
                errors,
                warnings: Vec::new(),
                stdout,
                stderr,
            };
        }
        if !self.spec.test_cmd.is_empty() {
            let out = Command::new(&self.spec.test_cmd[0])
                .args(&self.spec.test_cmd[1..])
                .current_dir(project_dir)
                .output();
            match out {
                Ok(o) => {
                    stdout.push_str(&String::from_utf8_lossy(&o.stdout));
                    stderr.push_str(&String::from_utf8_lossy(&o.stderr));
                    if !o.status.success() {
                        errors.push(CompileError {
                            code: "TEST_FAILURE".to_string(),
                            message: format!("{} reported failures", self.spec.test_cmd[0]),
                            file: String::new(),
                            line: 0,
                            col: 0,
                            suggestion: None,
                            source_line: None,
                            kind: ErrorKind::Other,
                        });
                    }
                }
                Err(e) => errors.push(CompileError {
                    code: "TEST_SPAWN_ERROR".to_string(),
                    message: format!("Failed to run tests: {}", e),
                    file: String::new(),
                    line: 0,
                    col: 0,
                    suggestion: None,
                    source_line: None,
                    kind: ErrorKind::Other,
                }),
            }
        }
        VerificationResult {
            clean: errors.is_empty(),
            errors,
            warnings: Vec::new(),
            stdout,
            stderr,
        }
    }
}

/// Parse compiler-ish output (`path:line:col: msg`, `File "p", line N`,
/// `path:line` + error on the next line) into structured errors.
fn parse_generic_errors(stdout: &str, stderr: &str) -> Vec<CompileError> {
    let mut errors = Vec::new();
    let lines: Vec<&str> = stdout.lines().chain(stderr.lines()).collect();
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i].trim();
        if let Some((file, line, msg)) = parse_path_line(l) {
            errors.push(CompileError {
                code: "VERIFY_ERROR".to_string(),
                message: if msg.is_empty() {
                    next_meaningful(&lines, i + 1)
                } else {
                    msg
                },
                file,
                line,
                col: 0,
                suggestion: None,
                source_line: None,
                kind: ErrorKind::Other,
            });
        } else if let Some(rest) = l.strip_prefix("File \"")
            && let Some(end) = rest.find("\", line ")
        {
            let file = rest[..end].to_string();
            let line = rest[end + "\", line ".len()..].trim().parse().unwrap_or(0);
            errors.push(CompileError {
                code: "VERIFY_ERROR".to_string(),
                message: next_meaningful(&lines, i + 1),
                file,
                line,
                col: 0,
                suggestion: None,
                source_line: None,
                kind: ErrorKind::Other,
            });
        }
        i += 1;
    }
    if errors.is_empty() {
        let blob = format!("{}\n{}", stdout, stderr);
        errors.push(CompileError {
            code: "VERIFY_ERROR".to_string(),
            message: blob
                .trim()
                .lines()
                .last()
                .unwrap_or("verify failed")
                .to_string(),
            file: String::new(),
            line: 0,
            col: 0,
            suggestion: None,
            source_line: None,
            kind: ErrorKind::Other,
        });
    }
    errors
}

fn parse_path_line(l: &str) -> Option<(String, u32, String)> {
    let mut parts = l.splitn(4, ':');
    let file = parts.next()?.trim();
    if file.is_empty() || file.contains(' ') || !file.contains('.') {
        return None;
    }
    let line: u32 = parts.next()?.trim().parse().ok()?;
    let rest = parts.next().unwrap_or("").trim();
    let (col, msg) = match rest.parse::<u32>() {
        Ok(_) => (0, parts.next().unwrap_or("").trim().to_string()),
        Err(_) => (
            0,
            format!("{}:{}", rest, parts.next().unwrap_or(""))
                .trim_matches(':')
                .to_string(),
        ),
    };
    let _ = col;
    Some((file.to_string(), line, msg))
}

fn next_meaningful(lines: &[&str], from: usize) -> String {
    lines
        .iter()
        .skip(from)
        .map(|s| s.trim())
        .find(|s| !s.is_empty() && !s.starts_with('^') && !s.starts_with('~'))
        .unwrap_or("verification failed")
        .to_string()
}

// --- Built-in registry (pure data) ---

/// The built-in registry. `const` can't hold `String`/`Vec`, so specs
/// are materialized once here.
fn registry_specs() -> Vec<LanguageSpec> {
    vec![
        LanguageSpec {
            name: "go".to_string(),
            extensions: vec!["go".to_string()],
            import_template: "import \"{p}\"".to_string(),
            import_prefixes: vec!["import ".to_string()],
            std_prefixes: vec![
                "fmt".to_string(),
                "os".to_string(),
                "strings".to_string(),
                "errors".to_string(),
                "io".to_string(),
                "net".to_string(),
                "encoding".to_string(),
                "sync".to_string(),
                "time".to_string(),
            ],
            std_exact: vec![],
            skip_lines: 1,
            verify_cmd: vec!["go".to_string(), "build".to_string(), "./...".to_string()],
            test_cmd: vec!["go".to_string(), "test".to_string(), "./...".to_string()],
        },
        LanguageSpec {
            name: "node".to_string(),
            extensions: vec!["js".to_string(), "mjs".to_string(), "cjs".to_string()],
            import_template: "const {p} = require('{p}');".to_string(),
            import_prefixes: vec![
                "import ".to_string(),
                "const ".to_string(),
                "require(".to_string(),
            ],
            std_prefixes: vec![],
            std_exact: vec![
                "fs".to_string(),
                "path".to_string(),
                "os".to_string(),
                "util".to_string(),
                "events".to_string(),
                "crypto".to_string(),
                "http".to_string(),
                "url".to_string(),
            ],
            skip_lines: 0,
            verify_cmd: vec![
                "node".to_string(),
                "--check".to_string(),
                "index.js".to_string(),
            ],
            test_cmd: vec![],
        },
    ]
}

/// Leak-once static views of the registry so selection can hand out
/// `&'static` references.
fn registry() -> &'static [LanguageSpec] {
    use std::sync::OnceLock;
    static REG: OnceLock<Vec<LanguageSpec>> = OnceLock::new();
    REG.get_or_init(registry_specs)
}

// --- Selection ---

/// Project-local override: `<project>/.grounding.toml` `[language]` table.
/// This is how ANY language plugs in with zero engine changes.
pub fn load_extra(project_dir: &Path) -> Vec<LanguageSpec> {
    let path = project_dir.join(".grounding.toml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(table) = text.parse::<toml_lang::Table>() else {
        return Vec::new();
    };
    table
        .get("language")
        .and_then(|v| v.clone().try_into().ok())
        .into_iter()
        .collect()
}

/// Minimal TOML subset parser (tables, strings, string arrays) so language
/// config needs no new dependency.
mod toml_lang {
    use std::collections::HashMap;

    #[derive(Debug, Clone)]
    pub enum Val {
        Str(String),
        List(Vec<String>),
        Table(HashMap<String, Val>),
    }

    #[derive(Debug, Clone, Default)]
    pub struct Table(pub HashMap<String, Val>);

    impl Table {
        pub fn get(&self, k: &str) -> Option<&Val> {
            self.0.get(k)
        }
    }

    impl TryFrom<Val> for super::LanguageSpec {
        type Error = ();
        fn try_from(v: Val) -> Result<Self, ()> {
            let Val::Table(m) = v else { return Err(()) };
            let str_field = |k: &str| match m.get(k) {
                Some(Val::Str(s)) => s.clone(),
                _ => String::new(),
            };
            let list_field = |k: &str| match m.get(k) {
                Some(Val::List(l)) => l.clone(),
                Some(Val::Str(s)) => vec![s.clone()],
                _ => Vec::new(),
            };
            Ok(super::LanguageSpec {
                name: str_field("name"),
                extensions: list_field("extensions"),
                import_template: str_field("import_template"),
                import_prefixes: list_field("import_prefixes"),
                std_prefixes: list_field("std_prefixes"),
                std_exact: list_field("std_exact"),
                skip_lines: match m.get("skip_lines") {
                    Some(Val::Str(s)) => s.parse().unwrap_or(0),
                    _ => 0,
                },
                verify_cmd: list_field("verify_cmd"),
                test_cmd: list_field("test_cmd"),
            })
        }
    }

    impl std::str::FromStr for Table {
        type Err = ();
        fn from_str(s: &str) -> Result<Self, ()> {
            let mut root: HashMap<String, Val> = HashMap::new();
            let mut current: Vec<String> = Vec::new();
            for raw in s.lines() {
                let line = raw.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if line.starts_with('[') && line.ends_with(']') {
                    current = line[1..line.len() - 1]
                        .split('.')
                        .map(|x| x.trim().to_string())
                        .collect();
                    continue;
                }
                let Some(eq) = line.find('=') else { continue };
                let key = line[..eq].trim().to_string();
                let val = parse_val(line[eq + 1..].trim());
                insert(&mut root, &current, key, val);
            }
            Ok(Table(root))
        }
    }

    fn insert(root: &mut HashMap<String, Val>, path: &[String], key: String, val: Val) {
        let mut node = root;
        for p in path {
            node = match node
                .entry(p.clone())
                .or_insert_with(|| Val::Table(HashMap::new()))
            {
                Val::Table(m) => m,
                _ => return,
            };
        }
        node.insert(key, val);
    }

    fn parse_val(s: &str) -> Val {
        let s = s.trim();
        if s.starts_with('[') && s.ends_with(']') {
            let inner = &s[1..s.len() - 1];
            Val::List(
                inner
                    .split(',')
                    .map(|x| x.trim().trim_matches('"').trim_matches('\'').to_string())
                    .filter(|x| !x.is_empty())
                    .collect(),
            )
        } else {
            Val::Str(s.trim_matches('"').trim_matches('\'').to_string())
        }
    }
}

/// Backend for a source file. Order: project override → hand-tuned
/// built-ins → registry specs → Rust fallback.
pub fn backend_for_file<'a>(file: &Path, extra: &'a [LanguageSpec]) -> Backend<'a> {
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
    if let Some(spec) = extra.iter().find(|s| s.extensions.iter().any(|e| e == ext)) {
        return Backend::Generic(GenericBackend { spec });
    }
    let builtin: Option<&'static dyn LanguageBackend> = match ext {
        "py" => Some(&PythonBackend),
        "kt" | "java" => Some(&KotlinBackend),
        "c" | "h" => Some(&CBackend),
        "html" | "htm" => Some(&HtmlBackend),
        "rs" => Some(&RustBackend),
        _ => None,
    };
    if let Some(b) = builtin {
        return Backend::Builtin(b);
    }
    if let Some(spec) = registry()
        .iter()
        .find(|s| s.extensions.iter().any(|e| e == ext))
    {
        return Backend::Generic(GenericBackend { spec });
    }
    Backend::Builtin(&RustBackend)
}

/// Backend for a whole project by layout markers (override wins).
pub fn backend_for_project<'a>(project_dir: &Path, extra: &'a [LanguageSpec]) -> Backend<'a> {
    if project_dir.join("Cargo.toml").exists() {
        return Backend::Builtin(&RustBackend);
    }
    if project_dir.join("pyproject.toml").exists()
        || project_dir.join("requirements.txt").exists()
        || project_dir.join("setup.py").exists()
        || has_source_files(project_dir, &["py"])
    {
        return Backend::Builtin(&PythonBackend);
    }
    if project_dir.join("build.gradle").exists()
        || project_dir.join("build.gradle.kts").exists()
        || project_dir.join("settings.gradle").exists()
        || has_source_files(project_dir, &["kt", "java"])
    {
        return Backend::Builtin(&KotlinBackend);
    }
    if project_dir.join("go.mod").exists() || has_source_files(project_dir, &["go"]) {
        return generic_by_name(extra, "go");
    }
    if project_dir.join("package.json").exists()
        || has_source_files(project_dir, &["js", "mjs", "cjs"])
    {
        return generic_by_name(extra, "node");
    }
    if project_dir.join("Makefile").exists()
        || project_dir.join("CMakeLists.txt").exists()
        || has_source_files(project_dir, &["c"])
    {
        return Backend::Builtin(&CBackend);
    }
    if has_source_files(project_dir, &["html", "htm"]) {
        return Backend::Builtin(&HtmlBackend);
    }
    // Last resort: first source file's backend wins over the Rust default.
    if let Some(ext) = first_source_extension(project_dir, extra) {
        let probe = Path::new("probe").with_extension(ext);
        return backend_for_file(&probe, extra);
    }
    Backend::Builtin(&RustBackend)
}

fn generic_by_name<'a>(extra: &'a [LanguageSpec], name: &str) -> Backend<'a> {
    if let Some(spec) = extra.iter().find(|s| s.name == name) {
        Backend::Generic(GenericBackend { spec })
    } else if let Some(spec) = registry().iter().find(|s| s.name == name) {
        Backend::Generic(GenericBackend { spec })
    } else {
        Backend::Builtin(&RustBackend)
    }
}

fn first_source_extension(project_dir: &Path, extra: &[LanguageSpec]) -> Option<String> {
    let mut exts: Vec<String> = KNOWN_EXTENSIONS.iter().map(|s| s.to_string()).collect();
    exts.extend(extra.iter().flat_map(|s| s.extensions.clone()));
    exts.extend(registry().iter().flat_map(|s| s.extensions.clone()));
    exts.sort();
    exts.dedup();
    let found = collect_with(
        project_dir,
        &exts.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    );
    found.into_iter().next().and_then(|f| {
        Path::new(&f)
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_string())
    })
}

/// Every extension the engine indexes for symbols.
pub const KNOWN_EXTENSIONS: &[&str] = &[
    "rs", "kt", "java", "py", "c", "h", "go", "js", "mjs", "cjs", "ts", "html", "htm",
];

/// Unified handle for dispatch without lifetime gymnastics at call sites.
pub enum Backend<'a> {
    Builtin(&'static dyn LanguageBackend),
    Generic(GenericBackend<'a>),
}

impl LanguageBackend for Backend<'_> {
    fn language(&self) -> &str {
        match self {
            Backend::Builtin(b) => b.language(),
            Backend::Generic(g) => g.language(),
        }
    }

    fn extensions(&self) -> &'static [&'static str] {
        match self {
            Backend::Builtin(b) => b.extensions(),
            Backend::Generic(_) => &[],
        }
    }

    fn import_line(&self, path: &str) -> String {
        match self {
            Backend::Builtin(b) => b.import_line(path),
            Backend::Generic(g) => g.import_line(path),
        }
    }

    fn is_import_line(&self, line: &str) -> bool {
        match self {
            Backend::Builtin(b) => b.is_import_line(line),
            Backend::Generic(g) => g.is_import_line(line),
        }
    }

    fn is_known_std(&self, symbol: &str) -> bool {
        match self {
            Backend::Builtin(b) => b.is_known_std(symbol),
            Backend::Generic(g) => g.is_known_std(symbol),
        }
    }

    fn verify(&self, project_dir: &Path) -> VerificationResult {
        match self {
            Backend::Builtin(b) => b.verify(project_dir),
            Backend::Generic(g) => g.verify(project_dir),
        }
    }

    fn skip_lines(&self) -> usize {
        match self {
            Backend::Builtin(b) => b.skip_lines(),
            Backend::Generic(g) => g.skip_lines(),
        }
    }
}

fn has_source_files(dir: &Path, extensions: &[&str]) -> bool {
    collect_with(dir, extensions).into_iter().next().is_some()
}

fn collect_with(dir: &Path, extensions: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    let ignore = [
        ".git",
        "target",
        "build",
        ".gradle",
        ".idea",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
    ];
    while let Some(current) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path
                    .file_name()
                    .is_none_or(|n| ignore.contains(&n.to_string_lossy().as_ref()))
                {
                    continue;
                }
                stack.push(path);
            } else if let Some(ext) = path.extension().and_then(|e| e.to_str())
                && extensions.contains(&ext)
            {
                out.push(path.to_string_lossy().to_string());
            }
        }
    }
    out.sort();
    out
}

fn command_available(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn has_pytest_module() -> bool {
    Command::new("python3")
        .arg("-c")
        .arg("import pytest")
        .output()
        .is_ok_and(|o| o.status.success())
}
