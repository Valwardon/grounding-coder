use std::path::{Path, PathBuf};
use std::process::Command;

use super::error::{CompileError, ErrorClassifier};

/// Verification result — grounded's `VerificationEvent` repurposed.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the code passed all verification checks.
    pub clean: bool,
    /// Compilation errors (empty if clean)
    pub errors: Vec<CompileError>,
    /// Warnings
    pub warnings: Vec<CompileError>,
    /// Raw stdout from the verification commands
    pub stdout: String,
    /// Raw stderr from the verification commands
    pub stderr: String,
}

impl VerificationResult {
    pub fn is_clean(&self) -> bool {
        self.clean && self.errors.is_empty()
    }
}

/// The deterministic code verifier — grounded's `VerificationLoop` repurposed.
///
/// In grounded, the VerificationLoop runs after each 16ms tick and checks
/// three things: energy conservation, path integrity, and structural faults.
/// Violations spike novelty + drop valence + mark edges for pruning.
///
/// Here, the verifier runs after each code generation attempt and checks
/// three equivalent guardrails:
///   1. **Syntax** — does the code parse? (cargo check)
///   2. **Types** — does it type-check? (cargo check / clippy)
///   3. **Behavior** — do tests pass? (cargo test)
///
/// The compiler (rustc/cargo) is the oracle. It never lies.
/// If verification fails, the errors are deterministic facts
/// that drive the CorrectionPipeline.
pub struct CodeVerifier {
    project_dir: PathBuf,
    /// Whether this is a Rust project (uses cargo) or Android (uses gradle)
    project_type: ProjectType,
}

#[derive(Debug, Clone, Copy)]
enum ProjectType {
    Rust,
    Android,
}

impl CodeVerifier {
    pub fn new(project_dir: PathBuf) -> Self {
        let project_type = detect_project_type(&project_dir);
        CodeVerifier {
            project_dir,
            project_type,
        }
    }

    /// Run all verification checks. This is the "oracle" call.
    ///
    /// The bot does NOT trust the LLM's or its own code generator.
    /// It trusts ONLY this verification.
    pub async fn verify(&self) -> VerificationResult {
        match self.project_type {
            ProjectType::Rust => self.verify_rust(),
            ProjectType::Android => self.verify_android(),
        }
    }

    /// Verify a Rust project: cargo check → clippy → test
    fn verify_rust(&self) -> VerificationResult {
        let mut all_errors = Vec::new();
        let mut all_warnings = Vec::new();
        let mut combined_stdout = String::new();
        let mut combined_stderr = String::new();

        // ── Guardrail 0: toolchain availability ──
        // If cargo is missing, that is a fact we must report — never a
        // silent "clean" (a false pass would let unverified code through).
        if !command_available("cargo") {
            return VerificationResult {
                clean: false,
                errors: vec![CompileError {
                    code: "CARGO_MISSING".to_string(),
                    message: "cargo is not available on PATH — cannot verify code".to_string(),
                    file: String::new(),
                    line: 0,
                    col: 0,
                    suggestion: None,
                    source_line: None,
                    kind: super::error::ErrorKind::Other,
                }],
                warnings: Vec::new(),
                stdout: String::new(),
                stderr: "cargo not found".to_string(),
            };
        }

        // ── Guardrail 1: Syntax + Types (cargo check) ──
        let check = Command::new("cargo")
            .arg("check")
            .arg("--all-targets")
            .current_dir(&self.project_dir)
            .output();

        if let Ok(output) = check {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            combined_stdout.push_str(&stdout);
            combined_stderr.push_str(&stderr);

            let errors = ErrorClassifier::parse(&stderr);
            let (errs, warns) = split_errors_warnings(errors);
            all_errors.extend(errs);
            all_warnings.extend(warns);
        }

        // If there are compilation errors, don't bother running tests
        // (grounded's "don't pile on errors" principle)
        if !all_errors.is_empty() {
            return VerificationResult {
                clean: false,
                errors: all_errors,
                warnings: all_warnings,
                stdout: combined_stdout,
                stderr: combined_stderr,
            };
        }

        // ── Guardrail 2: Linting (cargo clippy) ──
        // Clippy is optional: if it isn't installed we skip it rather than
        // failing the whole verification on a missing tool. Warnings are
        // reported but never promoted to hard errors (`-D warnings` would
        // gate the agent on its own stylistic output).
        if command_available("cargo-clippy") || command_available("clippy-driver") {
            let clippy = Command::new("cargo")
                .arg("clippy")
                .arg("--all-targets")
                .current_dir(&self.project_dir)
                .output();

            if let Ok(output) = clippy {
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                combined_stdout.push_str(&String::from_utf8_lossy(&output.stdout));
                combined_stderr.push_str(&stderr);

                let errors = ErrorClassifier::parse(&stderr);
                let (errs, warns) = split_errors_warnings(errors);
                all_errors.extend(errs);
                all_warnings.extend(warns);
            }
        } else {
            log::info!("cargo-clippy not installed — skipping lint guardrail.");
        }

        // If clippy found issues, don't run tests yet
        if !all_errors.is_empty() {
            return VerificationResult {
                clean: false,
                errors: all_errors,
                warnings: all_warnings,
                stdout: combined_stdout,
                stderr: combined_stderr,
            };
        }

        // ── Guardrail 3: Behavior (cargo test) ──
        let test = Command::new("cargo")
            .arg("test")
            .arg("--quiet")
            .current_dir(&self.project_dir)
            .output();

        if let Ok(output) = test {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            combined_stdout.push_str(&stdout);
            combined_stderr.push_str(&stderr);

            if !output.status.success() {
                // Test failures — parse test output
                let test_errors = parse_test_failures(&stdout, &stderr);
                all_errors.extend(test_errors);
            }
        }

        VerificationResult {
            clean: all_errors.is_empty(),
            errors: all_errors,
            warnings: all_warnings,
            stdout: combined_stdout,
            stderr: combined_stderr,
        }
    }

    /// Verify an Android project: gradle build + lint
    fn verify_android(&self) -> VerificationResult {
        // Prefer the project's gradle wrapper, fall back to a system `gradle`.
        let (cmd, args): (&str, Vec<&str>) = if self.project_dir.join("gradlew").exists() {
            ("./gradlew", vec!["build", "-x", "test"])
        } else {
            ("gradle", vec!["build", "-x", "test"])
        };
        let result = Command::new(cmd)
            .args(&args)
            .current_dir(&self.project_dir)
            .output();

        match result {
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();

                let errors = ErrorClassifier::parse(&stderr);
                let (errs, warns) = split_errors_warnings(errors);

                VerificationResult {
                    clean: errs.is_empty() && output.status.success(),
                    errors: errs,
                    warnings: warns,
                    stdout,
                    stderr,
                }
            }
            Err(e) => VerificationResult {
                clean: false,
                errors: vec![CompileError {
                    code: "GRADLE_ERROR".to_string(),
                    message: format!("Failed to run gradlew: {}", e),
                    file: String::new(),
                    line: 0,
                    col: 0,
                    suggestion: None,
                    source_line: None,
                    kind: super::error::ErrorKind::Other,
                }],
                warnings: vec![],
                stdout: String::new(),
                stderr: e.to_string(),
            },
        }
    }

    /// Re-run just the compilation check (fast path for retries).
    pub fn verify_changed(&self, _changed_files: &[String]) -> VerificationResult {
        // For now, just re-run the full check. A future optimization
        // could use `cargo check -p <changed package>` or incremental builds.
        self.verify_blocking()
    }

    fn verify_blocking(&self) -> VerificationResult {
        // Blocking version for synchronous callers
        match self.project_type {
            ProjectType::Rust => self.verify_rust(),
            ProjectType::Android => self.verify_android(),
        }
    }
}

fn detect_project_type(dir: &Path) -> ProjectType {
    if dir.join("Cargo.toml").exists() {
        ProjectType::Rust
    } else if dir.join("gradlew").exists()
        || dir.join("build.gradle").exists()
        || dir.join("settings.gradle").exists()
        || has_source_files(dir, &["kt", "java"])
    {
        ProjectType::Android
    } else {
        // Default to Rust for the MVP
        ProjectType::Rust
    }
}

/// Whether a binary is present on PATH (used to skip optional guardrails
/// without silently failing — a missing *required* tool is reported loudly).
fn command_available(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn has_source_files(dir: &Path, extensions: &[&str]) -> bool {
    let mut stack = vec![dir.to_path_buf()];
    let ignore = [
        ".git",
        "target",
        "build",
        ".gradle",
        ".idea",
        "node_modules",
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
            } else if let Some(ext) = path.extension().map(|e| e.to_string_lossy().to_string())
                && extensions.contains(&ext.as_str())
            {
                return true;
            }
        }
    }
    false
}

fn split_errors_warnings(errors: Vec<CompileError>) -> (Vec<CompileError>, Vec<CompileError>) {
    errors
        .into_iter()
        .partition(|e| !e.code.starts_with("warning"))
}

fn parse_test_failures(stdout: &str, stderr: &str) -> Vec<CompileError> {
    let mut failures = Vec::new();
    for line in stdout.lines().chain(stderr.lines()) {
        if line.contains("FAILED") || line.contains("test result: FAILED") {
            failures.push(CompileError {
                code: "TEST_FAILURE".to_string(),
                message: line.trim().to_string(),
                file: String::new(),
                line: 0,
                col: 0,
                suggestion: None,
                source_line: None,
                kind: super::error::ErrorKind::Other,
            });
        }
    }
    failures
}
