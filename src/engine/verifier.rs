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
        CodeVerifier { project_dir, project_type }
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
        let clippy = Command::new("cargo")
            .arg("clippy")
            .arg("--all-targets")
            .arg("--")
            .arg("-D")
            .arg("warnings")
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
        // For Android Kotlin/Java projects, use gradle
        let result = Command::new("./gradlew")
            .arg("build")
            .arg("-x")
            .arg("test") // skip unit tests for MVP, focus on compilation
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
    } else if dir.join("gradlew").exists() || dir.join("build.gradle").exists() {
        ProjectType::Android
    } else {
        // Default to Rust for the MVP
        ProjectType::Rust
    }
}

fn split_errors_warnings(errors: Vec<CompileError>) -> (Vec<CompileError>, Vec<CompileError>) {
    errors.into_iter()
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
