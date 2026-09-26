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
///
/// Internal trait: `async fn` in trait position is fine here since this
/// never crosses a public dyn boundary.
#[allow(async_fn_in_trait)]
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
    /// Build the project into a real artifact (binary, jar, apk…).
    /// `target` is a small closed vocabulary per backend (`""`/`native`,
    /// `release`, `android-apk`, …); anything else refuses honestly.
    /// The default refuses — a backend without a build oracle must not
    /// pretend. Artifacts are verified present + non-empty by the caller.
    async fn build(&self, _project_dir: &Path, _target: &str) -> Result<BuildOutcome, String> {
        Err(format!(
            "No build oracle for language {} — BLOCKED",
            self.language()
        ))
    }
    /// Run the language oracle over the project. The default refuses
    /// honestly — a backend without an oracle must not report clean.
    async fn verify(&self, _project_dir: &Path) -> VerificationResult {
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
            full: true,
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

    async fn build(&self, project_dir: &Path, target: &str) -> Result<BuildOutcome, String> {
        RustBackend::build_rust(project_dir, target).await
    }
}

impl RustBackend {
    async fn build_rust(project_dir: &Path, target: &str) -> Result<BuildOutcome, String> {
        let t = target.trim().to_lowercase();
        if t == "android-apk" || t == "apk" || t == "android" {
            return Self::build_android_apk(project_dir).await;
        }
        if !command_available("cargo") {
            return Err("cargo is not available on PATH — cannot build Rust — BLOCKED".to_string());
        }
        if !project_dir.join("Cargo.toml").exists() {
            return Err("No Cargo.toml in project — nothing to build — BLOCKED".to_string());
        }
        // Closed vocabulary: native/debug, release, or an explicit
        // `--target` triple. Anything else (flags, paths) refuses.
        let (release, triple): (bool, Option<String>) = match t.as_str() {
            "" | "native" | "debug" | "bin" => (false, None),
            "release" => (true, None),
            _ if t.contains([' ', '/', '\\', ';', '&', '|', '$', '`']) || t.starts_with('-') => {
                return Err(format!(
                    "Refusing target '{}' (not a plain profile or triple) — BLOCKED",
                    target
                ));
            }
            _ if t.contains('-') => (false, Some(t.clone())),
            _ => {
                return Err(format!(
                    "Rust backend has no target '{}' (native, release, <triple>, android-apk) — BLOCKED",
                    target
                ));
            }
        };
        let profile = if release { "release" } else { "debug" };
        let mut cmd = Command::new("cargo");
        cmd.arg("build");
        if release {
            cmd.arg("--release");
        }
        if let Some(tri) = &triple {
            cmd.arg("--target").arg(tri);
        }
        let o = cmd
            .current_dir(project_dir)
            .output()
            .map_err(|e| format!("Failed to run cargo: {}", e))?;
        let stdout = String::from_utf8_lossy(&o.stdout).to_string();
        let stderr = String::from_utf8_lossy(&o.stderr).to_string();
        if !o.status.success() {
            return Err(format!(
                "cargo build failed (exit {}): {}",
                o.status.code().unwrap_or(-1),
                tail2k(&stderr)
            ));
        }
        let name = cargo_package_name(project_dir).ok_or_else(|| {
            "cargo build succeeded but Cargo.toml has no parseable package name — BLOCKED"
                .to_string()
        })?;
        let mut artifact = project_dir.join("target");
        if let Some(tri) = triple {
            artifact = artifact.join(tri);
        }
        artifact = artifact.join(profile).join(&name);
        #[cfg(windows)]
        artifact.set_extension("exe");
        check_artifact(&artifact)
            .map(|_| BuildOutcome {
                artifact,
                stdout_tail: tail2k(&stdout),
                stderr_tail: tail2k(&stderr),
            })
            .map_err(|e| format!("cargo reported success but {}", e))
    }

    /// Android APK for Dioxus projects: evidence is `Dioxus.toml`
    /// carrying `[android]`; the driver is provisioned `dx`; the proof
    /// is a real `.apk` found in the tree (never a claimed path).
    async fn build_android_apk(project_dir: &Path) -> Result<BuildOutcome, String> {
        let toml = std::fs::read_to_string(project_dir.join("Dioxus.toml"))
            .map_err(|_| "No Dioxus.toml — not a Dioxus android project — BLOCKED".to_string())?;
        if !toml.contains("[android]") {
            return Err("Dioxus.toml has no [android] section — BLOCKED".to_string());
        }
        if std::env::var("ANDROID_HOME").is_err() {
            return Err(
                "ANDROID_HOME is not set — the operator must point at an SDK — BLOCKED".to_string(),
            );
        }
        let dx = ensure_dx().await?;
        let run_dx = |rustflags: Option<String>| {
            let mut cmd = Command::new(&dx);
            cmd.arg("build")
                .arg("--platform")
                .arg("android")
                .current_dir(project_dir);
            if let Some(flags) = rustflags {
                cmd.env("RUSTFLAGS", flags);
            }
            cmd.output()
        };
        let o = run_dx(None).map_err(|e| format!("Failed to run dx: {}", e))?;
        let mut ok = o.status.success();
        let mut stdout = String::from_utf8_lossy(&o.stdout).to_string();
        let mut stderr = String::from_utf8_lossy(&o.stderr).to_string();
        // Self-solving link step: NDK link lines put `-lunwind` before
        // the std archives, so `--as-needed` drops it and the link dies
        // on `_Unwind_*`. One bounded retry with the archive appended
        // last (pending symbols resolve left-to-right). Only on this
        // exact signature; anything else fails honestly below.
        // dx relays cargo's output on either stream; match on both.
        let combined = format!("{}\n{}", stdout, stderr);
        if !o.status.success() && needs_unwind_retry(&combined) {
            let mut flags = std::env::var("RUSTFLAGS").unwrap_or_default();
            if !flags.is_empty() {
                flags.push(' ');
            }
            flags.push_str("-C link-arg=-lunwind");
            let o2 = run_dx(Some(flags)).map_err(|e| format!("Failed to re-run dx: {}", e))?;
            stdout = String::from_utf8_lossy(&o2.stdout).to_string();
            stderr = String::from_utf8_lossy(&o2.stderr).to_string();
            ok = o2.status.success();
            if !ok {
                return Err(format!(
                    "dx build failed even with unwind retry (exit {}): {}",
                    o2.status.code().unwrap_or(-1),
                    tail2k(&stderr)
                ));
            }
        }
        if !ok {
            return Err(format!("dx build failed: {}", tail2k(&stderr)));
        }
        let mut apks = crate::github::find_apks(project_dir);
        apks.sort_by_key(|(_, p)| {
            std::fs::metadata(p)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });
        let (_, apk) = apks.into_iter().next_back().ok_or_else(|| {
            "dx reported success but no .apk exists in the tree — BLOCKED".to_string()
        })?;
        check_artifact(&apk)
            .map(|_| BuildOutcome {
                artifact: apk,
                stdout_tail: tail2k(&stdout),
                stderr_tail: tail2k(&stderr),
            })
            .map_err(|e| format!("dx reported success but {}", e))
    }
}

/// `name = "…"` under `[package]`. Naive line scan, sound for this
/// purpose: unparseable manifests block instead of guessing.
fn cargo_package_name(project_dir: &Path) -> Option<String> {
    let content = std::fs::read_to_string(project_dir.join("Cargo.toml")).ok()?;
    let mut in_package = false;
    for line in content.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_package = t == "[package]";
            continue;
        }
        if in_package && let Some(rest) = t.strip_prefix("name") {
            let rest = rest.trim_start().strip_prefix('=')?.trim();
            let name = rest.trim_matches('"').trim_matches('\'').trim();
            if !name.is_empty() && name != "..." {
                return Some(name.to_string());
            }
        }
    }
    None
}

/// Does this driver binary actually execute here? Runs `<bin> --version`
/// and reports success plus a diagnosis. A downloaded binary that cannot
/// start (stale glibc, wrong arch, dead link) is a fact to route on —
/// never a binary the builder trusts blindly.
fn driver_runs(bin: &std::path::Path) -> Result<(), String> {
    let o = Command::new(bin)
        .arg("--version")
        .output()
        .map_err(|e| format!("cannot execute {}: {}", bin.display(), e))?;
    if o.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
    let stdout = String::from_utf8_lossy(&o.stdout).to_string();
    let mut why = format!("exit {}", o.status.code().unwrap_or(-1));
    for line in stderr.lines().chain(stdout.lines()) {
        if line.contains("GLIBC") {
            why = format!("host libc too old ({})", line.trim());
            break;
        }
        if line.contains("No such file") || line.contains("not found") {
            why = format!("missing loader/interpreter ({})", line.trim());
            break;
        }
    }
    Err(format!("{} runs but fails: {}", bin.display(), why))
}

/// Provisioned `dx` 0.7.2 driver with a self-solving chain. Every
/// candidate is verified by execution (`--version` runs); every failure
/// is recorded in the trail. Order:
///
/// 1. `DX_BIN` (operator-owned).
/// 2. Checksum-pinned release binary for this arch.
/// 3. Source build from the `v0.7.2` tag (clone → dependency refresh on
///    oracle-reported failures → build → verify runs). This is how the
///    bot solves a stale-glibc host on its own instead of stopping.
/// 4. BLOCKED with the full trail — a builder that cannot prove its
///    driver must not run.
async fn ensure_dx() -> Result<std::path::PathBuf, String> {
    let mut trail: Vec<String> = Vec::new();
    if let Ok(bin) = std::env::var("DX_BIN")
        && !bin.trim().is_empty()
    {
        let p = std::path::PathBuf::from(&bin);
        if !p.is_file() {
            return Err(format!(
                "DX_BIN points at {} which is not a file — BLOCKED",
                bin
            ));
        }
        match driver_runs(&p) {
            Ok(()) => return Ok(p),
            Err(e) => {
                return Err(format!("DX_BIN driver broken: {} — BLOCKED", e));
            }
        }
    }
    let arch = std::env::consts::ARCH;
    let spec = match arch {
        "aarch64" => Some(crate::tool::DX_TOOL_AARCH64),
        "x86_64" => Some(crate::tool::DX_TOOL_X86_64),
        _ => None,
    };
    if let Some(spec) = spec {
        match ensure_dx_pinned(&spec).await {
            Ok(bin) => match driver_runs(&bin) {
                Ok(()) => return Ok(bin),
                Err(e) => trail.push(format!("pinned binary unusable: {}", e)),
            },
            Err(e) => trail.push(format!("pinned provision failed: {}", e)),
        }
    } else {
        trail.push(format!("no pinned dx driver for host arch {}", arch));
    }
    // A cached binary from an earlier run may already work (e.g. placed
    // by hand or built before) — check before the expensive path.
    let cached = crate::tool::tools_dir().join("dx").join("dx");
    if cached.is_file() {
        match driver_runs(&cached) {
            Ok(()) => return Ok(cached),
            Err(e) => trail.push(format!("cached dx unusable: {}", e)),
        }
    }
    match ensure_dx_from_source().await {
        Ok(bin) => Ok(bin),
        Err(e) => {
            trail.push(format!("source build failed: {}", e));
            Err(format!(
                "No working dx driver — BLOCKED\ntrail:\n- {}",
                trail.join("\n- ")
            ))
        }
    }
}

/// Download the pinned tarball, extract the `dx` binary, make it
/// executable. Returns the binary path (execution verified by caller).
async fn ensure_dx_pinned(spec: &crate::tool::ToolSpec) -> Result<std::path::PathBuf, String> {
    let tarball = spec.files.first().map(|f| f.name).unwrap_or("dx.tar.gz");
    let dir = crate::tool::ensure_tool(spec).await?;
    let bin = dir.join("dx");
    if bin.is_file() {
        return Ok(bin);
    }
    if !command_available("tar") {
        return Err("tar is not available — cannot extract dx — BLOCKED".to_string());
    }
    let tgz = dir.join(tarball);
    let o = Command::new("tar")
        .arg("xzf")
        .arg(&tgz)
        .arg("-C")
        .arg(&dir)
        .output()
        .map_err(|e| format!("Failed to run tar: {}", e))?;
    if !o.status.success() {
        return Err(format!(
            "Extracting {} failed: {} — BLOCKED",
            tarball,
            tail2k(String::from_utf8_lossy(&o.stderr).as_ref())
        ));
    }
    // Tarball may nest one level; accept `dx` at top or one dir down.
    let nested = std::fs::read_dir(&dir).ok().and_then(|entries| {
        entries.flatten().find_map(|e| {
            let p = e.path();
            (p.is_dir() && p.join("dx").is_file()).then(|| p.join("dx"))
        })
    });
    let found = if bin.is_file() {
        Some(bin.clone())
    } else {
        nested
    };
    let found = found.ok_or_else(|| {
        format!(
            "Pinned {} extracted but contains no dx binary — BLOCKED",
            tarball
        )
    })?;
    if found != bin {
        std::fs::rename(&found, &bin).map_err(|e| format!("Cannot place dx binary: {}", e))?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&bin)
            .map_err(|e| format!("Cannot stat dx binary: {}", e))?
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&bin, perms)
            .map_err(|e| format!("Cannot chmod dx binary: {}", e))?;
    }
    Ok(bin)
}

/// Source fallback: shallow-clone the `v0.7.2` tag, discover the `dx`
/// package via `cargo metadata`, then build with a bounded self-repair
/// loop — each `could not compile \`X\`` from the oracle triggers one
/// `cargo update -p X` before the next round (max 3 refreshes). The
/// resulting binary links the host libc by construction, which is
/// exactly the stale-glibc escape hatch. Result is cached under the
/// tools dir; verified by execution before return.
async fn ensure_dx_from_source() -> Result<std::path::PathBuf, String> {
    if !command_available("git") {
        return Err("git missing — cannot fetch dx sources — BLOCKED".to_string());
    }
    if !command_available("cargo") {
        return Err("cargo missing — cannot build dx — BLOCKED".to_string());
    }
    let base = crate::tool::tools_dir().join("dx-src");
    let out_bin = crate::tool::tools_dir().join("dx").join("dx");
    if out_bin.is_file()
        && let Ok(()) = driver_runs(&out_bin)
    {
        return Ok(out_bin);
    }
    std::fs::create_dir_all(&base)
        .map_err(|e| format!("Cannot create {}: {}", base.display(), e))?;
    let repo = base.join("dioxus");
    if !repo.join("Cargo.toml").is_file() {
        let o = Command::new("git")
            .arg("clone")
            .arg("--depth")
            .arg("1")
            .arg("--branch")
            .arg("v0.7.2")
            .arg("https://github.com/DioxusLabs/dioxus.git")
            .arg("dioxus")
            .current_dir(&base)
            .output()
            .map_err(|e| format!("Failed to run git: {}", e))?;
        if !o.status.success() {
            return Err(format!(
                "Cloning dioxus v0.7.2 failed: {} — BLOCKED",
                tail2k(String::from_utf8_lossy(&o.stderr).as_ref())
            ));
        }
    }
    // Discover the driver package: exact `dx` wins, else `dioxus-cli`.
    let meta_out = Command::new("cargo")
        .arg("metadata")
        .arg("--no-deps")
        .arg("--format-version")
        .arg("1")
        .current_dir(&repo)
        .output()
        .map_err(|e| format!("Failed to run cargo metadata: {}", e))?;
    if !meta_out.status.success() {
        return Err(format!(
            "cargo metadata failed: {} — BLOCKED",
            tail2k(String::from_utf8_lossy(&meta_out.stderr).as_ref())
        ));
    }
    let meta: serde_json::Value = serde_json::from_slice(&meta_out.stdout)
        .map_err(|e| format!("Unparseable cargo metadata: {} — BLOCKED", e))?;
    let names: Vec<String> = meta
        .get("packages")
        .and_then(|p| p.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|p| {
                    p.get("name")
                        .and_then(|n| n.as_str())
                        .map(|s| s.to_string())
                })
                .collect()
        })
        .unwrap_or_default();
    let pkg = names
        .iter()
        .find(|n| n.as_str() == "dx")
        .or_else(|| names.iter().find(|n| n.as_str() == "dioxus-cli"))
        .cloned()
        .ok_or_else(|| "No dx/dioxus-cli package in v0.7.2 sources — BLOCKED".to_string())?;
    // Build with bounded self-repair: the oracle names the failing crate,
    // one `cargo update -p` per round refreshes it, max 3 refresh rounds.
    let mut last_err = String::new();
    for round in 0..=3 {
        let o = Command::new("cargo")
            .arg("build")
            .arg("--release")
            .arg("-p")
            .arg(&pkg)
            .current_dir(&repo)
            .output()
            .map_err(|e| format!("Failed to run cargo build: {}", e))?;
        let stderr = String::from_utf8_lossy(&o.stderr).to_string();
        if o.status.success() {
            last_err.clear();
            break;
        }
        last_err = tail2k(&stderr);
        if round == 3 {
            break;
        }
        let crates = failing_crates(&stderr);
        if crates.is_empty() {
            break;
        }
        let mut upd = Command::new("cargo");
        upd.arg("update");
        for c in crates.iter().take(3) {
            upd.arg("-p").arg(c);
        }
        let uo = upd
            .current_dir(&repo)
            .output()
            .map_err(|e| format!("Failed to run cargo update: {}", e))?;
        if !uo.status.success() {
            break;
        }
    }
    if !last_err.is_empty() {
        return Err(format!("dx source build failed: {} — BLOCKED", last_err));
    }
    let built = repo.join("target").join("release").join("dx");
    if !built.is_file() {
        return Err("cargo build succeeded but target/release/dx is missing — BLOCKED".to_string());
    }
    if let Some(parent) = out_bin.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Cannot create tools dir: {}", e))?;
    }
    std::fs::copy(&built, &out_bin).map_err(|e| format!("Cannot cache dx binary: {}", e))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&out_bin)
            .map_err(|e| format!("Cannot stat dx binary: {}", e))?
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&out_bin, perms)
            .map_err(|e| format!("Cannot chmod dx binary: {}", e))?;
    }
    match driver_runs(&out_bin) {
        Ok(()) => Ok(out_bin),
        Err(e) => Err(format!("Source-built dx still fails: {} — BLOCKED", e)),
    }
}

/// Crate names from ``error: could not compile `X` `` lines. The oracle
/// tells us exactly what broke; the repair loop refreshes exactly that.
fn failing_crates(stderr: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in stderr.lines() {
        if let Some(rest) = line.strip_prefix("error: could not compile `")
            && let Some((name, _)) = rest.split_once('`')
            && !name.is_empty()
            && !out.contains(&name.to_string())
        {
            out.push(name.to_string());
        }
    }
    out
}

/// True when a failed Android link shows the stale-unwinder signature:
/// `undefined symbol: _Unwind_*` with the archive ordered before the
/// objects that need it. Pure predicate, unit-tested.
fn needs_unwind_retry(stderr: &str) -> bool {
    stderr.contains("undefined symbol") && stderr.contains("_Unwind_")
}

/// A built artifact: path on disk plus truncated oracle output.
// Evidence is the artifact's own bytes (caller checks presence + size),
// never a claim.
pub struct BuildOutcome {
    pub artifact: std::path::PathBuf,
    pub stdout_tail: String,
    pub stderr_tail: String,
}

/// Last 2 KiB on a char boundary — enough for the failing command line,
// never a dump.
fn tail2k(s: &str) -> String {
    const MAX: usize = 2048;
    if s.len() <= MAX {
        return s.to_string();
    }
    let mut i = s.len() - MAX;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    s[i..].to_string()
}

/// Binary name from the project dir; anything weird becomes `app`.
fn safe_bin_name(project_dir: &Path) -> String {
    let raw = project_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("app");
    let clean: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if clean.trim_matches('_').is_empty() {
        "app".to_string()
    } else {
        clean
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

    async fn verify(&self, project_dir: &Path) -> VerificationResult {
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
                full: true,
            };
        }

        // Guardrail 1: compile every module (syntax oracle).
        let files = collect_with(project_dir, &["py"]);
        if files.is_empty() {
            return VerificationResult {
                full: true,
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
                full: true,
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
                    full: true,
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
            full: true,
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

    async fn build(&self, project_dir: &Path, target: &str) -> Result<BuildOutcome, String> {
        CBackend::build_c(project_dir, target).await
    }

    async fn verify(&self, project_dir: &Path) -> VerificationResult {
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
                full: true,
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
                        let before = errors.len();
                        errors.extend(parse_gcc(&err, f));
                        // A failed exit with no parseable error is still a
                        // fact — never report clean on nonzero status.
                        if errors.len() == before {
                            errors.push(CompileError {
                                code: "GCC_ERROR".to_string(),
                                message: err
                                    .trim()
                                    .lines()
                                    .last()
                                    .unwrap_or("gcc failed")
                                    .to_string(),
                                file: f.clone(),
                                line: 0,
                                col: 0,
                                suggestion: None,
                                source_line: None,
                                kind: ErrorKind::Other,
                            });
                        }
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
            full: true,
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
    parse_colon_errors(stderr, fallback_file, "GCC_ERROR")
}

/// Parse `path:line:col: error: message` compiler output (gcc, kotlinc)
/// into structured errors. Anything unparseable becomes one generic error
/// rather than silence.
fn parse_colon_errors(stderr: &str, fallback_file: &str, code: &str) -> Vec<CompileError> {
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
    if errors.is_empty() {
        // Warning-only output (e.g. kotlinc's kotlin-home notices on a
        // successful build) is NOT failure. Only mint a fallback error
        // when some line actually claims to be one.
        if let Some(msg) = stderr
            .trim()
            .lines()
            .map(|l| l.trim())
            .rfind(|t| !t.is_empty() && t.contains("error"))
        {
            errors.push(CompileError {
                code: code.to_string(),
                message: msg.to_string(),
                file: fallback_file.to_string(),
                line: 0,
                col: 0,
                suggestion: None,
                source_line: None,
                kind: ErrorKind::Other,
            });
        }
    }
    errors
}

// --- Kotlin ---

pub struct KotlinBackend;

impl CBackend {
    async fn build_c(project_dir: &Path, target: &str) -> Result<BuildOutcome, String> {
        let t = target.trim().to_lowercase();
        if !t.is_empty() && t != "native" && t != "debug" && t != "bin" {
            return Err(format!(
                "C backend has no target '{}' (native only) — BLOCKED",
                target
            ));
        }
        if !command_available("gcc") {
            return Err("gcc is not available on PATH — cannot build C — BLOCKED".to_string());
        }
        let files = collect_with(project_dir, &["c"]);
        if files.is_empty() {
            return Err("No C sources in project — nothing to build — BLOCKED".to_string());
        }
        let out_dir = project_dir.join("target");
        std::fs::create_dir_all(&out_dir)
            .map_err(|e| format!("Cannot create target dir: {}", e))?;
        let out = out_dir.join(safe_bin_name(project_dir));
        let o = Command::new("gcc")
            .args(&files)
            .arg("-O2")
            .arg("-o")
            .arg(&out)
            .current_dir(project_dir)
            .output()
            .map_err(|e| format!("Failed to run gcc: {}", e))?;
        let stdout = String::from_utf8_lossy(&o.stdout).to_string();
        let stderr = String::from_utf8_lossy(&o.stderr).to_string();
        if !o.status.success() {
            return Err(format!(
                "gcc failed (exit {}): {}",
                o.status.code().unwrap_or(-1),
                tail2k(&stderr)
            ));
        }
        check_artifact(&out)
            .map(|_| BuildOutcome {
                artifact: out,
                stdout_tail: tail2k(&stdout),
                stderr_tail: tail2k(&stderr),
            })
            .map_err(|e| format!("gcc reported success but {}", e))
    }
}

/// Artifact post-condition: present, a file, non-empty.
fn check_artifact(path: &Path) -> Result<u64, String> {
    let meta = std::fs::metadata(path)
        .map_err(|e| format!("artifact {} missing: {}", path.display(), e))?;
    if !meta.is_file() {
        return Err(format!("artifact {} is not a file", path.display()));
    }
    if meta.len() == 0 {
        return Err(format!("artifact {} is empty", path.display()));
    }
    Ok(meta.len())
}

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

    /// `.kt` sources compile with provisioned kotlinc into one jar;
    /// pure-`.java` trees compile with `javac` into classes. Closed
    /// target vocabulary; missing toolchains block honestly.
    async fn build(&self, project_dir: &Path, target: &str) -> Result<BuildOutcome, String> {
        let t = target.trim().to_lowercase();
        if !t.is_empty() && t != "native" && t != "jar" && t != "classes" {
            return Err(format!(
                "JVM backend has no target '{}' (jar, classes) — BLOCKED",
                target
            ));
        }
        let kts = collect_with(project_dir, &["kt"]);
        if !kts.is_empty() {
            let Some((kotlinc_cp, stdlib)) = kotlin_toolchain().await else {
                return Err(
                    "No kotlinc found and provisioning failed — cannot build Kotlin — BLOCKED"
                        .to_string(),
                );
            };
            let out_jar = project_dir.join("target").join("grounding-app.jar");
            if let Some(parent) = out_jar.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Cannot create target dir: {}", e))?;
            }
            let o = Command::new("java")
                .arg("-cp")
                .arg(&kotlinc_cp)
                .arg("org.jetbrains.kotlin.cli.jvm.K2JVMCompiler")
                .args(&kts)
                .arg("-d")
                .arg(&out_jar)
                .arg("-cp")
                .arg(&stdlib)
                .current_dir(project_dir)
                .output()
                .map_err(|e| format!("Failed to run kotlinc: {}", e))?;
            let stdout = String::from_utf8_lossy(&o.stdout).to_string();
            let stderr = String::from_utf8_lossy(&o.stderr).to_string();
            if !o.status.success() {
                return Err(format!(
                    "kotlinc failed (exit {}): {}",
                    o.status.code().unwrap_or(-1),
                    tail2k(&stderr)
                ));
            }
            return check_artifact(&out_jar)
                .map(|_| BuildOutcome {
                    artifact: out_jar,
                    stdout_tail: tail2k(&stdout),
                    stderr_tail: tail2k(&stderr),
                })
                .map_err(|e| format!("kotlinc reported success but {}", e));
        }
        let javas = collect_with(project_dir, &["java"]);
        if !javas.is_empty() {
            if !command_available("javac") {
                return Err(
                    "javac is not available on PATH — cannot build Java — BLOCKED".to_string(),
                );
            }
            let out_dir = project_dir.join("target").join("classes");
            std::fs::create_dir_all(&out_dir)
                .map_err(|e| format!("Cannot create target dir: {}", e))?;
            let o = Command::new("javac")
                .arg("-d")
                .arg(&out_dir)
                .args(&javas)
                .current_dir(project_dir)
                .output()
                .map_err(|e| format!("Failed to run javac: {}", e))?;
            let stdout = String::from_utf8_lossy(&o.stdout).to_string();
            let stderr = String::from_utf8_lossy(&o.stderr).to_string();
            if !o.status.success() {
                return Err(format!(
                    "javac failed (exit {}): {}",
                    o.status.code().unwrap_or(-1),
                    tail2k(&stderr)
                ));
            }
            // Proof is a real .class file, newest wins.
            let mut classes: Vec<_> = walkdir::WalkDir::new(&out_dir)
                .into_iter()
                .flatten()
                .map(|e| e.path().to_path_buf())
                .filter(|p| p.extension().is_some_and(|e| e == "class"))
                .collect();
            classes.sort();
            let artifact = classes.into_iter().next_back().ok_or_else(|| {
                "javac reported success but no .class exists — BLOCKED".to_string()
            })?;
            return check_artifact(&artifact)
                .map(|_| BuildOutcome {
                    artifact,
                    stdout_tail: tail2k(&stdout),
                    stderr_tail: tail2k(&stderr),
                })
                .map_err(|e| format!("javac reported success but {}", e));
        }
        Err("No .kt or .java sources in project — nothing to build — BLOCKED".to_string())
    }

    async fn verify(&self, project_dir: &Path) -> VerificationResult {
        let Some((kotlinc_cp, stdlib)) = kotlin_toolchain().await else {
            return VerificationResult {
                clean: false,
                errors: vec![CompileError {
                    code: "KOTLINC_MISSING".to_string(),
                    message: "No kotlinc found and provisioning failed (tried env, Gradle cache, Maven Central) — cannot verify Kotlin".to_string(),
                    file: String::new(),
                    line: 0,
                    col: 0,
                    suggestion: None,
                    source_line: None,
                    kind: ErrorKind::Other,
                }],
                warnings: Vec::new(),
                stdout: String::new(),
                stderr: "kotlinc not found".to_string(),
                full: true,
            };
        };

        let mut errors: Vec<CompileError> = Vec::new();
        let mut stdout = String::new();
        let mut stderr = String::new();

        let files = collect_with(project_dir, &["kt"]);
        if files.is_empty() {
            return VerificationResult {
                clean: true,
                errors,
                warnings: Vec::new(),
                stdout,
                stderr,
                full: true,
            };
        }

        // Compile everything into one jar: kotlinc exit code is the oracle.
        let out_jar = project_dir.join("target").join("grounding-check.jar");
        let _ = std::fs::create_dir_all(out_jar.parent().unwrap());
        let mut cmd = Command::new("java");
        cmd.arg("-cp")
            .arg(&kotlinc_cp)
            .arg("org.jetbrains.kotlin.cli.jvm.K2JVMCompiler")
            .args(&files)
            .arg("-d")
            .arg(&out_jar)
            .arg("-cp")
            .arg(&stdlib)
            .current_dir(project_dir);
        match cmd.output() {
            Ok(o) => {
                stdout.push_str(&String::from_utf8_lossy(&o.stdout));
                let err = String::from_utf8_lossy(&o.stderr).to_string();
                stderr.push_str(&err);
                if !o.status.success() {
                    // Attribute diagnostics to the first file when the
                    // parser cannot (paths are absolute in kotlinc output).
                    let fallback = files.first().cloned().unwrap_or_default();
                    errors.extend(parse_colon_errors(&err, &fallback, "KOTLIN_ERROR"));
                    if errors.is_empty() {
                        errors.push(CompileError {
                            code: "KOTLIN_ERROR".to_string(),
                            message: "kotlinc failed".to_string(),
                            file: fallback,
                            line: 0,
                            col: 0,
                            suggestion: None,
                            source_line: None,
                            kind: ErrorKind::Other,
                        });
                    }
                    return VerificationResult {
                        clean: false,
                        errors,
                        warnings: Vec::new(),
                        stdout,
                        stderr,
                        full: true,
                    };
                }
            }
            Err(e) => {
                return VerificationResult {
                    clean: false,
                    errors: vec![CompileError {
                        code: "JAVA_ERROR".to_string(),
                        message: format!("Failed to run kotlinc: {}", e),
                        file: String::new(),
                        line: 0,
                        col: 0,
                        suggestion: None,
                        source_line: None,
                        kind: ErrorKind::Other,
                    }],
                    warnings: Vec::new(),
                    stdout,
                    stderr,
                    full: true,
                };
            }
        }

        // Behavior: run each `fun main` entry point; nonzero exit or a
        // `check()` throw fails the run. Bounded at 60s per entry via the
        // `timeout` utility when present (a hanging main must fail the
        // run, never the engine).
        for f in &files {
            let content = std::fs::read_to_string(f).unwrap_or_default();
            if !content.contains("fun main(") {
                continue;
            }
            let class = main_class_for(f);
            let cp = format!("{}:{}", out_jar.to_string_lossy(), stdlib);
            let run = if command_available("timeout") {
                Command::new("timeout")
                    .arg("60")
                    .arg("java")
                    .arg("-cp")
                    .arg(&cp)
                    .arg(&class)
                    .current_dir(project_dir)
                    .output()
            } else {
                Command::new("java")
                    .arg("-cp")
                    .arg(&cp)
                    .arg(&class)
                    .current_dir(project_dir)
                    .output()
            };
            match run {
                Ok(o) => {
                    stdout.push_str(&String::from_utf8_lossy(&o.stdout));
                    stderr.push_str(&String::from_utf8_lossy(&o.stderr));
                    if !o.status.success() {
                        let code = o.status.code().unwrap_or(-1);
                        errors.push(CompileError {
                            code: "TEST_FAILURE".to_string(),
                            message: if code == 124 {
                                format!("{} timed out after 60s", class)
                            } else {
                                format!("{} exited {}", class, code)
                            },
                            file: f.clone(),
                            line: 0,
                            col: 0,
                            suggestion: None,
                            source_line: None,
                            kind: ErrorKind::Other,
                        });
                    }
                }
                Err(e) => errors.push(CompileError {
                    code: "JAVA_ERROR".to_string(),
                    message: format!("Failed to run {}: {}", class, e),
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
            full: true,
        }
    }
}

/// Kotlin file facade: `Rng.kt` with top-level `fun main` → class `RngKt`.
fn main_class_for(path: &str) -> String {
    let stem = Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Main");
    let mut name: String = stem
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if name.chars().next().is_none_or(|c| c.is_ascii_digit()) {
        name = format!("File{}", name);
    }
    let mut chars = name.chars();
    let capitalized = match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => "Main".to_string(),
    };
    format!("{}Kt", capitalized)
}

/// Locate a working kotlinc: `KOTLIN_COMPILER_CP` + `KOTLIN_STDLIB` win;
/// else the Gradle module cache; else provision pinned jars from Maven
/// Central like pip fetching a build backend (checksum-verified, cached).
/// Returns (compiler classpath, stdlib jar).
async fn kotlin_toolchain() -> Option<(String, String)> {
    if let (Ok(cp), Ok(stdlib)) = (
        std::env::var("KOTLIN_COMPILER_CP"),
        std::env::var("KOTLIN_STDLIB"),
    ) {
        return Some((cp, stdlib));
    }
    if let Some(found) = search_gradle_kotlin() {
        return Some(found);
    }
    // Provision: download pinned artifacts, verify hashes, cache them.
    let dir = crate::tool::ensure_tool(&crate::tool::KOTLIN_TOOL)
        .await
        .ok()?;
    let get = |n: &str| dir.join(n).to_string_lossy().to_string();
    Some((
        [
            get("kotlin-compiler-embeddable-2.0.20.jar"),
            get("kotlin-stdlib-2.0.20.jar"),
            get("kotlinx-coroutines-core-jvm-1.6.4.jar"),
            get("annotations-13.0.jar"),
            get("trove4j-1.0.20200330.jar"),
        ]
        .join(":"),
        get("kotlin-stdlib-2.0.20.jar"),
    ))
}

fn search_gradle_kotlin() -> Option<(String, String)> {
    let home = dirs_home_fallback()?;
    let cache = home.join(".gradle/caches/modules-2/files-2.1");
    if !cache.exists() {
        return None;
    }
    let find = |name_part: &str| walk_find(&cache, name_part);
    let compiler = find("kotlin-compiler-embeddable-")?;
    // The stdlib must share the compiler's major.minor — a newer cached
    // stdlib (e.g. 2.2.0 from an Android build) against an older compiler
    // fails with incompatible-metadata errors. No match means fall back
    // to the pinned provisioned set, never a mixed one.
    let stdlib = compiler_major_minor(&compiler)
        .and_then(|mm| find(&format!("kotlin-stdlib-{}.", mm)))
        .or_else(|| {
            // Legacy loose match, kept only when it agrees with the
            // compiler; otherwise provisioning wins.
            let loose = find("kotlin-stdlib-2.")?;
            let same = compiler_major_minor(&compiler)
                .is_some_and(|mm| loose.contains(&format!("kotlin-stdlib-{}.", mm)));
            same.then_some(loose)
        })?;
    let coroutines = find("kotlinx-coroutines-core-jvm-")?;
    let annotations = find("annotations-13.0.jar")?;
    let trove = find("trove4j-")?;
    Some((
        [compiler, stdlib.clone(), coroutines, annotations, trove].join(":"),
        stdlib,
    ))
}

/// `…/kotlin-compiler-embeddable-2.0.20.jar` → `"2.0"`. Pure parsing,
/// unit-tested; None when the filename carries no version.
fn compiler_major_minor(path: &str) -> Option<String> {
    let file = path.rsplit('/').next()?;
    let ver = file
        .strip_prefix("kotlin-compiler-embeddable-")?
        .strip_suffix(".jar")?;
    let mut parts = ver.split('.');
    Some(format!("{}.{}", parts.next()?, parts.next()?))
}

fn dirs_home_fallback() -> Option<std::path::PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
}

fn walk_find(root: &Path, name_part: &str) -> Option<String> {
    let mut stack = vec![root.to_path_buf()];
    let mut best: Option<String> = None;
    let mut seen = 0;
    while let Some(current) = stack.pop() {
        if seen > 20000 {
            break;
        }
        seen += 1;
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if let Some(name) = path.file_name().and_then(|n| n.to_str())
                && name.starts_with(name_part)
                && name.ends_with(".jar")
                && !name.contains("sources")
            {
                let s = path.to_string_lossy().to_string();
                if best.as_ref().is_none_or(|b| s > *b) {
                    best = Some(s);
                }
            }
        }
    }
    best
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

    async fn verify(&self, project_dir: &Path) -> VerificationResult {
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
                full: true,
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
                    full: true,
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
                full: true,
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

    async fn verify(&self, project_dir: &Path) -> VerificationResult {
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
                full: true,
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
                full: true,
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
                full: true,
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
            full: true,
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
    match ext {
        "py" => Backend::Python,
        "kt" | "java" => Backend::Kotlin,
        "c" | "h" => Backend::C,
        "html" | "htm" => Backend::Html,
        "rs" => Backend::Rust,
        _ => {
            if let Some(spec) = registry()
                .iter()
                .find(|s| s.extensions.iter().any(|e| e == ext))
            {
                return Backend::Generic(GenericBackend { spec });
            }
            Backend::Rust
        }
    }
}

/// Backend for a whole project by layout markers (override wins).
pub fn backend_for_project<'a>(project_dir: &Path, extra: &'a [LanguageSpec]) -> Backend<'a> {
    if project_dir.join("Cargo.toml").exists() {
        return Backend::Rust;
    }
    if project_dir.join("pyproject.toml").exists()
        || project_dir.join("requirements.txt").exists()
        || project_dir.join("setup.py").exists()
        || has_source_files(project_dir, &["py"])
    {
        return Backend::Python;
    }
    if project_dir.join("build.gradle").exists()
        || project_dir.join("build.gradle.kts").exists()
        || project_dir.join("settings.gradle").exists()
        || has_source_files(project_dir, &["kt", "java"])
    {
        return Backend::Kotlin;
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
        return Backend::C;
    }
    if has_source_files(project_dir, &["html", "htm"]) {
        return Backend::Html;
    }
    // Last resort: first source file's backend wins over the Rust default.
    if let Some(ext) = first_source_extension(project_dir, extra) {
        let probe = Path::new("probe").with_extension(ext);
        return backend_for_file(&probe, extra);
    }
    Backend::Rust
}

fn generic_by_name<'a>(extra: &'a [LanguageSpec], name: &str) -> Backend<'a> {
    if let Some(spec) = extra.iter().find(|s| s.name == name) {
        Backend::Generic(GenericBackend { spec })
    } else if let Some(spec) = registry().iter().find(|s| s.name == name) {
        Backend::Generic(GenericBackend { spec })
    } else {
        Backend::Rust
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
/// Concrete variants (no trait objects) so the trait can stay `async`.
pub enum Backend<'a> {
    Rust,
    Python,
    Kotlin,
    C,
    Html,
    Generic(GenericBackend<'a>),
}

macro_rules! dispatch {
    ($self:ident, $method:ident ( $($arg:expr),* )) => {
        match $self {
            Backend::Rust => RustBackend.$method($($arg),*),
            Backend::Python => PythonBackend.$method($($arg),*),
            Backend::Kotlin => KotlinBackend.$method($($arg),*),
            Backend::C => CBackend.$method($($arg),*),
            Backend::Html => HtmlBackend.$method($($arg),*),
            Backend::Generic(g) => g.$method($($arg),*),
        }
    };
}

impl LanguageBackend for Backend<'_> {
    fn language(&self) -> &str {
        dispatch!(self, language())
    }

    fn extensions(&self) -> &'static [&'static str] {
        match self {
            Backend::Rust => RustBackend.extensions(),
            Backend::Python => PythonBackend.extensions(),
            Backend::Kotlin => KotlinBackend.extensions(),
            Backend::C => CBackend.extensions(),
            Backend::Html => HtmlBackend.extensions(),
            Backend::Generic(_) => &[],
        }
    }

    fn import_line(&self, path: &str) -> String {
        dispatch!(self, import_line(path))
    }

    fn is_import_line(&self, line: &str) -> bool {
        dispatch!(self, is_import_line(line))
    }

    fn is_known_std(&self, symbol: &str) -> bool {
        dispatch!(self, is_known_std(symbol))
    }

    async fn verify(&self, project_dir: &Path) -> VerificationResult {
        match self {
            Backend::Rust => RustBackend.verify(project_dir).await,
            Backend::Python => PythonBackend.verify(project_dir).await,
            Backend::Kotlin => KotlinBackend.verify(project_dir).await,
            Backend::C => CBackend.verify(project_dir).await,
            Backend::Html => HtmlBackend.verify(project_dir).await,
            Backend::Generic(g) => g.verify(project_dir).await,
        }
    }

    async fn build(&self, project_dir: &Path, target: &str) -> Result<BuildOutcome, String> {
        match self {
            Backend::Rust => RustBackend.build(project_dir, target).await,
            Backend::Python => PythonBackend.build(project_dir, target).await,
            Backend::Kotlin => KotlinBackend.build(project_dir, target).await,
            Backend::C => CBackend.build(project_dir, target).await,
            Backend::Html => HtmlBackend.build(project_dir, target).await,
            Backend::Generic(g) => g.build(project_dir, target).await,
        }
    }

    fn skip_lines(&self) -> usize {
        dispatch!(self, skip_lines())
    }
}

fn has_source_files(dir: &Path, extensions: &[&str]) -> bool {
    collect_with(dir, extensions).into_iter().next().is_some()
}

fn collect_with(dir: &Path, extensions: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![(dir.to_path_buf(), 0u8)];
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
    // Capped: same OOM discipline as the symbol scan. Depth 8, 5000 files.
    while let Some((current, depth)) = stack.pop() {
        if out.len() >= 5000 {
            break;
        }
        if depth > 8 {
            continue;
        }
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
                if name.starts_with('.') || ignore.contains(&name.as_str()) {
                    continue;
                }
                stack.push((path, depth + 1));
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

#[cfg(test)]
mod build_tests {
    use super::*;

    #[test]
    fn failing_crates_parses_oracle_output() {
        let err = "error: could not compile `rhai` (lib) due to 436 previous errors\n\
                   error: could not compile `lightningcss` (lib) due to 9 previous errors\n\
                   error: could not compile `rhai` (lib) due to 436 previous errors\n\
                   warning: unused import\n";
        assert_eq!(
            failing_crates(err),
            vec!["rhai".to_string(), "lightningcss".to_string()]
        );
        assert!(failing_crates("linking everything fine\n").is_empty());
    }

    #[test]
    fn compiler_major_minor_parses_jar_names() {
        assert_eq!(
            compiler_major_minor("/x/kotlin-compiler-embeddable-2.0.20.jar"),
            Some("2.0".to_string())
        );
        assert_eq!(compiler_major_minor("nope.jar"), None);
        assert_eq!(
            compiler_major_minor("/x/kotlin-compiler-embeddable-.jar"),
            None
        );
    }

    #[test]
    fn driver_runs_rejects_dead_binary() {
        let miss = std::path::Path::new("/tmp/gc-no-such-driver-xyz");
        assert!(driver_runs(miss).is_err());
    }

    #[test]
    fn unwind_retry_matches_only_its_signature() {
        assert!(needs_unwind_retry(
            "ld.lld: error: undefined symbol: _Unwind_DeleteException"
        ));
        assert!(!needs_unwind_retry("error[E0596]: cannot borrow"));
        assert!(!needs_unwind_retry("undefined symbol: foo_bar"));
        assert!(!needs_unwind_retry(""));
    }
}
