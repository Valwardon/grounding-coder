//! `.grounding.toml` extras are applied at selection time, and the
//! GenericBackend runs the configured verify command honestly: a command
//! that runs and succeeds is clean, a missing binary blocks, a failing
//! command is never clean.

use grounding_coder::engine::lang::{
    LanguageBackend, LanguageSpec, backend_by_name, backend_for_file, load_extra,
    registry_specs_with_extras,
};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static CTR: AtomicU64 = AtomicU64::new(93000);

fn tmp_dir() -> std::path::PathBuf {
    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("gc-langextra-{}-{}", std::process::id(), id));
    fs::create_dir_all(&base).expect("tmp");
    base
}

fn spec_toml(name: &str, ext: &str, verify_cmd: &str) -> String {
    format!(
        "[language]\n\
         name = \"{name}\"\n\
         extensions = [\"{ext}\"]\n\
         import_template = \"need {{p}};\"\n\
         import_prefixes = [\"need \"]\n\
         verify_cmd = [{verify_cmd}]\n"
    )
}

#[tokio::test]
async fn extra_verify_cmd_runs_and_cleans() {
    let base = tmp_dir();
    fs::write(
        base.join(".grounding.toml"),
        spec_toml("zz", "zz", "\"sh\", \"-c\", \"echo ok\""),
    )
    .unwrap();
    fs::write(base.join("prog.zz"), "say hello\n").unwrap();

    let extra = load_extra(&base);
    assert_eq!(extra.len(), 1, "the extra language must be parsed");

    let probe = Path::new("prog.zz").to_path_buf();
    let backend = backend_for_file(&probe, &extra);
    assert_eq!(backend.language(), "zz");

    let result = backend.verify(&base).await;
    assert!(result.is_clean(), "got: {:?}", result.errors);
    assert!(
        result.stdout.contains("ok"),
        "the configured verify command never ran: {:?}",
        result.stdout
    );
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn extra_overrides_a_registry_language_at_selection() {
    // `go` is in the registry with `go build ./...`; this project has no go
    // toolchain, so only the project's own override can make it clean.
    let base = tmp_dir();
    fs::write(
        base.join(".grounding.toml"),
        spec_toml("go", "go", "\"sh\", \"-c\", \"echo from-project\""),
    )
    .unwrap();
    fs::write(base.join("main.go"), "package main\n\nfunc main() {}\n").unwrap();

    let extra = load_extra(&base);

    // Merged view: the extra shadows the registry's `go`, registry lives on.
    let merged = registry_specs_with_extras(&extra);
    assert_eq!(
        merged.iter().filter(|s| s.name == "go").count(),
        1,
        "the extra must shadow the registered go, not sit beside it"
    );
    assert!(merged.iter().any(|s| s.name == "zig"), "registry stays");

    // Selection consults the extra first: same extension, project's spec.
    let backend = backend_for_file(Path::new("main.go"), &extra);
    assert_eq!(backend.language(), "go");
    let result = backend.verify(&base).await;
    assert!(result.is_clean(), "got: {:?}", result.errors);
    assert!(
        result.stdout.contains("from-project"),
        "the registry command ran instead of the project's: {:?}",
        result.stdout
    );

    // By name too: extras first, registry reachable for un-overridden names.
    let by_name = backend_by_name(&extra, "go").expect("extra go resolves");
    assert!(by_name.verify(&base).await.is_clean());
    let zig = backend_by_name(&extra, "zig").expect("registry zig resolves");
    assert_eq!(zig.language(), "zig");

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn missing_verify_binary_blocks_honestly() {
    let base = tmp_dir();
    fs::write(
        base.join(".grounding.toml"),
        spec_toml("zz", "zz", "\"gc-no-such-toolchain\", \"--check\""),
    )
    .unwrap();
    fs::write(base.join("prog.zz"), "say hello\n").unwrap();

    let extra = load_extra(&base);
    let backend = backend_for_file(Path::new("prog.zz"), &extra);
    let result = backend.verify(&base).await;
    assert!(!result.clean, "a missing tool must never be clean");
    assert_eq!(result.errors.len(), 1, "got: {:?}", result.errors);
    assert!(
        result.errors[0].code.ends_with("_MISSING"),
        "got: {}",
        result.errors[0].code
    );
    assert!(
        result.errors[0].message.contains("not available on PATH"),
        "got: {}",
        result.errors[0].message
    );
    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn failing_verify_command_is_never_clean() {
    // Nonzero exit with no parseable diagnostics (nothing to `path:line:`)
    // is still a failure — a broken tool must not read as a pass.
    let base = tmp_dir();
    fs::write(
        base.join(".grounding.toml"),
        spec_toml("zz", "zz", "\"sh\", \"-c\", \"exit 3\""),
    )
    .unwrap();
    fs::write(base.join("prog.zz"), "say hello\n").unwrap();

    let extra = load_extra(&base);
    let backend = backend_for_file(Path::new("prog.zz"), &extra);
    let result = backend.verify(&base).await;
    assert!(!result.clean, "a failing oracle must never be clean");
    assert!(
        result.errors.iter().any(|e| e.code == "VERIFY_ERROR"),
        "got: {:?}",
        result.errors
    );
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn unknown_language_has_no_backend() {
    // The selection API's refusal: no guess, no fallback language.
    let extra: Vec<LanguageSpec> = Vec::new();
    assert!(backend_by_name(&extra, "cobol-95").is_none());
    assert!(backend_by_name(&extra, "").is_none());
    assert!(backend_by_name(&extra, "rust").is_some());
    assert!(backend_by_name(&extra, "node").is_some());
}
