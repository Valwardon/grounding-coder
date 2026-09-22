# grounding-coder

A **deterministic, non-guessing coding agent**. The LLM is only an unverified natural-language→intent translator. All actual coding, verification, and correction is driven by grounded's deterministic engine patterns — no guessing, no hallucination.

## Core Principle

> The compiler is the oracle. Tests are the truth. The codebase is the answer key.

The LLM generates hypotheses (intent). The deterministic engine verifies every hypothesis against hard oracles (cargo check, tests, codebase patterns) and corrects failures using a recipe log — never guessing.

## Architecture

```
Human (messy language)
  │
  ▼
LLM via OpenRouter ──── UNVERIFIED ────► structured intent JSON
  │
  ▼
Intent validation
  │
  ▼
Research / resolve symbols
  │
  ▼
Deterministic EditPlan
  │
  ▼
Pre-edit snapshot
  │
  ▼
Apply ONLY precise edits
  │
  ▼
cargo check / gradle build
  │
  ├── clean → success
  │
  └── errors
        ↓
   structured diagnostic
       ↓
   known recipe?
      / \
    yes  no
     ↓    ↓
   exact    STOP
   edit     honestly
     ↓
   verify again
```

## Critical Invariant

> The agent may only modify bytes identified by a structured edit whose target and replacement are justified by project state, compiler diagnostics, or a verified recipe.

## Grounding Coder 0.2 — Castle + Project Intelligence

Version 0.2 introduces the Castle + Project Intelligence architecture, fixing the white screen crash and making the execution path evidence-driven and transactional.

### Key Features:

#### ✅ Enhanced Intent Understanding
- **Requirements Hypothesis**: LLM can indicate what it doesn't know via `unknown_requirements` field
- **Project Context**: StructuredIntent now includes platform, architecture, runtime, capabilities, domains, constraints, and dependencies
- **Confidence Scoring**: LLM can provide confidence levels for intent completeness

#### ✅ Knowledge Management
- **CastleStore**: Engineering knowledge base with compressed verified facts, symbols, patterns, and self-pruning
- **KnowledgeOracle**: Unified interface for knowledge adapters (Rust, Android, Solana, GitHub)
- **Relevance-Based Retrieval**: Retrieves only necessary knowledge, not entire documentation
- **Self-Pruning**: Automatically removes low-utility, stale, or contradicted knowledge

#### ✅ Android Project Intelligence
- **Project Workspace**: Real project selection mechanism instead of assuming "."
- **Project Manifest**: Comprehensive project metadata including platform, capabilities, domains, constraints
- **Android Capabilities**: Registry of device capabilities (network, filesystem, background execution, notifications, location, camera, audio, vibration)
- **Runtime Configuration**: SDK versions, package info, activities, permissions

#### ✅ Multi-Platform Verification
- **Rust Verification**: `cargo check`, `cargo test`, `cargo clippy`
- **Android Verification**: Gradle build, Android lint, resource validation, NDK compatibility
- **Platform-Aware**: Different verification oracles for Rust and Android projects

### New Components:

#### Android Modules
- `src/android/runtime.rs` — Android runtime with SDK info and capabilities
- `src/android/capabilities.rs` — Project capability registry
- `src/android/project.rs` — Project manifest and workspace management

#### Knowledge Management
- `src/knowledge/castle.rs` — CastleStore engineering knowledge base
- `src/knowledge/facts.rs` — Verified facts and code patterns
- `src/knowledge/sources.rs` — Source cache with compression

#### Knowledge Adapters
- `src/oracle/rust.rs` — RustOracle (crates.io, docs.rs)
- `src/oracle/android.rs` — AndroidOracle (Android SDK, AndroidX)
- `src/oracle/solana.rs` — SolanaOracle (Solana docs, examples)
- `src/oracle/github.rs` — GitHubOracle (repositories, code search)

### Build & Run

```bash
cargo build
cargo run --bin gc -- --help
```

### Android APK

The APK is built from the checked-in Gradle project in `android/` (mirrors the scaffold the Dioxus CLI generates), producing `dist/grounding-coder.apk`.

```bash
# 1. Build the Android cdylib (requires the Android Rust target + NDK toolchain)
cargo build --release --target aarch64-linux-android --features ui --lib

# 2. Assemble the APK (requires JDK 17, Android SDK at ANDROID_HOME)
cp target/aarch64-linux-android/release/libgrounding_coder.so \
   android/app/src/main/jniLibs/arm64-v8a/
gradle -p android :app:assembleDebug
cp android/app/build/outputs/apk/debug/app-debug.apk dist/grounding-coder.apk

# APK: dist/grounding-coder.apk  (~7.0 MB, arm64-v8a, debug-signed)
```

Requirements: Android SDK + NDK (`ANDROID_HOME`), JDK 17, Gradle 8.x, and the `aarch64-linux-android` Rust target. `dx bundle --android` works on hosts where the Dioxus CLI runs natively.

### Quality Gates

All checks run green in CI and locally:

```bash
cargo check --all-features
cargo clippy --all-targets --all-features   # zero warnings
cargo fmt --check
cargo test --all-features
```

### Deliverables
- **GitHub:** https://github.com/Valwardon/grounding-coder
- **Release:** https://github.com/Valwardon/grounding-coder/releases/tag/v0.2.0
- **APK:** `grounding-coder.apk` (arm64) uploaded to release

### Architecture Changes v0.2

The Grounding Coder 0.2 implements a significant architectural upgrade:

**Before (v0.1.0):**
- Tiny intent schema with only goal, file, language, actions, references, define, imports, test
- ResearchOracle limited to external sources (docs.rs, Android SDK docs)
- No project context or capabilities awareness
- Single verification oracle (cargo check/test/clippy)
- **write()/apply() could replace entire target files — bypassing determinism guarantees**

**After (v0.2.0):**
- Rich intent schema with platform, architecture, runtime, capabilities, domains, constraints, dependencies, and unknown_requirements
- CastleStore with compressed engineering memory
- Project workspace and manifest system
- Multiple platform-aware verification oracles
- Relevance-based knowledge retrieval
- Self-pruning knowledge base
- **Critical fix**: Transactional execution with snapshot/apply/verify/rollback loop
- **Write enforcement**: agent may only modify bytes identified by structured edit backed by evidence
- **BlockReason** for honest refusal when no safe fix exists
- **TaskState** machine for explicit state tracking

### Fixes in v0.2

1. **White screen crash**: Fixed by implementing all abstract methods in MainActivity.kt
2. **Architectural**: Replaced unsafe whole-file writes with EditPlan + exact byte-range edits
3. **Transactional**: Every task follows snapshot → apply → verify → commit/rollback
4. **Honesty principle**: Agent reports "I don't know" via BlockReason instead of guessing
5. **EditIntent enum**: Explicit operations instead of serde_json::Value escape hatch
6. **StructuredIntent**: unknown_requirements field for LLM to indicate uncertainty

### v0.2 Migration Notes

- The old `write()` and `apply()` methods are deprecated but kept for backward compatibility
- All new code should use `plan()` + `apply_plan()` pattern
- Tasks now follow the transactional loop: snapshot → apply → verify → commit/rollback
- If verification fails, the system rolls back to the pre-edit state
- BlockReason variants provide honest refusal when no safe fix exists
