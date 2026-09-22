# Grounding Coder — Progress

## Goal
Pivot grounded from an overly ambitious cognitive engine into a deterministic, non-guessing coding agent for Android — `grounding-coder`. The LLM is ONLY an unverified NL→intent translator; all code generation, verification, and correction is driven by grounded's deterministic engine patterns.

## Architecture

### Determinism Guarantee
- **LLM (unverified)**: OpenRouter API client → translates NL → structured intent JSON only. NEVER writes code.
- **Engine (deterministic)**: TaskDecomposer → RetryBudget → SymbolBinder → CodeVerifier → CorrectionPipeline
- **Compiler as oracle**: `cargo check/test/clippy` → never guesses, always verifies
- **EditPlan enforcement**: Agent may only modify bytes identified by structured edit backed by evidence
- **Transactional loop**: snapshot → apply → verify → commit OR rollback

### Key Components (all repurposed from grounded)
| Grounded Component | Grounding-Coder Repurposing |
|---|---|
| GraphArena | CodeArena — arena-backed graph of code symbols |
| KnowledgeStore | SymbolTable — embedded API ref + runtime cache |
| CuriosityBudget | RetryBudget — energy-bounded correction attempts |
| DefinitionResolver | SymbolBinder via SymbolTable.fetch() |
| VerificationLoop | CodeVerifier — cargo check/test/clippy |
| SelfHealingPipeline | CorrectionPipeline — 5-phase error correction |
| GoalFormationEngine | TaskDecomposer — breaks intent into sub-tasks |
| EpisodicRecorder | RecipeLog — error→fix recipe database |
| Motor System | CodeWriter — composes code from verified patterns |

### ResearchOracle (added during development)
Solves the "no code generation from fresh project" flaw. When the bot encounters unknown symbols, it fetches definitions from verified sources:
- docs.rs (Rust crate docs)
- developer.android.com (Android SDK)
- crates.io API (crate metadata)
- kotlinlang.org (Kotlin docs)

All fetched definitions are compiler-verified before caching. The bot never trusts broken code.

## Build & Run

### CLI (any platform)
```bash
cargo run --bin gc -- --help          # Show commands
cargo run --bin gc -- chat "prompt"   # Chat with the bot
cargo run --bin gc -- symbols -p .    # List indexed symbols
cargo run --bin gc -- recipes         # List fix recipes
```

### Android APK
Built from the Gradle project in `android/`, producing `dist/grounding-coder.apk`.
```bash
cargo build --release --target aarch64-linux-android --features ui --lib
cp target/aarch64-linux-android/release/libgrounding_coder.so android/app/src/main/jniLibs/arm64-v8a/
gradle -p android :app:assembleDebug
cp android/app/build/outputs/apk/debug/app-debug.apk dist/grounding-coder.apk
```
Output: `dist/grounding-coder.apk` (~7.0 MB, arm64-v8a, debug-signed).

## Quality Gates
```bash
cargo check --all-features
cargo clippy --all-targets --all-features   # zero warnings
cargo fmt --check
cargo test --all-features
```

## Deliverables
- GitHub: https://github.com/Valwardon/grounding-coder
- Release: https://github.com/Valwardon/grounding-coder/releases/tag/v0.2.0
- APK: `grounding-coder.apk` (arm64) uploaded to release

## v0.2.1 — White Screen Fix + Transactional Architecture

### Critical Fixes

1. **White screen crash on Android**: Fixed by implementing all abstract methods in `MainActivity.kt`
   - `setWebView(webView: RustWebView)`
   - `onWebViewCreate(webView: WebView)`
   - `onWebViewLoadUrl(url: String)` and overload with headers
   - `onCreate()` with Rust engine initialization
   - Companion object with `System.loadLibrary("grounding_coder")`

2. **Architectural enforcement**: Replaced unsafe whole-file writes with EditPlan + exact byte-range edits
   - New `src/engine/plan.rs` module with `EditPlan`, `SourceEdit`, `Evidence` structs
   - `CodeWriter.plan()` generates precise edit plans backed by evidence
   - `CodeWriter.apply_plan()` applies ONLY exact byte-range edits
   - Deprecated old `write()` and `apply()` methods

3. **Transactional execution**: Every task follows snapshot → apply → verify → commit/rollback
   - Pre-edit snapshots captured before any changes
   - Verification failure triggers automatic rollback
   - Never leaves partial/broken changes behind

4. **Honesty principle**: Agent reports "I don't know" via `BlockReason` instead of guessing
   - New `AgentOutcome` enum with `Success`, `Blocked`, `Failed` variants
   - `BlockReason` variants: `UnknownSymbol`, `NoVerifiedPattern`, `UnsupportedAction`, `NoSafeFix`, `VerificationToolUnavailable`, `StaleEdit`
   - `TaskState` machine for explicit state tracking

5. **Explicit EditIntents**: Replaced `serde_json::Value` escape hatch with typed operations
   - `EditIntent` enum: `AddImport`, `InsertAfterSymbol`, `ReplaceRange`, `AddDefinition`
   - `IntentDefinition` without `code` field — LLM only provides metadata, not implementation
   - `IntentTest` without `code` field — agent decides test construction
   - `StructuredIntent` with `unknown_requirements` field

### File Changes
- `android/app/src/main/kotlin/dev/dioxus/main/MainActivity.kt` — Fixed white screen crash
- `src/engine/plan.rs` — New module: EditPlan, SourceEdit, Evidence, TaskState
- `src/engine/writer.rs` — Added plan/snapshot/apply_plan/rollback/commit methods
- `src/engine/mod.rs` — Transactional loop in run_task(), added AgentOutcome and BlockReason
- `src/engine/tasks.rs` — Added EditIntent enum, removed code from IntentDefinition/IntentTest
- `README.md` — Updated with new architecture, critical invariant, fixes
- `PROGRESS.md` — Updated with v0.2.1 progress

## Next Steps
- [ ] Build Android APK
- [ ] Push APK as release on GitHub
- [ ] Integration tests for EditPlan
- [ ] End-to-end safety suite (tests/end_to_end.rs)
