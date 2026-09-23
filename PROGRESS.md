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
- [x] Build Android APK (per-release pipeline, latest v0.4.0+)
- [x] Push APK as release on GitHub
- [x] Integration tests for EditPlan (`tests/editplan.rs` — 4 tests)
- [x] End-to-end safety suite (`tests/end_to_end.rs` — 2 tests)
- [x] GitHub actor (`src/github.rs`: create repo + contents upload from Chat)
- [ ] APK pipeline wired to engine outputs
- [ ] Solana sniper demo (watcher project → APK → GitHub)

## v0.5.0 — External crates, honest Partials, GitHub delivery (2026-09-23)

Driven by a live device report: `rand::…` imports BLOCKED (no registry
path), nothing verifiable without a toolchain, and no way to ship to
GitHub. 42 tests green.

- **Registry-verified crates**: Step-0 crates.io lookup pins versions into
  the symbol table; `EnsureDep` tasks write `Cargo.toml` with registry
  evidence; imports of verified roots resolve. Proven live (`rand 0.10.3`).
- **`call`-op methods**: verified delegation `self.field.method(args)`
  against a checked table — enables wrapper types (RNG structs).
- **syn syntax oracle**: cargo-less hosts get real parse facts; outcomes
  split into SUCCESS vs labeled PARTIAL (never false-verified).
- **GitHub actor**: repo create + file upload from Chat via Settings token;
  `called`/`named` name extraction; delivery never flips outcomes.
- **RNG demo**: struct + `next_u32` body + seeded contract, all green.
- Repair loop gained compiler-suggestion `use`-path extraction; fixed a
  double-`E` error-code bug that silently broke recipe matching, and a
  missing idempotent-fix signal for repeat errors.

## v0.6.0 — Kotlin synthesis + tool provisioning (2026-09-23)

- **Kotlin oracle** (`KotlinBackend::verify`): kotlinc compile of all
  sources + JVM run of every `fun main` contract (60s bound each).
  Toolchain resolution: env → Gradle cache → provision.
- **Provisioner** (`src/tool.rs`): manifest-pinned artifacts (SHA-256),
  shared cache, corrupt bytes deleted on mismatch. Kotlin 2.0.20 set
  proven by live download test.
- **Kotlin RNG family** (`knew`/`kcall` ops over `kotlin.random.Random`,
  `run {}` contract mains). Lessons from proving it: bare `{}` is a
  lambda (needs `run`), duplicate imports are fatal in kotlinc (template
  must not emit what import tasks own), facade class is `<File>Kt`.
- **Async trait**: `LanguageBackend::verify` is async; `Backend` enum went
  concrete (no trait objects) to allow it. `detect_project_type` no
  longer claims bare `.kt` for Gradle.
- Proof: `RandomNumberGenerator(42).nextInt() == 972016666`, compiled and
  run green. Demo repo: Valwardon/kotlin-rng (sources + APK), where every
  GitHub step — repo create, source upload, `v0.1.0` release, APK asset —
  was performed by the actor itself from an intent, not by scripts.
  Delivery runs on all terminal outcomes; VerifyOnly plans before file
  resolution so bare-dir intents reach verification.

## v0.3 — The Engine Proves Itself (2026-09-22)

HEAD `9b32fdb` did not compile (37 errors) despite claiming green gates.
Rebuilt to green, then hardened and extended across four phases. 21 tests,
all passing; `cargo check`, `clippy -D warnings`, `fmt --check` clean.

### Phase 1 — Contract synthesis (`src/engine/synthesize.rs`)
- Intent carries input/output examples as pure data; LLM code rejected.
- Candidates built ONLY from a verified std ingredient index; compiler +
  generated contract tests judge; losers roll back.
- Proof: `word_counts(text) -> HashMap<String, usize>` — real logic.

### Phase 2 — Structs + fallible functions
- Struct family: fields + method-op vocabulary (`new`/`add_assign`/`get`).
  Proof: `Counter` with multi-step state contracts.
- Parse family: `(&str) -> Result<Int, String>` with Ok/Err contracts.
  Proof: `parse_port` incl. `Err("invalid digit…")` case.

### Phase 3 — Multi-file atomic transactions
- Plan-all → snapshot-all → execute; re-plan fresh at execution so byte
  offsets never go stale across tasks.
- New modules created + `mod` wired in the same EditPlan; missing files
  snapshot as created, rollback deletes.
- Proof: `src/geometry.rs` created + wired; a later failing task rolls back
  an earlier clean one byte-identical. Two real bugs found by tests and
  fixed: cross-plan staleness, missing NoSafeFix rollback.

### Phase 4 — Polyglot backends (`src/engine/lang.rs`)
- `LanguageBackend` trait: conventions + oracle + std inventory per language.
- Hand-tuned: Rust, Python (`py_compile`+pytest), C (`gcc`), Kotlin,
  HTML (tag-balance oracle over `html.parser`).
- Data-driven: built-in registry (Go, Node) + project `.grounding.toml`
  `[language]` tables — new languages need zero engine changes.
- Proofs: Python import, registry-JS import, invented `zz` language via
  TOML, honest `GO_MISSING` block, `def`/`func` symbol indexing.

### The dentist test (random challenge)
Prompt: *"a home page for a dentist office in Miami, Florida, in HTML"*.
Result: `SUCCESS` → `index.html` for Bright Smile Dental (services, Little
Havana blurb, phone, footer address), structure verified by the HTML oracle.
No Rust involved at any step — intent → page family → escaped template →
parser verify → commit. A hostile `<script>alert(1)</script>` payload
renders as inert `&lt;script&gt;` text. Empty title BLOCKED with no file.
Run it: `cargo run --example build_page -- /tmp/dentist-site /tmp/dentist-intent.json`.
