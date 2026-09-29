# Grounding Coder — Progress

## v0.16.0 — The loop sources its own base photos (2026-09-29)

- `engine/plates.rs`: Commons search → per-file metadata →
  provenance required complete (source URL, author, allowlisted
  license) → fetch → decode via the `image` crate (JPEG/PNG,
  resolved offline from the vendored cache) into the engine buffer.
  Incomplete records refuse naming the missing field; rate-limit
  refusals report and don't kill the plates that verified.
- `gc plate <query>`: ingests into a directory of BMPs plus a
  `provenance.json` manifest (file, source/page URLs, author,
  license), with downscale past 1280px so the repo stays lean.
- Proved live: "portrait" → JFK official portrait (Shikler,
  public domain, full chain) as `samples/plates/plate-00.bmp`.
- 181 tests green (108 lib + 73 integration), clippy `-D warnings`
  clean on default AND `ui`, fmt clean.

## v0.15.0 — The curiosity loop studies anatomy (2026-09-29)

- `BodyPlan`: the 7.5-heads canon as fractions of stature, each
  with cited sources (`ANATOMY_SOURCES`). `person_plan` builds
  posable individuals from it — capsule limbs, head with inset
  eyes, nose, mouth, offset hair cap, three distinct people in the
  group scene (looks and postures all differ, one waving).
- `study_anatomy`: renders a canon-built figure and measures its
  own photo via material labels — head/stature 0.128 (canon 0.133),
  span/stature 1.056 (canon 1.0), eye-spacing/head 0.35. Two real
  measurement bugs caught en route: bbox edges add eyeball diameter
  to the separation (centers are the canon quantity), and eyes go
  sub-pixel below 320px wide.
- `record_anatomy` files passing studies as Verified knowledge
  with the measurements as evidence; failures file Evidence, never
  promotion. `research_anatomy` does the web half on verified
  sources (dream loop calls it; tests stay offline).
- Three-point lighting (key casts shadows, fill/rim don't),
  capsule intersection, seeded film finish (grade/vignette/grain).
  The frozen demo renders byte-identical through the refactor.
- Samples: studied portrait (174 eye pixels) and three-person group
  join the demo in `samples/`.
- 179 tests green (106 lib + 73 integration), clippy `-D warnings`
  clean on default AND `ui`, fmt clean.

## v0.14.0 — Eyeballs: scenes, photos, manipulation (2026-09-29)

- `engine/vision.rs`: `Image` plus a hand-rolled 24-bit BMP codec
  (zero new dependencies), exact transforms (rect, disc, overlay,
  crop, nearest-neighbor resize), stats (brightness, histogram,
  dominant color, near-counts), Sobel edges, and SAD template
  matching. One real bug caught by its own tests: void-as-black
  borders printed phantom edges; borders now clamp.
- `engine/scene.rs`: scene graph (camera, light, ground plane,
  `person()` decomposed into head/torso/arms/legs, `tower()`
  shaft-plus-cap) rendered by a tiny deterministic raytracer
  (Lambert + hard shadows + gradient sky) with a per-material pixel
  receipt. Same scene twice renders byte-identical; adding the
  tower is proven to leave every person pixel untouched.
- `gc render --out photo.bmp`: renders the demo scene with receipt
  and BMP read-back verification. Proved live: 320×240 photo, sky
  up top, ground below, 3,826 tower pixels.
- What it honestly cannot do, stated in the code: name objects in
  arbitrary photographs. That takes a model; everything here is
  structure from arithmetic.
- 172 tests green (99 lib + 73 integration), clippy `-D warnings`
  clean on default AND `ui`, fmt clean.

## v0.13.0 — Compiler-probe evidence (2026-09-29)

- `engine/probe.rs`: `ProbeSpec` (claim + self-contained `main.rs`
  + required stdout markers) → `run_probe` builds a disposable
  scratch project and runs the full `CodeVerifier` oracle on it.
  The oracle runs on its own thread under a real `recv_timeout`
  deadline (the verifier is synchronous inside, so an async timeout
  could never fire — this is stated in the code, not hidden).
- `verify_concept_by_probe` walks Question/Hypothesis → Experiment
  → Evidence → Verified on a green probe, with the oracle
  transcript as the verification evidence. Red, uncompilable, and
  timed-out probes land in Evidence with a failure counted —
  never promoted, never deleted. Verified and Rejected concepts
  refuse re-probing with the reason stated.
- 6 new lib tests (green/red/uncompilable/timeout/chain-walk/
  failure-accounting), all passing against the real toolchain.
- 162 tests green (89 lib + 73 integration), clippy `-D warnings`
  clean on default AND `ui`, fmt clean.

## v0.12.0 — First learning experiment (2026-09-29)

- `engine/experiment.rs`: `run_family` runs a task family in two
  arms — dreaming (shared store + bounded idle passes between
  tasks) vs amnesiac (fresh store per task). Each task solves for
  real (fresh bot + scratch project); only the store carries state.
  The report records duration, budget consumed, research used,
  recognized vs new gaps, reused knowledge, and dream outcomes.
- `tests/learning_experiment.rs`: three word-count-shaped functions
  through the real bot. Dream arm: 3/3 solved, 6 recognized, 5 new,
  rediscovery rate **0.55**. Amnesiac: 3/3 solved, 0 recognized, 15
  new, rate **0.00**. Offline hypotheses verify nothing (firewall
  holds: 0 verified) and solving takes the same time — the measured
  learning is recognition, not speed. Speed is the probe follow-up.
- New `CodeBot::budget_remaining` accessor so attempts consumed are
  measured, not guessed.
- 156 tests green (83 lib + 73 integration), clippy `-D warnings`
  clean on default AND `ui`, fmt clean.

## v0.11.0 — Proactive curiosity and idle learning loop (2026-09-29)

- `engine/knowledge.rs`: every concept carries an explicit lifecycle
  (Unknown → Question → Hypothesis → Experiment → Evidence →
  Verified → Generalized, plus terminal Rejected). Promotion to
  Verified requires verification evidence — DREAM ≠ KNOWLEDGE is a
  store-enforced error, not a comment. `generalize` needs 2+
  verified supporters or the pattern is Rejected, never kept.
- `extract_gaps` turns a finished task (concepts used, unknowns
  seen, failures, related) into Questions with dependency seeds;
  solved and disproven concepts are left alone. `prioritize` studies
  breadth-first (least-visited, least-failed); `neighbors` walks the
  dependency graph — the what-next engine follows structure.
- `gc dream --project --budget [--research]`: bounded idle passes
  (at most `budget` investigations, sleeps when nothing is open),
  persisting to `.grounding/knowledge.jsonl` every step. `gc chat`
  now routes through `disposition()` like the rest.
- Deliberately v1-small: neighbor seeding across verified items is
  an explicit no-op hook (`guess_neighbors`) — inventing adjacency
  from spelling would be dreaming disguised as structure. Compiler
  probes as harder evidence are the named follow-up.
- 154 tests green (83 lib + 71 integration), clippy `-D warnings`
  clean on default AND `ui`, fmt clean.

## v0.10.4 — Benchmark pins the new capabilities (2026-09-29)

- Three new `intent_benchmark` rows so the v0.10.3 fixes cannot
  silently regress: head-noun kind rule ("settings page" → page,
  "counter struct" → struct), typo-verbs-never-names ("Build a bild
  page." asks instead of creating "Bild"), destructive-blocks-first
  ("Delete the file called Cleanup." blocks despite full
  specification).
- 145 tests green (77 lib + 68 integration), clippy `-D warnings`
  clean on default AND `ui`, fmt clean.

## v0.10.3 — Disposition engine + intent benchmark (2026-09-29)

- `disposition()` in `engine/understand.rs`: confidence + frame +
  prompt surface route to `Execute` / `Ask` / `Block`. Destructive
  verbs, contradiction pairs ("add but keep unchanged"), and
  multi-clause prompts never execute blind — even at high confidence.
  `ACT_THRESHOLD` (0.75) lives next to the lexicon; `--prose`,
  `--understand`, and chat refusal all route through it.
- Name-resolution fixes from the benchmark's typo case: the adjacent
  scan now skips typo'd verbs too (`fuzzy_verb` exclusion, so "Buld"
  stays an action), and kind comes from the head noun — the LAST kind
  word ("settings page" is a page about Settings, not a config).
- `tests/intent_benchmark.rs`: 8 categories, 10 cases (happy path,
  typo tolerance, vague, ambiguous, destructive, contradiction,
  multi-intent, unknown concepts) — all green.
- 142 tests green (77 lib + 65 integration), clippy `-D warnings`
  clean on default AND `ui`, fmt clean.

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
- [x] Clean-room authorship default + generalized Build op
- [x] Learning to rank + true-fix outcomes + precedent choice
- [x] Translator mode + config/component families
- [ ] Test the current build before any new APK work
- [ ] Solana sniper demo (watcher project → APK → GitHub)

## v0.7.0 — Clean-room, Build op, migration loop (2026-09-24/25)

- **Clean-room default** (`BlockReason::CleanRoom`): `files[]`
  replication intents refused before any fetch; explicit opt-out only.
  Proved: old port manifests refuse with disk untouched.
- **Generalized `TaskKind::Build`**: per-backend build oracles (C, Rust
  incl. triples, Kotlin jars, Java classes, Android APK via `dx`) with
  artifact-bytes proof. `tests/build.rs` (6 tests, incl. binaries that
  actually run).
- **Self-solving drivers**: `--version` verification, pinned per-arch
  downloads, source-build fallback (clone → oracle-driven dep refresh →
  build), NDK unwind-link retry. Full trails on failure.
- **Migration transforms**: rsx-let, resource-read, await-spawn,
  derive-clone, mut-binding, into-string, FnMut chains (param, call,
  capture), qualifier-aware fn detection, bracket-aware arg counting.
  Same-code batching with overlap guards, trigger selection across
  codes, anti-spin stall detection.
- **Proved live**: 1,142-line Dioxus 0.5 app migrated to green across
  9 repair rounds (10 + 84 errors → SUCCESS), then compiled through
  the full Android link.
- **Private delivery**: created repos are private; APK finder sees
  `target/dx/…` trees (512MB cap); release uploads get long windows;
  provisioner follows CDN redirects.

## v0.8.0 — Learning, translator, families (2026-09-25)

- **Learning to rank** (`src/engine/rank.rs`, `linfa` + `ndarray`,
  pure Rust): per-project `ranklog.jsonl` (plannability) and
  `outcomes.jsonl` (true fixes). Groups ordered by prediction, recipes
  chosen by learned P(fix); cold starts and ties fall back to
  deterministic order. The model proposes order — the compiler keeps
  every verdict.
- **Translator mode**: `build_page --prose "<prompt>"` — external model
  to intent metadata (validated, normalized, never code), key from
  `OPENROUTER_API_KEY` or config, honest refusal without one. Proven
  live end-to-end (prose → page).
- **Config family**: structs with `default()` from typed literals
  (bool/int/float validated, `String` gets `.to_string()` — the
  compiler caught the first draft).
- **Component family**: prop structs with `render()` over layout slots;
  bare `{name}` becomes positional `{i}` (named captures would bind
  locals — also caught by the compiler).
- **Kotlin consistency**: Gradle-cache stdlib must match the compiler
  major.minor, else fall back to the pinned provisioned set.
- 96 tests green, clippy `-D warnings` clean, fmt clean.

## v0.9.2 — Understander + messy-input hardening (2026-09-28)

- `engine/understand.rs`: verb/kind lexicons, typo-tolerant verbs
  (never names), frame dispatch, slot filling (quotes, titled names,
  adjacent nouns), constraint words, definable-kind guard, per-project
  translation history with overlap retrieval.
- `--understand` prints frame + confidence + precedent without touching
  anything; `--prose` records verified translations into history.
- Hardening from an 8-prompt messy battery (typos, vagueness, shouting,
  run-ons, gibberish): translator prompt refuses unasked replication,
  type normalization, kind coercion, op-slot completion, import targets.
- 120 tests green (65 lib), clippy clean, fmt clean.

## v0.10.2 — Curious research consortium (2026-09-28)

- `ResearchOracle::research_word`: Wikipedia summary API first, then
  DuckDuckGo search with page fetch — every summary carries its source.
- `html_to_text` strips scripts, styles, nav, headers, footers, asides
  (char-based, multibyte-safe, capped); DDG link extraction and wiki
  JSON parse unit-tested on canned inputs.
- Wired into `--understand --research` and the `--prose` refusal path:
  unknown words come back defined instead of merely questioned.
- 129 tests green (75 lib), clippy clean, fmt clean.

## v0.10.1 — No model anywhere (2026-09-28)

- Deleted the translator, its prompt, and every key: prose resolves
  through the understander or refuses with its receipt. Chat, UI, and
  examples all run model-free; `ui` feature still compiles clean.
- Below threshold the bot gets curious: unknown words return
  nearest-lexicon hypotheses plus verified history matches, and
  multi-request prose is detected and split by advice.
- 122 tests green (67 lib), clippy clean on default AND `ui`, fmt clean.

## v0.9.3 — Understand-first routing (2026-09-28)

- `--prose` runs the deterministic parser first; ≥0.75 confidence with
  a known frame executes with no model involved, the rest falls back.
  Content gate: parsable-but-unbuildable creates (no sections/fields)
  route to the model instead of a guaranteed block.
- Messy battery drove five robustness fixes: fuzzy verbs, adjacent +
  titled names (single resolution point), definable-kind guard,
  constraint words, type normalization, kind coercion, op-slot
  completion, import targets.
- 120 tests green (65 lib), clippy clean, fmt clean.

## v0.10.0 — LLM removed (2026-09-28)

- Deleted `src/llm.rs` outright: translator client, prompt,
  normalization, model config. No model dependency remains in any
  feature, default or otherwise.
- New `src/config.rs`: device-local settings only (project path,
  retries, GitHub token). Old config files still load.
- `scan_use_paths` moved into the engine (it was always pure
  byte-scanning, never intelligence).
- `gc chat`, the Chat tab, and `--prose` all run the deterministic
  understander; below-threshold parses become chat text or exit codes,
  never model calls. Settings screen lost its key/model fields.
- 119 tests green, clippy clean on default AND `ui` features, fmt clean.

## v0.9.1 — State machines + async tasks (2026-09-27)

- **State-machine family** (`kind: "statemachine"`): states plus
  `(event, from, to)` edges become `State`/`Event` enums and a machine
  with `new` (first state initial), a reader, and a total `transition`
  returning bool. Duplicate edges, unknown states, bad names, and
  smuggled methods/cases all block.
- **Async-task ops** (`due`/`mark`/`poll`) over exact clock fields
  (`last: std::time::Instant`, `interval: std::time::Duration`, full
  paths so bodies stay import-free). `poll` is a genuine `async fn`;
  the executor remains honestly out of scope.
- **Translator prompt** documents all kinds and ops, so prose can reach
  the new families without hand-written intents.
- 106 tests green (51 lib + 13 synthesize), clippy clean, fmt clean.

- Unknown identifiers met during repair are extracted, researched
  (canonical sources, compiler-verified), cataloged per-project, and
  utilized through `AddImport` — compiler judges next round as always.
- **Pathfinding** (`src/engine/pathfind.rs`): arena lookup plus locked
  dependency sources with layout-rebuilt module paths (incl. enclosing
  traits for method names). Proved live: unimported `greet()` from a
  path crate resolved to `helper::greet` and compiled clean.
- Dead ends carry a `RESEARCH_TRAIL` diagnostic (every source, hit,
  and miss) instead of a bare block.
- Fixed along the way: fabricated `use {bare_ident};` imports (matched
  call-site text, reported vacuous success), empty-import no-ops, nested
  crates.io JSON (versions read "unknown"), unverified flags on verified
  defs, and a `/dev/null` rustc output that sandboxes reject.
- Proved live: `serde_json` extracted → researched (v1.0.151, verified)
  → cataloged; missing-dep cases still block honestly (that's EnsureDep
  territory, not this loop's).

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
