# grounding-coder

A **deterministic, non-guessing coding agent — in any language it can check**.
The LLM is only an unverified natural-language→intent translator. All actual
coding, verification, and correction is driven by a deterministic engine:
no guessing, no hallucination.

## Core Principle

> The compiler is the oracle. Tests are the truth. The codebase is the answer key.

The LLM generates hypotheses (intent). The deterministic engine verifies every
hypothesis against hard oracles (compilers, interpreters, test runners,
structure checkers) and corrects failures using a recipe log — never guessing.
Anything it cannot prove, it refuses honestly (`BLOCKED`) with the file left
byte-identical.

## Architecture

```
Human (messy language)
  │
  ▼
LLM via OpenRouter ──── UNVERIFIED ────► structured intent JSON
  │                                        (metadata + literal values only,
  │                                         NEVER code)
  ▼
Intent validation + normalization
  │
  ▼
Research / resolve symbols
  │
  ▼
Language backend (Rust / Python / C / Kotlin / JS / HTML / *.grounding.toml)
  │
  ▼
Deterministic EditPlan
  [exact file] [exact byte range] [expected old text] [new text] [evidence]
  │
  ▼
Global snapshot (all planned files + all sources)
  │
  ▼
Apply ONLY precise edits (stale-check every range)
  │
  ▼
Oracle verify (compiler / tests / parser per language)
  │
  ├── clean → commit
  │
  └── dirty → CorrectionPipeline (bounded) → re-verify
                ├── fixed → commit
                └── unfixable → ROLLBACK EVERYTHING → BLOCKED honestly
```

## Critical Invariant

> The agent may only modify bytes identified by a structured edit whose
> target and replacement are justified by project state, compiler
> diagnostics, or a verified recipe. LLM-provided code is rejected on sight.

## What It Can Do (proven by tests)

- **Real functions from contracts** — `word_counts(text) -> HashMap<String, usize>`
  synthesized from input/output examples, verified by compiler + generated
  contract tests. (`tests/synthesize.rs`)
- **Stateful structs** — `Counter` with `new`/`incr`/`incr_by`/`value` built
  from a verified method-op vocabulary; multi-step state contracts pass.
- **Fallible functions** — `parse_port(s) -> Result<u16, String>` with `Ok`
  and `Err` cases both verified.
- **Multi-file atomic features** — new `src/geometry.rs` module created and
  `mod geometry;` wired into `lib.rs` in one transaction. If any task fails
  after others applied, every file rolls back byte-identical.
- **Any language with a checker** — Rust, Python (`py_compile`+pytest), C
  (`gcc`), Kotlin (`kotlinc` compile + JVM run of contracts), JavaScript
  (registry data + `node --check`), HTML (tag-balance oracle), Go (honest
  `GO_MISSING` without toolchain), plus any language at all via a project
  `.grounding.toml` `[language]` table.
- **Tool provisioning** — missing compilers are fetched pinned-by-SHA256
  from Maven Central into a shared cache (Gradle cache reused when present),
  like pip fetching a build backend. Corrupt bytes are deleted, never run.
- **Kotlin synthesis** — `RandomNumberGenerator` over `kotlin.random.Random`
  from constructor/delegation metadata, judged by `check()` contracts run
  on the JVM. Proved live: `Random(42).nextInt() == 972016666`.
- **Web pages from content slots** — title/sections/footer as pure data,
  engine-owned escaped template, parser oracle. Hostile
  `<script>alert(1)</script>` content renders as inert text.
- **External crates with registry evidence** — `rand::…` imports trigger a
  crates.io lookup; hits pin versions into `Cargo.toml` as evidence-backed
  edits, then imports resolve. Unknown crates block with disk untouched.
- **Real RNG struct** — `RandomNumberGenerator { rng: StdRng }` with a
  verified `next_u32` delegation body, judged by a fixed seeded contract.
- **Honest Partial outcomes** — hosts without a toolchain (Android) get
  `syn` parse verification and a labeled PARTIAL instead of a false
  SUCCESS or a useless BLOCKED.
- **GitHub actor** — "create a repo called X / upload to github" creates
  the repo and pushes project files from Chat with the Settings token.
  "Release … attach the apk" cuts the tag release and uploads project
  APKs as assets. Delivery runs on every terminal outcome (even BLOCKED)
  and never flips it; failures report plainly.
- **Clean-room authorship (default)** — intents carrying `files[]`
  replication manifests are refused *before any fetch* with
  `BLOCKED (CleanRoom)`, disk untouched. Every byte on disk is then
  authored by synthesis + verified transforms. Replication of owned
  templates requires explicit opt-out (`set_clean_room(false)` /
  `GROUNDING_ALLOW_REPLICATION=1`). (`tests/replicate.rs`)
- **Generalized Build op** — `TaskKind::Build` ("build …" actions) runs
  the language backend's build oracle: C/`gcc`, Rust/`cargo` (native,
  release, explicit triples), Kotlin/`kotlinc` jars, Java/`javac`,
  Android APKs via provisioned `dx`. Proof is the artifact's own bytes
  (present + non-empty); unknown languages/targets block honestly.
  (`tests/build.rs`: C hello builds *and runs*, Java classfiles, Rust
  bins, unknown-target block with no artifact)
- **Self-solving drivers** — every build driver is verified by execution
  (`--version` runs) with an ordered fallback chain: operator override
  → pinned hash-verified download → source build from the release tag
  (clone → oracle-driven `cargo update` repair loop → build → verify).
  Full diagnosis trails on failure, never a blind retry.
  (`src/engine/lang.rs`: `ensure_dx`, `driver_runs`, `failing_crates`)
- **Migration repair loop** — compiler-anchored structural rewrites
  (Dioxus 0.5→0.7 idioms and friends): rsx-let wraps, resource reads,
  await-spawn splits, `Clone` derives, `mut` bindings, `FnMut`
  relaxation chains, string-ambiguity fixes. Same-code batching with
  overlap guards, trigger selection across codes, anti-spin stall
  detection. Proved live: a 1,142-line app migrated to green.
- **Learning to rank (`linfa`)** — repair history trains a decision tree
  per project (`.grounding/ranklog.jsonl` + `outcomes.jsonl`,
  clean-room scoped). Groups are tried in predicted-plannability order;
  recipes are chosen by learned P(fix). ML proposes *order* —
  the compiler keeps every verdict. Cold starts and ties fall back to
  appearance/priority order. (`src/engine/rank.rs`)
- **True-fix outcomes** — every applied recipe is judged against the
  next verify (error gone or not), so the model learns what *worked*,
  not what was *busy*.
- **Translator mode** — `build_page --prose "<prompt>"` wires the
  external model (prose in, metadata out, never code) into the
  deterministic engine. Key from `OPENROUTER_API_KEY` or config; no key
  means an honest refusal, never a guess.
- **Config + component families** — `kind: "config"` synthesizes structs
  with a `default()` constructor from typed literals (bool/int/float
  validated, `String` correctly gets `.to_string()`); `kind:
  "component"` synthesizes prop structs with `render()` over layout
  slots (`{name}` becomes positional `{i}`, unknown slots/braces and
  non-Display props block). (`tests/synthesize.rs`)
- **Private GitHub delivery** — created repos are private; the APK
  finder sees into `target/dx/…` output trees (up to 512MB); release
  uploads get an 80-minute window for slow uplinks; the provisioner
  follows release-CDN redirects.
- **Kotlin toolchain consistency** — Gradle-cache kotlinc pairs only with
  a major.minor-matching stdlib, else falls back to the pinned
  provisioned set (a newer cached stdlib once broke metadata compat).

## Proof, Not Promises

96 tests, all green (`cargo test`), plus `cargo check`, `clippy -D warnings`,
`fmt --check` clean:

| Suite | Tests | What it proves |
|---|---|---|
| lib (unit) | 44 | recipes, ranker, parsers, manifests, guards |
| `build` | 6 | C/Java/Rust real builds + artifacts run; unknown targets block; dx-output APK discovery |
| `replicate` | 6 | byte-exact replication, hash-mismatch block, replace-exact rules, clean-room refusal + opt-out |
| `editplan` | 4 | byte-range apply, stale rejection, invalid-range rejection, LLM-code rejection |
| `end_to_end` | 2 | boring import path commits; unknown symbols block honestly |
| `synthesize` | 10 | word_counts, Counter struct, parse_port, new-module wiring, cross-task rollback, unknown-op block, unsupported-shape block, config defaults, component slots, unknown-slot block |
| `polyglot` | 5 | Python import, unknown-import block, registry-JS import, TOML-defined language, missing-toolchain block |
| `webpage` | 4 | homepage build, script-escape safety, empty-title block, strict-intent page injection |
| `repair` | 2 | missing-import repair + compile, unfixable block with untouched disk |
| `rng` | 2 | registry-crate RNG struct end to end, bogus-crate clean block |
| `progress` | 2 | ordered stage trail, silence without listener |
| `config` | 2 | config path resolution, save/load roundtrip |
| `fuzz` | 4 | deep JSON, unicode, hostile ranges, garbage intents |
| `kotlin` | 3 | RNG compiled+run on JVM, unknown-call block, pinned toolchain provisioning |

### The dentist test

A random challenge — *"a home page for a dentist office in Miami, Florida,
in HTML"* — produced `/tmp/dentist-site/index.html` via `SUCCESS`: Bright
Smile Dental with services, Little Havana blurb, phone, and footer address,
structure verified by the HTML oracle. No Rust was involved at any step:
intent → page family → escaped template → parser verify → commit. That run
is what proved the engine is a coding bot, not a Rust bot.

Run it yourself:

```bash
cargo run --example build_page -- /tmp/dentist-site /tmp/dentist-intent.json
```

## Language Backends

Per language there are exactly three things — edit conventions, an oracle
command, a std inventory — behind the `LanguageBackend` trait
(`src/engine/lang.rs`). New languages arrive as:

1. a hand-tuned struct (rich multi-stage oracles: Rust, Python, C, HTML), or
2. pure data — a registry entry, or a project `.grounding.toml`:
   ```toml
   [language]
   name = "zz"
   extensions = ["zz"]
   import_template = "need {p};"
   import_prefixes = ["need "]
   std_exact = ["std/io"]
   verify_cmd = ["zzc", "--check"]
   ```
Unknown languages with no spec refuse honestly instead of guessing.

## Build & Run

```bash
cargo build
cargo run --bin gc -- --help
cargo run --bin gc -- chat "Add HashMap support to src/lib.rs" --project /tmp/demo
```

## Android APK

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

## Quality Gates

```bash
cargo check --all-features
cargo clippy --all-targets --all-features   # zero warnings
cargo fmt --check
cargo test --all-features
```

## Learning (ML proposes, compiler disposes)

Three slots where machine learning is allowed — none of them can forge
evidence:

1. **Translator** (external model): prose → intent metadata. Never code.
2. **Ranking** (`linfa` decision trees, pure Rust, trains in
   milliseconds): which error group and which recipe to try first, learned
   from the project's own judged history. Order only.
3. **Retrieval** (planned): precedent search over verified outputs.
   Lexical/statistical first (`vtext`-shaped), embeddings (`tract`) later.

Refused: genetic search over code (guessing with extra steps), RL policies
(wrong data regime), anything with native dependencies beyond the existing
toolchain. `linfa` + `ndarray` are the only ML crates; both pure Rust.

## Components

- `src/engine/synthesize.rs` — deterministic synthesizer: verified ingredient
  index, composition families (map-accumulation, struct ops, fallible parse,
  web pages, config defaults, component render), contract-test renderer
- `src/engine/lang.rs` — language backends: trait + Rust/Python/C/Kotlin/HTML
  impls + data-driven registry + `.grounding.toml` loading; `build()` oracles
  (gcc, cargo, kotlinc/javac, provisioned dx) with self-solving drivers
- `src/engine/rank.rs` — learning-to-rank: per-project rank/outcome logs,
  linfa trees, cold-start fallbacks
- `src/engine/plan.rs` — `EditPlan` with `expected_old` stale-checks
- `src/engine/writer.rs` — planner-only `CodeWriter`, module wiring,
  creation-safe snapshots
- `src/engine/mod.rs` — plan-all → snapshot-all → execute transaction loop
- `src/engine/corrector.rs` — migration transforms with same-code
  batching, trigger selection, anti-spin stall detection, outcome judging
- `examples/build_page.rs` — drive the engine as a library, now with
  `--prose "<prompt>" translator mode (`OPENROUTER_API_KEY` or config key)
- `src/oracle/` — knowledge adapters (Rust, Android, Solana, GitHub)
- `src/knowledge/` — CastleStore engineering memory

## Deliverables

- **GitHub:** https://github.com/Valwardon/grounding-coder
- **Release:** https://github.com/Valwardon/grounding-coder/releases/tag/v0.2.0
- **APK:** `grounding-coder.apk` (arm64) uploaded to release
