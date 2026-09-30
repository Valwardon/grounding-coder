# grounding-coder

A **deterministic, non-guessing coding agent — in any language it can check**.
There is no language model anywhere in the system: prose is understood by a
built-in lexical parser, and all coding, verification, and correction is
driven by a deterministic engine. No guessing, no hallucination, no API keys
for intelligence — none needed.

## Core Principle

> The compiler is the oracle. Tests are the truth. The codebase is the answer key.

The deterministic parser turns prose into intent (with a confidence receipt).
The deterministic engine verifies every hypothesis against hard oracles
(compilers, interpreters, test runners, structure checkers) and corrects
failures using a recipe log — never guessing.
Anything it cannot prove, it refuses honestly (`BLOCKED`) with the file left
byte-identical.

## Architecture

```
Human (messy language)
  │
  ▼
Lexical parser (verbs, kinds, slots) ──► structured intent JSON
  │                                        (metadata + literal values only,
  │                                         NEVER code; confidence receipt
  │                                         attached, below-threshold
  │                                         parses refuse)
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
> diagnostics, or a verified recipe. There is no model to provide code;
> the concept does not exist in this codebase.

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
- **Idle learning loop** — `gc dream` turns finished tasks into
  questions (`extract_gaps`), investigates them on spare cycles, and
  promotes only what verifies. Knowledge carries an explicit
  lifecycle (Unknown → Question → Hypothesis → Experiment → Evidence
  → Verified → Generalized); DREAM ≠ KNOWLEDGE is enforced by the
  store — promotion without verification evidence is refused, and
  invalid generalizations are Rejected, never re-learned.
  (`tests/knowledge_loop.rs`)
- **Compiler-probe evidence** — code claims verify by experimental
  demonstration, not documentation: the smallest program exhibiting
  the behavior runs against the real toolchain oracle in a
  disposable scratch project under a real timeout. Green probes
  promote concepts to Verified with the oracle transcript as
  evidence;   red, uncompilable, or timed-out probes record negative
  evidence and never promote. (`src/engine/probe.rs`)
- **Deterministic eyeballs** — the raytracer builds worlds (tower,
  ground, sky, props), never bodies: "put a tower behind me" is a
  graph edit whose effect is proven by per-material pixel receipts.
  Classical perception without any model: hand-rolled BMP codec,
  crop/resize/overlay, histograms, Sobel edges, template matching —
  every op exact and repeatable. People come from researched
  photographs with ground-truth boxes. (`src/engine/vision.rs`,
  `src/engine/scene.rs`, `gc render`; sample photos in `samples/`)
- **Self-sourced plates** — the loop finds its own base photos:
  Commons search or general web image search (`--source web` via
  DDG), complete provenance required (source, author, allowlisted
  license — incomplete records refuse with the reason), fetch,
  decode via the `image` crate, BMP plus manifest carrying the
  license basis for audit. (`src/engine/plates.rs`, `gc plate`;
  first plate in `samples/plates/` with full provenance)
- **Photographic portraits from plates** — classical segmentation
  (skin locus + luminance floor, morphology, largest blob with a
  compactness gate), keep-aspect framing on a studio backdrop,
  feathered edges, matched finish. Refuses implausible masks with
  the reason instead of compositing furniture. Every output pixel
  traces to its source plate via the op log.
  (`src/engine/compose.rs`, `gc compose`;
  `samples/photo-composite-640x800.bmp` from a CC0 plate)
- **Queryable anatomy, no vectors** — a researched partonomy graph
  (bones, vessels, relations: part-of, articulates, supplied-by)
  answers "vessels near the distal phalanges" by graph walk, with
  Gray's Anatomy citations; joint ROM tables validate poses instead
  of imagining them, with hair and physique notes alongside.
  (`src/engine/anatomy.rs`)
- **Imagine** — one command from prose to photo: brief, reference
  research across Commons, web search, and Open Images (adult-
  filtered ground-truth boxes), palette/composition study, then a
  two-source montage of real people in real places. Refuses rather
  than meshing block figures. (`src/engine/imagine.rs`,
  `gc imagine`; samples in `samples/`)
- **Family restoration** — dust and scratch detection with
  classical operators, Laplacian inpainting, full defect reports.
  Your photos edit with no refusal path anywhere.
  (`src/engine/restore.rs`, `gc restore`)
- **Genome humans** — synthetic adults from validated parameters,
  never likenesses: parametric head meshes, smile blendshapes,
  evidence per property, `#[derive(GroundingType)]` schemas, ray-
  traced with skin wrap. (`src/engine/{mesh,human,figure}.rs`,
  `grounding-macros`, `gc figure`)
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
- **Prose without a model** — `build_page --prose "<prompt>"` resolves
  prose through the deterministic understander and runs the engine on
  it at ≥0.75 confidence; below that it refuses with the receipt.
  Verified translations record into per-project history.
- **Understanding without a model** — `build_page --understand
  "<prompt>"` parses prose with a hand-built lexicon (verbs, kinds,
  typo tolerance on verbs only), grammar frames (build/create/fix/
  publish/verify), and slot filling (quoted literals, titled names,
  adjacent nouns — never invented). Prints intent + confidence receipt
  + precedent; read-only, nothing touched. Below threshold it gets
  curious instead of silent: unknown words come back with
  nearest-lexicon hypotheses, verified history matches, and
  multi-request detection. (`src/engine/understand.rs`)
- **Messy-input hardening** — everyday spellings normalize to canonical
  types (`string`→`String`, mirroring rustc defaults; foreign
  spellings pass through); sloppy kinds coerce by shape; `new`/`get`/
  `add_assign` complete missing-but-determined slots; imports carry
  their target file. Each gap found by feeding the bot typos, and each
  fix proven by a test that fails without it.
- **Config + component families** — `kind: "config"` synthesizes structs
  with a `default()` constructor from typed literals (bool/int/float
  validated, `String` correctly gets `.to_string()`); `kind:
  "component"` synthesizes prop structs with `render()` over layout
  slots (`{name}` becomes positional `{i}`, unknown slots/braces and
  non-Display props block). (`tests/synthesize.rs`)
- **State-machine family** — `kind: "statemachine"` turns states plus
  `(event, from, to)` edges into `State`/`Event` enums and a machine
  struct with a total `transition()` (wildcard arm, so exhaustiveness
  holds by construction). Duplicate edges, unknown states, smuggled
  methods/cases all block.
- **Async-task ops** — `due`/`mark`/`poll` over exact `last:
  std::time::Instant` + `interval: std::time::Duration` fields (full
  paths, no imports needed). `poll` is a real `async fn`; anything
  else shaped refuses.
- **Private GitHub delivery** — created repos are private; the APK
  finder sees into `target/dx/…` output trees (up to 512MB); release
  uploads get an 80-minute window for slow uplinks; the provisioner
  follows release-CDN redirects.
- **Kotlin toolchain consistency** — Gradle-cache kotlinc pairs only with
  a major.minor-matching stdlib, else falls back to the pinned
  provisioned set (a newer cached stdlib once broke metadata compat).
- **Receive → catalog → utilize** — when repair meets an unknown
  identifier, the bot walks from A to B, cheapest source first: the
  compiler's suggestion, the arena, dependency sources (locked deps,
  paths rebuilt from file layout), then the network oracle — banking
  hits in the per-project catalog (`.grounding/catalog.jsonl`,
  reloaded into the symbol table on bootstrap) and applying through
  the normal `AddImport` plan. Bounded (3 network attempts per run,
  each symbol once); every step lands in a research trail attached to
  dead ends. (`src/engine/catalog.rs`, `src/engine/pathfind.rs`)

## Proof, Not Promises

267 tests, all green (`cargo test`), plus `cargo check`, `clippy -D warnings`,
`fmt --check` clean:

| Suite | Tests | What it proves |
|---|---|---|
| lib (unit) | 166 | recipes, ranker (importance-weighted), catalog, pathfind, parsers, manifests, guards, families, understander (slot satisfaction, vector synonyms, kind synonyms, material relations), scene-intent IR (lexicon, spec, planner, capabilities), disposition, web research, knowledge lifecycle (consolidation, evidence tiers), compiler probes, deterministic vision (pixels, film finish, masks, compositing, collages, roses, shadows, photo-first, ground-truth boxes, inpainting, restoration), plate sourcing + provenance (Commons, web search, Open Images), backend affordances, imagine trajectory, movement knowledge, genome humans (mesh, morphs, evidence, figures, type schemas) |
| `intent_benchmark` | 13 | happy path, typo tolerance, head-noun kind rule, typo-verbs-never-names, vague/ambiguous refusal, destructive block (even fully specified), contradiction ask, multi-intent split, unknown-concept curiosity |
| `messy_intent` | 10 | kind synonyms (webpage→page), plurals, typo'd kinds with verbatim names, verb inflections, visual verbs (imagine→create), material relations (macaroni-hat research plan), glossary, destructive-still-blocks |
| `photo_gate` | 6 | visual-prose routing (imagine/render/draw→create Ask), genome-waver human pixels, demo-world receipt + sky/ground pixel samples, byte-identical re-renders; photos commit under `samples/` |
| `scene_intent` | 11 | scene-requirements IR (subjects/actions/objects/materials/unresolved/confidence), per-requirement research queries, honest capability verdicts (pose/cloth missing, procedural humans deleted); macaroni binds as material, bare waving stays ambiguous |
| `knowledge_loop` | 3 | budget-bounded offline dreaming with persistence, sleep when nothing open, failures deprioritize without deleting |
| `learning_experiment` | 2 | dreaming arm rediscovers family patterns (rate 0.55) vs amnesiac baseline (0.00); both arms solve every task |
| `build` | 6 | C/Java/Rust real builds + artifacts run; unknown targets block; dx-output APK discovery |
| `replicate` | 6 | byte-exact replication, hash-mismatch block, replace-exact rules, clean-room refusal + opt-out |
| `editplan` | 4 | byte-range apply, stale rejection, invalid-range rejection, LLM-code rejection |
| `end_to_end` | 2 | boring import path commits; unknown symbols block honestly |
| `synthesize` | 13 | word_counts, Counter struct, parse_port, new-module wiring, cross-task rollback, unknown-op block, unsupported-shape block, config defaults, component slots, unknown-slot block, statemachine transition, statemachine refusal, asynctask scheduler |
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

Two slots where machine learning is allowed — neither can forge evidence:

1. **Ranking** (`linfa` decision trees, pure Rust, trains in
   milliseconds): which error group and which recipe to try first, learned
   from the project's own judged history. Order only.
2. **Retrieval**: precedent search over verified outputs — lexical
   history matching today (`translations.jsonl`), statistical
   (`vtext`-shaped) next, embeddings (`tract`) later.
3. **Open-web research**: Wikipedia summaries, DuckDuckGo search, and
   chrome-stripped page text for words the parser can't place —
   provenance attached, misses honest. (`ResearchOracle::research_word`)

The old third slot — an external model translating prose — is gone,
replaced by the deterministic understander below. Refused: genetic
search over code (guessing with extra steps), RL policies
(wrong data regime), anything with native dependencies beyond the existing
toolchain. `linfa` + `ndarray` are the only ML crates; both pure Rust.

## Understanding (no model)

Prose reaches the engine through `src/engine/understand.rs` — a
hand-built verb/kind lexicon, typo tolerance on verbs only (names pass
through verbatim), grammar frames (build/create/fix/publish/verify),
and slot filling from quotes, titled names, and adjacent nouns.
Kinds forgive too: everyday synonyms (`webpage`→`page`,
`options`→`config`, `func`→`function`), plurals, and near-miss
spellings (`stuct`→`struct`), while names still pass through verbatim.
Material relations parse as construction requirements — "a hat made of
macaroni" banks `material:macaroni`, references both concepts, states
seed glosses (hat: headwear; macaroni: pasta), and files a 3-question
research plan (base shape, material geometry, placement) instead of
dropping the words. WordNet / Wikidata / Wikipedia stay external
research sources (`ResearchOracle::research_word`); the built-in tables
are auditable seeds, never copies. Sample receipts live in
`samples/intent/`, battery in `tests/messy_intent.rs`.
Every parse carries a confidence receipt (quoted slots solid, inferred
slots soft); below 0.75 the system states what's missing instead of
acting. Verified translations accumulate per project and serve as
lexical precedent. `build_page --understand "<prompt>"` prints the
parse without touching anything; `--prose` runs the engine on it when
confidence clears, and refuses with the receipt when it doesn't.

Refused: genetic search over code (guessing with extra steps), RL policies
(wrong data regime), anything with native dependencies beyond the existing
toolchain. `linfa` + `ndarray` are the only ML crates; both pure Rust.

## Components

- `src/engine/synthesize.rs` — deterministic synthesizer: verified ingredient
  index, composition families (map-accumulation, struct ops, fallible parse,
  web pages, config defaults, component render, state machines, async-task
  ops), contract-test renderer
- `src/engine/catalog.rs` — receive/catalog/utilize: unknown-ident
  extraction, per-project researched-definitions log, symbol-table
  indexing on bootstrap
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
