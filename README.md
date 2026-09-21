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
CodeArena ──► CodebaseIndex ──► CodeVerifier
  │               │              (cargo check/test/clippy)
  │               │                    │
  │               ▼                    │ FAIL
  │          SymbolBinder          ▼
  │          (resolve refs)    ErrorCorrectionPipeline
  │                               (5-phase: classify → recipe → apply → verify → record)
  │                                   │
  │                              RecipeLog (error→fix DB)
  │                                   │
  └─────────── CodeWriter ◄──────────┘
             (compose from patterns)
```

## Grounding Coder 0.2 — Castle + Project Intelligence

Version 0.2 introduces the Castle + Project Intelligence architecture:

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

```bash
dx build --platform android --release --features ui
# APK: target/dx/gc/release/android/app/app/build/outputs/apk/debug/app-debug.apk
```

### Deliverables
- **GitHub:** https://github.com/Valwardon/grounding-coder
- **Release:** https://github.com/Valwardon/grounding-coder/releases/tag/v0.2.0
- **APK:** `app-debug.apk` (9.7MB) uploaded to release

### Architecture Changes

The Grounding Coder 0.2 implements a significant architectural upgrade:

**Before (v0.1.0):**
- Tiny intent schema with only goal, file, language, actions, references, define, imports, test
- ResearchOracle limited to external sources (docs.rs, Android SDK docs)
- No project context or capabilities awareness
- Single verification oracle (cargo check/test/clippy)

**After (v0.2.0):**
- Rich intent schema with platform, architecture, runtime, capabilities, domains, constraints, dependencies, and unknown_requirements
- CastleStore with compressed engineering memory
- Project workspace and manifest system
- Multiple platform-aware verification oracles
- Relevance-based knowledge retrieval
- Self-pruning knowledge base
