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

## Build & Run

```bash
cargo build
cargo run --bin gc -- --help
```
