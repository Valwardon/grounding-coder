Messy-intent sample receipts for the v0.25.1 understander hardening
(`src/engine/understand.rs`, battery in `tests/messy_intent.rs`).

Generated live, model-free, via:

  cargo run --example build_page -- /tmp/gc-sample-probe --understand "<prompt>"

1. samples/intent/macaroni-hat.json
   Prompt: "A hat made of macaroni." (one material among many —
   straw, glass, steel parse identically; see tests/messy_intent.rs.)
   Material relation banks `material:macaroni`, references hat+macaroni,
   and a 3-question research plan (base shape, material geometry,
   placement). Disposition: Ask.

2. samples/intent/webpage-synonym.json
   Prompt: "Create a webpage called Home with \"welcome back\"."
   Everyday spelling `webpage` normalizes to kind `page`; confidence 1.0.
   Disposition: Execute.
