# Working contract — the same loop as the image pipeline

This repo builds a mind that works one way, and the code is asked to
work that way too. The image pipeline is the model: a source is
**researched**, **measured** into structured facts, those facts are
**committed** as durable, reviewable memory, and action is taken only
from the measured facts. Coding is the same loop in a second domain.

## The loop

1. **Research** — read the real source (files, upstream, docs, the
   plate). Never assume; the source is the authority.
2. **Measure** — turn it into facts with numbers: tests, greps, repros,
   probes. No opinions, no vibes.
3. **Commit** — write the facts to a reviewable file a diff can carry
   (`data/…`, the decision log). Memory lives on disk, not in a head.
4. **Act** — the smallest change the measurements license. No invention.

## No dead-end failures

Failure is never terminal. When measurement is insufficient, ambiguous,
or a fork appears: **do not guess and do not stop. Clarify with the
user, then repeat with the clarification.** Every refusal names what to
clarify and invites the next run. (A plate that is a scene, not a
bounded subject, asks which region is the subject — it does not just
stop.)

## Rules that fall out

- Subject knowledge is researched, never coded: no per-subject
  branches, no proportion tables, no anatomy. Same for code facts —
  they come from measurement, not from invention.
- Deterministic or it does not count: same facts → same output, byte
  for byte.
- Prove by viewing, then push. A result is not done until it is seen
  and committed.
- Keyless, provenance-carrying sources only; every fact names where it
  came from.
- The decision log in `data/decisions.jsonl` is the durable memory:
  one measured decision per line, with its evidence and alternatives.
- The lesson memory in `data/lessons.jsonl` is what the loop has
  learned, told as rules: each refusal names its lesson by key
  (e.g. `L-substantial`), `gc reflect`/`gc encounter --reflect` cites
  the filed rule behind each open question, and `gc lessons` prints
  them all with where they live in the code. A rule is only filed once
  its measurement is in; the tests verify the engine never cites a
  lesson that is not filed.
