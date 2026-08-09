# `recall init`: a guided first-install wizard

`design · 2026-08-09`

## Why

Today, choosing an embedder, a reranker, SPLADE, or the entailment judge means reading the README
and hand-editing flags or environment variables, and there is no path from "I just installed this"
to "this is calibrated against my corpus" that does not involve reading `docs/CALIBRATION.md` and
`docs/GENERATIONS.md` end to end. A guided command that asks a short set of questions, resolves
them into the components this codebase already ships, and drives the existing build and
calibration machinery removes that gap without inventing new retrieval code.

The two things this explicitly is not: it is not a new configuration system (it writes the `.env`
knobs the codebase already reads), and it is not a way to make calibration happen without labeled
data, calibration's requirements (`recall/calibration.py`) do not change, the wizard just makes it
obvious when they have not been met.

## Command

A new `recall init --corpus <path>` subcommand in `recall/cli.py`, backed by a new module
`recall/setup_wizard.py`. Additive: nothing about `index`, `search`, `generation`, or `calibrate`
changes for a user who never runs it.

```
recall init --corpus <path>
    │
    ├─ 1. Data-handling question        (sets cloud_allowed, filters everything after it)
    ├─ 2. Embedder choice                (+ API key prompt if a cloud provider is picked)
    ├─ 3. Reranker yes/no
    ├─ 4. SPLADE sidecar yes/no
    ├─ 5. Entailment judge yes/no
    ├─ 6. Optional: existing S3 manifest path   (enterprise branch trigger, see below)
    ├─ 7. Optional: labeled query file          (enterprise branch only, see Calibration below)
    │
    ├─ writes .env                       (behavior knobs, read by index/search from now on)
    ├─ writes .recall/init_receipt.json  (human-readable record, never read back by code)
    │
    └─ 8. Build sequence, branches on step 6:
         no manifest given  -> quick path:      schema apply -> index <corpus>, dev mode always
         manifest given     -> enterprise path:  generation build -> validate -> calibrate? -> promote?
```

The branch is decided by data (was an existing manifest supplied), not by a separate "which path"
question, an enterprise deployment already has an S3-hosted corpus and a manifest for it, a fresh
local install does not, and the wizard should not ask the user to name something they may not
know they need.

**The quick path never calibrates**, by design, not by omission. `recall calibrate`
(`recall/cli.py:706`) is bound to a generation id end to end through
`CalibrationRepository.calibrate`, there is no generation-free calibration entry point in this
codebase today, and `recall index` on a local filesystem path already refuses to run in production
(`cli.py:727`, "local filesystem indexing is development-only"). The quick path is a development
path by the codebase's own existing design, so the wizard does not pretend otherwise: it always
lands in `TrustPolicy.development()` and says so, pointing at the enterprise path for anyone who
wants a certified threshold. Building a generation-free calibration mechanism would be new scope
nobody asked for, see Non-goals.

## The data-handling question

One question, three answers, it sets `cloud_allowed: bool` and nothing else:

```
How should RE-call treat your data?
  [1] Fully local, never call an external API      (recommended for sensitive data)
  [2] Local by default, cloud allowed when it clearly helps accuracy
  [3] Cloud is fine, prioritize retrieval quality
```

`[1]` -> `cloud_allowed = False`, the embedder list that follows shows local profiles only.
`[2]` / `[3]` -> `cloud_allowed = True`, both show the cloud options, `[3]` changes only which
option is pre-highlighted as suggested. No answer here turns a component on by itself, every
component question is still asked and still defaults to off. This mirrors the standing doctrine
already in the README: "each option is enabled by name rather than inferred for you."

## Component questions

**Embedder.** The list is built from `embedding_registry.REGISTERED_PROFILES` filtered to
`not rejected` (this excludes the rejected Qwen3 profile automatically, the registry already
flags it), plus, if `cloud_allowed`, `voyage` and `openai-compat`. Recommended default:
`bge-small-asymmetric-v1` (local).

If a cloud provider is chosen, the wizard checks for its required env var
(`VOYAGE_API_KEY`, or `OPENROUTER_API_KEY` / `OPENAI_API_KEY`). If it is already set, it moves on.
If it is missing, it prompts for the key right there, input is not echoed to the terminal and the
value is never written to `init_receipt.json`, only straight into `.env`. An empty answer at that
prompt falls back to printing the variable name and stopping, rather than proceeding with no key
and failing later at index time. Writing a secret into `.env` matches existing practice in this
repo (`recall/_env.py`'s own docstring: "for local secrets... never committed"), it is not a new
exposure.

**Reranker.** Yes/no, off by default. `CrossEncoderReranker` is the only real option today, so
"yes" means exactly that.

**SPLADE sidecar.** Yes/no, off by default, one line on cost (an extra local model) and the
measured R@100 gain from `docs/superpowers/specs/2026-08-06-learned-sparse-splade-design.md`.

**Entailment judge.** Yes/no, off by default, local QNLI cross-encoder
(`QnliEntailmentJudge`), there is no cloud judge in this codebase to choose between.

**Existing S3 manifest** (enterprise branch trigger). A path to a manifest JSON already built with
`recall manifest create`, or blank. The wizard does not upload a corpus to S3, no such helper
exists in this codebase (`recall/manifest.py` only reads), asking for one it cannot fulfil would be
worse than not asking. Blank means quick path, and the flow skips straight to `schema apply ->
index <corpus>` with no further questions.

**Labeled query file** (enterprise branch only, asked immediately after a manifest path is given).
A path to a small JSON file of `{"query": ..., "answerable": true|false}` entries, or blank. Blank
means the generation is built and validated but calibration is skipped, not attempted with
fabricated data, see Calibration outcomes.

## Persistence

`.env`, reusing `RECALL_RERANK` (`recall/profiles.py` already reads this exact variable, no reason
to invent a second name for the same switch), adding `RECALL_EMBEDDER_PROFILE` (a registry profile
id) or `RECALL_EMBEDDER_PROVIDER` (`voyage` / `openai-compat`), `RECALL_SPLADE`, `RECALL_ENTAIL`.

`.recall/init_receipt.json` is a plain audit record, what was chosen and when, and it is never read
back by any code path. That is deliberate: a config source that both drives behavior and gets read
back for its own bookkeeping is exactly the two-copies-of-one-fact failure mode
`embedding_registry.py`'s module docstring describes at length, elsewhere in this codebase. The
receipt exists for the human, not the program.

## Live wiring

A new `resolve_configured_embedder()` in `recall/setup_wizard.py` (or `recall/cli.py`, final
location is an implementation detail), modeled directly on the existing
`resolve_retrieval_profile()` in `recall/profiles.py`: read the env, resolve to a concrete object,
let an explicit flag override it. `index`, `search`, and `demo` call this instead of today's
hardcoded `hashing` / `fastembed` choice in `_make_embedder`.

For a local profile this resolves to a plain `FastEmbedEmbedder(model_name=..., asymmetric=...)`,
the same legacy, no-digest-pinning path the Quickstart already exercises, not the registry's
`.build()` (which demands a pre-provisioned artifact digest, a deliberately stricter contract that
belongs to the enterprise generation path, not this one). For a cloud profile it resolves to
`VoyageEmbedder()` or `OpenAICompatEmbedder()`, reading the key the wizard already placed in
`.env`.

## Calibration outcomes

Applies to the enterprise branch only, the quick path always ends in `TrustPolicy.development()`
(see the branch note above) and states that plainly, with no further outcome to report. On the
enterprise branch, three end states, always stated in plain text, never silently downgraded:

1. **Calibrated and promoted.** A query file was given, separability cleared the 0.90 bar
   (`recall/calibration.py:MIN_SEPARABILITY`), the threshold is published and the generation is
   promoted.
2. **Attempted and refused.** A query file was given but separability came in under the bar, the
   generation stays validated but unpromoted, and the actual measured AUC is printed, not just
   "failed."
3. **Skipped.** No query file was given, the generation stays validated but unpromoted, and the
   exact follow-up command (`recall calibrate --generation ... --queries ... --publish`, then
   `recall generation promote ...`) is printed so the gap is a known next step, not a silent one.

## Non-goals

- No non-interactive / scripted mode (`--yes`, flag-only answers) in this pass. The wizard is
  interactive-only; scripted installs still use the existing flags and env vars directly. Worth
  revisiting once the interactive flow is proven.
- No upload-to-S3 helper. The enterprise branch requires a manifest the operator already built.
- No new cloud reranker or cloud judge. Neither exists in this codebase today, the wizard offers
  what is real.

## Testing

- Unit tests for the question-to-config mapping and the cloud-allowed filtering, fed a canned
  sequence of answers, no terminal or network needed.
- An integration test running the full quick path against the real test Postgres
  (`RECALL_TEST_DSN`) with a throwaway corpus, asserting it lands in `TrustPolicy.development()`
  and that `.env` / `init_receipt.json` carry the chosen components.
- An integration test running the enterprise branch against a fixture manifest, once with a small
  labeled query file (outcome 1, calibrated and promoted) and once without one (outcome 3,
  validated but unpromoted with the follow-up command printed).
- A test that the enterprise branch, given no manifest, never triggers, quick path runs with no
  further questions asked.
