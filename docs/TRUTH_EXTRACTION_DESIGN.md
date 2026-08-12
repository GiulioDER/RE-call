# Truth extraction and reviewed rewrites: design, and the consolidation it starts from

Status: **design approved, not yet implemented.** The unusual part of this document is its
starting point. The feature described here has already been built three times, on three branches,
none merged. The first deliverable is therefore not code but a decision about which of the three
survives, and that decision is recorded here so the next session does not build a fourth.

## The problem

RE-call's model of truth is authored frontmatter, and exactly three keys are recognised
(`recall/frontmatter.py:12`): `supersedes`, `valid_from`, `valid_until`. The trust layer acts on
those and nothing else. Prose is retrieved, never interpreted.

On a real 792 memo corpus, **60 memos stated a supersession in prose against 2 declared
frontmatter edges** (`recall/fix.py:4`). Each undeclared relation is a stale memo served with
`verdict == "ok"`.

## The measured prior, which is a failure

The rule based attempt exists and its docstring is the most important prior art in the repo.
`recall/fix.py:10`, verbatim:

> "Measured on that corpus, it proposes ZERO edges. Four survived the mechanical rules and all
> four were wrong on review"

The four failures were reported speech, superseding a claim *inside* the target (twice), and
hedging. Every one of those is a semantic distinction: narrating versus declaring, part versus
whole, qualifying versus committing. No pattern over the text can see any of them.

That is the entire argument for a model here. It is also the reason the refusal rules are
restated in the prompt rather than assumed, and the reason the model's output is put through a
validation ladder rather than trusted.

## What already exists

| Branch | Commits | Scope |
|---|---|---|
| `claude/suspicious-ardinghelli-3d3e31` | 9 | Extraction only. A `recall/truth_extraction/` package (engine port, prompt, cache, normalize) with a refusing validation ladder, wired into `reasoning_proposals/_extracted.py`. 1037 line contract test. No CLI, no write path. |
| `claude/truth-extraction-prose-a7834c` | 5 | Write path only. `rewrite.py` (633 lines) with a 1502 line contract test, plus byte level `frontmatter.py` helpers and `atomic_write.py`. Four of its five commits are fixes for CommonMark fence info strings, BOM and CRLF. |
| `claude/truth-extraction-prose-1cfc81` | 3 | Both halves plus a CLI, but shallower on each. `extraction.py` (529), a rival `rewrite.py` (551), 272 lines of CLI. |

Roughly 8,000 lines. **Zero merged to master.** None of the three touches `recall_mcp/`.

`1cfc81` and `a7834c` converged independently on the same write path design: a delimited derived
block for relations the frontmatter has no vocabulary for, a SQLite rejection sidecar, a
`plan`/`apply` split, and a `PromotedFact` type gate. That convergence is treated here as
evidence the design is right, not as a coin flip.

## The consolidation

`a7834c` and `ardinghelli` touch **entirely disjoint file sets**. `git merge-tree --write-tree`
over the pair returns a clean tree. Both therefore land as real merges, preserving both
histories.

| Source | Contributes | Mechanism |
|---|---|---|
| `a7834c` | `rewrite.py`, `atomic_write.py`, byte level `frontmatter.py` helpers, `fix.py` fixes, `test_corpus_rewrite_contract.py` | `git merge` |
| `ardinghelli` | `recall/truth_extraction/`, the ladder, `reasoning_proposals/_extracted.py`, proposal schema v2 | `git merge` |
| `1cfc81` | the OpenAI compatible chat client and the recheck report, extracted from `extraction.py` | port one module |
| new | the verb CLI, the MCP surface, the end to end chain | write |

`1cfc81` is **not** merged wholesale. Its `rewrite.py` and `a7834c`'s are rival implementations of
one design, so taking both means resolving a conflict in order to then discard one side.
`a7834c`'s has five commits of fixes and three times the contract test.

Its `promoted_prose_edge` and `prose_edge_proposal_id` helpers are also dropped. They hand roll a
shortcut past `promotion.py`, and giving `promotion.py` a genuine first caller is a stated goal of
this work rather than an obstacle to it.

## Architecture

### Extraction is a port, and the model is one implementation of it

`ardinghelli` defined `ExtractionEngine` as a Protocol whose entire contract is
`run(prompt) -> str`, resolved from a `_ENGINES` registry through
`RECALL_TRUTH_EXTRACTION_ENGINE`, shipping OFF by default and mirroring
`resolve_entailment_judge`.

The model backed extractor is therefore a **new entry in that registry**, not a new architecture:

```
recall/truth_extraction/_openai_engine.py   ->  _ENGINES["openai"]
```

Whatever the model returns still passes the full ladder, unchanged:

**Batch rungs** (refuse the file's whole output): `json`, `top_level_shape`, `max_claims`,
`claim_shape`.
**Claim rungs** (refuse one claim, keep the rest): `quote_not_verbatim`, `quote_is_frontmatter`,
`target_not_in_corpus`, `date_not_in_body`.

The model gains no ability to skip a rung. A truncated batch is never returned in place of a
complete one: exceeding `MAX_CLAIMS_PER_FILE` (12) refuses the file rather than trimming it,
because a truncated batch is indistinguishable from a complete one downstream.

The deterministic rules engine stays. It is what makes the pipeline testable end to end with no
model and no network, and it is the fixed floor a model engine is measured against rather than
being measured against nothing.

### A refusal is a result, never an escaping exception

One memo whose output was malformed must not abort ingesting the other 791, and a refusal nobody
sees is a refusal nobody reviews. Batch level rejections are recorded on the returned
`FileExtraction`.

### The chain, which no branch currently runs end to end

```
ExtractedClaim  ->  InferenceProposal  ->  ReviewedProposal  ->  PromotedFact  ->  file edit
  truth_extraction    _extracted.py        promotion.py         promotion.py      rewrite.py
```

Every link exists on some branch. No branch runs all four. Closing this chain is the primary new
implementation work.

### Where a written value may land

`rewrite.py` adds **no frontmatter key**. Its `FRONTMATTER_KEYS` *is* `VALIDITY_KEYS`, imported
rather than restated, so the set cannot drift by someone editing that file. Relations the
frontmatter has no vocabulary for (`contradicts`, `same_entity`, `declares_status`) go to a
delimited derived block appended to the body, which `parse_frontmatter` sees as ordinary text and
the trust layer therefore never mistakes for authored metadata.

Inventing `contradicts:` as a fourth frontmatter key would enlarge the trust layer's input surface
without enlarging what the trust layer can act on.

### The relation to key resolver, which is new work

This is the one place the two merged branches do not already fit together, and it is worth
stating exactly rather than discovering during implementation.

`a7834c`'s `destination()` routes on a bare **key**: `FRONTMATTER_KEYS` is `VALIDITY_KEYS`, and
`DERIVED_KEYS` is `("contradicts", "same_entity", "status")`. `ardinghelli`'s `_extracted.py`
emits a **relation** and encodes the key as a prefix inside `object_id`: a validity claim arrives
as `object_id == "valid_from:2026-07-14"`, a status claim as `object_id == "status:deprecated"`.

Nothing currently bridges the two. The bridge is a single resolver in `rewrite.py`:

| Relation | Key | Value | File edited |
|---|---|---|---|
| `supersedes` | `supersedes` | `subject_id` | `object_id` |
| `declares_validity` | parsed from `object_id` prefix, must be `valid_from` or `valid_until` | rest of `object_id` | `subject_id` |
| `declares_status` | `status` | rest of `object_id` | `subject_id` |
| `contradicts` | `contradicts` | `object_id` | `subject_id` |
| `same_entity` | `same_entity` | `object_id` | `subject_id` |
| `references` | none | none | refused |

Two properties this table has to be tested for. A prefix that parses to anything other than the
two validity keys is **refused**, not coerced, so a malformed `object_id` cannot smuggle a fourth
frontmatter key past `destination()`. And `supersedes` is the only row whose edited file is
`object_id`; every other row edits `subject_id`. Getting that column wrong writes the right key
onto the wrong document, which is a quieter failure than an inverted edge and no less wrong.

The `status` hole in `DERIVED_KEYS` was left deliberately by `a7834c`, whose comment reads that
`status` is routable but no relation emits it yet. Schema v2 is what emits it. The hole and the
thing that fills it were built on separate branches by separate sessions, which is the clearest
single argument for consolidating rather than continuing to build in parallel.

### Direction

For `supersedes`, `subject_id` is the superseded document and `object_id` the superseding one,
matching the orientation `reasoning_graph.py` builds `authored_supersedes` with. The schema has no
`superseded_by`, so the key lands on `object_id` and names `subject_id`.

Inverting this declares the live memo stale and demotes it beneath the memo it replaced, which is
the exact failure the trust layer exists to prevent, caused by the tool meant to fix it.

## Invariants that must survive the merge

1. **No proposal reaches corpus metadata without a named human.** Enforced at
   `promotion.py:184` and pinned by tests.
2. **The planner never calls a model.** `max_model_calls` defaults to 0. Extraction runs on the
   ingest path, never the query path. `truth_extraction/__init__.py` imports
   `_extracted.py` rather than being re-exported from `reasoning_proposals`, so the dependency
   runs one way only and the query planner cannot pull a model backed component in behind it.
3. **Proposal ids are content hashes.** `_providers.py:86` recomputes and raises on mismatch. The
   model supplies semantics only; the library computes identity.
4. **Dry run by default.** `--apply` is opt in, for the reason `recall lint --fix` gives: a tool
   that rewrites your memory the first time you try it has earned distrust.

## CLI surface

Verb subparsers under a group noun, matching `cli.py:487`.

```
recall extract run <path> [--glob] [--limit N] [--recheck] [--cache PATH]
recall extract show <file>

recall rewrite plan <path> [--glob]
recall rewrite apply <path> --proposal <id> --reviewer <id> --note "..." [--apply]
recall rewrite reject --proposal <id> --reviewer <id> --note "..."
recall rewrite verify <path>

recall reasoning proposals --include-extracted        # existing command, new flag
```

`rewrite apply` takes **one proposal per invocation** and requires `--reviewer` and `--note` as
hard argparse requirements. That puts the named human gate at the argument parser, before any code
runs. Note that `required=True` accepts an empty string, so both are additionally checked for
non-empty content.

`recall extract run` refuses with an actionable message when the extra is absent, mirroring
`entailment.py:62`. Help text states the dry run default.

## MCP surface

Ship `recall_rewrite_plan` (read only). Deliberately do **not** ship `recall_rewrite_apply`.

The MCP client is the model. Letting it supply `reviewer_id` and `audit_note` makes the named
human gate a formality it satisfies by typing a string: the gate becomes a field, not a person.
The MCP surface proposes; a human applies at the CLI.

`recall_mcp/` currently contains zero file write calls. This work does not make it the first thing
that mutates a user's documents.

`recall_reasoning_proposals` gains `include_extracted: bool = False`, mirroring `include_text` at
`server.py:968`, so existing behaviour is byte identical by default. Annotations are pinned by a
test.

## Testing

House conventions apply without exception.

- Tests live flat in `tests/test_<area>_<subject>.py`. DB tests carry `@requires_db` and reach the
  DB through `make_store` / `cli_table`. `@requires_db` is a collection time optimisation; the
  refusal is `require_db()` inside the fixture (`conftest.py:156`).
- A boundary someone could violate gets a `tests/test_*_contract.py` whose docstring enumerates
  the properties, one test per property, written so a plausible wrong implementation fails.
- Anything writing an artifact gets a validator at the write site, one `pytest.raises(match=<field>)`
  per rejection path, one non over rejection test, and one test that the write site actually calls
  the validator. Pattern: `benchmarks/artifact_contract.py:17`.
- CLI tests use `cli_table`.
- `python -m ruff check .` and `mypy` clean. Bare `ruff` on this machine is a stale 0.6.9. Never
  run `ruff format`: 348 of 406 files fail it and CI does not run it.

Every guard is mutated and watched go red before it is claimed to work.

## Consequences and deferred costs, stated rather than discovered

**Proposal schema version 1 to 2.** Merging `ardinghelli` bumps `PROPOSAL_SCHEMA_VERSION`, which
rewrites every `ip_` id in existence, including in the checked in
`results/reasoning_session3_proposals.json`. That churn is already committed on the branch and
rides along with the merge. The bump is deliberate: an id minted under a vocabulary that could not
express validity must not be mistaken for one minted under a vocabulary that can.

**Rejections do not travel.** The ledger is a SQLite sidecar. Two people reviewing the same corpus
each decline the same bad proposal separately, and a fresh clone starts empty. This is a real
limitation and the thing to revisit if this ever runs somewhere with more than one reviewer.

**Temperature 0 is not a guarantee** from any hosted provider. `--recheck` exists to measure
whether determinism holds rather than to assume it: it re-calls the model on cached keys and
reports the mismatch rate. A non-zero rate means the cache, not the sampler, is what makes runs
reproducible, which is worth knowing before a cache eviction silently renumbers every proposal id
derived from it. Recheck is currently written against `1cfc81`'s cache and must be reimplemented
onto `ardinghelli`'s, which keys on engine plus prompt.

**Treat a non-empty proposal list as a question, not an answer.** The rule based prior narrowed 60
prose markers to four candidates and all four were wrong. This is a reviewing aid, not an
automation.
