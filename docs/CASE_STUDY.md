# Case study: where RE-call comes from

RE-call didn't start as a library. It's the extraction of a memory system I run for a **production,
long-running trading-research agent** — an autonomous Claude-based operator that has driven a
multi-strategy research program across many months and hundreds of experiments.

That agent's problem is the one RE-call solves: **its own memory outgrew its context window.** The
guards in this repo aren't hypothetical — each one exists because the agent failed a specific way
without it.

> **Honest boundary.** RE-call is a clean-room extraction of that engine — the same hybrid retrieval
> and the same guards (four as of v0.2, with trust verdicts + calibrated abstention), rebuilt as a
> standalone public library. The trading memory itself stays
> private; the corpus shipped in this repo is synthetic but mirrors the real one's shape. Nothing
> below reveals a strategy, threshold, or result — only the *system*, redacted.

## The memory it runs against

A persistent, human-readable markdown corpus — not a black-box vector blob:

```
memory/
├── MEMORY.md                    # always-loaded index (~100 lines): active work, gates, pointers
├── closed_hypotheses_index.md   # every falsified experiment — searched BEFORE proposing a new one
├── project_*.md      (~435)     # ongoing work, decisions, live state
├── feedback_*.md     (~108)     # how the operator should work (corrections, confirmed approaches)
├── reference_*.md    (~51)      # runbooks, external pointers
└── incident_*.md     (~22)      # what broke, and why
```

**Scale (all aggregate — no content):** ≈**660 typed memos**, ~**5 MB** of markdown, spanning months
of operation. Chunked, embedded, and stored in Postgres; **re-indexed daily** by a session-end hook.

Each memo is *one fact* with typed frontmatter, so retrieval relevance rides on a curated
one-line description:

```markdown
---
name: <kebab-slug>
description: <one-line summary — this is what retrieval matches against>
metadata: { type: project | feedback | reference | incident }
---
<the fact. For project/feedback: **Why** + **How to apply**. Links related memos via [[slug]].>
```

## Not just memory — code, too

The same *retrieve-before-you-act* pattern runs over the **codebase**, not only the memory: a sibling
code-RAG (production uses a code-tuned embedder + BM25) that the agent queries — *"where is X
handled?"* — **before grepping** across a large multi-service repository. RE-call carries this too:
point it at source and it chunks on `def`/`class` boundaries —

```bash
python -m recall.cli index ./src --glob "**/*.py"
python -m recall.cli code      # searches RE-call's own source: "where is RRF implemented?"
```

Same engine, same guards — only the content and the chunker change.

## Why each guard exists

Every guard is a scar. Before it existed, the agent failed this way:

| Guard | The failure it fixes |
|-------|----------------------|
| 🔁 **Anti-re-litigation** | The agent kept re-proposing experiments that had *already been falsified* — burning a day re-deriving a known-dead result. Now it searches a closed-hypotheses index *before* scaffolding anything; a settled decision resurfaces and it backs off. |
| 🕳️ **gap_warning** | Semantic search always returns *something*. On a question the memory had no answer for, the top hits were weakly-related memos — and the agent treated them as if they were relevant. Now, when the top matches all fall below the calibrated threshold (~0.50–0.70 depending on the embedder), the result is flagged a **probable corpus gap** and treated as noise. |
| ⏱️ **Freshness** | Between sessions the index went stale, and the agent silently retrieved facts that a newer memo had already overturned. Now every result reports index age, and a stale index warns instead of serving rot. |

## Anti-re-litigation in action (redacted)

A real interaction, with every strategy name, market, and finding scrubbed to a placeholder — the
*shape* is exact, the content is not:

```text
agent> considering a new experiment: <STRATEGY-X> on <MARKET-Y>
       → recall_search("<STRATEGY-X> on <MARKET-Y>?")  # check before scaffolding

[recall]  1 relevant memory hit (cosine 0.71 — NOT a gap):
  closed_hypotheses_index.md
  "<STRATEGY-X> on <MARKET-Y> — CLOSED, FALSIFIED (2026-0X-XX)
   why dead:  <redacted: failed out-of-sample validation>
   re-entry:  <redacted: only with a different data tier>"

agent> a settled, falsified decision surfaced — not a corpus gap.
       Dropping the idea instead of re-testing a known-dead result.
```

Contrast with a genuine gap, where the guard does the *opposite* job — stops the agent from trusting
noise:

```text
agent> recall_search("<question the memory has never covered>")

[recall]  abstained — top match below the calibrated threshold; probable corpus gap.
          Treat these hits as noise; do not rely on them.

agent> memory has no real answer here. Not going to fabricate one from weak hits.
```

## What's public vs private

| Public (this repo) | Private (stays out) |
|--------------------|---------------------|
| The engine: hybrid dense + sparse retrieval, RRF, reranking, the three guards | The trading memory corpus (strategies, thresholds, results) |
| A synthetic corpus that mirrors the real one's *shape* | Any live infrastructure, credentials, or data |
| The methodology: typed memos, gap calibration, freshness re-indexing | The domain edge itself |

The point of the split: you can read every line of *how* the self-recall loop works, run it against
your own memory in two minutes, and never see a single thing from mine.

— See the [engineering writeup](WRITEUP.md) for the design and the honest evaluation, or the
[README](../README.md) to run it.
