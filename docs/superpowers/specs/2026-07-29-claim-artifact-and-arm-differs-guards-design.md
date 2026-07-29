# Two guards ported from CCA: claim→artifact, and self-ablation

Date: 2026-07-29

Both guards are ports of anti-hallucination mechanisms from the CCA `/audit-fix` pipeline. The
shared idea is not "check more carefully". It is:

1. **A claim cannot enter the output without an artifact produced by execution** — not a second
   opinion. CCA's `NUM-*` rule: a numeric finding carries a Hypothesis falsifying example or it is
   escalated, because "a sign error reads fluently, which is precisely why this class needs
   execution rather than a second opinion."
2. **The verification's own failure modes must be non-silent.** CCA's tautology check exits
   non-zero; it is a gate, not a report. A test that errors is `INCONCLUSIVE`, never counted as
   proof.

## Why now — three defects from 2026-07-29 that these would have caught

- `benchmarks/SUITE-DESIGN.md` published a **loss as a tie** ("0.533 vs 0.533"); the Mem0 cell is
  **0.536**, re-derived from the n=70 table.
- Our **0.467 is not derivable from any committed artifact** — currently marked citation-pending in
  prose only.
- `results/gap/summary.json` recorded `usable: 1` beside a published `n=17`.

The third is an integer defect and is **out of scope** for guard 1 (see Limits).

## What already exists, and what is missing

`results/ARTIFACTS.md` plus `tests/test_results_artifact_provenance.py` and
`tests/test_results_artifact_model_stack.py` already enforce **artifact → declares itself**:
`_provenance` block, generation, status, named successor, model stack, and presence in the index.

Nothing enforces the other direction. A number can appear in a document without any artifact
containing it. That is the hole all three defects came through.

---

## Guard 1 — claim→artifact gate

### Marker syntax

An HTML comment, so it never renders:

```markdown
hit@5 rises to **0.777** <!--@ locomo_rerank/rerank_shipped.json # depth_curve.5.overall.hit -->
```

The path is relative to `results/`. The key is a dotted path into the JSON; numeric levels work
because JSON object keys are strings.

Two further forms:

| Form | Meaning |
|---|---|
| `<!--@ citation-pending: <reason> -->` | A figure with no artifact yet. A first-class state, not an exemption. **The 0.467 gets this.** |
| `<!--@ derived: <expression> -->` | A number computed from other cited numbers (a delta, a percentage lift). The test checks the arithmetic against the cited operands instead of looking the value up. |

### Match rule

The published value must equal the artifact value **rounded to the number of decimals published**.

- `0.777` against a stored `0.77714` → pass.
- `0.533` against a stored `0.536` → fail (0.536 to 3dp is 0.536).

That second case is the SUITE-DESIGN defect.

### Scope of "a number"

Decimals only: `\d+\.\d+`, excluding fenced code blocks, inline code spans, URLs, version strings,
ISO dates, and `#`-prefixed issue/PR references.

### Gated documents

- `results/RESULTS.md`
- `results/FINDINGS.md`
- `README.md`
- `benchmarks/SUITE-DESIGN.md`

Process records (`PREREGISTRATION*.md`, `REVIEW.md`, `ARTICLE_DRAFT.md`, `CHANGELOG.md`,
`docs/*.md`) are ungated. They are not published results, and starting with a large exemption list
produces a guard that stays mostly-exempted.

### Legacy ratchet

`results/CLAIMS_BASELINE.json` holds today's unmarked decimals as a **per-document multiset** — not
line numbers, which drift under prose edits. Two tests keep it honest, mirroring the `"unrecorded"`
ratchet already used by `test_results_artifact_model_stack.py`:

- **No dead entries.** Every baseline entry must still be present in its document *and still
  unmarked*. Marking a number therefore forces deleting its baseline row, so the file shrinks on its
  own.
- **No growth.** Total baseline size may not exceed a pinned constant.

### Deliverable

`tests/test_published_numbers_have_artifacts.py`. Runs in CI with the rest of the suite. No network,
no API keys.

### Limits, stated rather than buried

**Integers are not gated.** The `n=17` vs `n=18` defect was an integer, and this guard would not
have caught it. Integers cannot be scoped without a large exclusion list (k values, years, PR
numbers, sample sizes). Gating them is separate work.

**A green gate means derivable, not correct.** The guard proves a published number matches a
committed artifact. It says nothing about whether the artifact measured the right thing. CCA works
because code has a cheap oracle — execution. Semantic correctness of a benchmark result has no such
oracle; that is the answerability wall, and this guard does not touch it.

---

## Guard 2 — self-ablation preflight

### Why self-ablation and not cross-arm

`benchmarks/run.py` is **one arm per process** by design ("Splitting the arms across processes also
keeps a crash or a rate-limit in one arm from destroying the other arm's already-paid-for results").
The two arms never coexist, so a cross-arm comparison has nowhere to live.

Self-ablation is also the truer analogue of CCA's red-state proof: revert the mechanism, re-run,
require the output to differ — exactly as the tautology check reverts the fix and requires the test
to go red.

### Correcting the premise

The rule of thumb "`candidate_k == k` renders the reranker inert" is **exactly true only when the
realized fused pool size equals k**. `recall/retriever.py:111-125` reranks the whole fused pool and
truncates to `k` afterwards, and the fused pool can reach `2 * candidate_k` on a hybrid arm. Inert
is therefore a **runtime property to be measured**, not a config predicate to be asserted — and
measuring it also catches inertness nobody predicted.

### Interface

New module `recall/eval/arm_check.py`:

```python
ablation_verdicts(store, embedder, questions, *, k, candidate_k, reranker, use_sparse) -> list[Verdict]
```

For each configured mechanism it re-retrieves with that mechanism disabled and compares the returned
chunk-id lists across N sampled questions. **N defaults to 25**, overridable with
`--ablation-sample`; questions are taken deterministically (first N of the loaded question list, no
RNG) so the preflight verdict is reproducible for a given slice.

Mechanisms checked at launch: **the reranker** and **the sparse leg**. The sparse leg was silently
inert for an entire artifact generation (pre-`#81`/`#84`) through this same defect class, which is
why reranker-only would ship a guard that missed the larger incident.

### Verdicts

| Verdict | Meaning | Action |
|---|---|---|
| `DIFFERS` | ≥1 sampled question returns a different ordered id list | mechanism is live — proceed |
| `SET_IDENTICAL` | same ids, different order, on every question | inert for `hit@k` / `recall@k`; live only for rank-sensitive metrics. **Refuse if the run reports a set metric.** |

The caller declares its metric class when invoking the preflight — `metric_class="set"` or
`"ranked"` — rather than the module inferring it. `benchmarks/run.py` and `benchmarks/beam/run.py`
both pass `"set"` (LOCOMO hit@k, BEAM nugget coverage), so `SET_IDENTICAL` blocks there in
practice. Leaving this to inference would mean a new harness silently gets the permissive branch.
| `IDENTICAL` | byte-identical ordered ids on every question | **provably inert — refuse the run** |

### Placement and cost

Fires **after index build, before the first generator call**, in `benchmarks/run.py` (`--arm recall`)
and `benchmarks/beam/run.py`. Retrieval-only: no generator, no judge, so it costs $0 and runs ahead
of all LLM spend. This matters — BEAM best-config is currently blocked at 5/60 on exhausted
OpenRouter credits, and a post-hoc check would have spent them first.

### Override

`--allow-inert-arm`. The override flag **and every verdict** are stamped into the artifact's
`_provenance`, so a run that was let through cannot read as clean afterwards.

### Known failure direction

The check is only as good as its sample. If all N sampled questions happen to be ones where the
mechanism does not bite, it reports `IDENTICAL` on a live mechanism and blocks a legitimate run.
That is a false positive on the safe side — annoying, not dangerous — and `--allow-inert-arm` is the
escape hatch. The reverse failure direction (passing an inert arm) is the one that corrupts
published results, so the asymmetry is deliberate.

---

## Testing

Both guards exercise the **detection** path, not only the green path:

- A fixture document containing a deliberately wrong number must **fail** the claim gate.
- A fixture document with a correct number and a valid marker must pass.
- A `citation-pending` marker must pass; a bare unmarked new number must fail.
- A baseline entry that no longer appears unmarked in its document must fail (dead-entry test).
- A stub retriever with a dense-only arm at `candidate_k == k` must produce `IDENTICAL`.
- A stub retriever whose reranker genuinely reorders a wider pool must produce `DIFFERS`.
- A stub whose rerank permutes within an unchanged id set must produce `SET_IDENTICAL`.

## Out of scope

- Integer claims (see Limits).
- The wiki, which lives in a separate repository and cannot be gated by this repo's test suite.
- Any judgement about whether a benchmark result is *right* — only whether it is *derivable*.
