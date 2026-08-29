# Preregistration: folder and facet scoping, on a re-frozen population

Date: 2026-08-29   Status: predicted, not yet measured

## Why this exists rather than a rerun

`docs/preregistrations/2026-08-28-folder-scope-and-prior.md` ran and is recorded VOID on its folder
arm: it measured a bug (`metadata->>'file'` was a basename on the production build path, so
`folder=` matched nothing) rather than the idea. Its baseline also reproduced only 19 of 31
controls, because the population was frozen on 2026-08-26 against a corpus six generations back.

Both faults are fixed rather than argued away: the folder dimension is verified working on the
current generation (`folder=recall` and `folder=sentiment-agent` return disjoint, correctly
prefixed sources), and the population is re-frozen below against the generation being measured.

## The question

On the CURRENT corpus, does scoping retrieval to the folder or facet that contains the answer
recover memos that unscoped retrieval ranks outside the top 5, and at what cost to the ones it
already finds?

## How the population is re-frozen

The 46 (query, gold memo) pairs of `agent-ab-skill-001` are reused unchanged; only their LABELS are
re-derived, because a label is a claim about a corpus and this is a different corpus.

1. Run every query unscoped, k=5, against `gen_407f0e957c574939be4b61224294d895`.
2. A row whose gold memo appears in the hits is a **control**; a row whose gold memo does not is a
   **miss**.
3. Freeze that partition with the generation id and a digest, and measure the arms against it.

⚠️ **This makes the baseline arm true by construction, and that is the point rather than a flaw.**
Baseline rescues 0 of the misses and retains all of the controls BY DEFINITION. It is the
definition of the population, not a result, and it must not be reported as one. The only numbers
that carry information are the filter and facet arms.

## Arms

* `filter` — oracle folder, the store directory containing the gold memo. All eight gold memos are
  in `recall/`, which is **17.9%** of the corpus; `sentiment-agent` alone is 80.4%. A CEILING
  probe: the system is handed the answer's location.
* `facet` — oracle facet, the `type:` the gold memo declares (feedback ×4, project ×3,
  reference ×1). Also a ceiling probe.

The prior arm is again absent: `ScopePrior` is a retriever constructor argument with no MCP
surface, so it cannot be exercised over this transport.

## What I predict

Written before the baseline runs, so the denominator is unknown and the predictions are
proportional.

| arm | misses rescued | controls retained |
|---|---|---|
| `filter` (oracle folder) | **25%** | **≥ 95%** |
| `facet` (oracle facet) | **12%** | **≥ 95%** |

Higher for the folder than the facet, for a mechanical reason rather than a hunch: every gold memo
is in `recall/`, and the folder scope deletes `sentiment-agent`, which is 80.4% of the corpus and
therefore most of what a `recall/` memo is outranked BY. The facet scope cuts across stores and
removes less of the specific competition.

Deliberately low in absolute terms. Eleven of twelve past predictions here were falsified for being
two to four times too high (`[[i-over-predict-effect-magnitudes]]`), and three independent probes
have now returned 0 rescues for "make the right memo easier to find".

**Mechanism metric, predicted beside the outcome:** on the misses, the share of baseline top-5 hits
that come from OUTSIDE the oracle folder is **≥ 70%**. If that is high and rescues are low, the
competition is not what is keeping the memo out and no scope will fix it. If it is low, the memo is
being outranked from inside its own folder and scoping cannot help by construction.

## What would falsify this

* `filter` rescues 0, or fewer than 10% of misses. The ceiling has no signal even when handed the
  answer's folder, and the lane closes for this population.
* Either arm loses more than 5% of controls. A hard filter that costs answers already being found
  is worse than nothing.
* The mechanism metric is below 40% while rescues are at or above 25%: the gain would not be coming
  from removed competition, so something else is doing the work and the arm is measuring a confound.

## Confounds named now

* **The oracle is not a system.** Both arms are handed the answer's location, so they measure a
  ceiling. A shippable version has to infer the scope, and nothing here says it can.
* **One folder dominates the population.** All eight gold memos are in `recall/`, so `filter` is
  effectively one scope tested 46 times, not a folder dimension tested broadly.
* **Re-freezing changes the denominator.** Miss and control counts will not match the 15/31 of the
  previous record, so the two runs are not comparable arm to arm, only in direction.
* **k=5 with a 20-candidate pool.** A rescue means moving into the top 5 of a pool that already
  contained the memo. A memo absent from the pool entirely cannot be rescued by reordering.

## Observed results

**Status: measured 2026-08-29.** Generation `gen_407f0e957c574939be4b61224294d895`, k=5, all 138
responses `trust_state=trusted`, population digest verified against the registered value.
Artifacts: `results/scope_arms/scope-arms-refrozen-20260829.json` and
`results/scope_arms/mechanism-refrozen-20260829.json`.

Re-frozen partition: **27 misses, 19 controls** of 46.

| arm | misses rescued | controls retained |
|---|---:|---:|
| baseline | 0 of 27 — **by construction, not a result** | 19 of 19 |
| `filter` (oracle folder) | **0 of 27 (0%)** | 19 of 19 (100%) |
| `facet` (oracle facet) | **4 of 27 (15%)** | 19 of 19 (100%) |

**The prediction is falsified in DIRECTION, not merely in magnitude.** I predicted 25% for the
folder and 12% for the facet, and argued the folder would win because `sentiment-agent` is 80.4% of
the corpus and therefore most of what a `recall/` memo is outranked by. Measured: folder 0%, facet
15%. The arm I argued for hardest returned nothing, and the one I ranked second was the only one
that worked.

**The mechanism metric says exactly why, and it is the number that was wrong.** Predicted: at least
70% of the baseline top-5 on misses would come from OUTSIDE the oracle folder. Measured: **39%
outside, 61% inside** (53 and 82 of 135 hits over 27 queries, none empty).

So a `recall/` memo is outranked mostly by *other `recall/` memos*. The folder scope deletes 82% of
the corpus and removes only 39% of the actual competition, leaving the memo beaten by the same
folder-mates it was already losing to — which is precisely the registered condition under which
"scoping cannot help by construction". The facet scope cuts ACROSS folders and thins `recall/`
itself, so it removes the within-folder competition that the folder scope cannot touch. That is why
the weaker-looking scope is the one that rescued anything.

🔑 **The reusable finding: semantic competition is topically LOCAL.** A document's nearest rivals
live where it lives. A structural scope that removes distant material removes competitors that were
never competing, which is why "shrink the corpus by 82%" bought exactly zero. Any future scoping
work should be judged on how much of the *near* competition it removes, not on how much of the
corpus it deletes.

Neither arm cost a single control (19 of 19 both), so hard filtering did no damage here. Both
increased abstentions (baseline 4, filter 9, facet 11), which is the trust layer behaving correctly
on a thinner pool rather than a fault.

**Apparatus gap, recorded rather than hidden:** the runner logged `found` but not hit sources, so
the registered mechanism metric was not computable from the primary artifact and needed a second
27-query pass. A metric named in a registration should be instrumented by the runner that the
registration names.

**What this does NOT say.** Both arms are oracles handed the answer's location, so 15% is a
ceiling, not a feature. All eight gold memos sit in `recall/`, so `filter` is one scope tested 46
times rather than a folder dimension tested broadly. And 27 of 46 queries now miss unscoped, against
15 of 46 when the population was first frozen: the corpus got harder, and no arm here addresses that.

