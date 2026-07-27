# When is a cloud embedder worth it? — the answer is "measure your local one"

**Run:** 2026-07-27 · 17 BEIR/CQADupStack corpora · bge-small vs voyage-3 · hybrid arm ·
preregistered 2026-07-26 in `docs/superpowers/specs/2026-07-26-embedder-gap-predictor-design.md`
**Artifacts:** one JSON per corpus in this directory · `analysis.json` · reproduces from them exactly.

## The question

FINDINGS §8 established a conditional rule — a cloud embedder bought **+0.282** hit@5 on a private
corpus of internal codenames and **+0.022** on the public PEPs — and attributed the difference to
*vocabulary*: pay when your corpus uses words the model has never seen. That rule was never
operationalised, so it could not be applied by a reader or criticised by a reviewer.

This study asked whether any computable corpus statistic predicts the gap **better than simply
measuring the local embedder**, across corpora that did not generate the hypothesis.

## Result: no predictor beats the null

| predictor | partial r (control = local score) | permutation p | Holm p | |
|---|---|---|---|---|
| `oov_rate` | +0.265 | 0.327 | **0.653** | ✗ |
| `query_overlap` | +0.229 | 0.392 | **0.653** | ✗ |
| `crowding` | −0.655 | 0.006 | **0.019** | ✓ then ✗ — see below |

n = 17, above the preregistered power floor of 12. The null model is substantial in its own right:
the local score alone correlates **−0.512** with the gap, which is the ceiling effect the whole
design was built around.

**The vocabulary hypothesis fails cleanly.** `oov_rate` does not predict the gap, and the failure is
not an artefact: `oov_rate` correlates **−0.015** with corpus size, so nothing is hiding inside it.
The §7/§8 mechanism does not extend beyond the corpus that produced it — which is exactly why the
discovery corpus and PEPs were excluded from this analysis in advance.

## `crowding` survives Holm and then dies to its own confound

`crowding` (mean cosine to the nearest other document in local embedding space) beats the null under
the preregistered analysis. It is also **−0.613 correlated with corpus size**:

```
crowding -> gap | local score  = -0.655   p = 0.006   ✓
crowding -> gap | n_documents  = -0.413   p = 0.112   ✗
n_docs   -> gap | crowding     = +0.229   p = 0.388   ✗
```

Neither variable survives once the other is held fixed. **This design cannot separate "documents are
hard to tell apart" from "the haystack is larger."** The `haystack_confound` diagnostic was
preregistered for precisely this and reports +0.436 partial with a 5.51-fold size range; a common
20 000-document cap compresses that range but does not eliminate it, because corpora smaller than the
cap are used whole.

Reported as a confound rather than tuned away. Separating the two needs a design that varies
haystack size *within* a corpus while holding its content fixed — not attempted here.

## What to do instead

The preregistered decision table, applied: **no predictor beats the null, so skip the corpus
analysis and measure your local embedder on ~30 labelled questions.** It is cheaper than computing
any statistic here and it is the better estimator. Where the local model already scores well there is
little headroom for a cloud model to capture; where it scores badly, there is.

## Scope and limits

- **Subsampling.** Corpora are capped at 20 000 documents, so absolute scores are not comparable to
  published BEIR numbers. Only the within-corpus gap is, and only because both embedders saw the
  identical subsample.
- **One local model, one cloud model.** bge-small and voyage-3. Nothing here says a different pair
  behaves the same way.
- **Held-out questions.** The harness fits calibration on half the questions and scores on the other
  half, so per-corpus n is half what the corpus offers.
- **The excluded corpora are excluded on purpose.** The private memory corpus generated the
  hypothesis and PEPs was its first replication; scoring either as evidence would be circular.

## Reproduce

```bash
python -c "
import json, glob
from recall.eval.gap_study import analyse_records
recs=[json.load(open(p)) for p in sorted(glob.glob('results/gap/*.json')) if 'summary' not in p and 'analysis' not in p]
print(json.dumps(analyse_records(recs, arm='hybrid'), indent=2))"
```
