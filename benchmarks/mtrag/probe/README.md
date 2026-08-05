# MTRAGEval RB_alg probe — the byte-exact runners

These are the scripts that produced [`results/mtrag/PROBE_VERDICT.md`](../../../results/mtrag/PROBE_VERDICT.md),
committed so the verdict can be re-derived rather than trusted. Pre-registration:
[`benchmarks/PREREGISTRATION-mtrag-rbalg.md`](../../PREREGISTRATION-mtrag-rbalg.md).

## Input

One file, IBM's published baseline evaluations, not redistributed here:

```
mt-rag-benchmark/mtrag-human/evaluations/RAG.json
```

from the MT-RAG benchmark at revision `cc5b1d481b391181b89f7ced860308482e785463`. 842 tasks ×
10 models = 8,420 evaluations, each carrying `model_response` and per-metric `system` (raw) and
`composite` (answerability-conditioned) values. **No generation and no API call is involved**, so
every number below costs nothing and introduces no model of ours into the measurement.

## Order

```bash
python probe_invariant.py    /path/to/RAG.json   # exits 1 -- the invariant FAILS, by design
python probe_diagnose.py     /path/to/RAG.json   # P1, P2, P3, P4
python probe_abstention.py   /path/to/RAG.json   # the ceiling P4 exposed
python probe_p3_confound.py  /path/to/RAG.json   # P3 controlled for conditioning
python probe_p3_ceiling.py   /path/to/RAG.json   # the corrected length ceiling
```

Pure stdlib. No dependency beyond CPython.

## Two traps these scripts are the record of

**`probe_invariant.py` exits 1 and that is the correct result.** Recomputing the published MTRAG
table from `RAG.json` does not reproduce it. The aggregation formula is right, reproducing an
independently self-reported SemEval triple to four decimals; the instance set is not the published
one. **`RAG.json` licenses paired within-file comparison and no parity claim.**

**Do not read these exit codes through a pipe.** The first run of `probe_invariant.py` was invoked
as `python probe_invariant.py ... | head -40; echo $?`, which reports `head`'s status, not
Python's, and printed 0 for a run that returned 1. The conclusion survived only because the
verdict was read off the printed table rather than the exit code. Redirect to a file and read the
status directly.

## Known limits of the apparatus, stated rather than implied

- **The tokenizer is ours.** `norm_tokens` mirrors `run_algorithmic.py:normalize_text` and is used
  for **length binning only**. It is not the tokenizer any scorer uses, and no metric here is
  recomputed with it.
- **`Bert-Rec` is confirmed, not assumed.** `probe_diagnose.py` now reconstructs the exported
  `rb_agg` from `(RougeL, Bert-Rec, Bert-KPrec)` on **7,446 of 7,446 unconditioned rows, 100.0%**,
  within 1e-3. That single check confirms the naming, the `(x+1)/2` rescaling and the component
  set together.
- **`probe_diagnose.py`'s P3 does not control for the answerability conditioning.** That is why
  `probe_p3_confound.py` and `probe_p3_ceiling.py` exist: the longest quartile carries 396
  unanswerable instances against 22 to 39 elsewhere, so roughly half the headline gap was the
  conditioning gate rather than length. The uncontrolled figure is preserved in the verdict
  alongside the correction rather than being rewritten.
- **The ceiling statistic is upward-biased by construction** (max of four band means minus the
  overall mean). `probe_p3_ceiling.py` prints a 200-draw permutation null so the reader can
  subtract it: +0.0037 on subset (d), p = 0.000, bias-corrected ceiling +0.0315.
- **`P4` must be read off the MODELS row**, not the pooled row. The pooled figure includes the
  `Target` pseudo-model scoring the reference answer against itself, which cannot fail.
- **`composite()` in `probe_invariant.py` falls back to the `system` slot.** For `rl_f`, `rb_llm`
  and `rb_agg` that fallback never fires, since all 8,420 rows carry `composite` only. It is dead
  code kept so the reader can see the precedence that was intended.
- **Everything is measured on MTRAG.** MTRAG-UN is held out and no script here reads it.
- **`Answerability` is read for stratification and diagnosis only.** Nothing here performs
  inference, so it cannot leak into one; any later detector must derive answerability from the
  conversation and passages alone.
