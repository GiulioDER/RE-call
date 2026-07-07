# recall — evaluation findings

Interpreted results. The reproducible ablation numbers are in [`RESULTS.md`](RESULTS.md) (run
`make eval`); the per-embedder threshold numbers below come from `recall.eval.calibrate.calibrate()`.

## 1. Hybrid + rerank helps — where the embedder isn't already saturated

On the weak, non-semantic hashing embedder, quality climbs monotonically as we add the sparse leg
and then the cross-encoder reranker:

| fusion | MRR | nDCG@10 |
|---|---|---|
| dense only | 0.68 | 0.76 |
| + sparse (hybrid) | 0.79 | 0.84 |
| + cross-encoder rerank | 1.00 | 1.00 |

On the strong FastEmbed (bge-small) embedder, dense retrieval already scores a perfect nDCG@10 on
this corpus, so the fusion arms have nothing left to gain. Honest reading: **hybrid + rerank buys
the most on weaker embedders or harder corpora; on an easy corpus with a strong embedder it is
redundant.** A rigorous eval has to be able to show that, not just a win.

## 2. The honest negative result: a fixed gap threshold does NOT transfer across embedders

The gap guard fires when the best dense cosine for a query falls below a threshold (default 0.50).
We measured the top-cosine distribution for answerable vs. unanswerable queries per embedder:

| embedder | answerable cos (min–max) | unanswerable cos (min–max) | separable? | good threshold | FCR @0.50 | FCR @calibrated |
|---|---|---|---|---|---|---|
| hashing-64 | 0.30 – 0.68 | 0.35 – 0.45 | no (overlap) | — | 0.00 | — |
| bge-small (FastEmbed) | 0.70 – 0.90 | 0.50 – 0.64 | yes | ~0.70 | **0.80** | **0.00** |

Two lessons:

- **The default 0.50 is miscalibrated for a strong dense embedder.** bge-small's cosines live in
  roughly [0.50, 0.90]; 0.50 sits *below the entire distribution*, so the guard almost never fires
  and the false-confident rate on unanswerable queries is 0.80. Recalibrated to ~0.70 — the gap
  between the unanswerable ceiling (0.64) and the answerable floor (0.70) — the guard becomes
  perfect: FCR 0.00, with cleanly separable distributions.
- **Gap-detection quality is bounded by the embedder.** hashing-64's answerable and unanswerable
  distributions overlap, so no single threshold separates them: a weak, non-semantic embedder
  cannot support reliable gap detection at all, at any threshold. (0.50 scores FCR 0.00 only because
  every hashing cosine is low — it would also wrongly flag many *answerable* queries as gaps.)

Takeaway for anyone building gap/abstention into a RAG system: **calibrate the threshold per
embedding model against a small labeled answerable/unanswerable set; do not ship a hard-coded
constant, and do not assume a strong embedder's cosines are centered where a weak one's are.**
`recall.eval.calibrate.calibrate()` reproduces these numbers.
