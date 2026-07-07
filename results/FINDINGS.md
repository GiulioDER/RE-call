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
| hashing-64 | 0.30 – 0.68 | 0.35 – 0.45 | no (overlap) | — | 0.00\* | — |
| bge-small (FastEmbed) | 0.70 – 0.90 | 0.50 – 0.64 | yes | ~0.70 | **0.80** | **0.00** |
| voyage-3 | 0.53 – 0.70 | 0.09 – 0.32 | yes | ~0.50 | **0.00** | **0.00** |

Three embedders, three completely different cosine regimes. The fixed 0.50 threshold happens to sit
in Voyage's clean gap (unanswerable ≈ 0.1–0.3, answerable ≈ 0.5–0.7), sits *below the entire* bge
distribution (so the guard never fires — FCR 0.80), and lands inside hashing's overlap. **It works
for one strong model by luck, fails for another strong model, and cannot work for the weak one.**
(\* hashing's 0.00 at 0.50 is misleading: every hashing cosine is low, so it also wrongly flags many
*answerable* queries as gaps.) OpenAI is omitted — the available key had no quota, and the harness
skipped it cleanly (graceful degradation is itself a design goal).

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

## 3. Domain fine-tuning: an honest null result on this corpus

`finetune/train.py` fine-tunes `all-MiniLM-L6-v2` with OnlineContrastiveLoss on `(query, gold-chunk)`
positive / `(query, wrong-chunk)` negative pairs (recipe adapted from a proven production trainer),
then measures retrieval on a **held-out** set of differently-phrased queries:

| model | test MRR | test nDCG@10 |
|---|---|---|
| all-MiniLM-L6-v2 (base) | 1.00 | 1.00 |
| + fine-tuned | 1.00 | 1.00 |
| **Δ** | **+0.00** | **+0.00** |

**Zero lift — and that is the honest, expected outcome here.** The 14-document corpus is highly
separable; a modern small embedder already retrieves the correct chunk for every held-out query,
even when it is paraphrased with different vocabulary. There is no headroom to improve. Manufacturing
a win would have meant evaluating on the *training* queries (memorization) or crippling the base
model on purpose.

To demonstrate a *real* domain-adaptation lift you need a corpus the base model actually struggles
on — many mutually-confusable documents (a dozen near-identical policy variants), or genuine domain
jargon the base embedder never saw. The pipeline (held-out split, pre/post measurement, proven loss)
is built and runs end-to-end; on this saturated corpus it correctly reports no gain. This mirrors a
lesson from the production know-how the recipe came from: **embeddings only encode what they encode —
measure honestly, don't force a result.**
