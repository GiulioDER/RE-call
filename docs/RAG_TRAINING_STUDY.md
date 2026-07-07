# Does fine-tuning the embedder help? A controlled study

**Short answer: only as much as your corpus's vocabulary diverges from what the base model already
knows.** On a semantically-rich corpus a modern small embedder leaves *zero* headroom; on an
opaque-jargon corpus, fine-tuning nearly doubles retrieval. This is a controlled demonstration of
both, with the same pipeline, so the difference is the corpus — not the method.

> Reproduce everything below with `finetune/train.py` (see the commands at the end). Weights are
> gitignored; the numbers are committed.

## Method

`finetune/train.py` domain-adapts **`all-MiniLM-L6-v2`** for retrieval:

- **Loss**: `OnlineContrastiveLoss` over `(query, gold-chunk)` positives and `(query, wrong-chunk)`
  hard negatives (3 per query).
- **Honest split**: train on one set of queries; measure on a **held-out** set of *differently-phrased*
  queries for the same documents — so a lift is **generalization, not memorization**.
- **Measurement**: MRR and nDCG@10 on the held-out queries, **before** and **after** fine-tuning.

Same recipe for every run. Only the corpus and queries change.

## Result 1 — a rich corpus: zero lift (the honest null)

The default eval corpus (14 documents on distinct topics — caching, retries, incidents…), with
held-out paraphrased queries:

| model | test MRR | test nDCG@10 |
|---|---|---|
| all-MiniLM-L6-v2 (base) | 1.000 | 1.000 |
| + fine-tuned | 1.000 | 1.000 |
| **Δ** | **+0.00** | **+0.00** |

**Zero lift, and that is the correct outcome.** The corpus is semantically separable and the base
model already retrieves the right chunk for every held-out query, even when it's paraphrased with
fresh vocabulary. There is no headroom. Manufacturing a win here would have meant testing on the
*training* queries (memorization) or crippling the base on purpose.

## The lift is hard to find — the base is strong

Before concluding "fine-tuning is useless," I tried to *engineer* a corpus where it would help. First
attempt: **nine near-identical service runbooks** (same template — "the X service calls Y with an
N-second timeout, retries M times, pages team T"), differing only in entities and values, queried by
functional synonym ("how long does the *basket-to-order* flow wait on the *card processor*?").

Surface-confusable, but the base still scored **MRR 1.000**. A strong embedder distinguishes nine
near-duplicate documents effortlessly when the discriminators (service, dependency, team) are real
words it understands. **Structural confusability is not enough** — the base has to be missing actual
*knowledge*.

## Result 2 — an opaque-jargon corpus: fine-tuning nearly doubles retrieval

So I removed the knowledge. Nine documents that are **pure metadata behind an opaque codename** —
no description of what the feature does:

```
Codename Aster. Owner: Storefront. Feature flag: aster_ga. Rollout: 100%. Tier: 0. Status: GA.
Codename Basil. Owner: Storefront. Feature flag: basil_ga. Rollout: 50%.  Tier: 1. Status: GA.
… (Clove, Dill, Elm, Fern, Gorse, Holly, Iris)
```

The concept↔codename mapping (*Aster = one-tap reorder*, *Basil = guest checkout*, …) appears
**nowhere in the documents** — it lives only in the training queries. A held-out query asks about a
feature in plain language it has never seen paired with the codename:

| model | test MRR | test nDCG@10 |
|---|---|---|
| all-MiniLM-L6-v2 (base) | 0.306 | 0.465 |
| + fine-tuned | **0.547** | **0.657** |
| **Δ** | **+0.241** | **+0.192** |

**Base retrieval is near-random** (MRR 0.31 over 9 documents) — with no concept text in the docs, it
has nothing to match a plain-language query against. Fine-tuning on 18 `(concept-query → codename)`
pairs **injects the association**, and it **generalizes**: on *held-out, differently-phrased* queries
MRR climbs to 0.547 — a **+79% relative lift**.

It does **not** reach 1.0, and that's the honest part: 18 training pairs over opaque names is thin
signal, so fine-tuning helps a lot but doesn't fully solve it. A real, believable lift — not a
suspiciously perfect one.

## The takeaway

The two results are the same pipeline on two corpora. The variable that flipped the outcome is the
**gap between what the base model already encodes and what your corpus's vocabulary demands**:

| | rich corpus | opaque-jargon corpus |
|---|---|---|
| base–corpus vocabulary gap | none | large |
| base retrieval | already perfect | near-random |
| fine-tuning lift | **+0.00** | **+0.24 MRR** |

**Decision rule for anyone weighing an embedding fine-tune:** measure the base model's retrieval on a
held-out set *first*. If it's already strong, fine-tuning buys nothing — spend the effort on retrieval
(hybrid + rerank) or on the corpus. Fine-tuning pays off specifically when your corpus speaks a
language the base model never learned: internal codenames, domain acronyms, product jargon. **Measure
the gap; don't fine-tune on faith.**

## Reproduce

```bash
pip install -e ".[finetune]"

# Result 1 — the null (rich corpus)            -> Δ ≈ +0.00
python finetune/train.py

# Result 2 — the positive (opaque-jargon corpus) -> base MRR ~0.31, fine-tuned ~0.55
python finetune/train.py \
  --corpus finetune/confusable_corpus \
  --queries finetune/confusable_queries.json \
  --epochs 10
```

— See the [engineering writeup](WRITEUP.md) · the retrieval [evaluation findings](../results/FINDINGS.md)
· the [README](../README.md).
