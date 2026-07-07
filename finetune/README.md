# Fine-tuning demo

`train.py` domain-adapts a small embedding model (`all-MiniLM-L6-v2`) for retrieval on the eval
corpus, using `OnlineContrastiveLoss` over `(query, chunk)` pairs, and measures the lift on a
**held-out** query split (so it's generalization, not memorization).

    pip install -e ".[finetune]"
    python finetune/train.py --epochs 8

Output: base vs fine-tuned test MRR / nDCG@10 and the delta. The fine-tuned model is saved to
`finetune/model/` (gitignored — we commit the numbers, not the weights).

**Result on this corpus: zero lift.** The base model already saturates the (highly separable) corpus,
so there is no headroom — the honest, expected outcome. The recipe is adapted from a proven production
fine-tuning trainer (sentence-transformers + OnlineContrastiveLoss + pre/post eval).

For the **full study — including a second corpus where fine-tuning *does* lift retrieval** — see
[`docs/RAG_TRAINING_STUDY.md`](../docs/RAG_TRAINING_STUDY.md).

```bash
# the null (easy corpus)      -> Delta = +0.00
python finetune/train.py

# the positive (confusable corpus)
python finetune/train.py --corpus finetune/confusable_corpus --queries finetune/confusable_queries.json
```
