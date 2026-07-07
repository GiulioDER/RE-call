#!/usr/bin/env python3
"""Fine-tune a small embedding model for retrieval, and measure the lift on a held-out query split.

Recipe adapted from a proven production trainer: sentence-transformers + OnlineContrastiveLoss
over (query, chunk) pairs. Positives = query <-> its gold chunk; negatives = query <-> wrong chunks.
Trains on the --queries JSON ["train"] split over the --corpus folder, and evaluates retrieval on
the HELD-OUT ["test"] split (differently-phrased queries), so the measured lift is generalization,
not memorization. Two-corpus study: docs/RAG_TRAINING_STUDY.md.

CPU is fine for this tiny model/dataset.
    python finetune/train.py                                       # null (rich corpus) -> delta ~ +0.00
    python finetune/train.py --corpus finetune/confusable_corpus \
        --queries finetune/confusable_queries.json --epochs 10     # positive (opaque-jargon corpus)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

from recall.eval.metrics import mrr, ndcg_at_k
from recall.index import chunk_text

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT.parent / "recall" / "eval" / "corpus"
HARD = ROOT / "hard_queries.json"
OUT = ROOT / "model"


def load_chunks(corpus_dir: Path) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    for f in sorted(corpus_dir.glob("*.md")):
        for i, c in enumerate(chunk_text(f.read_text(encoding="utf-8"))):
            ids.append(f"{f.name}:{i}")
            texts.append(c)
    return ids, texts


def evaluate(model, ids: list[str], texts: list[str], queries: list[dict]) -> tuple[float, float]:
    import numpy as np

    chunk_emb = model.encode(texts, normalize_embeddings=True)
    q_emb = model.encode([q["query"] for q in queries], normalize_embeddings=True)
    mrrs, ndcgs = [], []
    for i, q in enumerate(queries):
        sims = chunk_emb @ q_emb[i]
        ranked = [ids[j] for j in np.argsort(-sims)]
        mrrs.append(mrr(ranked, q["relevant_ids"]))
        ndcgs.append(ndcg_at_k(ranked, q["relevant_ids"], 10))
    return mean(mrrs), mean(ndcgs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--negatives", type=int, default=3, help="wrong chunks per query")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", default=str(CORPUS), help="folder of .md docs to index")
    ap.add_argument("--queries", default=str(HARD), help="JSON with train/test query splits")
    ap.add_argument("--out", default=str(OUT), help="where to save the fine-tuned model")
    args = ap.parse_args()
    random.seed(args.seed)

    ids, texts = load_chunks(Path(args.corpus))
    if not ids:
        raise SystemExit(f"no .md chunks found under --corpus {args.corpus!r}")
    text_by_id = dict(zip(ids, texts))
    data = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    train_q, test_q = data["train"], data["test"]

    # --corpus and --queries are independent flags; guard the invariant that every query's gold
    # chunk exists in the loaded corpus. Without this a mismatch either crashes with a cryptic
    # KeyError (train) or, worse, silently collapses every metric to ~0 (test) — a misleading
    # "fine-tuning did nothing" result rather than an error.
    missing = sorted(
        {rid for q in train_q + test_q for rid in q["relevant_ids"] if rid not in text_by_id}
    )
    if missing:
        raise SystemExit(
            f"--queries {args.queries!r} references chunk ids absent from --corpus {args.corpus!r}: "
            f"{missing[:5]}{' and more' if len(missing) > 5 else ''}. Do --corpus and --queries match?"
        )

    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )

    model = SentenceTransformer(args.base)

    base_mrr, base_ndcg = evaluate(model, ids, texts, test_q)
    print(f"BASE       test MRR={base_mrr:.3f}  nDCG@10={base_ndcg:.3f}")

    s1, s2, labels = [], [], []
    for q in train_q:
        gold = q["relevant_ids"][0]
        s1.append(q["query"])
        s2.append(text_by_id[gold])
        labels.append(1)
        wrong = [cid for cid in ids if cid != gold]
        for neg in random.sample(wrong, min(args.negatives, len(wrong))):
            s1.append(q["query"])
            s2.append(text_by_id[neg])
            labels.append(0)
    train_ds = Dataset.from_dict({"sentence1": s1, "sentence2": s2, "label": labels})
    print(f"built {len(labels)} pairs from {len(train_q)} train queries")

    loss = losses.OnlineContrastiveLoss(model=model, margin=args.margin)
    out = Path(args.out)
    targs = SentenceTransformerTrainingArguments(
        # scratch/checkpoint dir tracks --out, so parallel runs (null vs confusable) don't collide
        output_dir=str(out.parent / f"{out.name}_ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        warmup_steps=10,
        learning_rate=2e-5,
        report_to=[],
        logging_steps=10_000,
        save_strategy="no",
    )
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=train_ds, loss=loss)
    trainer.train()
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))

    ft_mrr, ft_ndcg = evaluate(model, ids, texts, test_q)
    print(f"FINE-TUNED test MRR={ft_mrr:.3f}  nDCG@10={ft_ndcg:.3f}")
    print(f"delta MRR={ft_mrr - base_mrr:+.3f}   delta nDCG@10={ft_ndcg - base_ndcg:+.3f}")
    print(f"saved fine-tuned model to {out}")


if __name__ == "__main__":
    main()
