"""Fine-tune the shipped cross-encoder reranker on held-out LOCOMO, evaluate on the ladder's 200.

Prior work (searched before writing this, per CLAUDE.md):
`docs_search(source_type='memory', ...)` on "fine-tuning reranker cross-encoder training RE-call"
returned `gap_warning: true` for the reranker-fine-tuning question -> **no prior work**. Adjacent and
deliberately NOT this experiment:
  - [[project_ns4_finetune_bge_2026-04-26]] fine-tuned a BI-ENCODER (BGE-M3) against `sign(forward_return)`
  - [[archive/project_replug_lsr_2026-05-06]] trained a retriever adapter on LLM feedback about returns
  - `docs/RAG_TRAINING_STUDY.md` fine-tuned a bi-encoder on two toy corpora (14 and 9 documents)
Every one of those learned a NOISY or TOY target. This learns gold-evidence identity on a real corpus,
against a reranker with measured headroom. That difference is the whole reason it is worth running.
  - [[project-recall-nearmiss-signal-exhaustion-2026-07-29]] -- the reranker arc
  - `results/rerank_pool_arms.json` -- the matched-config baseline this builds on

============================================ SETUP =============================================
Train  : the 1,333 LOCOMO questions the ladder never used (of 1,533 total), gold `evidence` as
         positives, BM25 top-`--negatives` corpus-wide as hard negatives.
Eval   : the ladder's 200 answerable questions, scored over their REAL 50-chunk RRF pools from
         `ksweep_out.jsonl` -- i.e. the deployment distribution, not a training-shaped one.
Disjoint by construction on QUESTIONS. Not on the corpus: both halves draw on the same 10
conversations, so this is a held-out-question split (the design `docs/RAG_TRAINING_STUDY.md` uses),
NOT cross-dataset. LongMemEval-Oracle in `oracle_out/` is the stronger OOS check and is left to a
follow-up; a lift here is a necessary but not sufficient condition for one there.

Negatives are mined corpus-wide (5,882 docs) rather than within the gold's conversation because
184/400 of the eval pools cross conversation boundaries -- scoping negatives per-conversation would
train on an easier distribution than the one we score on.

======================================= PRE-REGISTRATION =======================================
Recorded before the first run, per [[feedback-predict-before-measuring-and-assert-the-invariant-2026-07-27]].

  Baseline (measured, `results/rerank_pool_arms.json`): hit@5 0.785, hit@1 0.585.
  Target the gap to `voyage:rerank-2.5`: hit@5 0.870 (+0.085), hit@1 0.710 (+0.125).

  PREDICTION: +0.03 hit@5 (range +0.01 to +0.05). It will NOT close the full 0.085 gap.
  Reasoning: ms-marco CE is already trained on a large relevance corpus, so the available gain is
  domain adaptation to conversational turns, not learning relevance from scratch; and 1,333 questions
  is thin. A NEGATIVE outcome is a live possibility -- fine-tuning a well-trained CE on ~1.3k
  examples can degrade it (catastrophic forgetting), which is why the LR is low and epochs few.

  SHIP FLOOR (pre-committed): mean delta hit@5 >= +0.02 AND the paired 95% CI excludes zero.
  Anything less is a null and gets reported as one.

========================================= INVARIANTS ===========================================
All hard failures. Each guards a way this script could print a plausible number while measuring
nothing -- the failure mode that matters more than the result.
  1. train and eval question sets are disjoint
  2. no gold document is ever mined as a negative
  3. the BASE arm reproduces hit@5 = 0.785 exactly (differential oracle: proves the eval path and
     `max_length` change nothing, so any delta belongs to the weights)
  4. fine-tuned weights actually differ from base
  5. the fine-tuned model reorders at least one pool differently from base

Usage:
    python benchmarks/rerank_finetune.py [--epochs 1] [--lr 2e-5] [--negatives 12] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

# NB: torch / sentence-transformers / datasets are imported INSIDE main(), never at module level.
# CI installs `.[dev]` only and deliberately omits the heavy extras, so a module-level import would
# break collection for any test that imports this file. Keeping the module importable under `.[dev]`
# is what lets the helpers below be tested in CI for real, rather than behind an `importorskip` that
# would make the job green while testing nothing.
from benchmarks.ladder.sources.locomo import load_locomo
from recall.eval.bm25 import BM25Index
from recall.rerank import DEFAULT_RERANKER_MODEL, DEFAULT_RERANKER_REVISION

REPO = Path(__file__).resolve().parent.parent
ANSWERABLE_RING = -2
KS = (1, 5, 10)
#: from results/rerank_pool_arms.json -- invariant 3 pins the eval path to this
BASE_HIT5 = 0.785
SEED = 0


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _hit_at(order: list[str], gold: set[str], k: int) -> float:
    return 1.0 if set(order[:k]) & gold else 0.0


def _paired_bootstrap(
    a: list[float], b: list[float], n_resamples: int = 10_000, seed: int = SEED
) -> tuple[float, float, float]:
    """Paired bootstrap on the per-instance difference b - a. Returns (mean, lo, hi) at 95%."""
    diffs = [x - y for x, y in zip(b, a)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    return sum(diffs) / n, means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples)]


def _score_pools(
    model: Any,
    ids: list[str],
    ksweep: dict[str, dict],
    q_text: dict[str, str],
    doc_text: dict[str, str],
    batch_size: int,
    tag: str,
) -> dict[str, list[str]]:
    """Score every (question, pool-member) pair and return per-instance reranked orderings.

    `model` is Any because it is a `sentence_transformers.CrossEncoder`, and that extra is
    deliberately absent from the type-check job (see the mypy overrides in pyproject.toml).
    """
    pairs: list[tuple[str, str]] = []
    spans: dict[str, tuple[int, int]] = {}
    for i in ids:
        qid = i.split("/", 1)[1].split("#", 1)[0]
        q = q_text[qid]
        start = len(pairs)
        for h in ksweep[i]["hits"]:
            pairs.append((q, doc_text[h["doc_id"]]))
        spans[i] = (start, len(pairs))
    t0 = time.time()
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    print(f"  [{tag}] {len(pairs)} pairs in {time.time() - t0:.1f}s")
    order = {}
    for i in ids:
        lo, hi = spans[i]
        member_ids = [h["doc_id"] for h in ksweep[i]["hits"]]
        s = {d: float(v) for d, v in zip(member_ids, scores[lo:hi])}
        order[i] = sorted(member_ids, key=lambda d: s[d], reverse=True)
    return order


def _arm(
    order: dict[str, list[str]], ids: list[str], gold_sets: dict[str, set[str]]
) -> tuple[dict[int, list[float]], dict[int | str, float]]:
    per = {k: [_hit_at(order[i], gold_sets[i], k) for i in ids] for k in KS}
    summary: dict[int | str, float] = {k: sum(v) / len(v) for k, v in per.items()}
    summary["ceiling@5"] = summary[5] + 0.5 * (1 - summary[5])
    return per, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--negatives", type=int, default=12, help="BM25 hard negatives per question")
    ap.add_argument("--batch-size", type=int, default=32, help="train batch")
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=192, help="pairs are p95=93 tokens")
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate plumbing")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "rerank_finetune.json")
    ap.add_argument("--model-out", type=Path, default=REPO / "benchmarks" / "finetune" / "model_rerank")
    args = ap.parse_args()

    import torch

    random.seed(SEED)
    torch.manual_seed(SEED)

    corpus = load_locomo(REPO / "locomo10.json")
    doc_text = dict(corpus.documents)
    q_text = {q.question_id: q.question for q in corpus.questions}
    gold_of = {q.question_id: set(q.gold_doc_ids) for q in corpus.questions}

    ksweep = {r["instance_id"]: r for r in _load_jsonl(REPO / "ksweep_out.jsonl")}
    eval_ids = sorted(i for i, r in ksweep.items() if r["ring"] == ANSWERABLE_RING)
    eval_qids = {i.split("/", 1)[1].split("#", 1)[0] for i in eval_ids}
    train_qids = sorted(set(q_text) - eval_qids)

    # --- INVARIANT 1 -----------------------------------------------------------------------------
    if set(train_qids) & eval_qids:
        raise SystemExit("INVARIANT 1 FAILED: train/eval question overlap")
    print(f"INVARIANT 1 ok: {len(train_qids)} train / {len(eval_qids)} eval questions, disjoint")

    if args.smoke:
        train_qids = train_qids[:40]
        eval_ids = eval_ids[:20]
        print(f"SMOKE: {len(train_qids)} train / {len(eval_ids)} eval")

    # --- mine hard negatives ----------------------------------------------------------------------
    print(f"building BM25 over {len(corpus.documents)} docs")
    bm25 = BM25Index(corpus.documents)
    rows_q, rows_d, rows_y = [], [], []
    t0 = time.time()
    for n, qid in enumerate(train_qids):
        gold = gold_of[qid] & set(doc_text)
        if not gold:
            continue
        question = q_text[qid]
        for g in gold:
            rows_q.append(question)
            rows_d.append(doc_text[g])
            rows_y.append(1.0)
        taken = 0
        for doc_id, _s in bm25.rank(question):
            if taken >= args.negatives:
                break
            if doc_id in gold:  # INVARIANT 2
                continue
            rows_q.append(question)
            rows_d.append(doc_text[doc_id])
            rows_y.append(0.0)
            taken += 1
        if n and n % 400 == 0:
            print(f"  mined {n}/{len(train_qids)} ({time.time() - t0:.0f}s)")
    n_pos, n_neg = int(sum(rows_y)), len(rows_y) - int(sum(rows_y))
    print(f"INVARIANT 2 ok: gold excluded from negatives — {n_pos} pos / {n_neg} neg pairs")

    # --- base arm ---------------------------------------------------------------------------------
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from datasets import Dataset

    gold_sets = {i: set(ksweep[i]["gold_doc_ids"]) for i in eval_ids}
    print(f"\nloading base {DEFAULT_RERANKER_MODEL} @ {DEFAULT_RERANKER_REVISION[:12]}")
    base = CrossEncoder(
        DEFAULT_RERANKER_MODEL, revision=DEFAULT_RERANKER_REVISION, max_length=args.max_length
    )
    base_order = _score_pools(
        base, eval_ids, ksweep, q_text, doc_text, args.eval_batch_size, "base"
    )
    base_per, base_sum = _arm(base_order, eval_ids, gold_sets)

    # --- INVARIANT 3: differential oracle on the eval path ---------------------------------------
    if not args.smoke:
        if abs(base_sum[5] - BASE_HIT5) >= 5e-4:
            raise SystemExit(
                f"INVARIANT 3 FAILED: base hit@5 = {base_sum[5]:.4f}, expected {BASE_HIT5:.4f}. "
                "The eval path changed; a delta could not be attributed to the weights."
            )
        print(f"INVARIANT 3 ok: base hit@5 {base_sum[5]:.3f} reproduces the measured baseline")

    base_fingerprint = base.model.state_dict()[
        "bert.encoder.layer.0.attention.self.query.weight"
    ].clone()

    # --- train ------------------------------------------------------------------------------------
    ds = Dataset.from_dict({"query": rows_q, "response": rows_d, "label": rows_y}).shuffle(seed=SEED)
    model = CrossEncoder(
        DEFAULT_RERANKER_MODEL, revision=DEFAULT_RERANKER_REVISION, max_length=args.max_length
    )
    loss = BinaryCrossEntropyLoss(model)
    targs = CrossEncoderTrainingArguments(
        output_dir=str(REPO / "benchmarks" / "finetune" / "_ce_run"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=False,
        bf16=False,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=SEED,
        dataloader_num_workers=0,
    )
    print(f"\ntraining: {len(ds)} pairs, {args.epochs} epoch(s), lr {args.lr}, batch {args.batch_size}")
    t0 = time.time()
    CrossEncoderTrainer(model=model, args=targs, train_dataset=ds, loss=loss).train()
    train_s = time.time() - t0
    print(f"trained in {train_s:.0f}s")

    # --- INVARIANT 4 ------------------------------------------------------------------------------
    ft_fingerprint = model.model.state_dict()["bert.encoder.layer.0.attention.self.query.weight"]
    drift = float((ft_fingerprint - base_fingerprint).abs().max())
    if drift == 0.0:
        raise SystemExit("INVARIANT 4 FAILED: weights identical to base — training was a no-op")
    print(f"INVARIANT 4 ok: weights moved (max |delta| = {drift:.2e})")

    # --- fine-tuned arm ---------------------------------------------------------------------------
    ft_order = _score_pools(model, eval_ids, ksweep, q_text, doc_text, args.eval_batch_size, "ft")
    ft_per, ft_sum = _arm(ft_order, eval_ids, gold_sets)

    # --- INVARIANT 5 ------------------------------------------------------------------------------
    changed = sum(1 for i in eval_ids if ft_order[i] != base_order[i])
    if changed == 0:
        raise SystemExit("INVARIANT 5 FAILED: fine-tuned ordering identical to base on every pool")
    print(f"INVARIANT 5 ok: ordering changed on {changed}/{len(eval_ids)} pools")

    # --- report -----------------------------------------------------------------------------------
    print(f"\n{'arm':<22} {'hit@1':>8} {'hit@5':>8} {'hit@10':>8} {'ceiling@5':>11}")
    print("-" * 62)
    for name, s in (("MiniLM base", base_sum), ("MiniLM fine-tuned", ft_sum)):
        print(f"{name:<22} {s[1]:>8.3f} {s[5]:>8.3f} {s[10]:>8.3f} {s['ceiling@5']:>11.4f}")
    print(f"{'voyage (paid, ref)':<22} {0.710:>8.3f} {0.870:>8.3f} {0.895:>8.3f} {0.9350:>11.4f}")

    print(f"\n{'paired delta (ft - base)':<26} {'mean':>9} {'CI95 lo':>9} {'CI95 hi':>9}")
    print("-" * 56)
    # Kept as floats, not read back out of the JSON-shaped dict below: the pre-committed floor is
    # evaluated from the numbers themselves, never from a re-parse of its own serialisation.
    stats: dict[int, tuple[float, float, float]] = {}
    deltas: dict[str, dict[str, object]] = {}
    for k in KS:
        mean, lo, hi = _paired_bootstrap(base_per[k], ft_per[k])
        stats[k] = (mean, lo, hi)
        deltas[f"hit@{k}"] = {"mean": mean, "ci95": [lo, hi]}
        flag = "" if lo <= 0 <= hi else "  *"
        print(f"{'hit@' + str(k):<26} {mean:>+9.4f} {lo:>+9.4f} {hi:>+9.4f}{flag}")
    print("\n* = 95% CI excludes zero")

    d5_mean, d5_lo, d5_hi = stats[5]
    passed = d5_mean >= 0.02 and not (d5_lo <= 0 <= d5_hi)
    verdict = "PASS" if passed else "NULL"
    print(f"\nPRE-COMMITTED FLOOR (mean >= +0.02 and CI excludes 0): {verdict}")
    print(f"  predicted +0.03 (range +0.01..+0.05) -> observed {d5_mean:+.4f}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.model_out))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "n_eval": len(eval_ids),
                "n_train_questions": len(train_qids),
                "train_pairs": {"positive": n_pos, "negative": n_neg},
                "negatives_per_question": args.negatives,
                "negative_source": "BM25 corpus-wide",
                "hyperparams": {
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "max_length": args.max_length,
                    "seed": SEED,
                },
                "base_model": DEFAULT_RERANKER_MODEL,
                "base_revision": DEFAULT_RERANKER_REVISION,
                "corpus_content_hash": corpus.content_hash,
                "base": {str(k): v for k, v in base_sum.items()},
                "finetuned": {str(k): v for k, v in ft_sum.items()},
                "paired_deltas": deltas,
                "pools_reordered_vs_base": changed,
                "weight_drift_max_abs": drift,
                "train_seconds": train_s,
                "prediction": {"hit@5": 0.03, "range": [0.01, 0.05]},
                "floor": {"mean_hit5": 0.02, "ci_excludes_zero": True},
                "verdict": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
        # See the note in rerank_pool_arms.py: newline=None would emit CRLF on Windows and make
        # every re-run a whole-file diff against a Linux-generated artifact.
        newline="\n",
    )
    print(f"\nwrote {args.out}\nsaved model to {args.model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
