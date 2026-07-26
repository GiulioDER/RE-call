"""The overnight run: one record per corpus, resumable, with failures kept in the table.

Usage on the rented box, after the BEIR zips are extracted under `--beir-root`::

    python -m recall.eval.gap_run --beir-root ./beir --out ./results/gap --dsn "$RECALL_DSN"

Each corpus writes `<out>/<dataset>.json` as it finishes, so the job resumes where it died rather
than re-embedding everything. Re-running is safe and cheap.

The parameters below are frozen in `docs/superpowers/specs/2026-07-26-embedder-gap-predictor-design.md`
and were fixed before any gap was measured. Changing one is a restatement, not a tweak.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from recall.eval.beir import materialize
from recall.eval.vocab import (
    bge_encoder,
    code_density,
    crowding,
    oov_rate,
    query_overlap,
    strip_code,
    subword_pieces,
)

_log = logging.getLogger(__name__)

#: Preregistered. See the spec's PREREGISTRATION section for why each value, decided before any
#: BEIR gap existed.
MAX_DOCS = 5_000          # subsample cap; both embedders see the identical subsample
TOKEN_BUDGET = 2_000      # oov_rate size control; rank ordering is stable 16k->2k (rho=1.000)
MIN_PIECES = 2            # "the tokenizer has no whole-word entry for this"
CROWDING_SAMPLE = 500     # fixed across corpora: sampling changes what "nearest" means
SEED = 20260726

#: The primary response arm. `hybrid` is what §7/§8 published and what a user deploys; `dense`
#: isolates the embedder; `bm25` is embedder-independent and therefore a free control — a gap that
#: tracks BM25 strength is a corpus-difficulty story, not a vocabulary one.
PRIMARY_ARM = "hybrid"

#: Below this, the preregistered power calculation says a null result means "underpowered" rather
#: than "no effect": at n=8, |partial r| must reach 0.85 to survive Holm over three predictors.
POWER_FLOOR = 12

DATASETS = [
    "nfcorpus", "scifact", "scidocs", "fiqa", "arguana",
    *(f"cqadupstack-{f}" for f in (
        "android", "english", "gaming", "gis", "mathematica", "physics",
        "programmers", "stats", "tex", "unix", "webmasters", "wordpress",
    )),
]


def pending_datasets(
    datasets: list[str], out_dir: Path, *, retry_failed: bool = False
) -> list[str]:
    """Which datasets still need running.

    A completed corpus is skipped so the job resumes instead of restarting; twenty corpora is most
    of a night and a crash at fourteen must not re-embed the first thirteen.

    A *failed* corpus is skipped too, unless asked for explicitly. Automatic retry is how a corpus
    that dies forty minutes into embedding burns the night twice. The manifest is what makes
    failures loud — retrying is not.
    """
    out: list[str] = []
    for name in datasets:
        path = out_dir / f"{name}.json"
        if not path.exists():
            out.append(name)
            continue
        if retry_failed:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") != "ok":
                out.append(name)
    return out


def failure_record(dataset: str, exc: BaseException) -> dict[str, Any]:
    """A failed corpus, kept in the table rather than dropped from it.

    The study reports `n`. A corpus that vanishes because its download 404'd silently lowers `n`
    and invalidates every power calculation in the spec without anything having raised.
    """
    return {
        "status": "failed",
        "dataset": dataset,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "traceback": traceback.format_exc(),
    }


def summarise(datasets: list[str], out_dir: Path) -> dict[str, Any]:
    """What actually landed, with `usable` stated rather than inferred from a list length."""
    ok, failed, missing = [], [], []
    for name in datasets:
        path = out_dir / f"{name}.json"
        if not path.exists():
            missing.append(name)
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        (ok if record.get("status") == "ok" else failed).append(name)

    underpowered = len(ok) < POWER_FLOOR
    return {
        "attempted": len(datasets),
        "usable": len(ok),
        "ok": ok,
        "failed": failed,
        "missing": missing,
        "underpowered": underpowered,
        "power_note": (
            f"n={len(ok)} is below the preregistered floor of {POWER_FLOOR}: a null result here "
            f"means UNDERPOWERED, not 'no effect' (at n=8 a partial r of 0.85 is needed to survive "
            f"Holm over three predictors). Report it as such."
            if underpowered else
            f"n={len(ok)} meets the preregistered floor of {POWER_FLOOR}."
        ),
    }


def compute_predictors(corpus_dir: Path, questions: list[dict], embedder: Any) -> dict[str, float]:
    """The three candidate predictors plus the code-density diagnostic, for one corpus."""
    texts = [p.read_text(encoding="utf-8", errors="ignore") for p in sorted(corpus_dir.glob("*.txt"))]
    encode = bge_encoder()
    tokenize = lambda word: subword_pieces(word, encode)  # noqa: E731

    stripped = [strip_code(t) for t in texts]
    by_name = {p.name: t for p, t in zip(sorted(corpus_dir.glob("*.txt")), stripped)}

    overlaps = []
    for question in questions:
        for filename in question["relevant_files"]:
            gold = by_name.get(filename)
            if gold is not None:
                value = query_overlap(question["query"], gold)
                if value == value:  # not NaN
                    overlaps.append(value)

    sample = stripped[:CROWDING_SAMPLE]
    vectors = embedder.embed(sample) if sample else []

    return {
        "oov_rate": oov_rate(stripped, tokenize, min_pieces=MIN_PIECES,
                             token_budget=TOKEN_BUDGET, seed=SEED),
        "query_overlap": (sum(overlaps) / len(overlaps)) if overlaps else float("nan"),
        "crowding": crowding(vectors, sample=CROWDING_SAMPLE, seed=SEED),
        "code_density": (sum(code_density(t) for t in texts) / len(texts)) if texts else float("nan"),
        "n_documents": len(texts),
        "n_overlap_pairs": len(overlaps),
    }


def default_embedders() -> tuple[Any, Any]:
    """The study's `(local, cloud)` pair, constructed lazily.

    Separated from `run_corpus` so the orchestration can be exercised end to end with a cheap
    embedder: the integration test needs to prove that materialise -> evaluate -> predictors fit
    together, which is a question about plumbing and not about retrieval quality. Discovering a
    wiring bug at 2am on a rented box is the failure this indirection buys out of.
    """
    from recall.embeddings import FastEmbedEmbedder, VoyageEmbedder

    return FastEmbedEmbedder(), VoyageEmbedder()


def run_corpus(
    dataset: str,
    beir_root: Path,
    work_root: Path,
    dsn: str,
    *,
    embedders: tuple[Any, Any] | None = None,
    max_docs: int = MAX_DOCS,
) -> dict[str, Any]:
    """Materialize one BEIR dataset, score it with both embedders, and compute its predictors."""
    from recall.eval.labelled import evaluate

    corpus_dir = work_root / dataset
    manifest = materialize(beir_root / dataset, corpus_dir, max_docs=max_docs, seed=SEED)
    questions = json.loads((corpus_dir / "questions.json").read_text(encoding="utf-8"))
    if not questions:
        raise RuntimeError(f"{dataset}: no scorable questions after materialisation")

    local, cloud = embedders if embedders is not None else default_embedders()
    started = time.perf_counter()
    scores: dict[str, Any] = {}
    for label, embedder in (("local", local), ("cloud", cloud)):
        report = evaluate(dsn, corpus_dir, questions, embedder, k=5, glob="**/*.txt")
        scores[label] = {
            arm: report["arms"][arm]["hit_at_5"]["rate"] for arm in ("bm25", "dense", "sparse", "hybrid")
        }
        scores[label]["mrr_hybrid"] = report["arms"]["hybrid"]["mrr"]
        scores[label]["held_out"] = report["questions"]["held_out"]

    return {
        "status": "ok",
        "dataset": dataset,
        "manifest": manifest,
        "scores": scores,
        "predictors": compute_predictors(corpus_dir, questions, local),
        "primary_arm": PRIMARY_ARM,
        "seconds": round(time.perf_counter() - started, 1),
        "params": {"max_docs": max_docs, "token_budget": TOKEN_BUDGET,
                   "min_pieces": MIN_PIECES, "crowding_sample": CROWDING_SAMPLE, "seed": SEED},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beir-root", required=True, type=Path, help="directory of EXTRACTED BEIR datasets")
    ap.add_argument("--out", required=True, type=Path, help="one <dataset>.json is written here per corpus")
    ap.add_argument("--work", type=Path, default=None, help="where materialised corpora go (default: <out>/corpora)")
    ap.add_argument("--dsn", default=None, help="Postgres DSN; falls back to RECALL_DSN")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import os

    dsn = args.dsn or os.environ.get("RECALL_DSN")
    if not dsn:
        raise SystemExit("no DSN: pass --dsn or set RECALL_DSN")

    datasets = args.datasets or DATASETS
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.work or (args.out / "corpora")

    todo = pending_datasets(datasets, args.out, retry_failed=args.retry_failed)
    _log.info("%d/%d dataset(s) to run", len(todo), len(datasets))

    for i, dataset in enumerate(todo, 1):
        _log.info("[%d/%d] %s", i, len(todo), dataset)
        try:
            record = run_corpus(dataset, args.beir_root, work, dsn)
            _log.info("  %s: local=%.3f cloud=%.3f (%s arm)", dataset,
                      record["scores"]["local"][PRIMARY_ARM],
                      record["scores"]["cloud"][PRIMARY_ARM], PRIMARY_ARM)
        except Exception as exc:  # one corpus must not take the night down
            _log.exception("  %s FAILED", dataset)
            record = failure_record(dataset, exc)
        (args.out / f"{dataset}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    summary = summarise(datasets, args.out)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log.info("done: %d usable of %d attempted; failed=%s missing=%s",
              summary["usable"], summary["attempted"], summary["failed"], summary["missing"])
    if summary["underpowered"]:
        _log.warning(summary["power_note"])


if __name__ == "__main__":
    main()
