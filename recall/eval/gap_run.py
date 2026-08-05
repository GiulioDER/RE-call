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
import math
import re
import os
import time
import traceback
from pathlib import Path
from typing import Any

from recall.eval.beir import materialize
from recall.eval.gap_study import POWER_FLOOR as _POWER_FLOOR
from recall.eval.gap_study import PRIMARY_ARM as _PRIMARY_ARM
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
MAX_DOCS = 20_000        # COMMON subsample cap; raised from 5_000 on 2026-07-26 (see spec restatement)
TOKEN_BUDGET = 2_000      # oov_rate size control; rank ordering is stable 16k->2k (rho=1.000)
MIN_PIECES = 2            # "the tokenizer has no whole-word entry for this"
CROWDING_SAMPLE = 500     # fixed across corpora: sampling changes what "nearest" means
SEED = 20260726

#: Re-exported from the analysis module, NOT redeclared. Both live here and there previously, and
#: both drive the same preregistered `underpowered` verdict on the same n — so a change to one
#: would have left the run summary and the analysis output disagreeing about the study's own power,
#: with nothing raising. A frozen constant with two definitions is not frozen.
#:
#: `PRIMARY_ARM`: `hybrid` is what §7/§8 published and what a user deploys; `dense` isolates the
#: embedder; `bm25` is embedder-independent and therefore a free control — a gap that tracks BM25
#: strength is a corpus-difficulty story, not a vocabulary one.
#: `POWER_FLOOR`: below this, the preregistered power calculation says a null result means
#: "underpowered" rather than "no effect": at n=8, |partial r| must reach 0.85 to survive Holm.
PRIMARY_ARM = _PRIMARY_ARM
POWER_FLOOR = _POWER_FLOOR

DATASETS = [
    "nfcorpus", "scifact", "scidocs", "fiqa", "arguana",
    *(f"cqadupstack-{f}" for f in (
        "android", "english", "gaming", "gis", "mathematica", "physics",
        "programmers", "stats", "tex", "unix", "webmasters", "wordpress",
    )),
]


def write_json(path: Path, payload: Any) -> None:
    """Write one artifact, atomically, and refuse to emit invalid JSON.

    Two guards, each for a failure this study already produced.

    `allow_nan=False`: `json.dumps` will otherwise emit a bare `NaN` token, which is not JSON
    (RFC 8259). Python reads it back happily, so the file looks fine from here and rejects in
    `jq`, `JSON.parse`, Go, Rust and Postgres `jsonb` — a reviewer who cannot open the artifact
    cannot check the claim. NaN means "not measured" and `null` is how JSON says that, so the
    payload is mapped rather than the check disabled.

    Temp-file + `os.replace`: the documented operating mode is sixteen `nohup`'d workers on a
    rented box, so a kill mid-write is the EXPECTED path, not the exotic one. A torn file is
    worse than a missing one because `pending_datasets` treats existence as completion and would
    skip that corpus as done forever.
    """
    encoded = json.dumps(_nan_to_null(payload), indent=2, allow_nan=False)
    tmp = path.with_name(path.name + ".tmp")
    # newline="\n" explicitly: the default translates on Windows, which would rewrite every line
    # of a committed artifact as CRLF and bury a one-value correction in a whole-file diff.
    tmp.write_text(encoded, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _nan_to_null(value: Any) -> Any:
    """Recursively replace non-finite floats with None, so `allow_nan=False` can stay on."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_to_null(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nan_to_null(v) for v in value]
    return value


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
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            # An unreadable file is not a completed corpus. Existence alone used to be the
            # completion marker, so a worker killed mid-write left a torn file that was skipped
            # as done forever and then raised a JSONDecodeError in `summarise` — after the
            # corpus had already been dropped from the run. Treating it as pending self-heals.
            _log.warning("%s: unreadable result, re-running", path)
            out.append(name)
            continue
        if record.get("status") is None or (retry_failed and record.get("status") != "ok"):
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
        "error": _scrub(str(exc)),
        "error_type": type(exc).__name__,
        # Redacted and bounded, because `results/gap/` is a git-TRACKED directory and this run
        # holds a password-bearing DSN and an API key in its environment. An exception message
        # that happens to carry either would otherwise be committed verbatim and published.
        "traceback": _scrub(traceback.format_exc())[-2000:],
    }


#: A Postgres URI's credentials, and long opaque token-shaped strings (API keys).
_SECRET_PATTERNS = (
    re.compile(r"(?P<scheme>[a-z+]+://)[^:/@\s]+:[^@/\s]+@", re.IGNORECASE),
    re.compile(r"\b(?:sk|pa|voy)-[A-Za-z0-9_\-]{16,}\b"),
)


def _scrub(text: str) -> str:
    """Remove credentials from text that is about to be written into a committed artifact."""
    text = _SECRET_PATTERNS[0].sub(r"\g<scheme>***:***@", text)
    return _SECRET_PATTERNS[1].sub("***", text)


def summarise(datasets: list[str], out_dir: Path) -> dict[str, Any]:
    """What actually landed, with `usable` stated rather than inferred from a list length."""
    ok: list[str] = []
    failed: list[str] = []
    missing: list[str] = []
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
        # Record the denominator of the RATES stored above, not the size of the fit/held split.
        # `held_out` is `len(questions[1::2])`; the rates are over `retrieval_scored_on`, which
        # follows `labelled.evaluate(score_retrieval_on=...)`. Under the default they coincide
        # for an all-answerable set like BEIR, and under `"all"` they differ by 2x — so recording
        # `held_out` beside a rate it does not divide is how an artifact silently mislabels its
        # own n. Both are kept: `held_out` for continuity with the committed pre-change records,
        # `retrieval_scored_on` because it is the one that names these rates.
        scores[label]["held_out"] = report["questions"]["held_out"]
        scores[label]["retrieval_scored_on"] = report["questions"]["retrieval_scored_on"]
        scores[label]["score_retrieval_on"] = report["questions"]["score_retrieval_on"]

    return {
        "status": "ok",
        "dataset": dataset,
        "manifest": manifest,
        "scores": scores,
        "predictors": compute_predictors(corpus_dir, questions, local),
        "primary_arm": PRIMARY_ARM,
        "seconds": round(time.perf_counter() - started, 1),
        # The two embedders are the ONE variable this study manipulates, and until now no
        # artifact recorded them: the names were only implied by `default_embedders()`'s
        # defaults. But that indirection exists precisely so a cheap test embedder can be
        # injected — so a real overnight record and a wired-up test record were indistinguishable
        # after the fact. Read off the objects actually used, not off the defaults.
        "embedders": {
            "local": getattr(local, "name", type(local).__name__),
            "cloud": getattr(cloud, "name", type(cloud).__name__),
        },
        "params": {"max_docs": max_docs, "token_budget": TOKEN_BUDGET,
                   "min_pieces": MIN_PIECES, "crowding_sample": CROWDING_SAMPLE, "seed": SEED},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # NOT `required=True`: `--summarise` reads only the records already on disk, and argparse
    # would reject that invocation before the flag was ever inspected. Since a `--datasets` worker
    # no longer writes summary.json, `--summarise` is now the ONLY path that produces it — an
    # unreachable one would leave the study with no summary at all. Checked below instead.
    ap.add_argument("--beir-root", type=Path, default=None,
                    help="directory of EXTRACTED BEIR datasets (required unless --summarise)")
    ap.add_argument("--out", required=True, type=Path, help="one <dataset>.json is written here per corpus")
    ap.add_argument("--work", type=Path, default=None, help="where materialised corpora go (default: <out>/corpora)")
    ap.add_argument("--dsn", default=None, help="Postgres DSN; falls back to RECALL_DSN")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument(
        "--summarise", action="store_true",
        help="write summary.json over the FULL preregistered roster and exit; run once, after "
             "all workers have finished",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.summarise:
        args.out.mkdir(parents=True, exist_ok=True)
        _write_summary(DATASETS, args.out)
        return

    if args.beir_root is None:
        raise SystemExit("--beir-root is required (it is optional only for --summarise)")

    dsn = args.dsn or os.environ.get("RECALL_DSN")
    if not dsn:
        raise SystemExit("no DSN: pass --dsn or set RECALL_DSN")

    # Checked HERE, next to the DSN, and not left to surface from inside the per-corpus
    # `except Exception` below. Without this a missing key produced one failure record per
    # corpus — each after materialising up to MAX_DOCS files to disk — then `usable: 0`,
    # `underpowered: true`, and exit 0. A whole night, a full artifact set, and a green exit
    # code for a run that could never have measured anything.
    if not os.environ.get("VOYAGE_API_KEY"):
        raise SystemExit(
            "no VOYAGE_API_KEY: the cloud arm cannot run. Export it before starting, or the "
            "whole roster fails one corpus at a time and the run still exits 0."
        )

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
        write_json(args.out / f"{dataset}.json", record)

    if args.datasets:
        # A worker that was handed a SUBSET must not write the study-wide summary. The launcher
        # runs one process per corpus against a shared `--out`, so every worker would overwrite
        # `summary.json` with a summary of its own single dataset and the last writer would win.
        # That is not hypothetical: the committed summary read `attempted: 1, usable: 1,
        # underpowered: true` while eighteen corpus records sat in the same directory and the
        # published finding reported n=17. The one artifact whose job is to state `n` instead of
        # letting it be inferred was the one misstating it.
        _log.info("subset run (%s): summary.json not written — run `--summarise` once at the end",
                  ",".join(datasets))
        return

    _write_summary(datasets, args.out)


def _write_summary(datasets: list[str], out_dir: Path) -> None:
    summary = summarise(datasets, out_dir)
    write_json(out_dir / "summary.json", summary)
    _log.info("done: %d usable of %d attempted; failed=%s missing=%s",
              summary["usable"], summary["attempted"], summary["failed"], summary["missing"])
    if summary["underpowered"]:
        _log.warning(summary["power_note"])


if __name__ == "__main__":
    main()
