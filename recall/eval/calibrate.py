"""Per-embedder gap-threshold calibration.

The gap guard fires when the best dense cosine for a query falls below a threshold. A fixed
threshold does not transfer across embedders: a model whose cosines cluster high needs a higher
threshold. This module measures the answerable vs. unanswerable max-cosine distributions and
suggests a threshold that separates them.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from recall.calibration import best_threshold
from recall.embeddings import Embedder
from recall.eval.metrics import false_confident_rate
from recall.index import Indexer
from recall.store import PgVectorStore

__all__ = ["Calibration", "CalibrationReport", "best_threshold", "calibrate",
           "measure_top_cosines"]

EVAL_DIR = Path(__file__).parent


@dataclass
class CalibrationReport:
    """Measurement REPORT (raw samples + FCRs) — distinct from the runtime artifact
    `recall.calibration.Calibration` (embedder/threshold/scale)."""

    embedder: str
    answerable_max_cos: list[float]
    unanswerable_max_cos: list[float]
    suggested_threshold: float
    fcr_at_050: float
    fcr_at_suggested: float


Calibration = CalibrationReport  # backward-compat alias (pre-v0.2 name)


def measure_top_cosines(
    store: PgVectorStore, embedder: Embedder, queries: list[dict]
) -> tuple[list[float], list[float]]:
    """Best dense cosine per labeled query -> (answerable, unanswerable) sample lists.

    Single source of the sampling rule for both `calibrate()` and the trust evaluation —
    validity-sensitive (`trust`) queries carry no answerable label and are skipped.
    """
    ans: list[float] = []
    unans: list[float] = []
    for q in queries:
        if q.get("trust"):
            continue
        hits = store.query_dense(embedder.embed([q["query"]])[0], k=1)
        top = hits[0].score if hits else 0.0
        (ans if q["answerable"] else unans).append(top)
    return ans, unans


def calibrate(
    dsn: str, embedder: Embedder, corpus_dir: Path | None = None, queries_path: Path | None = None
) -> CalibrationReport:
    """Measure the best dense cosine per query (answerable vs unanswerable) and suggest a gap
    threshold that separates them. `CalibrationReport.answerable_max_cos`/`unanswerable_max_cos` are the
    raw per-query top-cosine samples; `fcr_at_050`/`fcr_at_suggested` are the false-confident rates
    at the default 0.50 vs the suggested threshold.
    """
    corpus_dir = corpus_dir or (EVAL_DIR / "corpus")
    queries = json.loads(
        Path(queries_path or (EVAL_DIR / "queries.json")).read_text(encoding="utf-8")
    )
    table = "cal_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(dsn, dim=embedder.dim, table=table)
    try:
        store.ensure_schema()
        Indexer(store, embedder).index_path(corpus_dir)
        ans, unans = measure_top_cosines(store, embedder, queries)
    finally:
        try:
            store._conn.execute(f"DROP TABLE IF EXISTS {table}")
        except Exception:
            pass  # best-effort drop of the throwaway uuid table
        finally:
            store.close()

    thr = best_threshold(ans, unans)
    # gap_warning fires when the best cosine is below the threshold; on an unanswerable query a
    # working guard fires (gap=True). false_confident_rate counts the ones where it did NOT fire.
    fcr_050 = false_confident_rate([mc < 0.50 for mc in unans])
    fcr_sug = false_confident_rate([mc < thr for mc in unans])
    return CalibrationReport(embedder.name, ans, unans, thr, fcr_050, fcr_sug)
