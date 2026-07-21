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
from recall.eval.metrics import false_confident_rate, fraction_true
from recall.index import Indexer
from recall.store import PgVectorStore

__all__ = ["Calibration", "CalibrationReport", "best_threshold", "calibrate",
           "loo_threshold_rates", "measure_top_cosines"]

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
    #: IN-SAMPLE false-confident rate at `suggested_threshold`. `best_threshold` minimises
    #: misclassification on exactly these samples, so on separable data this is 0.00 by
    #: arithmetic, not by measurement. Kept as a diagnostic; `fcr_heldout` is the publishable
    #: number.
    fcr_at_suggested: float
    #: Leave-one-out cross-validated rates — the honest, out-of-sample read (NaN when a class
    #: has fewer than two samples, i.e. when a fold would fit on nothing).
    fcr_heldout: float = float("nan")
    false_abstain_heldout: float = float("nan")


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


def loo_threshold_rates(
    answerable: list[float], unanswerable: list[float]
) -> tuple[float, float]:
    """Leave-one-out cross-validated ``(fcr, false_abstain)`` for the fitted gap threshold.

    `best_threshold` is an optimiser: it minimises misclassification on the samples handed to
    it. Scoring it on those same samples therefore measures the optimiser's objective, not the
    threshold's ability to generalise — on separable data the answer is 0.00 before any data is
    collected. This refits the threshold once per sample, holding that sample out, and scores
    only the held-out one:

    - **fcr** — for each unanswerable sample, refit on everything else and count it as a
      failure when the guard does NOT fire on it (``cos >= thr``, i.e. wrongly confident).
    - **false_abstain** — the mirror on the answerable side: refit without it and count a
      failure when it falls below the threshold (a real answer wrongly abstained on).

    A class with fewer than two samples cannot be cross-validated (the fold would fit on an
    empty class), so its rate is NaN rather than a fabricated score — the same convention
    `fraction_true` uses. The two classes are reported independently: a corpus can support LOO
    on one side and not the other.
    """
    fcr_flags = [
        u >= best_threshold(answerable, unanswerable[:i] + unanswerable[i + 1:])
        for i, u in enumerate(unanswerable)
    ] if len(unanswerable) >= 2 else []
    abstain_flags = [
        a < best_threshold(answerable[:i] + answerable[i + 1:], unanswerable)
        for i, a in enumerate(answerable)
    ] if len(answerable) >= 2 else []
    return fraction_true(fcr_flags), fraction_true(abstain_flags)


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
    #
    # fcr_050 is a genuine measurement: 0.50 is a fixed constant, chosen before seeing these
    # samples. fcr_sug is NOT — `thr` was fitted to `unans`, so it is reported as an in-sample
    # diagnostic and the cross-validated `fcr_heldout` is the number to publish.
    fcr_050 = false_confident_rate([mc < 0.50 for mc in unans])
    fcr_sug = false_confident_rate([mc < thr for mc in unans])
    fcr_heldout, false_abstain_heldout = loo_threshold_rates(ans, unans)
    return CalibrationReport(
        embedder.name, ans, unans, thr, fcr_050, fcr_sug, fcr_heldout, false_abstain_heldout
    )
