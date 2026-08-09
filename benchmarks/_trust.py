"""The trust configuration every benchmark arm retrieves under, decided once.

Two facts have to hold together for a benchmark to measure anything, and getting either one alone
is worse than getting neither, which is why they live in one function instead of at four call
sites.

**Development mode, or nothing runs.** `TrustPolicy` defaults to strict, deliberately: a policy
object that defaulted to permissive would make every caller that forgot one fail open. A benchmark
corpus has no tenant generation and no published calibration artifact bound to it — that is what a
benchmark IS, a corpus nobody has certified yet — so under strict every query raises
`TrustRefusal: INDEX_NOT_READY` before retrieval runs. `recall.eval._research_trust.research_search`
is the seam that exists for this, and the arms opt into it there.

**An explicit calibration, or the measurement inverts.** Development mode with `calibration=None`
means "no threshold exists at all", and `recall.trust` answers that by overwriting every verdict to
``unverified`` and forcing ``abstained=False``. Both benchmark readings then break, in opposite
directions and both silently:

* an arm reading ``result.abstained`` (``benchmarks/systems.py``, ``recall_isolation``) sees a flat
  ZERO abstention rate, and reports a dead gate as a perfect one;
* an arm filtering on ``verdict == "ok"`` (``recall_temporal``, ``benchmarks/ladder``) sees NOTHING
  clear the filter, cites nothing, and scores zero everywhere.

Passing a `Calibration` takes the other branch in `recall.trust`, which preserves the verdicts and
the abstention flag — the code there calls it "the path every abstention benchmark measures".

The threshold is the library's own `DEFAULT_GAP_THRESHOLD`. That is the constant `evaluate` fell
back to before the trust gate existed, so an arm routed through here measures the same gate it
always did, now stated rather than inherited: `benchmarks/ladder/report.py` already DISCLOSES
`UNCALIBRATED_BGE_SMALL_FLOOR = 0.50` as the threshold its published numbers ran with.

⚠️ It is untuned and not comparable across embedders: 0.50 sits at the 0th percentile of five of
six measured top-1 distributions and the 16th of the sixth, so it barely fires on most embedders
and starves a sixth of queries on one. An arm that has fitted a real calibration passes it in and
keeps it. The long-term answer is a calibration step against the deployed corpus after the
embedder and reranker are chosen, not a better constant here.

If you are writing a SERVING path and you find yourself importing this, that is the bug — the same
warning `_research_trust` carries, for the same reason.
"""
from __future__ import annotations

from typing import Any

from recall.calibration import Calibration
from recall.embeddings import embedding_profile_id
from recall.eval._research_trust import RESEARCH_POLICY, research_search
from recall.guards import DEFAULT_GAP_THRESHOLD
from recall.types import TrustedResult


def uncalibrated_floor(embedder: Any) -> Calibration:
    """The library's untuned abstention floor, bound to this embedder's profile id.

    The PROFILE id, not the CLI string: a fitted calibration file is keyed by profile
    (`fastembed` resolves to `bge-small-symmetric-v1`), so anything recorded under the flag's
    spelling would name a key no calibration ever uses.
    """
    return Calibration(embedder=embedding_profile_id(embedder), threshold=DEFAULT_GAP_THRESHOLD)


def bench_search(store: Any, embedder: Any, query: str, **kwargs: Any) -> TrustedResult:
    """`research_search` with an explicit calibration. The entry point every arm retrieves through.

    Both defaults are defaults, not impositions: a caller passing a real `calibration` keeps it
    (`benchmarks/beam/systems.py` loads a fitted artifact), and a caller passing a real `policy`
    keeps it, which is how an arm that deliberately measures strict-mode refusal still works.
    Passing `None` for either is treated as not passing it at all — see below for why that is the
    dangerous spelling rather than the harmless one.
    """
    # `is None` rather than `setdefault`, on BOTH keys. `setdefault` leaves a PRESENT None alone,
    # and None is the spelling that reads like "use the default" while meaning the opposite in
    # each case: `calibration=None` is "no threshold exists at all", which blanks every verdict,
    # and `policy=None` resolves through `None or TrustPolicy()` to STRICT, which refuses every
    # query. An arm threading an optional `X | None = None` parameter through would hit whichever
    # one it forgot. Absent and None must mean the same thing here, or this helper hands back the
    # two failures it exists to prevent.
    if kwargs.get("calibration") is None:
        kwargs["calibration"] = uncalibrated_floor(embedder)
    if kwargs.get("policy") is None:
        kwargs["policy"] = RESEARCH_POLICY
    return research_search(store, embedder, query, **kwargs)
