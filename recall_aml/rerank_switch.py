"""RK-1: switch C9's Voyage reranker on for an experiment, and choose its model.

Round two, 2026-10-19 (recall-lab pre-registration ``2026-10-19-rk1-voyage-rerank``). C9 builds a
Voyage reranker at startup and never calls it: the variant sets ``reranker=False``. The AML reader
leans on the top of the returned list (BEAM probe, 2026-09-24), and answer accuracy with a reranker
was never measured on Textual. Two variables, both unset by default, so the served behaviour is
unchanged:

* ``RECALL_AML_RERANKER`` (1/0) overrides the variant's ``reranker`` per Search;
* ``RECALL_AML_RERANK_MODEL`` picks the Voyage model at startup, from ``RERANK_MODELS``.

A reranker reorders without rescoring, so `HostedRetriever.search` reassigns the fused scores in
descending order to the reranked list: the returned scores stay non-increasing.
"""

from __future__ import annotations

from collections.abc import Mapping

RERANKER_ENV = "RECALL_AML_RERANKER"
RERANK_MODEL_ENV = "RECALL_AML_RERANK_MODEL"
#: The served default since the reranker was wired; unchanged unless the variable is set.
DEFAULT_RERANK_MODEL = "rerank-2.5"
#: Voyage's current rerankers (docs.voyageai.com/docs/reranker, read 2026-10-19).
RERANK_MODELS = ("rerank-2.5", "rerank-2.5-lite", "rerank-3", "rerank-3-lite")


def rerank_model(environ: Mapping[str, str]) -> str:
    """The Voyage rerank model ``RECALL_AML_RERANK_MODEL`` names, else the served default."""
    value = environ.get(RERANK_MODEL_ENV, "").strip() or DEFAULT_RERANK_MODEL
    if value not in RERANK_MODELS:
        raise ValueError(f"{RERANK_MODEL_ENV} must be one of {', '.join(RERANK_MODELS)}, not {value!r}")
    return value
