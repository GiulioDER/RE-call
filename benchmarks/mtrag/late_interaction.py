"""Late-interaction (ColBERT/MaxSim) reranking arms for MTRAG-human dev.

Preregistration: `docs/superpowers/specs/2026-08-07-late-interaction-rerank-design.md`.

Why this reuses `rerank_offload.cmd_dump` rather than re-running retrieval. Pool width alone
moves reranker results here (`closed-hypothesis-recall-rerank-pool-interaction-2026-08-05`: the
same MiniLM got WORSE as the pool widened, entire 95% CI below threshold). Scoring the same frozen
pools means identical pools, identical tie rule and identical metrics, with the score source as the
only variable.

`li_jina` is cc-by-nc-4.0 and its effect is declared MONOTONE in the preregistration: it can
strengthen a null or weaken a positive claim, and it can never support a decision to build the
follow-on project. `holm_family` enforces that by refusing it, rather than trusting a reader.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from recall.rerank import LATE_INTERACTION_MODELS, PERMISSIVE_LICENCES, maxsim


@dataclass(frozen=True)
class LateArm:
    """One late-interaction arm. Frozen, and declared before any score exists."""

    name: str
    checkpoint: str

    @property
    def licence(self) -> str:
        licence = LATE_INTERACTION_MODELS.get(self.checkpoint)
        if licence is None:
            raise ValueError(
                f"arm {self.name!r} names unregistered checkpoint {self.checkpoint!r}; record it "
                f"in recall.rerank.LATE_INTERACTION_MODELS with its licence first"
            )
        return licence

    @property
    def deployable(self) -> bool:
        return self.licence in PERMISSIVE_LICENCES


#: Frozen before any score was observed, per the project's preregistration standard.
LATE_ARMS: tuple[LateArm, ...] = (
    LateArm("li_colbertv2", "colbert-ir/colbertv2.0"),
    LateArm("li_answerai", "answerdotai/answerai-colbert-small-v1"),
    LateArm("li_jina", "jinaai/jina-colbert-v2"),
)


def holm_family(arms: Sequence[LateArm]) -> tuple[str, ...]:
    """The arm names forming one Holm corrected family, refusing any non-deployable arm.

    This is the containment gate. The verdict that gates the follow-on project must not be
    computable from a family containing a non-commercial checkpoint, so the impossibility is
    mechanical rather than editorial.
    """
    # Materialised before it is read twice. A single-use iterator would be exhausted by the
    # blocked scan below and the return would then be an empty tuple, which is silent omission:
    # exactly what this gate exists to prevent. The annotation says Sequence, but a gate that
    # degrades to "quietly pass" on a type violation is not a gate.
    arms = list(arms)
    blocked = [a.name for a in arms if not a.deployable]
    if blocked:
        raise ValueError(
            f"non-deployable arms cannot enter a Holm family: {blocked}. Their licences "
            f"({[a.licence for a in arms if not a.deployable]}) make them diagnostic only, and "
            f"the preregistration fixes their effect as monotone: they may strengthen a null or "
            f"weaken a positive claim, never support a build decision. Report them separately."
        )
    return tuple(a.name for a in arms)


def arm_record(arm: LateArm) -> dict[str, object]:
    """The identity block stamped onto every emitted row, so a lifted number keeps its taint."""
    return {
        "arm": arm.name,
        "checkpoint": arm.checkpoint,
        "licence": arm.licence,
        "deployable": arm.deployable,
    }


def score_stream(
    encoder: Any,
    queries: dict[str, str],
    docs: Iterable[tuple[str, str]],
    pairs: dict[str, set[str]],
    batch_size: int = 32,
) -> Iterator[dict]:
    """Score every requested `(qid, doc_id)` pair, streaming the documents.

    This is the design decision that removes the GPU rental. A cross encoder runs one forward
    pass PER PAIR (241,270 of them on 2026-08-07). Late interaction encodes the two sides
    independently, so each document is encoded ONCE and MaxSim'd against only the queries that
    reference it.

    Document token matrices are discarded after each batch. Materialising them would cost roughly
    7 GB at 128 dims (unique docs x ~180 tokens x 128 floats), and holding them buys nothing: peak
    memory here is independent of corpus size.

    `pairs` maps doc_id to {qid}, the INVERTED form of `pairs.jsonl`. Inverting it is what makes a
    single pass over documents possible.

    A pair naming an unknown query raises: it means the dump and the scorer disagree, and any
    score emitted for it would be fabricated.
    """
    qids = list(queries)
    qmatrices = dict(zip(qids, encoder.query_embed([queries[q] for q in qids]), strict=True))
    for qid, matrix in qmatrices.items():
        if matrix.shape[0] == 0:
            raise ValueError(f"query {qid!r} has no tokens")

    batch: list[tuple[str, str]] = []

    def _flush() -> Iterator[dict]:
        if not batch:
            return
        matrices = list(encoder.passage_embed([text for _, text in batch]))
        for (doc_id, _), dmatrix in zip(batch, matrices, strict=True):
            for qid in sorted(pairs[doc_id]):
                if qid not in qmatrices:
                    raise KeyError(
                        f"pair references unknown query {qid!r} for document {doc_id!r}; the dump "
                        f"and the scorer disagree, and any score emitted here would be fabricated"
                    )
                # `-inf`, not a raise, and this MUST match `LateInteractionReranker.rerank`.
                # A zero-token document sorts last there rather than aborting the batch, and the
                # validate gate reranks a pool locally and compares it against these offloaded
                # scores. If this path raised instead, the gate would find the live reranker
                # ranking a document that has no offloaded score at all, and `rerank_order`
                # refuses a candidate with no score. The two paths agree or the offload is not a
                # substitute for the real reranker.
                score = (
                    float("-inf")
                    if dmatrix.shape[0] == 0
                    else maxsim(qmatrices[qid], dmatrix)
                )
                yield {"qid": qid, "doc_id": doc_id, "score": score}
        batch.clear()

    for doc_id, text in docs:
        if doc_id not in pairs:
            continue  # no query asked for this document; encoding it would be wasted work
        batch.append((doc_id, text))
        if len(batch) >= batch_size:
            yield from _flush()
    yield from _flush()
