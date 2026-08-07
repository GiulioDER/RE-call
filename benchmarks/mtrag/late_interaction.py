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

from collections.abc import Sequence
from dataclasses import dataclass

from recall.rerank import LATE_INTERACTION_MODELS, PERMISSIVE_LICENCES


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
