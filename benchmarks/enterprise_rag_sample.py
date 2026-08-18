"""Draw a reproducible stratified sample of EnterpriseRAG questions.

The published `dense_floor_strat100` sample is the **first ten question ids of each category
block**: `basic` qst_0001-0010 of 175, `semantic` qst_0176-0185 of 125, and so on. That is the
sorted-then-truncated head bias this project has been bitten by twice, and no sampler existed in
the tree at all, so the sample could not be reproduced or audited.

⚠️ The head of a category block is not a random ten. Question ids in this benchmark are assigned
in authoring order, so the head is whatever was written first, which correlates with whatever the
author was thinking about that day. Nothing guarantees it is representative, and nothing detects
it if it is not.

This module draws instead with an explicit seed, and records the seed so the draw can be repeated
exactly. It refuses a stratum it cannot fill rather than silently returning a short one, because a
short stratum is what makes a population-weighted rate quietly wrong.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Any, Protocol


class _Question(Protocol):
    question_id: str


def stratify(
    questions: Sequence[Any],
    *,
    per_stratum: int,
    seed: int,
    key: str = "question_type",
    type_of: Any = None,
    expect: Any = None,
) -> list[Any]:
    """`per_stratum` questions from every stratum, drawn with `seed`, ordered by question id.

    `type_of` is the accessor for a question's stratum, defaulting to `getattr(q, key)`. It is a
    parameter because the benchmark derives `question_type` through a helper rather than storing
    it on the object.

    Refuses rather than truncating:

    - a stratum with fewer than `per_stratum` members, because the caller's population weights
      assume the design it asked for and a short stratum silently changes the denominator;
    - `per_stratum` below 1, which would return an empty sample that every downstream rate would
      then divide by.

    The returned list is sorted by `question_id` so the artifact is diffable, but the SELECTION is
    random: sorting the output does not reintroduce head bias, sorting the input before truncating
    is what did.
    """
    if per_stratum < 1:
        raise ValueError(f"per_stratum must be >= 1, got {per_stratum}")
    if not questions:
        raise ValueError("cannot stratify an empty question set")

    accessor = type_of if type_of is not None else (lambda q: getattr(q, key))
    strata: dict[str, list[Any]] = {}
    for question in questions:
        strata.setdefault(str(accessor(question)), []).append(question)

    # ⚠️ `expect` closes the hole that the short-stratum check alone leaves open: a stratum with
    # ZERO members is absent from `strata`, so iterating what is present can never notice it.
    # `--limit-questions 300` produces exactly that input, because this benchmark's ids are laid
    # out in contiguous category blocks, and the run would then spend its whole embedding budget
    # before `summarize` refused the payload at the write.
    absent = sorted(set(expect) - set(strata)) if expect is not None else []
    short = {name: len(members) for name, members in strata.items() if len(members) < per_stratum}
    if absent or short:
        raise ValueError(
            f"cannot draw {per_stratum} from every stratum; absent: {absent}; "
            f"short: {sorted(short.items())}. Refusing rather than returning a partial sample, "
            "because a stratum missing or smaller than the design silently changes what a "
            "population-weighted rate is a rate OVER."
        )

    chosen: list[Any] = []
    for name in sorted(strata):
        # One Random per stratum, seeded by (seed, stratum name), so adding a stratum or changing
        # `per_stratum` does not reshuffle the strata drawn before it. A single shared generator
        # would make every sample depend on the iteration order of every other.
        rng = random.Random(f"{seed}:{name}")
        chosen.extend(rng.sample(strata[name], per_stratum))
    return sorted(chosen, key=lambda q: q.question_id)


def sampling_provenance(
    questions: Sequence[Any], chosen: Sequence[Any], *, per_stratum: int, seed: int
) -> dict[str, Any]:
    """What the artifact must record for the draw to be repeatable by someone else."""
    return {
        "design": "stratified without replacement, one RNG per stratum seeded by (seed, stratum)",
        "per_stratum": per_stratum,
        "seed": seed,
        "population_n": len(questions),
        "sample_n": len(chosen),
        "question_ids": [q.question_id for q in chosen],
    }


def head_of_each_stratum(
    questions: Sequence[Any], *, per_stratum: int, type_of: Any
) -> list[Any]:
    """The OLD selection, kept so the two designs can be compared rather than argued about.

    ⛔ Not for new measurements. It exists because the published artifact was drawn this way, and
    a claim that the head differs from a random draw should be measured on the same corpus rather
    than asserted from first principles.
    """
    strata: dict[str, list[Any]] = {}
    for question in questions:
        strata.setdefault(str(type_of(question)), []).append(question)
    chosen: list[Any] = []
    for name in sorted(strata):
        chosen.extend(sorted(strata[name], key=lambda q: q.question_id)[:per_stratum])
    return sorted(chosen, key=lambda q: q.question_id)


def strata_of(questions: Iterable[Any], *, type_of: Any) -> dict[str, int]:
    """Population count per stratum, which is what the weights must agree with."""
    counts: dict[str, int] = {}
    for question in questions:
        counts[str(type_of(question))] = counts.get(str(type_of(question)), 0) + 1
    return counts
