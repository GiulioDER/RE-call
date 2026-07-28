"""The installed versions of every package whose numerics can move a published result.

`_provenance` already records WHICH CODE produced an artifact — the `#81`/`#84` generation. That
is not enough for any result that passes through a model. `results/locomo/postfix_abstention.json`
publishes four abstention modes; two of them route through a QNLI cross-encoder, and the artifact
names no `sentence-transformers`, `transformers` or `torch` version at all. Re-running it on an
asserted-clean corpus moved the `entail` row by **0.0525** while the calibrated thresholds
reproduced exactly — so the corpus was not the variable, and nothing recorded says what was.

Two of four published rows are therefore not reproducible by anyone, on any machine, and the
artifact gives no signal that this is so. Same class as the `mem0ai` pin: a version that decides a
published number has to travel inside the artifact. `mem0ai` does, via `importlib.metadata`. These
did not.

Note what is already pinned and is therefore NOT the gap: the default judge's Hub revision
(`recall.entailment.DEFAULT_QNLI_REVISION`) fixes the weights, so the model itself is immutable,
and `git log -S` puts that pin before the artifact. The gap is the stack that runs it.

Three candidate causes were then eliminated by measurement rather than argument. Corpus doubling:
the calibrated thresholds are fit on the distribution doubling moves and reproduce exactly.
`sentence-transformers`: a second run pinned to 5.6.0 in an isolated venv returned **every mode to
four decimals identical** to the 5.6.1 run — so the patch is not the cause, and, more usefully,
**this pipeline is deterministic across independent environments**. Run-to-run noise is therefore
not the explanation either.

What remains is most consistent with an independent HNSW index build — a different graph returns
slightly different k=5 neighbours, which moves the fitted thresholds and the texts handed to the
judge, while leaving `default` (a floor) and `both` (threshold-dominated) almost unmoved. That is
nondeterminism §5b and §6 already document. It is also, now, **unfalsifiable**: the 07-26 index no
longer exists. Which is the whole argument for this module — the one record that could have settled
it is the one nobody wrote.

Absent packages are omitted rather than recorded as null: a run that never imported
`sentence-transformers` is not a run whose sentence-transformers version was unknown.
"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

#: Packages whose version can change a number this repo publishes. Embedders and cross-encoders
#: both land here: `fastembed`/`onnxruntime` decide the vectors, the torch stack decides the judge
#: and the reranker. Add to this list when a new dependency starts deciding a result — not when a
#: new dependency is merely installed.
TRACKED = (
    "sentence-transformers",
    "transformers",
    "torch",
    "fastembed",
    "onnxruntime",
)


def model_stack(packages: tuple[str, ...] = TRACKED) -> dict[str, str]:
    """Installed versions of `packages`, skipping the ones that are not installed."""
    found: dict[str, str] = {}
    for name in packages:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def generated_at() -> str:
    """When this run happened, in UTC.

    The companion question to `model_stack`, and the one that cost the most to answer without it.
    Deciding whether `postfix_abstention.json` predated the double-index guard meant reading git
    for the commit that ADDED the file — and a commit date is when someone committed, not when the
    run happened. For `3ee36ed` those differ by an unknown amount, which is precisely the gap that
    let a 07-26 run and a 07-28 guard pass each other unnoticed.

    Seconds resolution: a result is not identified by its milliseconds, and a shorter string is one
    a reader actually compares against a commit date.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
