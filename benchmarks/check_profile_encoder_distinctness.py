"""Do two registered embedding profiles actually produce different vectors?

`bge-small-symmetric-v1` and `bge-small-asymmetric-v1` are declared in
`recall/embedding_registry.py` with identical `model_name`, `dimension`, `context_mode`,
`normalization`, `instruction_version` and `chunker_version`. They differ in exactly two fields:
`query_mode` and `passage_mode`. Every other input to a stored vector is shared. So the WHOLE
retrieval difference between an arm on one and an arm on the other is whatever the backend's
`query_embed` / `passage_embed` do differently from `embed`.

If they do nothing differently, a paired comparison of those two profiles is a comparison of a
system with itself: `hit@5` is equal on every question by construction, the paired bootstrap
interval is exactly `[0, 0]`, and the promotion gate refuses for a reason that says nothing about
either profile. That is a null-difference control, and this repository already has one
(`results/promotion/decision.null-difference.json`). Running it a second time under a candidate's
name would publish the control as an experiment.

This script answers the question against the deployed artifact rather than against the code. It
deliberately does NOT import the `recall` package: the subject is what the backend does, and
asking the code under test whether its two configurations differ is the self-comparison this
program keeps recording.

**Prior work: [[project-recall-embedding-profile-registry-2026-08-05]]** (searched
`docs_search(source_type="memory")`, top cosine 0.81, no gap warning). Session 8 already measured
this on VPS2 and recorded "the asymmetric profiles are asymmetric in name only". This is not a
re-measurement for its own sake: that finding is now load-bearing for a whole campaign's design,
and a fact that decides whether an experiment is worth running is the one to confirm rather than
inherit. The two runs used different probe texts and this one is committed, so the next session
re-checks it with a command instead of a memory.

Two controls, both blocking:

* **Positive.** The same comparator, on two texts that must not agree. A comparator that reports
  "identical" for those is broken, and every "identical" above it would be worthless. The script
  refuses to report a verdict if this does not fire.
* **Determinism.** The same call twice must agree byte for byte. Without it an "identical"
  reading could be luck and a "different" reading could be float noise.

Run it where the artifact is provisioned::

    /opt/recall-enterprise/venv/bin/python -m benchmarks.check_profile_encoder_distinctness

Exit status is 0 whenever the controls held and a verdict was produced, whichever verdict that
is, matching `recall.eval.promotion`'s convention that an answer of "no" is an answer. A failed
control exits 2, because then there is no measurement at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections.abc import Callable, Iterable
from typing import Any

#: The provisioned tree on the host this program's artifacts live on. A default, not a claim: the
#: digest of whatever tree is used is reported in the output so a reader can tell which one ran.
DEFAULT_CACHE_DIR = "/opt/recall-enterprise/models/bge-fastembed-cache"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

#: Query-shaped and passage-shaped probes. Both shapes on purpose: an encoder that applied an
#: instruction prefix to queries only would show a difference on the first group and not the
#: second, and probing one shape could not tell those apart.
QUERIES: tuple[str, ...] = (
    "what did we decide about the cache?",
    "who approved the retry policy change",
    "why was postgres chosen over the alternatives",
)
PASSAGES: tuple[str, ...] = (
    "The cache decision was recorded after a two week trial: entries are keyed on the complete "
    "embedding identity rather than the profile id alone.",
    "Retry policy: exponential backoff capped at thirty seconds, with a dead letter queue after "
    "five attempts. Approved by the platform team.",
    "We chose PostgreSQL with pgvector because the operational surface was already understood "
    "and no second datastore had to be run.",
)


def _vector(encode: Callable[[list[str]], Iterable[Any]], text: str) -> Any:
    import numpy as np

    produced = list(encode([text]))
    if len(produced) != 1:
        raise SystemExit(f"expected exactly one vector for one text, got {len(produced)}")
    return np.asarray(produced[0], dtype=np.float32)


def _sha(vector: Any) -> str:
    return hashlib.sha256(vector.tobytes()).hexdigest()


def _cosine(left: Any, right: Any) -> float:
    import numpy as np

    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def measure(cache_dir: str, model_name: str) -> dict[str, Any]:
    """Probe the three encoders and both controls. Raises `SystemExit(2)` if a control fails."""
    import numpy as np
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, local_files_only=True)

    methods = {name: hasattr(model, name) for name in ("embed", "query_embed", "passage_embed")}
    absent = sorted(name for name, present in methods.items() if not present)
    if absent:
        raise SystemExit(
            f"backend exposes no {absent}; `bge-small-asymmetric-v1` would refuse to build here, "
            f"which is a different finding and not the one this script measures"
        )

    rows: list[dict[str, Any]] = []
    for kind, texts in (("query", QUERIES), ("passage", PASSAGES)):
        for text in texts:
            symmetric = _vector(model.embed, text)
            as_query = _vector(model.query_embed, text)
            as_passage = _vector(model.passage_embed, text)
            rows.append(
                {
                    "probe_kind": kind,
                    "text_prefix": text[:44],
                    "dimension": int(symmetric.shape[0]),
                    "sha256_embed": _sha(symmetric)[:16],
                    "sha256_query_embed": _sha(as_query)[:16],
                    "sha256_passage_embed": _sha(as_passage)[:16],
                    "bytes_embed_eq_query_embed": symmetric.tobytes() == as_query.tobytes(),
                    "bytes_embed_eq_passage_embed": symmetric.tobytes() == as_passage.tobytes(),
                    "cosine_embed_query_embed": round(_cosine(symmetric, as_query), 12),
                    "cosine_embed_passage_embed": round(_cosine(symmetric, as_passage), 12),
                }
            )

    left = _vector(model.embed, QUERIES[0])
    right = _vector(model.embed, PASSAGES[0])
    positive_control = {
        "bytes_equal": left.tobytes() == right.tobytes(),
        "cosine": round(_cosine(left, right), 12),
        "sha256_left": _sha(left)[:16],
        "sha256_right": _sha(right)[:16],
    }
    if positive_control["bytes_equal"] or float(positive_control["cosine"]) > 0.999:
        print(json.dumps({"positive_control": positive_control}, indent=2), file=sys.stderr)
        raise SystemExit(2)

    determinism = {
        "embed_twice_byte_identical": (
            _vector(model.embed, QUERIES[0]).tobytes()
            == _vector(model.embed, QUERIES[0]).tobytes()
        )
    }
    if not determinism["embed_twice_byte_identical"]:
        print(json.dumps({"determinism_control": determinism}, indent=2), file=sys.stderr)
        raise SystemExit(2)

    identical = all(
        row["bytes_embed_eq_query_embed"] and row["bytes_embed_eq_passage_embed"] for row in rows
    )
    return {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "fastembed": __import__("fastembed").__version__,
            "onnxruntime": __import__("onnxruntime").__version__,
            "numpy": np.__version__,
            "cache_dir": cache_dir,
            "model_name": model_name,
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        },
        "methods_present": methods,
        "probes": rows,
        "positive_control_two_different_texts": positive_control,
        "determinism_control": determinism,
        "all_probes_identical": identical,
        "verdict": (
            "query_embed and passage_embed are BYTE-IDENTICAL to embed on every probe: "
            "bge-small-asymmetric-v1 and bge-small-symmetric-v1 cannot produce different vectors "
            "on this backend, so a paired comparison of the two is a null by construction"
            if identical
            else "at least one probe differs: the two profiles are genuinely distinct here"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_profile_encoder_distinctness", description=__doc__)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    # Set before fastembed is imported, so a missing artifact fails as a missing artifact rather
    # than being quietly fetched. Runtime model downloads are on this program's out-of-scope list.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print(json.dumps(measure(args.cache_dir, args.model_name), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
