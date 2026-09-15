"""Fit the registered production shadow source model from a frozen audit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.source_conditioning import (  # noqa: E402
    REGISTERED_ALPHA,
    REGISTERED_CANDIDATE_K,
    REGISTERED_ITEM_BUDGET,
    REGISTERED_RAW_COSINE_FLOOR,
    REGISTERED_RRF_K,
    SOURCE_CONDITIONING_MODEL_ID,
    SOURCE_CONDITIONING_SCHEMA_VERSION,
    SOURCE_FEATURE_NAMES,
    SourceConditioningArtifact,
)
from scripts.run_live_source_conditioned_admission import (  # noqa: E402
    _fit_source_model,
    _source_rows,
)


EXPECTED_TRAINING_SHA256 = "531dfa31c4344df020ebda9be46e2b09365b0dd9c918aff6835f8c5b4e1bd666"
EXPECTED_TRAINING_GENERATION = "gen_18d5edd2e5e847c0af1ee37e40d27893"
EXPECTED_TRAINING_CALIBRATION = "cal_23d8708ac550444fa4274ac617df0870"
EXPECTED_PIPELINE = "57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86"


def fit_artifact(raw: bytes) -> SourceConditioningArtifact:
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_TRAINING_SHA256:
        raise ValueError("training artifact differs from the registered digest")
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("generation_id") != EXPECTED_TRAINING_GENERATION:
        raise ValueError("training generation differs")
    if payload.get("calibration_id") != EXPECTED_TRAINING_CALIBRATION:
        raise ValueError("training calibration differs")
    if payload.get("pipeline_fingerprint") != EXPECTED_PIPELINE:
        raise ValueError("training pipeline differs")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 50:
        raise ValueError("training artifact must contain the registered 50 rows")
    source_rows: list[dict[str, Any]] = [
        source_row for row in rows for source_row in _source_rows(row)
    ]
    means, scales, coefficients = _fit_source_model(source_rows)
    artifact = SourceConditioningArtifact(
        schema_version=SOURCE_CONDITIONING_SCHEMA_VERSION,
        model_id=SOURCE_CONDITIONING_MODEL_ID,
        feature_names=SOURCE_FEATURE_NAMES,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        intercept=float(coefficients[0]),
        coefficients=tuple(float(value) for value in coefficients[1:]),
        alpha=REGISTERED_ALPHA,
        raw_cosine_floor=REGISTERED_RAW_COSINE_FLOOR,
        rrf_k=REGISTERED_RRF_K,
        candidate_k=REGISTERED_CANDIDATE_K,
        item_budget=REGISTERED_ITEM_BUDGET,
        embedding_profile="voyage-context-4-v1",
        retrieval_profile="fast",
        training_generation_id=str(payload["generation_id"]),
        training_calibration_id=str(payload["calibration_id"]),
        training_pipeline_fingerprint=str(payload["pipeline_fingerprint"]),
        training_corpus_fingerprint=str(payload["corpus_fingerprint"]),
        training_query_set_digest=str(payload["query_set_sha256"]),
        training_fact_labels_digest=str(payload["fact_labels_sha256"]),
        training_artifact_sha256=digest,
        training_source_rows=len(source_rows),
        artifact_fingerprint="",
    )
    return artifact.with_fingerprint()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="docs/results/2026-09-13-live-source-scoped-expansion.json",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifact = fit_artifact(Path(args.input).read_bytes())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(artifact.to_json(), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "artifact_fingerprint": artifact.artifact_fingerprint,
                "training_source_rows": artifact.training_source_rows,
            }
        )
    )


if __name__ == "__main__":
    main()
