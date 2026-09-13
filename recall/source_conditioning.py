"""Versioned source support model and deterministic shadow selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


SOURCE_CONDITIONING_SCHEMA_VERSION = 1
SOURCE_CONDITIONING_MODEL_ID = "source-logistic-v1"
SOURCE_FEATURE_NAMES = (
    "max_cosine",
    "best_dense_rr",
    "best_sparse_rr",
    "rrf_mass",
    "cross_leg_fraction",
    "log_chunk_count",
    "source_margin",
)
REGISTERED_ALPHA = 0.08
REGISTERED_RAW_COSINE_FLOOR = 0.30
REGISTERED_RRF_K = 60
REGISTERED_CANDIDATE_K = 20
REGISTERED_ITEM_BUDGET = 5


class SourceConditioningArtifactError(ValueError):
    """A source conditioning artifact is malformed, tampered, or incompatible."""


def _finite_tuple(value: object, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise SourceConditioningArtifactError(f"{name} must contain exactly {length} numbers")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise SourceConditioningArtifactError(f"{name} contains a non-number")
        number = float(item)
        if not math.isfinite(number):
            raise SourceConditioningArtifactError(f"{name} contains a non-finite number")
        result.append(number)
    return tuple(result)


def _required_string(data: Mapping[str, object], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value:
        raise SourceConditioningArtifactError(f"{name} must be a non-empty string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SourceConditioningArtifactError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise SourceConditioningArtifactError(f"{name} must be finite")
    return number


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourceConditioningArtifactError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class SourceConditioningArtifact:
    """Immutable fitted source model with its training and serving contract."""

    schema_version: int
    model_id: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    alpha: float
    raw_cosine_floor: float
    rrf_k: int
    candidate_k: int
    item_budget: int
    embedding_profile: str
    retrieval_profile: str
    training_generation_id: str
    training_calibration_id: str
    training_pipeline_fingerprint: str
    training_corpus_fingerprint: str
    training_query_set_digest: str
    training_fact_labels_digest: str
    training_artifact_sha256: str
    training_source_rows: int
    artifact_fingerprint: str

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "alpha": self.alpha,
            "raw_cosine_floor": self.raw_cosine_floor,
            "rrf_k": self.rrf_k,
            "candidate_k": self.candidate_k,
            "item_budget": self.item_budget,
            "embedding_profile": self.embedding_profile,
            "retrieval_profile": self.retrieval_profile,
            "training_generation_id": self.training_generation_id,
            "training_calibration_id": self.training_calibration_id,
            "training_pipeline_fingerprint": self.training_pipeline_fingerprint,
            "training_corpus_fingerprint": self.training_corpus_fingerprint,
            "training_query_set_digest": self.training_query_set_digest,
            "training_fact_labels_digest": self.training_fact_labels_digest,
            "training_artifact_sha256": self.training_artifact_sha256,
            "training_source_rows": self.training_source_rows,
        }

    def computed_fingerprint(self) -> str:
        encoded = json.dumps(
            self._payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_fingerprint": self.artifact_fingerprint}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    def with_fingerprint(self) -> "SourceConditioningArtifact":
        return replace(self, artifact_fingerprint=self.computed_fingerprint())

    def assert_compatible(
        self,
        *,
        pipeline_fingerprint: str,
        embedding_profile: str,
        retrieval_profile: str,
        candidate_k: int,
    ) -> None:
        mismatches: list[str] = []
        if pipeline_fingerprint != self.training_pipeline_fingerprint:
            mismatches.append("pipeline_fingerprint")
        if embedding_profile != self.embedding_profile:
            mismatches.append("embedding_profile")
        if retrieval_profile != self.retrieval_profile:
            mismatches.append("retrieval_profile")
        if candidate_k != self.candidate_k:
            mismatches.append("candidate_k")
        if mismatches:
            raise SourceConditioningArtifactError(
                "incompatible source model: " + ", ".join(mismatches)
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SourceConditioningArtifact":
        expected_keys = {
            "schema_version",
            "model_id",
            "feature_names",
            "means",
            "scales",
            "intercept",
            "coefficients",
            "alpha",
            "raw_cosine_floor",
            "rrf_k",
            "candidate_k",
            "item_budget",
            "embedding_profile",
            "retrieval_profile",
            "training_generation_id",
            "training_calibration_id",
            "training_pipeline_fingerprint",
            "training_corpus_fingerprint",
            "training_query_set_digest",
            "training_fact_labels_digest",
            "training_artifact_sha256",
            "training_source_rows",
            "artifact_fingerprint",
        }
        if set(data) != expected_keys:
            missing = sorted(expected_keys - set(data))
            extra = sorted(set(data) - expected_keys)
            raise SourceConditioningArtifactError(
                f"source model fields differ: missing={missing}, extra={extra}"
            )
        if data["schema_version"] != SOURCE_CONDITIONING_SCHEMA_VERSION:
            raise SourceConditioningArtifactError("unsupported source model schema_version")
        if data["model_id"] != SOURCE_CONDITIONING_MODEL_ID:
            raise SourceConditioningArtifactError("unsupported source model model_id")
        feature_names_raw = data["feature_names"]
        if not isinstance(feature_names_raw, list) or tuple(feature_names_raw) != SOURCE_FEATURE_NAMES:
            raise SourceConditioningArtifactError("source model feature_names differ")
        means = _finite_tuple(data["means"], "means", len(SOURCE_FEATURE_NAMES))
        scales = _finite_tuple(data["scales"], "scales", len(SOURCE_FEATURE_NAMES))
        if any(value <= 0.0 for value in scales):
            raise SourceConditioningArtifactError("source model scales must be positive")
        coefficients = _finite_tuple(
            data["coefficients"], "coefficients", len(SOURCE_FEATURE_NAMES)
        )
        intercept_values = _finite_tuple([data["intercept"]], "intercept", 1)

        exact_numbers = {
            "alpha": REGISTERED_ALPHA,
            "raw_cosine_floor": REGISTERED_RAW_COSINE_FLOOR,
            "rrf_k": REGISTERED_RRF_K,
            "candidate_k": REGISTERED_CANDIDATE_K,
            "item_budget": REGISTERED_ITEM_BUDGET,
        }
        for name, expected in exact_numbers.items():
            value = data[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != expected:
                raise SourceConditioningArtifactError(f"source model {name} differs")
        training_source_rows = data["training_source_rows"]
        if (
            isinstance(training_source_rows, bool)
            or not isinstance(training_source_rows, int)
            or training_source_rows < 1
        ):
            raise SourceConditioningArtifactError("training_source_rows must be positive")

        artifact = cls(
            schema_version=SOURCE_CONDITIONING_SCHEMA_VERSION,
            model_id=SOURCE_CONDITIONING_MODEL_ID,
            feature_names=SOURCE_FEATURE_NAMES,
            means=means,
            scales=scales,
            intercept=intercept_values[0],
            coefficients=coefficients,
            alpha=_number(data["alpha"], "alpha"),
            raw_cosine_floor=_number(data["raw_cosine_floor"], "raw_cosine_floor"),
            rrf_k=_integer(data["rrf_k"], "rrf_k"),
            candidate_k=_integer(data["candidate_k"], "candidate_k"),
            item_budget=_integer(data["item_budget"], "item_budget"),
            embedding_profile=_required_string(data, "embedding_profile"),
            retrieval_profile=_required_string(data, "retrieval_profile"),
            training_generation_id=_required_string(data, "training_generation_id"),
            training_calibration_id=_required_string(data, "training_calibration_id"),
            training_pipeline_fingerprint=_required_string(
                data, "training_pipeline_fingerprint"
            ),
            training_corpus_fingerprint=_required_string(data, "training_corpus_fingerprint"),
            training_query_set_digest=_required_string(data, "training_query_set_digest"),
            training_fact_labels_digest=_required_string(data, "training_fact_labels_digest"),
            training_artifact_sha256=_required_string(data, "training_artifact_sha256"),
            training_source_rows=training_source_rows,
            artifact_fingerprint=_required_string(data, "artifact_fingerprint"),
        )
        if artifact.artifact_fingerprint != artifact.computed_fingerprint():
            raise SourceConditioningArtifactError("source model fingerprint mismatch")
        return artifact

    @classmethod
    def from_json(cls, value: str | bytes) -> "SourceConditioningArtifact":
        try:
            data = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SourceConditioningArtifactError("source model is not valid JSON") from exc
        if not isinstance(data, dict):
            raise SourceConditioningArtifactError("source model root must be an object")
        return cls.from_dict(data)


def load_source_conditioning_artifact(path: str | Path) -> SourceConditioningArtifact:
    return SourceConditioningArtifact.from_json(Path(path).read_bytes())


def source_features(
    pool: Sequence[Mapping[str, object]],
    dense: Sequence[Mapping[str, object]],
    sparse: Sequence[Mapping[str, object]],
    *,
    candidate_k: int = REGISTERED_CANDIDATE_K,
    rrf_k: int = REGISTERED_RRF_K,
) -> dict[str, tuple[float, ...]]:
    """Build the registered seven features from one query's candidate traces."""

    def ranks(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
        result: dict[str, int] = {}
        for item in rows:
            rank = _integer(item["rank"], "candidate rank")
            if rank <= candidate_k:
                result[str(item["chunk_id"])] = rank
        return result

    dense_ranks = ranks(dense)
    sparse_ranks = ranks(sparse)
    by_source: dict[str, list[Mapping[str, object]]] = {}
    for item in pool:
        by_source.setdefault(str(item["source"]), []).append(item)
    if not by_source:
        return {}

    mass: dict[str, float] = {}
    for source, items in by_source.items():
        total = 0.0
        for item in items:
            chunk_id = str(item["chunk_id"])
            if chunk_id in dense_ranks:
                total += 1.0 / (rrf_k + dense_ranks[chunk_id])
            if chunk_id in sparse_ranks:
                total += 1.0 / (rrf_k + sparse_ranks[chunk_id])
        mass[source] = total

    features: dict[str, tuple[float, ...]] = {}
    for source, items in by_source.items():
        chunk_ids = {str(item["chunk_id"]) for item in items}
        dense_values = [dense_ranks[value] for value in chunk_ids if value in dense_ranks]
        sparse_values = [sparse_ranks[value] for value in chunk_ids if value in sparse_ranks]
        both = sum(value in dense_ranks and value in sparse_ranks for value in chunk_ids)
        strongest_other = max((value for key, value in mass.items() if key != source), default=0.0)
        cosines = [_number(item["cosine"], "candidate cosine") for item in items]
        if any(not math.isfinite(value) for value in cosines):
            raise SourceConditioningArtifactError("candidate cosine is non-finite")
        features[source] = (
            max(cosines),
            0.0 if not dense_values else 1.0 / (rrf_k + min(dense_values)),
            0.0 if not sparse_values else 1.0 / (rrf_k + min(sparse_values)),
            mass[source],
            both / len(chunk_ids),
            math.log1p(len(chunk_ids)),
            mass[source] - strongest_other,
        )
    return features


def source_support(
    artifact: SourceConditioningArtifact,
    features: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for source, values in features.items():
        if len(values) != len(artifact.feature_names):
            raise SourceConditioningArtifactError("source feature vector has wrong length")
        logit = artifact.intercept
        for value, mean, scale, coefficient in zip(
            values, artifact.means, artifact.scales, artifact.coefficients
        ):
            number = float(value)
            if not math.isfinite(number):
                raise SourceConditioningArtifactError("source feature is non-finite")
            logit += ((number - mean) / scale) * coefficient
        clipped = min(40.0, max(-40.0, logit))
        result[source] = 1.0 / (1.0 + math.exp(-clipped))
    return result


def select_source_conditioned(
    artifact: SourceConditioningArtifact,
    pool: Sequence[Mapping[str, object]],
    dense: Sequence[Mapping[str, object]],
    sparse: Sequence[Mapping[str, object]],
    *,
    threshold: float,
) -> list[dict[str, object]]:
    features = source_features(
        pool,
        dense,
        sparse,
        candidate_k=artifact.candidate_k,
        rrf_k=artifact.rrf_k,
    )
    supports = source_support(artifact, features)
    eligible: list[dict[str, object]] = []
    for item in pool:
        if item.get("verdict") not in {"ok", "low_confidence"}:
            continue
        cosine = _number(item["cosine"], "candidate cosine")
        support = supports[str(item["source"])]
        adjusted = cosine + artifact.alpha * (support - 0.5)
        if cosine < artifact.raw_cosine_floor or adjusted < threshold:
            continue
        eligible.append(
            {**item, "source_support": support, "adjusted_score": adjusted}
        )
    eligible.sort(
        key=lambda item: (
            -_number(item["adjusted_score"], "adjusted score"),
            _integer(item["pool_rank"], "pool rank"),
        )
    )
    return eligible[: artifact.item_budget]


def chunk_identifier_hash(chunk_id: object) -> str:
    return hashlib.sha256(str(chunk_id).encode("utf-8")).hexdigest()


__all__ = [
    "REGISTERED_ALPHA",
    "REGISTERED_CANDIDATE_K",
    "REGISTERED_ITEM_BUDGET",
    "REGISTERED_RAW_COSINE_FLOOR",
    "REGISTERED_RRF_K",
    "SOURCE_CONDITIONING_MODEL_ID",
    "SOURCE_CONDITIONING_SCHEMA_VERSION",
    "SOURCE_FEATURE_NAMES",
    "SourceConditioningArtifact",
    "SourceConditioningArtifactError",
    "chunk_identifier_hash",
    "load_source_conditioning_artifact",
    "select_source_conditioned",
    "source_features",
    "source_support",
]
