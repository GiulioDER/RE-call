"""The compiled graph artifact: records, their vectors and edges, bound to one serving lineage.

Written once and never changed, like the atomic rescue artifact (``recall/atomic_rescue.py``), so
no database table or migration is needed and both backends share it. A directory is named by the
digest of its lineage and holds ``manifest.json``, ``records.jsonl`` and ``vectors.npy``; the
manifest carries each member's sha256, and a load refuses a mismatched lineage or a damaged member.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_SCHEMA_VERSION = 1
_MEMBERS = ("records.jsonl", "vectors.npy")


class CompiledGraphArtifactError(RuntimeError):
    """Base of the reasons an artifact cannot serve."""


class ArtifactMissing(CompiledGraphArtifactError):
    pass


class ArtifactLineageMismatch(CompiledGraphArtifactError):
    pass


class ArtifactDamaged(CompiledGraphArtifactError):
    pass


class ArtifactUnconfigured(CompiledGraphArtifactError):
    """The search process lacks the generation, model or root a serving lineage needs."""


@dataclass(frozen=True)
class ArtifactRecord:
    key: str
    source: str
    kind: str
    fields: tuple[str, ...]
    entities: tuple[str, ...]
    edges: tuple[str, ...]


@dataclass(frozen=True)
class CompiledGraphArtifact:
    lineage: Mapping[str, str]
    records: tuple[ArtifactRecord, ...]
    #: One unit-norm float32 row per record, in record order.
    vectors: Any


def artifact_dir(folder: Path, lineage: Mapping[str, str]) -> Path:
    digest = hashlib.sha256(json.dumps(dict(lineage), sort_keys=True).encode("utf-8")).hexdigest()
    return folder / f"cg-{digest[:24]}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_artifact(folder: Path, artifact: CompiledGraphArtifact) -> Path:
    """Write ``artifact`` under ``folder`` and return its directory; an existing one is kept.

    A directory without a manifest is what an erase that stopped part way leaves; it can never
    serve, so it is replaced rather than kept.
    """
    target = artifact_dir(folder, artifact.lineage)
    if (target / "manifest.json").is_file():
        return target
    if target.exists():
        shutil.rmtree(target)
    folder.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".cg-staging-", dir=folder))
    try:
        with (staging / "records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in artifact.records:
                handle.write(
                    json.dumps(
                        {
                            "key": record.key,
                            "source": record.source,
                            "kind": record.kind,
                            "fields": list(record.fields),
                            "entities": list(record.entities),
                            "edges": list(record.edges),
                        }
                    )
                    + "\n"
                )
        vectors = np.asarray(artifact.vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(artifact.records):
            raise ValueError("an artifact needs one vector row per record")
        np.save(staging / "vectors.npy", vectors, allow_pickle=False)
        manifest = {
            "schema": ARTIFACT_SCHEMA_VERSION,
            "lineage": dict(artifact.lineage),
            "records": len(artifact.records),
            "members": {name: _sha256(staging / name) for name in _MEMBERS},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def _record(row: object) -> ArtifactRecord:
    if not isinstance(row, dict):
        raise ArtifactDamaged("a record row is not an object")
    try:
        return ArtifactRecord(
            key=str(row["key"]),
            source=str(row["source"]),
            kind=str(row["kind"]),
            fields=tuple(str(value) for value in row["fields"]),
            entities=tuple(str(value) for value in row["entities"]),
            edges=tuple(str(value) for value in row["edges"]),
        )
    except (KeyError, TypeError) as exc:
        raise ArtifactDamaged("a record row lacks a field") from exc


_CACHE_MAX_ENTRIES = 2
# A search loads the artifact on every call, so a verified artifact is kept, stamped with its
# manifest's bytes: the manifest holds every member's sha256, and a directory removed by a forget
# has no manifest, so neither a rebuild nor an erase is ever served from memory.
_CACHE: OrderedDict[Path, tuple[bytes, CompiledGraphArtifact]] = OrderedDict()
_CACHE_LOCK = threading.Lock()


def load_artifact(folder: Path, lineage: Mapping[str, str]) -> CompiledGraphArtifact:
    """The artifact serving ``lineage``; raises Missing, LineageMismatch or Damaged."""
    target = artifact_dir(folder, lineage)
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactMissing(f"no compiled graph artifact for this lineage under {folder}")
    try:
        stamp = manifest_path.read_bytes()
        with _CACHE_LOCK:
            cached = _CACHE.get(target)
            if cached is not None and cached[0] == stamp:
                _CACHE.move_to_end(target)
                return cached[1]
        artifact = _verified(target, lineage, stamp)
    except FileNotFoundError as exc:
        # A forget in another process removed the directory between the check and the read.
        raise ArtifactMissing(f"the compiled graph artifact under {folder} was removed") from exc
    except OSError as exc:
        raise ArtifactDamaged(f"the compiled graph artifact could not be read: {exc}") from exc
    with _CACHE_LOCK:
        _CACHE[target] = (stamp, artifact)
        _CACHE.move_to_end(target)
        while len(_CACHE) > _CACHE_MAX_ENTRIES:
            _CACHE.popitem(last=False)
    return artifact


def _verified(target: Path, lineage: Mapping[str, str], stamp: bytes) -> CompiledGraphArtifact:
    try:
        manifest = json.loads(stamp.decode("utf-8"))
    except ValueError as exc:
        raise ArtifactDamaged("the manifest is not JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactDamaged("the manifest has an unknown schema")
    if manifest.get("lineage") != dict(lineage):
        raise ArtifactLineageMismatch("the artifact was built for another lineage")
    members = manifest.get("members")
    if not isinstance(members, dict):
        raise ArtifactDamaged("the manifest lists no members")
    for name in _MEMBERS:
        path = target / name
        if not path.is_file() or members.get(name) != _sha256(path):
            raise ArtifactDamaged(f"{name} is missing or does not match its checksum")
    rows = (target / "records.jsonl").read_text(encoding="utf-8").splitlines()
    try:
        records = tuple(_record(json.loads(row)) for row in rows if row)
    except ValueError as exc:
        raise ArtifactDamaged("a record row is not JSON") from exc
    try:
        vectors = np.load(target / "vectors.npy", allow_pickle=False)
    except ValueError as exc:
        raise ArtifactDamaged("vectors.npy is not a numpy array") from exc
    if vectors.ndim != 2 or vectors.shape[0] != len(records) or manifest.get("records") != len(records):
        raise ArtifactDamaged("vectors and records disagree")
    return CompiledGraphArtifact(lineage=dict(lineage), records=records, vectors=vectors)
