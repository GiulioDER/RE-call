"""Measure how far a frozen abstention threshold survives a corpus that keeps changing.

The registered question is in `docs/preregistrations/2026-08-21-calibration-drift-trigger.md`:
`recall.calibration_v2.DEFAULT_MAX_CORPUS_DELTA` is 0.25 by assertion, on one tenant at one delta,
and nothing in the tree measures what a delta costs a threshold that is not allowed to move.

The shape of the harness follows from three things the repository already measured.

* **The outcome must be the per-class error of the FIXED cut, never the AUC.** Separability is
  threshold-free, so a change that lifts every unanswerable score by the same amount leaves it at
  1.00 while sliding the whole class over the threshold. That is
  `memory/separability-cannot-see-a-shifted-class.md`, found by a test written to fail that passed.
* **Search is exact here, not HNSW.** Not because ANN is unrealistic, but because its build is
  nondeterministic (issue #26 measured coverage swinging 0.40 to 0.84 across rebuilds on one host),
  and a signal smaller than that noise cannot be seen through it. Every number this produces is
  therefore a LOWER bound on the drift a real deployment sees.
* **The chunker must be the one the index would use.** `recall/wizard/queryset.py` states why:
  generating questions with `chunk_text` and indexing with `chunk_code` produced 20 chunks against
  8 with no exact string in common, so every "answerable" query was about text that was not in the
  index it was scored against.

Snapshots come from real history rather than from simulated edits, because the thing under test is
whether *the way corpora actually change* moves a threshold, and a synthetic delta is a statement
about the generator. Git supplies two of the three; the memory store has no history, so its files
are revealed in mtime order, which is a reconstruction and is labelled as one.

    python -m benchmarks.calibration_drift --sizing          # chunk counts only, no embedding
    python -m benchmarks.calibration_drift --out results/calibration_drift_2026-08-21.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recall.calibration import (
    MIN_SEPARABILITY,
    from_samples,
    separability,
    separability_interval,
)
from recall.calibration_v2 import (
    DEFAULT_MAX_CARRY_FORWARD_ERROR,
    DEFAULT_MAX_CORPUS_DELTA,
    corpus_delta,
    threshold_error_rates,
)
from recall.index import chunk_code, chunk_text
from recall.wizard.queryset import QuerySetError, generate_offline

#: Queries per class. Twice the certification floor, matching `queryset.DEFAULT_PER_CLASS`: the
#: Hanley-McNeil interval is wide at n=20 and it is the interval's lower bound that certification
#: tests, so a genuinely separable set can still be refused for thinness.
PER_CLASS = 40

#: Fixed so the whole measurement replays. `generate_offline` samples chunks without replacement.
QUERY_SEED = 0

#: How many points on each corpus's history. Not a free parameter: embedding is the entire cost of
#: this harness, and every extra snapshot adds whatever chunks changed since the last one. Run
#: `--sizing` before raising it.
DEFAULT_SNAPSHOTS = 24

#: Smallest corpus the BASELINE may be fitted on. `generate_offline` oversamples to
#: `max(per_class * 4, per_class + 32)` chunks, so below this it either refuses outright or draws
#: every question from nearly every chunk.
#:
#: This is not a convenience. Both git corpora begin at a commit holding one file, and a threshold
#: fitted to a 15-chunk corpus is not a calibration anybody would deploy; measuring its decay would
#: be a statement about the first commit rather than about drift. Earlier snapshots are dropped and
#: the count is reported, so a reader can see how much history the baseline choice cost.
MIN_BASELINE_CHUNKS = 4 * 40

#: `BAAI/bge-small-en-v1.5` under fastembed. One embedder only, and thresholds are known not to
#: transfer across embedders (`memory/calibrated-thresholds-and-the-overlap.md`), so nothing here
#: claims the SHAPE of the curve transfers either.
EMBEDDER = "BAAI/bge-small-en-v1.5"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """One corpus state: a label, and every source in it as `(path, bytes)`."""

    label: str
    #: `path -> content`. Path is repository-relative and stable across snapshots, so a file edited
    #: in place keeps its identity and counts as `modified` rather than as a remove plus an add.
    sources: dict[str, bytes]

    def manifest_objects(self) -> list[dict[str, str]]:
        """The `(uri, sha256)` pairs `corpus_delta` compares, in manifest shape.

        `file://` URIs rather than bare paths, because that is what a local generation's manifest
        actually carries (`recall/lineage.py`), and a delta measured over a different identity than
        the one production uses would be a number about this harness.
        """
        return [
            {"uri": f"file:///{path}", "sha256": _sha256(body)}
            for path, body in sorted(self.sources.items())
        ]


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=True,
        # Blobs are read as bytes elsewhere; every call routed through here returns porcelain text.
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def _evenly_spaced(items: Sequence[Any], count: int) -> list[Any]:
    """`count` items spanning `items`, first and last always included.

    Evenly spaced over the COMMIT SEQUENCE rather than over calendar time. Both are defensible;
    this one is chosen because the predictor under test is a corpus delta, and commits are what
    move a corpus. Spacing by date would oversample the days nobody committed.
    """
    if count >= len(items):
        return list(items)
    if count < 2:
        return [items[-1]]
    step = (len(items) - 1) / (count - 1)
    # `round` can collide on adjacent positions for a short history, so the indices are deduped
    # BEFORE they are used. Deduping the drawn items instead would be wrong for any sequence that
    # can legitimately repeat a value, and deduping on `id()` would additionally depend on which
    # objects CPython happens to intern.
    positions = sorted({round(index * step) for index in range(count)})
    return [items[position] for position in positions]


class GitHistory:
    """Snapshots of one path prefix, taken at commits that actually touched it.

    Blobs are addressed by their git object id and read at most once, no matter how many snapshots
    contain them. That is what makes 16 snapshots of a 2,700-chunk corpus affordable: across a
    history, the overwhelming majority of files are unchanged from the previous snapshot, and an
    unchanged file must not be re-read, re-chunked or re-embedded.
    """

    def __init__(self, repo: Path, prefix: str, suffix: str) -> None:
        self.repo = repo
        self.prefix = prefix
        self.suffix = suffix
        self._blob_cache: dict[str, bytes] = {}

    def commits(self) -> list[str]:
        out = _git(self.repo, "rev-list", "--reverse", "HEAD", "--", self.prefix)
        return [line.strip() for line in out.splitlines() if line.strip()]

    def _tree(self, commit: str) -> list[tuple[str, str]]:
        """`(path, blob_id)` for every file under the prefix with the right suffix."""
        out = _git(self.repo, "ls-tree", "-r", commit, "--", self.prefix)
        entries: list[tuple[str, str]] = []
        for line in out.splitlines():
            if not line.strip():
                continue
            meta, _, path = line.partition("\t")
            fields = meta.split()
            if len(fields) < 3 or fields[1] != "blob":
                continue  # submodule or tree entry; neither is a source
            if not path.endswith(self.suffix):
                continue
            entries.append((path, fields[2]))
        return entries

    def _blob(self, blob_id: str) -> bytes:
        cached = self._blob_cache.get(blob_id)
        if cached is None:
            cached = subprocess.run(
                ["git", "-C", str(self.repo), "cat-file", "blob", blob_id],
                capture_output=True,
                check=True,
            ).stdout
            self._blob_cache[blob_id] = cached
        return cached

    def snapshots(self, count: int) -> list[Snapshot]:
        chosen = _evenly_spaced(self.commits(), count)
        out: list[Snapshot] = []
        for commit in chosen:
            tree = self._tree(commit)
            if not tree:
                continue  # a commit before the prefix existed; not a corpus state
            out.append(
                Snapshot(commit[:12], {path: self._blob(blob) for path, blob in tree})
            )
        return out


class MtimeGrowth:
    """Snapshots of a directory with no history, by revealing files in modification order.

    ⚠️ **A reconstruction, not a history.** An mtime is when a file was last written, so a memo
    edited on day fourteen appears on day fourteen even if it was created on day one. What this
    reproduces faithfully is the SHAPE of an append-mostly store growing, which is the change mode
    the other two corpora do not cover; it does not reproduce any particular past state.
    """

    def __init__(self, root: Path, suffix: str) -> None:
        self.root = root
        self.suffix = suffix

    def snapshots(self, count: int) -> list[Snapshot]:
        files = sorted(
            (path for path in self.root.rglob(f"*{self.suffix}") if path.is_file()),
            key=lambda path: (path.stat().st_mtime, str(path)),
        )
        if not files:
            return []
        # Cut points are over the file sequence, so each snapshot is a strict prefix of the next
        # and the delta between consecutive snapshots is pure addition by construction.
        cuts = _evenly_spaced(list(range(1, len(files) + 1)), count)
        out: list[Snapshot] = []
        for cut in cuts:
            visible = files[:cut]
            out.append(
                Snapshot(
                    f"n{cut:04d}",
                    {
                        str(path.relative_to(self.root)).replace("\\", "/"): path.read_bytes()
                        for path in visible
                    },
                )
            )
        return out


# --------------------------------------------------------------------------------------
# Embedding, cached by chunk content
# --------------------------------------------------------------------------------------


class ChunkEmbedder:
    """Embeds chunk text at most once per distinct chunk, across every snapshot and corpus.

    The cache key is the chunk's own sha256 rather than its file's, because a one-line edit to a
    long document leaves most of its chunks byte-identical. Measured while sizing this harness,
    that is the difference between an affordable run and an overnight one.

    Persisted to disk so an interrupted run resumes instead of restarting. The file holds only
    vectors keyed by content digest, so it is safe to delete and expensive to lose.
    """

    def __init__(self, cache_path: Path, model_name: str = EMBEDDER) -> None:
        import numpy as np

        self._np = np
        self.cache_path = cache_path
        self.model_name = model_name
        self._vectors: dict[str, Any] = {}
        self._model: Any = None
        self._dirty = 0
        if cache_path.exists():
            with np.load(cache_path) as data:
                self._vectors = {key: data[key] for key in data.files}

    @property
    def cached(self) -> int:
        return len(self._vectors)

    def _load_model(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name)
        return self._model

    def embed(self, texts: Sequence[str], *, progress: str = "") -> Any:
        """Unit-normalised row vectors for `texts`, in order."""
        np = self._np
        digests = [_sha256(text.encode("utf-8")) for text in texts]
        missing_by_digest: dict[str, str] = {}
        for digest, text in zip(digests, texts, strict=True):
            if digest not in self._vectors:
                missing_by_digest[digest] = text
        if missing_by_digest:
            model = self._load_model()
            pending = list(missing_by_digest.items())
            started = time.time()
            done = 0
            for offset in range(0, len(pending), 256):
                block = pending[offset : offset + 256]
                for (digest, _text), vector in zip(
                    block, model.embed([item[1] for item in block], batch_size=32), strict=True
                ):
                    array = np.asarray(vector, dtype=np.float32)
                    norm = float(np.linalg.norm(array))
                    self._vectors[digest] = array / norm if norm else array
                done += len(block)
                self._dirty += len(block)
                rate = done / max(time.time() - started, 1e-9)
                print(
                    f"  {progress} embedded {done}/{len(pending)} new chunks "
                    f"({rate:.1f}/s)",
                    file=sys.stderr,
                    flush=True,
                )
                if self._dirty >= 2000:
                    self.flush()
        return np.stack([self._vectors[digest] for digest in digests])

    def flush(self) -> None:
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Written to a sibling and moved, so an interrupt cannot leave a half-written cache that
        # `np.load` then refuses on the next run.
        temporary = self.cache_path.with_suffix(".tmp.npz")
        self._np.savez(temporary, **self._vectors)
        temporary.replace(self.cache_path)
        self._dirty = 0


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------


@dataclass
class CorpusSpec:
    id: str
    describe: str
    chunker: Callable[[str], list[str]]
    snapshots: Callable[[int], list[Snapshot]]
    change_mode: str


def _chunks_of(snapshot: Snapshot, chunker: Callable[[str], list[str]]) -> list[str]:
    chunks: list[str] = []
    for _path, body in sorted(snapshot.sources.items()):
        chunks.extend(chunker(body.decode("utf-8", errors="replace")))
    return chunks


def _top1(embedder: ChunkEmbedder, chunks: Sequence[str], queries: Any, *, tag: str) -> Any:
    """Exact best cosine per query against `chunks`.

    Exact rather than approximate on purpose; see the module docstring. Both sides are unit
    normalised at embedding time, so the inner product IS the cosine and no second normalisation
    can silently disagree with the first.
    """
    matrix = embedder.embed(chunks, progress=tag)
    return (queries @ matrix.T).max(axis=1)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation, ties averaged. Spearman rather than Pearson because the registered claim
    is about whether delta ORDERS the error, not about a linear coefficient with units."""

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        out = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            average = (position + end) / 2 + 1
            for index in range(position, end + 1):
                out[order[index]] = average
            position = end + 1
        return out

    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return num / den if den else float("nan")


@dataclass
class CorpusResult:
    corpus_id: str
    describe: str
    change_mode: str
    baseline: dict[str, Any] = field(default_factory=dict)
    points: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    #: Snapshots discarded because they preceded the first corpus big enough to calibrate on.
    dropped_snapshots: int = 0


def measure_corpus(
    spec: CorpusSpec, embedder: ChunkEmbedder, snapshot_count: int
) -> CorpusResult:
    result = CorpusResult(spec.id, spec.describe, spec.change_mode)
    every_snapshot = spec.snapshots(snapshot_count)
    # The baseline is the first snapshot big enough to fit a calibration anybody would deploy, not
    # simply the first one. See `MIN_BASELINE_CHUNKS`.
    chunk_counts = [len(_chunks_of(snapshot, spec.chunker)) for snapshot in every_snapshot]
    viable = [
        index for index, count in enumerate(chunk_counts) if count >= MIN_BASELINE_CHUNKS
    ]
    if not viable:
        result.error = (
            f"no snapshot reaches {MIN_BASELINE_CHUNKS} chunks; the largest has "
            f"{max(chunk_counts, default=0)}"
        )
        return result
    result.dropped_snapshots = viable[0]
    snapshots = every_snapshot[viable[0] :]
    if len(snapshots) < 3:
        result.error = f"only {len(snapshots)} snapshot(s) after the baseline; nothing to compare"
        return result

    base = snapshots[0]
    base_chunks = _chunks_of(base, spec.chunker)
    try:
        queries = generate_offline(base_chunks, per_class=PER_CLASS, seed=QUERY_SEED)
    except QuerySetError as exc:
        # Recorded rather than swallowed. A corpus whose own generator refuses it is a finding
        # about the wizard's reach, and silently substituting a different chunker to get past it
        # would break the invariant the module docstring names.
        result.error = f"generate_offline refused this corpus: {exc}"
        return result

    query_texts = [entry["query"] for entry in queries]
    answerable_mask = [bool(entry["answerable"]) for entry in queries]
    query_vectors = embedder.embed(query_texts, progress=f"[{spec.id} queries]")

    # One similarity matrix, read twice. The maximum is the score the threshold is fitted to; the
    # argmax names the chunk that produced it, and `evidence_survival` below asks whether that exact
    # text is still in the corpus later. That is what separates label rot (the evidence was edited
    # away, so the query is arguably no longer answerable) from retrieval drift (the evidence is
    # still there and scores differently).
    base_similarities = query_vectors @ embedder.embed(
        base_chunks, progress=f"[{spec.id} {base.label}]"
    ).T
    base_scores = base_similarities.max(axis=1)
    base_answerable = [float(s) for s, ok in zip(base_scores, answerable_mask, strict=True) if ok]
    base_unanswerable = [
        float(s) for s, ok in zip(base_scores, answerable_mask, strict=True) if not ok
    ]
    fitted = from_samples(EMBEDDER, base_answerable, base_unanswerable)
    evidence_digests = [
        _sha256(base_chunks[int(index)].encode("utf-8"))
        for index, ok in zip(base_similarities.argmax(axis=1), answerable_mask, strict=True)
        if ok
    ]

    result.baseline = {
        "label": base.label,
        "sources": len(base.sources),
        "chunks": len(base_chunks),
        "threshold": fitted.threshold,
        "scale": fitted.scale,
        "separability": fitted.separability,
        "separability_ci": list(fitted.separability_ci or ()),
        "certified": fitted.certified,
        "certification_reason": fitted.certification_reason,
        "n_answerable": fitted.n_answerable,
        "n_unanswerable": fitted.n_unanswerable,
        "in_sample_errors": threshold_error_rates(
            base_answerable, base_unanswerable, fitted.threshold
        ),
        "queries": queries,
    }

    base_objects = base.manifest_objects()
    baseline_max_error = max(
        result.baseline["in_sample_errors"]["false_abstain_rate"],
        result.baseline["in_sample_errors"]["false_confirm_rate"],
    )
    for snapshot in snapshots:
        chunks = _chunks_of(snapshot, spec.chunker)
        if not chunks:
            continue
        scores = _top1(embedder, chunks, query_vectors, tag=f"[{spec.id} {snapshot.label}]")
        answerable = [float(s) for s, ok in zip(scores, answerable_mask, strict=True) if ok]
        unanswerable = [float(s) for s, ok in zip(scores, answerable_mask, strict=True) if not ok]
        errors = threshold_error_rates(answerable, unanswerable, fitted.threshold)
        auc = separability(answerable, unanswerable)
        # `is not None`, never a truth test. An AUC of exactly 0.0 is a real and dramatic
        # measurement (the classes are perfectly inverted) and a falsy check would file it under
        # "could not judge", which is the failure `memory/missing-input-becomes-a-clean-null.md`
        # names: an absent input and a constant become indistinguishable in the output.
        ci = (
            separability_interval(auc, len(answerable), len(unanswerable))
            if auc is not None
            else (0.0, 1.0)
        )
        refit = from_samples(EMBEDDER, answerable, unanswerable)
        delta = corpus_delta(base_objects, snapshot.manifest_objects())
        present = {_sha256(text.encode("utf-8")) for text in chunks}
        survived = sum(1 for digest in evidence_digests if digest in present)
        result.points.append(
            {
                "label": snapshot.label,
                "sources": len(snapshot.sources),
                "chunks": len(chunks),
                **delta,
                "false_abstain_rate": errors["false_abstain_rate"],
                "false_confirm_rate": errors["false_confirm_rate"],
                "max_error": max(errors["false_abstain_rate"], errors["false_confirm_rate"]),
                # The RISE over what this threshold already cost on the day it was fitted. See the
                # warning in `trigger_analysis` for why an absolute bar is not enough on its own.
                "excess_max_error": max(
                    errors["false_abstain_rate"], errors["false_confirm_rate"]
                ) - baseline_max_error,
                "baseline_max_error": baseline_max_error,
                "separability": auc,
                "separability_ci_low": ci[0],
                "certified_now": (auc is not None and ci[0] >= MIN_SEPARABILITY),
                "refit_threshold": refit.threshold,
                "threshold_drift": refit.threshold - fitted.threshold,
                "evidence_survival": (
                    survived / len(evidence_digests) if evidence_digests else float("nan")
                ),
            }
        )
        embedder.flush()
        point = result.points[-1]
        print(
            f"  {spec.id} {snapshot.label}: delta={point['corpus_delta']:.3f} "
            f"fa={point['false_abstain_rate']:.3f} fc={point['false_confirm_rate']:.3f} "
            f"refit={point['refit_threshold']:.3f}",
            file=sys.stderr,
            flush=True,
        )
    return result


# --------------------------------------------------------------------------------------
# Trigger analysis
# --------------------------------------------------------------------------------------


def trigger_analysis(
    points: Sequence[dict[str, Any]],
    *,
    outcome: str = "max_error",
    error_bound: float = DEFAULT_MAX_CARRY_FORWARD_ERROR,
    target_recall: float = 0.90,
) -> dict[str, Any]:
    """How well `corpus_delta` alone can act as the recalibration trigger.

    The registered label is `max_error > DEFAULT_MAX_CARRY_FORWARD_ERROR`, because that is the bar
    `carry_forward` already refuses on, so it is the question an operator is actually asking.

    ⚠️ **A second `outcome` is offered, and the reason is an apparatus fact rather than a result.**
    The smoke run on the memory corpus put the BASELINE's own in-sample false-abstain rate at 0.100,
    exactly on the bound, before any drift at all. That is not surprising given the classes are
    known to overlap in 4 of 4 measured corpora, but it means an absolute bar can be met by a
    calibration on the day it is fitted, and a trigger scored against it would be labelling the fit
    rather than the drift. `excess_max_error`, the rise over the baseline's own in-sample error, is
    therefore recorded and analysed alongside. Both are reported; neither replaces the other, and
    the registered one is reported first.

    Precision and recall are both over their own denominators and both named, because a trigger
    tuned to catch everything by firing always has perfect recall and is worthless.
    """
    scored = [
        (float(point["corpus_delta"]), bool(point[outcome] > error_bound))
        for point in points
        # The baseline compared against itself is delta 0 and error in-sample; including it would
        # put a free true negative in every corpus and inflate precision.
        if point["corpus_delta"] > 0
    ]
    positives = sum(1 for _delta, over in scored if over)
    if not scored:
        return {"n": 0, "positives": 0}
    candidates = sorted({delta for delta, _over in scored})
    best: dict[str, Any] | None = None
    for cut in candidates:
        fired = [(delta, over) for delta, over in scored if delta >= cut]
        if not fired:
            continue
        true_positives = sum(1 for _delta, over in fired if over)
        recall = true_positives / positives if positives else float("nan")
        precision = true_positives / len(fired)
        if positives and recall >= target_recall:
            # Highest cut that still clears the recall target: the most precise trigger that does
            # not give up on the cases it exists to catch.
            if best is None or cut > best["cut"]:
                best = {
                    "cut": cut,
                    "recall": recall,
                    "precision": precision,
                    "fired": len(fired),
                    "true_positives": true_positives,
                }
    return {
        "n": len(scored),
        "positives": positives,
        "positive_rate": positives / len(scored),
        "outcome": outcome,
        "error_bound": error_bound,
        "spearman_delta_vs_outcome": _spearman(
            [delta for delta, _over in scored],
            [float(point[outcome]) for point in points if point["corpus_delta"] > 0],
        ),
        "under_default_max_delta_but_over_error": sum(
            1
            for delta, over in scored
            if delta < DEFAULT_MAX_CORPUS_DELTA and over
        ),
        "under_default_max_delta": sum(
            1 for delta, _over in scored if delta < DEFAULT_MAX_CORPUS_DELTA
        ),
        f"best_cut_at_recall_{target_recall}": best,
    }


# --------------------------------------------------------------------------------------
# Apparatus check
# --------------------------------------------------------------------------------------


def apparatus_check() -> dict[str, Any]:
    """Two cases whose answer is known before the harness runs. Exit code 0 is not a measurement.

    Both are about the predictor, not the model, because the predictor is where a silent error
    would be invisible: a delta computed over the wrong denominator still returns a plausible
    fraction for every input, and would produce a complete, wrong result set.
    """
    one = Snapshot("a", {"x.md": b"alpha", "y.md": b"beta"})
    identical = corpus_delta(one.manifest_objects(), one.manifest_objects())
    gutted = Snapshot("b", {"x.md": b"alpha"})
    emptied = corpus_delta(one.manifest_objects(), gutted.manifest_objects())
    edited = Snapshot("c", {"x.md": b"ALPHA", "y.md": b"beta"})
    in_place = corpus_delta(one.manifest_objects(), edited.manifest_objects())
    checks = {
        "identical_snapshot_is_zero": identical["corpus_delta"] == 0.0,
        "removal_counts_against_the_union": emptied["corpus_delta"] == 0.5,
        "in_place_edit_counts_as_modified": (
            in_place["sources_modified"] == 1 and in_place["corpus_delta"] == 0.5
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def build_specs(repo: Path, memory_root: Path | None) -> list[CorpusSpec]:
    specs = [
        CorpusSpec(
            id="docs",
            describe=f"{repo.name}/docs/**/*.md at commits that touched docs",
            chunker=chunk_text,
            snapshots=GitHistory(repo, "docs", ".md").snapshots,
            change_mode="add and edit",
        ),
        CorpusSpec(
            id="code",
            describe=f"{repo.name}/recall/**/*.py at commits that touched recall/",
            chunker=chunk_code,
            snapshots=GitHistory(repo, "recall", ".py").snapshots,
            change_mode="edit heavy",
        ),
    ]
    if memory_root is not None and memory_root.is_dir():
        specs.append(
            CorpusSpec(
                id="memory",
                describe=f"{memory_root} revealed in mtime order (a reconstruction)",
                chunker=chunk_text,
                snapshots=MtimeGrowth(memory_root, ".md").snapshots,
                change_mode="append only",
            )
        )
    return specs


def _sizing(specs: Sequence[CorpusSpec], count: int) -> None:
    total = 0
    for spec in specs:
        seen: set[str] = set()
        per_snapshot = []
        for snapshot in spec.snapshots(count):
            chunks = _chunks_of(snapshot, spec.chunker)
            fresh = {_sha256(text.encode("utf-8")) for text in chunks} - seen
            seen |= fresh
            per_snapshot.append((snapshot.label, len(snapshot.sources), len(chunks), len(fresh)))
        total += len(seen)
        print(f"\n{spec.id}: {len(per_snapshot)} snapshots, {len(seen)} distinct chunks to embed")
        for label, sources, chunks, fresh in per_snapshot:
            print(f"   {label:>14}  sources={sources:>4}  chunks={chunks:>5}  new={fresh:>5}")
    print(f"\nTOTAL distinct chunks to embed: {total}")


def analyze(payload: Mapping[str, Any]) -> None:
    """Print the figures the registered predictions are scored against, and nothing else.

    Separate from the measurement so a result can be re-read without re-running it, and so the
    numbers that go into the Result section come out of committed code rather than out of a
    snippet typed once and lost. Every rate is printed with its denominator, because a rate
    without one is not a result.
    """
    print(f"measured_at      {payload.get('measured_at')}")
    print(f"embedder         {payload.get('embedder')}")
    print(f"apparatus        {payload.get('apparatus_check', {}).get('passed')}")
    print(f"wall clock       {payload.get('wall_clock_seconds')}s")
    bound = float(payload.get("error_bound", DEFAULT_MAX_CARRY_FORWARD_ERROR))

    for corpus in payload.get("corpora", ()):
        points = [point for point in corpus.get("points", ()) if point["corpus_delta"] > 0]
        baseline = corpus.get("baseline") or {}
        print()
        print(f"=== {corpus['id']} ({corpus.get('change_mode')}) ===")
        if corpus.get("error"):
            print(f"  ERROR: {corpus['error']}")
            continue
        in_sample = baseline.get("in_sample_errors", {})
        print(
            f"  baseline {baseline.get('label')}: {baseline.get('sources')} sources, "
            f"{baseline.get('chunks')} chunks, threshold {baseline.get('threshold')}, "
            f"AUC {baseline.get('separability'):.4f}, in-sample "
            f"fa {in_sample.get('false_abstain_rate', 0):.3f} / "
            f"fc {in_sample.get('false_confirm_rate', 0):.3f}"
        )
        print(f"  dropped {corpus.get('dropped_snapshots', 0)} snapshot(s) before it")
        print(f"  {len(points)} non-baseline snapshot(s), delta range "
              f"{min((p['corpus_delta'] for p in points), default=float('nan')):.3f} to "
              f"{max((p['corpus_delta'] for p in points), default=float('nan')):.3f}")
        over = [point for point in points if point["max_error"] > bound]
        print(f"  over the {bound} bound: {len(over)} of {len(points)}")
        if over:
            first = min(over, key=lambda point: point["corpus_delta"])
            print(f"  smallest delta that is over the bound: {first['corpus_delta']:.3f} "
                  f"(fa {first['false_abstain_rate']:.3f}, fc {first['false_confirm_rate']:.3f})")
        else:
            print("  no snapshot crossed the bound")
        # P2 is a claim about WHICH error dominates, so it is scored by the counts and not by an
        # average that could hide one large excursion inside many quiet snapshots.
        abstain_led = sum(
            1 for point in points
            if point["false_abstain_rate"] > point["false_confirm_rate"]
        )
        confirm_led = sum(
            1 for point in points
            if point["false_confirm_rate"] > point["false_abstain_rate"]
        )
        print(f"  false abstain leads in {abstain_led}, false confirm leads in {confirm_led}, "
              f"tied in {len(points) - abstain_led - confirm_led}")
        print(f"  evidence survival at the last snapshot: "
              f"{points[-1]['evidence_survival']:.3f}" if points else "  no points")
        for label in ("trigger", "trigger_excess"):
            block = corpus.get(label) or {}
            if block:
                print(f"  {label}: spearman "
                      f"{block.get('spearman_delta_vs_outcome', float('nan')):.3f}, "
                      f"{block.get('positives')} positive of {block.get('n')}, "
                      f"best cut {block.get('best_cut_at_recall_0.9')}")

    for label in ("pooled_trigger", "pooled_trigger_excess"):
        block = payload.get(label) or {}
        if not block:
            continue
        print()
        print(f"=== {label} ===")
        for key, value in block.items():
            print(f"  {key}: {value}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="repository whose history supplies snapshots")
    parser.add_argument(
        "--memory-root",
        default=os.environ.get("RECALL_MEMORY_ROOT"),
        help="the memory store to use as the append-only corpus",
    )
    parser.add_argument("--snapshots", type=int, default=DEFAULT_SNAPSHOTS)
    parser.add_argument(
        "--cache",
        default=os.environ.get("RECALL_DRIFT_CACHE", ".drift-embeddings.npz"),
        help="chunk embedding cache; safe to delete, expensive to lose",
    )
    parser.add_argument("--out", default=None, help="write the result JSON here")
    parser.add_argument("--sizing", action="store_true", help="chunk counts only, no embedding")
    parser.add_argument("--only", default=None, help="comma-separated corpus ids to run")
    parser.add_argument(
        "--analyze",
        default=None,
        help="read a result JSON and print the figures the registered predictions are scored "
        "against, instead of measuring anything",
    )
    args = parser.parse_args(argv)

    if args.analyze:
        analyze(json.loads(Path(args.analyze).read_text(encoding="utf-8")))
        return 0

    repo = Path(args.repo).resolve()
    memory_root = Path(args.memory_root).expanduser() if args.memory_root else None
    specs = build_specs(repo, memory_root)
    if args.only:
        wanted = {item.strip() for item in args.only.split(",")}
        specs = [spec for spec in specs if spec.id in wanted]

    if args.sizing:
        _sizing(specs, args.snapshots)
        return 0

    apparatus = apparatus_check()
    print(f"apparatus check: {apparatus}", file=sys.stderr)
    if not apparatus["passed"]:
        print("apparatus check FAILED; refusing to measure", file=sys.stderr)
        return 2

    embedder = ChunkEmbedder(Path(args.cache))
    results = []
    started = time.time()
    for spec in specs:
        print(f"\n=== {spec.id} ===", file=sys.stderr, flush=True)
        results.append(measure_corpus(spec, embedder, args.snapshots))
        embedder.flush()

    every_point = [point for result in results for point in result.points]
    payload = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedder": EMBEDDER,
        "per_class": PER_CLASS,
        "query_seed": QUERY_SEED,
        "snapshots_requested": args.snapshots,
        "search": "exact top-1 cosine, not HNSW",
        "error_bound": DEFAULT_MAX_CARRY_FORWARD_ERROR,
        "default_max_corpus_delta": DEFAULT_MAX_CORPUS_DELTA,
        "apparatus_check": apparatus,
        "wall_clock_seconds": round(time.time() - started, 1),
        "corpora": [
            {
                "id": result.corpus_id,
                "describe": result.describe,
                "change_mode": result.change_mode,
                "error": result.error,
                "dropped_snapshots": result.dropped_snapshots,
                "baseline": result.baseline,
                "points": result.points,
                "trigger": trigger_analysis(result.points) if result.points else None,
                "trigger_excess": (
                    trigger_analysis(result.points, outcome="excess_max_error", error_bound=0.0)
                    if result.points
                    else None
                ),
            }
            for result in results
        ],
        "pooled_trigger": trigger_analysis(every_point) if every_point else None,
        "pooled_trigger_excess": (
            trigger_analysis(every_point, outcome="excess_max_error", error_bound=0.0)
            if every_point
            else None
        ),
    }
    text = json.dumps(payload, indent=2, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
