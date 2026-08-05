"""Do two registered embedding profiles actually produce different vectors?

Prior work: [[project-recall-embedding-profile-registry-2026-08-05]]. SEARCHED before writing
this, `docs_search(source_type="memory", "bge-small asymmetric symmetric embedding profile
query_embed passage_embed identical vectors RE-call")`, no gap_warning, top-3 cosine
0.809/0.689/0.672. Session 8 already measured this on VPS2 and recorded "the asymmetric profiles
are asymmetric in name only". This is not a re-measurement for its own sake: that finding is now
load-bearing for a whole campaign's design, and a fact that decides whether an experiment is worth
running is one to confirm rather than inherit. The two runs use different probe texts, and this
one is committed, so the next session re-checks it with a command instead of a memory.

`bge-small-symmetric-v1` and `bge-small-asymmetric-v1` are declared in
`recall/embedding_registry.py` with identical `model_name`, `dimension`, `context_mode`,
`normalization`, `instruction_version` and `chunker_version`. They differ in exactly two fields:
`query_mode` and `passage_mode`. So the WHOLE retrieval difference between an arm on one and an
arm on the other is whatever the backend's `query_embed` / `passage_embed` do differently from
`embed`. If that is nothing, a paired comparison of the two is a system compared with itself:
`hit@5` equal on every question by construction, paired bootstrap interval exactly `[0, 0]`, and
the promotion gate refusing for a reason that describes neither profile. This repository already
holds that artifact (`results/promotion/decision.null-difference.json`), produced deliberately as
the harness's null control, so running the campaign would publish the control as an experiment.

PRE-REGISTERED before running (predict-then-measure; recorded here so a convenient result cannot
be rationalised afterwards):

  P1  `embed`, `query_embed` and `passage_embed` return BYTE-IDENTICAL vectors for
      `BAAI/bge-small-en-v1.5` on every probe, query-shaped and passage-shaped alike.
  P2  All three names resolve to ONE underlying function, so P1 needs no cache hypothesis to
      explain it.
  P3  Positive control: two DIFFERENT texts through one encoder are not identical.
  P4  Sensitivity control: a deliberate rotation of 1e-4 rad is reported as DIFFERENT by the same
      comparison the probes use.

P3 and P4 are the controls and they are the point. "The encoders agree" is worthless on its own,
because a broken comparator reports exactly the same thing. P3 establishes that the comparator
separates grossly different vectors; **P4 establishes that it separates a SMALL difference along
the axis the null is actually drawn on**, which P3 cannot, because P3 varies the TEXT while the
probes vary the METHOD. The first version of this script had P3 and not P4: a positive control
validates the MECHANISM, not the SCOPE.

This script deliberately does NOT import the `recall` package. The subject is what the BACKEND
does, and asking the code under test whether its own two configurations differ is the
self-comparison this program keeps recording. The artifact tree digest is therefore reimplemented
rather than imported, which also makes it an independent recomputation of the manifest's value.

What this measurement cannot do by byte comparison alone is separate "one function under three
names" from "three genuinely different functions behind a cache keyed on the text alone", because
every probe of the second case is a cache hit by construction. So it also records the RESOLVED
METHOD IDENTITIES: if all three names resolve to one function, the cache hypothesis is not needed.
If they are distinct functions that nevertheless agree byte for byte, that is a different and more
interesting finding, and the verdict says so rather than eliding it.

Run it against a COPY of the artifact tree, never the provisioned one, and not as root.
fastembed 0.8.0's artifact-miss path deletes files before it checks `local_files_only`, so a
mis-pointed `--cache-dir` destroys operator data. `assert_provisioned` refuses that state before
the loader is constructed, and `--expect-artifact-sha256` is what makes a copy as good as the
original: a copy whose digest matches the manifest IS the tree, and one whose digest does not is
refused.

Run:  cp -a <provisioned tree> /var/tmp/tree && /opt/recall-enterprise/venv/bin/python -m
benchmarks.check_profile_encoder_distinctness --cache-dir /var/tmp/tree
--expect-artifact-sha256 <digest>

Exit codes, all distinct, because "no measurement", "a measurement that said no" and "you typed
the flag wrong" must never be confusable:  0 a verdict was produced (whichever verdict);
2 a control refused, so there is no measurement; 64 usage error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

#: The provisioned tree on the host this program's artifacts live on. A default, not a claim: the
#: digest of whatever tree is used is recomputed and reported, and `--expect-artifact-sha256`
#: turns that report into a binding.
DEFAULT_CACHE_DIR = "/opt/recall-enterprise/models/bge-fastembed-cache"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
#: `recall/embedding_registry.py` declares 384 for every bge-small profile. A width that disagrees
#: means the tree is not the model the result would be attributed to.
DEFAULT_DIMENSION = 384
#: The two profile ids this script exists to compare. Arguments rather than literals in the
#: verdict, so running with `--model-name` pointed elsewhere cannot publish a byte-identity result
#: about one model under a sentence naming another.
DEFAULT_SYMMETRIC_PROFILE = "bge-small-symmetric-v1"
DEFAULT_ASYMMETRIC_PROFILE = "bge-small-asymmetric-v1"

#: Above this cosine, two vectors are "the same" for the positive control's purposes. Not 1.0:
#: the control must fire on a comparator that returns 0.9999 for two different texts, not only on
#: one that returns exactly 1.0.
COSINE_IDENTICAL_FLOOR = 0.999
#: The rotation P4 injects. Far above the float64 comparison floor (~1e-8, see `_cosine`) and far
#: below any difference a real encoder change would produce, so a harness that cannot see it
#: cannot license a null.
SENSITIVITY_ANGLE_RAD = 1e-4
#: Reporting widths.
DIGEST_CHARS = 16
COSINE_DECIMALS = 12
TEXT_PREFIX_CHARS = 44

#: Query-shaped and passage-shaped probes. Both shapes on purpose: an encoder applying an
#: instruction prefix to queries only would differ on the first group and not the second, and
#: probing one shape could not tell those apart.
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

#: The symmetric encoder every comparison is made against, and the two the asymmetric profile
#: declares. One tuple, so adding an encoder is one edit rather than three.
BASE_ENCODER = "embed"
ASYMMETRIC_ENCODERS: tuple[str, ...] = ("query_embed", "passage_embed")
ENCODER_NAMES: tuple[str, ...] = (BASE_ENCODER, *ASYMMETRIC_ENCODERS)

#: Exit codes. See the module docstring.
EXIT_VERDICT = 0
EXIT_REFUSED = 2
EXIT_USAGE = 64

#: `Any` for vectors: numpy is an optional extra here, so the module must typecheck without its
#: stubs — the same reason `benchmarks/beam/dedup_probe.py` and `recall/eval/vocab.py:crowding`
#: annotate their arrays this way. `npt.NDArray` would need stubs the mypy override skips.
Encoder = Callable[[Sequence[str]], Iterable[Any]]


class ControlRefused(Exception):
    """A control did not hold, so there is no measurement to report.

    An exception rather than a bare `SystemExit` so the controls are testable in-process: the
    first version fused refusal to `print` and process exit, which meant the only proof that a
    control could fire was a file hand-edited on the host and thrown away.
    """

    def __init__(self, reason: str, detail: Mapping[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = dict(detail)

    def payload(self) -> dict[str, Any]:
        return {"refused": self.reason, **self.detail}


def script_sha256() -> str:
    """This file's own digest, over LF-normalised bytes.

    Provenance that travels IN the artifact rather than in a paragraph beside it, so a reader of
    an archived result can identify the code that produced it with one command:
    `git show <rev>:benchmarks/check_profile_encoder_distinctness.py | sha256sum`.

    Normalised because git stores this file with LF while a Windows working copy holds CRLF.
    Comparing a working copy against a git blob across that boundary is precisely the mistake that
    made one claim in `docs/ENTERPRISE_PROGRAM_STATUS.md` wrong three times, so this value is
    defined to be the one that matches on either platform.
    """
    crlf, lf = bytes([13, 10]), bytes([10])
    return hashlib.sha256(Path(__file__).read_bytes().replace(crlf, lf)).hexdigest()


def _package_version(name: str) -> str:
    """Never let recording the environment destroy a completed measurement.

    `__import__(name).__version__` used to be evaluated inside the returned dict, after every
    probe and both controls, so a package publishing no `__version__` lost the whole run with
    empty stdout. `recall/eval/provenance.py` tracks these same two packages this way.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def artifact_tree_sha256(path: str) -> str:
    """Sorted relative POSIX path, NUL, then file bytes.

    Reimplemented rather than imported from `recall.embeddings`, for the same reason the rest of
    this module does not import `recall`: an independent recomputation of the manifest's value is
    evidence, and code checking itself is not.
    """
    root = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ControlRefused("artifact tree has no files", {"cache_dir": str(root)})
    for file in files:
        resolved = file.resolve(strict=True)
        if root.is_dir() and not resolved.is_relative_to(root):
            raise ControlRefused("artifact symlink escapes its root", {"path": str(file)})
        relative = resolved.name if root.is_file() else resolved.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def assert_provisioned(cache_dir: str) -> dict[str, Any]:
    """Refuse a tree with no model snapshot, before a loader whose miss path DELETES sees it.

    Scope, stated because the first docstring claimed more: this checks that the tree holds SOME
    `models--*` snapshot and SOME `.onnx` graph. It does NOT check that they are the snapshot for
    `model_name`, and fastembed's destructive path is keyed on the model name, so a tree correctly
    provisioned for model A opened with `--model-name B` still passes here. The blast radius of
    that case is near zero (B's paths do not exist to delete), but the guard is narrower than
    "refuses every state that reaches the delete path".

    ⚠️ This guard is not tidiness. In fastembed 0.8.0 the artifact-miss path runs `shutil.rmtree`
    on `<cache_dir>/tmp/<model>` and `unlink` on `<cache_dir>/<model>.tar.gz` **before** it checks
    `local_files_only` (`fastembed/common/model_management.py:336-348`, guard at `:349`). The CCA
    security auditor demonstrated it against a seeded tree with `HF_HUB_OFFLINE=1` and
    `local_files_only=True`: the files were deleted and only then did the constructor raise. A
    correctly provisioned tree returns early and is untouched, so the hazard fires in exactly the
    situation this script would be run to diagnose, and the default `--cache-dir` is a production
    tree the docs said to open as root.

    So the snapshot is checked HERE, by reading, before `TextEmbedding` is ever constructed.
    """
    root = Path(cache_dir)
    if not root.is_dir():
        raise ControlRefused(
            "cache dir does not exist; refusing to hand it to a loader whose miss path deletes",
            {"cache_dir": cache_dir},
        )
    snapshots = sorted(p.name for p in root.glob("models--*") if p.is_dir())
    graphs = sorted(str(p.relative_to(root).as_posix()) for p in root.rglob("*.onnx"))
    detail = {"cache_dir": cache_dir, "snapshot_dirs": snapshots, "onnx_graphs": graphs}
    if not snapshots or not graphs:
        raise ControlRefused(
            "artifact tree is missing a model snapshot or its ONNX graph, which is the state that "
            "sends fastembed down a fallback path that DELETES files before checking "
            "local_files_only; refusing rather than opening it",
            detail,
        )
    return detail


def _vector(encode: Encoder, text: str) -> Any:
    import numpy as np

    produced = list(encode([text]))
    if len(produced) != 1:
        raise ControlRefused("expected exactly one vector for one text", {"returned": len(produced)})
    return np.asarray(produced[0], dtype=np.float32)


def _sha(vector: Any) -> str:
    return hashlib.sha256(vector.tobytes()).hexdigest()


def _cosine(left: Any, right: Any) -> float:
    """Cosine similarity, computed in float64 and clamped to [-1, 1].

    The first version computed entirely in float32: `np.dot` and `np.linalg.norm` both return
    `np.float32`, so the result was `S / fl(fl(sqrt(S))**2)` and the numerator and denominator
    disagreed by up to 1 ULP. The CCA numeric auditor measured the two consequences over 20000
    trials on 384-dim unit vectors: it returned values ABOVE 1.0 for byte-identical vectors in
    31.9% of them (max 1.000000119209), and it returned exactly 1.0 for vectors genuinely
    differing in 382 of 384 components (a rotation of 3e-5 rad). Upcasting is exact — every
    float32 is a float64 — so it cannot manufacture agreement, and it lowers the resolution floor
    from ~3e-5 rad to ~1e-8.
    """
    import numpy as np

    wide_left = np.asarray(left, dtype=np.float64)
    wide_right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(wide_left) * np.linalg.norm(wide_right))
    if denominator == 0.0:
        # Otherwise 0/0 is a NaN that serialises as a bare `NaN` token, which is not valid JSON,
        # and `nan > FLOOR` is False, so a degenerate vector would slip past the positive control.
        raise ControlRefused("a probe produced a zero-norm vector; cosine is undefined", {})
    return float(np.clip(float(np.dot(wide_left, wide_right)) / denominator, -1.0, 1.0))


def compare(left: Any, right: Any) -> dict[str, Any]:
    """The one comparison every probe and every control goes through.

    One function, so the controls demonstrate the resolution of the comparison the probes actually
    use rather than of a parallel one written beside it.
    """
    return {
        "bytes_equal": left.tobytes() == right.tobytes(),
        "cosine": round(_cosine(left, right), COSINE_DECIMALS),
        "sha256_left": _sha(left)[:DIGEST_CHARS],
        "sha256_right": _sha(right)[:DIGEST_CHARS],
    }


def calls_base_encoder(source: str) -> bool:
    """Does this source actually CALL `self.embed(...)`, rather than mention one?

    An AST walk, not a substring test. The first version asked `"self.embed(" in source` over the
    concatenated `inspect.getsource` of the whole delegation chain, and the anti-regression gate
    reproduced the consequence: a docstring reading "deliberately does NOT call self.embed(text)"
    and a comment reading "used to be `yield from self.embed(texts)`" both made it True, so a
    NEGATION read as a confirmation and the verdict escalated from "not settled" to "cannot produce
    different vectors". That mattered because on the real backend the three names are three
    distinct functions, so this check is the ONLY support for that escalation.

    Patching the pattern was available and is the wrong move: this repository has a recorded case
    of one domain check rewritten five times, producing a new defect each time, and the lesson is
    that three rounds on the same parser means the MECHANISM is wrong. Comments never enter an AST,
    and a docstring is a constant rather than a call, so both classes of false positive are
    excluded by construction rather than by a better regex.
    """
    import ast
    import textwrap

    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr == BASE_ENCODER
            and isinstance(function.value, ast.Name)
            and function.value.id == "self"
        ):
            return True
    return False


def resolved_encoders(model: Any) -> dict[str, Any]:
    """Which underlying function each declared encoder name resolves to.

    The only thing that separates "one function under three names" from "three functions behind a
    text-keyed cache". `callable`, not `hasattr`: a non-callable attribute of the right name would
    be published as present and then fail later with a bare `TypeError`, where
    `recall/embeddings.py` resolves the mode and refuses when it is not callable.
    """
    import inspect

    resolved: dict[str, Any] = {}
    groups: dict[int, str] = {}
    for name in ENCODER_NAMES:
        bound = getattr(model, name, None)
        underlying = getattr(bound, "__func__", bound)
        # Walk the delegation chain, not just the bound method. fastembed's public
        # `TextEmbedding` is a thin wrapper that forwards to `self.model`, so inspecting only the
        # outer layer finds `yield from self.model.query_embed(...)` and concludes nothing. The
        # implementation that matters is one level down.
        sources: list[str] = []
        chain: list[Any] = []
        target: Any = model
        for _ in range(4):  # bounded: a wrapper chain, not a general graph walk
            member = getattr(type(target), name, None) or getattr(target, name, None)
            if member is not None:
                chain.append(member)
            inner = getattr(target, "model", None)
            if inner is None or inner is target:
                break
            target = inner
        for member in chain:
            candidate = getattr(member, "__func__", member)
            if not callable(candidate):
                continue
            try:
                sources.append(inspect.getsource(candidate))
            except (OSError, TypeError):  # builtins, C extensions, anything without a file
                continue
        source = "\n".join(sources)
        resolved[name] = {
            "callable": callable(bound),
            "qualname": getattr(underlying, "__qualname__", None),
            "module": getattr(underlying, "__module__", None),
            # A stable label per distinct underlying function, so the artifact records how many
            # functions there are without publishing a memory address.
            "function_group": groups.setdefault(id(underlying), f"f{len(groups)}"),
            # Source-level delegation is the second way to settle the cache hypothesis, and the
            # only one that does not need the weights. In fastembed 0.8.0 `BAAI/bge-small-en-v1.5`
            # resolves to `OnnxTextEmbedding`, which does NOT override these two: they come from
            # `TextEmbeddingBase` and both `yield from self.embed(...)` with no instruction. So
            # byte-identity here is a property of the LIBRARY, not of one provisioned host.
            "source_delegates_to_base": name != BASE_ENCODER and calls_base_encoder(source),
            "source_available": bool(source),
            "delegation_chain": [getattr(m, "__qualname__", None) for m in chain],
        }
    resolved["distinct_function_count"] = len(groups)
    resolved["all_three_are_one_function"] = len(groups) == 1
    resolved["asymmetric_encoders_delegate_to_base"] = all(
        resolved[name]["source_delegates_to_base"] for name in ASYMMETRIC_ENCODERS
    )
    # Either route closes the question: one function cannot differ from itself, and a function
    # whose body defers to the base encoder cannot either.
    resolved["identity_is_explained"] = bool(
        resolved["all_three_are_one_function"] or resolved["asymmetric_encoders_delegate_to_base"]
    )
    return resolved


def probe_rows(encoders: Mapping[str, Encoder], *, expect_dimension: int) -> list[dict[str, Any]]:
    """One row per probe: the base encoder against each asymmetric encoder, same text."""
    rows: list[dict[str, Any]] = []
    for kind, texts in (("query", QUERIES), ("passage", PASSAGES)):
        for text in texts:
            base = _vector(encoders[BASE_ENCODER], text)
            if int(base.shape[0]) != expect_dimension:
                raise ControlRefused(
                    "observed vector width contradicts the registry's declared dimension; this "
                    "tree is not the model the result would be attributed to",
                    {"observed": int(base.shape[0]), "expected": expect_dimension},
                )
            row: dict[str, Any] = {
                "probe_kind": kind,
                "text_prefix": text[:TEXT_PREFIX_CHARS],
                "dimension": int(base.shape[0]),
                f"sha256_{BASE_ENCODER}": _sha(base)[:DIGEST_CHARS],
            }
            for name in ASYMMETRIC_ENCODERS:
                other = _vector(encoders[name], text)
                comparison = compare(base, other)
                row[f"sha256_{name}"] = comparison["sha256_right"]
                row[f"bytes_{BASE_ENCODER}_eq_{name}"] = comparison["bytes_equal"]
                row[f"cosine_{BASE_ENCODER}_{name}"] = comparison["cosine"]
            rows.append(row)
    return rows


def assert_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """`all([])` is True. A truncated probe set would narrow the evidence silently."""
    expected = len(QUERIES) + len(PASSAGES)
    kinds = sorted({str(row["probe_kind"]) for row in rows})
    detail = {"probe_count": len(rows), "expected": expected, "kinds": kinds}
    if len(rows) != expected or kinds != ["passage", "query"]:
        raise ControlRefused("probe coverage is incomplete; the reduction would be partial", detail)
    return detail


def positive_control(embed: Encoder) -> dict[str, Any]:
    """P3. Two DIFFERENT texts through ONE encoder must not be identical."""
    result = compare(_vector(embed, QUERIES[0]), _vector(embed, PASSAGES[0]))
    if bool(result["bytes_equal"]) or float(result["cosine"]) > COSINE_IDENTICAL_FLOOR:
        raise ControlRefused(
            "positive control: the comparator cannot distinguish two different texts", result
        )
    return result


def determinism_control(embed: Encoder) -> dict[str, Any]:
    """The same call twice must agree, or "identical" is luck and "different" is noise."""
    identical = _vector(embed, QUERIES[0]).tobytes() == _vector(embed, QUERIES[0]).tobytes()
    result = {"embed_twice_byte_identical": identical}
    if not identical:
        raise ControlRefused("determinism control: the backend is not deterministic", result)
    return result


def rotate(vector: Any, angle: float) -> Any:
    """Rotate `vector` by `angle` radians, deterministically, in its own norm."""
    import numpy as np

    wide = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(wide))
    if norm == 0.0:
        raise ControlRefused("cannot rotate a zero vector", {})
    unit = wide / norm
    # A fixed basis direction, projected off `unit`, gives a reproducible orthogonal axis. The
    # second basis vector covers the case where the first is (anti)parallel to `unit`.
    for index in (0, 1):
        candidate = np.zeros_like(wide)
        candidate[index] = 1.0
        candidate = candidate - float(np.dot(candidate, unit)) * unit
        magnitude = float(np.linalg.norm(candidate))
        if magnitude > 1e-6:
            orthogonal = candidate / magnitude
            break
    else:  # pragma: no cover - unreachable for dimension >= 2
        raise ControlRefused("could not build an orthogonal direction", {})
    return (np.cos(angle) * unit + np.sin(angle) * orthogonal) * norm


def sensitivity_control(embed: Encoder, *, angle: float = SENSITIVITY_ANGLE_RAD) -> dict[str, Any]:
    """P4. A deliberate small rotation must be reported as DIFFERENT by `compare`.

    This is the control the first version lacked. P3 varies the TEXT and holds the ENCODER fixed,
    at a cosine of about 0.65, so it establishes only that grossly different vectors separate. The
    probes vary the ENCODER and hold the text fixed, and the null verdict is drawn on that axis.
    A harness that cannot see a 1e-4 rad difference between two encoders has not earned the right
    to report that two encoders agree.
    """
    import numpy as np

    base = _vector(embed, QUERIES[0])
    perturbed = np.asarray(rotate(base, angle), dtype=np.float32)
    result = compare(base, perturbed)
    result["angle_rad"] = angle
    if bool(result["bytes_equal"]) or float(result["cosine"]) >= 1.0:
        raise ControlRefused(
            "sensitivity control: the comparator cannot see a deliberate small rotation, so it "
            "cannot license a null on the encoder axis",
            result,
        )
    return result


def build_verdict(
    *,
    identical: bool,
    resolved: Mapping[str, Any],
    symmetric_profile: str,
    asymmetric_profile: str,
) -> str:
    if not identical:
        return (
            f"at least one probe differs: {symmetric_profile} and {asymmetric_profile} are "
            f"genuinely distinct here"
        )
    if bool(resolved["identity_is_explained"]):
        why = (
            "all three names resolve to ONE underlying function"
            if resolved["all_three_are_one_function"]
            else f"both asymmetric encoders DELEGATE to {BASE_ENCODER} in source"
        )
        return (
            f"{', '.join(ASYMMETRIC_ENCODERS)} are BYTE-IDENTICAL to {BASE_ENCODER} on every "
            f"probe, and {why}: {asymmetric_profile} and {symmetric_profile} cannot produce "
            f"different vectors on this backend, so a paired comparison of the two is a null by "
            f"construction"
        )
    return (
        f"byte-identical on every probe, but the three names resolve to "
        f"{resolved['distinct_function_count']} distinct functions and no delegation was found in "
        f"source: {asymmetric_profile} and {symmetric_profile} agree here, and WHY they agree is "
        f"not settled by this measurement (see the module docstring on the text-keyed cache "
        f"hypothesis)"
    )


def measure(
    cache_dir: str,
    model_name: str,
    *,
    expect_dimension: int = DEFAULT_DIMENSION,
    expect_artifact_sha256: str | None = None,
    symmetric_profile: str = DEFAULT_SYMMETRIC_PROFILE,
    asymmetric_profile: str = DEFAULT_ASYMMETRIC_PROFILE,
    model_factory: Callable[[str, str], Any] | None = None,
) -> dict[str, Any]:
    """Load the backend, run every probe and every control, assemble the report.

    `model_factory` exists so a test can substitute an encoder without a real artifact tree. It
    defaults to the real loader; nothing in the shipped path goes through anything else.
    """
    # Inside `measure` and before the backend import, so the offline guard travels with the
    # measurement rather than with the CLI wrapper. Assignment, not `setdefault`: a caller who had
    # already exported HF_HUB_OFFLINE=0 would otherwise have that value RECORDED as the run's
    # provenance. The CCA security auditor measured which guard actually does the work here, with
    # a socket tripwire proven able to fire: `local_files_only=True` gave 0 network attempts with
    # both env vars removed, and omitting it gave 13. So these two are defence in depth and
    # `local_files_only` is the control; the report below states all three rather than publishing
    # the state of the guard that is not doing the work.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Both checks run BEFORE the loader is constructed: the provisioning guard because the miss
    # path deletes, and the identity binding so a run against the wrong tree cannot produce a
    # verdict at all.
    provisioning = assert_provisioned(cache_dir)
    observed_digest = artifact_tree_sha256(cache_dir)
    if expect_artifact_sha256 and observed_digest != expect_artifact_sha256:
        raise ControlRefused(
            "artifact tree digest does not match the expected one",
            {
                "cache_dir": cache_dir,
                "observed": observed_digest,
                "expected": expect_artifact_sha256,
            },
        )

    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "fastembed": _package_version("fastembed"),
        "onnxruntime": _package_version("onnxruntime"),
        "numpy": _package_version("numpy"),
        "cache_dir": cache_dir,
        "artifact_tree_sha256": observed_digest,
        "artifact_tree_sha256_expected": expect_artifact_sha256,
        "model_name": model_name,
        "symmetric_profile": symmetric_profile,
        "asymmetric_profile": asymmetric_profile,
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        "provisioning": provisioning,
    }

    if model_factory is None:

        def _load(name: str, directory: str) -> Any:
            from fastembed import TextEmbedding

            return TextEmbedding(model_name=name, cache_dir=directory, local_files_only=True)

        model_factory = _load
    model = model_factory(model_name, cache_dir)

    # `local_files_only` is absorbed by `**kwargs`, so a fastembed that stopped honouring it would
    # leave the guard vacuous and silent. Assert it was actually consumed; report what was found
    # either way rather than claiming a guard that may not be there.
    environment["local_files_only_consumed"] = bool(
        getattr(getattr(model, "model", None), "_local_files_only", False)
    )

    resolved = resolved_encoders(model)
    absent = sorted(name for name in ENCODER_NAMES if not resolved[name]["callable"])
    if absent:
        raise ControlRefused(
            f"backend exposes no such encoder; {asymmetric_profile} could not build here, which "
            f"is a different finding and not the one this script measures",
            {"absent": absent, "environment": environment},
        )

    encoders: dict[str, Encoder] = {name: getattr(model, name) for name in ENCODER_NAMES}
    rows = probe_rows(encoders, expect_dimension=expect_dimension)
    coverage = assert_coverage(rows)
    positive = positive_control(encoders[BASE_ENCODER])
    determinism = determinism_control(encoders[BASE_ENCODER])
    sensitivity = sensitivity_control(encoders[BASE_ENCODER])

    identical = all(
        all(row[f"bytes_{BASE_ENCODER}_eq_{name}"] for name in ASYMMETRIC_ENCODERS) for row in rows
    )
    return {
        # Named so a reader can tell this from a `PromotionDecision` without inferring from
        # the filename. It sits in results/promotion/ beside decisions and is NOT one: no arm
        # was scored and no gate was evaluated.
        "schema": "recall-encoder-distinctness-v1",
        # Which code produced this result. See `script_sha256`.
        "script_sha256": script_sha256(),
        "environment": environment,
        "resolved_encoders": resolved,
        "coverage_control": coverage,
        "probes": rows,
        "positive_control_two_different_texts": positive,
        "determinism_control": determinism,
        "sensitivity_control_small_rotation": sensitivity,
        "all_probes_identical": identical,
        "verdict": build_verdict(
            identical=identical,
            resolved=resolved,
            symmetric_profile=symmetric_profile,
            asymmetric_profile=asymmetric_profile,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="check_profile_encoder_distinctness", description=__doc__)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--expect-dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--symmetric-profile", default=DEFAULT_SYMMETRIC_PROFILE)
    parser.add_argument("--asymmetric-profile", default=DEFAULT_ASYMMETRIC_PROFILE)
    parser.add_argument(
        "--expect-artifact-sha256",
        default=None,
        help=(
            "bind the result to one provisioned tree. Without it the digest is still recomputed "
            "and reported, but nothing refuses a run against the wrong artifact."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exit_signal:
        # argparse exits 2 on a usage error, which is this script's "a control refused" code. A
        # mistyped flag and a failed control must not be the same observation.
        #
        # But argparse also raises SystemExit(0) for `--help`, and the first version of this fix
        # caught that too, so a CORRECT invocation started reporting a usage error. The
        # anti-regression gate measured it: `--help` exited 0 before the fix and 64 after. Fixing
        # a collision by breaking the success path is drift past the finding.
        if exit_signal.code in (0, None):
            raise
        raise SystemExit(EXIT_USAGE) from exit_signal

    try:
        result = measure(
            args.cache_dir,
            args.model_name,
            expect_dimension=args.expect_dimension,
            expect_artifact_sha256=args.expect_artifact_sha256,
            symmetric_profile=args.symmetric_profile,
            asymmetric_profile=args.asymmetric_profile,
        )
    except ControlRefused as refusal:
        # allow_nan=False everywhere: a bare `NaN` token is not valid JSON and this artifact is
        # archived and read back by other tools.
        print(json.dumps(refusal.payload(), indent=2, allow_nan=False), file=sys.stderr)
        return EXIT_REFUSED

    print(json.dumps(result, indent=2, allow_nan=False))
    return EXIT_VERDICT


if __name__ == "__main__":
    raise SystemExit(main())
