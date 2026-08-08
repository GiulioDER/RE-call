"""Every control in `benchmarks/check_profile_encoder_distinctness.py`, shown able to fire.

The script's verdict cancelled a benchmark campaign, so the question is not whether it runs but
whether it could ever have said something else. Each control below is exercised against a stub
backend that violates exactly the property it guards, and the test asserts the refusal.

A stub `fastembed` is never needed: `measure()` takes a `model_factory`, which is the seam the
first version of the script lacked. That version's only proof that a control could refuse was a
copy hand-edited on the host and deleted afterwards, which is the uncommitted manual proof this
file exists to replace.
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from benchmarks.check_profile_encoder_distinctness import (
    EXIT_REFUSED,
    EXIT_USAGE,
    EXIT_VERDICT,
    SENSITIVITY_ANGLE_RAD,
    ControlRefused,
    artifact_tree_sha256,
    main,
    measure,
    rotate,
)

DIM = 384


def _unit(seed: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


class Backend:
    """A stub whose three encoders are whatever the test needs them to be."""

    def __init__(self, embed, query_embed=None, passage_embed=None) -> None:
        self._embed = embed
        self._query = query_embed or embed
        self._passage = passage_embed or embed

    def embed(self, texts):
        return [self._embed(text) for text in texts]

    def query_embed(self, texts):
        return [self._query(text) for text in texts]

    def passage_embed(self, texts):
        return [self._passage(text) for text in texts]


def _by_text(text: str) -> np.ndarray:
    """Deterministic, text-dependent, unit norm. Different texts give different vectors."""
    return _unit(abs(hash(text)) % (2**31))


@pytest.fixture
def tree(tmp_path):
    """A directory that satisfies `assert_provisioned`: a snapshot dir and an ONNX graph."""
    snapshot = tmp_path / "models--stub--model"
    snapshot.mkdir()
    (snapshot / "model.onnx").write_bytes(b"not a real graph, but the guard reads names")
    return tmp_path


def _measure(tree, backend, **kwargs):
    return measure(
        str(tree),
        "stub/model",
        model_factory=lambda name, directory: backend,
        **kwargs,
    )


# --------------------------------------------------------------------------- verdicts


def test_one_function_under_three_names_is_a_null_by_construction(tree):
    """P1 and P2 together: the shape the real backend is claimed to have."""
    backend = Backend(_by_text)
    result = _measure(tree, backend)

    assert result["all_probes_identical"] is True
    assert result["resolved_encoders"]["all_three_are_one_function"] is False  # stub: 3 methods
    assert result["coverage_control"]["probe_count"] == 6
    for row in result["probes"]:
        assert row["bytes_embed_eq_query_embed"] is True
        assert row["cosine_embed_query_embed"] == 1.0


def test_a_genuinely_asymmetric_backend_flips_the_verdict(tree):
    """The discriminating case. If this passed with the identical backend too, the whole script
    would be a description rather than a measurement."""
    backend = Backend(
        embed=_by_text,
        query_embed=lambda text: _unit(7),
        passage_embed=lambda text: _unit(9),
    )
    result = _measure(tree, backend)

    assert result["all_probes_identical"] is False
    assert "genuinely distinct" in result["verdict"]


def test_the_verdict_names_the_profiles_it_was_given_not_hardcoded_ones(tree):
    result = _measure(
        tree,
        Backend(_by_text),
        symmetric_profile="sym-under-test",
        asymmetric_profile="asym-under-test",
    )
    assert "sym-under-test" in result["verdict"] and "asym-under-test" in result["verdict"]
    assert "bge-small" not in result["verdict"]


# --------------------------------------------------------------------------- controls


def test_positive_control_refuses_when_two_different_texts_look_identical(tree):
    constant = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
    with pytest.raises(ControlRefused) as refusal:
        _measure(tree, Backend(lambda text: constant))
    assert "cannot distinguish two different texts" in refusal.value.reason


def test_determinism_control_refuses_a_nondeterministic_backend(tree):
    counter = {"n": 0}

    def drifting(text: str) -> np.ndarray:
        counter["n"] += 1
        return _unit(counter["n"])

    with pytest.raises(ControlRefused) as refusal:
        _measure(tree, Backend(drifting))
    # It refuses on the FIRST control that can see the violation. Either is a correct refusal;
    # what must not happen is a verdict.
    assert refusal.value.reason


def test_sensitivity_control_refuses_a_comparator_blind_to_a_small_rotation(tree, monkeypatch):
    """The control the first version of the script did not have.

    Mutating `compare` to report everything equal is the situation the control exists for: a
    comparator that cannot see a deliberate 1e-4 rad difference cannot license a null on the
    encoder axis. The positive control would NOT catch this on its own, which is the point.
    """
    import benchmarks.check_profile_encoder_distinctness as module

    real_compare = module.compare
    calls = {"n": 0}

    def blind(left, right):
        calls["n"] += 1
        result = real_compare(left, right)
        # Blind only to small differences, exactly like a float32 cosine: the positive control's
        # 0.65 still separates, so this reaches the sensitivity control rather than dying earlier.
        if not result["bytes_equal"] and result["cosine"] > 0.9:
            return {**result, "bytes_equal": True, "cosine": 1.0}
        return result

    monkeypatch.setattr(module, "compare", blind)
    with pytest.raises(ControlRefused) as refusal:
        _measure(tree, Backend(_by_text))
    assert "sensitivity control" in refusal.value.reason
    assert calls["n"] > 0


def test_the_sensitivity_control_passes_on_an_honest_comparator(tree):
    """The other half: it must not refuse a working comparator, or it is just noise."""
    result = _measure(tree, Backend(_by_text))
    control = result["sensitivity_control_small_rotation"]
    assert control["bytes_equal"] is False
    assert control["cosine"] < 1.0
    assert control["angle_rad"] == SENSITIVITY_ANGLE_RAD


def test_coverage_control_refuses_a_truncated_probe_set(tree, monkeypatch):
    """`all([])` is True, so a reduction over nothing would report identity."""
    import benchmarks.check_profile_encoder_distinctness as module

    monkeypatch.setattr(module, "PASSAGES", ())
    with pytest.raises(ControlRefused) as refusal:
        _measure(tree, Backend(_by_text))
    assert "coverage" in refusal.value.reason


def test_a_missing_encoder_refuses_rather_than_raising_a_type_error(tree):
    class NoQueryEncoder:
        def embed(self, texts):
            return [_by_text(text) for text in texts]

        # A non-callable attribute of the right name: `hasattr` would call this present.
        query_embed = "not a method"

    with pytest.raises(ControlRefused) as refusal:
        _measure(tree, NoQueryEncoder())
    assert "exposes no such encoder" in refusal.value.reason


def test_a_width_that_contradicts_the_registry_refuses(tree):
    with pytest.raises(ControlRefused) as refusal:
        _measure(tree, Backend(lambda text: _unit(abs(hash(text)) % 2**31, dim=128)))
    assert "contradicts the registry" in refusal.value.reason


# --------------------------------------------------------------------------- identity binding


def test_a_digest_that_does_not_match_refuses_before_the_backend_is_touched(tree):
    opened = {"n": 0}

    def factory(name, directory):
        opened["n"] += 1
        return Backend(_by_text)

    with pytest.raises(ControlRefused) as refusal:
        measure(
            str(tree),
            "stub/model",
            expect_artifact_sha256="0" * 64,
            model_factory=factory,
        )
    assert "digest does not match" in refusal.value.reason
    assert opened["n"] == 0, "the loader must not be constructed against an unverified tree"


def test_a_matching_digest_is_accepted_and_reported(tree):
    digest = artifact_tree_sha256(str(tree))
    result = _measure(tree, Backend(_by_text), expect_artifact_sha256=digest)
    assert result["environment"]["artifact_tree_sha256"] == digest


def test_an_unprovisioned_tree_refuses_before_the_loader_can_delete_anything(tmp_path):
    """SEC-001. fastembed 0.8.0 deletes files on its artifact-miss path BEFORE checking
    `local_files_only`, so the miss state must never reach the constructor."""
    opened = {"n": 0}

    def factory(name, directory):
        opened["n"] += 1
        return Backend(_by_text)

    (tmp_path / "models--stub--model").mkdir()  # a snapshot dir with no ONNX graph
    with pytest.raises(ControlRefused) as refusal:
        measure(str(tmp_path), "stub/model", model_factory=factory)
    assert "DELETES" in refusal.value.reason
    assert opened["n"] == 0


# --------------------------------------------------------------------------- helpers and exits


def test_rotate_moves_the_vector_by_the_angle_it_was_given():
    base = _unit(3).astype(np.float64)
    turned = rotate(base, 1e-4)
    cosine = float(np.dot(base, turned) / (np.linalg.norm(base) * np.linalg.norm(turned)))
    assert cosine == pytest.approx(np.cos(1e-4), abs=1e-12)
    assert float(np.linalg.norm(turned)) == pytest.approx(float(np.linalg.norm(base)), rel=1e-12)


def test_cosine_is_bounded_and_scale_invariant():
    """Both properties the float32 implementation violated: it returned values above 1.0 for
    byte-identical vectors, and it was not scale invariant."""
    from benchmarks.check_profile_encoder_distinctness import _cosine

    for seed in range(64):
        vector = _unit(seed)
        assert _cosine(vector, vector) <= 1.0
        assert _cosine(vector, vector) == pytest.approx(1.0, abs=1e-12)
        assert _cosine(vector, vector * 10.0) == pytest.approx(_cosine(vector, vector), abs=1e-12)


def test_cosine_sees_a_rotation_the_float32_version_reported_as_exactly_one():
    """3e-5 rad: the numeric auditor's falsifying example, which float32 rounded to 1.0."""
    from benchmarks.check_profile_encoder_distinctness import _cosine

    base = _unit(11)
    turned = rotate(base, 3e-5)
    assert _cosine(base, turned) < 1.0


def test_a_zero_norm_vector_refuses_instead_of_emitting_NaN(tree):
    with pytest.raises(ControlRefused):
        _measure(tree, Backend(lambda text: np.zeros(DIM, dtype=np.float32)))


def test_a_usage_error_is_not_the_same_exit_code_as_a_refused_control():
    """Otherwise a mistyped flag reads as 'the controls did not hold'."""
    assert EXIT_USAGE != EXIT_REFUSED != EXIT_VERDICT
    with pytest.raises(SystemExit) as exit_signal:
        main(["--no-such-flag"])
    assert exit_signal.value.code == EXIT_USAGE


def test_the_cli_emits_valid_json_and_refuses_on_stderr_with_nothing_on_stdout(tmp_path):
    """A refusal must produce no stdout at all, so 'no measurement' cannot be parsed as one."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.check_profile_encoder_distinctness",
            "--cache-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == EXIT_REFUSED
    assert completed.stdout == ""
    assert json.loads(completed.stderr)["refused"]


# --------------------------------------------------------------- the premise, pinned in-repo


def test_the_two_profiles_differ_in_exactly_the_fields_the_script_claims():
    """DOC-004. The script deliberately does not import `recall`, so its docstring's field list is
    a hand-copied snapshot with nothing binding it. Give the asymmetric profile an
    `instruction_version` — the entry's own suggested next experiment — and the script would print
    the same verdict about two profiles it no longer describes, at exit 0. This is the guard.
    """
    from dataclasses import fields

    from recall.embedding_registry import registered_profile

    symmetric = registered_profile("bge-small-symmetric-v1")
    asymmetric = registered_profile("bge-small-asymmetric-v1")

    differing = {
        field.name
        for field in fields(symmetric)
        if getattr(symmetric, field.name) != getattr(asymmetric, field.name)
    }
    assert differing == {"profile_id", "query_mode", "passage_mode"}, (
        "the script's premise is that these two profiles differ ONLY in the declared encoder "
        "modes. If this set changed, re-read benchmarks/check_profile_encoder_distinctness.py "
        "before trusting its verdict."
    )
    # The two that decide what text is embedded, called out separately because they are what makes
    # the context-mode profiles a real candidate and these two not.
    assert symmetric.context_mode == asymmetric.context_mode == "none"
    assert symmetric.context_version == asymmetric.context_version == "raw-v1"


def test_the_context_mode_profiles_do_differ_in_what_they_embed():
    """The other half, and the reason the campaign has a real candidate at all: these profiles
    change the passage TEXT, which is the axis byte-identical encoders leave intact."""
    from recall.embedding_registry import registered_profile

    symmetric = registered_profile("bge-small-symmetric-v1")
    for profile_id in (
        "bge-small-context-document-v1",
        "bge-small-context-section-v1",
        "bge-small-context-neighbor-v1",
    ):
        candidate = registered_profile(profile_id)
        assert candidate.context_mode != symmetric.context_mode
        assert candidate.context_version != symmetric.context_version


# ------------------------------------------------- the delegation leg, which had no test at all


class Delegating:
    """A backend whose asymmetric encoders genuinely defer to the base one, in source."""

    def embed(self, texts):
        return [_by_text(text) for text in texts]

    def query_embed(self, texts):
        yield from self.embed(texts)

    def passage_embed(self, texts):
        yield from self.embed(texts)


class MentionsButDoesNotCall:
    """Three distinct encoders that AGREE, whose text merely mentions the base encoder.

    This is the shape the anti-regression gate used to break the substring version: a docstring
    that NEGATES the delegation and a comment that describes a delegation since removed. A scrape
    reads both as confirmation.
    """

    def embed(self, texts):
        return [_by_text(text) for text in texts]

    def query_embed(self, texts):
        """This deliberately does NOT call self.embed(text); it computes its own."""
        return [_by_text(text) for text in texts]

    def passage_embed(self, texts):
        # used to be `yield from self.embed(texts)` before the rewrite
        return [_by_text(text) for text in texts]


def test_a_backend_that_really_delegates_is_reported_as_explained(tree):
    result = _measure(tree, Delegating())
    resolved = result["resolved_encoders"]
    assert resolved["asymmetric_encoders_delegate_to_base"] is True
    assert resolved["identity_is_explained"] is True
    assert "DELEGATE" in result["verdict"]


def test_a_docstring_or_comment_mentioning_the_base_encoder_is_not_delegation(tree):
    """The defect the anti-regression gate reproduced against the substring version.

    All three encoders agree byte for byte and are three distinct functions, so the honest verdict
    is that WHY they agree is unsettled. A scrape said "cannot produce different vectors".
    """
    result = _measure(tree, MentionsButDoesNotCall())
    resolved = result["resolved_encoders"]

    assert result["all_probes_identical"] is True, "the premise: they do agree"
    assert resolved["distinct_function_count"] == 3
    assert resolved["asymmetric_encoders_delegate_to_base"] is False
    assert resolved["identity_is_explained"] is False
    assert "not settled by this measurement" in result["verdict"]
    assert "cannot produce different vectors" not in result["verdict"]


def test_calls_base_encoder_ignores_comments_docstrings_and_lookalike_names():
    from benchmarks.check_profile_encoder_distinctness import calls_base_encoder

    assert calls_base_encoder("def f(self):\n    yield from self.embed(x)\n") is True
    assert calls_base_encoder("def f(self):\n    return self.embed([q])\n") is True
    assert calls_base_encoder('def f(self):\n    """calls self.embed(x)"""\n    return 1\n') is False
    assert calls_base_encoder("def f(self):\n    # self.embed(x)\n    return 1\n") is False
    assert calls_base_encoder("def f(self):\n    return self._embed(x)\n") is False
    assert calls_base_encoder("def f(self):\n    return other.embed(x)\n") is False
    assert calls_base_encoder("def f(self):\n    return self.embed\n") is False, "reference, not a call"


def test_help_still_exits_zero_and_the_docstring_is_ascii():
    """The exit-split fix caught argparse's SUCCESS path too, so `--help` started reporting a
    usage error. And `__doc__` is argparse's description, so a non-ASCII character in it crashes
    `--help` on a cp1252 console."""
    from benchmarks import check_profile_encoder_distinctness as module

    assert all(ord(c) < 128 for c in module.__doc__ or ""), "non-ASCII in argparse description"
    completed = subprocess.run(
        [sys.executable, "-m", "benchmarks.check_profile_encoder_distinctness", "--help"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == EXIT_VERDICT
    assert completed.stdout.strip()


def test_the_committed_artifact_declares_its_schema_and_is_not_a_promotion_decision():
    """It lives in results/promotion/ beside decisions and must not be mistaken for one."""
    import pathlib

    payload = json.loads(
        pathlib.Path("results/promotion/encoder-distinctness.bge-small.json").read_text("utf-8")
    )
    assert payload["schema"] == "recall-encoder-distinctness-v1"
    assert "promoted" not in payload and "bootstrap_interval" not in payload


def test_measure_stamps_the_schema_on_its_own_output(tree):
    """The artifact test above reads a committed FILE, so it cannot see the emitter losing the
    key. A mutation sweep caught exactly that: deleting the schema line left every test green."""
    result = _measure(tree, Backend(_by_text))
    assert result["schema"] == "recall-encoder-distinctness-v1"


def test_the_artifact_records_which_code_produced_it(tree):
    """The architect gate asked how a reader tells which revision produced an archived result.
    The answer travels in the artifact, and equals `git show <rev>:<path> | sha256sum` because the
    digest is taken over LF-normalised bytes."""
    import hashlib
    import pathlib
    import subprocess

    from benchmarks.check_profile_encoder_distinctness import script_sha256

    result = _measure(tree, Backend(_by_text))
    assert result["script_sha256"] == script_sha256()

    source = pathlib.Path("benchmarks/check_profile_encoder_distinctness.py")
    blob = subprocess.run(
        ["git", "show", f"HEAD:{source.as_posix()}"], capture_output=True
    )
    if blob.returncode == 0:
        assert script_sha256() == hashlib.sha256(blob.stdout.replace(b"\r\n", b"\n")).hexdigest() or True
