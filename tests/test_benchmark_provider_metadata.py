from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.artifact_contract import reject_unauditable_cost_claims
from recall.provider_metadata import ProviderMetadata


def test_benchmark_cost_claim_rejects_missing_provider_metadata() -> None:
    with pytest.raises(ValueError, match="provider_metadata"):
        reject_unauditable_cost_claims({"cost_claims": [{"claim": "memory layer cost"}]})


def test_benchmark_cost_claim_rejects_missing_revision_or_cost() -> None:
    payload = {
        "cost_claims": [{"claim": "memory layer cost"}],
        "provider_metadata": [
            {
                "provider_id": "openrouter",
                "model_id": "openai/gpt-4o-mini",
                "model_revision": None,
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
                "latency_ms": 20,
                "monetary_cost_usd": None,
            }
        ],
    }

    with pytest.raises(ValueError, match="model_revision"):
        reject_unauditable_cost_claims(payload)

    payload["provider_metadata"][0]["model_revision"] = "rev"
    with pytest.raises(ValueError, match="monetary_cost_usd"):
        reject_unauditable_cost_claims(payload)


def test_provider_metadata_rejects_invalid_identity_cost_and_token_totals() -> None:
    valid = {
        "provider_id": "openrouter",
        "model_id": "openai/gpt-4o-mini",
        "model_revision": "rev",
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
        "latency_ms": 20,
        "monetary_cost_usd": 0.001,
    }

    for field in ("provider_id", "model_id"):
        bad = dict(valid)
        bad[field] = None
        with pytest.raises(ValueError, match="non-empty string"):
            reject_unauditable_cost_claims(
                {"cost_claims": [{"claim": "cost"}], "provider_metadata": [bad]}
            )

    with pytest.raises(ValueError, match="provider_id"):
        ProviderMetadata(provider_id=123, model_id="model", model_revision="rev")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model_id"):
        ProviderMetadata(provider_id="provider", model_id=True, model_revision="rev")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="provider_id"):
        ProviderMetadata(provider_id=" ", model_id="model", model_revision="rev")
    with pytest.raises(ValueError, match="model_id"):
        ProviderMetadata(provider_id="provider", model_id="", model_revision="rev")
    with pytest.raises(ValueError, match="prompt_tokens"):
        ProviderMetadata(
            provider_id="provider",
            model_id="model",
            model_revision="rev",
            prompt_tokens=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="latency_ms"):
        ProviderMetadata(
            provider_id="provider",
            model_id="model",
            model_revision="rev",
            latency_ms=1.5,  # type: ignore[arg-type]
        )

    bad = dict(valid)
    bad["monetary_cost_usd"] = "NaN"
    with pytest.raises(ValueError, match="finite"):
        reject_unauditable_cost_claims(
            {"cost_claims": [{"claim": "cost"}], "provider_metadata": [bad]}
        )

    bad = dict(valid)
    bad["total_tokens"] = 12
    with pytest.raises(ValueError, match="total_tokens"):
        reject_unauditable_cost_claims(
            {"cost_claims": [{"claim": "cost"}], "provider_metadata": [bad]}
        )


def test_benchmark_artifact_must_declare_cost_claims_even_when_empty() -> None:
    """An absent key is an undeclared posture, not a declaration of no monetary claim."""

    with pytest.raises(ValueError, match="cost_claims"):
        reject_unauditable_cost_claims({"provider_metadata": []})


def test_benchmark_cost_claims_must_be_an_array() -> None:
    with pytest.raises(ValueError, match="cost_claims"):
        reject_unauditable_cost_claims({"cost_claims": "$7.29 per run"})


def test_monetary_prose_is_audited_even_when_cost_claims_is_empty() -> None:
    """The hole this closes: a dollar figure in prose beside an empty `cost_claims` list."""

    with pytest.raises(ValueError, match="provider_metadata"):
        reject_unauditable_cost_claims(
            {"cost_claims": [], "headline": "full LOCOMO arm for $7.29 of tokens"}
        )


def test_monetary_prose_requires_revision_and_cost_fields() -> None:
    payload: dict[str, object] = {
        "cost_claims": [],
        "summary": {"note": "judge spend was 0.75 USD"},
        "provider_metadata": [
            {
                "provider_id": "openrouter",
                "model_id": "openai/gpt-4o-mini",
                "model_revision": None,
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
                "latency_ms": 20,
                "monetary_cost_usd": None,
            }
        ],
    }

    with pytest.raises(ValueError, match="model_revision"):
        reject_unauditable_cost_claims(payload)

    payload["provider_metadata"][0]["model_revision"] = "rev"  # type: ignore[index]
    with pytest.raises(ValueError, match="monetary_cost_usd"):
        reject_unauditable_cost_claims(payload)


def test_monetary_prose_scan_does_not_over_reject_ordinary_artifacts() -> None:
    """Digits in model ids, the word cost without a figure, and verbatim source text all pass."""

    reject_unauditable_cost_claims(
        {
            "cost_claims": [],
            "arm": "recall",
            "model": "openai/gpt-4o-mini",
            "config": {"k": 5, "notes": "cost per question was not measured on this run"},
            "aggregate": {"accuracy": 0.42, "usd": None},
            "outcomes": [
                {"question": "how much did the jacket cost?", "gold": "it was $40"},
            ],
            "provider_metadata": [],
        }
    )


def test_verbatim_source_exclusion_applies_only_at_the_top_level() -> None:
    """A nested key named `outcomes` must not hide a cost claim.

    The exclusion exists for the one top level array of copied in LOCOMO source text. Applied at
    every depth it becomes an audit bypass, and `config["system"]` is `describe()` output from a
    duck typed adapter, so that key namespace is not ours to trust.
    """

    with pytest.raises(ValueError, match="provider_metadata"):
        reject_unauditable_cost_claims(
            {"cost_claims": [], "config": {"system": {"outcomes": "billed $7.29 per run"}}}
        )


def test_monetary_prose_is_found_inside_sets() -> None:
    with pytest.raises(ValueError, match="provider_metadata"):
        reject_unauditable_cost_claims({"cost_claims": [], "tags": {"$7.29 per run"}})


def test_monetary_prose_covers_the_other_currencies_this_project_could_publish() -> None:
    for prose in ("spend was EUR 6.60", "cost €6.60 per run", "£5.00 of tokens"):
        with pytest.raises(ValueError, match="provider_metadata"):
            reject_unauditable_cost_claims({"cost_claims": [], "headline": prose})


def test_sterling_written_in_words_is_still_a_cost_claim() -> None:
    """Dropping the bare word `pounds` must not drop sterling entirely. The disambiguated form
    leaves one weight phrasing matching, "N pounds sterling silver", accepted because this
    project will not publish it."""

    with pytest.raises(ValueError, match="provider_metadata"):
        reject_unauditable_cost_claims(
            {"cost_claims": [], "headline": "total spend: 12 pounds sterling"}
        )


def test_pounds_as_a_unit_of_weight_is_not_a_cost_claim() -> None:
    """A false positive costs the operator a republish, so the bare word `pounds` is
    deliberately not a currency form. The £ symbol and `GBP` still are."""

    reject_unauditable_cost_claims(
        {"cost_claims": [], "provider_metadata": [], "aggregate": {"note": "the box weighs 5 pounds"}}
    )


def test_self_referential_payload_does_not_blow_the_stack() -> None:
    """A cycle must not surface as a RecursionError from the contract. `benchmarks.run`
    serialises first, so json reports the cycle there; this covers direct callers."""

    payload: dict[str, object] = {"cost_claims": []}
    payload["self"] = payload

    reject_unauditable_cost_claims(payload)


def test_write_site_calls_the_validator_before_writing() -> None:
    """A validator the write path stopped calling is a validator that cannot fail."""

    import inspect

    from benchmarks import run as run_module

    source = inspect.getsource(run_module.main)

    assert "reject_unauditable_cost_claims(payload)" in source
    # `_write_atomic(path, ...)`, not `path.write_text`: the publish goes through a temp file and
    # an `os.replace` so a failed write cannot truncate or destroy the target.
    assert source.index("reject_unauditable_cost_claims(payload)") < source.index(
        "_write_atomic(path"
    )


def test_benchmark_without_monetary_claim_may_report_token_metadata_only() -> None:
    reject_unauditable_cost_claims(
        {
            "cost_claims": [],
            "provider_metadata": [
                {
                    "provider_id": "openrouter",
                    "model_id": "openai/gpt-4o-mini",
                    "model_revision": None,
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                    "latency_ms": 20,
                    "monetary_cost_usd": None,
                }
            ],
        }
    )


def test_a_refused_artifact_is_refused_by_every_reader(tmp_path) -> None:
    """The in-band mark is only a contract if the readers enforce it.

    `benchmarks.run` quarantines a refused artifact outside the publishable glob AND marks it,
    but the glob only protects the documented invocation. A reader handed the file directly —
    which is how all of these are invoked — is the case the mark exists for, and it was honoured
    by exactly one of them.
    """
    import json

    from benchmarks.artifact_contract import load_published_artifact

    path = tmp_path / "refused.json"
    path.write_text(
        json.dumps(
            {
                "arm": "recall",
                "aggregate": {"answerable_accuracy": {"rate": 0.99, "n": 2}},
                "unpublished": True,
                "unpublished_reason": "benchmark cost claims require provider_metadata",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="REFUSED publication"):
        load_published_artifact(path)

    ordinary = tmp_path / "fine.json"
    ordinary.write_text(json.dumps({"arm": "recall"}), encoding="utf-8")
    assert load_published_artifact(ordinary)["arm"] == "recall"


def _refused_artifact(tmp_path: Path) -> Path:
    import json

    path = tmp_path / "refused.json"
    path.write_text(
        json.dumps(
            {
                "arm": "recall",
                "aggregate": {"answerable_accuracy": {"rate": 0.99, "n": 2}},
                "outcomes": [],
                "unpublished": True,
                "unpublished_reason": "benchmark cost claims require provider_metadata",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_every_run_artifact_reader_refuses_a_refused_artifact(tmp_path) -> None:
    """Behavioural, per reader. The textual guard below can be evaded by reformatting; this
    cannot, because it calls the real entry points."""
    from benchmarks import (
        analyze,
        h2h_artifact,
        judge_quality,
        locomo_audit,
        rejudge,
        token_f1,
    )

    refused = _refused_artifact(tmp_path)

    readers = {
        "analyze.curve_points": lambda: analyze.curve_points([refused]),
        "h2h_artifact.load_run": lambda: h2h_artifact.load_run(refused),
        "judge_quality._load_document": lambda: judge_quality._load_document(refused),
        "rejudge._load_document": lambda: rejudge._load_document(refused),
        "token_f1.compare": lambda: token_f1.compare(refused, refused),
        "locomo_audit.main": lambda: locomo_audit.main(
            ["--results", str(refused), "--data", str(refused), "--audit", str(refused)]
        ),
    }
    refused_by = []
    for name, call in readers.items():
        try:
            call()
        except SystemExit as exc:
            assert "REFUSED publication" in str(exc), name
            refused_by.append(name)
    assert refused_by == list(readers), f"these did not refuse: {set(readers) - set(refused_by)}"


def test_the_claim_gate_reports_a_refused_artifact_as_a_claim_error(tmp_path) -> None:
    """`resolve` promises ClaimError, and its caller loops collecting failures.

    A SystemExit from the loader would escape that loop and abort the gate on the first refused
    artifact, leaving every later claim unchecked and reporting a bare exception in place of the
    accumulated list.
    """
    from benchmarks.claim_gate import Claim, ClaimError, Marker, resolve

    refused = _refused_artifact(tmp_path)
    claim = Claim(
        doc="docs/RESULTS.md",
        line=42,
        text="0.99",
        marker=Marker(kind="artifact", artifact=refused.name, key="aggregate.answerable_accuracy.rate"),
    )

    with pytest.raises(ClaimError, match="REFUSED publication"):
        resolve(claim, tmp_path)


def test_no_benchmark_tool_reads_a_run_artifact_without_the_publication_check() -> None:
    """Two guards over the whole package, because this contract decays one `json.loads` at a time.

    NEGATIVE: every module under `benchmarks/`, recursively, that parses JSON and names a
    run-artifact key must import the checker. An allowlist of known readers cannot catch the
    thing most worth catching, a new one, and a top-level-only glob misses six subpackages.
    The condition is the literal IMPORT line, not the mere presence of the name: a substring
    check is satisfied by a `# TODO: use load_published_artifact` that says the opposite.

    POSITIVE: `claim_gate` is invisible to the heuristic — it looks keys up from markers and so
    names none of them — and any reader could become invisible the same way. So the readers we
    already know about are asserted to still import the checker.
    """
    from pathlib import Path

    import ast

    def imports_checker(text: str) -> bool:
        """True if the module really IMPORTS the checker.

        Parsed, not grepped. A substring is satisfied by a `# TODO: use load_published_artifact`
        that says the opposite; a literal import LINE is defeated by wrapping the import in
        parentheses, which is exactly what happened to salvage.py the moment it needed a second
        name from that module.
        """
        try:
            tree = ast.parse(text)
        except SyntaxError:  # pragma: no cover - a syntax error is somebody else's test failing
            return False
        guarded: set[int] = set()
        for node in ast.walk(tree):
            # An import under `if TYPE_CHECKING:` never runs, so it is not a use of the checker;
            # it is a cheaper bypass than the parenthesised import that motivated this rewrite.
            if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
                guarded.update(id(inner) for inner in ast.walk(node) if inner is not node)
        return any(
            isinstance(node, ast.ImportFrom)
            and node.module == "benchmarks.artifact_contract"
            and any(alias.name == "load_published_artifact" for alias in node.names)
            and id(node) not in guarded
            for node in ast.walk(tree)
        )

    #: Keys that only a `benchmarks.run` results artifact carries.
    run_keys = (
        "outcomes",
        "aggregate",
        "provider_metadata",
        "cost_claims",
        "adversarial_abstention",
        "answerable_accuracy",
    )
    #: The two modules that legitimately name those keys without reading an artifact.
    exempt = {
        "artifact_contract.py": "defines load_published_artifact",
        "run.py": "writes artifacts; reads only the LOCOMO source corpus",
    }
    known_readers = {
        "analyze.py",
        "claim_gate.py",
        "h2h_artifact.py",
        "judge_quality.py",
        "locomo_audit.py",
        "rejudge.py",
        "salvage.py",
        "token_f1.py",
    }

    # LIVENESS, both directions. `imports_checker` returning True unconditionally makes this
    # whole test pass vacuously — verified: that mutant survived. So prove it accepts a real
    # import and rejects the two spellings that are not one.
    assert imports_checker("from benchmarks.artifact_contract import load_published_artifact\n")
    assert not imports_checker("# TODO: use load_published_artifact one day\n")
    assert not imports_checker(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from benchmarks.artifact_contract import load_published_artifact\n"
    ), "a TYPE_CHECKING import is not a runtime call site"

    root = Path(__file__).resolve().parents[1] / "benchmarks"
    offenders = []
    for module in sorted(root.rglob("*.py")):
        relative = module.relative_to(root).as_posix()
        if relative in exempt:
            continue
        text = module.read_text(encoding="utf-8")
        if "json.load" not in text or imports_checker(text):
            continue
        if any(f'"{key}"' in text or f"'{key}'" in text for key in run_keys):
            offenders.append(relative)

    assert not offenders, (
        "these parse JSON and name run-artifact keys without importing the publication check. "
        "Route them through `load_published_artifact`, or add them to `exempt` with the reason: "
        + ", ".join(offenders)
    )

    # The behavioural test above and this textual list are two hand-maintained inventories of
    # the same thing, which is how they drift. Tie them: every behavioural entry must be a known
    # reader, and the readers deliberately absent from the behavioural dict are named here so
    # adding one forces a decision about the other.
    behavioural = {"analyze", "h2h_artifact", "judge_quality", "locomo_audit", "rejudge", "token_f1"}
    only_textual = {"claim_gate", "salvage"}
    assert {f"{name}.py" for name in behavioural | only_textual} == known_readers, (
        "the behavioural reader dict and known_readers have drifted; update both"
    )

    renamed = sorted(name for name in known_readers if not (root / name).is_file())
    assert not renamed, (
        "a known artifact reader was renamed, moved or deleted: "
        + ", ".join(renamed)
        + ". If it still reads run artifacts, update `known_readers` to its new path — do not "
        "just drop the entry, because it may be one the heuristic above cannot see."
    )
    missing = sorted(
        name
        for name in known_readers
        if not imports_checker((root / name).read_text(encoding="utf-8"))
    )
    assert not missing, f"these known readers dropped the publication check: {missing}"


def test_a_json_file_that_is_not_an_object_is_refused_cleanly(tmp_path: Path) -> None:
    """`doc.get` on a list is an AttributeError, which is not the caller's clean error."""
    import json

    from benchmarks.artifact_contract import load_published_artifact

    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(SystemExit, match="not a JSON object"):
        load_published_artifact(path)


def test_the_claim_gate_reports_a_malformed_artifact_as_a_claim_error(tmp_path: Path) -> None:
    """Refusal is not the only way the load can fail, and the gate accumulates failures.

    A malformed or mis-encoded artifact raises JSONDecodeError/UnicodeDecodeError out of the
    loader. Catching only SystemExit lets those escape `except ClaimError` and abort the gate on
    the first bad file, leaving every later claim unchecked — the same defect the SystemExit
    translation was added to fix, wearing a different exception.
    """
    from benchmarks.claim_gate import Claim, ClaimError, Marker, resolve

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{oops", encoding="utf-8")
    claim = Claim(
        doc="docs/RESULTS.md",
        line=7,
        text="0.99",
        marker=Marker(kind="artifact", artifact=malformed.name, key="aggregate.rate"),
    )

    with pytest.raises(ClaimError):
        resolve(claim, tmp_path)
