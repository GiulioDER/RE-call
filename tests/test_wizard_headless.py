"""The headless driver: one config file in, one report out, and what it refuses to guess.

This is the entry point CI and the installer both use, so its contract is narrower than it looks.
Three properties carry most of the weight.

**A missing or malformed config is refused by name.** The user of this file is a first-run installer
or a CI job, neither of which can interpret a `KeyError` on line 40 of a module they have never
heard of.

**One corpus refusing does not abort the others.** The corpora are independent, and a driver that
dies on the second of three leaves the user with less than one that reports which corpus failed and
why. That is the same reasoning as the degraded path inside `run_corpus`, one level up.

**The schema is applied at the embedder's own dimension.** The vector dimension is welded to the
table, so a hardcoded dimension here would produce a schema no chosen embedder fits, discovered as
an insert error minutes into the first build.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from recall.wizard.headless import HeadlessConfig, PipelineRefusal, load_config


def _config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dsn": "postgresql://recall:recall@127.0.0.1:1/recall",
        "migration_dsn": "postgresql://recall:recall@127.0.0.1:1/recall",
        "embedder": "hashing",
        "corpus_version": "2026-08-18",
        "docs_root": str(tmp_path / "docs"),
        "code_root": str(tmp_path / "repo"),
        "memory_root": str(tmp_path / "memory"),
    }
    payload.update(overrides)
    return payload


def _write(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "wizard.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------------------------
# The config contract
# ----------------------------------------------------------------------------------------------


def test_a_complete_config_loads(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, _config(tmp_path)))

    assert isinstance(config, HeadlessConfig)
    assert config.embedder == "hashing"
    assert config.corpus_version == "2026-08-18"
    assert config.docs_root == tmp_path / "docs"


@pytest.mark.parametrize(
    "missing",
    ["dsn", "migration_dsn", "embedder", "corpus_version", "docs_root", "code_root", "memory_root"],
)
def test_every_required_key_is_named_when_absent(tmp_path: Path, missing: str) -> None:
    """Named individually, because "invalid config" sends the reader to read the whole file.

    Parametrised over the real key list rather than a hand-picked sample, so a key added to the
    dataclass without a message is a failing test rather than a silent gap.
    """
    payload = _config(tmp_path)
    del payload[missing]

    with pytest.raises(PipelineRefusal, match=missing):
        load_config(_write(tmp_path, payload))


def test_a_relative_root_is_refused_with_the_key_that_holds_it(tmp_path: Path) -> None:
    """`CorpusSpec` refuses a relative root because it would stamp the wizard's own commit.

    That refusal names the path and not the config key, so the driver has to say which key to fix.
    """
    with pytest.raises(PipelineRefusal, match="docs_root"):
        load_config(_write(tmp_path, _config(tmp_path, docs_root="relative/docs")))


def test_a_config_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "wizard.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(PipelineRefusal, match="a JSON object"):
        load_config(path)


def test_unreadable_or_malformed_json_says_which(tmp_path: Path) -> None:
    absent = tmp_path / "nope.json"
    with pytest.raises(PipelineRefusal, match="cannot read"):
        load_config(absent)

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(PipelineRefusal, match="cannot read"):
        load_config(broken)


# ----------------------------------------------------------------------------------------------
# What the driver does with the plan
# ----------------------------------------------------------------------------------------------


class _Spy:
    """Captures what the driver asked for, without touching a database."""

    def __init__(self, *, refuse: set[str] | None = None) -> None:
        self.schema_dims: list[int] = []
        self.corpora: list[str] = []
        self._refuse = refuse or set()

    def apply_schema(self, dsn: str, *, dim: int) -> None:
        self.schema_dims.append(dim)

    def run(self, spec: Any, **kwargs: Any) -> Any:
        from recall.wizard.pipeline import CorpusOutcome

        self.corpora.append(spec.tenant)
        if spec.tenant in self._refuse:
            raise PipelineRefusal(f"corpus {spec.tenant!r} refused for the test")
        return CorpusOutcome(
            tenant=spec.tenant,
            generation_id=f"gen_{spec.tenant}",
            calibration_id=f"cal_{spec.tenant}",
            certified=True,
            promoted=True,
        )


def test_only_the_calibrated_corpora_are_driven(tmp_path: Path) -> None:
    """`memory` is indexed into the legacy table and has no generation.

    Handing it to `run_corpus` produces a promoted generation nothing can calibrate, which the
    pipeline refuses; the driver must not offer it in the first place.
    """
    from recall.wizard.headless import run_headless

    spy = _Spy()
    report = run_headless(load_config(_write(tmp_path, _config(tmp_path))), services=spy)

    assert spy.corpora == ["docs", "code"]
    assert "memory" not in spy.corpora
    assert report.skipped == ("memory",)


def test_the_schema_is_applied_at_the_embedders_own_dimension(tmp_path: Path) -> None:
    """The dimension is welded to the table, so a hardcoded one fits no embedder but its own.

    `hashing` is 64. Asserted against `resolve_embedder`'s answer rather than the literal, so the
    test cannot drift from the embedder it names.
    """
    from recall.embeddings import resolve_embedder
    from recall.wizard.headless import run_headless

    spy = _Spy()
    run_headless(load_config(_write(tmp_path, _config(tmp_path))), services=spy)

    assert spy.schema_dims == [resolve_embedder("hashing").dim]


def test_one_corpus_refusing_does_not_abort_the_others(tmp_path: Path) -> None:
    """The corpora are independent, so a refusal is reported rather than raised.

    A driver that dies on the first of two leaves the user with nothing, and with no way to tell
    whether the second would have worked.
    """
    from recall.wizard.headless import run_headless

    spy = _Spy(refuse={"docs"})
    report = run_headless(load_config(_write(tmp_path, _config(tmp_path))), services=spy)

    assert spy.corpora == ["docs", "code"], "the second corpus must still be attempted"
    assert [r.tenant for r in report.refused] == ["docs"]
    assert [o.tenant for o in report.outcomes] == ["code"]
    assert report.ok is False, "a refusal is not a successful install"


def test_a_fully_certified_run_reports_ok(tmp_path: Path) -> None:
    """The allow path, so `ok` cannot be satisfied by always being False."""
    from recall.wizard.headless import run_headless

    report = run_headless(load_config(_write(tmp_path, _config(tmp_path))), services=_Spy())

    assert report.ok is True
    assert [o.tenant for o in report.outcomes] == ["docs", "code"]
    assert report.refused == ()


def test_a_degraded_corpus_is_not_a_failed_install(tmp_path: Path) -> None:
    """Certification failing is a measurement. The install completes and says which corpus is thin.

    This is the distinction the exit code turns on: a REFUSAL is a configuration problem the user
    must act on, a DEGRADED corpus is a working install with a named limitation.
    """
    from recall.wizard.headless import run_headless
    from recall.wizard.pipeline import CorpusOutcome

    class _Degrading(_Spy):
        def run(self, spec: Any, **kwargs: Any) -> Any:
            self.corpora.append(spec.tenant)
            return CorpusOutcome(
                tenant=spec.tenant,
                generation_id=f"gen_{spec.tenant}",
                calibration_id=f"cal_{spec.tenant}",
                certified=False,
                promoted=False,
                degraded_reason="separability below the bar",
            )

    report = run_headless(load_config(_write(tmp_path, _config(tmp_path))), services=_Degrading())

    assert report.ok is True, "degraded is not refused"
    assert report.degraded == ("docs", "code")
    assert all(o.certified is False for o in report.outcomes)


def test_the_report_renders_every_corpus_and_its_state(tmp_path: Path) -> None:
    """The text a headless caller actually reads, so it must name each corpus and what happened."""
    from recall.wizard.headless import run_headless

    report = run_headless(load_config(_write(tmp_path, _config(tmp_path))), services=_Spy(refuse={"code"}))
    rendered = report.render()

    assert "docs" in rendered and "code" in rendered
    assert "memory" in rendered, "a skipped corpus must still be accounted for"
    assert "refused" in rendered.lower()


# ----------------------------------------------------------------------------------------------
# The CLI surface. The exit code is the only thing CI reads, so it is the part worth pinning.
# ----------------------------------------------------------------------------------------------


def _cli(config_path: Path, *extra: str) -> tuple[int, str]:
    """Run `recall wizard`, returning its exit code and any message it exited with.

    `SystemExit` carries either an int or a string: `SystemExit("message")` prints the message and
    exits 1. Coercing that with `int()` raises `ValueError` and turns an operator-facing refusal
    into a test-harness crash, which is what the first version of this helper did.
    """
    from recall.cli import main as cli_main

    try:
        cli_main(
            [
                "--serving-dsn",
                "postgresql://recall:recall@127.0.0.1:1/recall",
                "wizard",
                "--config",
                str(config_path),
                *extra,
            ]
        )
    except SystemExit as exit_info:
        code = exit_info.code
        if isinstance(code, str):
            return 1, code
        return int(code or 0), ""
    return 0, ""


def test_a_refused_corpus_exits_nonzero_and_a_degraded_one_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The distinction the exit code exists to carry.

    A REFUSAL is a configuration problem the user must act on. A DEGRADED corpus is a working
    install with a named limitation: built, validated, not certified, deliberately not promoted.
    Collapsing the two would make "did the install work" unanswerable from the exit code, which is
    the only thing a CI job reads.
    """
    import recall.wizard.headless as headless
    from recall.wizard.pipeline import CorpusOutcome

    config = _write(tmp_path, _config(tmp_path))

    monkeypatch.setattr(
        headless,
        "run_headless",
        lambda cfg, services=None: headless.HeadlessReport(
            outcomes=(CorpusOutcome(tenant="docs", generation_id="g", certified=False),),
            refused=(headless.Refusal(tenant="code", reason="corpus is empty"),),
            skipped=("memory",),
        ),
    )
    assert _cli(config, "--headless")[0] == 1
    assert "REFUSED" in capsys.readouterr().out

    monkeypatch.setattr(
        headless,
        "run_headless",
        lambda cfg, services=None: headless.HeadlessReport(
            outcomes=(CorpusOutcome(tenant="docs", generation_id="g", certified=False,
                                    degraded_reason="separability below the bar"),),
            skipped=("memory",),
        ),
    )
    assert _cli(config, "--headless")[0] == 0, "a degraded corpus completes, it does not fail"
    assert "DEGRADED" in capsys.readouterr().out


def test_a_bad_config_is_a_message_naming_the_key_not_a_traceback(tmp_path: Path) -> None:
    """The reader is an installer or a CI log, and neither can act on a stack trace.

    ⚠️ The first version of this assertion ended in `or True`, which made it pass for any input
    whatsoever — a guard that cannot fail, written into the very test meant to prove the message is
    useful. It asserts the key by name now.
    """
    payload = _config(tmp_path)
    del payload["embedder"]

    code, message = _cli(_write(tmp_path, payload), "--headless")

    assert code == 1
    assert "embedder" in message, "the refusal must name the key the operator has to fix"
    assert "Traceback" not in message


def test_the_command_refuses_to_run_unattended_without_being_asked(
    tmp_path: Path,
) -> None:
    """`--headless` is required and named rather than implied.

    The interactive and GUI front ends do not exist yet, so a bare `recall wizard` that silently ran
    unattended would be the wrong surprise for an installer: it builds, calibrates and promotes.
    """
    from recall.cli import main as cli_main

    with pytest.raises(SystemExit) as exit_info:
        cli_main(
            [
                "--serving-dsn",
                "postgresql://recall:recall@127.0.0.1:1/recall",
                "wizard",
                "--config",
                str(_write(tmp_path, _config(tmp_path))),
            ]
        )

    assert "headless" in str(exit_info.value)


def test_the_wizard_subcommand_is_registered_and_documented() -> None:
    """`--config` is required, so a bare invocation cannot start guessing at paths."""
    from recall.cli import main as cli_main

    with pytest.raises(SystemExit) as exit_info:
        cli_main(["wizard"])

    assert exit_info.value.code == 2, "argparse must refuse a missing --config at parse time"
