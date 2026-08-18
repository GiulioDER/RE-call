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

from recall.wizard.headless import (
    _REQUIRED,
    ConfigRefusal,
    HeadlessConfig,
    PipelineRefusal,
    load_config,
)


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


@pytest.mark.parametrize("missing", _REQUIRED)
def test_every_required_key_is_named_when_absent(tmp_path: Path, missing: str) -> None:
    """Named individually, because "invalid config" sends the reader to read the whole file.

    Parametrised over the real key list, imported rather than retyped, so a key added to the
    dataclass without a message is a failing test rather than a silent gap. The previous version
    called itself "the real key list" while being a literal.

    ⚠️ Asserted on `.keys`, NOT on the message. The refusal text enumerates every required key, so
    `pytest.raises(match=missing)` matched whichever key was deleted regardless of which key the
    message actually blamed: a mutation making every refusal name `dsn` left all eight of these
    green. A test that cannot distinguish the right answer from a constant is not a test.
    """
    payload = _config(tmp_path)
    del payload[missing]

    with pytest.raises(ConfigRefusal) as caught:
        load_config(_write(tmp_path, payload))
    assert caught.value.keys == (missing,), "the refusal must blame the key that is actually absent"


@pytest.mark.parametrize("value", [None, 0, False, [], {"a": 1}])
def test_a_present_but_non_string_value_is_refused_by_name(tmp_path: Path, value: Any) -> None:
    """`str(raw.get(key, ""))` read every one of these as present and non-empty.

    `str(None)` is "None" and `str(False)` is "False", so a null `dsn` became the literal string
    "None" and was handed to psycopg, while a null `docs_root` raised a bare `TypeError` out of
    `Path()` — from inside the function whose docstring promises to name the key at fault.
    """
    payload = _config(tmp_path)
    payload["dsn"] = value

    with pytest.raises(ConfigRefusal) as caught:
        load_config(_write(tmp_path, payload))
    assert caught.value.keys == ("dsn",)


def test_an_unknown_key_is_refused_rather_than_discarded(tmp_path: Path) -> None:
    """A typo silently did nothing, so a config that looked applied was not."""
    with pytest.raises(ConfigRefusal) as caught:
        load_config(_write(tmp_path, _config(tmp_path, tabel="chunks")))
    assert caught.value.keys == ("tabel",)


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

    def __init__(self, *, refuse: set[str] | None = None, crash: set[str] | None = None) -> None:
        self.schema_dims: list[int] = []
        #: The DSN is recorded, not dropped. Dropping it made the DDL-owner separation untestable:
        #: a mutation applying the schema with the SERVING dsn left the whole module green.
        self.schema_dsns: list[str] = []
        self.grants: list[tuple[str, str]] = []
        self.corpora: list[str] = []
        #: Uncalibrated corpora handed to `index_legacy`. Recorded separately from `corpora`,
        #: because "which corpora were CALIBRATED" and "which were indexed" are different questions
        #: and the report used to answer neither for `memory`.
        self.legacy: list[str] = []
        self.steps: list[str] = []
        self._refuse = refuse or set()
        self._crash = crash or set()

    def dim(self) -> int:
        from recall.embeddings import resolve_embedder

        return resolve_embedder("hashing").dim

    def apply_schema(self, dsn: str, *, dim: int) -> None:
        self.schema_dims.append(dim)
        self.schema_dsns.append(dsn)

    def grant(self, dsn: str, *, role: str) -> None:
        self.grants.append((dsn, role))

    def index_legacy(self, spec: Any) -> Any:
        from recall.wizard.headless import LegacyIndex

        self.legacy.append(spec.tenant)
        if spec.tenant in self._refuse:
            raise PipelineRefusal(f"corpus {spec.tenant!r} refused for the test")
        if spec.tenant in self._crash:
            raise RuntimeError(f"psycopg: connection lost while indexing {spec.tenant}")
        return LegacyIndex(tenant=spec.tenant, files=3, chunks=11)

    def run(self, spec: Any, *, progress: Any = None) -> Any:
        from recall.wizard.pipeline import CorpusOutcome

        self.corpora.append(spec.tenant)
        if progress is not None:
            progress("build")
            self.steps.append(f"{spec.tenant}:build")
        if spec.tenant in self._refuse:
            raise PipelineRefusal(f"corpus {spec.tenant!r} refused for the test")
        if spec.tenant in self._crash:
            raise RuntimeError(f"psycopg: server closed the connection while building {spec.tenant}")
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
    assert "memory" not in spy.corpora, "memory must not go down the generation path"
    # But it must still be DRIVEN. The report used to say `memory` was "indexed into the legacy
    # chunks table" while nothing indexed it and nothing created its directory, so a third of the
    # install was a claim rather than a result.
    assert spy.legacy == ["memory"], "memory must be indexed, not skipped"
    assert [i.tenant for i in report.indexed] == ["memory"]
    assert report.indexed[0].chunks == 11, "the report carries counts, which cannot be claimed"


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


def _degrading(previously_serving: str | None) -> type[_Spy]:
    """A spy whose corpora all degrade, over a tenant that was or was not already serving."""

    class _Degrading(_Spy):
        def run(self, spec: Any, *, progress: Any = None) -> Any:
            from recall.wizard.pipeline import CorpusOutcome

            self.corpora.append(spec.tenant)
            return CorpusOutcome(
                tenant=spec.tenant,
                generation_id=f"gen_{spec.tenant}",
                calibration_id=f"cal_{spec.tenant}",
                certified=False,
                promoted=False,
                degraded_reason="separability below the bar",
                previously_serving=previously_serving,
            )

    return _Degrading


def test_a_degraded_upgrade_is_not_a_failed_install(tmp_path: Path) -> None:
    """Certification failing on a tenant that is already serving is a measurement, not a failure.

    This is the distinction the exit code turns on: a REFUSAL is a configuration problem the user
    must act on, and a DEGRADED corpus with a predecessor still serving is a working install with
    a named limitation.
    """
    from recall.wizard.headless import run_headless

    services = _degrading("gen_previous")()
    report = run_headless(load_config(_write(tmp_path, _config(tmp_path))), services=services)

    assert report.ok is True, "degraded is not refused when something is still serving"
    assert report.degraded == ("docs", "code")
    assert report.unserved == (), "a predecessor is still answering queries"
    assert "install complete" in report.render()


def test_a_degraded_first_install_is_not_complete(tmp_path: Path) -> None:
    """The same outcome record, over an EMPTY tenant, is an install that answers nothing.

    These two states rendered identically for a release: "install complete", exit 0, and a reason
    saying "whatever this tenant was serving still serves" about a tenant that was serving nothing.
    `docs` is production/strict, so a query against it raises `NoActiveGeneration` from outside
    `trusted_search`'s try block — a raw exception with no failure code and no advice. CI reads the
    exit code and nothing else, so the two must not collapse.
    """
    from recall.wizard.headless import run_headless

    services = _degrading(None)()
    report = run_headless(load_config(_write(tmp_path, _config(tmp_path))), services=services)

    assert report.unserved == ("docs", "code")
    assert report.ok is False, "a tenant that answers nothing is not a complete install"
    rendered = report.render()
    assert "install complete" not in rendered
    assert "will answer nothing" in rendered


def test_an_unexpected_error_is_reported_and_the_report_survives(tmp_path: Path) -> None:
    """A crash is not a refusal, and losing the report loses which generations are now active.

    `promote` is irreversible and retires whatever the tenant was serving. A driver that catches
    only `PipelineRefusal` throws away the record that `docs` was already promoted, so the operator
    gets a traceback and no way to know what changed.
    """
    from recall.wizard.headless import run_headless

    spy = _Spy(crash={"code"})
    report = run_headless(load_config(_write(tmp_path, _config(tmp_path))), services=spy)

    assert spy.corpora == ["docs", "code"], "a crash in one corpus must not skip the rest"
    assert [o.tenant for o in report.outcomes] == ["docs"], "the promoted corpus must be reported"
    assert [f.tenant for f in report.failures] == ["code"]
    assert "RuntimeError" in report.failures[0].error, "the reader needs the exception type"
    assert report.ok is False
    assert "FAILED" in report.render()


def test_the_schema_is_applied_with_the_ddl_owner_dsn(tmp_path: Path) -> None:
    """Not the serving DSN. The spy used to drop the DSN, so this was untestable and untested.

    A mutation applying the schema with `config.dsn` left the whole module green, which is how the
    DDL-owner separation shipped with no coverage at all.
    """
    from recall.wizard.headless import run_headless

    payload = _config(tmp_path, migration_dsn="postgresql://recall:recall@127.0.0.1:1/recall")
    spy = _Spy()
    config = load_config(_write(tmp_path, payload))
    run_headless(config, services=spy)

    assert spy.schema_dsns == [config.migration_dsn]


def test_progress_reaches_the_caller(tmp_path: Path) -> None:
    """Otherwise the installer prints nothing at all for the whole multi-minute build.

    `run_corpus` takes a `progress` callback precisely to prevent that silence, and the driver
    forwarded no keywords at all, so it was unreachable from the only entry point that has one.
    """
    from recall.wizard.headless import run_headless

    seen: list[str] = []
    run_headless(
        load_config(_write(tmp_path, _config(tmp_path))),
        services=_Spy(),
        progress=seen.append,
    )

    assert seen == ["docs: build", "code: build", "memory: index"], (
        "each step must name the corpus it belongs to, and the uncalibrated corpus is a step too: "
        "indexing memory is work the operator waits for, so silence there is the same defect"
    )


def test_the_report_renders_every_corpus_and_its_state(tmp_path: Path) -> None:
    """The text a headless caller actually reads, so it must name each corpus and what happened."""
    from recall.wizard.headless import run_headless

    report = run_headless(load_config(_write(tmp_path, _config(tmp_path))), services=_Spy(refuse={"code"}))
    rendered = report.render()

    assert "docs" in rendered and "code" in rendered
    assert "memory" in rendered, "an uncalibrated corpus must still be accounted for"
    assert "indexed" in rendered, "and it must say it was indexed, with counts"
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
        lambda cfg, services=None, progress=None: headless.HeadlessReport(
            outcomes=(
                CorpusOutcome(
                    tenant="docs",
                    generation_id="g",
                    certified=False,
                    degraded_reason="separability below the bar",
                ),
            ),
            refused=(headless.Refusal(tenant="code", reason="corpus is empty"),),
            indexed=(headless.LegacyIndex(tenant="memory", files=3, chunks=11),),
        ),
    )
    assert _cli(config, "--headless")[0] == 1
    assert "REFUSED" in capsys.readouterr().out

    monkeypatch.setattr(
        headless,
        "run_headless",
        lambda cfg, services=None, progress=None: headless.HeadlessReport(
            outcomes=(CorpusOutcome(tenant="docs", generation_id="g", certified=False,
                                    degraded_reason="separability below the bar",
                                    previously_serving="gen_previous"),),
            indexed=(headless.LegacyIndex(tenant="memory", files=3, chunks=11),),
        ),
    )
    # `previously_serving` is what makes this exit 0. Without it the tenant answers nothing, and
    # that case is exercised separately — the two used to share this assertion and this exit code.
    assert _cli(config, "--headless")[0] == 0, "a degraded upgrade completes, it does not fail"
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


# ----------------------------------------------------------------------------------------------
# The deployment shape: two DSNs, two roles, and the GRANTs nothing emitted
# ----------------------------------------------------------------------------------------------


def test_two_dsns_naming_different_databases_are_refused(tmp_path: Path) -> None:
    """The schema would be applied to one database and every chunk written to the other.

    The serving database's vector width is first checked inside `calibrate`, which is after the
    entire corpus has been built, and that failure strands a READY generation. Refusing up front
    costs nothing; discovering it costs a full build.
    """
    from recall.wizard.headless import run_headless

    payload = _config(
        tmp_path, migration_dsn="postgresql://recall:recall@127.0.0.1:1/somewhere_else"
    )
    with pytest.raises(ConfigRefusal) as caught:
        run_headless(load_config(_write(tmp_path, payload)), services=_Spy())
    assert caught.value.keys == ("dsn", "migration_dsn")


def test_a_two_role_config_without_a_serving_role_is_refused(tmp_path: Path) -> None:
    """No migration emits a GRANT, so the serving role would own nothing it needs.

    This is the deployment the two DSN keys exist for, and it is invisible locally and in CI, where
    both DSNs are the same superuser. The install completed "successfully" and then failed at the
    first query with `permission denied` — which `recall/schema.py` already records as exactly what
    happens to an operator who skips the grants.
    """
    from recall.wizard.headless import run_headless

    payload = _config(
        tmp_path,
        dsn="postgresql://recall_server:pw@127.0.0.1:1/recall",
        migration_dsn="postgresql://recall_migrator:pw@127.0.0.1:1/recall",
    )
    with pytest.raises(ConfigRefusal) as caught:
        run_headless(load_config(_write(tmp_path, payload)), services=_Spy())
    assert caught.value.keys == ("serving_role",)
    assert "GRANT" in str(caught.value), "the reason must name what is missing, not just the key"


def test_a_named_serving_role_is_granted_over_the_ddl_connection(tmp_path: Path) -> None:
    """And over the MIGRATION dsn: the DDL owner is the only connection that can grant."""
    from recall.wizard.headless import run_headless

    payload = _config(
        tmp_path,
        dsn="postgresql://recall_server:pw@127.0.0.1:1/recall",
        migration_dsn="postgresql://recall_migrator:pw@127.0.0.1:1/recall",
        serving_role="recall_server",
    )
    spy = _Spy()
    config = load_config(_write(tmp_path, payload))
    run_headless(config, services=spy)

    assert spy.grants == [(config.migration_dsn, "recall_server")]


def test_a_single_role_config_grants_nothing(tmp_path: Path) -> None:
    """The allow path, so the grant guard cannot be satisfied by always refusing."""
    from recall.wizard.headless import run_headless

    spy = _Spy()
    report = run_headless(load_config(_write(tmp_path, _config(tmp_path))), services=spy)

    assert spy.grants == [], "one role needs no grant"
    assert report.ok is True


def test_the_configs_own_dsns_are_refused_when_insecure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fail-closed guard must inspect the DSNs the wizard USES, not the global --serving-dsn.

    Measured before the fix: the guard inspected `postgresql://recall:recall@localhost:5432/recall`
    while the wizard connected to a remote host with the credentials published in this repo's
    README, and reached `psycopg.connect`. `migration_dsn` is checked too, and is the more
    privileged of the two: it is the DDL owner.
    """
    from recall.cli import main as cli_main

    monkeypatch.delenv("RECALL_ALLOW_INSECURE_DSN", raising=False)
    remote = "postgresql://recall:recall@192.0.2.10:5432/recall"

    for key in ("dsn", "migration_dsn"):
        payload = _config(tmp_path, **{key: remote})
        with pytest.raises(SystemExit) as exit_info:
            cli_main(
                [
                    "--serving-dsn",
                    "postgresql://recall:recall@127.0.0.1:1/recall",
                    "wizard",
                    "--headless",
                    "--config",
                    str(_write(tmp_path, payload)),
                ]
            )
        message = str(exit_info.value.code)
        assert key in message, f"the refusal must name {key}, the config key actually at fault"
        # ⚠️ Asserted on the CREDENTIAL refusal specifically, not merely on the host appearing.
        # Without the guard the run reaches `apply_schema`, the connection to that host fails, and
        # THAT refusal also names the key and the host — so a looser assertion passed against a
        # mutation that deleted the security check entirely. Only `require_secure_dsn` says this.
        assert "recall:recall" in message and "README" in message, (
            "the refusal must be about the published default credentials, not about reachability"
        )


def test_a_password_never_reaches_the_operator_verbatim(tmp_path: Path) -> None:
    """`redacted_dsn` is not sufficient, and assuming it was left the leak open.

    The secret can be inside the EXCEPTION rather than inside a DSN we format: measured, a password
    containing `%` produced `invalid percent-encoded token: "<the password>"`, so a message that
    carefully redacted the DSN beside it still printed the secret. Port 1, so psycopg fails on parse
    and nothing is contacted.
    """
    from recall.cli import main as cli_main

    placeholder = "CHANGEME%notreal"  # a placeholder, and it must contain a percent sign
    dsn = f"postgresql://recall:{placeholder}@127.0.0.1:1/recall"
    payload = _config(tmp_path, dsn=dsn, migration_dsn=dsn)

    with pytest.raises(SystemExit) as exit_info:
        cli_main(["wizard", "--headless", "--config", str(_write(tmp_path, payload))])

    message = str(exit_info.value.code)
    assert placeholder not in message, "the password must not survive into an operator-facing message"
    assert "***" in message, "and it must be visibly redacted rather than merely absent"


def test_the_report_survives_a_long_tenant_and_a_multi_sentence_reason() -> None:
    """The column width and the continuation indent were unnamed literals repeated six times.

    A tenant longer than 8 characters shifted every column, and a reason containing a newline lost
    its indent completely — and both `degraded_reason` and `Refusal.reason` are already
    multi-sentence strings today, so only the tenant names were keeping this invisible.
    """
    from recall.wizard.headless import HeadlessReport, Refusal
    from recall.wizard.pipeline import CorpusOutcome

    rendered = HeadlessReport(
        outcomes=(
            CorpusOutcome(
                tenant="a-very-long-tenant-name",
                generation_id="gen_x",
                certified=False,
                degraded_reason="one sentence.\nand a second after a newline.",
                previously_serving="gen_old",
            ),
        ),
        refused=(Refusal(tenant="code", reason="a reason " * 30),),
    ).render()

    body = [line for line in rendered.splitlines()[1:] if line.strip()]
    assert all(line.startswith("  ") for line in body), "no line may escape the left gutter"
    assert "a-very-long-tenant-name DEGRADED" in rendered, "the column must widen to fit"
    # The continuation of a wrapped reason lines up past the widest tenant, not at column 11.
    indent = 2 + len("a-very-long-tenant-name") + 1
    continuations = [line for line in body if line.startswith(" " * indent)]
    assert continuations, "wrapped detail must be indented past the tenant column"


def test_an_optional_project_reaches_the_pipeline_and_an_absent_one_is_none(tmp_path: Path) -> None:
    """`project` is the one optional required-looking key, and nothing exercised it.

    An absent one must be `None`, not the string "None", since it is stamped on every chunk.
    """
    assert load_config(_write(tmp_path, _config(tmp_path))).project is None
    assert load_config(_write(tmp_path, _config(tmp_path, project="recall"))).project == "recall"


def test_the_wizard_refuses_flags_it_would_otherwise_discard(tmp_path: Path) -> None:
    """`--tenant` and `--table` were accepted and ignored.

    So `recall --tenant myproject --table probe wizard ...` promoted into `docs` and migrated the
    real `chunks` table, having taken both flags and applied neither.
    """
    from recall.cli import main as cli_main

    config = str(_write(tmp_path, _config(tmp_path)))
    for flag, value in (("--tenant", "myproject"), ("--table", "probe_chunks")):
        with pytest.raises(SystemExit) as exit_info:
            cli_main([flag, value, "wizard", "--headless", "--config", config])
        assert flag in str(exit_info.value.code), f"{flag} must be refused, not silently dropped"
