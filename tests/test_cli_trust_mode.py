"""The CLI is strict by default, and development mode has to be asked for by name.

`tests/test_cli.py` sets `RECALL_TRUST_MODE=development` in an autouse fixture so it can reach
the index and forget behaviour it actually tests. That fixture would hide a regression in the
default, so the default is pinned here instead, in a file with no such fixture.
"""

from __future__ import annotations

import pytest

from recall.cli import main
from recall.trust_policy import TrustFailureCode, TrustMode, TrustPolicy
from tests.conftest import TEST_DSN, requires_db


class TestPolicyResolution:
    def test_unset_is_strict(self) -> None:
        assert TrustPolicy.from_env({}).mode is TrustMode.STRICT

    def test_exact_development_string_opts_in(self) -> None:
        assert TrustPolicy.from_env({"RECALL_TRUST_MODE": "development"}).mode is (
            TrustMode.DEVELOPMENT
        )

    @pytest.mark.parametrize("value", ["dev", "developement", "1", "true", "", "strict"])
    def test_anything_else_stays_strict(self, value: str) -> None:
        """A typo must not be the thing that opens the gate."""
        assert TrustPolicy.from_env({"RECALL_TRUST_MODE": value}).mode is TrustMode.STRICT

    def test_case_and_whitespace_are_tolerated_for_the_exact_word(self) -> None:
        assert TrustPolicy.from_env({"RECALL_TRUST_MODE": " Development "}).mode is (
            TrustMode.DEVELOPMENT
        )


@requires_db
def test_cli_search_refuses_by_default_on_an_uncalibrated_table(tmp_path, cli_table, monkeypatch):
    """No env var: the CLI must refuse rather than score against the 0.50 floor."""
    monkeypatch.delenv("RECALL_TRUST_MODE", raising=False)
    (tmp_path / "note.md").write_text("the caching layer decision was adopted", encoding="utf-8")
    main(["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table,
          "index", str(tmp_path)])

    from recall.trust_policy import TrustRefusal

    with pytest.raises(TrustRefusal) as excinfo:
        main(["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table,
              "search", "caching layer"])

    assert excinfo.value.code in {
        TrustFailureCode.INDEX_NOT_READY,
        TrustFailureCode.CALIBRATION_MISSING,
    }
    # The refusal names a remedy rather than merely failing.
    assert "no trustworthy decision" in excinfo.value.advice.lower()


@requires_db
def test_cli_search_degrades_when_development_is_requested(tmp_path, capsys, cli_table,
                                                           monkeypatch):
    monkeypatch.setenv("RECALL_TRUST_MODE", "development")
    (tmp_path / "note.md").write_text("the caching layer decision was adopted", encoding="utf-8")
    main(["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table,
          "index", str(tmp_path)])
    capsys.readouterr()

    main(["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table,
          "search", "caching layer"])

    assert capsys.readouterr().out, "development mode should still produce output"
