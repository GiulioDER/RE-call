"""Three flags that were accepted and then ignored are refused before anything runs (#1438, #1440, #1441).

Invariant: a command refuses a flag it would not apply, naming it, before it opens a store, reads
an input file or embeds anything. The failure each test catches is the silent no-op the issue
records: `generation build --chunker code --overlap 200` building as if no overlap were asked for,
`provenance apply --source-root` on a Lite or PostgreSQL store hashing nothing beneath that root,
and `--table` or `--tenant` reaching `recall calibrate` or `recall lint --semantic`, which embed
into a throwaway table of their own.

Each command's first piece of real work is replaced by a recorder that stops the command, so no
database or model is needed; every refusal test has a positive control proving the recorder is
what that command reaches when the flag is absent or applies.

Red proof, recorded 2026-11-19, red in the intended assertion and green once restored:
- Baseline `origin/master` at 2e546bec1 (before the fix): every
  `::test_*_is_refused*` case failed on its `assert reached == []` (the recorder was reached, so
  the flag passed silently): `GenerationManager` for `--chunker code --overlap 20`,
  `read the cards` for `--source-root` on both stores, `calibrate_from_files` and
  `semantic_lint` for `--table scratch` and `--tenant other`. Every positive control passed.
- Mutations, one at a time, each red in the named test:
  `recall.runtime_route.throwaway_table_refusal`, its `--tenant` clause deleted:
  `::test_calibrate_and_semantic_lint_refuse_table_and_tenant[*-tenant]` failed on
  `assert reached == []`.
  `recall.runtime_route.throwaway_table_refusal`, `table.casefold()` replaced by `table`:
  `::test_calibrate_and_semantic_lint_still_run_on_the_default_table[*-Chunks]` failed on
  `assert reached == [...]`.
  `recall/cli_commands/generation_cmd.py`, the overlap check's `args.chunker == "code"` replaced
  by `True`: `::test_overlap_still_reaches_the_text_chunker` failed on `assert reached == [...]`.
  `recall/cli_commands/provenance_cmd.py`, the `--source-root` check's `not args.sqlite_path`
  replaced by `True`: `::test_source_root_still_reaches_the_sqlite_path` failed on
  `assert reached == [...]`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import recall.cli
import recall.generations
import recall.semantic_lint
import recall.setup
from recall.cli_commands import provenance_cmd
from tests.lite_cli_support import _offline, run_cli  # noqa: F401  # _offline: autouse fixture

# Nothing listens on port 1, and every store is replaced before anything could connect.
DSN = "postgresql://tester:unused@127.0.0.1:1/recall"

_REACHED = "REACHED THE WORK"


@pytest.fixture
def reached(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Each command's first piece of real work records being reached and stops the command."""
    calls: list[str] = []

    def recorder(name: str):  # type: ignore[no-untyped-def]
        def _stop(*args: object, **kwargs: object) -> None:
            calls.append(name)
            raise SystemExit(_REACHED)

        return _stop

    monkeypatch.delenv("RECALL_SOURCE_POLICY_FILE", raising=False)
    monkeypatch.setattr(recall.cli, "require_secure_dsn", lambda dsn: None)
    monkeypatch.setattr(recall.generations, "GenerationManager", recorder("GenerationManager"))
    monkeypatch.setattr(recall.setup, "calibrate_from_files", recorder("calibrate_from_files"))
    monkeypatch.setattr(recall.semantic_lint, "semantic_lint", recorder("semantic_lint"))
    # `provenance apply` reads its claim, then its cards; reaching the cards means no refusal came first.
    monkeypatch.setattr(provenance_cmd, "_read_cards", recorder("read the cards"))
    return calls


def _refused(out: str, flag: str) -> None:
    assert "EXIT" in out and _REACHED not in out and flag in out, out


# --- #1438: `generation build --overlap` with `--chunker code` -------------------------------


def _build(tmp_path: Path, *flags: str) -> str:
    manifest = tmp_path / "man.json"
    manifest.write_text("{}", encoding="utf-8")
    return run_cli(["--dsn", DSN, "--embedder", "hashing", "generation", "build", str(manifest), *flags])


def test_overlap_with_the_code_chunker_is_refused(tmp_path: Path, reached: list[str]) -> None:
    out = _build(tmp_path, "--chunker", "code", "--overlap", "20")
    assert reached == [], f"the build started despite --overlap with --chunker code: {out}"
    _refused(out, "--overlap")


def test_the_code_chunker_without_overlap_still_builds(tmp_path: Path, reached: list[str]) -> None:
    """Positive control: the default overlap is not an explicit one, so the code chunker runs."""
    out = _build(tmp_path, "--chunker", "code")
    assert reached == ["GenerationManager"], out


def test_overlap_still_reaches_the_text_chunker(tmp_path: Path, reached: list[str]) -> None:
    out = _build(tmp_path, "--chunker", "text", "--overlap", "20")
    assert reached == ["GenerationManager"], out


# --- #1440: `provenance apply --source-root` without `--sqlite-path` --------------------------


def _apply(tmp_path: Path, dsn: str, *flags: str) -> str:
    return run_cli([
        "--dsn", dsn, "--embedder", "hashing", "provenance", "apply",
        "--claim", str(tmp_path / "claim.json"), "--cards", str(tmp_path / "cards.json"),
        "--request-id", "r1", "--generation", "g1", *flags,
    ])


@pytest.fixture
def claim_files(tmp_path: Path) -> Path:
    (tmp_path / "claim.json").write_text("{}", encoding="utf-8")
    (tmp_path / "cards.json").write_text("[]", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("store", ["postgresql", "lite"])
def test_source_root_without_sqlite_path_is_refused(
    claim_files: Path, reached: list[str], store: str
) -> None:
    dsn = DSN if store == "postgresql" else "sqlite:///" + (claim_files / "memory.db").as_posix()
    out = _apply(claim_files, dsn, "--source-root", str(claim_files))
    assert reached == [], f"the cards were read despite --source-root on {store}: {out}"
    _refused(out, "--source-root")


def test_source_root_still_reaches_the_sqlite_path(claim_files: Path, reached: list[str]) -> None:
    out = _apply(
        claim_files, DSN,
        "--sqlite-path", str(claim_files / "facts.sqlite"), "--source-root", str(claim_files),
    )
    assert reached == ["read the cards"], out


def test_apply_without_source_root_still_reads_the_cards(claim_files: Path, reached: list[str]) -> None:
    out = _apply(claim_files, DSN)
    assert reached == ["read the cards"], out


# --- #1441: `--table` and `--tenant` on `recall calibrate` and `recall lint --semantic` -------


def _throwaway_command(tmp_path: Path, command: str) -> list[str]:
    if command == "calibrate":
        queries = tmp_path / "queries.json"
        queries.write_text(
            '[{"query": "a", "answerable": true}, {"query": "b", "answerable": false}]',
            encoding="utf-8",
        )
        return ["calibrate", str(queries)]
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    return ["lint", str(corpus), "--semantic", "--threshold", "0.7"]


_WORK = {"calibrate": "calibrate_from_files", "lint --semantic": "semantic_lint"}


@pytest.mark.parametrize("flag", ["table", "tenant"])
@pytest.mark.parametrize("command", sorted(_WORK))
def test_calibrate_and_semantic_lint_refuse_table_and_tenant(
    tmp_path: Path, reached: list[str], command: str, flag: str
) -> None:
    value = "scratch" if flag == "table" else "other"
    out = run_cli([
        "--dsn", DSN, "--embedder", "hashing", f"--{flag}", value,
        *_throwaway_command(tmp_path, command),
    ])
    assert reached == [], f"{command} ran despite --{flag} {value!r}: {out}"
    _refused(out, f"--{flag}")


@pytest.mark.parametrize("table", ["chunks", "Chunks"])
@pytest.mark.parametrize("command", sorted(_WORK))
def test_calibrate_and_semantic_lint_still_run_on_the_default_table(
    tmp_path: Path, reached: list[str], command: str, table: str
) -> None:
    """Positive control: the defaults are not a request, so each command reaches its work.
    `Chunks` is the default table as PostgreSQL folds it, as `generations_table_refusal` accepts."""
    out = run_cli([
        "--dsn", DSN, "--embedder", "hashing", "--table", table, "--tenant", "default",
        *_throwaway_command(tmp_path, command),
    ])
    assert reached == [_WORK[command]], out
