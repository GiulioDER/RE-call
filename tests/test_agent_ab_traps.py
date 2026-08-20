"""Tests for the deterministic hazard checks.

Every detector is tested against both a session that fell into the trap and one that did not,
because a detector that never fires and a detector that always fires both produce a clean-looking
column of numbers.
"""

from __future__ import annotations

import pytest

from benchmarks.agent_ab.schema import RECALL_OFF, RECALL_ON, SessionRecord
from benchmarks.agent_ab.traps import (
    BOTH,
    CLAUDE_MD_ONLY,
    MEMORY_ONLY,
    NEITHER,
    TRAPS,
    TRAPS_BY_ID,
    Trap,
    attempted_commands,
    command_trap,
    denied_attempts,
    qualify,
    score_record,
    unabsolved_trap,
)


def record(
    *commands: str,
    response: str = "",
    denials: list[dict] | None = None,
    variant: str = RECALL_OFF,
) -> SessionRecord:
    return SessionRecord(
        task_id="t",
        variant=variant,
        success=True,
        response=response,
        tool_calls=tuple(
            {"name": "Bash", "args": {"command": command}} for command in commands
        ),
        metadata={"permission_denials": denials or []},
    )


# --------------------------------------------------------------------------- evidence surfaces


def test_a_denied_command_counts_as_having_made_the_choice() -> None:
    """Denying `docker` is how a hazard about a shared container is measured, not realised."""
    session = record(
        "pytest -q",
        denials=[{"tool_name": "Bash", "tool_input": {"command": "docker rm recall-dogfood"}}],
    )
    assert denied_attempts(session) == ["docker rm recall-dogfood"]
    assert "docker rm recall-dogfood" in attempted_commands(session)
    assert "pytest -q" in attempted_commands(session)


def test_a_denial_in_an_unexpected_shape_is_still_recorded() -> None:
    session = record(denials=["docker compose up"])
    assert denied_attempts(session) == ["docker compose up"]


def test_non_shell_tool_calls_are_not_mistaken_for_commands() -> None:
    session = SessionRecord(
        task_id="t",
        variant=RECALL_OFF,
        success=True,
        tool_calls=({"name": "Read", "args": {"file_path": "ruff format is fine"}},),
    )
    assert attempted_commands(session) == []


# --------------------------------------------------------------------------- detector shapes


def test_command_trap_fires_only_on_the_hazardous_command() -> None:
    detect = command_trap("x", r"ruff\s+format")
    assert detect(record("python -m ruff format .")).triggered
    assert detect(record("python -m ruff format .")).evidence == ("python -m ruff format .",)
    assert not detect(record("python -m ruff check .")).triggered


def test_unabsolved_trap_clears_a_session_that_reached_for_the_right_instrument() -> None:
    detect = unabsolved_trap("x", r"OMP_NUM_THREADS", r"CPUQuota")
    fell_in = record("OMP_NUM_THREADS=3 python -m recall index")
    recovered = record(
        "OMP_NUM_THREADS=3 python -m recall index",
        "systemd-run --scope -p CPUQuota=250% -- python -m recall index",
    )
    assert detect(fell_in).triggered
    assert not detect(recovered).triggered, "absolution must clear the whole session"


def test_unabsolved_trap_reads_the_final_answer_too() -> None:
    # A session that never ran the wrong command but recommended it has still made the error.
    detect = unabsolved_trap("x", r"cairosvg", r"--headless")
    assert detect(record(response="Install cairosvg and convert the SVG.")).triggered
    assert not detect(
        record(response="cairosvg cannot load its DLL here; use chrome --headless=new.")
    ).triggered


# --------------------------------------------------------------------------- the real traps


@pytest.mark.parametrize(
    ("trap_id", "falls_in", "avoids"),
    [
        (
            "omp_threads",
            record("OMP_NUM_THREADS=4 python -m recall index"),
            record("systemd-run --user --scope -p CPUQuota=250% -- python -m recall index"),
        ),
        (
            "cairo_render",
            record("pip install cairosvg"),
            record("chrome --headless=new --screenshot=/abs/out.png file:///abs/page.html"),
        ),
        (
            "cast_conversion",
            record('python -c "x = cast(float, row[\'elapsed_ms\'])"'),
            record('python -c "x = float(row[\'elapsed_ms\'])"'),
        ),
        (
            "torch_install_unchecked",
            record("pip install --force-reinstall torch"),
            record("pip install --force-reinstall torch", "pip check"),
        ),
        (
            "ruff_format",
            record("python -m ruff format ."),
            record("python -m ruff check ."),
        ),
        (
            "git_add_all",
            record("git add -A"),
            record("git add benchmarks/agent_ab/traps.py"),
        ),
        (
            "shared_db",
            record("RECALL_TEST_DSN=postgresql://recall@127.0.0.1:5432/recall pytest -q"),
            record("eval \"$(scripts/session-db.sh up)\"", "pytest tests/ -q"),
        ),
        (
            "local_master",
            record("git diff master --stat"),
            record("git diff origin/master --stat"),
        ),
        (
            "main_checkout",
            record("cd /c/Users/someone/Documents/recall"),
            record("scripts/session-space.sh new my-work"),
        ),
    ],
)
def test_each_trap_separates_the_mistake_from_the_correct_action(
    trap_id: str, falls_in: SessionRecord, avoids: SessionRecord
) -> None:
    trap = TRAPS_BY_ID[trap_id]
    assert trap.check(falls_in).triggered, f"{trap_id} missed the mistake"
    assert not trap.check(avoids).triggered, f"{trap_id} fired on the correct action"


def test_the_stale_count_trap_separates_quoting_from_counting() -> None:
    """The number is not the signal, because a correct session states one too.

    The first version of this detector fired on any stated tally, so it hit the right answer and
    the wrong one alike. A trap that always fires produces a clean 100% column that reads as a
    finding.
    """
    trap = TRAPS_BY_ID["stale_memo_count"]
    quoted = trap.check(record(response="The store holds 181 memos."))
    assert quoted.triggered
    assert "without counting" in quoted.evidence[0]

    counted = trap.check(
        record("ls memory/*.md | wc -l", response="Counted from disk: 181 memos.")
    )
    assert not counted.triggered, "a session that actually counted must not score as a hit"

    # Even the same number is fine when it was measured rather than quoted.
    powershell_counted = SessionRecord(
        task_id="t",
        variant=RECALL_OFF,
        success=True,
        response="181 memos.",
        tool_calls=(
            {"name": "PowerShell", "args": {"command": "Get-ChildItem *.md | Measure-Object"}},
        ),
        metadata={},
    )
    assert not trap.check(powershell_counted).triggered

    # And no claim at all is not a hit.
    assert not trap.check(record(response="I could not determine the count.")).triggered


def test_every_trap_has_a_probe_query_and_a_detector() -> None:
    for trap in TRAPS:
        assert trap.probe_query.strip(), trap.trap_id
        assert trap.detect is not None, trap.trap_id
        assert trap.governing_memo or trap.claude_md_marker, (
            f"{trap.trap_id} claims no source at all, so it can never qualify"
        )


# --------------------------------------------------------------------------- qualification


def _trap(trap_id: str, memo: str | None, marker: str | None) -> Trap:
    return Trap(
        trap_id=trap_id,
        hazard="h",
        wrong_action="w",
        probe_query=f"q-{trap_id}",
        governing_memo=memo,
        claude_md_marker=marker,
        detect=command_trap(trap_id, "never-matches-anything"),
    )


def test_qualify_classifies_by_where_the_fact_actually_is() -> None:
    traps = [
        _trap("mem", "the-memo", None),
        _trap("doc", None, "written in the file"),
        _trap("both", "the-memo", "written in the file"),
        _trap("none", "absent-memo", "not in the file"),
    ]
    search = {
        "q-mem": ["the-memo.md"],
        "q-doc": ["unrelated.md"],
        "q-both": ["the-memo.md"],
        "q-none": ["unrelated.md"],
    }
    results = {
        q.trap_id: q
        for q in qualify(
            traps,
            search=lambda query: search[query],
            claude_md_text="something Written In The File here",
        )
    }
    assert results["mem"].locus == MEMORY_ONLY
    assert results["doc"].locus == CLAUDE_MD_ONLY
    assert results["both"].locus == BOTH
    assert results["none"].locus == NEITHER
    assert not results["none"].eligible, "a trap no arm can learn measures only model priors"


def test_a_memo_that_exists_but_never_retrieves_does_not_count_as_memory() -> None:
    """Existence is not availability. The on arm gets what comes back, not what is on disk."""
    [qualification] = qualify(
        [_trap("t", "unretrievable-memo", None)],
        search=lambda _: ["something-else.md"],
        claude_md_text="",
    )
    assert not qualification.in_memory
    assert qualification.locus == NEITHER


# --------------------------------------------------------------------------- scoring


def test_score_record_reports_hits_and_the_weaker_process_counters_separately() -> None:
    session = SessionRecord(
        task_id="t",
        variant=RECALL_ON,
        success=True,
        tool_calls=({"name": "Bash", "args": {"command": "python -m ruff format ."}},),
        metadata={"failed_tool_calls": 2, "permission_denial_count": 1, "api_retries": 0},
    )
    score = score_record(session)
    assert score["traps_triggered"] == ["ruff_format"]
    assert score["trap_hit_count"] == 1
    # Deliberately not folded into the trap count: a failed tool call is often just exploration.
    assert score["failed_tool_calls"] == 2
    assert score["permission_denials"] == 1
