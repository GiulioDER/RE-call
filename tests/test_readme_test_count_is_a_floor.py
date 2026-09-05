"""The README should describe quality gates without using a stale test-count badge."""

from __future__ import annotations

import re

from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def test_the_readme_does_not_advertise_a_numeric_test_count() -> None:
    text = README.read_text(encoding="utf-8")
    assert "tests-1300" not in text
    assert "1,300+ tests" not in text


def test_the_schema_migrations_claim_matches_the_readme_body() -> None:
    text = README.read_text(encoding="utf-8")
    assert "no versioned upgrade path" not in text
    assert "ordered SQL migration path" in text
    assert "pre-tenancy tables are migrated in place" in text


def test_the_readme_has_a_clear_quickstart_and_surface_split() -> None:
    text = README.read_text(encoding="utf-8")
    assert "## Quickstart" in text
    assert "## Product surface" in text
    assert "recall setup" in text
    assert "When the wizard asks whether to calibrate" in text
    # Matched case-insensitively on the instruction rather than on one sentence's opening words.
    # The quickstart now leads with the database, because its first command needed a compose file
    # a pip-only reader did not have, so the wizard sentence moved rather than went away.
    assert "run the guided setup wizard" in text.lower()
    assert "Declared supersession makes the current memory win" in text


def test_the_install_path_does_not_change_invocation_style_halfway() -> None:
    """One spelling of the command, from the first section to the last.

    The README used to open with `recall quickstart` and then switch to
    `python -m recall.cli setup` four sections later. Both work, and that is the problem: a reader
    following the page top to bottom has no way to tell whether the change of style carries meaning
    (a different entry point? a Windows workaround?) or is an accident of two sections being
    written on different days. It was the second.

    The console script is the one that survives, because the very first command on the page is
    already `recall quickstart`: if `recall` does not resolve, the reader never reaches the install
    section at all, so `python -m` buys nothing there that it has not already failed to buy above.

    Scoped to FENCED CODE BLOCKS, not the whole file, and that distinction is the test rather than
    an implementation detail of it. Prose is where the alternative spelling has to be explained
    ("`python -m recall.cli` is the same program under a longer name"), and a check that could not
    tell that sentence from a command would forbid the explanation the reader needs. What is being
    pinned is the set of lines somebody COPIES.
    """
    text = README.read_text(encoding="utf-8")
    blocks = re.findall(r"^```[a-z]*\n(.*?)^```", text, re.S | re.M)
    assert blocks, "the README has no fenced code blocks at all"
    commands = "\n".join(blocks)
    console_script = re.search(r"(?m)^(?:\S+=\S+ )*recall(?:-\S+)? ", commands) is not None
    module_form = "python -m recall.cli " in commands
    assert console_script or module_form, "no code block names a runnable recall command"
    assert not (console_script and module_form), (
        "the README's code blocks mix `recall <cmd>` and `python -m recall.cli <cmd>`; pick one "
        "so the reader is not asked to guess whether the difference means something"
    )


def test_the_readme_says_its_numbers_are_claim_gated() -> None:
    text = README.read_text(encoding="utf-8")
    assert "tied to committed artifacts" in text
    assert "claim gate checks them in CI" in text


def test_the_readme_names_apache_and_the_citation_path() -> None:
    text = README.read_text(encoding="utf-8")
    assert "Apache 2.0 license" in text
    assert "## Citation" in text
    assert "NOTICE" in text


def test_the_readme_names_the_actual_gate_shapes() -> None:
    text = README.read_text(encoding="utf-8")
    assert "Real pgvector integration tests" in text
    assert "type checking" in text
    assert "dependency audit" in text
