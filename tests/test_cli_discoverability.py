"""`recall --help` has to answer "which one do I want", not only "what exists".

Four commands in this CLI can each be read as "the install" (`quickstart`, `setup`, `wizard`, and
the `recall-install` window that is not even in this parser), and the list of twenty subcommands
with one line each does not distinguish them. A newcomer's question is never "what exists".

The specific failure this was written after: `recall wizard --help` printed a usage line and its
flags and nothing else, so the command most likely to be reached by somebody unsure whether they
wanted `setup` was the one that explained itself least.
"""

from __future__ import annotations

import argparse

import pytest

from recall.cli import build_parser


def _subparsers() -> dict[str, argparse.ArgumentParser]:
    actions = [
        a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)
    ]
    assert len(actions) == 1, "the CLI grew a second subparser group; this test assumes one"
    return dict(actions[0].choices)


def test_the_top_level_help_says_which_command_to_start_with() -> None:
    """An epilogue naming the ordering, not just the inventory.

    Asserted on content rather than on presence: an epilogue that existed and did not name the
    canonical install would satisfy `epilog is not None` while leaving the reader exactly where
    they were.
    """
    text = build_parser().format_help()
    for command in ("quickstart", "setup", "doctor", "wizard"):
        assert f"recall {command}" in text, f"{command} is not placed for a newcomer"
    assert "THE install" in text, "nothing in the help says which of the four to run"


#: Commands somebody reaches BEFORE they have a working install, plus the two they reach
#: immediately after. Every one of these must explain itself, because the reader who types
#: `recall <cmd> --help` here has no other source: they do not yet have a working system to
#: experiment against.
INSTALL_PATH = frozenset(
    {"quickstart", "setup", "wizard", "doctor", "uninstall", "index", "search", "demo", "code",
     "forget"}
)

#: ⚠️ **A ratchet, not an exemption, and the difference is the point.** These eleven had no
#: `description` when this test was written (measured 2026-08-25: 11 of 21 subcommands), and every
#: one of them operates an install that already exists, so none is on the path a new user walks.
#: Writing them is worth doing and was out of scope for the change that added this file.
#:
#: What the test below enforces is that this set never GROWS. Listing them by name rather than
#: asserting a count means a new undocumented subcommand fails immediately instead of being
#: absorbed into a number, and documenting one of these fails too, which is the correct nuisance:
#: it costs one line to delete a name, and it stops the list from quietly becoming permanent.
UNDOCUMENTED = frozenset(
    {"calibrate", "calibration", "check", "extract", "generation", "graph", "lint", "manifest",
     "reasoning", "rewrite", "schema"}
)


@pytest.mark.parametrize("name", sorted(INSTALL_PATH))
def test_every_install_path_subcommand_explains_itself_in_its_own_help(name: str) -> None:
    """`help=` shows in `recall --help`; `description=` is what `recall <cmd> --help` shows.

    They are different strings and only the first is easy to remember to write, so a subcommand can
    look documented from the top level and be blank where somebody actually goes looking. That is
    what `recall wizard` was.

    The length floor is there because a description that merely restates the one-line help has
    satisfied the check without adding anything for the person who typed the longer command
    precisely because the short line was not enough.
    """
    parsers = _subparsers()
    assert name in parsers, f"{name} is named as install-path but no longer exists"
    description = (parsers[name].description or "").strip()
    assert description, f"`recall {name} --help` explains nothing"
    assert len(description) > 60, f"`recall {name}`'s description is too thin to be worth reading"


def test_the_undocumented_list_never_grows() -> None:
    """The ratchet. A new subcommand with no description fails here rather than joining a count."""
    parsers = _subparsers()
    bare = {name for name, p in parsers.items() if not (p.description or "").strip()}

    new = sorted(bare - UNDOCUMENTED)
    assert not new, (
        f"these subcommands have no `description=`, so `recall <cmd> --help` explains nothing: "
        f"{new}. Add one, or if it is genuinely not worth documenting say so in UNDOCUMENTED."
    )

    fixed = sorted(UNDOCUMENTED - bare)
    assert not fixed, (
        f"{fixed} now have descriptions; remove them from UNDOCUMENTED so the list stays an "
        "accurate account of what is missing rather than a stale one"
    )

    gone = sorted(UNDOCUMENTED - set(parsers))
    assert not gone, f"{gone} are named in UNDOCUMENTED but are not subcommands any more"


def test_setup_and_wizard_each_say_how_they_differ_from_the_other() -> None:
    """⚠️ The specific confusion, pinned by name.

    These two are the pair that cannot be told apart from a one-line help, because both are true
    descriptions of "install recall". Whichever one a reader lands on first has to mention the
    other, or the reader has no way to learn that a choice was even being made.
    """
    parsers = _subparsers()
    assert "wizard" in (parsers["setup"].description or "").lower() or "recall setup" in (
        parsers["wizard"].description or ""
    )
    assert "recall setup" in (parsers["wizard"].description or ""), (
        "`recall wizard --help` must name `recall setup`, since a reader who wanted the "
        "one-off interactive install lands here by guessing"
    )
