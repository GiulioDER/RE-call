"""The questions `recall wizard` asks when nobody handed it a config file.

The engine has always supported more than one shape of install: an existing PostgreSQL, or a
container this wizard provisions. Only one of those was ever reachable by a person, because the
only way to express the choice was to hand-write JSON, which is exactly what this product exists to
spare its audience. So the options were not missing. **The surface that chooses between them was.**

Three decisions worth stating, because each is where an interactive installer usually goes wrong:

**Every prompt is answered through an injected reader.** Not `input()`. A prompt loop that reads the
real stdin cannot be tested, and an installer whose question flow is untested is one where the
seventh question is wrong forever. `Prompter` is the seam; the tests drive it with a list.

**A non-interactive stdin is refused, loudly, before the first question.** Piping into this, or
running it from a CI job, would otherwise either hang waiting for a line that never comes or read
EOF and take every default. Both look like the installer working. A person is either present to
answer or they should be using `--headless` with a config file, and those are the only two states.

**The answers are written to a config file, and the install runs from that file.** Not from the
answers directly. The interactive path therefore produces the same artefact the headless path
consumes, which means a user can re-run their own install, send it to somebody, or put it in CI,
and it means there is exactly one engine rather than an interactive one that drifts from it.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from recall.wizard.database import probe_database
from recall.wizard.questions import (
    Question,
    build_config,
    default_for,
    question_plan,
    visible_questions,
)

__all__ = [
    "InteractiveRefusal",
    "Prompter",
    "ask_config",
    "stdin_prompter",
]


class InteractiveRefusal(RuntimeError):
    """Raised when the flow cannot sensibly continue. Carries a sentence a user can act on."""


@dataclass
class Prompter:
    """Where questions go and answers come from.

    A dataclass rather than bare callables so a test can hold the transcript, and so the two halves
    cannot be wired up separately by accident.
    """

    read: Callable[[str], str]
    write: Callable[[str], None]

    def ask(self, question: str, *, default: str = "") -> str:
        """One line, with `default` used when the answer is empty.

        The default is shown, always. A prompt that silently applies one is a prompt that lies about
        what the user chose, and this file's whole job is to record what they chose.
        """
        suffix = f" [{default}]" if default else ""
        answer = self.read(f"{question}{suffix}: ").strip()
        return answer or default

    def choose(self, question: str, options: Sequence[tuple[str, str]], *, default: str) -> str:
        """Pick one key from `options`, which are `(key, description)` pairs.

        Re-asks on an unrecognised answer rather than falling back to the default. Silently taking
        the default when someone typed something is how an installer ends up doing the opposite of
        what it was told, and the person never learns their answer was discarded.
        """
        keys = [key for key, _ in options]
        if default not in keys:  # pragma: no cover - a programming error in the caller
            raise ValueError(f"default {default!r} is not one of {keys}")
        self.write(question)
        for key, description in options:
            self.write(f"  {key}) {description}")
        for _ in range(_MAX_RETRIES):
            answer = self.ask("choice", default=default).lower()
            if answer in keys:
                return answer
            self.write(f"  '{answer}' is not one of {', '.join(keys)}.")
        raise InteractiveRefusal(
            f"no valid choice after {_MAX_RETRIES} attempts; nothing was installed"
        )

    def confirm(self, question: str, *, default: bool = True) -> bool:
        shown = "Y/n" if default else "y/N"
        for _ in range(_MAX_RETRIES):
            answer = self.read(f"{question} [{shown}]: ").strip().lower()
            if not answer:
                return default
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no"}:
                return False
            self.write("  answer y or n.")
        raise InteractiveRefusal(f"no valid answer after {_MAX_RETRIES} attempts")


#: Bounded because an unbounded re-ask loop against a stdin that returns EOF forever is a hang, and
#: a hang during an install is the failure mode with no diagnostic at all.
_MAX_RETRIES = 5


def stdin_prompter() -> Prompter:
    """The real one. Refuses immediately when stdin is not a terminal.

    ⚠️ **This check is the reason the flow is safe to add to an existing command.** Without it,
    `recall wizard` in a script either blocks on a line that never arrives or reads EOF and takes
    every default, and both of those look like a successful install from the outside.
    """
    if not sys.stdin.isatty():
        raise InteractiveRefusal(
            "recall wizard needs a terminal to ask questions. This session's input is not one, so "
            "it would either hang or silently accept every default. Use `--headless --config "
            "<file>` for an unattended install; the interactive flow writes exactly that file, so "
            "you can produce one on a machine with a terminal and reuse it here."
        )
    return Prompter(read=input, write=lambda line: print(line))


def ask_config(
    prompter: Prompter,
    *,
    default_root: Path,
    probe: Callable[..., object] = probe_database,
) -> dict[str, object]:
    """Ask what the install needs and return the config document, ready to write.

    Returns a plain dict rather than a `HeadlessConfig` so the caller can WRITE it before running
    anything. The file is the artefact: it is what makes the install repeatable and what a user can
    hand to somebody else.

    `probe` is injected for the same reason `Prompter` is. The database question is the only one
    with a side effect — it opens a connection — and a test of the question flow must not need a
    PostgreSQL to run.

    ⛔ **The questions come from `recall.wizard.questions`, not from this function.** They used to
    live here, and then the graphical installer needed the same set: the same defaults, the same
    branch between a provisioned database and an existing one, the same rule that all three corpus
    roots are mandatory. Re-typing any of that would have made a second installer that drifts, which
    is the exact shape of three separate defects fixed in this repository already. This function is
    now one RENDERER of that plan; `recall/desktop/install_ui.py` is the other.
    """
    prompter.write("recall install\n")

    plan = question_plan(default_root=default_root)
    answers: dict[str, str] = {}

    for question in plan:
        if not _applies(question, answers):
            continue
        if question.help:
            prompter.write(f"\n{question.help}")
        if question.kind == "choice":
            answers[question.key] = prompter.choose(
                f"\n{question.prompt}",
                [(choice.key, choice.label) for choice in question.choices],
                default=default_for(question, answers),
            )
        elif question.kind == "dsn":
            answers[question.key] = _ask_working_dsn(
                prompter, embedder=answers.get("embedder", "fastembed"), probe=probe
            )
        else:
            answers[question.key] = prompter.ask(
                f"\n{question.prompt}", default=default_for(question, answers)
            )

    try:
        return build_config(answers)
    except ValueError as exc:
        # ⛔ **Translated, because `recall/cli.py` handles `InteractiveRefusal` and not `ValueError`.**
        # The data-folder validation added alongside this is reachable by TYPING a relative path,
        # unlike the three corpus-root refusals which are effectively unreachable from a prompt that
        # offers a non-empty default. So the terminal interview ended in a traceback with every
        # answer already given thrown away. The GUI was fine; it catches `ValueError` itself.
        raise InteractiveRefusal(str(exc)) from exc


def _applies(question: Question, answers: dict[str, str]) -> bool:
    """Whether `question` is reachable given the answers so far.

    Delegates to `visible_questions` rather than re-reading `depends_on`, so a change to how a
    dependency is expressed cannot make the terminal and the graphical front end disagree about
    which questions exist. That disagreement is not a cosmetic one: it is asking for a DSN on the
    Docker path, or failing to ask for one on the other.
    """
    return question in visible_questions((question,), answers)


def _ask_working_dsn(
    prompter: Prompter, *, embedder: str, probe: Callable[..., object]
) -> str:
    """Ask for a DSN until one is usable, or give up with the reason attached.

    ⚠️ **The probe runs before the answer is accepted, and a blocking finding re-asks.** Taking a
    DSN on faith is what makes the whole existing-database path a trap: every way it can be wrong
    fails minutes later, during a build, naming something other than the connection string.
    """
    expected = _dimension_for(embedder)
    for _ in range(_MAX_RETRIES):
        dsn = prompter.ask(
            "\nConnection string, for example postgresql://user:password@127.0.0.1:5432/recall"
        )
        if not dsn:
            prompter.write("  a connection string is required for an existing database.")
            continue
        prompter.write("  checking...")
        report = probe(dsn, expected_dimension=expected)
        prompter.write(report.render())  # type: ignore[attr-defined]
        if report.usable:  # type: ignore[attr-defined]
            return dsn
        prompter.write("  that database cannot host this install yet. Fix the above, or try another.")
    raise InteractiveRefusal(
        f"no usable database after {_MAX_RETRIES} attempts; nothing was installed"
    )


def _dimension_for(embedder: str) -> int | None:
    """The embedder's vector width, or None when it cannot be determined without loading it.

    Resolved through the real registry rather than a table here, because a second table of
    dimensions is a second thing to keep in step with the first, and the failure when they disagree
    is the one this whole preflight exists to prevent.
    """
    try:
        from recall.embeddings import resolve_embedder

        return int(resolve_embedder(embedder).dim)
    except Exception:  # noqa: BLE001 - an embedder that cannot be resolved here is one the install
        # will refuse by name later, with a better message than this function could produce. The
        # dimension check simply degrades to "not compared", which the report states explicitly.
        return None


def write_config(document: dict[str, object], path: Path) -> Path:
    """Write the answers where the headless installer can read them back.

    Atomic (this was a plain `write_text`): the file is the input to a later install run, and
    a crash mid-write used to leave truncated JSON the headless wizard would refuse.
    """
    from recall.atomic_write import atomic_write_bytes

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, (json.dumps(document, indent=2) + "\n").encode("utf-8"))
    return path
