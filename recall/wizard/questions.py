"""What the installer needs to know, declared once so every front end asks the same thing.

⛔ **This module exists because a second front end is a second chance to drift.** The terminal
installer in `recall/wizard/interactive.py` already encoded the question set, the defaults, the
branch between a provisioned database and an existing one, and the rule that all three corpus roots
are mandatory. A graphical installer that re-encoded any of that would be a second installer, and
this session has already fixed three separate defects whose whole shape was one rule with two
implementations: the certification gate keyed on the wrong environment while the wizard's pipeline
checked the right one, the desktop upload assembling its own pipeline identity, and
`recall/wizard/projects.py` carrying its own copy of the compose file's name.

So the questions are DATA. `interactive.py` renders them through a `Prompter`; the desktop installer
renders them as widgets. Neither owns them.

**`build_config` is the load-bearing half, not the question list.** Turning answers into the config
document is where the real rules live: which key each answer lands under, that roots are resolved to
absolute paths, that the corpus version is stamped, and that a blank root becomes a directory under
the install rather than an absent key. A front end that asked identical questions and assembled the
document itself would still produce a different artefact, and the failure would be a config the
engine refuses one step later — which is exactly what happened to the terminal flow once, and was
found by round-tripping through `load_config` rather than by any of its ten passing question tests.

**Conditional questions are declared, not branched.** `depends_on` marks the two that belong to one
side of the database choice. A front end that renders every question in the plan and hides the ones
`visible_questions` filters out cannot accidentally ask for a DSN on the Docker path, and cannot
accidentally OMIT one on the other path either.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

__all__ = [
    "Choice",
    "Question",
    "QuestionKind",
    "build_config",
    "default_for",
    "question_plan",
    "visible_questions",
]

#: `directory` and `dsn` are separate kinds from `text` because a front end can offer more than a
#: line edit for them: a folder picker, and a connection test. The terminal treats all three as a
#: line of input, which is the right degradation rather than a missing feature.
QuestionKind = Literal["text", "choice", "directory", "dsn"]


@dataclass(frozen=True)
class Choice:
    """One option of a `choice` question, with the text a person reads to decide."""

    key: str
    label: str


@dataclass(frozen=True)
class Question:
    """One thing the installer needs to know.

    `default` is a callable rather than a string wherever it depends on an earlier answer. The
    corpus roots are the case that forces this: their suggestion sits under `data_root` when the
    wizard provisions a database and under the fallback root when it does not, so a static default
    would be wrong on one of the two paths.
    """

    key: str
    prompt: str
    kind: QuestionKind = "text"
    default: str | Callable[[Mapping[str, str]], str] = ""
    choices: tuple[Choice, ...] = ()
    #: Shown to the user beside the question. Not a comment: it is the only explanation a graphical
    #: front end has room for, and the terminal prints it too.
    help: str = ""
    #: `(key, value)` — this question applies only when a previous answer equals `value`.
    depends_on: tuple[str, str] | None = None
    #: An empty answer is accepted and means "not set". Only `project_root` is optional today.
    optional: bool = False
    #: Extra keys this question's answer is stored under. Empty for all of them; kept because a
    #: front end iterating the plan must not have to know which answers are also config keys.
    metadata: Mapping[str, str] = field(default_factory=dict)


def question_plan(*, default_root: Path) -> tuple[Question, ...]:
    """Every question the installer can ask, in order, including both sides of the branch.

    Returns the WHOLE plan rather than the visible subset, so a front end can build its widgets once
    and show or hide them as the answers change. `visible_questions` does the filtering.
    """

    def _root_default(leaf: str) -> Callable[[Mapping[str, str]], str]:
        def _default(answers: Mapping[str, str]) -> str:
            # ⚠️ The base is the user's chosen `data_root` when there is one. Suggesting a folder
            # under a directory the install does not use would be a confusing default on the
            # Docker path, which is the recommended one and therefore the common one.
            base = answers.get("data_root") or str(default_root)
            return str(Path(base).expanduser() / leaf)

        return _default

    return (
        Question(
            key="project",
            prompt="Project name",
            default=_DEFAULT_PROJECT,
            help="Names the tenants and is stamped on every chunk.",
        ),
        Question(
            key="embedder",
            prompt="Which embedder?",
            kind="choice",
            default="fastembed",
            choices=(
                Choice(
                    "fastembed",
                    "local, 384 dimensions, downloads a small model once (recommended)",
                ),
                Choice("hashing", "fully offline, no download, lower quality"),
            ),
        ),
        Question(
            key="database",
            prompt="Where should the database live?",
            kind="choice",
            default="docker",
            choices=(
                Choice(
                    "docker",
                    "create one for me in Docker (recommended if you have no PostgreSQL)",
                ),
                Choice("existing", "I already have a PostgreSQL with pgvector"),
            ),
        ),
        Question(
            key="data_root",
            prompt="Where should recall keep its data?",
            kind="directory",
            default=str(default_root),
            depends_on=("database", "docker"),
        ),
        Question(
            key="dsn",
            prompt="Connection string",
            kind="dsn",
            help="For example postgresql://user:password@127.0.0.1:5432/recall",
            depends_on=("database", "existing"),
        ),
        Question(
            key="docs_root",
            prompt="Documents folder (notes, markdown, PDFs)",
            kind="directory",
            default=_root_default("docs"),
            help="An empty folder is fine and can be filled later.",
        ),
        Question(
            key="code_root",
            prompt="Code folder",
            kind="directory",
            default=_root_default("code"),
            help="An empty folder is fine and can be filled later.",
        ),
        Question(
            key="memory_root",
            prompt="Agent memory folder",
            kind="directory",
            default=_root_default("memory"),
            help="An empty folder is fine and can be filled later.",
        ),
        Question(
            key="project_root",
            prompt="Register the MCP servers for which project?",
            kind="directory",
            optional=True,
            help="The folder of a project you use with Claude Code. Leave blank to skip wiring.",
        ),
    )


def visible_questions(
    plan: Sequence[Question], answers: Mapping[str, str]
) -> tuple[Question, ...]:
    """The subset of `plan` that applies given the answers so far.

    A question whose `depends_on` key has not been answered yet is HIDDEN rather than shown. The
    alternative is showing both sides of a branch until the branch is answered, which in a terminal
    means asking for a DSN before asking whether there is a database to connect to.
    """
    return tuple(
        question
        for question in plan
        if question.depends_on is None
        or answers.get(question.depends_on[0]) == question.depends_on[1]
    )


def default_for(question: Question, answers: Mapping[str, str]) -> str:
    """The default to offer, resolving a callable against the answers given so far."""
    if callable(question.default):
        return question.default(answers)
    return question.default


def build_config(
    answers: Mapping[str, str], *, today: Callable[[], str] = lambda: date.today().isoformat()
) -> dict[str, object]:
    """Turn answers into the config document the headless installer reads.

    ⛔ **This is the shared piece that matters.** Two front ends asking identical questions and each
    assembling their own document would still produce different artefacts, and the failure arrives
    one step later as a validation error about JSON keys — an installer handing the user a complaint
    about its own output. The terminal flow shipped exactly that bug once: all ten of its question
    tests passed, and the config it wrote would not load, because `docs_root`, `code_root` and
    `memory_root` are mandatory and a blank answer omitted them.

    Roots are resolved to absolute paths here rather than at the prompt, because `load_config`
    requires absolute and a front end that forgot would produce a config that fails to load. Note
    that `data_root` is deliberately NOT resolved, only expanded: it is the user's chosen install
    location and `resolve()` would follow a symlink they chose on purpose.

    `today` is injected so a test can pin the corpus version. A stamped date that changes between
    the assertion and the run is a flaky test, and worse, a value nobody can reproduce.
    """
    config: dict[str, object] = {
        "project": answers.get("project") or _DEFAULT_PROJECT,
        "embedder": answers.get("embedder") or "fastembed",
        "corpus_version": today(),
    }

    if answers.get("database") == "existing":
        dsn = (answers.get("dsn") or "").strip()
        if not dsn:
            raise ValueError(
                "an existing database needs a connection string; nothing else identifies it"
            )
        config["dsn"] = dsn
    else:
        config["data_root"] = str(Path(answers.get("data_root") or "").expanduser())

    for key in ("docs_root", "code_root", "memory_root"):
        value = (answers.get(key) or "").strip()
        if not value:
            raise ValueError(
                f"{key} is required: `load_config` names all three corpus roots as mandatory, so a "
                "config without one fails to load rather than defaulting. An EMPTY folder is the "
                "way to say 'nothing here yet'."
            )
        config[key] = str(Path(value).expanduser().resolve())

    project_root = (answers.get("project_root") or "").strip()
    if project_root:
        config["project_root"] = str(Path(project_root).expanduser().resolve())

    return config


def _default_project() -> str:
    """Imported lazily so this module does not pull the corpus layer in at import time."""
    from recall.wizard.corpora import DEFAULT_PROJECT

    return str(DEFAULT_PROJECT)


#: Read once at import. `corpora` is a light module and the value is a constant, but the indirection
#: above keeps the import in one place if that stops being true.
_DEFAULT_PROJECT = _default_project()
