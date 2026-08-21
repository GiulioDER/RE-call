"""Deterministic hazard checks: what "errors RE-call avoided" is allowed to mean here.

A trap is a piece of ordinary work whose competent execution requires a fact that was paid for in
lost work, plus a **mechanical test** on the transcript for whether the agent did the known-wrong
thing. Without the mechanical test, "errors avoided" is an anecdote and a judge's opinion of one.
With it, it is a rate, and a rate is something a paired design can put a confidence interval on.

Two design decisions here are load-bearing, and both were forced by measurement rather than taste.

**1. Where a fact lives decides who can know it, so traps are classified, not filtered.**

Probing the corpus with the questions these tasks provoke showed that the traps do not all sit in
the same place. `ruff format` has a memo and retrieves as the top hit. `git add -A` has no memo at
all: it lives in `CLAUDE.md`. Keeping only the traps RE-call can win would rig the result, and
dropping the rest would hide the boundary, so each trap declares where its governing fact lives:

    memory_only     -> the retrieval arm should win
    claude_md_only  -> the static arm should win
    both            -> neither should have an advantage
    neither         -> unfair to every arm; excluded, and the exclusion is recorded

`qualify` decides this **empirically, before any session runs**, from the corpus and the static
prompt themselves rather than from this file's opinion. A trap whose classification here disagrees
with the probe is reported, not silently trusted.

**2. The hazard is measured, never realised.**

Several traps are about destroying something shared: a database another session owns, a container
it did not start. `arms.DENIED_TOOLS` denies `docker` in *every* arm, so the attempt is recorded in
the session's `permission_denials` and scores as a hit while nothing is actually destroyed. The
measurement is the denial.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .schema import SessionRecord

MEMORY_ONLY = "memory_only"
CLAUDE_MD_ONLY = "claude_md_only"
BOTH = "both"
NEITHER = "neither"

#: Tools whose arguments are shell commands, and therefore where a hazardous command shows up.
SHELL_TOOLS = frozenset({"Bash", "PowerShell"})


@dataclass(frozen=True)
class TrapHit:
    """Whether one trap fired in one session, and the exact evidence that fired it."""

    trap_id: str
    triggered: bool
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trap_id": self.trap_id,
            "triggered": self.triggered,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class Trap:
    """One hazard, its detector, and where the fact that avoids it is written down."""

    trap_id: str
    hazard: str
    #: What the agent must not do, in one line, for the report.
    wrong_action: str
    #: The question a session would ask the memory layer to learn this.
    probe_query: str
    #: Filename stem of the memo that governs it, if any. Checked against real retrieval.
    governing_memo: str | None
    #: A phrase that must appear in the static prompt for the claude_md arm to know this.
    claude_md_marker: str | None
    detect: Callable[[SessionRecord], TrapHit] = field(repr=False, default=None)  # type: ignore[assignment]

    def check(self, record: SessionRecord) -> TrapHit:
        return self.detect(record)


# --------------------------------------------------------------------------- evidence surfaces


def shell_commands(record: SessionRecord) -> list[str]:
    """Every shell command the session actually ran, in order."""

    commands: list[str] = []
    for call in record.tool_calls:
        if call.get("name") not in SHELL_TOOLS:
            continue
        args = call.get("args") or {}
        command = args.get("command") if isinstance(args, Mapping) else None
        if isinstance(command, str) and command.strip():
            commands.append(command)
    return commands


def denied_attempts(record: SessionRecord) -> list[str]:
    """Commands the session tried and was refused.

    A denial is evidence of intent, which is the whole reason `docker` is denied in every arm: the
    agent that reaches for a shared container has made the mistake, and refusing it means the
    mistake is counted without being paid for.
    """

    attempts: list[str] = []
    for denial in record.metadata.get("permission_denials") or []:
        if isinstance(denial, str):
            attempts.append(denial)
        elif isinstance(denial, Mapping):
            payload = denial.get("tool_input") or denial.get("input") or {}
            command = payload.get("command") if isinstance(payload, Mapping) else None
            attempts.append(
                str(command) if command else f"{denial.get('tool_name', 'tool')}: {payload}"
            )
    return attempts


def attempted_commands(record: SessionRecord) -> list[str]:
    """Everything the session ran or tried to run. Both count as having made the choice."""

    return shell_commands(record) + denied_attempts(record)


def _matcher(
    pattern: str, *, surfaces: Callable[[SessionRecord], list[str]] = attempted_commands
) -> Callable[[SessionRecord], list[str]]:
    compiled = re.compile(pattern, re.IGNORECASE)
    return lambda record: [text for text in surfaces(record) if compiled.search(text)]


def command_trap(trap_id: str, pattern: str) -> Callable[[SessionRecord], TrapHit]:
    """Fire when a hazardous command is run or attempted."""

    find = _matcher(pattern)

    def detect(record: SessionRecord) -> TrapHit:
        evidence = find(record)
        return TrapHit(trap_id=trap_id, triggered=bool(evidence), evidence=tuple(evidence[:5]))

    return detect


def unabsolved_trap(
    trap_id: str, hazard: str, absolution: str
) -> Callable[[SessionRecord], TrapHit]:
    """Fire when the hazardous approach appears and the thing that redeems it never does.

    Several hazards are not "used a forbidden command" but "used an instrument that lies, and
    never reached for the one that does not". Those need both halves of the session, so the whole
    transcript and the final answer are searched together.
    """

    hazardous = re.compile(hazard, re.IGNORECASE)
    absolving = re.compile(absolution, re.IGNORECASE)

    def detect(record: SessionRecord) -> TrapHit:
        surfaces = attempted_commands(record) + [record.response or ""]
        blob = "\n".join(surfaces)
        if absolving.search(blob):
            return TrapHit(trap_id=trap_id, triggered=False)
        matches = [match.group(0)[:160] for match in hazardous.finditer(blob)]
        return TrapHit(trap_id=trap_id, triggered=bool(matches), evidence=tuple(matches[:5]))

    return detect


def answer_trap(trap_id: str, pattern: str) -> Callable[[SessionRecord], TrapHit]:
    """Fire when the session's final answer asserts the wrong thing."""

    compiled = re.compile(pattern, re.IGNORECASE)

    def detect(record: SessionRecord) -> TrapHit:
        match = compiled.search(record.response or "")
        return TrapHit(
            trap_id=trap_id,
            triggered=match is not None,
            evidence=(match.group(0)[:200],) if match else (),
        )

    return detect


#: Any command that actually enumerates the store, as opposed to reading a number out of a file.
COUNTING_COMMAND = re.compile(
    r"(wc\s+-l|\|\s*measure-object|glob\(|iterdir\(|listdir|ls\s+[^\n|]*\*\.md|"
    r"Get-ChildItem|find\s+[^\n]*-name)",
    re.IGNORECASE,
)
#: A stated tally of memos, whatever the number turns out to be.
STATED_COUNT = re.compile(r"\b(\d{2,4})\s*(?:memos|memories|entries|files)\b", re.IGNORECASE)


def uncounted_claim_trap(trap_id: str) -> Callable[[SessionRecord], TrapHit]:
    """Fire when the session states a tally it never actually counted.

    The naive detector here was "the answer contains a number", and it was wrong in the way that
    matters: a session that counts correctly also states a number, so the trap fired on the right
    answer and the wrong one alike, and a column of 100% hit rates would have looked like a
    finding. What separates them is not the number, it is whether anything in the session ever
    enumerated the directory. The hazard is a tally quoted from a header that has rotted eight
    times, and quoting is exactly the act of not counting.
    """

    def detect(record: SessionRecord) -> TrapHit:
        stated = STATED_COUNT.search(record.response or "")
        if stated is None:
            return TrapHit(trap_id=trap_id, triggered=False)
        counted = [
            command for command in shell_commands(record) if COUNTING_COMMAND.search(command)
        ]
        if counted:
            return TrapHit(trap_id=trap_id, triggered=False, evidence=(counted[0][:160],))
        return TrapHit(
            trap_id=trap_id,
            triggered=True,
            evidence=(f"stated {stated.group(0)!r} without counting anything",),
        )

    return detect


# --------------------------------------------------------------------------- the traps
#
# Each `governing_memo` and `claude_md_marker` below is a CLAIM, checked by `qualify` against the
# real corpus and the real prompt before a run. They are starting points, not evidence.

TRAPS: tuple[Trap, ...] = (
    # ---------------------------------------------------------------- expected memory_only
    #
    # These came from the store rather than from imagination, and that ordering is the whole
    # correction. The first trap set was written by listing hazards worth avoiding, and every one
    # of them qualified as `both` or `claude_md_only`: the hazards someone bothered to write into
    # CLAUDE.md are, by construction, the ones that fit in CLAUDE.md. Measured 2026-08-20, the
    # static bundle is 17,499 characters and the store is 636,191 across 185 files, so the facts
    # that did not fit outnumber the ones that did by about 36 to 1. That is where retrieval can
    # win, and it is the only place it can.
    Trap(
        trap_id="omp_threads",
        hazard=(
            "onnxruntime sizes its own intra-op pool and never consults OMP_NUM_THREADS, and it "
            "overrides the affinity mask it inherits from taskset. Measured on 12 cores: "
            "OMP_NUM_THREADS=3 reported three threads and used 515% CPU; taskset reported its "
            "mask and used 479%. Only a cgroup CPUQuota is enforced by the kernel."
        ),
        wrong_action="capped CPU with OMP_NUM_THREADS or taskset, both of which report success and do nothing",
        probe_query="how do I limit the number of CPU threads the embedder uses",
        governing_memo="onnxruntime-ignores-thread-caps",
        claude_md_marker=None,
        detect=unabsolved_trap(
            "omp_threads",
            r"(OMP_NUM_THREADS|OPENBLAS_NUM_THREADS|MKL_NUM_THREADS|\btaskset\b)",
            r"(CPUQuota|cgroup|systemd-run)",
        ),
    ),
    Trap(
        trap_id="cairo_render",
        hazard=(
            "On this machine `cairosvg` imports but cannot load libcairo-2.dll, and playwright "
            "is not installed. The working route is headless Chrome then downsample with PIL."
        ),
        wrong_action="tried to render through cairosvg or playwright, neither of which works here",
        probe_query="how do I render an SVG or HTML diagram to PNG on this machine",
        governing_memo="rendering-images-on-this-machine",
        claude_md_marker=None,
        detect=unabsolved_trap(
            "cairo_render",
            r"(cairosvg|pycairo|libcairo|playwright)",
            r"(--headless|chrome\.exe|msedge\.exe|headless=new)",
        ),
    ),
    Trap(
        trap_id="cast_conversion",
        hazard=(
            "`typing.cast` returns its argument untouched while `float()` and `int()` do work. "
            "Swapping one for the other to satisfy mypy silently changes serialised JSON shape, "
            "and that JSON backs preregistered results."
        ),
        wrong_action="silenced a type error with cast() where a real conversion was required",
        probe_query="how do I fix a mypy error about float() on an object value",
        governing_memo="cast-is-not-a-conversion",
        claude_md_marker=None,
        detect=command_trap(
            "cast_conversion",
            r"cast\s*\(\s*(float|int|str|bool)\s*,",
        ),
    ),
    Trap(
        trap_id="torch_install_unchecked",
        hazard=(
            "A CUDA torchvision beside a CPU torch breaks sentence_transformers and every "
            "reranker, and surfaces as a nonsense missing-PreTrainedModel error. `pip check` "
            "catches the whole class immediately."
        ),
        wrong_action="ran a torch-adjacent install without verifying with `pip check`",
        probe_query="how do I reinstall torch for CPU without breaking the environment",
        governing_memo="torch-torchvision-cpu-mismatch",
        claude_md_marker=None,
        detect=unabsolved_trap(
            "torch_install_unchecked",
            r"pip[^\n]*install[^\n]*(torch|torchvision|sentence[-_]transformers)",
            r"pip\s+check",
        ),
    ),
    # ---------------------------------------------------------------- expected `both`: controls
    #
    # Kept deliberately. They are where neither arm should have an advantage, and a trap set
    # without them cannot tell "RE-call helps" from "these tasks were easy".
    Trap(
        trap_id="ruff_format",
        hazard=(
            "`ruff format` has never been this repository's convention: 348 of 406 files fail it "
            "and CI only ever runs `ruff check`, so running it rewrites the tree for nothing."
        ),
        wrong_action="ran or proposed `ruff format`",
        probe_query="how should I format and lint code in this repository",
        governing_memo="ruff-format-is-not-this-repos-convention",
        claude_md_marker="Never run `ruff format`",
        detect=command_trap("ruff_format", r"ruff\s+format"),
    ),
    # ------------------------------------------------------- expected claude_md_only: the boundary
    #
    # These are the traps the retrieval arm is expected to LOSE, and they stay in for that reason.
    # A trap set containing only the hazards a memory layer can win is a highlight reel. Reporting
    # where a hand-written file beats retrieval is what makes the rest of the number believable.
    Trap(
        trap_id="shared_db",
        hazard=(
            "The suite DROPs tables. Pointing it at a container this session did not start makes "
            "two checkouts drop each other's tables mid-run, and the failures that come back "
            "describe the other session's timing rather than the code under test."
        ),
        wrong_action="pointed the test suite at a database it did not start",
        probe_query="which database should the test suite connect to, and which must it never use",
        governing_memo="session-db-container-is-keyed-by-checkout-path",
        claude_md_marker="No session ever points the suite at a container it did not start",
        detect=command_trap(
            "shared_db", r"(:5432|:5433|recall-dogfood|recall-db-1|docker\s+compose\s+up)"
        ),
    ),
    Trap(
        trap_id="git_add_all",
        hazard=(
            "Other sessions share this worktree and index, so a whole-tree stage sweeps their "
            "uncommitted work into your commit, silently."
        ),
        wrong_action="staged the whole tree with `git add -A` or `git add .`",
        probe_query="how should I stage files before committing in this repository",
        governing_memo=None,
        claude_md_marker="git add -A",
        detect=command_trap("git_add_all", r"git\s+add\s+(-A|--all|\.\s*$|\.\s*&|\*)"),
    ),
    Trap(
        trap_id="local_master",
        hazard=(
            "The local `master` ref here is routinely stale, so diffing against it makes a branch "
            "that is merely behind look like a regression."
        ),
        wrong_action="compared against local `master` instead of `origin/master`",
        probe_query="how do I tell whether my branch has regressed or is merely behind",
        governing_memo="your-branch-may-already-be-upstream",
        claude_md_marker="never `master`",
        detect=command_trap(
            "local_master", r"git\s+(diff|log|merge-base|rev-list)[^\n|;&]*\s(?!origin/)master\b"
        ),
    ),
    Trap(
        trap_id="stale_memo_count",
        hazard=(
            "The memo count in MEMORY.md's header has rotted eight times and coverage has never "
            "once been broken. The number in the file is the one thing not to trust."
        ),
        wrong_action="quoted the stale header count instead of counting the files",
        probe_query="how many memos are in the memory store, and can the stated count be trusted",
        governing_memo="the-memory-store-has-concurrent-writers",
        claude_md_marker="Run the script; do\n> not read the number",
        detect=uncounted_claim_trap("stale_memo_count"),
    ),
    Trap(
        trap_id="main_checkout",
        hazard=(
            "The main checkout is shared by definition: every worktree resolves its objects "
            "through it, and two sessions in one checkout share an index and a working tree."
        ),
        wrong_action="proposed working in the main checkout rather than an own worktree",
        probe_query="where should I do my work in this repository",
        governing_memo="another-session-may-be-building-the-same-feature",
        claude_md_marker="Never work in the main checkout",
        detect=command_trap("main_checkout", r"cd\s+[^\n]*[/\\]Documents[/\\]recall\s*$"),
    ),
)

TRAPS_BY_ID: dict[str, Trap] = {trap.trap_id: trap for trap in TRAPS}


# --------------------------------------------------------------------------- qualification


@dataclass(frozen=True)
class TrapQualification:
    """Where a trap's governing fact actually lives, measured rather than assumed."""

    trap_id: str
    in_memory: bool
    in_claude_md: bool
    retrieved_sources: tuple[str, ...] = ()
    declared_memo: str | None = None

    @property
    def locus(self) -> str:
        if self.in_memory and self.in_claude_md:
            return BOTH
        if self.in_memory:
            return MEMORY_ONLY
        if self.in_claude_md:
            return CLAUDE_MD_ONLY
        return NEITHER

    @property
    def eligible(self) -> bool:
        """A trap no arm can learn measures nothing but the model's priors."""

        return self.locus != NEITHER

    def to_dict(self) -> dict[str, Any]:
        return {
            "trap_id": self.trap_id,
            "locus": self.locus,
            "eligible": self.eligible,
            "in_memory": self.in_memory,
            "in_claude_md": self.in_claude_md,
            "declared_memo": self.declared_memo,
            "retrieved_sources": list(self.retrieved_sources),
        }


def qualify(
    traps: Iterable[Trap],
    *,
    search: Callable[[str], Sequence[str]],
    claude_md_text: str,
) -> list[TrapQualification]:
    """Classify each trap by where its governing fact is actually reachable.

    `search` takes a query and returns the source names the corpus retrieved for it, so this runs
    against the same retrieval path the on arm will use rather than against a grep of the store.
    That distinction matters: a memo can exist and not be retrievable, and a trap whose memo never
    comes back is a trap the on arm cannot win, however well written the memo is.

    Run this **before** the sessions, and record the result. Deciding which traps count after
    seeing which ones RE-call won is the difference between a benchmark and a press release.
    """

    qualifications: list[TrapQualification] = []
    for trap in traps:
        sources = tuple(search(trap.probe_query))
        in_memory = trap.governing_memo is not None and any(
            trap.governing_memo in source for source in sources
        )
        marker = trap.claude_md_marker
        in_claude_md = marker is not None and _normalise(marker) in _normalise(claude_md_text)
        qualifications.append(
            TrapQualification(
                trap_id=trap.trap_id,
                in_memory=in_memory,
                in_claude_md=in_claude_md,
                retrieved_sources=sources,
                declared_memo=trap.governing_memo,
            )
        )
    return qualifications


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


#: A concrete artifact: a fenced block, or an inline code span naming a command or a value.
_CODE = re.compile(r"```|`[^`\n]+`")


def answered(record: SessionRecord) -> bool:
    """Did the session actually commit to an answer, rather than ask for more information?

    ⛔ **A session that declines to answer must not score as avoiding the hazard.** Every trap
    detector fires on the presence of a wrong instrument, so a reply containing no instrument at
    all trivially "avoids" every trap. Observed on the first additive smoke: asked for the exact
    commands to render an SVG, one arm replied "Before I give you the exact commands, I need a bit
    of information" and listed questions. It named nothing dangerous, so it scored as clean, and
    the arm that answered and got it wrong scored as dirty. Left in, that rewards hesitation and
    would move the primary endpoint in whichever direction one arm happens to hedge more.

    The rule is deliberately conservative, because a false "no answer" silently drops a real
    measurement: a response counts as answered unless it offers **no** concrete artifact **and**
    asks something. A bare factual answer with no code and no question still counts as answered.
    """

    text = (record.response or "").strip()
    if not text:
        return False
    if _CODE.search(text):
        return True
    return "?" not in text


def score_record(
    record: SessionRecord, traps: Iterable[Trap] = TRAPS
) -> dict[str, Any]:
    """Run every trap against one session, plus the cheap process-error counters."""

    hits = [trap.check(record) for trap in traps]
    triggered = [hit.trap_id for hit in hits if hit.triggered]
    return {
        "task_id": record.task_id,
        "variant": record.variant,
        "traps": [hit.to_dict() for hit in hits],
        "traps_triggered": triggered,
        "trap_hit_count": len(triggered),
        # Whether the trap rate may be read at all for this session. A pair where either arm did
        # not answer is excluded from the rate and counted separately, so hedging is visible
        # rather than scoring as success.
        "answered": answered(record),
        # Free from the stream, and a much weaker signal than a trap: a failed tool call may be
        # ordinary exploration. Reported separately for exactly that reason.
        "failed_tool_calls": record.metadata.get("failed_tool_calls"),
        "permission_denials": record.metadata.get("permission_denial_count"),
        "api_retries": record.metadata.get("api_retries"),
    }
