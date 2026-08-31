"""`UserPromptSubmit`: search this project's memory with the user's own words, and inject it.

**Why this event, when `write_time` already exists.** The write-time hook fires on `Write`, `Edit`
and `Bash`, which is *after* the decision has been made. The complaint this hook answers is about
the decision itself: proposing an approach that was settled weeks ago, re-running an experiment
that already has a committed result, rebuilding something that exists. By the time a tool call
carries a draft, the wrong plan is already written.

`UserPromptSubmit` is the only event that carries the user's own words *and* precedes every
proposal in the turn. The `SessionStart` docstring in this package has said since 2026-08-19 that
per-turn retrieval belongs here; this is that.

## What it targets, and what it does NOT

⚠️ **This does not replace the write-time hook and cannot.** Measured in this project
(`search-with-the-draft-not-the-goal`), a *goal-shaped* query surfaces the governing hazard memo
for 1 of 14 sessions where the *draft text* surfaces it for 11 of 11. A user prompt is goal-shaped
by construction, so the hazard case stays with `write_time`.

What a goal-shaped query is good at is the *topical* case, which is the one being complained about:
"has this project already decided something about X", where X is named in the prompt. Lexical
overlap between a prompt and a memo written about the same subject is high, and the memo's
`description` frontmatter is a one-line summary written for exactly this purpose.

🔑 **State that separation when reporting on this hook.** Its benefit is unmeasured, and it is a
different mechanism from the one the write-time A/B measured. Nothing here should be quoted as
evidence for it.

## No database, no network, no embedder, and that is a measurement rather than a preference

Measured on this workstation, 2026-08-31, which is why this reads files rather than the corpus:

| path to the corpus | measured |
|---|---|
| the `dsn` in `recall-hook.json` (`127.0.0.1:55432`) | **refused**, 2.04s to fail |
| `ssh vps2 true`, three runs | **2475 / 3227 / 3455 ms** (two hops, via a jump host) |
| the project's memory store on local disk | **~280 ms** to read and index 329 memos |

The consequence, found while writing this: `write_time.enabled` is `true` in the config and the
hook has been a **silent no-op**, because every call fails to connect, starts the 5-minute
cooldown, and returns nothing. The cooldown stamp was live and in the future when this was written.
See `the-write-time-hook-has-no-corpus-on-this-workstation`, whose conclusion still holds.

So this hook reads the memo files directly. They are the same content the `memory` tenant is built
from, minus chunking and embeddings, and they are on the disk the hook already runs on.

## What it costs, per user turn

Medians of five, same machine and day, against the 329-memo store this was developed on:

| | wall clock |
|---|---|
| bare interpreter, the floor any hook pays | **305 ms** |
| plus importing this module | 312 ms |
| plus a prompt too short to rank ("ok") | 320 ms |
| plus reading and ranking the whole store | **754 ms** |

🔑 **The marginal cost of the feature is therefore about 430 ms, once per user turn**, against the
write-time hook's ~2s once per TOOL CALL against a remote corpus. In-process the split is 427 ms to
read and tokenize 329 files and 67 ms to rank them, so the cost is I/O and tokenizing, not the
ranker.

⚠️ **Deliberately not cached.** An inverted index on disk would cut the 427 ms, and it would buy a
staleness bug in exchange: the memo written by the session that is running right now is exactly the
one the next prompt should find. A cache invalidated wrongly here fails silently and looks like
"nothing matched", which is the failure this hook exists to remove.

⛔ **Three properties this must never lose:**

1. **It never blocks a prompt.** Exit 2 on `UserPromptSubmit` discards the user's message. A memory
   layer that can swallow what the user typed is worse than no memory layer. Every path returns 0.
2. **It never raises.** A hook that raises is charged to the client.
3. **It stays ASCII on the wire.** `json.dumps` defaults to `ensure_ascii=True` and that default is
   load-bearing here: the prototype of this scorer crashed with `UnicodeEncodeError` printing a
   memo description through a cp1252 stdout on this machine. Memos are full of `⛔` and `🔑`.
"""

from __future__ import annotations

import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import claude_config_home, load_config

#: The four hub and sub-index files. They are pointers to memos rather than memos, they repeat
#: every memo's hook line, and in the first prototype they out-scored the memos they point at.
INDEX_STEMS = frozenset({"MEMORY", "feedback_index", "project_index", "reference_index"})

#: How many memos to name. Three one-line descriptions is a few hundred tokens; the cost of a
#: wrong hit is one line the reader dismisses, and the cost of a miss is the whole complaint this
#: hook exists for. The asymmetry is why this fires generously rather than precisely.
TOP_K = 3
#: Below this a prompt is "ok", "yes", "go on": no query, and nothing worth reading 329 files for.
MIN_PROMPT_CHARS = 20
#: Distinct content tokens a prompt needs before it is worth ranking at all.
MIN_QUERY_TOKENS = 3
#: A hit must match at least this many distinct query tokens. One shared word is a coincidence.
MIN_MATCHED_TOKENS = 2
#: And it must score at least this fraction of the top hit, so a single strong memo is not padded
#: out to three with whatever came next.
RELATIVE_FLOOR = 0.45
#: Characters of each memo's one-line summary.
SUMMARY_CHARS = 220
#: Ranking never reads more of a prompt than this. A pasted stack trace is not a better query.
MAX_QUERY_CHARS = 4096

BM25_K1 = 1.2
BM25_B = 0.75
#: Title and description are authored summaries; the body is prose around them. Repeating them
#: this many times is the whole of the field weighting, which keeps the ranker one loop.
FIELD_WEIGHT = 3

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
DESCRIPTION_RE = re.compile(r"^description:\s*(.+)$", re.MULTILINE)

#: Function words plus the conversational filler a prompt is made of. Not a linguistic stoplist:
#: every word here was watched matching a memo for no reason during development.
STOPWORDS = frozenset("""
the and for that with this from have has had are was were will would should could into your you
not but they them their there here what when where which who whom how why all any can does did
done doing else its our ours out over under again more most other some such only own same than too
very just also been being both each few nor once during before after above below off then about
against between through because need needs make makes made want wants like likes use uses used
using get gets got let lets please one two now old ways thing things thanks good okay yes sure
really much many since while still yet even ever never always today tomorrow yesterday
""".split())


def settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """The `prompt_time` block of the hook config, with defaults.

    Absent means ENABLED, matching `write_time`: a config written by an older installer should
    still get the feature the upgrade was for. `enabled: false` turns it off everywhere.
    """

    config = load_config() if config is None else config
    block = config.get("prompt_time")
    block = block if isinstance(block, dict) else {}
    return {
        "enabled": bool(block.get("enabled", True)),
        "k": int(block.get("k", TOP_K)),
        "min_chars": int(block.get("min_chars", MIN_PROMPT_CHARS)),
        "store": str(block.get("store", "")),
    }


#: Suffixes stripped to fold a prompt's wording onto a memo's. Longest first, so `decisions`
#: loses `s` after `ions` has had its chance. Deliberately NOT Porter: a real stemmer is another
#: dependency on the critical path of every user turn, and the cases that matter here are the
#: plural and the participle, which four rules cover. `bus` and `gas` are the classic casualties
#: and MIN_LENGTH is what keeps them whole.
SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ers", "er", "ed", "s")
#: The root left behind must be at least this long. Three, not five: at five, `tests` and `runs`
#: kept their plural while `test` and `run` did not, so the two most common words in this corpus
#: did not fold onto each other. At three, `bus` and `gas` are still safe, their roots being two.
STEM_MIN_LENGTH = 3


@lru_cache(maxsize=1 << 16)
def stem(token: str) -> str:
    """Fold a word onto a crude root. `decisions` and `decision` must be the same token.

    Without this the ranker misses on morphology alone, which on goal-shaped prompts is common:
    a user writes "previous decisions" and the memo says "the decision", and a pure lexical
    ranker scores that as no match at all. Measured on the store this was written against, adding
    it moved the memo that governs the prompt from unranked to rank 1.

    ⚠️ Cached because it is called once per WORD of the whole store, not once per vocabulary
    entry: unmemoised it cost ~350 ms of the ~1,000 ms a prompt takes, on ~2 MB of memos whose
    words repeat heavily. The cache is per process and the process is one hook invocation.
    """

    for suffix in SUFFIXES:
        if len(token) - len(suffix) >= STEM_MIN_LENGTH and token.endswith(suffix):
            root = token[: -len(suffix)]
            return root + "y" if suffix == "ies" else root
    return token


def tokenize(text: str) -> list[str]:
    return [
        stem(t) for t in (w.lower() for w in TOKEN_RE.findall(text)) if t not in STOPWORDS
    ]


def project_slug(path: Path) -> str:
    """The client's project-directory name for an absolute path.

    Every character outside `[A-Za-z0-9-]` becomes `-`, so a Windows path under
    `Documents/recall` becomes `C--Users-gde00-Documents-recall`.
    """

    return "".join(c if (c.isascii() and (c.isalnum() or c == "-")) else "-" for c in str(path))


def find_store(cwd: str, override: str = "") -> Path | None:
    """The memory store for this project, or None.

    ⚠️ **The cwd of a session is routinely a WORKTREE, and a worktree has its own slug.** This
    project keeps its worktrees under `<repo>/.claude/worktrees/<name>`, which slugifies to a
    directory that does not exist while the repository's own does. So the search walks UP from the
    cwd and takes the first ancestor with a store, which finds the repository from any worktree
    without asking git anything. A subprocess per prompt to answer a question that is pure path
    arithmetic is not a trade worth making.

    ⛔ It walks up, never down, and stops at the filesystem root. It cannot reach another
    project's memories, only an enclosing one's.
    """

    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_dir() else None
    if not cwd:
        return None
    projects = claude_config_home() / "projects"
    try:
        here = Path(cwd).resolve()
    except OSError:
        return None
    for directory in (here, *here.parents):
        store = projects / project_slug(directory) / "memory"
        if store.is_dir():
            return store
    return None


def load_memos(store: Path) -> list[tuple[str, str, list[str]]]:
    """`(stem, one-line summary, tokens)` per memo. Index files are skipped, see INDEX_STEMS."""

    memos: list[tuple[str, str, list[str]]] = []
    for path in sorted(store.glob("*.md")):
        if path.stem in INDEX_STEMS:
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # One unreadable memo must not cost the other 328.
            continue
        match = DESCRIPTION_RE.search(raw)
        summary = match.group(1).strip().strip('"').strip("'") if match else ""
        title = path.stem.replace("-", " ")
        tokens = tokenize(title) * FIELD_WEIGHT + tokenize(summary) * FIELD_WEIGHT + tokenize(raw)
        memos.append((path.stem, summary or title, tokens))
    return memos


def rank(
    memos: list[tuple[str, str, list[str]]], query: str, k: int = TOP_K
) -> list[tuple[str, str, float]]:
    """BM25 over the memo files. `(stem, summary, score)`, best first.

    ⛔ Deliberately lexical and deliberately not the corpus. The dense leg would need an embedder,
    which on this machine is an ONNX load measured at ~11 seconds in this project, and the corpus
    is a database this host cannot reach; see the module docstring's table.
    """

    total = len(memos)
    if not total:
        return []
    frequencies: list[dict[str, int]] = []
    document_frequency: dict[str, int] = {}
    for _, _, tokens in memos:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        frequencies.append(counts)
        for token in counts:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    average_length = sum(len(m[2]) for m in memos) / total

    # ⚠️ The length gate is on the PROMPT's content words, not on the ones the corpus happens to
    # know. Filtering first and counting after conflates "the user said almost nothing" with "this
    # corpus has never heard of the subject", and the second is the case a memory layer most needs
    # to keep answering: a narrow, unfamiliar question still deserves whatever two words match.
    prompt_tokens = list(dict.fromkeys(tokenize(query[:MAX_QUERY_CHARS])))
    if len(prompt_tokens) < MIN_QUERY_TOKENS:
        return []
    query_tokens = [t for t in prompt_tokens if t in document_frequency]
    if not query_tokens:
        return []
    inverse: dict[str, float] = {}
    for token in query_tokens:
        frequency = document_frequency[token]
        inverse[token] = math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))

    scored: list[tuple[str, str, float]] = []
    for position, (stem, summary, tokens) in enumerate(memos):
        length = len(tokens)
        score = 0.0
        matched = 0
        for token, weight in inverse.items():
            found = frequencies[position].get(token, 0)
            if not found:
                continue
            matched += 1
            saturation = (found * (BM25_K1 + 1)) / (
                found + BM25_K1 * (1 - BM25_B + BM25_B * length / average_length)
            )
            score += weight * saturation
        if matched >= MIN_MATCHED_TOKENS:
            scored.append((stem, summary, score))
    if not scored:
        return []
    scored.sort(key=lambda row: row[2], reverse=True)
    floor = scored[0][2] * RELATIVE_FLOOR
    return [row for row in scored[:k] if row[2] >= floor]


def render(hits: list[tuple[str, str, float]], store: Path) -> str:
    """The injected text.

    It names the files rather than pasting them. A memo is up to several thousand words and most
    hits do not apply; a one-line summary is enough to decide, and the path is enough to read the
    rest. Saying up front that most will not apply is what makes an irrelevant hit cheap to drop
    instead of something to be reconciled.
    """

    lines = [
        "Project memory was searched with this prompt before you saw it. "
        f"{len(hits)} prior record(s) came back. Most searches return nothing that applies, so "
        "scan the summaries and drop the ones that are about something else.",
        "",
        "⛔ If one of these already settles the question, say so and follow it. Do not re-derive a "
        "decision, re-run a measurement, or rebuild something that already exists. Read the file "
        "before proposing anything that touches its subject.",
        "",
    ]
    for stem, summary, score in hits:
        lines.append(f"- {stem} (score {score:.1f})")
        lines.append(f"  {summary[:SUMMARY_CHARS]}")
        lines.append(f"  {store / (stem + '.md')}")
    return "\n".join(lines)


def user_prompt_submit(payload: dict[str, Any]) -> int:
    """Entry point. Returns 0 always: this hook may decline to speak, never to pass the prompt on."""

    prompt = str(payload.get("prompt") or "").strip()
    options = settings()
    if not options["enabled"] or len(prompt) < options["min_chars"]:
        return 0
    # ⚠️ The rendering and the serialisation are INSIDE the guard, not only the retrieval. The
    # first version of this function caught the search and then formatted outside it, and the
    # test that breaks `render` went red immediately: a formatting bug is exactly as fatal to the
    # user's message as a retrieval one, and "the risky part is the I/O" was the wrong instinct.
    try:
        store = find_store(str(payload.get("cwd") or ""), options["store"])
        if store is None:
            # Unconfigured is silent BY DESIGN: a project with no memory store must behave
            # exactly as it would without this hook.
            return 0
        hits = rank(load_memos(store), prompt, options["k"])
        if not hits:
            return 0
        # `ensure_ascii` left at its default on purpose; see the module docstring's third property.
        document = json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": render(hits, store),
                }
            }
        )
        sys.stdout.write(document)
    except Exception:  # noqa: BLE001 - a retrieval failure must never eat the user's message
        return 0
    return 0


__all__ = [
    "find_store",
    "load_memos",
    "project_slug",
    "rank",
    "render",
    "settings",
    "tokenize",
    "user_prompt_submit",
]
