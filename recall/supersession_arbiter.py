"""Propose the supersession edges a corpus never declared, for a human to accept or reject.

The trust layer acts on `supersedes:` frontmatter and on nothing else (`recall/trust.py`), and a
memory store written over months declares only a fraction of the replacements it contains. The
memo that corrected a value and the memo it corrected both stay `active`, both match a query,
and a reader has no way to tell which one to believe. This module finds candidates for the
missing edges and asks a model to judge them. It never writes one: what it produces are
`InferenceProposal`s with `status="requires_review"`, through
`recall/reasoning_proposals/_arbiter.py`, and the only door from a proposal to the corpus is
`recall rewrite apply`, which needs a named reviewer.

**The pipeline, in the order money is spent.** Each stage is free until the last.

1. *Neighbours.* Every memo's `RECALL_ARBITER_NEIGHBOURS` nearest memos by TF-IDF cosine over the
   human-written body. Lexical rather than embedded, because `recall rewrite` is filesystem-only
   and has no vectors to read; on a store whose authored edges served as ground truth, ten lexical
   neighbours already contained nearly every authored pair.
2. *Trigger.* A pair goes forward only when its file names say the two memos are about one
   thing: the same leading `YYYY-MM-DD`, or slug-word Jaccard of at least `SLUG_JACCARD`. Metadata
   only, and it is what makes the model gate affordable: without it, a model sees every neighbour
   pair, and on these stores most neighbours are related work rather than a replacement.
3. *Direction.* Decided from metadata, never from the model: a frontmatter `modified:` stamp when
   both memos carry one, else the leading file-name date. A pair whose direction cannot be read
   is dropped BEFORE any call, because an undirected supersession is a proposal nobody can apply,
   and reversing one declares the live memo stale.
4. *Gate.* One model call per pair, asked for a stated probability that the two form a
   supersession pair, with two verbatim quotes showing the two versions of the claim. A claim of
   50 or more whose quotes are not verbatim, at least `MIN_QUOTE_CHARS` long, in the text the model
   was shown, scores zero. A pair is proposed at `threshold` or above. The notes are shown in an
   order fixed by a hash of their names, so file order cannot leak which one is newer.

**What the threshold means, and what it does not.** The default model, prompt and threshold
are the configuration a private measurement selected on one engineer's memory store. On a second
store the same threshold did not transfer: thresholds on a stated probability are a property of
a store as much as of a model. So the output is a REVIEW QUEUE, and nothing in it is evidence on
its own. That is also why the proposals carry the quotes: a reviewer should be able to accept or
reject one by reading two lines.

**Spend is bounded and cached.** At most `max_pairs` pairs are judged per run, the most similar
first, and every answer is cached on disk keyed by the model's identity, the prompt text and the
two texts shown. `recall rewrite` re-derives its proposals on every verb, so without the cache
`apply` would re-pay for the whole run, and a model that answered differently the second time
would make the id a reviewer copied from `plan` disappear. A pair whose call FAILED is not cached,
so the next run retries it.

**Off by default, and it sends text off the machine.** `RECALL_SUPERSESSION_ARBITER=1` turns it
on, `RECALL_ARBITER_API_KEY` is required, and every judged pair sends up to `BODY_CHARS` of both
memos to `RECALL_ARBITER_BASE_URL`. It shares `recall.truth_extraction._openai_engine`'s HTTP client, retry policy and
timeout handling, under the `RECALL_ARBITER_*` names. Requires ``pip install "recall-rag[extract]"``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from collections import Counter
from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from recall.document import parse_document
from recall.frontmatter import frontmatter_span, supersedes_key, supersedes_targets
from recall.observability import get_logger
from recall.truth_extraction._openai_engine import ChatClient, _host_of, _setting

_log = get_logger("supersession_arbiter")

ARBITER_PROVIDER_ID = "recall.supersession_arbiter"
#: Bumped by hand with any change to `SYSTEM_PROMPT` or to `score`. The cache key hashes the
#: prompt text itself, so a forgotten bump cannot serve a stale answer; this label is for the
#: audit record, where a hash means nothing to a reader.
PROMPT_REVISION = "stated-probability-v1"
DEFAULT_ARBITER_MODEL = "openai/gpt-6.1-sol"
DEFAULT_ARBITER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_THRESHOLD = 0.90
DEFAULT_NEIGHBOURS = 10
DEFAULT_MAX_PAIRS = 200
#: Concurrent calls. Hosted providers rate-limit per key, and `retry_with_backoff` absorbs a 429.
WORKERS = 8
#: Characters of each body shown to the model. Long memos are cut, not summarised.
BODY_CHARS = 6000
MIN_QUOTE_CHARS = 20
SLUG_JACCARD = 0.3

ENV_ENABLED = "RECALL_SUPERSESSION_ARBITER"
ENV_PREFIX = "RECALL_ARBITER"
ENV_CACHE_PATH = "RECALL_ARBITER_CACHE"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})
_CACHE_DISABLED = frozenset({"", "0", "off", "no", "false", "none"})

SYSTEM_PROMPT = (
    "You compare two notes from one engineer's working memory. Decide only whether they form a "
    "SUPERSESSION PAIR: one of them replaces, corrects, retires or updates a specific claim, "
    "decision, value or status stated in the other, on the same specific subject, so a reader "
    "should trust one instead of the other. Sharing a topic, citing each other, or continuing the "
    "same line of work is NOT a pair. Do not decide which note is newer. "
    'Reply with JSON only: {"probability": <integer 0 to 100, how likely they form a pair>, '
    '"subject": "...", "quote_a": "...", "quote_b": "..."}. When probability is 50 or more, quote_a '
    "and quote_b are copied verbatim from Note A and Note B, at least 20 characters each, showing the "
    "two versions of the claim."
)

_DATE_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})")
_TOKEN = re.compile(r"[a-z][a-z0-9_]{2,}")
_STOPWORDS = frozenset(
    {"the", "a", "an", "of", "to", "is", "and", "in", "on", "for", "not", "no", "it", "be", "by",
     "with", "from", "at", "as", "its", "or", "md"}
)
_MODIFIED = re.compile(r"^\s*modified:\s*(\S+)")


# ---------------------------------------------------------------- notes and candidates (pure)


@dataclass(frozen=True)
class Note:
    """What the arbiter reads of one memo."""

    name: str
    body: str
    modified: datetime | None
    day: date | None
    declares: tuple[str, ...]


def basename(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def file_day(name: str) -> date | None:
    """The leading `YYYY-MM-DD` of a file name, or None; an impossible date is None too."""
    match = _DATE_PREFIX.match(basename(name))
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def modified_stamp(text: str) -> datetime | None:
    """The frontmatter `modified:` value as an aware datetime, or None.

    `recall/frontmatter.py` recognises only the keys the trust layer acts on, and `modified` is
    not one of them, so it is read here from the same span. A naive value is taken as UTC: a
    comparison between a naive and an aware datetime raises, and one memo written by a tool that
    omits the offset must not stop the run.
    """
    span = frontmatter_span(text)
    if span is None:
        return None
    for line in text.split("\n")[1:span]:
        found = _MODIFIED.match(line)
        if not found:
            continue
        raw = found.group(1).strip().strip("'\"").replace("Z", "+00:00")
        try:
            stamp = datetime.fromisoformat(raw)
        except ValueError:
            return None
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    return None


def read_note(name: str, text: str) -> Note:
    document = parse_document(text)
    return Note(
        name=name,
        body=document.human_body,
        modified=modified_stamp(text),
        day=file_day(name),
        declares=supersedes_targets(document.meta.get("supersedes")),
    )


def slug_words(name: str) -> set[str]:
    stem = _DATE_PREFIX.sub("", basename(name).removesuffix(".md")).lstrip("-")
    return {
        token
        for token in re.split(r"[-_.]", stem.lower())
        if token and token not in _STOPWORDS and not token.isdigit()
    }


def triggered(left: str, right: str) -> bool:
    """Same leading file-name date, or slug-word Jaccard at least `SLUG_JACCARD`."""
    day_left, day_right = file_day(left), file_day(right)
    if day_left is not None and day_left == day_right:
        return True
    words_left, words_right = slug_words(left), slug_words(right)
    union = words_left | words_right
    return bool(union) and len(words_left & words_right) / len(union) >= SLUG_JACCARD


def orient(left: Note, right: Note) -> tuple[Note, Note, str] | None:
    """`(older, newer, source)`, or None when metadata cannot say. Never a guess.

    One signal at a time: `modified` decides only when BOTH memos carry it and it differs, and
    the file-name date is consulted only when `modified` could not decide. Mixing a stamp from
    one memo with a date from the other compares two different clocks.
    """
    if left.modified is not None and right.modified is not None and left.modified != right.modified:
        older, newer = (left, right) if left.modified < right.modified else (right, left)
        return older, newer, "modified"
    if left.day is not None and right.day is not None and left.day != right.day:
        older, newer = (left, right) if left.day < right.day else (right, left)
        return older, newer, "file-name date"
    return None


def declared_pair(left: Note, right: Note) -> bool:
    """Whether either memo already declares the other as the one it supersedes."""
    for note, other in ((left, right), (right, left)):
        other_key = supersedes_key(basename(other.name))
        if any(supersedes_key(target) == other_key for target in note.declares):
            return True
    return False


def _tfidf(bodies: Mapping[str, str]) -> dict[str, dict[str, float]]:
    tokens = {
        name: [t for t in _TOKEN.findall(body.lower()) if t not in _STOPWORDS]
        for name, body in bodies.items()
    }
    total = len(tokens)
    df = Counter(t for words in tokens.values() for t in set(words))
    vectors: dict[str, dict[str, float]] = {}
    for name, words in tokens.items():
        weights = {
            t: (1 + math.log(c)) * math.log(total / df[t])
            for t, c in Counter(words).items()
            if df[t] < total
        }
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        vectors[name] = {t: w / norm for t, w in weights.items()}
    return vectors


def lexical_neighbours(
    bodies: Mapping[str, str], k: int, focus: Collection[str] | None = None
) -> dict[tuple[str, str], float]:
    """Each focus memo's `k` nearest memos by TF-IDF cosine, as `{(a, b): similarity}`, a < b.

    `focus` defaults to every memo. A pair is kept once whichever side found it.
    """
    vectors = _tfidf(bodies)
    postings: dict[str, list[tuple[str, float]]] = {}
    for name in sorted(vectors):
        for term, weight in vectors[name].items():
            postings.setdefault(term, []).append((name, weight))
    pairs: dict[tuple[str, str], float] = {}
    for name in sorted(focus if focus is not None else vectors):
        if name not in vectors:
            continue
        scores: dict[str, float] = {}
        for term, weight in vectors[name].items():
            for other, other_weight in postings[term]:
                if other != name:
                    scores[other] = scores.get(other, 0.0) + weight * other_weight
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        for other, similarity in ranked:
            if similarity <= 0:
                continue
            key = (name, other) if name < other else (other, name)
            pairs[key] = max(pairs.get(key, 0.0), similarity)
    return pairs


def blinded_order(left: str, right: str) -> tuple[str, str]:
    """Which memo is shown as Note A: fixed by a hash of the two names, never by their order."""
    first, second = sorted((left, right))
    digest = hashlib.sha256(f"{first}\x00{second}".encode("utf-8", "surrogatepass")).digest()
    return (first, second) if digest[0] % 2 == 0 else (second, first)


# ---------------------------------------------------------------- the gate (pure)


def _normalise(text: str) -> str:
    return " ".join(text.split())


def grounded(quote: object, shown: str) -> bool:
    """A quote counts only if it is at least `MIN_QUOTE_CHARS` long and verbatim in `shown`.

    Whitespace is normalised on both sides, because a model reflows a line break into a space
    and that is not a fabrication.
    """
    if not isinstance(quote, str):
        return False
    wanted = _normalise(quote)
    return len(wanted) >= MIN_QUOTE_CHARS and wanted in _normalise(shown)


def parse_answer(text: str) -> dict[str, Any] | None:
    """The JSON object in a reply, or None. Tolerates prose around one object."""
    for candidate in (text, *re.findall(r"\{.*\}", text, re.DOTALL)[:1]):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def score(answer: Mapping[str, Any] | None, shown_a: str, shown_b: str) -> float:
    """The stated probability in [0, 1]. A claim of 0.5 or more without grounded quotes is 0.

    A boolean is not a number here, though Python says it is: `{"probability": true}` would
    otherwise read as 1 percent.
    """
    if answer is None:
        return 0.0
    value = answer.get("probability")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
        return 0.0
    probability = min(max(float(value), 0.0), 100.0) / 100.0
    if probability >= 0.5 and not (
        grounded(answer.get("quote_a"), shown_a) and grounded(answer.get("quote_b"), shown_b)
    ):
        return 0.0
    return probability


def user_message(shown_a: str, shown_b: str) -> str:
    return f"Note A:\n{shown_a}\n\n---\n\nNote B:\n{shown_b}"


# ---------------------------------------------------------------- settings and cache


@dataclass(frozen=True)
class ArbiterSettings:
    model_id: str
    revision: str
    base_url: str
    threshold: float
    neighbours: int
    max_pairs: int


def _bounded(source: Mapping[str, str], name: str, default: str, kind: type, low: float, high: float) -> Any:
    raw = _setting(source, name, default)
    try:
        value = kind(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a {kind.__name__}") from None
    if not low <= value <= high:
        raise ValueError(f"{name}={raw!r} must be between {low} and {high}")
    return value


def arbiter_settings(env: Mapping[str, str] | None = None) -> ArbiterSettings | None:
    """The arbiter's settings, or None when it is off (the default). Malformed values refuse."""
    source = env if env is not None else os.environ
    raw = source.get(ENV_ENABLED, "").strip().lower()
    if raw in _FALSE:
        return None
    if raw not in _TRUE:
        raise ValueError(
            f"{ENV_ENABLED}={raw!r} is not a boolean. Use one of {sorted(_TRUE)} to enable it "
            "or leave it unset."
        )
    return ArbiterSettings(
        model_id=_setting(source, f"{ENV_PREFIX}_MODEL", DEFAULT_ARBITER_MODEL),
        revision=_setting(source, f"{ENV_PREFIX}_REVISION", "unpinned"),
        base_url=_setting(source, f"{ENV_PREFIX}_BASE_URL", DEFAULT_ARBITER_BASE_URL),
        # Zero is refused: it would propose every candidate the model answered at all.
        threshold=_bounded(source, f"{ENV_PREFIX}_THRESHOLD", str(DEFAULT_THRESHOLD), float, 0.5, 1.0),
        neighbours=_bounded(source, f"{ENV_PREFIX}_NEIGHBOURS", str(DEFAULT_NEIGHBOURS), int, 1, 100),
        max_pairs=_bounded(source, f"{ENV_PREFIX}_MAX_PAIRS", str(DEFAULT_MAX_PAIRS), int, 1, 100_000),
    )


def default_cache_path(env: Mapping[str, str] | None = None) -> Path | None:
    """`RECALL_ARBITER_CACHE`, or the platform cache directory; None when switched off."""
    source = env if env is not None else os.environ
    raw = source.get(ENV_CACHE_PATH)
    if raw is not None:
        if raw.strip().lower() in _CACHE_DISABLED:
            return None
        return Path(raw).expanduser()
    if os.name == "nt":
        base = source.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    else:
        base = source.get("XDG_CACHE_HOME")
        root = Path(base) if base else Path.home() / ".cache"
    return root / "recall" / "arbiter.sqlite3"


class ArbiterCache:
    """Model answers, keyed by everything that can change one. Deleting the file costs a re-run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, answer TEXT NOT NULL)"
        )
        self._db.commit()

    def get(self, key: str) -> str | None:
        row = self._db.execute("SELECT answer FROM answers WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def put(self, key: str, answer: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO answers VALUES (?, ?)", (key, answer))
        self._db.commit()

    def close(self) -> None:
        self._db.close()


def open_cache(env: Mapping[str, str] | None = None) -> ArbiterCache | None:
    """Open the cache, or None. Never raises: an unwritable cache costs a re-run, not the run."""
    path = default_cache_path(env)
    if path is None:
        return None
    try:
        return ArbiterCache(path)
    except (OSError, sqlite3.Error) as exc:
        _log.warning("arbiter cache unavailable at %s (%s); every pair will be called", path, exc)
        return None


# ---------------------------------------------------------------- the run


@dataclass(frozen=True)
class ArbiterVerdict:
    """One pair the gate passed. `older` is the SUPERSEDED memo, `newer` the superseding one."""

    older: str
    newer: str
    probability: float
    subject: str
    quote_older: str
    quote_newer: str
    direction_source: str


@dataclass(frozen=True)
class ArbiterRun:
    """What one run did, so a reviewer can see what was and was not looked at."""

    verdicts: tuple[ArbiterVerdict, ...]
    candidates: int
    already_declared: int
    undirected: int
    over_budget: int
    cached: int
    called: int
    failed: int

    def summary(self) -> str:
        return (
            f"arbiter: {self.candidates} candidate pair(s); {self.already_declared} already "
            f"declared, {self.undirected} without a readable direction, {self.over_budget} over "
            f"the {ENV_PREFIX}_MAX_PAIRS budget; judged {self.cached} from cache and "
            f"{self.called} by the model, {self.failed} failed; {len(self.verdicts)} proposed"
        )


class SupersessionArbiter:
    def __init__(
        self,
        *,
        client: ChatClient,
        settings: ArbiterSettings,
        cache: ArbiterCache | None = None,
        workers: int = WORKERS,
    ) -> None:
        self._client = client
        self.settings = settings
        self._cache = cache
        self._workers = workers
        #: The endpoint is part of the identity, for the reason `OpenAIExtractionEngine` gives:
        #: two endpoints serving one model name are two judges, and one cache key for both would
        #: serve one's answers as the other's.
        self.provider_id = f"{ARBITER_PROVIDER_ID}@{_host_of(settings.base_url)}"
        self.model_id = settings.model_id
        self.provider_revision = (
            f"{settings.revision}+{PROMPT_REVISION}+threshold={settings.threshold:g}"
        )

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()

    def cache_key(self, shown_a: str, shown_b: str) -> str:
        digest = hashlib.sha256()
        for part in (
            self.provider_id, self.settings.model_id, self.settings.revision, SYSTEM_PROMPT,
            shown_a, shown_b,
        ):
            digest.update(part.encode("utf-8", "surrogatepass"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def _ask(self, shown_a: str, shown_b: str) -> str:
        return self._client.complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message(shown_a, shown_b)},
            ],
            # No temperature: the configuration this gate was selected under sent none, and some
            # reasoning models refuse any value but their default.
            response_format={"type": "json_object"},
        )

    def run(
        self, documents: Mapping[str, str], *, focus: Collection[str] | None = None
    ) -> ArbiterRun:
        """Judge the candidate pairs among `documents`; at least one side of each is in `focus`."""
        notes = {name: read_note(name, text) for name, text in documents.items()}
        similar = lexical_neighbours(
            {name: note.body for name, note in notes.items()}, self.settings.neighbours, focus
        )
        candidates = sorted(
            (pair for pair in similar if triggered(*pair)),
            key=lambda pair: (-similar[pair], pair),
        )
        declared = undirected = 0
        jobs: list[tuple[Note, Note, str]] = []
        for left, right in candidates:
            if declared_pair(notes[left], notes[right]):
                declared += 1
                continue
            oriented = orient(notes[left], notes[right])
            if oriented is None:
                undirected += 1
                continue
            jobs.append(oriented)
        over_budget = max(0, len(jobs) - self.settings.max_pairs)
        jobs = jobs[: self.settings.max_pairs]

        shown: dict[str, str] = {name: notes[name].body[:BODY_CHARS] for name in notes}
        answers: dict[int, str] = {}
        misses: list[tuple[int, str, str, str]] = []
        cached = 0
        for index, (older, newer, _source) in enumerate(jobs):
            note_a, note_b = blinded_order(older.name, newer.name)
            key = self.cache_key(shown[note_a], shown[note_b])
            hit = self._cache.get(key) if self._cache is not None else None
            if hit is not None:
                answers[index] = hit
                cached += 1
            else:
                misses.append((index, key, shown[note_a], shown[note_b]))

        failed = 0
        if misses:
            with ThreadPoolExecutor(max_workers=self._workers) as pool:
                futures = [
                    (index, key, pool.submit(self._ask, text_a, text_b))
                    for index, key, text_a, text_b in misses
                ]
                for index, key, future in futures:
                    try:
                        reply = future.result()
                    except Exception as exc:  # noqa: BLE001  # BROAD-CATCH: fail-open (one pair, counted)
                        failed += 1
                        _log.warning("arbiter call failed: %s", type(exc).__name__)
                        continue
                    answers[index] = reply
                    if self._cache is not None:
                        # The cache is an optimisation; a full disk must not cost the answers.
                        with suppress(sqlite3.Error):
                            self._cache.put(key, reply)

        verdicts: list[ArbiterVerdict] = []
        for index, (older, newer, source) in enumerate(jobs):
            if index not in answers:
                continue
            note_a, note_b = blinded_order(older.name, newer.name)
            answer = parse_answer(answers[index])
            probability = score(answer, shown[note_a], shown[note_b])
            if answer is None or probability < self.settings.threshold:
                continue
            quote_a, quote_b = str(answer.get("quote_a")), str(answer.get("quote_b"))
            older_is_a = note_a == older.name
            verdicts.append(
                ArbiterVerdict(
                    older=older.name,
                    newer=newer.name,
                    probability=probability,
                    subject=str(answer.get("subject") or "")[:200],
                    quote_older=quote_a if older_is_a else quote_b,
                    quote_newer=quote_b if older_is_a else quote_a,
                    direction_source=source,
                )
            )
        return ArbiterRun(
            verdicts=tuple(verdicts),
            candidates=len(candidates),
            already_declared=declared,
            undirected=undirected,
            over_budget=over_budget,
            cached=cached,
            called=len(misses),
            failed=failed,
        )


def resolve_arbiter(env: Mapping[str, str] | None = None) -> SupersessionArbiter | None:
    """The arbiter configured by `env` (default: the process environment), or None when off."""
    from recall.truth_extraction._openai_engine import _client_from_env

    source = env if env is not None else os.environ
    settings = arbiter_settings(source)
    if settings is None:
        return None
    client = _client_from_env(source, prefix=ENV_PREFIX, default_model=DEFAULT_ARBITER_MODEL)
    return SupersessionArbiter(client=client, settings=settings, cache=open_cache(source))


__all__ = [
    "ARBITER_PROVIDER_ID",
    "DEFAULT_ARBITER_MODEL",
    "DEFAULT_THRESHOLD",
    "PROMPT_REVISION",
    "SYSTEM_PROMPT",
    "ArbiterCache",
    "ArbiterRun",
    "ArbiterSettings",
    "ArbiterVerdict",
    "SupersessionArbiter",
    "arbiter_settings",
    "resolve_arbiter",
]
