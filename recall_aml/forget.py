"""R2-1: honour a user's in-conversation request to forget a detail, behind a default-off flag.

An evaluation's memory system has no delete call, so a request such as "Please forget that I lift
weights for strength training." reaches it only as an ordinary user turn inside an Add. C9 then
ranks the exchange that used the detail, and the request itself (which restates the detail), near
the top of a later Search, and a reader asked to personalise uses it. This module is the design's
Option B (recall-lab ``research/designs/2026-09-28-r2-1-forgetting.md``):

* **Detection at Add** (``find_forget_requests``): a pattern over USER messages only, refusing
  negations ("don't forget"), whole-memory resets ("forget everything", "forget the previous
  instructions"), idioms ("forget it, never mind"), task-document deletes ("delete my draft from
  the archive") and anything that looks like a coding trajectory. It is the stage-0 detector
  tightened by the design's scope rules. Measured 2026-09-28, read-only on the testbench's
  PersonaMem-v2 32k histories: 707 of the 707 benchmark forget turns detected, and 2,313 user
  turns flagged in those 126 histories, the same count as the stage-0 detector there; 0 in
  BEAM-100K, LoCoMo-10 and CL-bench user turns, 1 in 93,931
  distinct LongMemEval-S user turns (a scoped "forget what I have said about ..."). Replayed over
  X-1's stored C9 lists, ``drop`` cut forgotten details in the top 10 from 33 of 42 questions to 1
  and removed 7 evidence windows of 137 other questions, the design's rule RL figures. ``ForgetConfirmer`` is the hook for an optional gpt-4o-mini confirmation; nothing
  here calls a model, and a confirmer that fails leaves the pattern verdict standing.
* **A ledger** (``ledger_chunks``): one row per request, ``record_type == "forget_request"``, with
  a deterministic id from (tenant, session, message ordinal, target), so a retried Add rewrites the
  same row. The rows live in their own derived tenant (``identity.forget_ledger_tenant``), beside
  the corpus and never inside it, so no retrieval leg can return one as evidence.
* **Suppression at Search**, in one of three modes (``FORGET_MODES``):

  - ``drop``: remove every item that states the target (rule R+text: the target's content stems
    covered at ``TARGET_MATCH_FRACTION``), the request's own window (it shares a word run with the
    request sentence), and the immediately preceding exchange when it also matches the target at
    the lower ``PRECEDING_MATCH_FRACTION`` (rule R). The caller renders from what is left, so the
    list backfills from lower ranks.
  - ``stub``: keep every item, but replace the matching sentences with ``STUB_SENTENCE`` (the
    design's Option C).
  - ``annotate``: keep everything and prepend a note naming what the user asked to forget (the
    control arm).

Two parts of the design are deliberately not here. Suppression is not ordered by Add sequence (a
later re-assertion by the user is suppressed too), because raw windows carry no Add sequence and
a timestamp proxy would leak the acknowledgement turn when Adds arrive one message at a time.
Compiled records are matched by their text only, not by ``evidence_spans`` pointing into a
suppressed message.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any, Literal

from recall.types import Chunk, ScoredChunk
from recall_aml.identity import canonical_digest, session_digest
from recall_aml.models import Message, SearchItem, TextContentPart
from recall_aml.window_format import looks_like_coding

ForgetMode = Literal["off", "drop", "stub", "annotate"]
#: What ``RECALL_AML_FORGET`` and ``HostedVariant.forget_suppression`` may be; ``off`` first.
FORGET_MODES: tuple[ForgetMode, ...] = ("off", "drop", "stub", "annotate")
FORGET_ENV = "RECALL_AML_FORGET"
FORGET_RECORD_TYPE = "forget_request"
FORGET_DETECTOR = "pattern-v1"
#: Share of a target's content stems an item must contain to state it (the design's rule L).
TARGET_MATCH_FRACTION = 0.6
#: The lower share for the exchange right before the request, which rule R drops only when it
#: also mentions the target.
PRECEDING_MATCH_FRACTION = 0.4
#: A window belongs to a message when they share a run of this many words.
SHINGLE_WORDS = 6
#: How far back the "preceding exchange" reaches: up to and including the previous user turn,
#: never more than this many messages.
PRECEDING_MAX_MESSAGES = 4
MAX_TARGET_CHARS = 200
STUB_SENTENCE = (
    "[The user asked the assistant to stop using one personal detail shared here; "
    "do not personalise on it.]"
)
ANNOTATION_PREFIX = "Note: the user asked the assistant to forget the following and not to use it: "

_SENTENCE_START = r"(?:^|[.!?]+[\"')\]]*\s+|\n)\s*"
_POLITE = (
    r"(?:(?:please|kindly|also|and|so|ok(?:ay)?|now)\s*,?\s+)*"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?(?:please\s+)?"
)
_VERB = (
    r"(?P<verb>forget|stop\s+remembering|(?:do\s+not|don't|dont)\s+remember"
    r"|erase|delete|remove|drop|disregard)"
)
#: ``rest`` is a lookahead, so a second request later on the same line is still found.
_REQUEST = re.compile(
    _SENTENCE_START + r"(?P<request>" + _POLITE + _VERB + r")\b(?=(?P<rest>[^\n]{0,300}))",
    re.I,
)
#: A memory verb the request's own words negate. The request pattern cannot start with one, so
#: this only refuses a sentence that also carries one ("Please forget it, no, don't forget ...").
_NEGATED = re.compile(
    r"\b(?:do\s+not|don't|dont|never|won't|wont|wouldn't|shouldn't|can't|cannot|couldn't|not)"
    r"\s+(?:ever\s+|just\s+|to\s+)?forget\b",
    re.I,
)
#: A first-person or memory object, which a "forget" request must carry: it is what separates
#: "Please forget that I lift weights" from "Forget about heavy garments".
_MEMORY_OBJECT = re.compile(
    r"\b(?:that\s+(?:i|my|me)\b|about\s+(?:me|my|myself)\b|my\b"
    r"|that\s+you\s+(?:remember(?:ed)?|know|knew)\b"
    r"|the\s+(?:detail|fact|part|info(?:rmation)?|bit|thing|preference|memory)\s+(?:about|for|of|that)\b"
    r"|what\s+i\s+(?:said|told|mentioned|shared)\b|from\s+(?:your\s+|the\s+)?memory\b"
    r"|i\s*(?:am|'m|have|'ve|like|love|enjoy|prefer|hate|used\s+to|was|mentioned|said|told)\b)",
    re.I,
)
#: What an erase, delete, remove, drop or disregard request needs: an explicit memory object.
#: "Please delete my earlier draft from the archive" and "Remove the foil" are tasks, not
#: requests to forget.
_STRONG_MEMORY_OBJECT = re.compile(
    r"\b(?:from\s+(?:your\s+|the\s+)?memory|(?:what|that)\s+i\s+(?:said|told\s+you|mentioned|shared)"
    r"|the\s+(?:detail|fact|info(?:rmation)?)\s+(?:about|that))\b",
    re.I,
)
#: A whole-memory reset or an idiom, read at the start of what follows the verb: "everything",
#: "all of that", "it, never mind", "about it", "the previous instructions", "what we discussed".
#: "My past fracture" and "that you remember I ..." are not resets.
_BLANKET = re.compile(
    r"^\s*(?:about\s+)?(?:everything|anything|all\b|what\s+(?:we|you)\b"
    r"|(?:it|this|that|these|those)\s*(?:[.!?,;:]|$)"
    r"|(?:the\s+)?(?:previous|prior|above|earlier|last|past)\s*(?:[.!?,;:]|$)"
    r"|(?:(?:the|my|your|our)\s+)?(?:previous|prior|above|earlier|last|past)\s+(?:few\s+)?"
    r"(?:ones?|instructions?|messages?|conversations?|chats?|prompts?|questions?|requests?"
    r"|answers?|responses?|information|context|turns?)\b)",
    re.I,
)
_PREFIX = re.compile(
    r"^\s*(?:about\s+|that\s+you\s+(?:remember(?:ed)?|know|knew)\s+(?:that\s+|about\s+)?|that\s+"
    r"|the\s+(?:detail|fact|info(?:rmation)?|part|bit|thing|preference|memory)\s+(?:about|for|of|that)\s+"
    r"|my\s+(?:preference|detail|fact)\s+(?:about|for|of|that)\s+"
    r"|what\s+i\s+(?:said|told\s+you|mentioned|shared)\s+(?:about\s+)?"
    r"|i\s+(?:mentioned|said|told\s+you)\s+(?:that\s+|about\s+)?)",
    re.I,
)
_SUFFIX = re.compile(
    r"(?:\s*,?\s*(?:from\s+(?:your\s+|the\s+)?memory|going\s+forward|please|for\s+now|for\s+good"
    r"|entirely|completely|as\s+well|too)\b)*\s*[.!?,;:]*\s*$",
    re.I,
)
_FIRST_PERSON = re.compile(r"\b(?:i|i'm|i've|my|me|myself)\b", re.I)
_FENCED_CODE = re.compile(r"```.*?(?:```|$)", re.S)

#: Words that carry no content of their own (r21_lib.STOP, unchanged).
_STOP = frozenset(
    """a an the i im my me to of in on for and or but with from at by as is are was were be been
    being that this it its about over than more most less very really often occasionally sometimes
    usually have has had do does enjoy enjoys enjoying like likes liking prefer prefers preferring
    preference preferences love loves tend tends especially particularly into whenever when while
    who which what feel feeling feelings someone one lot""".split()
)
#: Words that name the scope of a request rather than its subject; a target made only of these is
#: a reset or an idiom ("everything", "the previous instructions", "anything I said").
_SCOPE = frozenset(
    """everything anything something all previous prior above earlier before instruction
    instructions message messages ones conversation chat information info stuff thing things
    whatever mind said told asked mentioned discussed talked shared just now today okay rest last
    whole entire detail details fact facts part bit memory remember remembered forget please never
    you your""".split()
)


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_stems(text: str) -> frozenset[str]:
    """The content stems of ``text``: the stage-0 stemmer, without stop and scope words."""
    return frozenset(
        _stem(word)
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in _STOP and word not in _SCOPE
    )


def covers(target: frozenset[str], text: frozenset[str], fraction: float) -> bool:
    """Whether ``text`` holds enough of ``target``'s stems: all of one or two, else the fraction."""
    if not target:
        return False
    have = len(target & text)
    need = len(target) if len(target) <= 2 else max(2, math.ceil(fraction * len(target)))
    return have >= need


def shingles(text: str) -> frozenset[tuple[str, ...]]:
    """Every run of ``SHINGLE_WORDS`` words; a shorter text of three or more words is one run."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < SHINGLE_WORDS:
        return frozenset({tuple(words)}) if len(words) >= 3 else frozenset()
    return frozenset(
        tuple(words[start : start + SHINGLE_WORDS]) for start in range(len(words) - SHINGLE_WORDS + 1)
    )


def _sentence_end(text: str, start: int) -> int:
    found = re.search(r"[.!?](?=\s|$)|\n", text[start:])
    return len(text) if found is None else start + found.end()


@dataclass(frozen=True)
class DetectedRequest:
    """One forget request inside one message: the target and the request sentence."""

    target: str
    sentence: str


def detect_forget_requests(text: str) -> list[DetectedRequest]:
    """Every forget request in one message's text, in order; empty when there is none.

    A request is a sentence-initial imperative addressed to the assistant, with a first-person or
    memory object, whose target (the text after the verb, cut at the sentence end and stripped of
    "that", "the detail about", "from your memory" and the like) names at least one content word.
    A single content word is accepted only with a first-person object ("Please forget that I'm
    vegan"); resets, idioms and negations are refused.
    """
    text = _FENCED_CODE.sub(" ", text)
    found: list[DetectedRequest] = []
    for match in _REQUEST.finditer(text):
        verb = re.sub(r"\s+", " ", match.group("verb").lower())
        rest = match.group("rest")
        start = match.start("request")
        sentence = text[start : _sentence_end(text, start)].strip()
        if _NEGATED.search(sentence):
            continue
        cut = re.split(r"(?<=[.!?])\s+", rest, maxsplit=1)[0]
        if verb in {"forget", "stop remembering", "do not remember", "don't remember", "dont remember"}:
            if not _MEMORY_OBJECT.search(rest[:120]):
                continue
        elif not _STRONG_MEMORY_OBJECT.search(rest[:160]):
            continue
        if _BLANKET.match(cut):
            continue
        target = cut
        for _ in range(3):
            stripped = _PREFIX.sub("", target, count=1)
            if stripped == target:
                break
            target = stripped
        target = _SUFFIX.sub("", target).strip(" \t,;:\"'")[:MAX_TARGET_CHARS].strip()
        stems = content_stems(target)
        if not stems or (len(stems) == 1 and not _FIRST_PERSON.search(cut)):
            continue
        found.append(DetectedRequest(target=target, sentence=sentence))
    return found


#: The hook for an optional model confirmation: given the whole message and the pattern's
#: request, return the request to keep (possibly with a corrected target) or None to reject it.
#: A gpt-4o-mini confirmer would reject quotes of someone else and idioms the pattern let through.
#: Nothing in this package supplies one; ``pattern_only`` is the default.
ForgetConfirmer = Callable[[str, DetectedRequest], DetectedRequest | None]


def pattern_only(message: str, request: DetectedRequest) -> DetectedRequest | None:
    """The default confirmer: the pattern's verdict stands."""
    del message
    return request


@dataclass(frozen=True)
class ForgetRequest:
    """A detected request, located in its Add."""

    session_id: str
    #: The message's position inside its Add (not inside the session: every Add counts from 0).
    message_ordinal: int
    target: str
    request_text: str
    preceding_text: str
    detector: str = FORGET_DETECTOR


def _message_text(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(part.text for part in message.content if isinstance(part, TextContentPart))


def _is_user(message: Message) -> bool:
    return message.role.strip().lower() == "user"


def _preceding_text(messages: Sequence[Message], ordinal: int) -> str:
    """The exchange right before a message: back to and including the previous user turn."""
    parts: list[str] = []
    index = ordinal - 1
    while index >= 0 and len(parts) < PRECEDING_MAX_MESSAGES:
        parts.insert(0, _message_text(messages[index]))
        if _is_user(messages[index]):
            break
        index -= 1
    return "\n".join(part for part in parts if part.strip())


def find_forget_requests(
    messages: Sequence[Message],
    session_id: str,
    *,
    confirm: ForgetConfirmer | None = None,
) -> list[ForgetRequest]:
    """The forget requests in one Add's USER messages; none for a coding trajectory.

    ``confirm`` is the optional model check. If it raises, the pattern verdict stands: a failed
    confirmation must never turn into "no forgetting".
    """
    if looks_like_coding(messages):
        return []
    checker = confirm or pattern_only
    requests: list[ForgetRequest] = []
    for ordinal, message in enumerate(messages):
        if not _is_user(message):
            continue
        text = _message_text(message)
        for detected in detect_forget_requests(text):
            try:
                confirmed = checker(text, detected)
            except Exception:  # BROAD-CATCH: the pattern verdict stands when the check fails
                confirmed = detected
            if confirmed is None:
                continue
            requests.append(
                ForgetRequest(
                    session_id=session_id,
                    message_ordinal=ordinal,
                    target=confirmed.target,
                    request_text=confirmed.sentence,
                    preceding_text=_preceding_text(messages, ordinal),
                )
            )
    return requests


def ledger_id(tenant: str, request: ForgetRequest) -> str:
    """Deterministic in (tenant, session, message ordinal, target), so a retry is idempotent."""
    return "forget_" + canonical_digest(
        {
            "tenant": tenant,
            "session_id": request.session_id,
            "message_ordinal": request.message_ordinal,
            "target": request.target,
        }
    )


def ledger_chunks(
    tenant: str, requests: Iterable[ForgetRequest], *, created_at: str
) -> list[Chunk]:
    """The ledger rows for ``requests``, one per distinct id."""
    rows: dict[str, Chunk] = {}
    for request in requests:
        row_id = ledger_id(tenant, request)
        rows[row_id] = Chunk(
            id=row_id,
            source="aml://session/" + session_digest(request.session_id),
            text=request.target,
            metadata={
                "record_type": FORGET_RECORD_TYPE,
                "kind": FORGET_RECORD_TYPE,
                "source_session_id": request.session_id,
                "session_digest": session_digest(request.session_id),
                "message_ordinal": request.message_ordinal,
                "target_text": request.target,
                "target_stems": sorted(content_stems(request.target)),
                "request_text": request.request_text,
                "preceding_text": request.preceding_text,
                "detector": request.detector,
                "created_at": created_at,
                "file": f"{row_id}.md",
            },
        )
    return list(rows.values())


@dataclass(frozen=True)
class ForgetEntry:
    """A ledger row as Search uses it."""

    entry_id: str
    session_id: str
    target: str
    stems: frozenset[str]
    request_shingles: frozenset[tuple[str, ...]]
    preceding_shingles: frozenset[tuple[str, ...]]


def ledger_entries(rows: Iterable[Chunk]) -> list[ForgetEntry]:
    """The usable ledger rows, in id order; a row of another type or without a target is skipped."""
    entries: list[ForgetEntry] = []
    for row in sorted(rows, key=lambda chunk: chunk.id):
        metadata = row.metadata
        target = metadata.get("target_text")
        if metadata.get("record_type") != FORGET_RECORD_TYPE or not isinstance(target, str):
            continue
        stems = content_stems(target)
        if not stems:
            continue
        entries.append(
            ForgetEntry(
                entry_id=row.id,
                session_id=str(metadata.get("source_session_id", "")),
                target=target,
                stems=stems,
                request_shingles=shingles(str(metadata.get("request_text", ""))),
                preceding_shingles=shingles(str(metadata.get("preceding_text", ""))),
            )
        )
    return entries


class _Text:
    """One item's text, analysed once and matched against every entry."""

    def __init__(self, text: str, session_id: str) -> None:
        self.session_id = session_id
        self.stems = content_stems(text)
        self.shingles = shingles(text)

    def states(self, entry: ForgetEntry) -> bool:
        """Rule R+text, the request window, and rule R, in that order."""
        if covers(entry.stems, self.stems, TARGET_MATCH_FRACTION):
            return True
        if self.session_id != entry.session_id or not self.session_id:
            return False
        if entry.request_shingles & self.shingles:
            return True
        return bool(entry.preceding_shingles & self.shingles) and covers(
            entry.stems, self.stems, PRECEDING_MATCH_FRACTION
        )


@dataclass
class ForgetOutcome:
    """What one Search's suppression did; counts only, never text."""

    mode: str = "off"
    #: Ledger rows of this tenant that Search read.
    requests_available: int = 0
    #: Ledger rows that changed at least one item.
    requests_applied: int = 0
    items_dropped: int = 0
    items_stubbed: int = 0
    items_annotated: int = 0
    failed: bool = False

    @property
    def items_changed(self) -> int:
        return self.items_dropped + self.items_stubbed + self.items_annotated


def drop_hits(
    hits: Sequence[ScoredChunk], entries: Sequence[ForgetEntry]
) -> tuple[list[ScoredChunk], set[str]]:
    """``hits`` without every item that states a forgotten target; also the entries that acted."""
    kept: list[ScoredChunk] = []
    applied: set[str] = set()
    for hit in hits:
        analysed = _Text(hit.chunk.text, str(hit.chunk.metadata.get("source_session_id", "")))
        acting = [entry.entry_id for entry in entries if analysed.states(entry)]
        if acting:
            applied.update(acting)
            continue
        kept.append(hit)
    return kept, applied


_SENTENCE = re.compile(r"[^.!?\n]*(?:[.!?]+[\"')\]]*|\n|$)")


def _stub_text(text: str, session_id: str, entries: Sequence[ForgetEntry]) -> tuple[str, set[str]]:
    analysed = _Text(text, session_id)
    acting = [entry for entry in entries if analysed.states(entry)]
    if not acting:
        return text, set()
    pieces: list[str] = []
    stubbed_any = False
    previous_stub = False
    for match in _SENTENCE.finditer(text):
        sentence = match.group(0)
        if not sentence:
            continue
        sentence_stems = content_stems(sentence)
        sentence_shingles = shingles(sentence)
        hide = any(
            covers(entry.stems, sentence_stems, PRECEDING_MATCH_FRACTION)
            or bool(entry.request_shingles & sentence_shingles)
            for entry in acting
        )
        if hide:
            stubbed_any = True
            if not previous_stub:
                lead = sentence[: len(sentence) - len(sentence.lstrip())]
                trailing = sentence[len(sentence.rstrip()) :]
                pieces.append(lead + STUB_SENTENCE + trailing)
            previous_stub = True
        else:
            pieces.append(sentence)
            previous_stub = False if sentence.strip() else previous_stub
    stubbed = "".join(pieces).rstrip() if stubbed_any else STUB_SENTENCE
    return stubbed, {entry.entry_id for entry in acting}


def stub_items(
    items: Sequence[SearchItem], entries: Sequence[ForgetEntry]
) -> tuple[list[SearchItem], set[str], int]:
    """Items whose forgotten detail is replaced by ``STUB_SENTENCE``; order and count unchanged."""
    output: list[SearchItem] = []
    applied: set[str] = set()
    changed = 0
    for item in items:
        if isinstance(item.content, str):
            text, acting = _stub_text(item.content, item.session_id, entries)
            content: Any = text
        else:
            parts: list[Any] = []
            acting = set()
            for part in item.content:
                if isinstance(part, TextContentPart):
                    part_text, part_acting = _stub_text(part.text, item.session_id, entries)
                    acting |= part_acting
                    parts.append(part.model_copy(update={"text": part_text}))
                else:
                    parts.append(part)
            content = parts
        if acting:
            applied |= acting
            changed += 1
            output.append(item.model_copy(update={"content": content}))
        else:
            output.append(item)
    return output, applied, changed


def annotate_items(
    items: Sequence[SearchItem], entries: Sequence[ForgetEntry]
) -> tuple[list[SearchItem], set[str], int]:
    """Items unchanged, with a note naming each forgotten target they state put before the first.

    The control arm: the reader is told what the user asked to forget, and still sees it.
    """
    stated: list[ForgetEntry] = []
    for item in items:
        text = (
            item.content
            if isinstance(item.content, str)
            else "\n".join(p.text for p in item.content if isinstance(p, TextContentPart))
        )
        analysed = _Text(text, item.session_id)
        for entry in entries:
            if entry not in stated and analysed.states(entry):
                stated.append(entry)
    if not items or not stated:
        return list(items), set(), 0
    note = ANNOTATION_PREFIX + "; ".join(f'"{entry.target}"' for entry in stated) + "."
    first = items[0]
    content: Any
    if isinstance(first.content, str):
        content = note + "\n\n" + first.content
    else:
        content = [TextContentPart(type="text", text=note), *first.content]
    return (
        [first.model_copy(update={"content": content}), *items[1:]],
        {entry.entry_id for entry in stated},
        1,
    )


def parse_mode(value: str) -> ForgetMode:
    """A mode name, or ValueError naming the variable and the accepted values."""
    mode = value.strip().lower()
    if mode not in FORGET_MODES:
        raise ValueError(f"{FORGET_ENV} must be one of {', '.join(FORGET_MODES)}, not {value!r}")
    return mode
