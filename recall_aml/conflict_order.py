"""K-2: put returned items about the same subject but from different days next to each other.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-relative-dates-and-conflict-adjacency.md

Relevance order can leave an old and a new statement about the same thing many ranks apart, and
nothing else in C9 ever sets them side by side (the compiler's ``supersedes`` field was used 0
times in 263,662 compiled rows). Inside the top ``window`` items only, this groups items whose
subject words overlap and whose dates differ, and moves each group to the rank of its best member,
newest first. It adds no text and drops no item; items past the window are untouched.

Every parameter is the pre-registered value and is fixed, not tuned: subject words are lowercased
alphabetic words of at least four letters, minus a stopword list and minus any word present in at
least half of the window's items (speaker names, a conversation's constant vocabulary); a link
needs Jaccard at least 0.35 and different UTC days; a group holds at most four items.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timezone
import re

from recall_aml.models import SearchItem, TextContentPart

WINDOW = 30
JACCARD_THRESHOLD = 0.35
MAX_GROUP = 4

_WORD = re.compile(r"[a-z]+")
_STOPWORDS = frozenset(
    """
    about above after again against also although always another anything around because been
    before being below between both came cannot come could does doing done down during each
    either else even ever every from further gets getting going gone good great have having here
    hers herself himself into itself just know like made make many maybe more most much must
    myself need never next nothing once only other ours ourselves over really same says should
    since some something still such sure take than that thats their theirs them themselves then
    there these they thing things think this those though through told took under until upon very
    want well went were what whatever when where whether which while will with within without
    would yeah your yours yourself yourselves
    """.split()
)


def _text(item: SearchItem) -> str:
    if isinstance(item.content, str):
        return item.content
    return " ".join(part.text for part in item.content if isinstance(part, TextContentPart))


def _words(item: SearchItem) -> set[str]:
    return {word for word in _WORD.findall(_text(item).lower()) if len(word) >= 4} - _STOPWORDS


def _day(item: SearchItem) -> date | None:
    if item.created_at is None:
        return None
    created = item.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created.astimezone(timezone.utc).date()


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def same_subject_adjacent(items: Sequence[SearchItem], *, window: int = WINDOW) -> list[SearchItem]:
    """Reorder only the top ``window`` items so same-subject, different-day items sit together."""
    head, tail = list(items[:window]), list(items[window:])
    count = len(head)
    if count < 2:
        return head + tail
    words = [_words(item) for item in head]
    frequency: dict[str, int] = {}
    for bag in words:
        for word in bag:
            frequency[word] = frequency.get(word, 0) + 1
    common = {word for word, seen in frequency.items() if seen * 2 >= count}
    subjects = [bag - common for bag in words]
    days = [_day(item) for item in head]

    weight: dict[tuple[int, int], float] = {}
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(count):
        for right in range(left + 1, count):
            if days[left] is None or days[right] is None or days[left] == days[right]:
                continue
            similarity = _jaccard(subjects[left], subjects[right])
            if similarity >= JACCARD_THRESHOLD:
                weight[(left, right)] = similarity
                parent[find(right)] = find(left)

    components: dict[int, list[int]] = {}
    for index in range(count):
        components.setdefault(find(index), []).append(index)
    group_of: dict[int, list[int]] = {}
    for members in components.values():
        if len(members) < 2:
            continue
        best = min(members)
        # Keep the best-ranked member plus the members most strongly linked to it, then by rank.
        others = sorted(
            (member for member in members if member != best),
            key=lambda member: (-weight.get((best, member), 0.0), member),
        )
        kept = [best, *others[: MAX_GROUP - 1]]
        if len(kept) < 2:
            continue
        newest_first = sorted(kept, key=lambda member: (-days[member].toordinal(), member))  # type: ignore[union-attr]
        for member in kept:
            group_of[member] = newest_first

    ordered: list[SearchItem] = []
    placed: set[int] = set()
    for index in range(count):
        if index in placed:
            continue
        group = group_of.get(index)
        if group is None:
            ordered.append(head[index])
            placed.add(index)
            continue
        for member in group:
            ordered.append(head[member])
            placed.add(member)
    return ordered + tail
