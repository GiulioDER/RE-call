"""W1: resolve a newly extracted ``subject|attribute`` key onto one the conversation already has.

Round two, 2026-09-28. The W1 key-link test sent the conversation's known keys in the extraction
prompt and the model still named one slot two ways: of 26 knowledge-update pairs with both sides
extracted, 5 shared a key, and the misses read like ``zoom call|date and time`` against ``zoom call
with the creative director|scheduled time``. A prompt cannot make a model reuse a string; code can.

The resolver is deterministic and online, in Add order, exactly as it would run at Add time. A new
key maps onto the existing canonical key with the best score among those passing two gates, else it
becomes a canonical key of its own:

- the **subjects** agree when one's word set contains the other's, or their Jaccard reaches
  ``subject_jaccard``;
- the **attributes** agree when their Jaccard reaches ``attribute_jaccard``, after an optional fixed
  synonym map folds time, quantity and duration words onto one slot word each.

A key is compared with every alias its candidate already holds, so a slot that has been named three
ways is matched by any of them. Nothing here calls a model or an embedder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re

STOPWORDS = frozenset(
    {
        "a", "about", "an", "and", "at", "by", "current", "for", "from", "her", "his", "in", "initial",
        "is", "its", "latest", "my", "new", "of", "old", "on", "original", "our", "recent", "revised",
        "s", "the", "their", "to", "updated", "user", "users", "with", "your",
    }
)
#: Words that name the same slot differently. Each folds onto one class word.
SLOT_SYNONYMS = {
    **dict.fromkeys(
        ("date", "day", "deadline", "due", "rescheduled", "schedule", "scheduled", "time", "timeline",
         "timing", "when"),
        "when",
    ),
    **dict.fromkeys(("amount", "count", "number", "quantity", "total"), "count"),
    **dict.fromkeys(("duration", "hour", "length", "minute", "spent"), "duration"),
}
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ResolverConfig:
    subject_jaccard: float = 0.5
    #: 0.0 turns the attribute gate off ("same subject, any attribute").
    attribute_jaccard: float = 0.5
    synonyms: bool = True

    @property
    def name(self) -> str:
        return f"ts{self.subject_jaccard:g}-ta{self.attribute_jaccard:g}-{'syn' if self.synonyms else 'nosyn'}"


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


@lru_cache(maxsize=65_536)
def key_words(part: str, *, synonyms: bool) -> frozenset[str]:
    """Content words of one half of a key, stemmed, stopwords dropped (before and after the stem,
    so "users" goes too), synonyms folded if asked."""
    words = (_stem(w) for w in _WORD.findall(part.casefold()) if w not in STOPWORDS)
    return frozenset(SLOT_SYNONYMS.get(w, w) if synonyms else w for w in words if w not in STOPWORDS)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def split_key(key: str) -> tuple[str, str]:
    subject, _, attribute = key.partition("|")
    return subject, attribute


@dataclass
class KeyResolver:
    """One conversation's (one tenant's) canonical keys and the raw keys mapped onto each."""

    config: ResolverConfig = field(default_factory=ResolverConfig)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    _canonical_of: dict[str, str] = field(default_factory=dict)

    def _score(self, key: str, alias: str) -> float | None:
        cfg = self.config
        (s1, a1), (s2, a2) = split_key(key), split_key(alias)
        subj1, subj2 = key_words(s1, synonyms=False), key_words(s2, synonyms=False)
        if not subj1 or not subj2:
            return None
        contained = subj1 <= subj2 or subj2 <= subj1
        subject = 1.0 if contained else jaccard(subj1, subj2)
        if not contained and subject < cfg.subject_jaccard:
            return None
        attr1, attr2 = key_words(a1, synonyms=cfg.synonyms), key_words(a2, synonyms=cfg.synonyms)
        attribute = jaccard(attr1, attr2)
        if cfg.attribute_jaccard > 0 and attribute < cfg.attribute_jaccard:
            return None
        return subject + attribute

    def resolve(self, key: str) -> str:
        """The canonical key for ``key``, registering it (as a new canonical key or an alias)."""
        if key in self._canonical_of:
            return self._canonical_of[key]
        best: tuple[float, str] | None = None
        for canonical, names in self.aliases.items():
            scores = [s for s in (self._score(key, name) for name in names) if s is not None]
            if scores and (best is None or max(scores) > best[0]):
                best = (max(scores), canonical)
        canonical = best[1] if best else key
        self.aliases.setdefault(canonical, []).append(key)
        self._canonical_of[key] = canonical
        return canonical

    def largest_cluster(self) -> int:
        return max((len(names) for names in self.aliases.values()), default=0)
