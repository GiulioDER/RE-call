"""Near-miss coverage: what a calibration's certificate does and does not vouch for. Report only.

`recall.wizard.queryset.generate_offline` builds the unanswerable class from subjects the corpus
never touches, on purpose (`recall/eval/synthetic.py`, 70-76): a gap class sharing the corpus's
vocabulary is not separable by score, and fitting a threshold to it fits noise. The price is that
the certificate says nothing about questions that ARE in the corpus's vocabulary and still have no
answer in it, and those are the questions an agent actually asks. Measured 2026-10-08: on one
corpus the offline probes certified at 0.960 while realistic near-misses separated at 0.699; on
another, 20 of 20 questions about invented subjects got a trusted hit about something else.

This module measures that gap at calibration time without changing the fit. Two probe classes are
built from the corpus, never from any labelled set:

* **unknown entity**: an answerable probe asked about a code the corpus never contains
  ("... in QK4817?"). The unknown-term gate (`recall.unknown_terms`) exists to reject these.
* **in-vocabulary**: one chunk's heading words joined to another chunk's distinctive term, kept
  only when no single chunk contains every content word. Every word is the corpus's own; no chunk
  is the source. Score alone rarely rejects these, and nothing else in RE-call does yet.

Each probe counts as rejected when the trust gate would not give it an `ok` hit: its best cosine is
below the threshold, or the unknown-term gate names one of its terms. The rates are stored with the
calibration and printed with its status. A certificate whose in-vocabulary rejection is below
`PROVISIONAL_BAR` is reported as provisional for in-vocabulary near-misses. The certified/
uncertified decision is untouched: stores that cannot run the unknown-term gate (PostgreSQL today)
must not lose a certification they hold because of a measurement they cannot act on.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from recall.embeddings import Embedder, embed_query
from recall.eval.vocab import word_tokens
from recall.unknown_terms import check_unknown_terms
from recall.wizard.queryset import (
    _ANSWERABLE_TEMPLATES,
    _STOPWORDS,
    _distinctive_terms,
    _document_frequency,
    _heading_of,
)

PROVISIONAL_BAR = 0.90
CLASSES = ("unknown_entity", "in_vocabulary")


def _invented_codes(corpus_words: set[str], rng: random.Random, count: int) -> list[str]:
    """Codes like QK4817 that occur nowhere in the corpus, deterministic for the seed."""
    letters = "BCDFGHJKLMNPQRSTVWXZ"
    codes: list[str] = []
    while len(codes) < count:
        code = rng.choice(letters) + rng.choice(letters) + f"{rng.randrange(1000, 10000)}"
        if code.lower() not in corpus_words and code not in codes:
            codes.append(code)
    return codes


def near_miss_probes(
    chunks: Sequence[str], answerable_queries: Sequence[str], *, per_class: int, seed: int = 0
) -> dict[str, list[str]]:
    """Up to `per_class` probes of each class, built from the corpus and its answerable probes."""
    rng = random.Random(seed)
    corpus_words = set(word_tokens(chunks))
    base = list(answerable_queries)[:per_class]
    codes = _invented_codes(corpus_words, rng, len(base))
    unknown = [f"{q.rstrip(' ?')} in {code}?" for q, code in zip(base, codes, strict=True)]

    df = _document_frequency(chunks)
    total = len(chunks)
    chunk_words = [set(word_tokens([c])) for c in chunks]
    order = rng.sample(range(total), total)
    in_vocab: list[str] = []
    seen: set[str] = set()
    for i in order:
        if len(in_vocab) >= per_class:
            break
        heading = _heading_of(chunks[i])
        if not heading:
            continue
        j = rng.randrange(total)
        if j == i:
            continue
        own = set(heading.split())
        term = next((t for t in _distinctive_terms(chunks[j], df, total, 3)
                     if t not in own and t not in chunk_words[i]), None)
        if term is None:
            continue
        subject = f"{heading} {term}"
        content = {w for w in word_tokens([subject]) if w not in _STOPWORDS}
        if subject in seen or any(content <= words for words in chunk_words):
            continue
        seen.add(subject)
        template = _ANSWERABLE_TEMPLATES[len(in_vocab) % len(_ANSWERABLE_TEMPLATES)]
        in_vocab.append(template.format(terms=subject))
    return {"unknown_entity": unknown, "in_vocabulary": in_vocab}


def near_miss_coverage(
    store: Any, embedder: Embedder, threshold: float, probes: dict[str, list[str]]
) -> dict[str, Any]:
    """How many probes of each class the trust gate rejects at `threshold`."""
    report: dict[str, Any] = {}
    gate = "unavailable"
    for name in CLASSES:
        queries = probes.get(name, [])
        rejected = 0
        for query in queries:
            check = check_unknown_terms(store, query)
            gate = check.status
            score = float(store.top_cosine(embed_query(embedder, query)))
            rejected += bool(check.unknown) or score < threshold
        n = len(queries)
        report[name] = {"n": n, "rejected": rejected, "rate": round(rejected / n, 3) if n else None}
    report["unknown_term_gate"] = gate
    in_vocab = report["in_vocabulary"]["rate"]
    report["provisional_in_vocabulary"] = in_vocab is not None and in_vocab < PROVISIONAL_BAR
    return report


def describe(report: dict[str, Any]) -> str:
    """One line for a status message."""
    parts = [
        f"{name.replace('_', '-')} {report[name]['rejected']}/{report[name]['n']} rejected"
        for name in CLASSES if report.get(name, {}).get("n")
    ]
    line = "near-miss coverage: " + (", ".join(parts) or "no probes could be built")
    if report.get("provisional_in_vocabulary"):
        line += "; provisional for in-vocabulary near-misses"
    if report.get("unknown_term_gate") != "checked":
        line += f"; unknown-term gate {report.get('unknown_term_gate')}"
    return line
