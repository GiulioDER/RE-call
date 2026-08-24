"""Produce the labelled query set calibration needs, without asking a human to write it.

Calibration binds to a generation and requires at least `MIN_CALIBRATION_SAMPLES` answerable and
the same number of unanswerable labelled queries. Nothing in this repository produced them:
`recall setup` prompts for a path to a file the user must already have written, and
`recall/eval/labelled.py` states outright that a hand-labelled set "is the only way to measure the
thing that actually matters". That is the step that makes calibration too hard to be worth doing,
and it is the step a first-run wizard has to remove.

Two generators share one contract, so the caller does not branch on which ran:

* **offline** — derived from the corpus itself, no network, no key, usable when the answer to "is
  data security necessary" was yes.
* **LLM** — authored by a model (`recall/wizard/llm.py`), which produces more natural questions.

Whether the two are interchangeable is a measured question, not an assumed one; the prediction is
registered in `docs/preregistrations/2026-08-16-generated-calibration-query-sets.md` and says they
are not.

**The one design constraint that is not negotiable** comes from a measurement recorded at
`recall/eval/synthetic.py:69-75`: an unanswerable query built by perturbing an answerable one
(suffixing a nonsense token) produced a set that was *not separable at all* — median top cosine
0.830 against answerable 0.923, with 0% of it below the weakest answerable query. Genuinely
off-topic questions sit at median 0.570 with 78% below that floor. Unanswerable queries therefore
come from a domain the corpus does not touch, and that disjointness is checked against the corpus
rather than assumed.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from recall.calibration import MIN_CALIBRATION_SAMPLES
from recall.eval.synthetic import _OFFTOPIC_SUBJECTS, _OFFTOPIC_TEMPLATES
from recall.eval.vocab import word_tokens
from recall.index import chunk_text
from recall.lint import DEFAULT_GLOB
from recall.observability import get_logger
from recall.errors import RecallError

_log = get_logger("wizard.queryset")

#: The certification floor, imported rather than restated. A local `20` here could drift away from
#: the number `Calibration.certified` actually tests, and the failure would be a set that this
#: module called adequate and certification rejected after a full retrieval run.
MIN_PER_CLASS = MIN_CALIBRATION_SAMPLES

#: Twice the floor. The Hanley-McNeil interval is wide at n=20, and it is the interval's LOWER
#: bound that certification tests, so a genuinely separable set can still be refused for thinness.
#: Headroom here is cheaper than a failed install.
DEFAULT_PER_CLASS = 2 * MIN_CALIBRATION_SAMPLES

#: Keys `canonical_query_set` keeps. Everything else is dropped there, so carrying extras produces
#: a file whose digest looks stable while its inputs are not.
_KEPT_KEYS = ("query", "answerable", "relevant_ids")

#: Questions asked of a real chunk. Kept short and varied so the shared template words are a small
#: fraction of each query; a long shared stem would make every answerable query resemble every
#: other one more than it resembles its own chunk.
_ANSWERABLE_TEMPLATES = (
    "what does this project say about {terms}",
    "how is {terms} handled here",
    "why was {terms} decided this way",
    "what is the behaviour of {terms}",
    "where is {terms} described",
)

#: Words too common in technical prose to identify a chunk. Deliberately short: the distinctive
#: term selection below is a document-frequency test, which already removes anything that appears
#: across the corpus, so this only has to catch words frequent WITHIN one chunk.
_STOPWORDS = frozenset(
    """
    the a an and or but if then than that this these those is are was were be been being
    it its as at by for from in into of on to with without within over under after before
    do does did done can could should would may might must will shall not no yes any all
    each every some such only also more most less least other another same very much many
    when where which who whom whose what how why while during about against between both
    """.split()
)

_MIN_TERM_LENGTH = 4
_TERMS_PER_QUERY = 3

#: A markdown ATX heading. Setext headings are not matched: `chunk_text` splits on blank lines, so
#: an underlined heading and its rule usually survive together, and the rule alone is not a topic.
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")

#: Inline code inside a heading (`recall_search`) is a symbol, not prose, and is dropped before
#: the heading's words are taken.
_CODE_SPAN = re.compile(r"`[^`]*`")


class QuerySetError(ValueError, RecallError):
    """A query set that certification would refuse, refused earlier and with a reason."""


def canonicalize(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce `entries` to the exact shape `canonical_query_set` accepts, or explain why not.

    Shape only. Class balance is `require_balance`, and the two are separate because they answer
    different questions and fail for different reasons: this one is about whether each record is
    well formed, that one is about whether there are enough of them. Fused, neither could be
    exercised without satisfying the other.

    Three things are enforced here because v2's loader raises on the WHOLE file rather than
    skipping an entry, so one bad record costs the run:

    * `trust` entries are dropped. Both `recall/eval/queries.json` and `synthetic.generate` emit
      them, they carry no `answerable` key, and the legacy `recall calibrate` path filters them
      (`recall/setup.py:1146`) while v2 refuses first.
    * `answerable` must be a real bool. `"yes"` is truthy in Python and rejected by the loader.
    * duplicates are removed, compared case- and whitespace-insensitively, because the loader
      refuses a duplicate outright.
    """
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            # Raised, not skipped. Skipping turned a JSON array of strings into "0 answerable and
            # 0 unanswerable", which reads as a corpus problem and sends the reader to fix the
            # wrong thing.
            raise QuerySetError(f"entry {position} is {type(entry).__name__}, not an object")
        # A `trust` entry is dropped only when it carries no usable label. Dropping on the flag
        # alone discarded a well-formed labelled entry that happened to also be marked, silently,
        # and the loss then showed up as a class-count shortfall blamed on the corpus.
        if entry.get("trust") and not isinstance(entry.get("answerable"), bool):
            continue
        query = entry.get("query")
        if not isinstance(query, str) or not query.strip():
            raise QuerySetError(f"every entry needs a non-empty query, got {query!r}")
        answerable = entry.get("answerable")
        if not isinstance(answerable, bool):
            raise QuerySetError(
                f"'answerable' must be a boolean, got {answerable!r} for {query!r}. "
                "A truthy string passes here and is refused by the calibration loader."
            )
        record: dict[str, Any] = {"query": query, "answerable": answerable}
        # `relevant_ids` is type-checked here rather than passed through. `canonical_query_set`
        # requires a list of strings and aborts the WHOLE file on a bad one, which is the failure
        # this function exists to prevent; passing it through unvalidated meant a `null` from the
        # LLM generator (the caller this module is written to feed) produced a set that looked
        # clean here and was refused wholesale downstream.
        if entry.get("relevant_ids") is not None and "relevant_ids" in entry:
            relevant = entry["relevant_ids"]
            if not isinstance(relevant, list) or not all(isinstance(v, str) for v in relevant):
                raise QuerySetError(
                    f"'relevant_ids' must be a list of strings, got {relevant!r} for {query!r}"
                )
            record["relevant_ids"] = relevant

        fingerprint = " ".join(query.lower().split())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        kept.append(record)
    return kept


def require_balance(
    entries: Sequence[dict[str, Any]],
    *,
    min_per_class: int = MIN_PER_CLASS,
) -> list[dict[str, Any]]:
    """Return `entries` if both classes clear `min_per_class`, else say which side is thin.

    Both counts are named. Knowing WHICH class is short is what tells the caller whether to widen
    the corpus or the off-topic pool, and it is the difference between an actionable message and
    "calibration failed". Refused here rather than at certification, which only finds out after a
    full retrieval run over every query.
    """
    answerable = sum(1 for e in entries if e.get("answerable"))
    unanswerable = len(entries) - answerable
    if answerable < min_per_class or unanswerable < min_per_class:
        raise QuerySetError(
            f"calibration needs at least {min_per_class} of each class; this set has "
            f"{answerable} answerable and {unanswerable} unanswerable. "
            "Certification would refuse it after a full retrieval run."
        )
    return list(entries)


def prepare_for_calibration(
    entries: Iterable[dict[str, Any]],
    *,
    min_per_class: int = MIN_PER_CLASS,
) -> list[dict[str, Any]]:
    """`canonicalize` then `require_balance`: everything a set needs before it is written out."""
    return require_balance(canonicalize(entries), min_per_class=min_per_class)


def chunks_from_directory(
    root: str | Path,
    glob: str = DEFAULT_GLOB,
    chunker: "Callable[[str], list[str]] | None" = None,
) -> list[str]:
    """Every chunk of the corpus at `root`, chunked exactly as indexing will chunk it.

    `chunker` defaults to `chunk_text` and MUST be the callable the generation will actually be
    built with. A caller indexing code with `chunk_code` while generating queries from `chunk_text`
    breaks the invariant this docstring states two paragraphs down, and it breaks it invisibly:
    measured on this repository's own `pipeline.py`, the two produce 20 chunks against 8 with no
    exact string in common, so every "answerable" query would be generated from text that is not in
    the index it is about to be measured against.

    `chunk_text` rather than a local splitter: a query generated from text that was never a chunk
    cannot retrieve that chunk, so the two must be the same function and not merely similar ones.
    """
    from recall.index import candidate_files

    root_path = Path(root)
    if not root_path.exists():
        # Checked first so the message names the real problem. `candidate_files` reaches its
        # non-directory branch for any path that fails `is_dir()`, including one that does not
        # exist, and then blames the glob for a missing directory.
        raise QuerySetError(f"{str(root_path)!r} does not exist")
    try:
        files = candidate_files(root_path, glob)
    except ValueError as exc:
        raise QuerySetError(str(exc)) from exc

    chunks: list[str] = []
    split = chunker if chunker is not None else chunk_text
    unreadable: list[str] = []
    vanished = 0
    for file in files:
        try:
            body = file.read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, NotADirectoryError) as exc:
            # Raced away between the walk and the read, so it is not part of the corpus. Skipped
            # rather than fatal, matching `index_path` (`recall/index.py:575`), because a wizard
            # is pointed at a live user directory where an editor swap file or a cloud-sync
            # placeholder can vanish mid-run, and aborting the whole install over one is worse
            # than generating questions from the rest.
            #
            # Logged AND counted, which is the rest of what `index_path` does and what a bare
            # `continue` here dropped. A half-vanished corpus that generated questions from the
            # remainder with no signal is the same silence this function refuses elsewhere.
            _log.warning("skipping %s: it vanished before it could be read (%s)", file, exc)
            vanished += 1
            continue
        except OSError as exc:
            # Everything else IS collected. A permission error or an I/O error means the file is
            # still there and still part of the corpus, so skipping it would let a partly
            # unreadable corpus shrink in silence and the sampling below would run on a corpus the
            # caller never learned was truncated.
            unreadable.append(f"{file} ({type(exc).__name__})")
            continue
        chunks.extend(split(body))

    if unreadable:
        shown = ", ".join(unreadable[:3])
        more = f" and {len(unreadable) - 3} more" if len(unreadable) > 3 else ""
        raise QuerySetError(
            f"{len(unreadable)} of {len(files)} file(s) under {str(root_path)!r} could not be "
            f"read: {shown}{more}. A corpus that is partly unreadable would silently generate "
            "questions about only the readable part."
        )
    if not chunks:
        # The vanished count is named here, because without it an unmounted share or an offline
        # cloud-sync folder — where every file is found by the walk and gone by the read — reports
        # as a glob mismatch and sends the reader to fix a pattern that was correct.
        raced = (
            f" ({vanished} of {len(files)} file(s) disappeared while being read)" if vanished else ""
        )
        raise QuerySetError(
            f"no chunks under {str(root_path.resolve())!r} matching {glob!r}{raced}, so there is "
            "nothing to build answerable questions from."
        )
    return chunks


def offtopic_subjects_absent_from(texts: Sequence[str]) -> list[str]:
    """The off-topic subjects this corpus does not touch, in declaration order.

    Filtered against the corpus rather than trusted, because the shipped list is off-topic for a
    software corpus and squarely ON topic for a corpus about glaciers or beekeeping. A gap class
    that overlaps the corpus is the exact failure `synthetic.py` measured: shared vocabulary
    destroys separability, and the resulting calibration is fitted to noise.
    """
    corpus_words = set(word_tokens(texts))
    surviving: list[str] = []
    for subject in _OFFTOPIC_SUBJECTS:
        content = [w for w in word_tokens([subject]) if len(w) > 3]
        if not corpus_words.intersection(content):
            surviving.append(subject)
    return surviving


def _pool_collisions(texts: Sequence[str]) -> set[str]:
    """The pool's own content words that this corpus contains, which is what the refusal must name.

    A count of surviving subjects says how bad it is. These words say WHY, and the two commonest
    causes are told apart by reading them: a corpus genuinely about food and music, or a corpus
    that has ingested the subject list itself.
    """
    corpus_words = set(word_tokens(texts))
    return {
        word
        for subject in _OFFTOPIC_SUBJECTS
        for word in word_tokens([subject])
        if len(word) > 3 and word in corpus_words
    }


def _document_frequency(chunks: Sequence[str]) -> Counter[str]:
    df: Counter[str] = Counter()
    for chunk in chunks:
        df.update(set(word_tokens([chunk])))
    return df


def _heading_of(chunk: str) -> str:
    """The first markdown heading in `chunk`, cleaned, or "".

    A heading is the one place a document states its own topic, so it beats anything inferred from
    word statistics. Without it the generator produced questions like "why was doubly
    sources_not_found fell decided this way": grammatical scaffolding wrapped around a bag of rare
    tokens, which is not what anybody would ask and does not describe the chunk.
    """
    for line in chunk.splitlines():
        match = _HEADING.match(line)
        if match:
            text = _CODE_SPAN.sub(" ", match.group(1))
            words = [w for w in word_tokens([text]) if w not in _STOPWORDS and _is_prose(w)]
            if words:
                return " ".join(words[:6])
    return ""


def _is_prose(word: str) -> bool:
    """Whether `word` reads as prose rather than as a symbol a reader would never say aloud.

    Rare-token ranking surfaced `_tenant_isolation`, `test_parity_reports_a_missing_shadow…`,
    `promo_`, `uuid8` and `id_rsa` — all genuinely rare, none of them a topic. A question built
    from identifiers retrieves the one chunk that happens to contain that symbol, which looks like
    excellent separability while measuring string matching rather than meaning.
    """
    if len(word) < _MIN_TERM_LENGTH or len(word) > 24:
        return False
    if "_" in word or any(ch.isdigit() for ch in word):
        return False
    return word.isalpha()


def _distinctive_terms(chunk: str, df: Counter[str], total: int, count: int) -> list[str]:
    """The words that best identify `chunk`, most distinctive first.

    Ranked by `tf * log(total / df)`, not by document frequency alone. Pure rarity ranked a word
    occurring ONCE in one chunk above a word occurring five times in that chunk and twice
    elsewhere, and the second is far more likely to be what the chunk is about. Term frequency is
    what distinguishes a topic from a passing mention or a typo.
    """
    from math import log

    counts = Counter(w for w in word_tokens([chunk]) if w not in _STOPWORDS and _is_prose(w))
    if not counts:
        return []
    scored = sorted(
        counts.items(),
        key=lambda item: (-(item[1] * log(total / max(1, df.get(item[0], 1)))), item[0]),
    )
    return [word for word, _ in scored[:count]]


def _subject_of(chunk: str, df: Counter[str], total: int) -> str:
    """What a question about `chunk` should ask about: its heading, sharpened by its own terms."""
    heading = _heading_of(chunk)
    terms = _distinctive_terms(chunk, df, total, _TERMS_PER_QUERY)
    if heading:
        # The heading names the topic; one distinctive term separates this chunk from its
        # siblings, since a long document's chunks can share a heading.
        extra = [t for t in terms if t not in heading.split()][:1]
        return " ".join([heading, *extra])
    return " ".join(terms)


def generate_offline(
    chunks: Sequence[str],
    *,
    per_class: int = DEFAULT_PER_CLASS,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """A labelled set derived from `chunks` alone. No network, no key, no model.

    Deterministic for a `seed`, because the registered measurement re-runs it and two runs that
    disagreed would make the comparison meaningless.

    Both refusals below are refusals rather than quiet narrowing. Reusing a chunk or a subject
    emits duplicate queries, which the loader rejects, and which `synthetic.generate` refuses for
    the further reason that duplicate gap queries understate the variance of any rate measured on
    them.
    """
    if per_class < 1:
        raise QuerySetError("per_class must be at least 1")
    if len(chunks) < per_class:
        raise QuerySetError(
            f"{per_class} answerable questions need {per_class} distinct chunks, and this corpus "
            f"has {len(chunks)}. Index more documents, or lower per_class (the certification "
            f"floor is {MIN_PER_CLASS})."
        )

    subjects = offtopic_subjects_absent_from(chunks)
    capacity = len(subjects) * len(_OFFTOPIC_TEMPLATES)
    if capacity < per_class:
        # The colliding WORDS, not just the count. Without them this refusal said the corpus
        # overlaps the pool and told the operator to supply a domain-specific subject list, which
        # was a confident diagnosis and, in the case that actually occurred, the wrong one: the
        # corpus was a Python tree that happened to contain recall's own source. Naming the words
        # makes a corpus that has swallowed a copy of the pool distinguishable at a glance from a
        # corpus genuinely about food and music, and those two want opposite fixes.
        #
        # ⚠️ Do NOT quote example subjects here. An earlier version of this comment named the three
        # words it had collided on, which put them back into every code corpus rooted at this
        # repository and disqualified those three subjects for every user. Measured: survivors fell
        # from 12 to 11 and the words traced to this file. The pool lives in
        # `recall/eval/offtopic_subjects.json` precisely so it is not source; a comment quoting it
        # is source again.
        collisions = _pool_collisions(chunks)
        sample = ", ".join(sorted(collisions)[:8]) or "none identified"
        raise QuerySetError(
            f"only {len(subjects)} of {len(_OFFTOPIC_SUBJECTS)} off-topic subjects are disjoint "
            f"from this corpus, giving {capacity} distinct gap questions against the "
            f"{per_class} needed. A gap class drawn from the rest would share vocabulary with the "
            f"corpus and would not be separable. The corpus contains these pool words: {sample}. "
            "If they read as an unrelated subject list rather than as your domain, the corpus "
            "probably includes a copy of the pool itself (recall's own source does), and the fix "
            "is to exclude that path rather than to supply a new subject list."
        )

    rng = random.Random(seed)
    df = _document_frequency(chunks)
    total = len(chunks)

    # Sampled without replacement rather than taking the first `per_class`: the head of a corpus
    # is its front matter, and a set of questions all drawn from README-shaped text would measure
    # one region rather than the corpus.
    #
    # OVERSAMPLED, because some chunks yield no subject (all symbols, or all stopwords) and some
    # yield a subject another chunk already used. Drawing exactly `per_class` and refusing on the
    # first shortfall failed 2.3% of seeds at the default on this repository's own `docs/`, and
    # told the reader their 1811-chunk corpus was "too repetitive" while ~1800 usable chunks sat
    # unasked.
    pool_size = min(len(chunks), max(per_class * 4, per_class + 32))
    pool = rng.sample(range(len(chunks)), pool_size)

    entries: list[dict[str, Any]] = []
    used_subjects: set[str] = set()
    # Iterated in SAMPLED order, not sorted order. Sorting the pool and stopping at `per_class`
    # takes only its lowest-indexed quarter, which is the head-of-corpus bias the sample exists to
    # avoid — and oversampling made it worse rather than better, because the tail of the pool is
    # never reached. Measured on this repository's `docs/` at the default: sorted order drew all
    # 40 questions from 3 of 51 files, sampled order from 21 of 51.
    for index in pool:
        if len(entries) == per_class:
            break
        subject = _subject_of(chunks[index], df, total)
        if not subject:
            continue
        # Deduplicated on the SUBJECT, not on the rendered query. Two chunks with one subject
        # under different templates render as different strings and both survived, giving two
        # answerable queries with the same ground truth — a correlated pair, which is the variance
        # problem `synthetic.generate` refuses a repeated subject to avoid.
        if subject in used_subjects:
            continue
        used_subjects.add(subject)
        # Template chosen by ACCEPTED count, so skipping a chunk does not skip a template and
        # leave the five unevenly used.
        template = _ANSWERABLE_TEMPLATES[len(entries) % len(_ANSWERABLE_TEMPLATES)]
        entries.append({"query": template.format(terms=subject), "answerable": True})

    if len(entries) < per_class:
        raise QuerySetError(
            f"only {len(entries)} distinct subjects came from {pool_size} sampled chunks, against "
            f"the {per_class} needed. The corpus is too repetitive, or too much of it is code and "
            "identifiers, for questions generated from it to identify one chunk."
        )

    for i in range(per_class):
        subject = subjects[i % len(subjects)]
        template = _OFFTOPIC_TEMPLATES[(i // len(subjects)) % len(_OFFTOPIC_TEMPLATES)]
        entries.append({"query": template.format(s=subject), "answerable": False})

    return prepare_for_calibration(entries, min_per_class=per_class)
