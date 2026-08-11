"""Measure the JUDGE, not the memory system — on cases whose correct verdict is known by
construction.

::

    python -m benchmarks.judge_quality --artifact benchmarks/results/recall_...json \\
        --models openai/gpt-4o-mini,openai/gpt-4o --sample 100 --out judge_quality.json
    python -m benchmarks.judge_quality --artifact ... --dry-run --out probes.json   # no LLM calls

Why this exists
---------------
The head-to-head is graded by an LLM judge, and everything known about that judge so far is one
opinion versus another. An independent audit (github.com/dial481/locomo-audit) measured
``openai/gpt-4o-mini`` accepting **62.81% of deliberately wrong-but-on-topic answers**;
`benchmarks.rejudge` then re-scored our own artifacts with ``openai/gpt-4o`` and the two judges
disagreed on ~10% of the judged records — asymmetrically by arm (94 RE-call answers flipped to
correct against 70 for Mem0). A judge choice that moves one arm more than the other is a judge
choice that picks the winner.

Which judge is RIGHT cannot be settled by that comparison, because neither side is ground truth.
This module manufactures ground truth instead. Every item it builds carries a verdict that is
known before any model is called, so a deviation is not a disagreement — it is a measured judge
error, attributable to the judge and to nothing else.

The three probes
----------------
Each is derived from a finished results artifact (`benchmarks.run`), using only its ``question``
and ``gold`` fields. No human labelling, no LLM-generated distractors, nothing about the memory
systems, and nothing that depends on which arm produced the file.

1. **VERBATIM GOLD** — the prediction IS the gold answer. A correct judge accepts 100% of these.
   This is the floor test and the strongest probe in the set precisely because it is not
   arguable: there is no phrasing question, no completeness question and no partial credit to
   debate. A rejection here is an unambiguous defect. ``verbatim_accept_rate`` should be 1.0.

2. **REWORDED GOLD** — the gold answer put through a transform that PROVABLY preserves the facts,
   by construction rather than by an LLM's opinion of it (see `TRANSFORMS`): reorder the items of
   a comma list, swap a date between ``14 July 2023`` and ``July 14, 2023``, or wrap the whole
   thing in a carrier phrase. A correct judge accepts all of them — the judge prompt explicitly
   says to grade the facts, not the wording, and cites the date case verbatim. Rejections are
   FALSE REJECTS: answers the benchmark would have marked wrong for being phrased differently.
   ``reworded_accept_rate`` should be 1.0.

3. **SWAPPED GOLD** — the question paired with the gold answer of a DIFFERENT question from the
   SAME conversation: on-topic, same speakers, same register, plausible, and wrong. A correct
   judge rejects all of them, so ``swapped_accept_rate`` IS the false-accept rate the audit
   measured, on our own data and our own judge prompt. It should be 0.0.

Why the swap filter is not optional
-----------------------------------
Two questions from one conversation can share an answer ("Who is Melanie's partner?" /
"Who does Melanie live with?"), and one gold can contain another ("7 May 2023" inside "She went
on 7 May 2023"). Pairing those would produce a prediction that is genuinely CORRECT while the
harness expects a rejection — and every such pair would be counted as a false accept, inflating
exactly the number this module exists to measure. `swap_reject_reason` therefore screens every
candidate pair on normalised text (case, punctuation and whitespace folded) and the rejects are
counted BY REASON into the output, so the screen is auditable rather than trusted.

Comparability between judges
----------------------------
The probe items are built ONCE, from a seeded sample, and every named model is then run over the
identical list. Sampling per model would make the columns incomparable: a difference between two
judges could be a difference between two samples. Every item and every per-item verdict is dumped
into the output for the same reason the results artifact dumps its per-question records — the
constructions are meant to be inspected, not believed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.llm import Completer, OpenRouterLLM
from benchmarks.pipeline import _json_safe_rate, judge_correct
from benchmarks.artifact_contract import load_published_artifact
from recall.eval.locomo import _rate

#: Sample size when ``--sample`` is not given. Deliberately small: the CLI spends real money, one
#: judge call per item per probe per model (a run of N is up to 3*N calls per model).
DEFAULT_SAMPLE = 25

#: Fixed so two invocations that name different models still build the identical item list. It is
#: written into the output, so a reader can rebuild the same probes offline with ``--dry-run``.
DEFAULT_SEED = 20260724

#: Probe names, in report order. Also the keys of the ``probes`` block and of every per-model
#: ``verdicts`` block.
VERBATIM = "verbatim"
REWORDED = "reworded"
SWAPPED = "swapped"
PROBE_NAMES = (VERBATIM, REWORDED, SWAPPED)


@dataclass(frozen=True)
class ProbeItem:
    """One constructed judge case, with the verdict a correct judge must return.

    ``expected`` is set by the CONSTRUCTION, never by a model and never by a human: True for the
    two gold-preserving probes, False for the swap. That is the whole design — the item is only
    worth judging because its answer was decided before the judge saw it.

    ``construction`` records HOW the prediction was made (``"verbatim"``, the transform name, or
    ``"swap:<partner question_id>"``), so a reader auditing the dump can reproduce the prediction
    from the artifact by hand.
    """

    item_id: str
    probe: str
    question_id: str
    question: str
    gold: str
    prediction: str
    expected: bool
    construction: str


# --------------------------------------------------------------------------------------------
# Probe 2: fact-preserving transforms
# --------------------------------------------------------------------------------------------

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)

#: ``14 July 2023`` / ``14th July, 2023`` -> day, month, year.
_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTHS}),?\s+(\d{{4}})\b", re.IGNORECASE)

#: ``July 14, 2023`` / ``July 14th 2023`` -> month, day, year.
_MDY = re.compile(rf"\b({_MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE)

#: A comma list whose items are sequenced rather than enumerated ("cooked, then ate") must not be
#: reordered: rotating it changes the claim instead of restating it. Checked on the first word of
#: every item, which is where a sequencing cue lands.
_ORDERING_CUES = frozenset(
    {
        "then",
        "next",
        "after",
        "afterwards",
        "afterward",
        "before",
        "later",
        "finally",
        "first",
        "firstly",
        "second",
        "secondly",
        "third",
        "thirdly",
        "followed",
        "subsequently",
        "eventually",
    }
)

#: A closing conjunction on the last item of a list ("apples, pears, and plums"). It marks the
#: END of the enumeration, not the item, so it is detached before the rotation and re-attached to
#: whichever item ends up last — otherwise the rotation strands it at the front ("and plums,
#: apples, pears"), which is a sentence no reader would write and no judge should have to parse.
_LIST_CONJUNCTIONS = frozenset({"and", "or"})

_DIGITS = re.compile(r"\d+")
_PUNCT = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_END = (".", "!", "?")

#: An item longer than this is a clause, not a list element; reordering clauses is not provably
#: meaning-preserving, so a gold that contains one is left to the carrier-phrase transform.
_MAX_LIST_ITEM_WORDS = 4


def has_date(text: str) -> bool:
    """True iff `text` contains a full-month calendar date in either supported order."""
    return bool(_DMY.search(text) or _MDY.search(text))


def reformat_date(gold: str) -> str | None:
    """`gold` with its dates written in the OTHER order, or None if it carries no date.

    Fact-preserving by construction: day, month and year are carried across unchanged, only their
    order and the comma move. An ordinal suffix ("14th") is dropped, which denotes the same day.

    One direction per call — if a gold contains day-first dates the whole string is converted to
    month-first, and vice versa — so the transform is idempotent in intent and never converts a
    date it has just written back again.
    """
    if _DMY.search(gold):
        return _DMY.sub(lambda m: f"{m.group(2)} {int(m.group(1))}, {m.group(3)}", gold)
    if _MDY.search(gold):
        return _MDY.sub(lambda m: f"{int(m.group(2))} {m.group(1)} {m.group(3)}", gold)
    return None


def reorder_list(gold: str) -> str | None:
    """`gold` with its comma-separated items rotated by one, or None if it is not a plain list.

    A rotation is a permutation, so the SET of items is identical and the judge prompt's stated
    rule ("correct only if it covers every one of them") is satisfied by construction. What makes
    that argument valid is the screening, and each guard exists for a gold that would otherwise be
    mangled into a different claim:

    * a date anywhere in the string — ``May 8, 2023`` splits on its own comma, and rotating gives
      ``2023, May 8``. The date transform is the right one for those and runs first anyway.
    * a purely numeric item — the year half of a date the month-name check did not catch.
    * an item longer than `_MAX_LIST_ITEM_WORDS` — a clause, not an element.
    * a sequencing cue on any item (`_ORDERING_CUES`) — an ordered narrative, where the order IS
      part of the fact.

    A trailing sentence mark is detached before the split and re-attached after the join, and a
    closing ``and``/``or`` is moved to whichever item ends up last (`_LIST_CONJUNCTIONS`), so the
    rotation cannot strand punctuation or a conjunction in the middle of the answer.
    """
    body = gold.strip()
    tail = ""
    if body.endswith(_SENTENCE_END):
        body, tail = body[:-1].rstrip(), body[-1]
    if has_date(body):
        return None

    parts = [part.strip() for part in body.split(",")]
    if len(parts) < 2 or any(not part for part in parts):
        return None

    conjunction = ""
    head, _, remainder = parts[-1].partition(" ")
    if head.casefold() in _LIST_CONJUNCTIONS:
        if not remainder.strip():
            return None  # a dangling "and" with nothing after it: malformed, not a list
        conjunction = f"{head} "
        parts[-1] = remainder.strip()

    for part in parts:
        words = part.split()
        if len(words) > _MAX_LIST_ITEM_WORDS or _DIGITS.fullmatch(part):
            return None
        if _PUNCT.sub("", words[0]).casefold() in _ORDERING_CUES:
            return None

    rotated = [parts[-1], *parts[:-1]]
    if rotated == parts:
        return None
    rotated[-1] = conjunction + rotated[-1]
    return ", ".join(rotated) + tail


def carrier_phrase(gold: str) -> str | None:
    """`gold` wrapped in a neutral carrier sentence — the always-applicable fallback.

    The gold text is embedded verbatim, so nothing about the facts can change; what changes is the
    verbosity, which the judge prompt explicitly says not to penalise. None only for a gold that
    is empty once stripped, which `eligible_records` has already excluded.
    """
    body = gold.strip()
    if not body:
        return None
    carried = f"The answer is {body}"
    return carried if body.endswith(_SENTENCE_END) else carried + "."


#: Tried in order; the first transform that applies AND actually changes the string wins. Date
#: first because it is the case the judge prompt calls out by name and therefore the one most
#: worth measuring; the carrier phrase is last because it applies to everything and would
#: otherwise starve the other two.
TRANSFORMS: tuple[tuple[str, Callable[[str], str | None]], ...] = (
    ("date_reformat", reformat_date),
    ("list_reorder", reorder_list),
    ("carrier_phrase", carrier_phrase),
)


def reword(gold: str) -> tuple[str, str] | None:
    """(transform name, reworded gold) for the first applicable transform, or None if none apply.

    A transform whose output equals its input is treated as INAPPLICABLE rather than applied: an
    item identical to the verbatim probe measures nothing new, and would quietly pad the reworded
    denominator with easier cases than the ones it claims to contain.
    """
    for name, transform in TRANSFORMS:
        out = transform(gold)
        if out is not None and out.strip() != gold.strip():
            return name, out
    return None


# --------------------------------------------------------------------------------------------
# Probe 3: swap screening
# --------------------------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Casefolded, punctuation-stripped, whitespace-collapsed text — for equivalence checks only.

    Never used to build a prediction, only to decide whether two golds are too close to be safely
    treated as a wrong answer for each other.
    """
    return _WHITESPACE.sub(" ", _PUNCT.sub(" ", text).casefold()).strip()


def _contains(outer: str, inner: str) -> bool:
    """True iff `inner` appears in `outer` as a whole run of words (both already normalised).

    Padded rather than a bare substring test so ``may`` does not "appear inside" ``maybe``: a
    spurious containment would discard a perfectly good swap pair, which costs coverage, while a
    missed one would keep a pair whose answer is actually right, which costs correctness.
    """
    return f" {inner} " in f" {outer} "


def swap_reject_reason(source: Mapping[str, str], mate: Mapping[str, str]) -> str | None:
    """Why `mate`'s gold may NOT be used as a wrong answer to `source`'s question, or None.

    Every rejection is a case where the "wrong" answer could be right, and would therefore have
    been counted as a judge false-accept when the judge was correct — the one failure mode that
    would make the headline false-accept rate an artefact of this module rather than of the judge.
    """
    source_gold = normalise(source["gold"])
    mate_gold = normalise(mate["gold"])
    if not source_gold or not mate_gold:
        return "empty_after_normalisation"
    if normalise(source["question"]) == normalise(mate["question"]):
        return "identical_question"
    if source_gold == mate_gold:
        return "identical_gold"
    if _contains(source_gold, mate_gold) or _contains(mate_gold, source_gold):
        return "gold_containment"
    return None


# --------------------------------------------------------------------------------------------
# Source records and sampling
# --------------------------------------------------------------------------------------------


def conversation_of(question_id: str) -> str:
    """The conversation id inside a `benchmarks.run` question_id (``{sample_id}:{index}``)."""
    return question_id.rsplit(":", 1)[0]


def _sort_key(record: Mapping[str, Any]) -> tuple[str, int, str]:
    """Stable ordering: conversation, then numeric position, then the raw id as a tie-break.

    Numeric rather than lexicographic on the index, so ``conv-26:9`` precedes ``conv-26:10`` and
    the "next question in the conversation" that probe 3 pairs with is the actual next one.
    """
    question_id = str(record["question_id"])
    _, _, tail = question_id.rpartition(":")
    return (conversation_of(question_id), int(tail) if tail.isdigit() else -1, question_id)


def eligible_records(doc: Mapping[str, Any]) -> list[dict[str, str]]:
    """The (question, gold) pairs in `doc` that can carry a probe, in a deterministic order.

    Adversarial records are excluded because they HAVE no gold answer — LOCOMO gives them a
    plausible distractor instead, and a probe built on one would be measuring the trap. Records
    whose question or gold is blank are excluded for the same reason `benchmarks.run` skips them:
    there is nothing to be correct about.

    The model's own answer is deliberately not read. Whether the arm answered, abstained or got it
    wrong is irrelevant to a probe built from the gold, so an ABSTAINED record is a perfectly good
    source — and excluding those would quietly bias the item pool toward the questions the system
    under test found easy.
    """
    records: list[dict[str, str]] = []
    for record in doc["outcomes"]:
        if bool(record.get("is_adversarial")):
            continue
        question = str(record.get("question") or "").strip()
        gold = str(record.get("gold") or "").strip()
        if not question or not gold:
            continue
        question_id = str(record["question_id"])
        records.append(
            {
                "question_id": question_id,
                "conversation": conversation_of(question_id),
                "question": question,
                "gold": gold,
            }
        )
    records.sort(key=_sort_key)
    return records


def select_sources(
    records: Sequence[Mapping[str, str]], sample: int, seed: int
) -> list[dict[str, str]]:
    """A deterministic `sample`-sized draw from `records` (all of them if there are fewer).

    Seeded and re-sorted, so the SAME items come out for every model, every run and every machine.
    A per-model draw would make the two judges' columns incomparable — a gap between them could
    then be a gap between two samples rather than between two judges, and nothing in the output
    would distinguish those.
    """
    chosen = list(records)
    if sample < len(chosen):
        chosen = random.Random(seed).sample(chosen, sample)
    chosen.sort(key=_sort_key)
    return [dict(record) for record in chosen]


# --------------------------------------------------------------------------------------------
# Probe construction
# --------------------------------------------------------------------------------------------


def build_verbatim(sources: Sequence[Mapping[str, str]]) -> list[ProbeItem]:
    """One item per source: the gold answer offered back as the prediction."""
    return [
        ProbeItem(
            item_id=f"{VERBATIM}:{source['question_id']}",
            probe=VERBATIM,
            question_id=source["question_id"],
            question=source["question"],
            gold=source["gold"],
            prediction=source["gold"],
            expected=True,
            construction="verbatim",
        )
        for source in sources
    ]


def build_reworded(
    sources: Sequence[Mapping[str, str]],
) -> tuple[list[ProbeItem], dict[str, Any]]:
    """Items whose prediction is a fact-preserving rewrite, plus the per-transform breakdown.

    A source no transform applies to yields no item, and is counted under ``no_transform`` — the
    reworded denominator is the number of items actually built, never the sample size.
    """
    items: list[ProbeItem] = []
    by_transform: dict[str, int] = {}
    skipped = 0
    for source in sources:
        reworded = reword(source["gold"])
        if reworded is None:
            skipped += 1
            continue
        name, prediction = reworded
        by_transform[name] = by_transform.get(name, 0) + 1
        items.append(
            ProbeItem(
                item_id=f"{REWORDED}:{source['question_id']}",
                probe=REWORDED,
                question_id=source["question_id"],
                question=source["question"],
                gold=source["gold"],
                prediction=prediction,
                expected=True,
                construction=name,
            )
        )
    report = {
        "n_items": len(items),
        "by_transform": dict(sorted(by_transform.items())),
        "no_transform": skipped,
    }
    return items, report


def build_swapped(
    sources: Sequence[Mapping[str, str]], pool: Sequence[Mapping[str, str]]
) -> tuple[list[ProbeItem], dict[str, Any]]:
    """Items pairing each source question with another question's gold, from the SAME conversation.

    Partners come from `pool` — every eligible record in the artifact — not from the sample, so a
    small ``--sample`` does not also shrink the set of available wrong answers. Scanning starts at
    the source's own position and wraps, so consecutive sources get different partners instead of
    every item in a conversation being paired with its first question.

    Candidates are screened by `swap_reject_reason` and every rejection is counted by reason. A
    source with no usable partner (a one-question conversation, or one where everything screened
    out) yields no item and is counted under ``no_usable_partner``.
    """
    by_conversation: dict[str, list[Mapping[str, str]]] = {}
    for record in pool:
        by_conversation.setdefault(record["conversation"], []).append(record)

    items: list[ProbeItem] = []
    rejected: dict[str, int] = {}
    for source in sources:
        mates = by_conversation.get(source["conversation"], [])
        offset = next(
            (i for i, m in enumerate(mates) if m["question_id"] == source["question_id"]),
            0,
        )
        ordered = [*mates[offset + 1 :], *mates[:offset]]

        chosen: Mapping[str, str] | None = None
        for mate in ordered:
            reason = swap_reject_reason(source, mate)
            if reason is None:
                chosen = mate
                break
            rejected[reason] = rejected.get(reason, 0) + 1
        if chosen is None:
            rejected["no_usable_partner"] = rejected.get("no_usable_partner", 0) + 1
            continue

        items.append(
            ProbeItem(
                item_id=f"{SWAPPED}:{source['question_id']}",
                probe=SWAPPED,
                question_id=source["question_id"],
                question=source["question"],
                gold=source["gold"],
                prediction=chosen["gold"],
                expected=False,
                construction=f"swap:{chosen['question_id']}",
            )
        )
    report = {
        "n_items": len(items),
        "rejected_candidates": {
            "total": sum(rejected.values()),
            "by_reason": dict(sorted(rejected.items())),
        },
    }
    return items, report


def build_probes(
    doc: Mapping[str, Any], sample: int, seed: int
) -> tuple[dict[str, list[ProbeItem]], dict[str, Any]]:
    """All three probes from one results artifact, plus the construction report.

    The report is published beside the rates because the rates are only readable with it: a
    reworded accept rate over 4 items and one over 90 are not the same claim, and the reason a
    probe is short (no applicable transform, no screenable partner) is itself a finding about the
    artifact rather than about the judge.
    """
    pool = eligible_records(doc)
    sources = select_sources(pool, sample, seed)
    verbatim = build_verbatim(sources)
    reworded, reworded_report = build_reworded(sources)
    swapped, swapped_report = build_swapped(sources, pool)

    probes = {VERBATIM: verbatim, REWORDED: reworded, SWAPPED: swapped}
    report = {
        "seed": seed,
        "sample_requested": sample,
        "eligible_records": len(pool),
        "sources_used": len(sources),
        "conversations": len({source["conversation"] for source in sources}),
        VERBATIM: {"n_items": len(verbatim)},
        REWORDED: reworded_report,
        SWAPPED: swapped_report,
    }
    return probes, report


# --------------------------------------------------------------------------------------------
# Judging
# --------------------------------------------------------------------------------------------


def run_probe(items: Sequence[ProbeItem], completer: Completer) -> list[dict[str, Any]]:
    """Ask `completer` to grade every item, through the SAME `judge_correct` the benchmark uses.

    Reused rather than reimplemented: a private copy of the judge prompt here would measure a
    judge the benchmark never runs, and would keep measuring the old one after the real prompt
    changed. ``judge_error`` is the whole output — the verdict differing from the one the
    construction guarantees.
    """
    verdicts: list[dict[str, Any]] = []
    for item in items:
        accepted = judge_correct(completer, item.question, item.gold, item.prediction)
        verdicts.append(
            {
                "item_id": item.item_id,
                "question_id": item.question_id,
                "construction": item.construction,
                "expected": item.expected,
                "accepted": accepted,
                "judge_error": accepted != item.expected,
            }
        )
    return verdicts


def measure_model(
    probes: Mapping[str, Sequence[ProbeItem]], completer: Completer
) -> dict[str, Any]:
    """One judge's scorecard: the three accept rates with n and Wilson 95% CI, plus every verdict.

    All three are ACCEPT rates — the fraction of items the judge said YES to — deliberately, so
    they are the same quantity measured on three different populations and can be read in one
    row. The targets differ: 1.0, 1.0 and 0.0, the last being the false-accept rate the audit
    reported. Rates come from `recall.eval.locomo._rate` (Wilson, not bootstrap: these samples are
    small and often degenerate) and are sanitised for JSON the same way `benchmarks.pipeline`
    sanitises its own.
    """
    result: dict[str, Any] = {}
    verdicts_by_probe: dict[str, list[dict[str, Any]]] = {}
    for name in PROBE_NAMES:
        items = probes.get(name, [])
        verdicts = run_probe(items, completer)
        verdicts_by_probe[name] = verdicts
        result[f"{name}_accept_rate"] = _json_safe_rate(
            _rate([bool(v["accepted"]) for v in verdicts])
        )
    result["judge_errors"] = {
        name: sum(1 for v in verdicts_by_probe[name] if v["judge_error"]) for name in PROBE_NAMES
    }
    result["verdicts"] = verdicts_by_probe
    return result


def judge_quality_document(
    doc: Mapping[str, Any],
    judges: Mapping[str, Completer],
    *,
    sample: int,
    seed: int,
    source: str,
) -> dict[str, Any]:
    """The full measurement: constructions, per-model scorecards, and the provenance to re-run it.

    `judges` is an ordered mapping of model name to completer, and every one of them is run over
    the SAME `probes` list — built once, before any judge is constructed. An empty mapping is
    legal and is what ``--dry-run`` uses: the probes are constructed and dumped, no judge is
    called, and nothing is spent.
    """
    probes, construction = build_probes(doc, sample, seed)
    models = {name: measure_model(probes, judge) for name, judge in judges.items()}
    return {
        "source": source,
        "source_arm": doc.get("arm"),
        "source_judge_model": (doc.get("config") or {}).get("model"),
        "judge_system_prompt": (doc.get("config") or {}).get("judge_system_prompt"),
        "construction": construction,
        "probes": {
            name: [asdict(item) for item in probes[name]] for name in PROBE_NAMES
        },
        "models": models,
    }


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _load_document(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = load_published_artifact(path)
    if "outcomes" not in doc:
        raise ValueError(f"{path} has no 'outcomes' array — it is not a results artifact")
    return doc


def _model_list(raw: str) -> list[str]:
    """argparse type for ``--models``: a comma-separated list, de-duplicated, order preserved."""
    names: list[str] = []
    for part in raw.split(","):
        name = part.strip()
        if name and name not in names:
            names.append(name)
    if not names:
        raise argparse.ArgumentTypeError("--models needs at least one model name")
    return names


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _print_summary(payload: Mapping[str, Any]) -> None:
    construction = payload["construction"]
    print(
        f"probes built from {construction['eligible_records']} eligible records "
        f"({construction['sources_used']} sampled, seed {construction['seed']}): "
        f"verbatim={construction[VERBATIM]['n_items']} "
        f"reworded={construction[REWORDED]['n_items']} "
        f"swapped={construction[SWAPPED]['n_items']}"
    )
    for name, result in payload["models"].items():
        parts = " ".join(
            f"{probe}={result[f'{probe}_accept_rate']['rate']}"
            f"(n={result[f'{probe}_accept_rate']['n']})"
            for probe in PROBE_NAMES
        )
        print(f"{name}: {parts}  want verbatim=1.0 reworded=1.0 swapped=0.0")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.judge_quality",
        description=(
            "Measure judge error on constructed cases whose correct verdict is known in advance."
        ),
    )
    parser.add_argument(
        "--artifact", type=Path, required=True, help="results artifact to build probes from"
    )
    parser.add_argument(
        "--models",
        type=_model_list,
        default=None,
        help="comma-separated judge models, e.g. openai/gpt-4o-mini,openai/gpt-4o",
    )
    parser.add_argument("--sample", type=_positive_int, default=DEFAULT_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, required=True, help="where to write the measurement")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and dump the probes without calling any judge (free, offline, no key needed)",
    )
    args = parser.parse_args(argv)

    if not args.artifact.exists():
        parser.error(f"{args.artifact} not found")
    # Never clobber: this file is meant to be published beside the artifact it measures, and a
    # second invocation writing over the first would destroy a paid-for measurement.
    if args.out.exists():
        parser.error(f"{args.out} already exists - refusing to overwrite a measurement")
    if not args.dry_run and not args.models:
        parser.error("--models is required unless --dry-run is given")

    judges: dict[str, Completer] = {}
    if not args.dry_run:
        # Imported here, as `benchmarks.rejudge` does, so the module stays importable (and
        # testable) without the bench extra installed.
        from benchmarks.run import validate_openrouter_key

        try:
            key = validate_openrouter_key(os.environ.get("OPENROUTER_API_KEY"))
        except ValueError as exc:
            parser.error(str(exc))
        judges = {
            name: OpenRouterLLM(model=name, api_key=key).complete for name in args.models or []
        }

    doc = _load_document(args.artifact)
    payload = judge_quality_document(
        doc, judges, sample=args.sample, seed=args.seed, source=str(args.artifact)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_summary(payload)
    print(f"judge-quality measurement -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
