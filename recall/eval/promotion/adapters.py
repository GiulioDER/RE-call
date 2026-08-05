"""Corpus adapters: every supported benchmark, reduced to frozen questions and question records.

An adapter's whole job is to answer two questions in a way the rest of the harness does not have
to know the corpus to understand:

``freeze()``  what questions exist, what they are called, and what a correct retrieval must
              surface. Runs BEFORE any candidate result exists.
``queries()`` the query text for each frozen question, so the runner can verify it still hashes
              to the frozen `input_hash` before spending an embedding on it.

Nothing here executes retrieval. Scoring lives in `run.py`, which is corpus-agnostic by
construction: it takes frozen questions plus a search callable and emits `QuestionRecord`s. That
split is what stops five corpora from growing five subtly different definitions of hit@5.

Availability is explicit, never silent. Every adapter reports `available()` and, when it is not,
`unavailable_reason()` says which file is missing. A corpus that quietly contributes zero
questions would shrink the gate's evidence base while the run still printed success — the same
failure shape as a benchmark that drops a corpus whose download 404'd.

Corpus coverage, and the honest state of each:

=================  =======================================================================
`labelled`         `recall/eval/corpus/` + `recall/eval/queries.json`. In-repo, 22 documents,
                   answerable and unanswerable queries. This is the small corpus the harness
                   is proven on end to end.
`peps`             `recall/eval/peps_questions.json`, which was in the tree and referenced by
                   NO code before this module. Now wired. Needs an external PEP corpus
                   checkout to run.
`locomo`           `locomo10.json` in the working directory, the layout
                   `recall/eval/locomo.py` already reads.
`ladder`           the frozen manifests under `results/ladder/`. Its instances are already
                   frozen, so `freeze()` re-expresses rather than re-derives them.
`longmemeval`      a fresh post-fix run; needs the LongMemEval instance file, which is not in
                   the tree.
`mtrag`            via the runner committed in session 1 (`benchmarks/mtrag/run.py`), for the
                   part of its format that is compatible: per-task queries with qrels. Its
                   nDCG/Recall scoring is NOT reused — see `MtragAdapter`.
=================  =======================================================================
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from recall.eval.promotion.manifest import FrozenQuestion, file_digest, question_input_hash


class CorpusUnavailable(RuntimeError):
    """The adapter's source data is not present, named rather than skipped."""


@dataclass(frozen=True)
class SourceQuestion:
    """One question as its source corpus states it, before freezing."""

    question_id: str
    query: str
    #: Empty means unanswerable. See `FrozenQuestion.expected_relevance_labels`.
    relevance_labels: tuple[str, ...]


class CorpusAdapter(ABC):
    """One benchmark corpus, expressed as frozen questions."""

    #: The `corpus` value stamped on every frozen question and record. It is a grouping key in the
    #: gate's stratified bootstrap, so it must be stable across sessions.
    name: str

    #: Which id space this corpus's labels live in — a key of `search.LABEL_KEYS`. Declared on
    #: every adapter with NO inherited default, because a wrong one scores the whole corpus as a
    #: total miss and reports a successful run. `label_kind()` below refuses an adapter that
    #: forgot to declare it.
    label_kind: str

    @abstractmethod
    def source_paths(self) -> tuple[Path, ...]:
        """Files whose digests go into the manifest's provenance block."""

    @abstractmethod
    def _questions(self) -> Iterator[SourceQuestion]:
        """Yield every question this corpus contributes."""

    def available(self) -> bool:
        return all(path.exists() for path in self.source_paths())

    def unavailable_reason(self) -> str | None:
        missing = [str(path) for path in self.source_paths() if not path.exists()]
        if not missing:
            return None
        return f"{self.name}: source data not present: {missing}"

    def _require(self) -> None:
        reason = self.unavailable_reason()
        if reason is not None:
            raise CorpusUnavailable(reason)

    def corpus_hashes(self) -> dict[str, str]:
        self._require()
        return {
            f"{self.name}:{path.name}": file_digest(path) for path in self.source_paths()
        }

    def corpus_of(self, question: SourceQuestion) -> str:
        """The stratification key for one question. One corpus per adapter by default.

        `MtragAdapter` overrides it to split its domains, because the gate's macro average is the
        unweighted mean over corpora and MT-RAG's domains differ in judged-query count by nearly
        a factor of two.
        """
        return self.name

    def freeze(self) -> tuple[FrozenQuestion, ...]:
        self._require()
        frozen: list[FrozenQuestion] = []
        seen: set[str] = set()
        for question in self._questions():
            if question.question_id in seen:
                raise ValueError(
                    f"{self.name}: duplicate question id {question.question_id!r}"
                )
            seen.add(question.question_id)
            corpus = self.corpus_of(question)
            frozen.append(
                FrozenQuestion(
                    question_id=question.question_id,
                    corpus=corpus,
                    input_hash=question_input_hash(
                        question_id=question.question_id,
                        corpus=corpus,
                        query=question.query,
                        expected_relevance_labels=question.relevance_labels,
                    ),
                    expected_relevance_labels=tuple(question.relevance_labels),
                )
            )
        if not frozen:
            raise CorpusUnavailable(
                f"{self.name}: source data is present but yielded no questions. A corpus that "
                f"contributes nothing must say so rather than shrink the evidence base silently."
            )
        return tuple(frozen)

    def queries(self) -> Mapping[str, str]:
        """question_id -> query text, for the runner's `FrozenQuestion.verify` check."""
        self._require()
        return {question.question_id: question.query for question in self._questions()}


class LabelledAdapter(CorpusAdapter):
    """The in-repo 22-document corpus and its labelled query set.

    This is the corpus the harness is proven on end to end, because it is the only one whose
    documents ship in the repository. Its labels are `"<file>:<ord>"`, which is the key
    `recall/eval/harness.py` derives from a hit's metadata and NOT the store's chunk id: chunk
    ids are content hashes. The first end-to-end run of this harness compared the two and scored
    0 of 14 answerable questions, which is where `label_kind` and the vacuity refusal in
    `aggregate.build_outcomes` both come from.
    """

    name = "labelled"
    #: `queries.json` labels are "<file>:<ord>" — `recall/eval/harness.py`'s own key. NOT the
    #: chunk id, which is a content hash: the first run of this harness compared the two and
    #: scored 0/14.
    label_kind = "file_ord"

    def __init__(self, queries_path: Path | None = None, corpus_dir: Path | None = None) -> None:
        root = Path(__file__).resolve().parent.parent
        self.queries_path = queries_path or root / "queries.json"
        self.corpus_dir = corpus_dir or root / "corpus"

    def source_paths(self) -> tuple[Path, ...]:
        return (self.queries_path,)

    def _questions(self) -> Iterator[SourceQuestion]:
        payload = json.loads(self.queries_path.read_text(encoding="utf-8"))
        for entry in payload:
            answerable = bool(entry.get("answerable", True))
            labels = tuple(entry.get("relevant_ids", ())) if answerable else ()
            yield SourceQuestion(
                question_id=str(entry["id"]), query=entry["query"], relevance_labels=labels
            )


class PepsAdapter(CorpusAdapter):
    """`recall/eval/peps_questions.json` — in the tree since it was written, referenced by no code.

    Its labels are FILE names (`pep-0690.rst`), not chunk ids. The runner therefore compares
    against the `source` of each retrieved chunk rather than its id, which is why `label_kind` is
    part of an adapter's contract instead of being assumed uniform: a harness that silently
    compared file names against chunk ids would score every question a miss and report a corpus
    that simply does not work.
    """

    name = "peps"
    #: Labels are file names ("pep-0690.rst").
    label_kind = "source"

    def __init__(self, questions_path: Path | None = None) -> None:
        self.questions_path = (
            questions_path or Path(__file__).resolve().parent.parent / "peps_questions.json"
        )

    def source_paths(self) -> tuple[Path, ...]:
        return (self.questions_path,)

    def _questions(self) -> Iterator[SourceQuestion]:
        payload = json.loads(self.questions_path.read_text(encoding="utf-8"))
        for entry in payload:
            answerable = bool(entry.get("answerable", True))
            labels = tuple(entry.get("relevant_files", ())) if answerable else ()
            yield SourceQuestion(
                question_id=str(entry["id"]), query=entry["query"], relevance_labels=labels
            )


class LocomoAdapter(CorpusAdapter):
    """LOCOMO, from `locomo10.json` in the working directory.

    Question ids are `"{sample_id}/{qa_index}"` — LOCOMO's own question objects carry no id, and
    the index alone is not unique across samples. Labels are the BARE evidence turn ids (`"D1:3"`),
    which is what `recall/eval/locomo.py::_retrieved_dia_ids` recovers from an indexed filename.
    Bare rather than sample-prefixed because a LOCOMO run indexes one conversation at a time, and
    that is what makes a turn id unique within the index being scored.

    Category 5 questions ("adversarial", unanswerable by construction in LOCOMO's own taxonomy)
    are frozen with empty labels, so they land on the safety axis rather than being dropped.
    """

    name = "locomo"
    #: Labels are bare turn ids ("D1:3"), recovered from the indexed filename exactly the way
    #: `recall/eval/locomo.py::_retrieved_dia_ids` recovers them. Bare rather than
    #: sample-prefixed because a LOCOMO run indexes ONE conversation at a time, which is what
    #: makes a turn id unique within the index being scored.
    label_kind = "locomo_dia"

    def __init__(self, data_path: Path | None = None) -> None:
        self.data_path = data_path or Path("locomo10.json")

    def source_paths(self) -> tuple[Path, ...]:
        return (self.data_path,)

    def _questions(self) -> Iterator[SourceQuestion]:
        payload = json.loads(self.data_path.read_text(encoding="utf-8"))
        for sample in payload:
            sample_id = str(sample["sample_id"])
            for index, qa in enumerate(sample.get("qa", ())):
                evidence = qa.get("evidence") or ()
                labels = tuple(str(dia_id) for dia_id in evidence)
                if int(qa.get("category", 0)) == 5:
                    labels = ()
                yield SourceQuestion(
                    question_id=f"{sample_id}/{index}",
                    query=str(qa["question"]),
                    relevance_labels=labels,
                )


class LadderAdapter(CorpusAdapter):
    """The Answerability Ladder, from its own frozen manifests under `results/ladder/`.

    The ladder's instances are ALREADY frozen, with their own digest. This adapter re-expresses
    them; it does not re-derive them, and `benchmarks.ladder.manifest.read_manifest` verifies the
    ladder digest before a single question crosses over. A ladder instance whose gold documents
    were excised is unanswerable by construction, so its labels are empty and it scores on the
    safety axis, which is the axis the whole benchmark was built to measure.
    """

    name = "ladder"
    #: Ladder doc ids ("c1/D1:3"), recovered by the inverse of the escaping its own adapter
    #: applies when it writes them to disk.
    label_kind = "ladder_doc"

    def __init__(self, manifest_path: Path | None = None) -> None:
        self.manifest_path = manifest_path or Path("results/ladder/manifest.jsonl")

    def source_paths(self) -> tuple[Path, ...]:
        return (self.manifest_path,)

    def _questions(self) -> Iterator[SourceQuestion]:
        from benchmarks.ladder.manifest import LABEL_UNANSWERABLE, read_manifest

        instances, _header = read_manifest(self.manifest_path)
        for instance in instances:
            labels = (
                ()
                if instance.label == LABEL_UNANSWERABLE
                else tuple(instance.gold_doc_ids)
            )
            yield SourceQuestion(
                question_id=instance.instance_id,
                query=instance.question,
                relevance_labels=labels,
            )


class LongMemEvalAdapter(CorpusAdapter):
    """LongMemEval, run fresh rather than read from a prior artifact.

    The instance file is not in the tree; point `--longmemeval` at a downloaded
    `longmemeval_s.json` (or the oracle variant). Ids are the dataset's own `question_id`. Labels
    are the answer session ids rendered the way `recall/eval/longmemeval.py::convert` writes them,
    so they line up with what an index built by that converter contains.

    `question_type` ending in `_abs` is LongMemEval's abstention class; those freeze with empty
    labels.
    """

    name = "longmemeval"
    #: `recall/eval/longmemeval.py::convert` writes one "<session_id>.md" per session.
    label_kind = "source"

    def __init__(self, data_path: Path | None = None) -> None:
        self.data_path = data_path or Path("longmemeval_s.json")

    def source_paths(self) -> tuple[Path, ...]:
        return (self.data_path,)

    def _questions(self) -> Iterator[SourceQuestion]:
        payload = json.loads(self.data_path.read_text(encoding="utf-8"))
        for instance in payload:
            question_id = str(instance["question_id"])
            abstention = str(instance.get("question_type", "")).endswith("_abs")
            sessions = instance.get("answer_session_ids") or ()
            labels = () if abstention else tuple(f"{sid}.md" for sid in sessions)
            yield SourceQuestion(
                question_id=question_id,
                query=str(instance["question"]),
                relevance_labels=labels,
            )


class MtragAdapter(CorpusAdapter):
    """MT-RAG, through the runner committed in session 1, for the compatible part of its format.

    Compatible: per-task queries and the qrels that say which corpus documents answer them. Read
    through `benchmarks.mtrag.run`'s own loaders, so the two never drift apart on how a task's
    domain or query text is derived.

    **Not** reused: that runner's `score_predictions`, which pools all judged queries into one
    list and therefore weights each domain by its judged-query count. This repository defines a
    macro average as the UNWEIGHTED mean of per-corpus values (`recall/promotion.py`), and the
    difference was already caught once as AUD-1. Each MT-RAG domain therefore becomes its own
    corpus here, `mtrag:<domain>`, so the gate's own stratification does the averaging.

    A task whose qrels are empty is dropped rather than frozen as unanswerable: MT-RAG's empty
    qrels mean "not judged", not "no document answers this", and the two are opposite claims.
    """

    name = "mtrag"
    #: `benchmarks/mtrag/run.py::index_domain` indexes each corpus document under its own id, so
    #: the qrels line up with chunk ids directly.
    label_kind = "chunk_id"

    def __init__(self, root: Path | None = None, *, mode: str = "last_turn") -> None:
        self.root = root or Path("mtrag")
        self.mode = mode

    def source_paths(self) -> tuple[Path, ...]:
        from benchmarks.mtrag.run import tasks_path

        try:
            return (tasks_path(self.root),)
        except Exception:
            return (self.root / "human" / "generation" / "tasks.jsonl",)

    def _questions(self) -> Iterator[SourceQuestion]:
        from benchmarks.mtrag.run import load_qrels, load_tasks, query_text, task_domain

        qrels = load_qrels(self.root)
        for task in load_tasks(self.root):
            domain = task_domain(task)
            task_id = str(task.get("task_id") or task.get("taskId") or "")
            relevant = qrels.get(domain, {}).get(task_id)
            if not relevant:
                continue
            yield SourceQuestion(
                question_id=f"{domain}/{task_id}",
                query=query_text(task, self.mode),
                relevance_labels=tuple(sorted(relevant)),
            )

    def corpus_of(self, question: SourceQuestion) -> str:
        """`mtrag:<domain>`, so the gate's unweighted macro average is over domains."""
        return f"{self.name}:{question.question_id.split('/', 1)[0]}"


#: Every adapter the CLI can name. A corpus absent from here cannot be frozen, which is
#: deliberate: adding one is an edit to this table, visible in a diff.
ADAPTERS: dict[str, type[CorpusAdapter]] = {
    "labelled": LabelledAdapter,
    "peps": PepsAdapter,
    "locomo": LocomoAdapter,
    "ladder": LadderAdapter,
    "longmemeval": LongMemEvalAdapter,
    "mtrag": MtragAdapter,
}


def label_kind(adapter: CorpusAdapter) -> str:
    """The id space this adapter's labels live in, refusing an adapter that declared none.

    There is deliberately no fallback. A default here would let a new adapter inherit a
    comparison that is silently wrong for its corpus, and the symptom of that is a clean run
    reporting that retrieval found nothing at all.
    """
    declared = getattr(adapter, "label_kind", None)
    if not isinstance(declared, str) or not declared:
        raise TypeError(
            f"{type(adapter).__name__} declares no label_kind. Every adapter must say which id "
            f"space its labels live in; see recall/eval/promotion/search.py::LABEL_KEYS."
        )
    return declared
