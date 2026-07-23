"""Convert LongMemEval into a corpus + labelled question set this repo can score.

LongMemEval (MIT, https://github.com/xiaowu0162/LongMemEval) is the only public benchmark in
the agent-memory field whose question taxonomy names the two things this library is actually
about: **knowledge-update** (78 instances — a fact the user later revised) and **abstention**
(30 instances — nothing in the history answers the question). The standard retrieval protocol
published alongside it *skips every abstention instance*, on the grounds that they have no
answer location. That is a reasonable choice for a system that cannot abstain, and it discards
precisely the class this repo exists to serve.

This module is a format adapter, nothing more. It turns

    {"question_id", "question_type", "question", "haystack_session_ids",
     "haystack_dates", "haystack_sessions", "answer_session_ids", ...}

into a directory of markdown sessions plus the question file `recall.eval.labelled` already
consumes, so the existing harness scores it with no changes:

    python -m recall.eval.longmemeval --dataset longmemeval_oracle.json --out /tmp/lme
    python -m recall.eval.labelled --corpus /tmp/lme/corpus --questions /tmp/lme/questions.json

**Three things about the resulting number, stated here so they cannot be discovered later.**

1. **It is a retrieval number, not the benchmark's number.** LongMemEval scores an LLM's answer
   with a judge. This scores whether the evidence session came back in the top *k*. It does not
   belong in the same column as anyone's LLM-judged accuracy, and reporting it as though it did
   would be the exact error this repository's README spends a section retracting.

2. **The haystack is merged, which makes it HARDER than the official protocol.** Officially each
   question is retrieved against its own ~40-session haystack. Here every unique session across
   every instance lands in one corpus and each question must find its evidence among all of
   them. That is a bigger haystack and a strictly harder task, so a low score here is not
   comparable to a per-question-haystack score either. It is chosen because it is what one index
   pass can afford, and because a shared corpus is closer to how a real memory store is used.

3. **Ground truth is at session level.** `recall.eval.labelled` scores file-level hit@k, so a
   multi-session question (133 of them) counts as a hit when *any* of its evidence sessions is
   retrieved — weaker than the benchmark's requirement that the answer be derivable. Read the
   multi-session category accordingly.

4. **Temporal-reasoning questions are NOT scoreable in this mode.** The benchmark reuses a
   distractor session across many haystacks and stamps it with a *different* date in each. One
   merged corpus can hold one document per session, so a session seen at several dates keeps
   the first and records the rest in `session_dates_all`. The converter reports how many
   sessions this affected (`sessions_at_multiple_dates`); if that count is non-zero — it will
   be — the 133 temporal-reasoning questions are measuring a timeline the corpus does not
   faithfully represent, and their numbers should not be published from this harness.

The dataset is not vendored. Fetch it yourself from the `xiaowu0162/longmemeval-cleaned`
HuggingFace repository; `longmemeval_oracle.json` (evidence sessions only) is the cheap first
run and the one to validate the converter against.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# A session id becomes a filename. The ids come out of a downloaded file, so joining one onto a
# path without checking is an arbitrary-write primitive: `../../.ssh/authorized_keys` is a
# perfectly good JSON string. Anything outside this class is refused loudly rather than
# sanitised quietly — a silently rewritten id would break the `answer_session_ids` → filename
# correspondence the score depends on.
_SAFE_SESSION_ID = re.compile(r"\A[A-Za-z0-9._-]+\Z")

# Instances whose question_id carries this suffix are the benchmark's abstention class: nothing
# in the haystack answers them, and they have no answer location by construction.
_ABSTENTION_SUFFIX = "_abs"


class ConversionError(ValueError):
    """The dataset does not have the shape this converter can score."""


@dataclass
class ConversionReport:
    sessions_written: int = 0
    sessions_deduplicated: int = 0
    #: Sessions the benchmark placed at more than one timestamp across haystacks. The merged
    #: corpus holds one document per session, so only the first date is the document's date and
    #: the rest are recorded beside it. Any temporal-reasoning question over such a session is
    #: not scoreable in this mode — see the module docstring.
    sessions_at_multiple_dates: int = 0
    questions: int = 0
    answerable: int = 0
    abstention: int = 0
    by_type: Counter = field(default_factory=Counter)


def _check_session_id(sid: str) -> None:
    if not isinstance(sid, str) or not _SAFE_SESSION_ID.match(sid) or sid in {".", ".."}:
        raise ConversionError(
            f"unsafe session id {sid!r}: a session id becomes a filename and must be a plain "
            "name of letters, digits, dot, underscore or hyphen"
        )


def render_turns(turns: list[dict]) -> str:
    """The conversation itself, role-labelled and in order — no frontmatter.

    This is what identity is judged on: the same session reused across haystacks must compare
    equal even though the benchmark stamps it with a different date each time.

    `has_answer` is deliberately not rendered: it is the benchmark's turn-level ground truth,
    and writing it into the corpus would put the answer key inside the documents being searched.
    """
    lines: list[str] = []
    for turn in turns:
        lines.append(f"**{turn.get('role', 'unknown')}**")
        lines.append("")
        lines.append(str(turn.get("content", "")).strip())
        lines.append("")
    return "\n".join(lines)


def render_session(session_id: str, dates: list[str], turns: list[dict]) -> str:
    """A session as a markdown document: frontmatter, then the conversation.

    `parse_frontmatter` recognises only the validity keys and strips the whole block from the
    body, so these keys are a record for the reader and never reach the index.
    """
    head = ["---", f"session_id: {session_id}", f"session_date: {dates[0] if dates else ''}"]
    if len(dates) > 1:
        head.append("session_dates_all: " + " | ".join(dates))
    head += ["---", ""]
    return "\n".join(head) + render_turns(turns)


def convert(instances: list[dict], out_dir: Path) -> ConversionReport:
    """Write `out_dir/corpus/*.md` and `out_dir/questions.json`; return what was written."""
    out_dir = Path(out_dir)
    corpus = out_dir / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)

    report = ConversionReport()
    # session id -> (digest of the turns, turns, dates seen in order of first appearance)
    seen: dict[str, tuple[str, list[dict], list[str]]] = {}
    questions: list[dict] = []

    for inst in instances:
        qid = inst["question_id"]
        ids = inst["haystack_session_ids"]
        sessions = inst["haystack_sessions"]
        dates = inst.get("haystack_dates") or [""] * len(ids)
        if not (len(ids) == len(sessions) == len(dates)):
            raise ConversionError(
                f"{qid}: haystack is inconsistent — {len(ids)} session ids, {len(sessions)} "
                f"sessions, {len(dates)} dates"
            )

        for sid, date, turns in zip(ids, dates, sessions):
            _check_session_id(sid)
            digest = hashlib.sha256(render_turns(turns).encode("utf-8")).hexdigest()
            if sid not in seen:
                seen[sid] = (digest, turns, [date])
                continue
            known_digest, _, known_dates = seen[sid]
            if known_digest != digest:
                # Keeping either copy would leave a document in the corpus that no longer
                # matches the haystack some question was written against, and the run would
                # still print a number. The DATE differing is expected and handled above; the
                # conversation differing is not.
                raise ConversionError(
                    f"session id {sid!r} appears twice with different content "
                    f"(second occurrence in {qid})"
                )
            report.sessions_deduplicated += 1
            if date not in known_dates:
                known_dates.append(date)

        question = {
            "id": qid,
            "query": inst["question"],
            # Carried through for per-category analysis. `recall.eval.labelled` ignores keys it
            # does not know, so this costs nothing and is the only way to read the
            # knowledge-update and abstention categories out of a run.
            "question_type": inst.get("question_type", "unknown"),
        }
        if qid.endswith(_ABSTENTION_SUFFIX):
            question["answerable"] = False
            report.abstention += 1
        else:
            gold = inst.get("answer_session_ids") or []
            if not gold:
                raise ConversionError(
                    f"{qid}: answerable instance has no answer_session_ids — only the "
                    f"abstention class (ids ending {_ABSTENTION_SUFFIX!r}) may omit them"
                )
            unknown = [s for s in gold if s not in ids]
            if unknown:
                raise ConversionError(
                    f"{qid}: answer session(s) {unknown} are not in this instance's haystack, "
                    "so they would be scored against a corpus that cannot contain them"
                )
            question["answerable"] = True
            question["relevant_files"] = [f"{s}.md" for s in gold]
            report.answerable += 1
        report.by_type[question["question_type"]] += 1
        questions.append(question)

    # Written last: a session's full date list is only known once every instance has been read.
    for sid, (_, turns, dates) in seen.items():
        (corpus / f"{sid}.md").write_text(render_session(sid, dates, turns), encoding="utf-8")
        report.sessions_written += 1
        if len(dates) > 1:
            report.sessions_at_multiple_dates += 1

    report.questions = len(questions)
    (out_dir / "questions.json").write_text(
        json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="recall.eval.longmemeval",
        description="Convert a LongMemEval json into a corpus + labelled questions.",
    )
    ap.add_argument("--dataset", required=True,
                    help="longmemeval_oracle.json | longmemeval_s_cleaned.json | ..._m_...")
    ap.add_argument("--out", required=True, help="output directory (corpus/ + questions.json)")
    args = ap.parse_args()

    instances = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    report = convert(instances, Path(args.out))

    print(f"sessions written      {report.sessions_written}")
    print(f"  deduplicated        {report.sessions_deduplicated}")
    print(f"  at >1 date          {report.sessions_at_multiple_dates} "
          f"(temporal-reasoning questions over these are not scoreable here)")
    print(f"questions             {report.questions} "
          f"({report.answerable} answerable, {report.abstention} abstention)")
    for qtype, n in sorted(report.by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {qtype:28} {n}")
    print(f"\nnext:\n  python -m recall.eval.labelled --corpus {args.out}/corpus "
          f"--questions {args.out}/questions.json")


if __name__ == "__main__":
    main()
