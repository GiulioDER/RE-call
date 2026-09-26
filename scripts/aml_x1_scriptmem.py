"""X-1 ScriptMem, measured on locally reconstructed script texts (a labelled proxy, not ScriptMem).

Pre-registration: docs/preregistrations/2026-09-26-aml-c9-scriptmem-reconstructed.md. The texts are
held privately (the user approved private local use on 2026-09-26) and never committed; this script
reads them from ``--texts``, where ``normalize`` has written one ``<work>.jsonl`` per work, each line
``{"session": str, "date": "YYYY-MM-DD" | null, "speaker": str | null, "text": str}`` in reading order.

``normalize``  convert the fetched source files into that format (one converter per source layout)
``coverage``   Stage 0: gold-option proper nouns and numbers found in each work's text (free)
``collect``    per work, on the X-1 C9 service: every Add, then every question's Search at top_k 100,
               then delete and verify an empty Search; items stored per question
``answer``     arms C9, C9p, N0, NK, F with AML's ScriptMem answer prompt (DeepSeek, capped)
``score``      ScriptMem's exact scorer as vendored by AML, per arm, work and question type

    python scripts/aml_x1_scriptmem.py coverage --data-dir DATA --texts TEXTS --out coverage.json
    python scripts/aml_x1_scriptmem.py collect  --data-dir DATA --texts TEXTS --out-dir OUT --port 18034
    python scripts/aml_x1_scriptmem.py answer   --data-dir DATA --texts TEXTS --out-dir OUT --aml-repo AML --arms C9,C9p,N0,NK,F
    python scripts/aml_x1_scriptmem.py score    --data-dir DATA --out-dir OUT --aml-repo AML
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import aml_x1_sources as x1  # noqa: E402

RUN_ID = "x1sm"
WORKS = ("angry", "enemy", "friends", "man_earth")
TITLES = {
    "angry": "12 Angry Men",
    "enemy": "An Enemy of the People",
    "friends": "Friends",
    "man_earth": "The Man from Earth",
}
SINGLE_WORKS = ("angry", "enemy", "man_earth")
ARMS = ("C9", "C9p", "N0", "NK", "F")
LINES_PER_SESSION = 60
#: Works without a calendar: one fixed day, one minute per session, so order survives and no false
#: gap between sessions appears (pre-registration, "Sessions").
UNDATED_BASE_MS = int(datetime(2000, 1, 1, tzinfo=UTC).timestamp() * 1000)
AML_COMMIT = "1b8142bfe0f20f1c5218d6b554aa0012de34e504"
COVERAGE_GATE = 0.80
CAP_USD = 10.0
WORKERS = 2

#: ScriptMem renames the six Friends leads; inferred from its public questions ("Bennett's ex-wife"
#: Carol, "Dexter's mom" Nora Bing). Full names first, then first names and two nicknames.
FRIENDS_RENAME_FULL = (
    ("Ross Geller", "Bennett Geller"), ("Monica Geller", "Chloe Geller"), ("Rachel Green", "Ariel Green"),
    ("Chandler Bing", "Dexter Bing"), ("Joey Tribbiani", "Ethan Tribbiani"), ("Phoebe Buffay", "Fiona Buffay"),
)
FRIENDS_RENAME_FIRST = (
    ("Ross", "Bennett"), ("Monica", "Chloe"), ("Rachel", "Ariel"), ("Chandler", "Dexter"),
    ("Joey", "Ethan"), ("Phoebe", "Fiona"), ("Rach", "Ariel"), ("Pheebs", "Fiona"),
)
#: First US air dates, Friends season 1, episodes 1 to 24 (16 and 17 aired the same night).
FRIENDS_S1_AIR_DATES = (
    "1994-09-22", "1994-09-29", "1994-10-06", "1994-10-13", "1994-10-20", "1994-10-27",
    "1994-11-03", "1994-11-10", "1994-11-17", "1994-12-15", "1995-01-05", "1995-01-12",
    "1995-01-19", "1995-02-09", "1995-02-16", "1995-02-23", "1995-02-23", "1995-03-02",
    "1995-03-09", "1995-04-06", "1995-04-27", "1995-05-04", "1995-05-11", "1995-05-18",
)


def rename_friends(text: str) -> str:
    for old, new in FRIENDS_RENAME_FULL:
        text = text.replace(old, new)
    for old, new in FRIENDS_RENAME_FIRST:
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


def read_lines(texts: Path, work: str) -> list[dict[str, Any]]:
    path = texts / f"{work}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(texts: Path, work: str, lines: list[dict[str, Any]]) -> None:
    texts.mkdir(parents=True, exist_ok=True)
    with (texts / f"{work}.jsonl").open("w", encoding="utf-8") as sink:
        for line in lines:
            sink.write(json.dumps(line, ensure_ascii=False) + "\n")


def line_text(line: dict[str, Any]) -> str:
    """``Speaker: text``; unattributed dialogue (film subtitles) bare; anything else is narration."""
    speaker = line.get("speaker")
    if speaker:
        return f"{speaker}: {line['text']}"
    if line.get("kind") == "dialogue":
        return str(line["text"])
    return f"Narration: {line['text']}"


def full_text(texts: Path, work: str) -> str:
    return "\n".join(line_text(line) for line in read_lines(texts, work))


# ------------------------------------------------------------------------------------ normalize


def normalize_friends(paths: list[Path]) -> list[dict[str, Any]]:
    """EmoryNLP character-mining JSON: episodes -> scenes -> utterances (speakers, transcript)."""
    out: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for episode in data["episodes"]:
            match = re.search(r"s(\d+)_e(\d+)$", episode["episode_id"])
            season, number = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
            date = FRIENDS_S1_AIR_DATES[number - 1] if season == 1 and 1 <= number <= 24 else None
            for scene in episode["scenes"]:
                for utterance in scene["utterances"]:
                    text = (utterance.get("transcript") or "").strip()
                    if not text:
                        continue
                    speakers = [s for s in (utterance.get("speakers") or []) if s]
                    speaker = ", ".join(speakers) if speakers else None
                    if speaker and "scene" in speaker.lower():
                        speaker = None
                    out.append({"session": episode["episode_id"], "date": date,
                                "speaker": rename_friends(speaker) if speaker else None,
                                "text": rename_friends(text)})
    return out


def normalize_gutenberg_play(path: Path) -> list[dict[str, Any]]:
    """Project Gutenberg #2446: acts as sessions, "Speaker." or "Speaker (direction)." paragraphs."""
    raw = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    start = re.search(r"\*\*\* ?START OF[^\n]*\n", raw)
    end = re.search(r"\*\*\* ?END OF", raw)
    raw = raw[start.end() if start else 0: end.start() if end else len(raw)]
    out: list[dict[str, Any]] = []
    act = "front-matter"
    # An honorific name ("Mrs. Stockmann") is tried first, so its own full stop is not taken for
    # the one that ends the speaker label; otherwise one to five words ("Voices from the crowd").
    speaker_re = re.compile(
        r"^((?:Dr|Mrs|Mr)\.\s[A-Z][a-z]+|[A-Z][A-Za-z]+(?:\s[A-Za-z][a-z]+){0,4})"
        r"\s*(?:\(([^)]*)\))?\.\s+(.*)$", re.S)
    for paragraph in re.split(r"\n\s*\n", raw):
        text = " ".join(paragraph.split())
        if not text:
            continue
        act_match = re.fullmatch(r"ACT\s+([IVX]+)", text)
        if act_match:
            act = f"act-{act_match.group(1)}"
            continue
        match = speaker_re.match(text)
        if match and act != "front-matter" and not text.startswith("("):
            direction = f"({match.group(2)}) " if match.group(2) else ""
            out.append({"session": act, "date": None, "speaker": match.group(1).strip(),
                        "text": f"{direction}{match.group(3).strip()}"})
        elif act != "front-matter":
            out.append({"session": act, "date": None, "speaker": None, "text": text})
    return out


def normalize_subtitles(path: Path) -> list[dict[str, Any]]:
    """A film's subtitle-style dialogue page: no speakers, sentences broken across lines.

    Fragments are rejoined until a line ends a sentence; a line opening with a dash starts a new
    utterance. Every utterance is dialogue with no speaker, so it is never labelled as narration.
    """
    page = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r'<div class="scrolling-script-container">(.*?)</div>', page, re.S)
    if not match:
        raise SystemExit(f"{path.name}: no script container")
    fragments = [html.unescape(re.sub(r"<[^>]+>", "", part)).strip() for part in re.split(r"<br\s*/?>", match.group(1))]
    out: list[dict[str, Any]] = []
    buffer: list[str] = []
    for fragment in (f for f in fragments if f):
        if fragment.startswith("-") and buffer:
            out.append(" ".join(buffer))
            buffer = []
        buffer.append(fragment.lstrip("- ").strip())
        if re.search(r"[.?!…\"')\]]$", fragment) and not fragment.endswith(("Mr.", "Mrs.", "Dr.")):
            out.append(" ".join(buffer))
            buffer = []
    if buffer:
        out.append(" ".join(buffer))
    return [{"session": "film", "date": None, "speaker": None, "kind": "dialogue", "text": text} for text in out]


#: The OCR reads a 1 as "l" in "#11", "#12", "#10"; those are mapped back before use.
_CUE = re.compile(r"^\s*(#\s?[\dl]{1,2}|FOREMAN|GUARD|JUDGE|CLERK)\s*(?:\(cont\.?\))?\s*$")
_WORD = re.compile(r"[a-z]+")
#: Function words say nothing about which speech a line belongs to, and a tiny block made of them
#: would otherwise win on the size-normalised score.
_STOP = frozenset(
    "a an the and or but if so to of in on at by for with from as is are was were be been am i you he "
    "she it we they me him her us them my your his its our their this that these those what who how "
    "do does did not no yes don t s m ll re ve d just all there here then than about up out get got "
    "go going know well oh like can could would will".split()
)


def content_words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _STOP and len(word) > 1}
ALIGN_MIN_SHARE = 0.6
ALIGN_GAP_SHARE = 0.5
#: Only a block of ordinary size (the 90th percentile is 84 distinct words) lends its speaker
#: to an unmatched line between two lines it holds.
ALIGN_FILL_MAX_WORDS = 60
ALIGN_MAX_RUN = 30


def shooting_script_blocks(path: Path) -> list[tuple[str, set[str]]]:
    """Speaker cues and the words under each, from the 12 Angry Men shooting script's OCR text.

    The OCR is too noisy to serve as the text itself, but its cue lines survive, and each block's
    word set is enough to recognise the matching film line.
    """
    blocks: list[tuple[str, list[str]]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        cue = _CUE.match(raw_line)
        if cue:
            label = cue.group(1).replace(" ", "").replace("l", "1")
            speaker = f"Juror {label}" if label.startswith("#") else label.title()
            blocks.append((speaker, []))
        elif blocks:
            blocks[-1][1].append(raw_line.lower())
    return [(speaker, content_words(" ".join(lines))) for speaker, lines in blocks]


def _best_block(words: set[str], blocks: list[tuple[str, set[str]]], low: int, high: int) -> tuple[int | None, float]:
    """The block chosen by overlap over sqrt(|line| * |block|), so a block that swallowed its
    neighbours' text through a lost cue does not win by size; returned with the line's share."""
    best, best_score, best_share = None, 0.0, 0.0
    for index in range(low, high):
        overlap = len(words & blocks[index][1])
        score = overlap / math.sqrt(len(words) * max(1, len(blocks[index][1])))
        if score > best_score:
            best, best_score, best_share = index, score, overlap / len(words)
    return best, best_share


def align_speakers(lines: list[dict[str, Any]], blocks: list[tuple[str, set[str]]]) -> list[dict[str, Any]]:
    """Give each film line the speaker of the script block that holds most of its words.

    1. Anchors: every line of at least four distinct words is matched against all blocks
       (share at least ALIGN_MIN_SHARE), and the longest run of those matches that is increasing in
       block order is kept, so a common line cannot jump across the film.
    2. Between two anchors, a line of at least three words is matched only against the blocks that
       lie between them, at ALIGN_GAP_SHARE.
    3. A line still unmatched takes a speaker only when the nearest matched lines on both sides sit
       in the same block. Everything else stays unattributed dialogue.
    """
    word_sets = [content_words(line["text"]) for line in lines]
    candidates = []
    for position, words in enumerate(word_sets):
        if len(words) >= 3:
            block, share = _best_block(words, blocks, 0, len(blocks))
            if block is not None and share >= ALIGN_MIN_SHARE:
                candidates.append((position, block))
    tails: list[int] = []
    previous: list[int | None] = [None] * len(candidates)
    for k, (_, block) in enumerate(candidates):
        slot = bisect.bisect_right([candidates[t][1] for t in tails], block)
        if slot == len(tails):
            tails.append(k)
        else:
            tails[slot] = k
        previous[k] = tails[slot - 1] if slot else None
    matched: dict[int, int] = {}
    cursor = tails[-1] if tails else None
    while cursor is not None:
        matched[candidates[cursor][0]] = candidates[cursor][1]
        cursor = previous[cursor]
    anchors = sorted(matched.items())
    for (left_line, left_block), (right_line, right_block) in zip(anchors, anchors[1:], strict=False):
        for position in range(left_line + 1, right_line):
            if len(word_sets[position]) >= 2:
                block, share = _best_block(word_sets[position], blocks, left_block, right_block + 1)
                if block is not None and share >= ALIGN_GAP_SHARE:
                    matched[position] = block
    ordered = sorted(matched)
    for position in range(len(lines)):
        if position in matched:
            continue
        slot = bisect.bisect_left(ordered, position)
        if (0 < slot < len(ordered) and matched[ordered[slot - 1]] == matched[ordered[slot]]
                and len(blocks[matched[ordered[slot]]][1]) <= ALIGN_FILL_MAX_WORDS):
            matched[position] = matched[ordered[slot]]
    out = [{**line, "speaker": blocks[matched[i]][0]} if i in matched else line for i, line in enumerate(lines)]
    # 4. A run longer than ALIGN_MAX_RUN lines on one speaker is a block that swallowed a scene
    #    through lost cues (the opening chatter all went to #5), not a speech: unattribute it.
    start = 0
    for end in range(1, len(out) + 1):
        if end == len(out) or out[end].get("speaker") != out[start].get("speaker"):
            if out[start].get("speaker") and end - start > ALIGN_MAX_RUN:
                for position in range(start, end):
                    out[position] = {**out[position], "speaker": None}
            start = end
    return out


def chunk_sessions(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For a source with no act or scene marks: consecutive runs of LINES_PER_SESSION lines."""
    return [{**line, "session": f"part-{index // LINES_PER_SESSION + 1:03d}"} for index, line in enumerate(lines)]


def cmd_normalize(args: argparse.Namespace) -> None:
    raw = args.raw
    if args.work == "friends":
        lines = normalize_friends(sorted(raw.glob("friends_season_*.json")))
    elif args.work == "enemy":
        lines = normalize_gutenberg_play(raw / "enemy_pg2446.txt")
    else:
        lines = normalize_subtitles(raw / f"{args.work}_springfield.html")
        script = raw / f"{args.work}_shooting_script.txt"
        if script.exists():
            lines = align_speakers(lines, shooting_script_blocks(script))
    write_lines(args.texts, args.work, lines)
    sessions = Counter(line["session"] for line in lines)
    print(json.dumps({"work": args.work, "lines": len(lines), "sessions": len(sessions),
                      "speakers": len({line['speaker'] for line in lines if line['speaker']}),
                      "unattributed_dialogue": sum(1 for line in lines
                                                   if not line["speaker"] and line.get("kind") == "dialogue"),
                      "narration_lines": sum(1 for line in lines
                                             if not line["speaker"] and line.get("kind") != "dialogue"),
                      "chars": sum(len(line["text"]) for line in lines)}))


# ------------------------------------------------------------------------------------ questions


def questions(data_dir: Path) -> list[dict[str, Any]]:
    """X-1's own ScriptMem questions (ids, Search requests), under this record's run id."""
    out = x1.scriptmem_questions(data_dir, RUN_ID)
    options = {}
    for work in WORKS:
        data = json.loads((data_dir / "scriptmem" / f"{work}.json").read_text(encoding="utf-8"))
        for sample_index, sample in enumerate(data):
            sample_id = sample.get("sample_id") or f"{work}-{sample_index}"
            for qa_index, qa in enumerate(sample.get("qa", [])):
                options[f"{work}:{sample_id}#q{qa_index:04d}"] = qa
    for question in out:
        question["qa"] = options[question["question_id"]]
    return out


# ------------------------------------------------------------------------------------ coverage

_NAME = re.compile(r"\b(?:[A-Z][a-z]+(?:[-'][A-Za-z]+)?|\d[\d,.]*)\b")
_COMMON = set(
    "The A An And But Or If In On At To Of For With By From As He She They His Her Their It Its This That These "
    "Those What Which Who Whom Why When Where How Not No Yes Only Both Each All Some Any None Cannot Because "
    "After Before During While Then Later Earlier First Second Third Last Next Instead Rather Although So "
    "Juror Mr Mrs Dr".split()
)


def names_in(text: str) -> set[str]:
    return {token for token in _NAME.findall(text) if token not in _COMMON}


def option_body(option: str) -> str:
    """The option without its letter and without its first word, which is capitalised as the
    start of a sentence rather than as a name ("Earliest", "Institutional")."""
    body = re.sub(r"^\s*[A-F]\.\s*", "", option)
    return body.split(" ", 1)[1] if " " in body else ""


def cmd_coverage(args: argparse.Namespace) -> None:
    by_work: dict[str, Counter[str]] = defaultdict(Counter)
    missing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    haystacks = {work: full_text(args.texts, work) for work in WORKS if (args.texts / f"{work}.jsonl").exists()}
    for question in questions(args.data_dir):
        work = question["script"]
        if work not in haystacks:
            continue
        qa = question["qa"]
        stem = qa["question"].split("\n\n")[0]
        gold = set(x1_gold_letters(qa["answer"]))
        stem_names = names_in(stem)
        for option in qa["option"]:
            letter = option.strip()[:1]
            if "cannot infer" in option.lower():
                continue
            wanted = names_in(option_body(option)) - stem_names
            found = {name for name in wanted if re.search(rf"\b{re.escape(name)}\b", haystacks[work])}
            kind = "gold" if letter in gold else "wrong"
            by_work[work][f"{kind}_names"] += len(wanted)
            by_work[work][f"{kind}_found"] += len(found)
            if kind == "gold" and wanted - found:
                missing[work].append({"question_id": question["question_id"], "missing": sorted(wanted - found)})
    report: dict[str, Any] = {"gate": COVERAGE_GATE, "works": {}}
    for work, counts in by_work.items():
        gold_share = counts["gold_found"] / counts["gold_names"] if counts["gold_names"] else None
        wrong_share = counts["wrong_found"] / counts["wrong_names"] if counts["wrong_names"] else None
        report["works"][work] = {**counts, "gold_share": gold_share, "wrong_share": wrong_share,
                                 "kept": gold_share is not None and gold_share >= COVERAGE_GATE,
                                 "gold_missing": missing[work]}
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({w: {k: v for k, v in r.items() if k != "gold_missing"} for w, r in report["works"].items()},
                     indent=2))


def x1_gold_letters(answer: Any) -> list[str]:
    """The same parse as AML's ``gold_letters``: a leading ``X.`` on each answer part."""
    parts = answer if isinstance(answer, list) else [answer]
    return [m.group(1) for part in parts if (m := re.match(r"\s*([A-F])\.", str(part)))]


# ------------------------------------------------------------------------------------ collect


def sessions_of(texts: Path, work: str) -> list[tuple[str, int, list[dict[str, Any]]]]:
    lines = read_lines(texts, work)
    if len({line["session"] for line in lines}) <= 1:
        lines = chunk_sessions(lines)
    ordered: dict[str, list[dict[str, Any]]] = {}
    for line in lines:
        ordered.setdefault(line["session"], []).append(line)
    out = []
    for index, (session, members) in enumerate(ordered.items()):
        date = members[0].get("date")
        stamp = (int(datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp() * 1000)
                 if date else UNDATED_BASE_MS + index * 60_000)
        out.append((session, stamp, members))
    return out


def adds_for(texts: Path, work: str) -> list[dict[str, Any]]:
    user_id = x1.user_id_for(RUN_ID, "scriptmem", work)
    return [
        x1.add_request(user_id, index, f"{work}:{session}",
                       [{"role": "user", "content": line_text(line), "timestamp": stamp} for line in members])
        for index, (session, stamp, members) in enumerate(sessions_of(texts, work))
    ]


def cmd_collect(args: argparse.Namespace) -> None:
    import aml_x1_stageb as stageb

    token = stageb.token_from_env(args.serve_env)
    kept = json.loads(args.coverage.read_text(encoding="utf-8"))["works"]
    works = [w for w in WORKS if kept.get(w, {}).get("kept")]
    all_questions = questions(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "collected.jsonl"
    done = {json.loads(line)["work"] for line in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    for work in works:
        if work in done:
            continue
        user_id = x1.user_id_for(RUN_ID, "scriptmem", work)
        stageb.call(args.port, "/v1/delete", {"user_id": user_id}, token)
        adds = adds_for(args.texts, work)
        add_rows = []
        for add in adds:
            reply = stageb.call(args.port, "/v1/add", add, token)
            add_rows.append({"request_id": add["request_id"], "session_id": add["session_id"],
                             "messages": len(add["messages"]), "status": reply.status, "ms": round(reply.ms)})
        work_lines = read_lines(args.texts, work)
        canary_text = next(line["text"] for line in work_lines[len(work_lines) // 2:] if len(line["text"]) > 80)
        canary = stageb.call(args.port, "/v1/search", {"query": canary_text, "user_id": user_id, "top_k": 10}, token)
        canary_hit = any(canary_text[:50] in str(item.get("content", "")) for item in canary.body.get("data", []))

        def search(question: dict[str, Any]) -> dict[str, Any]:
            reply = stageb.call(args.port, "/v1/search", question["search"], token)
            items = reply.body.get("data", []) if reply.status == 200 else []
            return {"question_id": question["question_id"], "status": reply.status, "ms": round(reply.ms),
                    "items": [stageb.compact(item) for item in items]}

        mine = [q for q in all_questions if q["script"] == work]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            searches = list(pool.map(search, mine))
        deleted = stageb.call(args.port, "/v1/delete", {"user_id": user_id}, token)
        after = stageb.call(args.port, "/v1/search", {"query": "cleanup verification", "user_id": user_id, "top_k": 1}, token)
        row = {"work": work, "user_id": user_id, "adds": add_rows, "canary_hit_top10": canary_hit,
               "searches": searches,
               "deleted_and_empty": deleted.status == 200 and after.status == 200 and not after.body.get("data")}
        with out.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"work": work, "adds": len(add_rows), "add_non_200": sum(r["status"] != 200 for r in add_rows),
                          "searches": len(searches), "search_non_200": sum(s["status"] != 200 for s in searches),
                          "canary_hit_top10": canary_hit, "deleted_and_empty": row["deleted_and_empty"]}), flush=True)


# ------------------------------------------------------------------------------------ answer


def load_scriptmem_pipeline(repo: Path) -> ModuleType:
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    if head != AML_COMMIT:
        raise SystemExit(f"AML checkout is at {head}, expected {AML_COMMIT}")
    spec = importlib.util.spec_from_file_location("aml_scriptmem_pipeline", repo / "data" / "scriptmem" / "pipeline.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(repo))
    spec.loader.exec_module(module)
    return module


NK_LINE_OLD = "1. Use only the provided memories. Prefer the memories that answer the question most directly."


def prompt_for(pipeline: ModuleType, arm: str, question: dict[str, Any], memories: str) -> str:
    work = question["script"]
    item = {"question": question["qa"]["question"],
            "speaker_1_name": f"the characters of {TITLES[work]}", "speaker_1_memories": memories,
            "speaker_2_name": "(none)", "speaker_2_memories": "(all memories are listed above)"}
    prompt = str(pipeline.render_answer_prompt(item))
    if arm == "NK":
        if NK_LINE_OLD not in prompt:
            raise SystemExit("the pinned answer prompt no longer carries the line NK replaces")
        prompt = prompt.replace(NK_LINE_OLD, f"1. Answer from what you know about {TITLES[work]}.")
    return prompt


def cmd_answer(args: argparse.Namespace) -> None:
    from aml_locomo_loss_diagnosis import render_memories
    import aml_t1k2_locomo as t1k2

    pipeline = load_scriptmem_pipeline(args.aml_repo)
    arms = [arm for arm in args.arms.split(",") if arm]
    if not set(arms) <= set(ARMS):
        raise SystemExit(f"--arms must name only {ARMS}")
    collected = {}
    path = args.out_dir / "collected.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for search in row["searches"]:
                collected[search["question_id"]] = search["items"]
    kept = {w for w, r in json.loads(args.coverage.read_text(encoding="utf-8"))["works"].items() if r["kept"]}
    texts = {work: full_text(args.texts, work) for work in SINGLE_WORKS if work in kept}
    sink_path = args.out_dir / "answers.jsonl"
    done = set()
    spent = 0.0
    if sink_path.exists():
        for line in sink_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add((record["question_id"], record["arm"]))
            spent += float((record.get("usage") or {}).get("cost") or 0.0)
    reader = t1k2.Reader(spent, CAP_USD)
    balance = reader.balance()
    print(json.dumps({"openrouter_balance_usd": round(balance, 2), "spent_so_far": round(spent, 4)}), flush=True)
    if balance < t1k2.CREDIT_FLOOR_USD:
        raise SystemExit(f"balance {balance:.2f} is below the floor {t1k2.CREDIT_FLOOR_USD}")
    todo = []
    for index, question in enumerate(q for q in questions(args.data_dir) if q["script"] in kept):
        rotated = arms[index % len(arms):] + arms[: index % len(arms)]
        for arm in rotated:
            if (question["question_id"], arm) in done:
                continue
            if arm in ("C9", "C9p") and question["question_id"] not in collected:
                continue
            if arm == "F" and question["script"] not in texts:
                continue
            todo.append((question, arm))
    lock = threading.Lock()

    def one(job: tuple[dict[str, Any], str]) -> None:
        question, arm = job
        if arm in ("C9", "C9p"):
            memories = render_memories(collected[question["question_id"]])
        elif arm == "F":
            memories = texts[question["script"]]
        else:
            memories = "(no memories)"
        try:
            answer, usage = reader.complete(prompt_for(pipeline, arm, question, memories), 300)
        except RuntimeError as exc:
            print(f"stopped: {exc}", flush=True)
            return
        with lock, sink_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps({"question_id": question["question_id"], "arm": arm, "script": question["script"],
                                   "qa_type": question["category"], "answer": answer, "usage": usage},
                                  ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(one, todo))
    print(json.dumps({"answered_this_run": len(todo), "spent_usd": round(reader.spent, 4),
                      "stopped": reader.stopped.is_set()}), flush=True)


# ------------------------------------------------------------------------------------ score


def cmd_score(args: argparse.Namespace) -> None:
    pipeline = load_scriptmem_pipeline(args.aml_repo)
    qas = {q["question_id"]: q for q in questions(args.data_dir)}
    cannot = {qid: {o.strip()[:1] for o in q["qa"]["option"] if "cannot infer" in o.lower()} for qid, q in qas.items()}
    table: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    per_question: dict[str, dict[str, float]] = defaultdict(dict)
    for line in (args.out_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        question = qas[record["question_id"]]
        qa_type = question["category"]
        gold = pipeline.gold_letters(question["qa"]["answer"])
        pred, malformed = pipeline.predicted_letters(record["answer"], qa_type)
        score = pipeline.score_item(qa_type, gold, pred, malformed)
        per_question[record["question_id"]][record["arm"]] = score
        for key in ("all", f"work:{question['script']}", f"type:{qa_type}"):
            cell = table[record["arm"]][key]
            cell["n"] += 1
            cell["score"] += score
            cell["unparsed"] += int(malformed or not pred)
            cell["cannot_infer"] += int(bool(pred) and set(pred) <= cannot[record["question_id"]])
    summary = {arm: {key: {"n": c["n"], "accuracy": c["score"] / c["n"], "unparsed": c["unparsed"],
                           "cannot_infer": c["cannot_infer"]} for key, c in cells.items()}
               for arm, cells in table.items()}
    contrasts = {}
    for a, b in (("C9p", "C9"), ("C9", "N0"), ("C9", "NK"), ("F", "C9")):
        pairs = [(s[a], s[b]) for s in per_question.values() if a in s and b in s]
        if pairs:
            diffs = [x - y for x, y in pairs]
            contrasts[f"{a}-{b}"] = {"n": len(pairs), "mean": sum(diffs) / len(diffs), "ci95": bootstrap(diffs)}
    result = {"summary": summary, "contrasts": contrasts,
              "note": "reconstructed ScriptMem, a proxy; F covers the single works only"}
    (args.out_dir / "score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def bootstrap(diffs: list[float], rounds: int = 10_000, seed: int = 20260926) -> list[float]:
    import random

    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(rounds))
    return [means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("normalize")
    p.add_argument("--work", choices=WORKS, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--texts", type=Path, required=True)
    p.set_defaults(func=cmd_normalize)
    p = commands.add_parser("coverage")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--texts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_coverage)
    p = commands.add_parser("collect")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--texts", type=Path, required=True)
    p.add_argument("--coverage", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--port", type=int, default=18034)
    p.add_argument("--serve-env", type=Path, default=Path(os.path.expanduser("~/mm1-mm3/serve/base.env")))
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_collect)
    p = commands.add_parser("answer")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--texts", type=Path, required=True)
    p.add_argument("--coverage", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--aml-repo", type=Path, required=True)
    p.add_argument("--arms", default=",".join(ARMS))
    p.set_defaults(func=cmd_answer)
    p = commands.add_parser("score")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--aml-repo", type=Path, required=True)
    p.set_defaults(func=cmd_score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
