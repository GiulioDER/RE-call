"""X-1 Stage A: fetch, sample and dry-run the six AML sources C9 has never been measured on.

Pre-registration: ``docs/preregistrations/2026-09-25-aml-c9-source-coverage-baseline.md``.

Stage A is free: it downloads public data at pinned revisions, draws the pre-registered subsets,
and builds every AML Add and Search request for them WITHOUT sending anything. No model, no
embedding, no C9 service and no paid API is called anywhere in this file; the only network
traffic is to Hugging Face and GitHub for the pinned files.

One adapter per source turns a sampled item into (a) the Adds for its tenant, one Add per
session or conversational round, with epoch-millisecond timestamps where the data carries dates
and images as ``data:image/...;base64`` URLs in ``{"type": "image_url"}`` content parts, and (b)
the Search request plus everything its scorer needs (gold, rubrics, options, category, labelled
evidence). Tenants follow the record: one per question for LongMemEval-S and MemLens, one per
script for ScriptMem, one per persona for PersonaMem-v2, one per user for MobileMem-Omni, and one
per task for CLBench.

Commands, in order (``materialize`` needs the draw, because it fetches only the drawn items'
chat histories and images)::

    python scripts/aml_x1_sources.py fetch --data-dir DIR
    python scripts/aml_x1_sources.py draw --data-dir DIR --out draw.json
    python scripts/aml_x1_sources.py materialize --data-dir DIR --draw draw.json
    python scripts/aml_x1_sources.py dryrun --data-dir DIR --draw draw.json --out dryrun.json \
        [--questions-out questions.jsonl.gz]

Every large file is read as a stream (LongMemEval-S item by item, CLBench and MobileMem line by
line, MemLens in record batches), and ``--max-rss-mb`` stops the process before it can press on a
shared host's memory. MobileMem's 6.25 GB ``image.zip`` is never downloaded whole: only the drawn
users' members are read through HTTP range requests and CRC-checked by ``zipfile``.
"""

from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import statistics
import sys
import tarfile
import threading
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
import zipfile

PREREGISTRATION = "docs/preregistrations/2026-09-25-aml-c9-source-coverage-baseline.md"
SEED = 20260925
TOP_K = 100
HF = "https://huggingface.co/datasets"
GH_RAW = "https://raw.githubusercontent.com"
USER_AGENT = "RE-call-x1-stage-a/1"

# C9 request limits, from recall_aml/models.py and recall_aml/app.py at this commit.
MAX_QUERY_CHARS = 20_000
MAX_OPTIONS = 20
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_ADD_MEDIA_BYTES = 30 * 1024 * 1024
MAX_BODY_BYTES = 44 * 1024 * 1024
#: An Add is split, at message boundaries, before its decoded images reach this.
ADD_MEDIA_SPLIT_BYTES = 28 * 1024 * 1024

SOURCES = ("longmemeval_s", "scriptmem", "clbench", "personamem_v2", "memlens_32k", "mobilemem_omni")
SHORT = {
    "longmemeval_s": "lme",
    "scriptmem": "sm",
    "clbench": "clb",
    "personamem_v2": "pm2",
    "memlens_32k": "ml32",
    "mobilemem_omni": "mmo",
}


@dataclass(frozen=True)
class Pinned:
    url: str
    local: str
    sha256: str


REVISIONS = {
    "longmemeval_s": {"host": "huggingface", "repo": "xiaowu0162/longmemeval-cleaned",
                      "revision": "98d7416c24c778c2fee6e6f3006e7a073259d48f"},
    "scriptmem": {"host": "github", "repo": "memorax-ai/ScriptMem",
                  "revision": "22ac7e7e70124280d8af6100262ee7f88fff3436"},
    "clbench": {"host": "huggingface", "repo": "tencent/CL-bench",
                "revision": "b28a5832a09b0d96c0cf4c22e90d7c60ede25b80"},
    "personamem_v2": {"host": "huggingface", "repo": "bowen-upenn/PersonaMem-v2",
                      "revision": "ed956dea41521fc4499acbc63f966e0fd3c053ba"},
    "memlens_32k": {"host": "huggingface", "repo": "xiyuRenBill/MEMLENS",
                    "revision": "afa101a1907cc37db40b50d649547964387b96b7"},
    "mobilemem_omni": {"host": "huggingface", "repo": "zjunlp/MobileMem",
                       "revision": "14c086312c61b0e13cf588afd2b67a5b1664b33a"},
}


def _hf(source: str, path: str) -> str:
    meta = REVISIONS[source]
    return f"{HF}/{meta['repo']}/resolve/{meta['revision']}/{quote(path)}"


def _gh(source: str, path: str) -> str:
    meta = REVISIONS[source]
    return f"{GH_RAW}/{meta['repo']}/{meta['revision']}/{quote(path)}"


PINNED: dict[str, list[Pinned]] = {
    "longmemeval_s": [
        Pinned(_hf("longmemeval_s", "longmemeval_s_cleaned.json"),
               "longmemeval_s/longmemeval_s_cleaned.json",
               "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"),
    ],
    "scriptmem": [
        Pinned(_gh("scriptmem", "data/raw/angry.json"), "scriptmem/angry.json",
               "f9234a9d4bbd48e5d74fa94f22031ad4867a90756d285afeb1cf3bd039a11b88"),
        Pinned(_gh("scriptmem", "data/raw/enemy.json"), "scriptmem/enemy.json",
               "459f7d91a2af8f23cf01954f4bdd8afc2f5a9ab32c596571249f85e5b918294c"),
        Pinned(_gh("scriptmem", "data/raw/friends.json"), "scriptmem/friends.json",
               "58da6f3dc20644aa35ee17b427e9cab3b6596b97171c6bf28ffc5889c1d0cf04"),
        Pinned(_gh("scriptmem", "data/raw/man_earth.json"), "scriptmem/man_earth.json",
               "4d2f51bcc5a817ab38d2195845458fdb379b593928efa844032911580b2b070b"),
        Pinned(_gh("scriptmem", "data/public/manifest.json"), "scriptmem/manifest.json",
               "aafe50e7b003c08e03d951f824929564e4ecd12c997c23fd576a1e03096f1fce"),
    ],
    "clbench": [
        Pinned(_hf("clbench", "CL-bench.jsonl"), "clbench/CL-bench.jsonl",
               "d5fc88d4b2eea75c61dd40862021b6ae2fba26bd21b58e8c5e18377a763943be"),
    ],
    "personamem_v2": [
        Pinned(_hf("personamem_v2", "benchmark/text/benchmark.csv"),
               "personamem_v2/benchmark/text/benchmark.csv",
               "95f2a8a324aab7baf2af937feae12731369e2abf7cad5ab3e170594cb25a3e52"),
    ],
    "memlens_32k": [
        Pinned(_hf("memlens_32k", "dataset_32k.parquet"), "memlens_32k/dataset_32k.parquet",
               "f8d766bfc67f3c191939c7baeb00e0f2c7e89bb8461224dd9c305d306f988d44"),
        # A small git-tracked file: checked against the git blob sha1 Hugging Face states.
        Pinned(_hf("memlens_32k", "agent_subset_195.json"), "memlens_32k/agent_subset_195.json", ""),
    ],
    "mobilemem_omni": [
        Pinned(_hf("mobilemem_omni", "omni/data.jsonl"), "mobilemem_omni/data.jsonl",
               "f32049c2fc818507faba7c4eff349539cd43e4f61783a9c06deed581a1e6dab1"),
        Pinned(_hf("mobilemem_omni", "omni/filtered_questions.jsonl"),
               "mobilemem_omni/filtered_questions.jsonl",
               "113ee8b002939706ed7dd338b62a7d27bd0ab576eb7662474ee166dfe2b47af4"),
        Pinned(_hf("mobilemem_omni", "omni/questions.jsonl"), "mobilemem_omni/questions.jsonl",
               "413aeec3707f07af3b26882e9e47e96426e9916962127b6937acf3c6d670cd6b"),
    ],
}

MEMLENS_IMAGES = Pinned(_hf("memlens_32k", "release_images.tar.gz"),
                        "memlens_32k/release_images.tar.gz",
                        "8d0be814ab3ffe99ed3ea84ed04eacd9f06423c4068da2c24e4749c7ca3940f6")
#: Hugging Face LFS object id of omni/image.zip. Only members are read, never the whole archive.
MOBILEMEM_ZIP_URL = _hf("mobilemem_omni", "omni/image.zip")
MOBILEMEM_ZIP_SHA256 = "83be9d4ce715bd7bb27882d3e9e22790f8c8dbd1dd90c3444699dbead4048e17"
MOBILEMEM_ZIP_SIZE = 6_253_516_306

#: Counts each source states about itself, compared against the data in ``dryrun``.
PUBLISHED: dict[str, dict[str, Any]] = {
    "longmemeval_s": {
        "citation": "github.com/xiaowu0162/LongMemEval README: 500 questions, 30 abstention",
        "questions": 500,
        "abstention": 30,
    },
    "scriptmem": {
        "citation": "ScriptMem README 'Dataset Statistics' table and data/public/manifest.json",
        "per_script": {
            "angry": {"single_choice": 71, "multi_select": 20, "ordering": 8},
            "enemy": {"single_choice": 59, "multi_select": 27, "ordering": 8},
            "friends": {"single_choice": 99, "multi_select": 61, "ordering": 14},
            "man_earth": {"single_choice": 69, "multi_select": 15, "ordering": 6},
        },
        "questions": 457,
    },
    "clbench": {
        "citation": "tencent/CL-bench dataset card: 1,899 tasks, 4 categories, 18 sub-categories, "
                    "63.2 rubrics per context, 3.8 tasks per context",
        "tasks": 1899,
        "categories": 4,
        "sub_categories": 18,
        "rubrics_per_context": 63.2,
        "tasks_per_context": 3.8,
    },
    "personamem_v2": {
        "citation": "PersonaMem-v2 dataset card: benchmark/text/benchmark.csv holds 5000 user "
                    "queries; each has 3 incorrect answers",
        "queries": 5000,
        "incorrect_answers": 3,
    },
    "memlens_32k": {
        "citation": "MEMLENS dataset card and DATASHEET: 789 questions, 5 types; agent subset "
                    "61 IE / 35 MSR / 48 TR / 29 KU / 22 AR preserves type shares within 0.2 pp; "
                    "4,695 unique images in release_images/",
        "questions": 789,
        "types": 5,
        "agent_subset": {"information_extraction": 61, "multi_session_reasoning": 35,
                         "temporal_reasoning": 48, "knowledge_update": 29, "answer_refusal": 22},
        "unique_images": 4695,
    },
    "mobilemem_omni": {
        "citation": "github.com/zjunlp/MobileMem omni/README 'Benchmark Statistics': 16 users "
                    "(8 English, 8 Chinese), 1,589 events, 155,670 turns (48.2 per session), "
                    "19,060 images (5.88 per session), 7,415 questions by type",
        "users": 16,
        "english_users": 8,
        "events": 1589,
        "turns": 155670,
        "turns_per_session": 48.2,
        "images": 19060,
        "images_per_session": 5.88,
        "questions": 7415,
        "questions_by_type": {"single_hop": 986, "multi_hop": 1135, "knowledge_update": 1010,
                              "temporal_reasoning": 773, "implicit_preference": 1226,
                              "abstention": 1208, "visual_reasoning": 1077},
    },
}

UNAVAILABLE = {
    "scriptmem": (
        "The public release omits the script conversations. Every data/raw/*.json holds only "
        "`conversation.format_example`, a two-utterance synthetic schema example, and the "
        "README says 'the released data does not include the original script conversation "
        "text'. The questions, options and gold answers are public; the memory to Add is not, "
        "so no Add can be formed and no tenant can be measured."
    ),
}

MAPPING_NOTES = {
    "longmemeval_s": "One tenant per question (its own haystack). One Add per haystack session, "
                     "in haystack order, session_id = haystack_session_id; every message carries "
                     "the session's date as epoch ms (UTC). Search query = question.",
    "scriptmem": "Unavailable for ingest (see reason). Search requests and scorer fields are "
                 "still built: query = question text, options = the `option` list.",
    "clbench": "AML's CLBench ingest mapping is not in the pinned AML checkout; the pipeline "
               "reads `system_prompt`, `question` and `rubrics` and renders memories 'from "
               "previous conversations'. X-1 maps: one tenant per task; every message before "
               "the final user turn except the system prompt becomes Adds, one Add per "
               "user-led round; the final user turn is the Search query and the reader's "
               "question; the system prompt goes to the reader, not to memory. No dates, so no "
               "timestamps. Single-turn tasks therefore have zero Adds.",
    "personamem_v2": "Text benchmark, 32k chat histories (the official inference default). One "
                     "tenant per persona. The history is one flat message list with no dates "
                     "or session markers, so one Add per user-led round; the leading system "
                     "(persona profile) message rides in the first Add. No timestamps. Search "
                     "query = user_query content; options = correct plus 3 incorrect answers, "
                     "shuffled by a sha256 seed (AML's own shuffle uses Python hash(), which "
                     "varies per process).",
    "memlens_32k": "One tenant per question (MemLens's own memory-agent convention). One Add "
                   "per haystack session in the original order, timestamps from haystack_dates; "
                   "<image> placeholders are replaced in place by image parts, as MemLens's "
                   "build_messages does; roles lower-cased ('User' occurs). Search query = "
                   "question.",
    "mobilemem_omni": "One tenant per user. One Add per session in file order (chronological), "
                      "timestamp = event_start_time read as UTC; each dialogue turn is one "
                      "message, image turns become an image part. Question image_refs and "
                      "evidence image_path use a different directory naming from the dialogue, "
                      "so images are matched by (user, file name). Search query = question.",
}


# ---------------------------------------------------------------------------------------------
# Small utilities


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def rss_watchdog(limit_mb: int) -> None:
    """Exit hard if this process's resident set passes ``limit_mb`` (Linux only)."""
    status = Path("/proc/self/status")
    if limit_mb <= 0 or not status.exists():
        return

    def watch() -> None:
        while True:
            for line in status.read_text().splitlines():
                if line.startswith("VmRSS:") and int(line.split()[1]) > limit_mb * 1024:
                    sys.stderr.write(f"x1: RSS above {limit_mb} MB, stopping\n")
                    sys.stderr.flush()
                    os._exit(3)
            time.sleep(0.25)

    threading.Thread(target=watch, daemon=True).start()


def peak_rss_mb() -> float | None:
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def hf_etag(url: str) -> str:
    """The object id Hugging Face states for a file: LFS sha256, or the git blob sha1."""
    opener = build_opener(_NoRedirect)
    request = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with opener.open(request, timeout=60) as response:
            headers = response.headers
    except HTTPError as exc:  # the 302/307 carries the headers
        headers = exc.headers
    value = headers.get("X-Linked-ETag") or headers.get("ETag") or ""
    return value.strip().removeprefix("W/").strip('"')


def download(url: str, target: Path, sha256: str = "", *, retries: int = 4) -> str:
    """Stream ``url`` to ``target`` and return its sha256, refusing a mismatch with ``sha256``."""
    if target.exists():
        existing = sha256_file(target)
        if not sha256 or existing == sha256:
            return existing
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            digest = hashlib.sha256()
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=300) as response, partial.open("wb") as out:
                for block in iter(lambda: response.read(1 << 20), b""):
                    digest.update(block)
                    out.write(block)
            got = digest.hexdigest()
            if sha256 and got != sha256:
                partial.unlink()
                raise SystemExit(f"sha256 mismatch for {url}: {got} != {sha256}")
            partial.replace(target)
            return got
        except OSError as exc:
            if attempt == retries:
                raise
            sys.stderr.write(f"retry {attempt} for {url}: {exc}\n")
            time.sleep(2.0 * attempt)
    raise AssertionError("unreachable")


def iter_json_array(path: Path, chunk: int = 1 << 20) -> Iterator[Any]:
    """Yield the elements of a top-level JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = handle.read(chunk)
        pos = buffer.index("[") + 1
        while True:
            while True:
                while pos < len(buffer) and buffer[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buffer):
                    break
                more = handle.read(chunk)
                if not more:
                    return
                buffer, pos = more, 0
            if buffer[pos] == "]":
                return
            while True:
                try:
                    item, end = decoder.raw_decode(buffer, pos)
                    break
                except json.JSONDecodeError:
                    more = handle.read(chunk)
                    if not more:
                        raise
                    buffer, pos = buffer[pos:] + more, 0
            yield item
            buffer, pos = buffer[end:], 0


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def epoch_ms(value: datetime) -> int:
    return int(value.replace(tzinfo=timezone.utc).timestamp() * 1_000)


_SLASH_DATE = re.compile(r"^(\d{4})/(\d{2})/(\d{2}) \(\w+\) (\d{2}):(\d{2})$")


def slash_date_ms(text: str) -> int:
    """``2023/05/20 (Sat) 02:21`` (LongMemEval, MemLens) as epoch ms, read as UTC."""
    match = _SLASH_DATE.match(text.strip())
    if match is None:
        raise ValueError(f"unreadable date {text!r}")
    year, month, day, hour, minute = (int(part) for part in match.groups())
    return epoch_ms(datetime(year, month, day, hour, minute))


def dash_date_ms(text: str) -> int:
    """``2025-01-01 10:00:00`` (MobileMem) as epoch ms, read as UTC."""
    return epoch_ms(datetime.strptime(text.strip(), "%Y-%m-%d %H:%M:%S"))


def request_id(user_id: str, index: int, session_id: str) -> str:
    return "x1:" + hashlib.sha256(f"{user_id}|{index}|{session_id}".encode()).hexdigest()[:40]


def add_request(user_id: str, index: int, session_id: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "request_id": request_id(user_id, index, session_id),
        "user_id": user_id,
        "session_id": session_id,
        "messages": messages,
    }


def search_request(query: str, user_id: str, options: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query, "user_id": user_id, "top_k": TOP_K}
    if options:
        payload["options"] = options
    return payload


def sniff_mime(head: bytes) -> str | None:
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return None


@dataclass
class ImageRef:
    """An image the Add will carry; encoded only when the Add is built."""

    path: Path
    name: str

    def part(self) -> tuple[dict[str, Any], int, str | None]:
        raw = self.path.read_bytes()
        mime = sniff_mime(raw[:16])
        encoded = base64.b64encode(raw).decode("ascii")
        url = f"data:{mime or 'application/octet-stream'};base64,{encoded}"
        return {"type": "image_url", "image_url": {"url": url}}, len(raw), mime


# ---------------------------------------------------------------------------------------------
# Tenants and questions


@dataclass
class Tenant:
    source: str
    key: str
    user_id: str
    #: Yields (session_id, messages) where a message's content may hold ``ImageRef`` parts.
    sessions: Callable[[], Iterator[tuple[str, list[dict[str, Any]]]]]
    questions: list[dict[str, Any]] = field(default_factory=list)
    item_counts: dict[str, int] = field(default_factory=dict)


def user_id_for(run_id: str, source: str, key: str) -> str:
    return f"{run_id}-{SHORT[source]}-{key}"


def lme_tenant(item: dict[str, Any], run_id: str) -> Tenant:
    qid = str(item["question_id"])
    uid = user_id_for(run_id, "longmemeval_s", qid)

    def sessions() -> Iterator[tuple[str, list[dict[str, Any]]]]:
        for sid, date, turns in zip(
            item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"], strict=True
        ):
            stamp = slash_date_ms(date)
            yield str(sid), [
                {"role": str(turn["role"]), "content": str(turn["content"]), "timestamp": stamp}
                for turn in turns
            ]

    abstention = qid.endswith("_abs")
    question = {
        "source": "longmemeval_s",
        "question_id": qid,
        "user_id": uid,
        "category": item["question_type"],
        "abstention": abstention,
        "search": search_request(str(item["question"]), uid),
        "scorer": {
            "question": item["question"],
            "gold_answer": str(item["answer"]),
            "question_date": item["question_date"],
            "question_type": item["question_type"],
            "judge": "AML data/longmemeval-s/pipeline.py ACCURACY_PROMPT (binary)",
        },
        # LongMemEval scores retrieval on non-abstention questions only (README).
        "evidence": None if abstention else {"sessions": [str(s) for s in item["answer_session_ids"]]},
        "evidence_labelled_for_abstention": [str(s) for s in item["answer_session_ids"]]
        if abstention else None,
        "evidence_note": "abstention: LongMemEval scores retrieval on non-abstention questions only"
        if abstention else None,
    }
    return Tenant("longmemeval_s", qid, uid, sessions, [question],
                  {"sessions": len(item["haystack_sessions"])})


def rounds(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group a flat message list into user-led rounds; anything before the first user turn
    (a system prompt) rides with the first round."""
    groups: list[list[dict[str, Any]]] = []
    pending: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "user" and any(m.get("role") == "user" for m in pending):
            groups.append(pending)
            pending = []
        pending.append(message)
    if pending:
        groups.append(pending)
    return groups


def clbench_tenant(item: dict[str, Any], run_id: str) -> Tenant:
    meta = item["metadata"]
    task = str(meta["task_id"])
    uid = user_id_for(run_id, "clbench", task)
    messages = item["messages"]
    system_prompt = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    body = messages[1:] if system_prompt else list(messages)
    if not body or body[-1]["role"] != "user":
        raise ValueError(f"CLBench task {task} does not end with a user turn")
    prior, final = body[:-1], body[-1]
    groups = rounds(prior)

    def sessions() -> Iterator[tuple[str, list[dict[str, Any]]]]:
        for index, group in enumerate(groups):
            yield f"{task}:r{index}", [
                {"role": str(m["role"]), "content": str(m["content"])} for m in group
            ]

    question = {
        "source": "clbench",
        "question_id": task,
        "user_id": uid,
        "category": meta["context_category"],
        "search": search_request(str(final["content"]), uid),
        "scorer": {
            "system_prompt": system_prompt,
            "question": final["content"],
            "rubrics": item["rubrics"],
            "context_id": meta["context_id"],
            "context_category": meta["context_category"],
            "sub_category": meta["sub_category"],
            "judge": "AML data/clbench/pipeline.py rubric_judge_prompt (strict all-or-nothing, "
                     "plus requirement ratio)",
        },
        "evidence": None,
        "turns": len(messages),
    }
    return Tenant("clbench", task, uid, sessions, [question],
                  {"messages": len(messages), "prior_messages": len(prior), "rounds": len(groups)})


PERSONAMEM_RECALL_SUFFIX = (
    " Please recall my related preferences from our conversation history "
    "to give personalized responses."
)


def personamem_query(raw: str) -> str:
    """The query text out of the CSV's Python-literal ``{'role': ..., 'content': ...}`` string.

    Parsed as syntax only (``ast.parse`` of a dict display of string constants); nothing is
    evaluated, and anything else is returned as it stands.
    """
    text = str(raw).strip()
    if text.startswith("{"):
        try:
            node = ast.parse(text, mode="eval").body
        except SyntaxError:
            return text
        if isinstance(node, ast.Dict):
            fields = {
                key.value: value.value
                for key, value in zip(node.keys, node.values, strict=True)
                if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
            }
            return str(fields.get("content", fields.get("text", text)))
    return text


def personamem_options(row: dict[str, Any], query: str) -> tuple[list[str], str]:
    correct = str(row["correct_answer"])
    incorrect = json.loads(row["incorrect_answers"]) if row["incorrect_answers"] else []
    options = [correct, *(str(answer) for answer in incorrect)]
    seed_text = f"{row['persona_id']}_{query}{PERSONAMEM_RECALL_SUFFIX}"
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
    random.Random(seed).shuffle(options)
    letter = chr(ord("A") + options.index(correct))
    return options, letter


def content_digest(text: str) -> str:
    return hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest()


def personamem_tenant(persona: str, rows: list[dict[str, Any]], history_path: Path,
                      run_id: str) -> Tenant:
    uid = user_id_for(run_id, "personamem_v2", persona)
    history = json.loads(history_path.read_text(encoding="utf-8"))["chat_history"]
    groups = rounds(history)

    def sessions() -> Iterator[tuple[str, list[dict[str, Any]]]]:
        for index, group in enumerate(groups):
            yield f"persona{persona}:r{index}", [
                {"role": str(m["role"]), "content": str(m["content"])} for m in group
            ]

    questions = []
    for row in rows:
        query = personamem_query(row["user_query"])
        options, letter = personamem_options(row, query)
        snippet = json.loads(row["related_conversation_snippet"])
        user_turns = [str(m["content"]) for m in snippet if m.get("role") == "user"]
        # A distance of 0 marks a snippet the source kept OUT of the history: 508 of the 511
        # sensitive_info rows, whose target answer must not use the leaked detail.
        in_history = str(row["distance_from_related_snippet_to_query_32k"]).strip() not in {"", "0"}
        questions.append({
            "source": "personamem_v2",
            "question_id": row["_qid"],
            "persona_id": persona,
            "user_id": uid,
            "category": row["pref_type"],
            "search": search_request(query, uid, options),
            "scorer": {
                "user_query": query,
                "reader_query": query + PERSONAMEM_RECALL_SUFFIX,
                "options": options,
                "correct_letter": letter,
                "correct_answer": row["correct_answer"],
                "preference": row["preference"],
                "pref_type": row["pref_type"],
                "topic_query": row["topic_query"],
                "conversation_scenario": row["conversation_scenario"],
                "who": row["who"],
                "updated": row["updated"],
                "judge": "AML data/personamem/pipeline_v2.py MCQ mode, exact mapped answer",
            },
            "evidence": {"message_digests": [content_digest(text) for text in user_turns]}
            if in_history else None,
            "evidence_note": None if in_history else "snippet not in the 32k history by construction "
                                                     "(distance_from_related_snippet_to_query_32k = 0)",
        })
    return Tenant("personamem_v2", persona, uid, sessions, questions,
                  {"history_messages": len(history), "rounds": len(groups)})


def _interleave(text: str, images: list[ImageRef]) -> Any:
    """MemLens's `_append_turn_blocks`, without labels: each ``<image>`` becomes that image."""
    if not images:
        return text
    parts: list[Any] = []
    index = 0
    if "<image>" in text:
        for piece in re.split(r"(<image>)", text):
            if piece == "<image>":
                if index < len(images):
                    parts.append(images[index])
                    index += 1
            elif piece.strip():
                parts.append({"type": "text", "text": piece})
    elif text.strip():
        parts.append({"type": "text", "text": text})
    parts.extend(images[index:])
    return parts


def memlens_tenant(item: dict[str, Any], image_root: Path, run_id: str) -> Tenant:
    qid = str(item["question_id"])
    uid = user_id_for(run_id, "memlens_32k", qid)
    answer_sessions = {str(s) for s in item["answer_session_ids"] or []}

    def sessions() -> Iterator[tuple[str, list[dict[str, Any]]]]:
        for sid, date, turns in zip(
            item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"], strict=True
        ):
            stamp = slash_date_ms(date)
            messages = []
            for turn in turns:
                images = [ImageRef(image_root / str(image["file"]), str(image["file"]))
                          for image in turn["images"] or []]
                messages.append({
                    "role": str(turn["role"]).lower(),
                    "content": _interleave(str(turn["content"]), images),
                    "timestamp": stamp,
                })
            yield str(sid), messages

    evidence_images = sorted({
        str(image["file"])
        for sid, turns in zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True)
        if str(sid) in answer_sessions
        for turn in turns
        if turn["has_answer"]
        for image in turn["images"] or []
    })
    answer = str(item["answer"]).strip()
    question = {
        "source": "memlens_32k",
        "question_id": qid,
        "user_id": uid,
        "category": item["question_type"],
        "image_evidence": bool(evidence_images),
        "search": search_request(str(item["question"]), uid),
        "scorer": {
            "question": item["question"],
            "gold_answer": item["answer"],
            "question_date": item["question_date"],
            "question_type": item["question_type"],
            "answer_form": "choice (A/B)" if answer in {"A", "B"} else "open",
            "evidence_images": evidence_images,
            "judge": "MemLens answer_extraction.py + llm_judge.py at github.com/xrenaf/MEMLENS "
                     "77f3ab9a52fa2d6a17978e2dffe80438a4ecced2",
        },
        "evidence": {"sessions": sorted(answer_sessions),
                     "image_names": sorted(PurePosixPath(f).name for f in evidence_images)}
        if answer_sessions else None,
        "evidence_note": None if answer_sessions else f"no answer_session_ids ({item['question_type']})",
    }
    return Tenant("memlens_32k", qid, uid, sessions, [question],
                  {"sessions": len(item["haystack_sessions"])})


def mobilemem_uid_of(path: str) -> str:
    match = re.search(r"(?:^|/)uid(\d+)/", path)
    if match is None:
        raise ValueError(f"no uid in MobileMem image path {path!r}")
    return match.group(1)


def mobilemem_tenant(record: dict[str, Any], questions: list[dict[str, Any]],
                     image_index: dict[str, str], image_root: Path, run_id: str) -> Tenant:
    user = str(record["uuid"])
    uid = user_id_for(run_id, "mobilemem_omni", user)

    def image_for(path: str) -> ImageRef:
        key = f"{mobilemem_uid_of(path)}/{PurePosixPath(path).name}"
        local = image_index.get(key)
        return ImageRef(_safe_join(image_root, local) if local else image_root / "__absent__", path)

    def sessions() -> Iterator[tuple[str, list[dict[str, Any]]]]:
        for session in record["sessions"]:
            stamp = dash_date_ms(session["event_start_time"])
            messages = []
            for turn in session["dialogue"]:
                if turn["content_type"] == "image":
                    content: Any = [image_for(str(turn["image_inline"]))]
                else:
                    content = str(turn.get("content") or "")
                messages.append({"role": str(turn["role"]), "content": content, "timestamp": stamp})
            yield str(session["session_id"]), messages

    built = []
    for q in questions:
        evidence_images = sorted({
            PurePosixPath(str(e["image_path"])).name for e in q["evidence"] if e.get("image_path")
        } | {PurePosixPath(str(r)).name for r in q["image_refs"]})
        built.append({
            "source": "mobilemem_omni",
            "question_id": q["question_id"],
            "user_id": uid,
            "category": q["question_type"],
            "image_evidence": any(e.get("image_path") for e in q["evidence"]),
            "search": search_request(str(q["question"]), uid, q.get("options") or None),
            "scorer": {
                "question": q["question"],
                "gold_answer": q["answer"],
                "question_type": q["question_type"],
                "difficulty": q["difficulty"],
                "evidence": [
                    e["explanation"] if isinstance(e["explanation"], str)
                    else json.dumps(e["explanation"], ensure_ascii=False)
                    for e in q["evidence"]
                ],
                "image_refs": q["image_refs"],
                "judge": "MobileMem omni/eval/eval/question_answering_and_judge_prompts.txt "
                         "(binary, with evidence)",
            },
            "evidence": {"sessions": [str(s) for s in q["source_session_ids"]],
                         "image_names": evidence_images} if q["source_session_ids"] else None,
            "evidence_note": None if q["source_session_ids"]
            else f"no source_session_ids ({q['question_type']})",
        })
    return Tenant("mobilemem_omni", user, uid, sessions, built,
                  {"sessions": len(record["sessions"])})


def scriptmem_questions(data_dir: Path, run_id: str) -> list[dict[str, Any]]:
    """AML's `load_gold_records` ids (``source:sample_id#qNNNN``); no Adds exist to pair with."""
    out = []
    for name in ("angry", "enemy", "friends", "man_earth"):
        data = json.loads((data_dir / "scriptmem" / f"{name}.json").read_text(encoding="utf-8"))
        uid = user_id_for(run_id, "scriptmem", name)
        for sample_index, sample in enumerate(data):
            sample_id = sample.get("sample_id") or f"{name}-{sample_index}"
            for qa_index, qa in enumerate(sample.get("qa", [])):
                out.append({
                    "source": "scriptmem",
                    "question_id": f"{name}:{sample_id}#q{qa_index:04d}",
                    "user_id": uid,
                    "category": qa["qa_type"],
                    "script": name,
                    "search": search_request(str(qa["question"]), uid, [str(o) for o in qa["option"]]),
                    "scorer": {"question": qa["question"], "answer": qa["answer"],
                               "qa_type": qa["qa_type"],
                               "judge": "AML data/scriptmem/pipeline.py evaluate_official (exact)"},
                    "evidence": None,
                })
    return out


# ---------------------------------------------------------------------------------------------
# fetch


def cmd_fetch(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    for source in args.sources:
        for pinned in PINNED[source]:
            target = data_dir / pinned.local
            got = download(pinned.url, target, pinned.sha256)
            if not pinned.sha256:
                expected = hf_etag(pinned.url)
                if expected not in {got, git_blob_sha1(target)}:
                    raise SystemExit(f"{pinned.url}: neither sha256 nor blob sha1 match {expected}")
            print(f"{source}: {pinned.local} {got}", flush=True)


# ---------------------------------------------------------------------------------------------
# draw


def allocate(population: dict[str, int], total: int, mode: str) -> dict[str, int]:
    """Per-category sample sizes: ``equal`` (total / categories each) or ``proportional``
    (largest remainder, ties broken by category name)."""
    categories = sorted(population)
    if mode == "equal":
        base = total // len(categories)
        if base * len(categories) != total:
            raise ValueError("equal allocation needs total divisible by the category count")
        want = {c: base for c in categories}
    else:
        size = sum(population.values())
        exact = {c: total * population[c] / size for c in categories}
        want = {c: int(exact[c]) for c in categories}
        rest = total - sum(want.values())
        for c in sorted(categories, key=lambda c: (-(exact[c] - want[c]), c))[:rest]:
            want[c] += 1
    for c in categories:
        if want[c] > population[c]:
            raise ValueError(f"category {c} has {population[c]} < {want[c]} requested")
    return want


def stratified(ids_by_category: dict[str, list[str]], total: int, mode: str) -> dict[str, Any]:
    rng = random.Random(SEED)
    population = {c: len(v) for c, v in ids_by_category.items()}
    want = allocate(population, total, mode)
    drawn: dict[str, list[str]] = {}
    for category in sorted(ids_by_category):
        drawn[category] = sorted(rng.sample(sorted(ids_by_category[category]), want[category]))
    return {"population": dict(sorted(population.items())), "drawn_per_category": want,
            "ids_by_category": drawn}


def file_record(data_dir: Path, source: str) -> dict[str, Any]:
    out = {}
    for pinned in PINNED[source]:
        path = data_dir / pinned.local
        out[pinned.local] = {"url": pinned.url, "sha256": sha256_file(path), "bytes": path.stat().st_size,
                             "pinned_sha256": pinned.sha256 or None}
    return out


def personamem_rows(data_dir: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    csv.field_size_limit(sys.maxsize)
    with (data_dir / PINNED["personamem_v2"][0].local).open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            qid = f"row{index:04d}"
            row["_qid"] = qid
            yield qid, row


def drawn_ids(draw: dict[str, Any], source: str) -> set[str]:
    return {q for ids in draw["sources"][source]["ids_by_category"].values() for q in ids}


def cmd_draw(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    rss_watchdog(args.max_rss_mb)
    data_dir = Path(args.data_dir)
    result: dict[str, Any] = {
        "preregistration": PREREGISTRATION,
        "seed": SEED,
        "rng": "random.Random(20260925), a fresh generator per source; categories visited in "
               "sorted order; ids sorted before rng.sample",
        "script_sha256": sha256_file(Path(__file__)),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {},
    }
    sources = result["sources"]

    by_type: dict[str, list[str]] = defaultdict(list)
    for item in iter_json_array(data_dir / PINNED["longmemeval_s"][0].local):
        by_type[item["question_type"]].append(str(item["question_id"]))
    sources["longmemeval_s"] = {
        "status": "available", **REVISIONS["longmemeval_s"],
        "files": file_record(data_dir, "longmemeval_s"),
        "category_field": "question_type", "allocation": "equal, 20 per type (record)",
        "tenant": "per question", **stratified(by_type, 120, "equal"),
    }

    sm_ids: dict[str, list[str]] = defaultdict(list)
    for question in scriptmem_questions(data_dir, "x1"):
        sm_ids[question["category"]].append(question["question_id"])
    sources["scriptmem"] = {
        "status": "unavailable", "reason": UNAVAILABLE["scriptmem"], **REVISIONS["scriptmem"],
        "files": file_record(data_dir, "scriptmem"), "category_field": "qa_type",
        "allocation": "all questions (record)", "tenant": "per script",
        "population": {c: len(v) for c, v in sorted(sm_ids.items())},
        "drawn_per_category": {c: len(v) for c, v in sorted(sm_ids.items())},
        "ids_by_category": {c: sorted(v) for c, v in sorted(sm_ids.items())},
    }

    cl_ids: dict[str, list[str]] = defaultdict(list)
    for item in iter_jsonl(data_dir / PINNED["clbench"][0].local):
        cl_ids[item["metadata"]["context_category"]].append(str(item["metadata"]["task_id"]))
    sources["clbench"] = {
        "status": "available", **REVISIONS["clbench"], "files": file_record(data_dir, "clbench"),
        "category_field": "metadata.context_category",
        "allocation": "proportional, largest remainder (record: 200 items, stratified)",
        "tenant": "per task", **stratified(cl_ids, 200, "proportional"),
    }

    pm_ids: dict[str, list[str]] = defaultdict(list)
    pm_persona: dict[str, str] = {}
    for qid, row in personamem_rows(data_dir):
        pm_ids[row["pref_type"]].append(qid)
        pm_persona[qid] = row["persona_id"]
    pm = stratified(pm_ids, 200, "proportional")
    drawn_pm = [q for ids in pm["ids_by_category"].values() for q in ids]
    sources["personamem_v2"] = {
        "status": "available", **REVISIONS["personamem_v2"],
        "files": file_record(data_dir, "personamem_v2"),
        "split": "benchmark/text/benchmark.csv, chat_history_32k",
        "question_id_rule": "0-based row index in benchmark.csv as 'row<NNNN>' (the CSV has no id)",
        "category_field": "pref_type",
        "allocation": "proportional, largest remainder (record: 200 questions, stratified)",
        "tenant": "per persona", **pm,
        "personas": sorted({pm_persona[q] for q in drawn_pm}, key=int),
    }

    table = pq.read_table(data_dir / PINNED["memlens_32k"][0].local,
                          columns=["question_id", "question_type"])
    ml_ids: dict[str, list[str]] = defaultdict(list)
    for qid, qtype in zip(table.column("question_id").to_pylist(),
                          table.column("question_type").to_pylist(), strict=True):
        ml_ids[qtype].append(qid)
    del table
    sources["memlens_32k"] = {
        "status": "available", **REVISIONS["memlens_32k"], "files": file_record(data_dir, "memlens_32k"),
        "category_field": "question_type",
        "allocation": "proportional, largest remainder (record: 120 of 789, stratified by type)",
        "tenant": "per question", **stratified(ml_ids, 120, "proportional"),
    }

    question_file = data_dir / "mobilemem_omni/filtered_questions.jsonl"
    english = sorted(int(r["uuid"]) for r in iter_jsonl(question_file) if r["language"] == "en")
    users = sorted(random.Random(SEED).sample(english, 2))
    mm_ids: dict[str, list[str]] = defaultdict(list)
    for record in iter_jsonl(question_file):
        if int(record["uuid"]) in users:
            for q in record["questions"]:
                mm_ids[q["question_type"]].append(str(q["question_id"]))
    sources["mobilemem_omni"] = {
        "status": "available", **REVISIONS["mobilemem_omni"],
        "files": file_record(data_dir, "mobilemem_omni"),
        "question_file": "omni/filtered_questions.jsonl",
        "english_users": english,
        "user_rule": "random.Random(20260925).sample(sorted English uuids, 2)",
        "users": users, "category_field": "question_type",
        "allocation": "all questions of the drawn users (record)", "tenant": "per user",
        "population": {c: len(v) for c, v in sorted(mm_ids.items())},
        "drawn_per_category": {c: len(v) for c, v in sorted(mm_ids.items())},
        "ids_by_category": {c: sorted(v) for c, v in sorted(mm_ids.items())},
        "image_zip": {"url": MOBILEMEM_ZIP_URL, "lfs_sha256": MOBILEMEM_ZIP_SHA256,
                      "bytes": MOBILEMEM_ZIP_SIZE},
    }
    for entry in sources.values():
        entry["drawn_total"] = sum(entry["drawn_per_category"].values())
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({s: v["drawn_total"] for s, v in sources.items()}))


# ---------------------------------------------------------------------------------------------
# materialize


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file through HTTP range requests."""

    def __init__(self, url: str, size: int) -> None:
        self.url, self.size, self.pos, self.requests = url, size, 0, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer: Any) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(buffer)) - 1
        data = b""
        for attempt in range(1, 6):
            try:
                request = Request(self.url, headers={"Range": f"bytes={self.pos}-{end}",
                                                     "User-Agent": USER_AGENT})
                with urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise OSError(f"expected 206, got {response.status}")
                    data = response.read()
                break
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(2.0 * attempt)
        self.requests += 1
        count = len(data)
        buffer[:count] = data
        self.pos += count
        return count


def _safe_join(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe member path {relative!r}")
    return root.joinpath(*path.parts)


def memlens_items(data_dir: Path, wanted: set[str]) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(data_dir / PINNED["memlens_32k"][0].local)
    for batch in parquet.iter_batches(batch_size=16):
        for item in batch.to_pylist():
            if item["question_id"] in wanted:
                yield item


def materialize_personamem(data_dir: Path, draw: dict[str, Any]) -> dict[str, Any]:
    """The drawn personas' 32k chat histories, each checked against the object id Hugging Face
    states for it (the git blob sha1, for these small files)."""
    wanted = drawn_ids(draw, "personamem_v2")
    links = {row["persona_id"]: row["chat_history_32k_link"]
             for qid, row in personamem_rows(data_dir) if qid in wanted}
    histories = {}
    for persona, link in sorted(links.items(), key=lambda kv: int(kv[0])):
        url = _hf("personamem_v2", link)
        target = _safe_join(data_dir / "personamem_v2", link)
        digest = download(url, target)
        expected = hf_etag(url)
        blob = git_blob_sha1(target)
        if expected not in {digest, blob}:
            raise SystemExit(f"{link}: stated object id {expected} matches neither sha256 nor blob")
        histories[link] = {"sha256": digest, "git_blob_sha1": blob}
    print(f"personamem_v2: {len(histories)} chat histories", flush=True)
    return {"chat_histories": len(histories), "files": histories}


def materialize_memlens(data_dir: Path, draw: dict[str, Any]) -> dict[str, Any]:
    """The pinned image tarball, then only the drawn questions' images out of it."""
    tar_digest = download(MEMLENS_IMAGES.url, data_dir / MEMLENS_IMAGES.local, MEMLENS_IMAGES.sha256)
    needed = {
        "release_images/" + str(image["file"])
        for item in memlens_items(data_dir, drawn_ids(draw, "memlens_32k"))
        for turns in item["haystack_sessions"] for turn in turns for image in turn["images"] or []
    }
    members = image_members = extracted = 0
    with tarfile.open(data_dir / MEMLENS_IMAGES.local, mode="r|gz") as archive:
        for member in archive:
            members += 1
            name = member.name.removeprefix("./")
            if member.isfile() and name.startswith("release_images/"):
                image_members += 1
            if member.isfile() and name in needed:
                target = _safe_join(data_dir / "memlens_32k", name)
                if not (target.exists() and target.stat().st_size == member.size):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    assert source is not None
                    target.write_bytes(source.read())
                extracted += 1
    missing = sorted(n for n in needed if not (data_dir / "memlens_32k" / n).exists())
    print(f"memlens_32k: {extracted}/{len(needed)} images, {len(missing)} missing", flush=True)
    return {"tarball_sha256": tar_digest, "tar_members": members, "tar_image_files": image_members,
            "needed": len(needed), "extracted": extracted, "missing": missing}


def materialize_mobilemem(data_dir: Path, draw: dict[str, Any]) -> dict[str, Any]:
    """The drawn users' dialogue images, read out of image.zip by HTTP range request."""
    users = {str(u) for u in draw["sources"]["mobilemem_omni"]["users"]}
    remote = RangeFile(MOBILEMEM_ZIP_URL, MOBILEMEM_ZIP_SIZE)
    archive_zip = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20))
    by_key: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    for info in archive_zip.infolist():
        if info.is_dir():
            continue
        match = re.match(r"uid(\d+)/", info.filename)
        if match and match.group(1) in users:
            by_key[f"{match.group(1)}/{PurePosixPath(info.filename).name}"].append(info)
    needed_paths: set[str] = set()
    for record in iter_jsonl(data_dir / PINNED["mobilemem_omni"][0].local):
        if str(record["uuid"]) in users:
            needed_paths.update(
                str(turn["image_inline"]) for session in record["sessions"]
                for turn in session["dialogue"] if turn["content_type"] == "image"
            )
    index: dict[str, str] = {}
    ambiguous: list[str] = []
    absent: list[str] = []
    fetched = 0
    image_root = data_dir / "mobilemem_omni/images"
    for number, path in enumerate(sorted(needed_paths), 1):
        key = f"{mobilemem_uid_of(path)}/{PurePosixPath(path).name}"
        infos = by_key.get(key, [])
        if len(infos) != 1:
            (ambiguous if infos else absent).append(path)
            continue
        info = infos[0]
        target = _safe_join(image_root, info.filename)
        if not (target.exists() and target.stat().st_size == info.file_size):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive_zip.read(info))  # zipfile verifies the CRC-32
            fetched += 1
        index[key] = info.filename
        if number % 250 == 0:
            print(f"mobilemem_omni: {number}/{len(needed_paths)} images", flush=True)
    image_root.mkdir(parents=True, exist_ok=True)
    index_path = image_root / "index.json"
    index_path.write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    return {"users": sorted(users, key=int), "needed": len(needed_paths), "resolved": len(index),
            "fetched_now": fetched, "ambiguous": ambiguous, "absent": absent,
            "range_requests": remote.requests, "zip_members": len(archive_zip.infolist()),
            "zip_lfs_sha256": MOBILEMEM_ZIP_SHA256, "index_sha256": sha256_file(index_path)}


MATERIALIZERS: dict[str, Callable[[Path, dict[str, Any]], dict[str, Any]]] = {
    "personamem_v2": materialize_personamem,
    "memlens_32k": materialize_memlens,
    "mobilemem_omni": materialize_mobilemem,
}


def cmd_materialize(args: argparse.Namespace) -> None:
    rss_watchdog(args.max_rss_mb)
    data_dir = Path(args.data_dir)
    draw = json.loads(Path(args.draw).read_text(encoding="utf-8"))
    out = data_dir / "materialize.json"
    report: dict[str, Any] = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for source in args.sources:
        report[source] = MATERIALIZERS[source](data_dir, draw)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "files"} for k, v in report.items()}
    print(json.dumps(summary, default=str)[:4000])


# ---------------------------------------------------------------------------------------------
# dryrun


def _hits() -> list[int]:
    return [0, 0]


@dataclass
class SourceStats:
    items: int = 0
    tenants: int = 0
    adds: int = 0
    text_only_adds: int = 0
    empty_adds: int = 0
    split_adds: int = 0
    messages: int = 0
    blank_messages: int = 0
    image_parts: int = 0
    image_bytes: int = 0
    text_chars: int = 0
    text_only_add_chars: int = 0
    body_bytes: int = 0
    max_add_body_bytes: int = 0
    max_add_media_bytes: int = 0
    max_image_bytes: int = 0
    missing_images: int = 0
    unknown_mime: int = 0
    mime_extension_mismatch: int = 0
    over_limit: list[str] = field(default_factory=list)
    zero_add_tenants: int = 0
    timestamps: int = 0
    questions: int = 0
    per_category: Counter[str] = field(default_factory=Counter)
    query_over_20k: int = 0
    query_chars: list[int] = field(default_factory=list)
    evidence_labelled: int = 0
    evidence_present: int = 0
    evidence_by_category: defaultdict[str, list[int]] = field(default_factory=lambda: defaultdict(_hits))
    image_evidence: Counter[str] = field(default_factory=Counter)
    image_evidence_present: list[int] = field(default_factory=_hits)
    evidence_notes: Counter[str] = field(default_factory=Counter)
    validation_failures: int = 0
    validation_examples: list[str] = field(default_factory=list)
    adds_per_tenant: list[int] = field(default_factory=list)
    duplicate_request_ids: int = 0
    digest: Any = field(default_factory=hashlib.sha256)


def _text_len(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    return sum(len(p["text"]) for p in content if isinstance(p, dict) and p.get("type") == "text")


def _blank(content: Any) -> bool:
    return not content.strip() if isinstance(content, str) else not content


_EXTENSION_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                   ".webp": "image/webp", ".gif": "image/gif"}


def build_adds(tenant: Tenant, stats: SourceStats) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Materialize each Add with its images encoded, splitting a session whose images would pass
    the media budget at a message boundary; yield (add, facts)."""
    index = 0
    for session_id, messages in tenant.sessions():
        encoded: list[tuple[dict[str, Any], int, list[str]]] = []
        for message in messages:
            content = message["content"]
            media = 0
            names: list[str] = []
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, ImageRef):
                        parts.append(part)
                        continue
                    if not part.path.is_file():
                        stats.missing_images += 1
                        continue
                    image, size, mime = part.part()
                    media += size
                    stats.max_image_bytes = max(stats.max_image_bytes, size)
                    if size > MAX_IMAGE_BYTES:
                        stats.over_limit.append(f"image over 10 MiB: {part.name}")
                    if mime is None:
                        stats.unknown_mime += 1
                    elif _EXTENSION_MIME.get(part.path.suffix.lower()) != mime:
                        stats.mime_extension_mismatch += 1
                    names.append(PurePosixPath(part.name).name)
                    parts.append(image)
                content = parts
            encoded.append(({**message, "content": content}, media, names))
        chunks: list[list[tuple[dict[str, Any], int, list[str]]]] = [[]]
        chunk_media = 0
        for entry in encoded:
            if chunks[-1] and chunk_media + entry[1] > ADD_MEDIA_SPLIT_BYTES:
                chunks.append([])
                chunk_media = 0
            chunks[-1].append(entry)
            chunk_media += entry[1]
        stats.split_adds += len(chunks) - 1
        for chunk in chunks:
            add = add_request(tenant.user_id, index, session_id, [m for m, _, _ in chunk])
            index += 1
            yield add, {"session_id": session_id, "media": sum(m for _, m, _ in chunk),
                        "image_names": [n for _, _, names in chunk for n in names]}


def c9_validators() -> tuple[Any, Any] | None:
    """C9's own request models, when this checkout's ``recall_aml`` is importable."""
    try:
        from recall_aml.models import AddRequest, SearchRequest
    except ImportError:
        return None
    return AddRequest, SearchRequest


VALIDATORS = c9_validators()


def _validate(model: Any, payload: dict[str, Any], stats: SourceStats) -> None:
    if model is None:
        return
    try:
        model.model_validate(payload)
    except Exception as exc:  # BROAD-CATCH: counted and reported, the dry run continues
        stats.validation_failures += 1
        if len(stats.validation_examples) < 5:
            stats.validation_examples.append(f"{type(exc).__name__}: {str(exc)[:300]}")


def dry_run_tenant(tenant: Tenant, stats: SourceStats, seen_ids: set[str]) -> dict[str, Any]:
    add_model, search_model = VALIDATORS or (None, None)
    session_ids: set[str] = set()
    user_digests: set[str] = set()
    image_names: set[str] = set()
    adds = 0
    for add, facts in build_adds(tenant, stats):
        adds += 1
        if add["request_id"] in seen_ids:
            stats.duplicate_request_ids += 1
        seen_ids.add(add["request_id"])
        session_ids.add(facts["session_id"])
        image_names.update(facts["image_names"])
        _validate(add_model, add, stats)
        body = canonical(add)
        stats.digest.update(hashlib.sha256(body).digest())
        stats.body_bytes += len(body)
        stats.max_add_body_bytes = max(stats.max_add_body_bytes, len(body))
        stats.max_add_media_bytes = max(stats.max_add_media_bytes, facts["media"])
        if len(body) > MAX_BODY_BYTES:
            stats.over_limit.append(f"Add body over 44 MiB: {add['request_id']}")
        if facts["media"] > MAX_ADD_MEDIA_BYTES:
            stats.over_limit.append(f"Add media over 30 MiB: {add['request_id']}")
        images_here = 0
        chars = 0
        for message in add["messages"]:
            stats.messages += 1
            if _blank(message["content"]):
                stats.blank_messages += 1
            if "timestamp" in message:
                stats.timestamps += 1
            chars += _text_len(message["content"])
            if isinstance(message["content"], list):
                images_here += sum(1 for p in message["content"] if p.get("type") == "image_url")
            if message["role"] == "user" and isinstance(message["content"], str):
                user_digests.add(content_digest(message["content"]))
        stats.adds += 1
        stats.image_parts += images_here
        stats.image_bytes += facts["media"]
        stats.text_chars += chars
        if not add["messages"] or all(_blank(m["content"]) for m in add["messages"]):
            stats.empty_adds += 1
        elif images_here == 0:
            stats.text_only_adds += 1
            stats.text_only_add_chars += chars
    stats.tenants += 1
    stats.adds_per_tenant.append(adds)
    if adds == 0:
        stats.zero_add_tenants += 1
    present_count = labelled_count = 0
    for question in tenant.questions:
        stats.questions += 1
        stats.per_category[question["category"]] += 1
        _validate(search_model, question["search"], stats)
        query = question["search"]["query"]
        stats.query_chars.append(len(query))
        if len(query) > MAX_QUERY_CHARS:
            stats.query_over_20k += 1
        if len(question["search"].get("options") or []) > MAX_OPTIONS:
            stats.over_limit.append(f"options over 20: {question['question_id']}")
        if "image_evidence" in question:
            stats.image_evidence["image" if question["image_evidence"] else "text"] += 1
        if question.get("evidence_note"):
            stats.evidence_notes[question["evidence_note"]] += 1
        evidence = question.get("evidence")
        if not evidence:
            continue
        present = True
        if "sessions" in evidence:
            present = set(evidence["sessions"]) <= session_ids
        if "message_digests" in evidence:
            present = present and set(evidence["message_digests"]) <= user_digests
        if evidence.get("image_names"):
            stats.image_evidence_present[0] += set(evidence["image_names"]) <= image_names
            stats.image_evidence_present[1] += 1
        stats.evidence_labelled += 1
        stats.evidence_present += present
        cell = stats.evidence_by_category[question["category"]]
        cell[0] += present
        cell[1] += 1
        present_count += present
        labelled_count += 1
    return {"adds": adds, "evidence_present": present_count, "evidence_labelled": labelled_count}


def summarize(stats: SourceStats) -> dict[str, Any]:
    q = sorted(stats.query_chars)
    per_tenant = stats.adds_per_tenant
    return {
        "items": stats.items,
        "tenants": stats.tenants,
        "zero_add_tenants": stats.zero_add_tenants,
        "adds": stats.adds,
        "text_only_adds": stats.text_only_adds,
        "adds_with_images": stats.adds - stats.text_only_adds - stats.empty_adds,
        "empty_adds": stats.empty_adds,
        "adds_split_for_media_budget": stats.split_adds,
        "adds_per_tenant": {"min": min(per_tenant, default=0),
                            "median": statistics.median(per_tenant) if per_tenant else 0,
                            "max": max(per_tenant, default=0)},
        "messages": stats.messages,
        "blank_messages_dropped_by_c9": stats.blank_messages,
        "messages_with_timestamp": stats.timestamps,
        "image_parts": stats.image_parts,
        "image_decoded_bytes": stats.image_bytes,
        "missing_images": stats.missing_images,
        "unknown_image_format": stats.unknown_mime,
        "image_extension_mime_mismatch": stats.mime_extension_mismatch,
        "max_image_bytes": stats.max_image_bytes,
        "max_add_media_bytes": stats.max_add_media_bytes,
        "max_add_body_bytes": stats.max_add_body_bytes,
        "total_text_chars": stats.text_chars,
        "total_add_body_bytes": stats.body_bytes,
        "over_c9_limits": stats.over_limit[:50],
        "over_c9_limits_count": len(stats.over_limit),
        "duplicate_request_ids": stats.duplicate_request_ids,
        "c9_model_validation": {
            "checked": VALIDATORS is not None,
            "failures": stats.validation_failures,
            "examples": stats.validation_examples,
            "models": "recall_aml.models.AddRequest and SearchRequest of this checkout",
        },
        "ingest_compile_estimate": {
            "text_only_adds": stats.text_only_adds,
            "text_only_add_chars": stats.text_only_add_chars,
            "note": "C9 compiles each text-only Add once (DeepSeek); Adds carrying an image skip "
                    "the compiler, and an Add of blank messages stores nothing.",
        },
        "questions": stats.questions,
        "questions_per_category": dict(sorted(stats.per_category.items())),
        "search_query_chars": {"median": statistics.median(q) if q else 0, "max": max(q, default=0),
                               "over_20000_bounded_by_c9": stats.query_over_20k},
        "evidence": {
            "labelled_questions": stats.evidence_labelled,
            "present_in_tenant_adds": stats.evidence_present,
            "share": round(stats.evidence_present / stats.evidence_labelled, 4)
            if stats.evidence_labelled else None,
            "by_category": {c: {"present": v[0], "labelled": v[1]}
                            for c, v in sorted(stats.evidence_by_category.items())},
            "evidence_images_present": {"present": stats.image_evidence_present[0],
                                        "labelled": stats.image_evidence_present[1]}
            if stats.image_evidence_present[1] else None,
            "unlabelled_by_source": dict(stats.evidence_notes) or None,
        },
        "evidence_modality": dict(stats.image_evidence) or None,
        "adds_sha256": stats.digest.hexdigest(),
    }


def check(name: str, published: Any, observed: Any, *, tolerance: float | None = None) -> dict[str, Any]:
    if published is None:
        return {"what": name, "published": None, "observed": observed, "match": None}
    if tolerance is None:
        match = published == observed
    else:
        match = abs(float(published) - float(observed)) <= tolerance
    return {"what": name, "published": published, "observed": observed, "match": match}


def published_checks(source: str, data_dir: Path) -> list[dict[str, Any]]:
    pub = PUBLISHED[source]
    checks: list[dict[str, Any]] = []
    if source == "longmemeval_s":
        total = abstention = 0
        for item in iter_json_array(data_dir / PINNED[source][0].local):
            total += 1
            abstention += str(item["question_id"]).endswith("_abs")
        checks += [check("questions", pub["questions"], total),
                   check("abstention questions", pub["abstention"], abstention)]
    elif source == "scriptmem":
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for question in scriptmem_questions(data_dir, "x1"):
            counts[question["script"]][question["category"]] += 1
        for script, expected in pub["per_script"].items():
            checks.append(check(f"{script} questions by qa_type", expected, dict(sorted(counts[script].items()))))
        checks.append(check("questions", pub["questions"], sum(sum(c.values()) for c in counts.values())))
        manifest = json.loads((data_dir / "scriptmem/manifest.json").read_text(encoding="utf-8"))
        for name, entry in manifest["datasets"].items():
            checks.append(check(f"manifest sha256 of {name}.json vs the released file",
                                entry["sha256"], sha256_file(data_dir / "scriptmem" / f"{name}.json")))
    elif source == "clbench":
        tasks = rubrics = 0
        categories: set[str] = set()
        subs: set[str] = set()
        contexts: Counter[str] = Counter()
        for item in iter_jsonl(data_dir / PINNED[source][0].local):
            tasks += 1
            categories.add(item["metadata"]["context_category"])
            subs.add(item["metadata"]["sub_category"])
            contexts[item["metadata"]["context_id"]] += 1
            rubrics += len(item["rubrics"])
        checks += [check("tasks", pub["tasks"], tasks),
                   check("context categories", pub["categories"], len(categories)),
                   check("sub-categories", pub["sub_categories"], len(subs)),
                   check("tasks per context", pub["tasks_per_context"], round(tasks / len(contexts), 2),
                         tolerance=0.05),
                   check("rubrics per context", pub["rubrics_per_context"],
                         round(rubrics / len(contexts), 2), tolerance=0.05)]
    elif source == "personamem_v2":
        rows = wrong = 0
        for _, row in personamem_rows(data_dir):
            rows += 1
            wrong += len(json.loads(row["incorrect_answers"] or "[]")) != pub["incorrect_answers"]
        checks += [check("benchmark queries", pub["queries"], rows),
                   check("rows without exactly 3 incorrect answers", 0, wrong)]
    elif source == "memlens_32k":
        import pyarrow.parquet as pq

        table = pq.read_table(data_dir / PINNED[source][0].local, columns=["question_id", "question_type"])
        types = Counter(table.column("question_type").to_pylist())
        checks += [check("questions", pub["questions"], table.num_rows),
                   check("question types", pub["types"], len(types))]
        agent = json.loads((data_dir / "memlens_32k/agent_subset_195.json").read_text(encoding="utf-8"))
        ids = agent.get("question_ids", agent) if isinstance(agent, dict) else agent
        in_32k = set(ids) & set(table.column("question_id").to_pylist())
        checks.append(check("agent subset ids present in the 32k file", len(set(ids)), len(in_32k)))
        size = sum(types.values())
        agent_total = sum(pub["agent_subset"].values())
        worst = max(abs(pub["agent_subset"][t] / agent_total - types[t] / size) for t in pub["agent_subset"])
        checks.append(check("agent-subset type shares vs full set, max gap in pp (published: under 0.2)",
                            0.0, round(worst * 100, 3), tolerance=0.2))
        materialized = data_dir / "materialize.json"
        if materialized.exists():
            tar = json.loads(materialized.read_text(encoding="utf-8"))["memlens_32k"]
            checks.append(check("image files in release_images.tar.gz", pub["unique_images"],
                                tar["tar_image_files"]))
    elif source == "mobilemem_omni":
        # The benchmark's users are the 16 in filtered_questions.jsonl; questions.jsonl and
        # data.jsonl also carry uuids 8, 9, 18 and 19, which the published figures leave out.
        benchmark: set[int] = set()
        english: set[int] = set()
        filtered: Counter[str] = Counter()
        for record in iter_jsonl(data_dir / "mobilemem_omni/filtered_questions.jsonl"):
            benchmark.add(int(record["uuid"]))
            if record["language"] == "en":
                english.add(int(record["uuid"]))
            filtered.update(q["question_type"] for q in record["questions"])
        by_type: Counter[str] = Counter()
        all_users: set[int] = set()
        for record in iter_jsonl(data_dir / "mobilemem_omni/questions.jsonl"):
            all_users.add(int(record["uuid"]))
            if int(record["uuid"]) in benchmark:
                by_type.update(q["question_type"] for q in record["questions"])
        checks += [check("users (filtered_questions.jsonl)", pub["users"], len(benchmark)),
                   check("English users (filtered_questions.jsonl)", pub["english_users"], len(english)),
                   check("questions.jsonl users, all (reference only; no published figure)", None,
                         len(all_users)),
                   check("questions (questions.jsonl, benchmark users)", pub["questions"],
                         sum(by_type.values())),
                   check("questions by type (questions.jsonl, benchmark users)",
                         pub["questions_by_type"], dict(sorted(by_type.items()))),
                   check("filtered questions (no published count; for reference)", None,
                         sum(filtered.values()))]
        sessions = turns = images = 0
        events: set[tuple[int, str]] = set()
        for record in iter_jsonl(data_dir / PINNED[source][0].local):
            if int(record["uuid"]) not in benchmark:
                continue
            for session in record["sessions"]:
                sessions += 1
                events.add((int(record["uuid"]), str(session["event_id"]).split("_")[0]))
                turns += len(session["dialogue"])
                images += sum(t["content_type"] == "image" for t in session["dialogue"])
        checks += [check("dialogue turns, benchmark users (data.jsonl)", pub["turns"], turns),
                   check("turns per session", pub["turns_per_session"], round(turns / sessions, 2),
                         tolerance=0.05),
                   check("image turns, benchmark users (data.jsonl)", pub["images"], images),
                   check("images per session", pub["images_per_session"], round(images / sessions, 2),
                         tolerance=0.01),
                   check("root events (user, event_id before '_'), benchmark users", pub["events"],
                         len(events))]
    return checks


def tenants_for(source: str, data_dir: Path, draw: dict[str, Any], run_id: str) -> Iterator[Tenant]:
    wanted = drawn_ids(draw, source)
    if source == "longmemeval_s":
        for item in iter_json_array(data_dir / PINNED[source][0].local):
            if str(item["question_id"]) in wanted:
                yield lme_tenant(item, run_id)
    elif source == "clbench":
        for item in iter_jsonl(data_dir / PINNED[source][0].local):
            if str(item["metadata"]["task_id"]) in wanted:
                yield clbench_tenant(item, run_id)
    elif source == "personamem_v2":
        by_persona: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for qid, row in personamem_rows(data_dir):
            if qid in wanted:
                by_persona[row["persona_id"]].append(row)
        for persona in sorted(by_persona, key=int):
            rows = by_persona[persona]
            path = _safe_join(data_dir / "personamem_v2", rows[0]["chat_history_32k_link"])
            yield personamem_tenant(persona, rows, path, run_id)
    elif source == "memlens_32k":
        for item in memlens_items(data_dir, wanted):
            yield memlens_tenant(item, data_dir / "memlens_32k/release_images", run_id)
    elif source == "mobilemem_omni":
        users = {int(u) for u in draw["sources"][source]["users"]}
        questions = {int(r["uuid"]): r["questions"]
                     for r in iter_jsonl(data_dir / "mobilemem_omni/filtered_questions.jsonl")
                     if int(r["uuid"]) in users}
        index_path = data_dir / "mobilemem_omni/images/index.json"
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
        for record in iter_jsonl(data_dir / PINNED[source][0].local):
            if int(record["uuid"]) in users:
                chosen = [q for q in questions[int(record["uuid"])] if str(q["question_id"]) in wanted]
                yield mobilemem_tenant(record, chosen, index, data_dir / "mobilemem_omni/images", run_id)


def cmd_dryrun(args: argparse.Namespace) -> None:
    rss_watchdog(args.max_rss_mb)
    data_dir = Path(args.data_dir)
    draw = json.loads(Path(args.draw).read_text(encoding="utf-8"))
    started = time.time()
    out: dict[str, Any] = {
        "preregistration": PREREGISTRATION,
        "draw_sha256": sha256_file(Path(args.draw)),
        "script_sha256": sha256_file(Path(__file__)),
        "run_id": args.run_id,
        "sent": "nothing: no Add, Search, model or embedding call is made by this command",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "c9_limits": {"max_query_chars": MAX_QUERY_CHARS, "max_options": MAX_OPTIONS,
                      "max_image_bytes": MAX_IMAGE_BYTES, "max_add_media_bytes": MAX_ADD_MEDIA_BYTES,
                      "max_body_bytes": MAX_BODY_BYTES, "split_add_media_at": ADD_MEDIA_SPLIT_BYTES},
        "c9_models_module": None,
        "sources": {},
    }
    if VALIDATORS is not None:
        out["c9_models_module"] = sys.modules[VALIDATORS[0].__module__].__file__
    materialized = data_dir / "materialize.json"
    if materialized.exists():
        out["materialize"] = json.loads(materialized.read_text(encoding="utf-8"))
        out["materialize"]["personamem_v2"].pop("files", None)
        out["materialize_sha256"] = sha256_file(materialized)
    questions_out = gzip.open(args.questions_out, "wt", encoding="utf-8") if args.questions_out else None
    seen_ids: set[str] = set()
    try:
        for source in args.sources:
            entry: dict[str, Any] = {"status": draw["sources"][source]["status"],
                                     "revision": REVISIONS[source],
                                     "mapping": MAPPING_NOTES[source]}
            stats = SourceStats()
            wanted = drawn_ids(draw, source)
            if source == "scriptmem":
                entry["reason"] = UNAVAILABLE[source]
                for question in scriptmem_questions(data_dir, args.run_id):
                    if question["question_id"] not in wanted:
                        continue
                    stats.items += 1
                    stats.questions += 1
                    stats.per_category[question["category"]] += 1
                    _validate(VALIDATORS[1] if VALIDATORS else None, question["search"], stats)
                    stats.query_chars.append(len(question["search"]["query"]))
                    if questions_out:
                        questions_out.write(json.dumps(question, ensure_ascii=False) + "\n")
                entry.update(summarize(stats))
                entry.update({"tenants": None, "adds": None, "text_only_adds": None,
                              "adds_sha256": None})
            else:
                item_counts: Counter[str] = Counter()
                for tenant in tenants_for(source, data_dir, draw, args.run_id):
                    stats.items += len(tenant.questions)
                    item_counts.update(tenant.item_counts)
                    per = dry_run_tenant(tenant, stats, seen_ids)
                    if questions_out:
                        for question in tenant.questions:
                            questions_out.write(json.dumps(question, ensure_ascii=False) + "\n")
                    if args.verbose:
                        print(f"{source} {tenant.key}: {per}", flush=True)
                entry.update(summarize(stats))
                entry["source_item_totals"] = dict(item_counts)
            entry["drawn_questions_built"] = check("drawn questions built",
                                                   draw["sources"][source]["drawn_total"], stats.questions)
            entry["published_count_checks"] = published_checks(source, data_dir)
            out["sources"][source] = entry
            print(f"{source}: done at {time.time() - started:.0f}s", flush=True)
    finally:
        if questions_out:
            questions_out.close()
    out["peak_rss_mb"] = peak_rss_mb()
    out["wall_seconds"] = round(time.time() - started, 1)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    keys = ("tenants", "adds", "text_only_adds", "image_parts", "total_text_chars", "questions")
    print(json.dumps({s: {k: v.get(k) for k in keys} for s, v in out["sources"].items()}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "draw", "materialize", "dryrun"):
        command = sub.add_parser(name)
        command.add_argument("--data-dir", required=True)
        command.add_argument("--max-rss-mb", type=int, default=900)
        if name == "fetch":
            command.add_argument("--sources", nargs="+", default=list(SOURCES), choices=SOURCES)
        if name == "draw":
            command.add_argument("--out", required=True)
        if name == "materialize":
            command.add_argument("--draw", required=True)
            command.add_argument("--sources", nargs="+", default=list(MATERIALIZERS),
                                 choices=list(MATERIALIZERS))
        if name == "dryrun":
            command.add_argument("--draw", required=True)
            command.add_argument("--out", required=True)
            command.add_argument("--questions-out", default="")
            command.add_argument("--run-id", default="x1")
            command.add_argument("--sources", nargs="+", default=list(SOURCES), choices=SOURCES)
            command.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    commands = {"fetch": cmd_fetch, "draw": cmd_draw, "materialize": cmd_materialize, "dryrun": cmd_dryrun}
    commands[args.command](args)


if __name__ == "__main__":
    main()
