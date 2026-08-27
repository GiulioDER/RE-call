"""Recover the frozen benchmark sources and author their discoverability variants.

    python -u scripts/agent_ab_prepare_discoverability_corpus.py \
        --evidence-dsn postgresql://recall:recall@127.0.0.1:5407/agent_ab \
        --out benchmarks/artifacts/agent_ab/discoverability-probe

Preregistered in `docs/preregistrations/2026-08-27-memo-discoverability-authoring.md`. Four
outputs, side by side and diffable:

- `sources-control/`: the frozen corpus recovered exactly as the alias probe recovered it (sha
  verified live files, drifted files reconstructed from the generation's own chunks with the
  learned joiner and flagged in the report).
- `sources-retitle/`: each memo gains a searcher-oriented TITLE as its first body line, and its
  frontmatter `description:` is replaced by a searcher-oriented one. Nothing else changes.
- `sources-restructured/`: retitle, plus a "You need this when" section of 5 task-intent
  phrasings immediately after the title, BEFORE the original body. The placement is the point:
  the failed alias probe appended at the bottom, where the section merges into the last chunk.
- `sources-pointer/`: each memo verbatim, plus a separate `<stem>--tasks.md` pointer document
  holding the title, the task phrasings, and a link to the memo. Separate documents are the
  variant the alias probe's dilution mechanism predicts behaves differently.

All three variants are assembled from ONE generation per memo, so the arms differ only in
structure, never in generated content. The four index files are never touched.

⛔ This script never reads the benchmark archive. The generator must not see the recorded
queries it will later be judged against, and the cleanest guarantee is structural: nothing here
imports or opens them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from agent_ab_prepare_alias_corpus import (  # noqa: E402
    INDEX_FILES,
    joiner_hits,
    learn_joiner,
    rows,
    sha256_bytes,
)

REWRITE_MODEL = "anthropic/claude-sonnet-5"
TASKS_HEADER = "## You need this when"
#: Fixed and generic: it teaches the searcher's point of view and names nothing from any task.
REWRITE_PROMPT = (
    "You write retrieval surfaces for engineering postmortem notes. The reader who needs a note "
    "has NOT hit its failure yet: they are planning or starting an ordinary task, they search in "
    "the task's own vocabulary, and they do not know the note exists. From the note below, and "
    "nothing else, write:\n"
    '- "title": one line naming the situation the reader is in when the note should reach them. '
    "Task vocabulary first, naming the concrete operations involved (tools, commands, file "
    "types), then the surprise.\n"
    '- "description": one sentence matchable from both directions: what the reader is trying to '
    "do, and what will go wrong.\n"
    '- "tasks": 5 short task phrases an engineer could be starting when this note should '
    "interrupt them. Phrase them as goals in plain task vocabulary, naming the concrete "
    "artifacts the note involves; do not mention the failure, its symptoms, or its fix in "
    "these.\n"
    "Return strict JSON with exactly those three keys and no other text.\n\nNote:\n"
)


def generate_rewrite(text: str, key: str) -> dict:
    body = json.dumps(
        {
            "model": REWRITE_MODEL,
            "temperature": 0,
            "max_tokens": 2000,
            # Sonnet 5 reasons by default through OpenRouter, and the reasoning spend counts
            # toward max_tokens: an 800 budget returned 90 characters of content cut mid-string.
            "reasoning": {"enabled": False},
            "messages": [{"role": "user", "content": REWRITE_PROMPT + text[:6000]}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(4):
        # Transport retries only: the prompt, model and inputs never change between attempts.
        try:
            with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
            break
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            time.sleep(5 * (attempt + 1))
    else:
        raise SystemExit(f"generation failed after retries: {last_error}")
    content = payload["choices"][0]["message"]["content"].strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Deterministic repair for the one malformation seen: a literal newline inside a JSON
        # string, which temperature 0 reproduces identically on every retry.
        parsed = json.loads(re.sub(r"[\r\n]+", " ", content))
    title = str(parsed["title"]).strip()
    description = str(parsed["description"]).strip()
    tasks = [str(t).strip() for t in parsed["tasks"] if str(t).strip()][:5]
    if not title or not description or len(tasks) < 3:
        raise SystemExit(f"generation returned an unusable triple: {content[:300]}")
    return {"title": title, "description": description, "tasks": tasks}


FRONTMATTER = re.compile(r"\A(---\r?\n.*?\r?\n---\r?\n)", re.DOTALL)
DESCRIPTION_LINE = re.compile(r"^description:.*$", re.MULTILINE)


def split_frontmatter(text: str) -> tuple[str, str]:
    match = FRONTMATTER.match(text)
    if not match:
        return "", text
    return match.group(1), text[match.end() :]


def with_description(frontmatter: str, description: str) -> str:
    replacement = f"description: {json.dumps(description)}"
    if DESCRIPTION_LINE.search(frontmatter):
        return DESCRIPTION_LINE.sub(replacement.replace("\\", "\\\\"), frontmatter, count=1)
    # A reconstructed file can lack the field; add it before the closing fence.
    return re.sub(r"\r?\n---\r?\n\Z", f"\n{replacement}\n---\n", frontmatter, count=1)


def retitle_text(text: str, rewrite: dict) -> str:
    frontmatter, body = split_frontmatter(text)
    head = with_description(frontmatter, rewrite["description"]) if frontmatter else ""
    return f"{head}\n# {rewrite['title']}\n\n{body.lstrip()}"


def restructured_text(text: str, rewrite: dict) -> str:
    frontmatter, body = split_frontmatter(text)
    head = with_description(frontmatter, rewrite["description"]) if frontmatter else ""
    bullets = "\n".join(f"- {task}" for task in rewrite["tasks"])
    return (
        f"{head}\n# {rewrite['title']}\n\n{TASKS_HEADER}\n\n{bullets}\n\n{body.lstrip()}"
    )


def pointer_text(stem: str, original_description: str, rewrite: dict) -> str:
    bullets = "\n".join(f"- {task}" for task in rewrite["tasks"])
    described = f": {original_description}" if original_description else "."
    return (
        "---\n"
        f"name: {stem}--tasks\n"
        f"description: {json.dumps(rewrite['description'])}\n"
        "---\n\n"
        f"# {rewrite['title']}\n\n"
        f"{TASKS_HEADER}\n\n{bullets}\n\n"
        f"Read [[{stem}]] before proceeding{described}\n"
    )


def original_description(text: str) -> str:
    frontmatter, _ = split_frontmatter(text)
    match = DESCRIPTION_LINE.search(frontmatter)
    if not match:
        return ""
    value = match.group(0).split(":", 1)[1].strip()
    return value.strip("\"'")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dsn", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    from urllib.parse import unquote, urlparse

    out = Path(args.out)
    dirs = {
        name: out / f"sources-{name}"
        for name in ("control", "retitle", "restructured", "pointer")
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    by_source: dict[str, dict] = {}
    for uri, sha, ordinal, text in rows(args.evidence_dsn):
        entry = by_source.setdefault(uri, {"sha": sha, "chunks": {}})
        entry["chunks"][int(ordinal)] = text
    print(f"{len(by_source)} sources in the frozen generation")

    verified: list[tuple[str, list[str]]] = []
    for uri, entry in by_source.items():
        path = Path(unquote(urlparse(uri).path.lstrip("/")))
        if len(entry["chunks"]) < 2 or not path.is_file():
            continue
        data = path.read_bytes()
        if sha256_bytes(data) == entry["sha"]:
            ordered = [entry["chunks"][k] for k in sorted(entry["chunks"])]
            verified.append((data.decode("utf-8"), ordered))
    joiner = learn_joiner(verified)
    reproduced = sum(joiner_hits(joiner, verified))
    print(
        f"joiner learned from {len(verified)} sha-verified multi-chunk files: "
        f"{joiner!r} reproduces {reproduced}/{len(verified)}"
    )

    kept, reconstructed = 0, []
    for uri, entry in sorted(by_source.items()):
        path = Path(unquote(urlparse(uri).path.lstrip("/")))
        name = path.name
        if path.is_file() and sha256_bytes(path.read_bytes()) == entry["sha"]:
            data = path.read_bytes()
            kept += 1
        else:
            ordered = [entry["chunks"][k] for k in sorted(entry["chunks"])]
            data = joiner.join(ordered).encode("utf-8")
            reconstructed.append(name)
        (dirs["control"] / name).write_bytes(data)
    print(f"recovered: {kept} sha-verified live files, {len(reconstructed)} reconstructed")

    memos = sorted(p for p in dirs["control"].glob("*.md") if p.name not in INDEX_FILES)
    print(f"generating searcher-oriented surfaces for {len(memos)} memos with {REWRITE_MODEL}")

    def one(path: Path) -> tuple[str, dict]:
        try:
            rewrite = generate_rewrite(path.read_text(encoding="utf-8"), key)
        except Exception as error:
            raise SystemExit(f"generation failed for {path.name}: {error}") from error
        return path.name, rewrite

    rewrites: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (name, rewrite) in enumerate(pool.map(one, memos), start=1):
            rewrites[name] = rewrite
            print(f"  [{index}/{len(memos)}] {name}: {rewrite['title'][:70]}")

    for path in sorted(dirs["control"].glob("*.md")):
        raw = path.read_bytes()
        if path.name in INDEX_FILES:
            for name in ("retitle", "restructured", "pointer"):
                (dirs[name] / path.name).write_bytes(raw)
            continue
        text = raw.decode("utf-8")
        rewrite = rewrites[path.name]
        stem = path.stem
        variants = {
            "retitle": retitle_text(text, rewrite),
            "restructured": restructured_text(text, rewrite),
        }
        for name, content in variants.items():
            # newline='\n' on purpose: python-write-text-crlf-churn.md is IN this corpus.
            with (dirs[name] / path.name).open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
        (dirs["pointer"] / path.name).write_bytes(raw)
        pointer = pointer_text(stem, original_description(text), rewrite)
        with (dirs["pointer"] / f"{stem}--tasks.md").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(pointer)

    report = {
        "sources": len(by_source),
        "sha_verified": kept,
        "reconstructed": reconstructed,
        "joiner": joiner,
        "joiner_reproduced": f"{reproduced}/{len(verified)}",
        "rewrite_model": REWRITE_MODEL,
        "rewrite_prompt": REWRITE_PROMPT,
        "rewritten_files": len(rewrites),
        "rewrites": rewrites,
    }
    (out / "rewrites.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"\nwrote {out / 'rewrites.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
