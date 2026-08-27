"""Recover the frozen benchmark sources and author their discoverability variants.

    OPENROUTER_API_KEY=... python -u scripts/agent_ab_prepare_discoverability_corpus.py \
        --evidence-dsn postgresql://recall:recall@127.0.0.1:<evidence-port>/agent_ab \
        --out benchmarks/artifacts/agent_ab/discoverability-probe

`OPENROUTER_API_KEY` is required. `--workers` (default 4) bounds the concurrent generations, and
`--allow-lossy-reconstruction` is the explicit override for a corpus whose drifted sources cannot
be rebuilt faithfully. The evidence port is per checkout, from `scripts/session-db.sh`.

Preregistered in `docs/preregistrations/2026-08-27-memo-discoverability-authoring.md`. Four
outputs, side by side and diffable:

- `sources-control/`: the frozen corpus recovered exactly as the alias probe recovered it (sha
  verified live files, drifted files reconstructed from the generation's own chunks with the
  learned joiner and flagged in the report).
- `sources-retitle/`: each memo gains a searcher-oriented TITLE as its first body line and a
  searcher-oriented DESCRIPTION as the line under it, and its frontmatter `description:` is
  replaced by the same sentence (synthesizing the frontmatter block when the source has none).
  Nothing else changes.
- `sources-restructured/`: retitle, plus a "You need this when" section of 5 task-intent
  phrasings immediately after the description, BEFORE the original body. The placement is the
  point: the failed alias probe appended at the bottom, where the section merges into the last
  chunk.
- `sources-pointer/`: each memo verbatim, plus a separate `<stem>--tasks.md` pointer document
  holding the title, the description, the task phrasings, the memo's ORIGINAL description and a
  link to the memo. Separate documents are the variant the alias probe's dilution mechanism
  predicts behaves differently.

⛔ **Every treated surface is emitted as BODY text, never as frontmatter alone.** The corpus
builder chunks `parse_document(text).human_body`, so a `description:` key reaches neither the
embedding nor the lexical index. The 2026-08-27 run wrote the description only into frontmatter
and therefore measured the retitle arm as title-only; see `treated_head`.

All three variants are assembled from ONE generation per memo, so the arms differ only in
structure, never in generated content. The four index files are never touched.

Outputs beside the four trees: `rewrites.json` (the audit record — model, prompt, every sampling
parameter, a terms fingerprint, per-memo task counts, the truncated and synthesized-frontmatter
lists, and every generated triple verbatim) and `rewrites.partial.json` (an atomically written
resume cache, stamped with the generation terms and discarded when they change).

⛔ This script never reads the benchmark archive. The generator must not see the recorded
queries it will later be judged against, and the cleanest guarantee is structural: nothing here
imports or opens them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agent_ab_prepare_alias_corpus import (  # noqa: E402
    INDEX_FILES,
    joiner_hits,
    learn_joiner,
    rows,
    sha256_bytes,
)
from recall.manifest import local_path_for  # noqa: E402

REWRITE_MODEL = "anthropic/claude-sonnet-5"
REWRITE_TEMPERATURE = 0
REWRITE_MAX_TOKENS = 2000
REWRITE_REASONING = False
REWRITE_INPUT_CHARS = 6000
TASKS_PER_MEMO = 5
MIN_TASKS = 3
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


class GenerationError(Exception):
    """A generation that failed for this memo.

    Deliberately an Exception rather than a SystemExit: `SystemExit` derives from BaseException,
    so the `except Exception` in the worker could not catch it, and the handler that attaches the
    memo name was dead on exactly the two failures it was written for (retry exhaustion and an
    unusable triple). Found by the 2026-08-27 audit.
    """


def terms_fingerprint(
    *,
    model: str | None = None,
    prompt: str | None = None,
    temperature: int | None = None,
    max_tokens: int | None = None,
    reasoning: bool | None = None,
    input_chars: int | None = None,
) -> str:
    """A digest of everything that decides what the generator returns.

    The resume cache is keyed on this, not on the filename alone. The generation terms changed
    twice in two days on this file (a JSON parser repair, then reasoning disabled), while
    `rewrites.json` attests the CURRENT constants over whatever the cache happened to hold, so a
    reused cache could silently make that attestation false.
    """

    payload = json.dumps(
        {
            "model": REWRITE_MODEL if model is None else model,
            "prompt": REWRITE_PROMPT if prompt is None else prompt,
            "temperature": REWRITE_TEMPERATURE if temperature is None else temperature,
            "max_tokens": REWRITE_MAX_TOKENS if max_tokens is None else max_tokens,
            "reasoning": REWRITE_REASONING if reasoning is None else reasoning,
            "input_chars": REWRITE_INPUT_CHARS if input_chars is None else input_chars,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_cache(path: Path, rewrites: dict) -> None:
    """Persist the resume cache atomically, stamped with the terms that produced it.

    `write_text` truncates in place, and this runs once per completed memo, so an interrupt
    during a write left unparseable JSON: the cache existed to survive exactly that interrupt.
    """

    payload = {"terms": terms_fingerprint(), "rewrites": rewrites}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def load_cache(path: Path) -> dict:
    """Return the cached rewrites, or an empty dict when the cache cannot be trusted.

    Three ways it cannot: it is torn (an interrupted write), it predates the stamped format, or
    it was produced under different generation terms. None of those is fatal, because the only
    cost of discarding a cache is re-buying the generations it held; silently REUSING one is what
    would corrupt the audit record.
    """

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        print(f"  resume cache unreadable ({error}); regenerating from scratch")
        return {}
    if not isinstance(payload, dict) or payload.get("terms") != terms_fingerprint():
        print("  resume cache was written under different generation terms; discarding it")
        return {}
    cached = payload.get("rewrites")
    return cached if isinstance(cached, dict) else {}


def reconstruction_is_trustworthy(*, reproduced: int, verified: int) -> tuple[bool, str]:
    """Is the learned joiner good enough to rebuild a drifted source from its chunks?

    `learn_joiner` returns the FIRST candidate on an all-zero tie, which is the empty string, the
    one that glues the tail of one chunk to the head of the next. The 2026-08-27 run recorded
    `'' reproduces 0/167` and rebuilt 25 of 194 sources with it anyway, stripping their
    frontmatter and gluing their chunk boundaries. Zero measured fidelity licenses nothing.
    """

    if verified <= 0:
        return False, "no sha-verified multi-chunk file exists, so no joiner can be measured"
    if reproduced < verified:
        return (
            False,
            f"the learned joiner reproduces only {reproduced}/{verified} sha-verified files, "
            "so every reconstruction from it is known-approximate",
        )
    return True, ""


def generate_rewrite(text: str, key: str) -> dict:
    body = json.dumps(
        {
            "model": REWRITE_MODEL,
            "temperature": REWRITE_TEMPERATURE,
            "max_tokens": REWRITE_MAX_TOKENS,
            # Sonnet 5 reasons by default through OpenRouter, and the reasoning spend counts
            # toward max_tokens: an 800 budget returned 90 characters of content cut mid-string.
            "reasoning": {"enabled": REWRITE_REASONING},
            "messages": [
                {"role": "user", "content": REWRITE_PROMPT + text[:REWRITE_INPUT_CHARS]}
            ],
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
        raise GenerationError(f"generation failed after retries: {last_error}")
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
    tasks = [str(t).strip() for t in parsed["tasks"] if str(t).strip()][:TASKS_PER_MEMO]
    if not title or not description or len(tasks) < MIN_TASKS:
        raise GenerationError(f"generation returned an unusable triple: {content[:300]}")
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


def treated_head(text: str, rewrite: dict, stem: str) -> str:
    """Frontmatter carrying the generated description, synthesized when the source has none.

    ⛔ **Frontmatter alone is NOT a treatment.** `recall/generations.py` chunks
    `parse_document(text).human_body`, which is the document with its frontmatter removed, so a
    `description:` key reaches neither the embedding nor the lexical index. Measured on the
    2026-08-27 corpora: the generated description appeared in indexed chunk text for 0 of 40
    memos while the generated title appeared in 40 of 40. That is why every caller of this
    function also emits the description as BODY text; the frontmatter copy is for a human reader
    and for `recall_search`'s metadata, not for retrieval.

    Synthesizing the block when it is absent closes the second half of the same defect: 27 of 190
    memos in the shipped run had no frontmatter (21 of them because chunk reconstruction had
    destroyed it), and for those the description was dropped instead of inserted.
    """

    frontmatter, _ = split_frontmatter(text)
    if frontmatter:
        return with_description(frontmatter, rewrite["description"])
    return (
        f"---\nname: {json.dumps(stem)}\n"
        f"description: {json.dumps(rewrite['description'])}\n---\n"
    )


def retitle_text(text: str, rewrite: dict, stem: str) -> str:
    _, body = split_frontmatter(text)
    head = treated_head(text, rewrite, stem)
    return (
        f"{head}\n# {rewrite['title']}\n\n{rewrite['description']}\n\n{body.lstrip()}"
    )


def restructured_text(text: str, rewrite: dict, stem: str) -> str:
    _, body = split_frontmatter(text)
    head = treated_head(text, rewrite, stem)
    bullets = "\n".join(f"- {task}" for task in rewrite["tasks"])
    return (
        f"{head}\n# {rewrite['title']}\n\n{rewrite['description']}\n\n"
        f"{TASKS_HEADER}\n\n{bullets}\n\n{body.lstrip()}"
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
        # As body text, not only in the frontmatter above, for the reason in `treated_head`.
        f"{rewrite['description']}\n\n"
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
    parser.add_argument(
        "--allow-lossy-reconstruction",
        action="store_true",
        help="build even when the learned joiner cannot reproduce the files it is measured on",
    )
    args = parser.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

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
        # `local_path_for` is recall's single implementation of this decision; re-deriving it with
        # urlparse + unquote is the WRONG decoder (its docstring says so) and yields a RELATIVE
        # path on POSIX, where every sha check then fails and the whole corpus is silently
        # reconstructed instead.
        path = local_path_for(uri)
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

    # Classify BEFORE writing anything, so a refused run leaves no lossy corpus on disk for a
    # later reader to mistake for a good one.
    resolved: list[tuple[str, bytes]] = []
    kept, reconstructed = 0, []
    for uri, entry in sorted(by_source.items()):
        path = local_path_for(uri)
        name = path.name
        # ONE read, hashed and kept. Reading twice meant the bytes copied into the corpus were
        # not the bytes whose digest was checked, against a store other sessions write during a run.
        data = path.read_bytes() if path.is_file() else None
        if data is not None and sha256_bytes(data) == entry["sha"]:
            kept += 1
        else:
            ordered = [entry["chunks"][k] for k in sorted(entry["chunks"])]
            data = joiner.join(ordered).encode("utf-8")
            reconstructed.append(name)
        resolved.append((name, data))
    print(f"recovered: {kept} sha-verified live files, {len(reconstructed)} to reconstruct")

    trustworthy, reason = reconstruction_is_trustworthy(
        reproduced=reproduced, verified=len(verified)
    )
    if reconstructed and not trustworthy:
        if not args.allow_lossy_reconstruction:
            raise SystemExit(
                f"REFUSED: {len(reconstructed)} source(s) need reconstruction and {reason}.\n"
                "  Every arm would be built on documents whose frontmatter is gone and whose "
                "chunk boundaries are glued together.\n"
                "  Re-run with --allow-lossy-reconstruction only if the affected sources cannot "
                "govern any scored session, and say so in the record.\n"
                f"  affected: {', '.join(sorted(reconstructed))}"
            )
        print(f"  WARNING: building on approximate reconstructions anyway: {reason}")

    for name, data in resolved:
        (dirs["control"] / name).write_bytes(data)

    memos = sorted(p for p in dirs["control"].glob("*.md") if p.name not in INDEX_FILES)

    # Resume cache: two runs have already died mid-pass (a malformed response, then a network
    # drop), and at temperature 0 a finished triple never changes FOR THE SAME TERMS, so completed
    # memos are kept across runs rather than re-bought. `load_cache` enforces the "same terms"
    # half, which the first version of this cache did not.
    partial = out / "rewrites.partial.json"
    rewrites: dict[str, dict] = load_cache(partial)
    if rewrites:
        print(f"resuming: {len(rewrites)} rewrites already generated under the current terms")
    todo = [p for p in memos if p.name not in rewrites]
    print(
        f"generating searcher-oriented surfaces for {len(todo)} of {len(memos)} memos "
        f"with {REWRITE_MODEL}"
    )

    def one(path: Path) -> tuple[str, dict]:
        try:
            rewrite = generate_rewrite(path.read_text(encoding="utf-8"), key)
        except Exception as error:
            raise GenerationError(f"{path.name}: {error}") from error
        return path.name, rewrite

    # `submit` + `as_completed`, never `map`: map yields strictly in submission order, so a
    # failure discarded the already-paid results of every worker that had finished behind it.
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, path): path for path in todo}
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                name, rewrite = future.result()
            except GenerationError as error:
                failures.append(str(error))
                print(f"  [{index}/{len(todo)}] FAILED {error}")
                continue
            rewrites[name] = rewrite
            write_cache(partial, rewrites)
            print(f"  [{index}/{len(todo)}] {name}: {rewrite['title'][:70]}")
    if failures:
        raise SystemExit(
            f"{len(failures)} memo(s) failed to generate; the cache holds the rest, so a re-run "
            "resumes rather than re-buying:\n  " + "\n  ".join(failures)
        )

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
            "retitle": retitle_text(text, rewrite, stem),
            "restructured": restructured_text(text, rewrite, stem),
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
        # The rest of what decides a generation, so the artifact can distinguish surfaces built
        # under different terms rather than attesting the current constants over all of them.
        "rewrite_temperature": REWRITE_TEMPERATURE,
        "rewrite_max_tokens": REWRITE_MAX_TOKENS,
        "rewrite_reasoning": REWRITE_REASONING,
        "rewrite_input_chars": REWRITE_INPUT_CHARS,
        "terms_fingerprint": terms_fingerprint(),
        "truncated_sources": sorted(
            p.name
            for p in memos
            if len(p.read_text(encoding="utf-8")) > REWRITE_INPUT_CHARS
        ),
        "synthesized_frontmatter": sorted(
            p.name for p in memos if not split_frontmatter(p.read_text(encoding="utf-8"))[0]
        ),
        "task_counts": {name: len(r["tasks"]) for name, r in sorted(rewrites.items())},
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
