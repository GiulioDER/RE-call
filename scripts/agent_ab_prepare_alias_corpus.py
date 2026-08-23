"""Recover the frozen benchmark sources and build their alias-augmented twin.

    python -u scripts/agent_ab_prepare_alias_corpus.py \
        --evidence-dsn postgresql://recall:recall@127.0.0.1:5407/agent_ab \
        --out benchmarks/artifacts/agent_ab/alias-probe

Two outputs, side by side and diffable:

- `sources-control/`: the corpus EXACTLY as `agent-ab-skill-001` searched it. The live memory
  store has drifted since the 2026-08-21 build (it now contains memos ABOUT the benchmark, which
  name the tasks and their governing memos), so the live directory is contaminated as a source.
  Recovery goes through the evidence corpus instead, read-only: for each of the 194 `source_uri`s
  in the frozen generation, the live file is used only when its sha256 still equals the recorded
  `source_sha256`; otherwise the text is reconstructed from the generation's own chunks and the
  file is flagged in the report. The joiner used for reconstruction is not guessed: it is learned
  from the sha-verified multi-chunk sources, where the true file text is known.
- `sources-aliased/`: the same files, each memo appended with a "Tasks that can lead here"
  section of task-intent queries generated from THE MEMO TEXT ALONE by the fixed prompt below.
  The four index files are left unaliased. Every generated alias is recorded verbatim in
  `aliases.json` for audit.

⛔ This script never reads the benchmark archive. The alias generator must not see the recorded
queries it will later be judged against, and the cleanest guarantee is structural: nothing here
imports or opens them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_FILES = {"MEMORY.md", "feedback_index.md", "project_index.md", "reference_index.md"}
ALIAS_MODEL = "anthropic/claude-haiku-4.5"
ALIAS_HEADER = "## Tasks that can lead here"
#: Fixed and generic: it teaches the memo-to-goals move and names nothing from any task.
ALIAS_PROMPT = (
    "You index engineering postmortem notes for retrieval. Agents search this store with "
    "task-intent queries describing what they are about to do, not what will go wrong. Given one "
    "note, write 5 short queries an agent might issue while planning or starting a task that this "
    "note's failure would strike. Phrase them as goals in plain task vocabulary; do not mention "
    "the failure, its symptoms, or its fix. One per line, no numbering, no other text.\n\n"
    "Note:\n{text}"
)


def rows(evidence_dsn: str) -> list[tuple[str, str, int, str]]:
    import psycopg

    with psycopg.connect(evidence_dsn, connect_timeout=20) as conn:
        return conn.execute(
            "SELECT source_uri, source_sha256, chunk_ordinal, text FROM recall_chunks_v1 "
            "WHERE tenant_id = 'default' ORDER BY source_uri, chunk_ordinal"
        ).fetchall()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def learn_joiner(verified: list[tuple[str, list[str]]]) -> str:
    """Find the joiner that reproduces sha-verified multi-chunk files from their chunks."""

    candidates = ["", "\n", "\n\n"]
    for joiner in candidates:
        ok = 0
        for text, chunks in verified:
            if joiner.join(chunks) == text:
                ok += 1
        if verified and ok == len(verified):
            return joiner
    # No joiner reproduces every file byte for byte; take the best and let the caller report it.
    best = max(candidates, key=lambda j: sum(joiner_hits(j, verified)))
    return best


def joiner_hits(joiner: str, verified: list[tuple[str, list[str]]]) -> list[bool]:
    return [joiner.join(chunks) == text for text, chunks in verified]


def expand_aliases(text: str, key: str) -> list[str]:
    body = json.dumps(
        {
            "model": ALIAS_MODEL,
            "temperature": 0,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": ALIAS_PROMPT.format(text=text[:6000])}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed https URL
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    lines = [line.strip().lstrip("-*0123456789. ") for line in content.splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"alias generation returned nothing: {content[:200]}")
    return lines[:5]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dsn", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    out = Path(args.out)
    control = out / "sources-control"
    aliased = out / "sources-aliased"
    control.mkdir(parents=True, exist_ok=True)
    aliased.mkdir(parents=True, exist_ok=True)

    by_source: dict[str, dict] = {}
    for uri, sha, ordinal, text in rows(args.evidence_dsn):
        entry = by_source.setdefault(uri, {"sha": sha, "chunks": {}})
        entry["chunks"][int(ordinal)] = text
    print(f"{len(by_source)} sources in the frozen generation")

    # Learn the reconstruction joiner from files that are still bit-identical on disk.
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
    print(f"joiner learned from {len(verified)} sha-verified multi-chunk files: "
          f"{joiner!r} reproduces {reproduced}/{len(verified)}")

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
        (control / name).write_bytes(data)
    print(f"recovered: {kept} sha-verified live files, {len(reconstructed)} reconstructed from chunks")
    if reconstructed:
        print("  reconstructed:", ", ".join(reconstructed[:10]) + ("..." if len(reconstructed) > 10 else ""))

    aliases: dict[str, list[str]] = {}
    for path in sorted(control.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if path.name in INDEX_FILES:
            (aliased / path.name).write_bytes(path.read_bytes())
            continue
        lines = expand_aliases(text, key)
        aliases[path.name] = lines
        augmented = text.rstrip() + f"\n\n{ALIAS_HEADER}\n\n" + "\n".join(f"- {line}" for line in lines) + "\n"
        # newline='\n' on purpose: python-write-text-crlf-churn.md is IN this corpus.
        with (aliased / path.name).open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(augmented)
        print(f"  aliased {path.name}")

    report = {
        "sources": len(by_source),
        "sha_verified": kept,
        "reconstructed": reconstructed,
        "joiner": joiner,
        "joiner_reproduced": f"{reproduced}/{len(verified)}",
        "alias_model": ALIAS_MODEL,
        "alias_prompt": ALIAS_PROMPT,
        "aliased_files": len(aliases),
        "aliases": aliases,
    }
    (out / "aliases.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"\nwrote {out / 'aliases.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
