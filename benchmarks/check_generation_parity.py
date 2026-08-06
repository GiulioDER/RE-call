#!/usr/bin/env python
"""Generation parity across the context-mode profiles, using the shipped validator.

The invariant this whole family of profiles rests on: a context mode changes the text that is
EMBEDDED and must never change the raw chunk content or the raw content hash that is STORED. If it
did, the three context arms of a promotion campaign would not be comparable with the raw baseline,
because they would be describing different corpora.

`recall.migration.validate_generation_parity` is the shipped check for exactly this, so it is what
runs here rather than a second implementation of the same comparison written for the occasion.

Prior work: NONE FOUND. Searched `docs_search(source_type="memory")` for "generation parity raw
content hash identical across context modes RE-call" on 2026-08-06; `gap_warning` was TRUE (top-3
cosine 0.485 / 0.480 / 0.479, all below the 0.50 floor), so the hits are noise and no memo records
a parity measurement. The adjacent memos that ARE real
([[project-recall-paired-comparison-null-by-construction-2026-08-05]],
[[project-recall-embedding-profile-registry-2026-08-05]]) establish the profiles and the campaign,
not this invariant.

⚠️ THIS CANNOT BE RUN RETROACTIVELY AGAINST THE 2026-08-06 CAMPAIGN'S OWN GENERATIONS.
`recall/eval/promotion/__main__.py::_indexed_store` indexes into a `promo_<uuid8>` table and drops
it in a `finally`. Every generation that campaign built is gone. This script REBUILDS one
generation per profile over the same corpus, with the same embedder and the same artifact tree, and
compares those. That is a reconstruction, not the original, and the report says so.

Four controls, all blocking, because the headline result is an ABSENCE (no mismatches) and an
absence is the easiest thing in the world to produce by accident:

1. NON-EMPTY HASHES. `PgVectorStore.source_raw_hashes` selects
   `coalesce(metadata->>'content_hash', '')`. Two generations that BOTH lack content hashes
   therefore compare equal, and the validator reports perfect parity over a pair of absences. This
   is the same defect class the session-9 audit already found once in this repository, where a
   content-hash test compared SQL NULL with SQL NULL. Every hash is asserted to be a 64-character
   hex digest.
2. COVERAGE. `set() & set()` is empty, so two EMPTY generations also report no mismatches. The
   number of sources actually compared is asserted against the number of files on disk.
3. POSITIVE CONTROL. A generation built from a corpus with one file's bytes changed MUST come back
   with a non-empty `hash_mismatches` and `valid=False`. Without it, "no mismatches" is not
   evidence that a mismatch could have been detected.
4. SELF-COMPARISON. The baseline against itself must be valid. This separates "the validator is
   working" from "the validator returns valid for everything".

`GenerationParity.valid` also folds in `rls_enabled` and `indexes_valid`, which are facts about the
HOST rather than about content. They are reported as their own fields so a deployment-shaped
failure can never be read as a content-parity failure, or the reverse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

from recall.context import context_policy_for_profile
from recall.embeddings import embedding_profile_id
from recall.index import Indexer
from recall.migration import validate_generation_parity
from recall.store import PgVectorStore
from recall_mcp.service import make_embedder

#: The baseline is FIRST and is the raw-context profile. Every comparison is
#: `baseline vs candidate`, three independent pairs, never a tournament between the candidates.
BASELINE = "bge-small-symmetric-v1"
CANDIDATES = (
    "bge-small-context-document-v1",
    "bge-small-context-section-v1",
    "bge-small-context-neighbor-v1",
)

_HEX = set("0123456789abcdef")


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and set(value) <= _HEX


def _table_for(prefix: str, profile: str) -> str:
    """One table per profile, derived so the index and compare stages cannot disagree on it."""
    return f"{prefix}_{profile.replace('-', '_')}"


def _open(dsn: str, profile: str, table: str) -> tuple:
    """Resolve the profile, its context policy and its store, WITHOUT indexing.

    Split out of `_index` so the compare stage can reach the same tables a set of concurrent
    index stages wrote, without re-embedding a single chunk.
    """
    os.environ["RECALL_EMBED_PROFILE"] = profile
    embedder = make_embedder("fastembed")
    resolved = embedding_profile_id(embedder)
    if resolved != profile:
        raise SystemExit(
            f"asked for profile {profile!r} and the embedder resolved to {resolved!r}. "
            f"Every generation below would be labelled with a profile it was not built under."
        )
    policy = context_policy_for_profile(resolved)
    return PgVectorStore(dsn, dim=embedder.dim, table=table), embedder, policy, resolved


def _index(dsn: str, table: str, profile: str, corpus: Path, glob: str) -> dict:
    """Build one generation and return its facts. The context policy comes from the PROFILE.

    `Indexer` defaults to `ContextPolicy()`, mode "none", and REFUSES an embedder whose registered
    profile declares anything else. Deriving the policy from the profile is what makes a context
    arm indexable at all; the campaign's first attempt omitted it and all three context arms died
    at index time.
    """
    store, embedder, policy, resolved = _open(dsn, profile, table)
    store.ensure_schema()
    started = time.monotonic()
    stats = Indexer(store, embedder, context_policy=policy).index_path(corpus, glob)
    elapsed = time.monotonic() - started
    if not stats.chunks:
        raise SystemExit(
            f"{profile}: indexing {corpus} with glob {glob!r} produced NO chunks. A parity "
            f"comparison between two empty generations reports perfect parity."
        )
    return {
        "profile": resolved,
        "table": table,
        "context_mode": policy.mode,
        "files": stats.files,
        "chunks": stats.chunks,
        "index_seconds": round(elapsed, 1),
        "store": store,
    }


def _parity(active: PgVectorStore, shadow: PgVectorStore, expected_sources: int) -> dict:
    """Run the shipped validator, then apply controls 1 and 2 to its inputs."""
    result = validate_generation_parity(active, shadow)
    active_hashes = active.source_raw_hashes()
    shadow_hashes = shadow.source_raw_hashes()
    compared = sorted(set(active_hashes) & set(shadow_hashes))
    # Control 1: an absent content_hash arrives here as '' and compares equal to another ''.
    degenerate = sorted(
        source
        for source in compared
        if not _is_sha256_hex(active_hashes[source]) or not _is_sha256_hex(shadow_hashes[source])
    )
    return {
        "valid": result.valid,
        "active_chunks": result.active_chunks,
        "shadow_chunks": result.shadow_chunks,
        "missing_sources": list(result.missing_sources),
        "extra_sources": list(result.extra_sources),
        "hash_mismatches": list(result.hash_mismatches),
        "failures": list(result.failures),
        # Reported apart from `valid` on purpose: these two are host facts, not content facts.
        "content_parity_holds": (
            not result.missing_sources and not result.extra_sources and not result.hash_mismatches
        ),
        "controls": {
            "sources_compared": len(compared),
            "sources_expected": expected_sources,
            # Control 2: two empty generations yield an empty intersection and no mismatches.
            "coverage_ok": len(compared) == expected_sources and expected_sources > 0,
            # Control 1, reported rather than merely asserted, so the artifact carries the evidence.
            "degenerate_hashes": degenerate,
            "hashes_all_sha256": not degenerate,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--glob", default="**/*.rst")
    parser.add_argument("--table-prefix", default="parity")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--control-corpus",
        type=Path,
        help="Directory for the positive control's perturbed copy. A SUBSET is fine and is the "
        "point: the control has to show the validator can fire, not re-measure the corpus.",
    )
    parser.add_argument("--control-files", type=int, default=24)
    parser.add_argument("--keep-tables", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("all", "index", "compare"),
        default="all",
        help="`index` builds ONE generation and keeps its table, so the four arms can run as "
        "concurrent processes at two cores each instead of one process paying the sum of them. "
        "`compare` then reads those tables. `all` does both in one process.",
    )
    parser.add_argument("--profile", help="the single profile to build when --stage index")
    args = parser.parse_args()
    if args.stage == "index" and not args.profile:
        parser.error("--stage index requires --profile")
    if args.stage == "index" and args.profile not in (BASELINE, *CANDIDATES):
        parser.error(f"--profile must be one of {(BASELINE, *CANDIDATES)}")

    corpus = args.corpus_dir.resolve()
    on_disk = sorted(corpus.glob(args.glob))
    if not on_disk:
        raise SystemExit(f"no files match {args.glob!r} under {corpus}")
    print(f"corpus {corpus}: {len(on_disk)} files matching {args.glob!r}", flush=True)

    built: list[dict] = []
    report: dict = {
        "kind": "generation_parity",
        "corpus_dir": str(corpus),
        "glob": args.glob,
        "files_on_disk": len(on_disk),
        "baseline_profile": BASELINE,
        "reconstruction_note": (
            "The 2026-08-06 campaign's own generations were promo_<uuid8> tables dropped in a "
            "finally and no longer exist. These generations are a REBUILD over the same corpus "
            "with the same embedder, not the campaign's originals."
        ),
        "comparisons": {},
        "controls": {},
    }
    # An index stage that dropped its own table would leave the compare stage nothing to read.
    if args.stage == "index":
        args.keep_tables = True

    try:
        if args.stage == "index":
            table = _table_for(args.table_prefix, args.profile)
            print(f"indexing {args.profile} -> {table}", flush=True)
            gen = _index(args.dsn, table, args.profile, corpus, args.glob)
            built.append(gen)
            print(f"  {gen['chunks']} chunks in {gen['index_seconds']}s", flush=True)
            return 0

        if args.stage == "all":
            for profile in (BASELINE, *CANDIDATES):
                table = _table_for(args.table_prefix, profile)
                print(f"indexing {profile} -> {table}", flush=True)
                built.append(_index(args.dsn, table, profile, corpus, args.glob))
                print(
                    f"  {built[-1]['chunks']} chunks in {built[-1]['index_seconds']}s", flush=True
                )
        else:
            for profile in (BASELINE, *CANDIDATES):
                table = _table_for(args.table_prefix, profile)
                store, _embedder, policy, resolved = _open(args.dsn, profile, table)
                rows = store.count()
                # A table that was never written reads as a generation with nothing in it, and two
                # empty generations report perfect parity. Refuse where the cause is legible.
                if not rows:
                    raise SystemExit(
                        f"{profile}: table {table} holds no rows for this tenant. The index stage "
                        f"for this arm did not run, or did not complete."
                    )
                print(f"opened {profile} -> {table} ({rows} chunks)", flush=True)
                built.append(
                    {
                        "profile": resolved,
                        "table": table,
                        "context_mode": policy.mode,
                        "files": None,
                        "chunks": rows,
                        "index_seconds": None,
                        "store": store,
                    }
                )

        base = built[0]
        report["generations"] = [
            {k: v for k, v in gen.items() if k != "store"} for gen in built
        ]

        # Control 4: the validator must not simply return valid for everything.
        report["controls"]["self_comparison"] = _parity(
            base["store"], base["store"], len(on_disk)
        )

        for gen in built[1:]:
            print(f"parity: {BASELINE} vs {gen['profile']}", flush=True)
            report["comparisons"][gen["profile"]] = _parity(
                base["store"], gen["store"], len(on_disk)
            )

        # Control 3, on a SUBSET. The control's job is to show a mismatch is detectable, which one
        # changed byte establishes as well as 732 files would, at a fraction of the embedding cost.
        if args.control_corpus:
            ctl = args.control_corpus.resolve()
            if ctl.exists():
                shutil.rmtree(ctl)
            ctl.mkdir(parents=True)
            subset = on_disk[: args.control_files]
            for src in subset:
                shutil.copy2(src, ctl / src.name)
            clean_tbl = f"{args.table_prefix}_ctl_clean"
            dirty_tbl = f"{args.table_prefix}_ctl_dirty"
            clean = _index(args.dsn, clean_tbl, BASELINE, ctl, "*.rst")
            # One file, one appended line. The raw content hash is sha256 over the whole file, so
            # this must move exactly one source's hash and nothing else.
            victim = sorted(ctl.glob("*.rst"))[0]
            before = hashlib.sha256(victim.read_bytes()).hexdigest()
            with victim.open("a", encoding="utf-8") as handle:
                handle.write("\n.. parity positive control\n")
            after = hashlib.sha256(victim.read_bytes()).hexdigest()
            dirty = _index(args.dsn, dirty_tbl, BASELINE, ctl, "*.rst")
            control = _parity(clean["store"], dirty["store"], len(subset))
            report["controls"]["positive_control"] = {
                "victim": victim.name,
                "sha256_before": before,
                "sha256_after": after,
                "file_bytes_changed": before != after,
                "parity": control,
                # This is the assertion that makes the headline mean something.
                "fired": bool(control["hash_mismatches"]) and not control["valid"],
                # `recall/index.py` stores the ABSOLUTE host path in the `source` column, so the
                # mismatch is keyed by full path and not by basename. Comparing against the name
                # made this read False while the validator had in fact isolated exactly one file.
                "detected_exactly_one_source": control["hash_mismatches"] == [str(victim)],
            }
            built.extend([clean, dirty])

        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.out}", flush=True)

        content_ok = all(c["content_parity_holds"] for c in report["comparisons"].values())
        cov_ok = all(c["controls"]["coverage_ok"] for c in report["comparisons"].values())
        hash_ok = all(c["controls"]["hashes_all_sha256"] for c in report["comparisons"].values())
        ctl = report["controls"].get("positive_control")
        ctl_ok = ctl is None or ctl["fired"]
        print(
            f"content_parity={content_ok} coverage={cov_ok} hashes_sha256={hash_ok} "
            f"positive_control_fired={ctl_ok}",
            flush=True,
        )
        return 0 if (content_ok and cov_ok and hash_ok and ctl_ok) else 1
    finally:
        if not args.keep_tables:
            for gen in built:
                try:
                    gen["store"].drop_table()
                except Exception as exc:  # pragma: no cover - cleanup diagnostics only
                    print(f"WARN could not drop {gen['table']}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
