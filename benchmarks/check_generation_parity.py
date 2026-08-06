#!/usr/bin/env python
"""Generation parity across the context-mode profiles, using the shipped validator.

The invariant this whole family of profiles rests on: a context mode changes the text that is
EMBEDDED and must never change the raw chunk content or the raw content hash that is STORED. If it
did, the three context arms of a promotion campaign would not be comparable with the raw baseline,
because they would be describing different corpora.

`recall.migration.validate_generation_parity` is the shipped check for exactly this, so it is what
runs here rather than a second implementation of the same comparison written for the occasion.

⚠️ PRIOR WORK EXISTS, and an earlier version of this docstring said it did not.
`tests/test_context_modes_index.py::test_raw_text_and_content_hashes_are_identical_across_generations_and_modes`
already asserts this invariant over four independently indexed generations against real PostgreSQL
rows, and it is STRICTER than this script on what it compares: stored text, `content_hash`,
`text_start`/`text_end` and `heading_hierarchy`, where this script compares source sets, raw
hashes and chunk counts. `docs/ENTERPRISE_PROGRAM_STATUS.md` records that invariant as "held".

The claim "Prior work: NONE FOUND" was written from `docs_search(source_type="memory")` for
"generation parity raw content hash identical across context modes RE-call" on 2026-08-06, which
returned `gap_warning` TRUE (top-3 cosine 0.485 / 0.480 / 0.479, all under the 0.50 floor). That
search was scoped to the MEMORY corpus, which cannot see repository tests. "No memo records this"
was true; "no prior work exists" did not follow from it, and the two were not the same question.

What this script adds over that test, which is the honest scope of its contribution: the test runs
one small in-memory corpus through a stub embedder inside pytest, and this runs the CAMPAIGN's
corpus through the real fastembed profiles at the pinned artifact tree, comparing promotion-shaped
generations with the shipped validator rather than with hand-written assertions. It is a
scale-and-realism check on an invariant already covered at unit scale, NOT a first look at it.

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
4. SELF-COMPARISON. The baseline against itself must come back parity-holding. ⚠️ This does NOT
   distinguish "the validator works" from "the validator returns valid for everything": a
   validator hardcoded to `valid=True` passes it identically. Only control 3, which demands a
   NEGATIVE verdict, discriminates those two. Control 4's actual job is the opposite direction, a
   validator or a host that fails EVERYTHING: with `active is shadow` the three content sets are
   empty by construction, so anything it does report is a host fact (`rls_enabled`,
   `indexes_valid`) rather than a content difference.

`GenerationParity.valid` also folds in `rls_enabled` and `indexes_valid`, which are facts about the
HOST rather than about content. `_parity` therefore reports `content_parity_holds` separately AND
emits the two host facts as their own fields, read from `readiness_facts()`, so a deployment-shaped
failure can never be read as a content-parity failure or the reverse. Chunk count is deliberately
on the CONTENT side of that split: `contextual_passages` returns one chunk per input chunk in every
mode, so a count difference is a content divergence, not a host fact.
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

import psycopg

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
    # Case-insensitive: a valid digest is valid in either case, and rejecting an uppercase one
    # would fail the run for a defect that is not there.
    return len(value) == 64 and set(value.lower()) <= _HEX


def _table_for(prefix: str, profile: str) -> str:
    """One table per profile, derived so the index and compare stages cannot disagree on it."""
    return f"{prefix}_{profile.replace('-', '_')}"


def _table_exists(dsn: str, table: str) -> bool:
    """Whether `table` is in the catalog, asked BEFORE any migration runs.

    This is what makes `owned` mean "THIS PROCESS CREATED IT". Deriving ownership from the fact
    that `ensure_schema()` returned would be wrong: it silently ADOPTS a pre-existing table,
    because every migration reads as already applied and it returns without creating anything. A
    re-run over tables another process built would then mark them owned and drop them.
    """
    # `connect_timeout` matches recall/store.py and recall/schema.py, whose comment records why:
    # without it a dead host hangs the caller on the TCP handshake indefinitely.
    with psycopg.connect(dsn, autocommit=True, connect_timeout=10) as conn:
        row = conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()
    return bool(row and row[0] is not None)


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


def _index(
    dsn: str, table: str, profile: str, corpus: Path, glob: str, *, scratch: bool = False
) -> dict:
    """Build one generation and return its facts. The context policy comes from the PROFILE.

    `Indexer` defaults to `ContextPolicy()`, mode "none", and REFUSES an embedder whose registered
    profile declares anything else. Deriving the policy from the profile is what makes a context
    arm indexable at all; the campaign's first attempt omitted it and all three context arms died
    at index time.
    """
    store, embedder, policy, resolved = _open(dsn, profile, table)
    # Asked BEFORE `ensure_schema`, which would otherwise adopt an existing table. See
    # `_table_exists`: this is the difference between "we made it" and "we migrated it".
    pre_existing = _table_exists(dsn, table)
    store.ensure_schema()
    started = time.monotonic()
    stats = Indexer(store, embedder, context_policy=policy).index_path(corpus, glob)
    elapsed = time.monotonic() - started
    if not stats.chunks:
        # `stats.skipped` is what separates the two causes. A table that is ALREADY complete
        # returns 0 written and N skipped, because the fingerprint guard skips every file, and an
        # earlier version reported that as though the corpus or the glob were at fault.
        if stats.skipped:
            raise SystemExit(
                f"{profile}: {table} already holds this corpus at this fingerprint "
                f"({stats.skipped} files skipped, 0 written), so nothing was re-indexed. This is a "
                f"COMPLETE generation, not an empty one. Drop the table or point --table-prefix "
                f"somewhere fresh."
            )
        raise SystemExit(
            f"{profile}: indexing {corpus} with glob {glob!r} produced NO chunks and skipped "
            f"nothing, so the glob matched no file. A parity comparison between two empty "
            f"generations reports perfect parity."
        )
    return {
        "profile": resolved,
        "table": table,
        "context_mode": policy.mode,
        "files": stats.files,
        "chunks": stats.chunks,
        "index_seconds": round(elapsed, 1),
        "store": store,
        # THIS process created this table, so THIS process may drop it. False when the table was
        # already in the catalog, which `ensure_schema` alone could not have told us.
        "owned": not pre_existing,
        # A control table is disposable by design; a generation cost about an hour of embedding.
        # The cleanup treats the two differently, so the distinction is recorded rather than
        # inferred from the name.
        "scratch": scratch,
    }


def _parity(active: PgVectorStore, shadow: PgVectorStore, expected_sources: int) -> dict:
    """Run the shipped validator, then apply controls 1 and 2 to its inputs."""
    result = validate_generation_parity(active, shadow)
    active_facts = active.readiness_facts()
    shadow_facts = shadow.readiness_facts()
    # ⚠️ A SECOND read. `validate_generation_parity` does not return the hash maps it compared, so
    # the controls below attest to a later read of the same tables rather than to the validator's
    # own inputs. Sound here only because nothing writes between the two reads: the compare stage
    # is read-only and the index stages have exited.
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
        # `rls_enabled` and `indexes_valid` are the two HOST facts `valid` folds in. They are
        # emitted separately so a deployment-shaped failure is never read as a content failure.
        "rls_enabled": bool(active_facts["rls_enabled"] and shadow_facts["rls_enabled"]),
        "indexes_valid": bool(active_facts["indexes_valid"] and shadow_facts["indexes_valid"]),
        # ⚠️ CHUNK COUNT IS A CONTENT FACT and belongs here. An earlier version omitted it, so a
        # pair the shipped validator failed on `chunk counts differ between generations` was
        # written to the artifact with `valid: false` and a populated `failures` list, and the
        # process still exited 0. The verdict dropped a term the validator had already computed.
        "content_parity_holds": (
            not result.missing_sources
            and not result.extra_sources
            and not result.hash_mismatches
            and result.active_chunks == result.shadow_chunks
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
        "point: the control has to show the validator can fire, not re-measure the corpus. It is "
        "REQUIRED for --stage all/compare, because a run that never demonstrated a mismatch is "
        "detectable would rest its verdict on an absence.",
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
    parser.add_argument(
        "--drop-generations",
        action="store_true",
        help="Also drop the four GENERATION tables, whether or not this process created them. Off "
        "by default: they cost about an hour of embedding each, and keeping them is what makes a "
        "re-run of the compare stage free. This is the supported way to remove them, because "
        "PgVectorStore.drop_table() also clears the migration ledger rows that a bare psql DROP "
        "TABLE would orphan.",
    )
    args = parser.parse_args()
    if args.stage == "index" and not args.profile:
        parser.error("--stage index requires --profile")
    if args.stage == "index" and args.profile not in (BASELINE, *CANDIDATES):
        parser.error(f"--profile must be one of {(BASELINE, *CANDIDATES)}")
    if args.control_files < 1:
        # A negative bound inverts the slice: `on_disk[:-1]` is EVERY FILE BUT THE LAST, so the
        # "control" would silently index the whole corpus.
        parser.error("--control-files must be >= 1")
    if args.stage in ("all", "compare") and not args.control_corpus:
        parser.error(
            "--stage all/compare requires --control-corpus: without the positive control nothing "
            "in the run demonstrates that a mismatch is detectable, and the verdict would rest on "
            "an absence."
        )

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
                        # ⚠️ NOT owned. A concurrent index stage built this table, at roughly an
                        # hour of embedding each. An earlier version dropped these in the cleanup
                        # `finally`, on every exit path including the deliberate refusals and a
                        # failed report write, so one dead arm destroyed the three that succeeded
                        # and the compare stage could never be re-run.
                        "owned": False,
                        "scratch": False,
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

        # Control 3, on a SUBSET. The control's job is to show a mismatch is detectable, which ONE
        # CHANGED FILE establishes as well as the whole corpus would, at a fraction of the
        # embedding cost. (An earlier version of this comment said "one changed byte" while the
        # code appends a 27-byte line, and cited "732 files" for a corpus the run measures at 746.)
        if args.control_corpus:
            # ⚠️ `rmtree` on an argv path, on a host that also carries unrelated live production.
            # An earlier version deleted whatever `--control-corpus` named, with `resolve()`
            # following a symlink first. `--control-corpus /var/tmp` would have taken out the
            # venv, the checkout and the output directory this very script runs from.
            if args.control_corpus.is_symlink():
                raise SystemExit(
                    f"--control-corpus {args.control_corpus} is a symlink; refusing to resolve "
                    f"and delete its target."
                )
            ctl = args.control_corpus.resolve()
            if ctl == corpus or corpus in ctl.parents or ctl in corpus.parents:
                raise SystemExit(
                    f"--control-corpus {ctl} overlaps --corpus-dir {corpus}. Deleting it would "
                    f"destroy the corpus under measurement, or leave mutated copies inside it."
                )
            marker = ctl / ".parity-scratch"
            if ctl.exists():
                # Only a directory this script created, or one that is EMPTY, may be recursively
                # deleted. The empty case matters: a directory left by an earlier harness, or by a
                # crash in the one-syscall window between `mkdir` and the marker write below,
                # would otherwise abort the run AFTER the comparisons and BEFORE the artifact is
                # written. Refusing a path that previously worked is a regression, and deleting an
                # empty directory destroys nothing.
                if not ctl.is_dir():
                    raise SystemExit(f"--control-corpus {ctl} exists and is not a directory.")
                # `Path.is_file()` swallows OSError but `iterdir()` raises it, so a directory that
                # cannot be listed would abort with a bare PermissionError traceback instead of
                # the refusal this block exists to produce. FAILING TO LIST MUST READ AS
                # "NOT EMPTY": the one thing that must never follow from an unknown directory is
                # a recursive delete.
                try:
                    empty = not any(ctl.iterdir())
                except OSError as exc:
                    raise SystemExit(
                        f"--control-corpus {ctl} cannot be listed ({exc}); refusing to delete a "
                        f"directory whose contents are unknown."
                    ) from exc
                if not marker.is_file() and not empty:
                    raise SystemExit(
                        f"--control-corpus {ctl} is a NON-EMPTY directory carrying no "
                        f"{marker.name} marker, so it was not created by this harness. Refusing "
                        f"to delete it. Point the flag at a fresh path."
                    )
                shutil.rmtree(ctl)
            ctl.mkdir(parents=True)
            marker.write_text("written by check_generation_parity.py\n", encoding="utf-8")
            subset = on_disk[: args.control_files]
            for src in subset:
                shutil.copy2(src, ctl / src.name)
            clean_tbl = f"{args.table_prefix}_ctl_clean"
            dirty_tbl = f"{args.table_prefix}_ctl_dirty"
            # DERIVED from `--glob`, not the literal "*.rst" this used to hardcode. The control
            # corpus is a FLAT copy, so only the last path component applies, which is the same
            # rule `recall.index` matches on. With the literal, any non-.rst corpus made the
            # positive control impossible, and it failed only after every comparison had been paid
            # for, blaming the control corpus rather than the disagreement with --glob.
            control_glob = args.glob.rsplit("/", 1)[-1]
            clean = _index(args.dsn, clean_tbl, BASELINE, ctl, control_glob, scratch=True)
            # Registered IMMEDIATELY, not after the block succeeds. `_index` creates the table
            # before it indexes, so anything that raised between here and a later registration
            # leaked a table the cleanup block claims to drop.
            built.append(clean)
            # One file, one appended line. The raw content hash is sha256 over the whole file, so
            # this must move exactly one source's hash and nothing else.
            # Exclude the marker explicitly rather than relying on the glob. pathlib `*` DOES match
            # dotfiles, so a `--glob '**/*'` derives a control glob that pulls `.parity-scratch`
            # in, and `.` sorts first, so the harness would mutate its OWN marker and score a
            # blocking control green while measuring nothing from the corpus.
            candidates = [p for p in sorted(ctl.glob(control_glob)) if p.name != marker.name]
            if not candidates:
                raise SystemExit(
                    f"control corpus {ctl} matched no file under {control_glob!r} once the "
                    f"{marker.name} marker is excluded."
                )
            victim = candidates[0]
            before = hashlib.sha256(victim.read_bytes()).hexdigest()
            with victim.open("a", encoding="utf-8") as handle:
                handle.write("\n.. parity positive control\n")
            after = hashlib.sha256(victim.read_bytes()).hexdigest()
            dirty = _index(args.dsn, dirty_tbl, BASELINE, ctl, control_glob, scratch=True)
            built.append(dirty)
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

        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.out}", flush=True)

        comparisons = report["comparisons"]
        # `all([])` is True, so every aggregate below is vacuously satisfied over an empty set. A
        # run that compared NOTHING must not report success.
        complete = len(comparisons) == len(CANDIDATES)
        content_ok = complete and all(c["content_parity_holds"] for c in comparisons.values())
        cov_ok = complete and all(c["controls"]["coverage_ok"] for c in comparisons.values())
        hash_ok = complete and all(
            c["controls"]["hashes_all_sha256"] for c in comparisons.values()
        )
        # Control 4 is declared blocking in the docstring, so it has to appear in the verdict.
        # An earlier version computed it, wrote it to the artifact, and never read it.
        self_c = report["controls"]["self_comparison"]
        self_ok = bool(
            self_c["content_parity_holds"]
            and self_c["controls"]["coverage_ok"]
            and self_c["controls"]["hashes_all_sha256"]
        )
        # Named apart from the `ctl` Path above: rebinding one name to two unrelated types in one
        # scope is unchecked, because `args` is an untyped Namespace.
        control_report = report["controls"].get("positive_control")
        # ⚠️ NOT `ctl is None or ...`. An absent positive control means the run never demonstrated
        # a mismatch is detectable at all, which is precisely what control 3 exists to establish;
        # scoring that as a pass is the vacuous-true shape this whole script is written against.
        # `detected_exactly_one_source` is part of the gate too: exactly one file's bytes were
        # changed, so any other cardinality means the control fired for the wrong reason.
        ctl_ok = (
            control_report is not None
            and control_report["fired"]
            and control_report["detected_exactly_one_source"]
        )
        print(
            f"comparisons={len(comparisons)}/{len(CANDIDATES)} content_parity={content_ok} "
            f"coverage={cov_ok} hashes_sha256={hash_ok} self_comparison={self_ok} "
            f"positive_control_fired={ctl_ok}",
            flush=True,
        )
        return 0 if (content_ok and cov_ok and hash_ok and self_ok and ctl_ok) else 1
    finally:
        # Three rules, in this order, and the order is the point:
        #   1. `--keep-tables` drops nothing at all.
        #   2. NEVER drop a table this process did not create. A compare stage is HANDED four
        #      generations another process built, at about an hour of embedding each.
        #   3. A SCRATCH (control) table is disposable and always goes. A GENERATION goes only when
        #      explicitly asked for, because keeping it is what makes a compare re-run free.
        # An earlier version had two loops and got rule 3 backwards on the success path of
        # `--stage all`: every arm there is created by this process, so all four were dropped on a
        # clean run, which contradicted the rationale `--drop-generations` states for its default.
        # ⚠️ Ownership is NOT the gate, and an earlier version of this loop made it one. That
        # skipped every not-owned table first, which silently killed `--drop-generations` in the
        # ONLY stage that can reach it: a compare stage marks all four generations `owned=False`
        # by construction, so the flag became a no-op in the one path the driver runs, while its
        # own help text still advertised the old meaning. Safety here comes from the
        # scratch/generation split (a generation is never dropped implicitly), not from refusing
        # an operator who asked explicitly. `owned` is reported, and narrates the loud case.
        for gen in built:
            if args.keep_tables:
                continue
            if not gen.get("scratch") and not args.drop_generations:
                continue
            if not gen.get("scratch") and not gen.get("owned"):
                print(
                    f"NOTE dropping {gen['table']}, which this process did NOT create, because "
                    f"--drop-generations was passed",
                    file=sys.stderr,
                )
            try:
                gen["store"].drop_table()
            except Exception as exc:  # pragma: no cover - cleanup diagnostics only
                print(f"WARN could not drop {gen['table']}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
