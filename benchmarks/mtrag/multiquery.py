"""Multi-query diversity with nested RRF on MTRAG-human dev, with no LLM and no GPU.

The arms, contrasts, decision rule and predicted outcome were frozen before any score was
observed; `MQ_ARMS` below is the executable copy and `tests/test_multiquery.py` pins it.

Prior work (searched 2026-08-06, `docs_search(source_type="memory")`, no gap warning):

  - [[project-recall-mtrag-retrieval-coverage-bottleneck-2026-08-06]] -- coverage, not ranking, is
    the bottleneck: R@100 0.687 against a ~0.95 saturation threshold.
  - [[project-recall-splade-learned-sparse-measured-2026-08-06]] -- SPLADE moved reranked R@100 to
    0.7599 (+0.0331, CI [+0.0202, +0.0457]). Coverage improved, not solved.
  - [[reference-validation-standards]] -- bootstrap CI, sign-flip permutation, Holm-Bonferroni.
    `analyse_contrasts` implements them and is imported rather than reimplemented.

🔑 Why this is cheap. MTRAG-human ships three ALIGNED query files per domain -- last user turn,
full-conversation concatenation, and a gold human rewrite -- on the same `_id`s. They are three
genuine reformulations, already written, so the fusion mechanism is testable for a few CPU-hours
instead of a GPU rental.

🔑 Why it dumps LEGS and not pools. `rerank_offload.cmd_dump` records each arm's already-fused
pool, which means one retrieval pass per arm. Here the expensive thing (retrieval) is done ONCE
per query variant and every fusion topology is then computed offline from the same per-leg
rankings. Eight arms cost three variants of retrieval, every arm sees byte-identical raw material
so no contrast is contaminated by retrieval jitter, and a new topology costs zero database work.

    dump      -- retrieve each variant, record per-leg ID rankings and any leg shortfall
    validate  -- rebuild `mq_last` from the dump and require it to match `HybridRetriever.search`
    fuse      -- compute every declared arm from the dumped legs and score it against the qrels
    contrast  -- the preregistered paired CIs, permutation tests and Holm correction

⚠️ `validate` is not optional, for the reason `rerank_offload` states about its own gate: a fusion
that merely looks reasonable produces publishable-looking metrics that RE-call would never
compute, and nothing about the output looks wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.mtrag.analyse_contrasts import holm, ndcg_at, paired_stats, recall_at
from benchmarks.mtrag.run import (
    DEFAULT_SPARSE_MODEL,
    DOMAINS,
    EMBEDDER_MODEL,
    load_dsn,
    load_qrels,
    strip_speaker,
    table_name,
    utc_now,
)

#: RRF damping constant, at BOTH nesting levels, for every arm.
#:
#: 60 is `recall.retriever._rrf`'s default and therefore the control's fusion. Rank 1 used
#: `k_internal = 40` for their weak-consensus group; adopting that here would change the control
#: as well and break comparability with the archived `hybrid_splade` baseline, so it is a
#: deliberate and recorded deviation from the paper rather than an oversight.
RRF_K = 60

#: How deep each leg retrieves, and the metric cutoff. Both 100: coverage is the bottleneck and
#: R@100 is the decision metric.
LEG_DEPTH = 100
EVAL_K = 100

#: The three shipped query files, in the order every weighted arm's weight vector assumes.
VARIANT_FILES = {"last": "lastturn", "full": "questions", "rewrite": "rewrite"}

#: The retrieval legs of the SPLADE configuration, in `HybridRetriever.search`'s own fusion order.
#:
#: Order is load bearing, not cosmetic. `_rrf` accumulates into a plain dict and the caller sorts
#: it with a STABLE sort, so RRF ties resolve by first appearance -- which is dense, then learned
#: sparse. Reversing these two names would silently re-break every tie in the control.
LEGS = ("dense", "splade")


@dataclass(frozen=True)
class MultiQueryArm:
    """One frozen arm: which reformulations to fuse, and how.

    `topology`:
      nested -- fuse a variant's legs into one per-variant ranking, then fuse the variants.
                This is the two-level shape rank 1 describes, mapped onto RE-call: the inner
                level is dense+SPLADE, the outer is the reformulations.
      flat   -- one RRF over every leg ranking of every variant, all peers.
      Both reduce to `HybridRetriever`'s own fusion when there is a single variant.

    `weights` applies at the OUTER level only, one per variant. `leg_truncate` caps every leg
    ranking before any fusion, which is how total retrieval budget is held fixed.
    """

    name: str
    variants: tuple[str, ...]
    topology: str
    weights: tuple[float, ...] | None = None
    leg_truncate: int | None = None
    role: str = "ablation"

    def pool_bound(self) -> int:
        """UPPER BOUND on distinct chunks this arm's fusion can hold, assuming NO overlap.

        ⚠️ This is a bound, not a measurement, and it must not be read as certifying that a
        metric cut at `EVAL_K` measures ranking rather than pool. The variants overlap heavily
        (they are reformulations of one question), and where they are byte-identical they overlap
        completely: `mq_nested3_budget33` on such a query fuses three copies of the same two
        33-deep legs and realises at most 66 distinct documents against a cut at 100, while this
        method still returns 198.

        `realised_pool_sizes` is the measurement. Use that to decide whether a metric is
        pool-limited; use this only to reason about the ceiling.
        """
        return len(self.variants) * len(LEGS) * (self.leg_truncate or LEG_DEPTH)


#: Frozen 2026-08-06, before any score from this experiment was observed.
#:
#: `mq_last` is configuration-identical to the archived `hybrid_splade` and is the control the
#: whole family is measured against; `validate` requires it to reproduce that run exactly.
#: `mq_rewrite` and `mq_full` are re-measured here rather than cited from the 2026-08-06 2x2,
#: which ran at candidate_k=20 on the pre-SPLADE lexical hybrid and recorded no R@100 at all.
#:
#: ⚠️ Do not add an arm to this tuple after observing a score. A post-hoc arm is a different
#: experiment wearing a preregistered experiment's name; declare it separately and label it.
MQ_ARMS = (
    MultiQueryArm("mq_last", ("last",), "nested", role="control"),
    MultiQueryArm("mq_rewrite", ("rewrite",), "nested", role="ceiling"),
    MultiQueryArm("mq_full", ("full",), "nested"),
    MultiQueryArm("mq_nested3", ("last", "full", "rewrite"), "nested", role="primary"),
    MultiQueryArm("mq_nested2", ("last", "rewrite"), "nested", role="robustness"),
    MultiQueryArm("mq_flat6", ("last", "full", "rewrite"), "flat", role="topology"),
    # w_full = 0.5, fixed A PRIORI because `full` is the only variant with a measured negative
    # effect. Not tuned; no other weight setting will be run.
    MultiQueryArm(
        "mq_nested3_vw", ("last", "full", "rewrite"), "nested",
        weights=(1.0, 0.5, 1.0), role="variance_aware",
    ),
    # 3 variants x 2 legs x 33 = 198 ~ the control's 200, so diversity is separated from budget.
    MultiQueryArm(
        "mq_nested3_budget33", ("last", "full", "rewrite"), "nested",
        leg_truncate=33, role="budget_control",
    ),
)

#: ⚠️ POST-HOC. Declared 2026-08-06 AFTER the preregistered results were observed, and kept out of
#: `MQ_ARMS` and out of the Holm family for that reason. It is exploratory and reported separately.
#:
#: Why it exists: every preregistered fusion arm consumes `rewrite`, which is a GOLD HUMAN rewrite
#: and is not available at serving time. `last` and `full` both are: they are just the current turn
#: and the conversation so far. So this arm asks the deployment question the frozen set cannot,
#: namely whether query diversity buys anything with NO gold and NO LLM at all.
#:
#: Its control is `mq_last`, and its contrast is reported with the same paired statistics, labelled
#: post-hoc. A result here is a hypothesis for a future preregistration, not a decision.
POST_HOC_ARMS = (
    MultiQueryArm("mq_nested2_nogold", ("last", "full"), "nested", role="post_hoc_deployable"),
)

#: The preregistered Holm family: five contrasts, all on R@100. nDCG@5 and the shallower recalls
#: are recorded as descriptive secondaries and cannot trigger a ship.
#:
#: The last field is the contrast's DECIDING CELL -- the queries on which its delta can be
#: non-zero at all. `variants_differ` means a query whose three variants are byte-identical
#: contributes an exact structural zero, because identical text yields identical leg rankings and
#: RRF over copies of one ranking is order-preserving (`test_multiquery` pins that).
#:
#: ⚠️ B1 is the exception and it is easy to get wrong. It compares TRUNCATED legs (33) against
#: untruncated ones (100), so its delta is driven by the truncation and is perfectly capable of
#: being non-zero on a query whose variants are identical. Reporting B1 over the
#: `variants_differ` cell would silently drop 102 queries that carry real signal for it.
CONTRASTS = (
    ("P1", "mq_nested3", "mq_last",
     "does multi-query fusion move coverage at all", "variants_differ"),
    ("M1", "mq_nested3", "mq_rewrite",
     "fusion vs rewriting quality", "variants_differ"),
    ("T1", "mq_nested3", "mq_flat6",
     "does the nesting topology matter", "variants_differ"),
    ("R1", "mq_nested2", "mq_nested3",
     "does including a harmful variant hurt", "variants_differ"),
    ("B1", "mq_nested3_budget33", "mq_last",
     "diversity at fixed retrieval budget", "all"),
)

#: The preregistered ship bar on P1, in R@100. Sits just above the ~0.019 minimum detectable
#: effect, so the gate can both fire and fail.
SHIP_BAR = 0.020

DECISION_METRIC = "R@100"

#: Ranking metrics that can BLOCK a ship but can never trigger one.
#:
#: 🔑 Amendment, 2026-08-06, made while the dump was still retrieving and BEFORE any fused score
#: existed. Without it the rule was one-sided: an arm that lifted R@100 while degrading the
#: ranking would clear the bar. That is not hypothetical on this project. The archived SPLADE run
#: is exactly that shape ("the headline reverses under reranking, and that is the finding"), and
#: rank 1 likewise reports added retrieval raising recall while ranking falls.
#:
#: ⚠️ These are deliberately NOT in the Holm family and NOT multiplicity-corrected. Holm exists to
#: suppress false POSITIVES among discovery claims; correcting a safety veto would make harm
#: harder to detect and bias the procedure toward shipping. A guard has to be easy to trip.
VETO_METRICS = ("nDCG@5", "nDCG@10", "R@5", "R@10")


def rrf_scores(
    rankings: Sequence[Sequence[str]],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Weighted Reciprocal Rank Fusion over best-first ID rankings.

    With `weights=None` this is `recall.retriever._rrf` exactly, including the insertion order
    that decides ties downstream. It is re-expressed here rather than imported because the outer
    level needs per-ranking weights, which `_rrf` does not take; `test_multiquery` pins the
    unweighted case against `_rrf` so the two cannot drift.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError(
            f"weights has {len(weights)} entries for {len(rankings)} rankings; a recycled or "
            f"truncated weight vector would mis-weight an arm invisibly"
        )
    scores: dict[str, float] = {}
    for position, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[position]
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank + 1)
    return scores


def sorted_by_score(scores: dict[str, float]) -> list[str]:
    """Best-first, ties by first appearance -- `HybridRetriever`'s own rule.

    Python's sort is stable and `scores` preserves insertion order, so equal RRF scores keep the
    order the rankings introduced them in. RRF ties are common rather than exotic: any two
    documents at the same rank of equally weighted rankings tie exactly.
    """
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def fuse_arm(arm: MultiQueryArm, legs: dict[str, dict[str, list[str]]]) -> list[str]:
    """This arm's fused ranking over one query's dumped per-variant, per-leg rankings.

    A variant the arm declares but the dump lacks raises: fusing whatever is present under a
    three-variant arm's name would report a diversity result for a diversity the run never had,
    and the arm name in the metrics file would still say three.
    """
    missing = [v for v in arm.variants if v not in legs]
    if missing:
        raise KeyError(
            f"arm {arm.name!r} declares variants {missing} that the dump does not contain"
        )

    def leg_rankings(variant: str) -> list[list[str]]:
        present = legs[variant]
        return [list(present.get(leg, []))[: arm.leg_truncate] for leg in LEGS]

    if arm.topology == "flat":
        flattened = [r for variant in arm.variants for r in leg_rankings(variant)]
        # Weights are declared per VARIANT; a flat fusion has no variant level to apply them at,
        # so an arm that asked for both would be silently mis-specified.
        if arm.weights is not None:
            raise ValueError(f"arm {arm.name!r} is flat, so per-variant weights have no level")
        return sorted_by_score(rrf_scores(flattened))
    if arm.topology != "nested":
        raise ValueError(f"unknown topology {arm.topology!r} on arm {arm.name!r}")

    inner = [sorted_by_score(rrf_scores(leg_rankings(v))) for v in arm.variants]
    if arm.weights is not None and len(arm.weights) != len(arm.variants):
        raise ValueError(
            f"arm {arm.name!r} has {len(arm.weights)} weights for {len(arm.variants)} variants"
        )
    return sorted_by_score(rrf_scores(inner, weights=arm.weights))


# --------------------------------------------------------------------------------------------
# dump
# --------------------------------------------------------------------------------------------


def variant_path(root: Path, domain: str, variant: str) -> Path:
    return root / "mtrag-human" / "retrieval_tasks" / domain / f"{domain}_{VARIANT_FILES[variant]}.jsonl"


def load_variant(root: Path, domain: str, variant: str) -> dict[str, str]:
    """One domain's queries for one variant, keyed by task id, speaker prefix removed.

    `strip_speaker` is imported rather than re-spelled: MTRAG-human prefixes `|user|: ` to every
    turn, and left in, that literal token reaches both encoders on every query and depresses the
    entire run with nothing failing.
    """
    out: dict[str, str] = {}
    path = variant_path(root, domain, variant)
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "_id" not in row or "text" not in row:
                raise RuntimeError(f"{path}:{line_no} is missing _id or text")
            out[str(row["_id"])] = strip_speaker(str(row["text"]))
    return out


#: Provenance keys that must agree across every variant of one dump. If two variants disagree on
#: any of these they were retrieved from different indexes or by different encoders, and fusing
#: them compares arms across two systems while reporting one number.
PROVENANCE_KEYS = ("table_prefix", "sparse_model", "sparse_profile", "embedder", "leg_depth")


def manifest_path(out: Path) -> Path:
    return out / "dump_manifest.json"


def read_manifest(out: Path) -> dict[str, dict[str, Any]] | None:
    """Per-variant provenance, or None when there is no readable manifest at all.

    🔑 None and `{}` are DIFFERENT and the distinction is load bearing. None means "this dump
    kept no record, so nothing can be verified". `{}` means "the record parsed and vouches for
    nothing", which is what an interrupted dump leaves behind and which must be a refusal. An
    earlier version collapsed the two, so the default `dump` invocation, whose pre-loop write
    legitimately empties the map, reopened the hole the guard exists to close.

    Migrates the LEGACY shape, in which `variants` was a flat list of names and provenance sat at
    the top level. Migrating rather than discarding matters: every dump directory created before
    per-variant provenance existed is in that shape, including the one behind the published
    numbers, and dropping it would destroy the shortfall counts this manifest exists to preserve.

    Never raises on a damaged manifest. An unreadable manifest is a missing record, not evidence
    of a mixed dump, and killing a fusion whose leg files are intact would be the worse failure.
    """
    def usable(entry: Any) -> bool:
        # An entry vouching for nothing would compare equal to every other such entry and pass as
        # "verified" while verifying nothing. Dropping it makes it a `missing` refusal instead.
        return isinstance(entry, dict) and (
            bool(entry.get("provenance_invalid"))
            or any(key in entry for key in PROVENANCE_KEYS)
        )

    path = manifest_path(out)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    variants = raw.get("variants")
    if isinstance(variants, dict):
        return {k: v for k, v in variants.items() if usable(v)}
    if isinstance(variants, list):
        # Only keys actually PRESENT. Injecting them as None would make `usable` trivially true
        # on the legacy branch, so a legacy manifest that lists variants but records no
        # provenance would migrate into entries that vouch for nothing, compare equal to each
        # other, and pass as verified: exactly what `usable` exists to refuse.
        base = {key: raw[key] for key in
                ("leg_depth", "legs", "embedder", "sparse_model", "sparse_profile",
                 "table_prefix", "split") if key in raw}
        short = raw.get("legs_short_of_depth") or {}
        empty = raw.get("empty_sparse_encodings") or {}
        migrated = {
            name: {
                **base,
                "legs_short_of_depth": short.get(name) if isinstance(short, dict) else None,
                "empty_sparse_encodings": empty.get(name) if isinstance(empty, dict) else None,
                "at": raw.get("at"), "migrated_from_legacy_manifest": True,
            }
            for name in variants if isinstance(name, str)
        }
        return {k: v for k, v in migrated.items() if usable(v)}
    return None


def write_manifest(out: Path, entries: dict[str, dict[str, Any]]) -> None:
    """Replace the manifest atomically.

    It is rewritten once per variant now rather than once per run, and it is the only provenance
    record there is. A truncated body reads back as "no manifest", which is a silent pass, so an
    in-place truncating write turns every interruption into a lost guard.

    Covers process death. NOT full power-loss safety: the containing directory is not fsynced
    (and cannot be portably on Windows), so a kernel-level crash can still lose the rename.
    """
    path = manifest_path(out)
    # PID-suffixed: two dumps sharing an --output-dir would otherwise truncate each other's tmp
    # file and could promote a partial body, which reads back as "no manifest" and is the silent
    # pass. Concurrent dumps into one directory are operator error, but this project ran two
    # sessions against this box on the day it was written, so it is not hypothetical.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump({"variants": entries}, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def cmd_dump(args: argparse.Namespace) -> int:
    """Retrieve every variant of every judged query and record the per-leg ID rankings.

    Both legs are asked for `LEG_DEPTH` rows and any shortfall is COUNTED, per leg per variant.
    That is the `ef_search` trap: `_query_learned_sparse` once returned 6 of 100 candidates and
    nothing errored -- no test caught it, a timing anomaly did. A leg that quietly returns a
    twentieth of what it was asked for reads on a benchmark as "this idea does not help".
    """
    from recall.embeddings import FastEmbedEmbedder, embed_query
    from recall.sparse import SpladeEncoder
    from recall.store import PgVectorStore

    root = args.mtrag_root.resolve()
    out = args.output_dir.resolve()
    (out / "legs").mkdir(parents=True, exist_ok=True)
    dsn = load_dsn(args.dsn_env_file)

    embedder = FastEmbedEmbedder(EMBEDDER_MODEL)
    encoder = SpladeEncoder.from_pretrained(args.sparse_model or DEFAULT_SPARSE_MODEL)
    qrels = load_qrels(root, "dev")

    variants = args.variants or list(VARIANT_FILES)
    shortfalls: dict[str, dict[str, int]] = {}
    empty_sparse: dict[str, int] = {}
    # Mark the variants about to be re-dumped INVALID, and persist that BEFORE writing a single
    # leg file. Leg files are overwritten inside the loop, so a manifest written only at the end
    # would describe freshly overwritten legs with the PREVIOUS run's provenance if the run died
    # partway, and the check would then certify a dump that is half one index and half another.
    #
    # 🔑 Marked, not deleted, for two reasons the first version got wrong. Deleting the entries of
    # a FULL dump leaves `{"variants": {}}`, and an empty map used to be indistinguishable from
    # "no manifest", which is a silent pass: the default invocation reopened the hole. And
    # deleting throws away the shortfall counts of leg files the run may never reach, leaving a
    # previously-good directory strictly worse than before. An explicit invalid flag refuses
    # downstream while keeping the data.
    superseded = utc_now()
    entries = dict(read_manifest(out) or {})
    for variant in variants:
        entries[variant] = {
            **entries.get(variant, {}),
            "provenance_invalid": True, "superseded_at": superseded,
        }
    write_manifest(out, entries)
    provenance = {
        "leg_depth": LEG_DEPTH, "legs": list(LEGS), "embedder": EMBEDDER_MODEL,
        "sparse_model": args.sparse_model or DEFAULT_SPARSE_MODEL,
        "sparse_profile": encoder.profile.profile_id,
        "table_prefix": args.table_prefix, "split": "dev",
    }
    # Constructed inside the try: `PgVectorStore.__init__` opens a live connection, so building
    # these in a comprehension outside it leaks every connection already opened when a later
    # domain's constructor raises.
    stores: dict[str, Any] = {}
    try:
        for domain in DOMAINS:
            stores[domain] = PgVectorStore(
                dsn, embedder.dim, table=table_name(args.table_prefix, domain)
            )
        for variant in variants:
            started = time.perf_counter()
            rows = []
            short = {leg: 0 for leg in LEGS}
            empty = 0
            for domain in DOMAINS:
                texts = load_variant(root, domain, variant)
                for task_id in sorted(qrels[domain]):
                    if task_id not in texts:
                        raise RuntimeError(
                            f"judged query {task_id} absent from {variant} file for {domain}; "
                            f"the three files are supposed to be aligned on _id"
                        )
                    query = texts[task_id]
                    qvec = embed_query(embedder, query)
                    dense = stores[domain].query_dense(qvec, k=LEG_DEPTH)
                    weights = encoder.encode([query])[0]
                    if weights:
                        learned = stores[domain].query_learned_sparse(
                            weights, k=LEG_DEPTH,
                            profile_id=encoder.profile.profile_id, vec=qvec,
                        )
                    else:
                        # A query of pure stopwords legitimately encodes to no terms.
                        # `HybridRetriever` treats that as an empty leg, not a failure.
                        learned = []
                        empty += 1
                    ranking = {"dense": [h.chunk.id for h in dense],
                               "splade": [h.chunk.id for h in learned]}
                    short["dense"] += int(len(ranking["dense"]) < LEG_DEPTH)
                    short["splade"] += int(bool(weights) and len(ranking["splade"]) < LEG_DEPTH)
                    rows.append({
                        "task_id": task_id, "domain": domain, "variant": variant,
                        "query": query, **ranking,
                    })
                    if len(rows) % 100 == 0:
                        print(json.dumps({
                            "event": "progress", "variant": variant, "queries": len(rows),
                            "elapsed_s": round(time.perf_counter() - started, 1),
                        }), flush=True)
            path = out / "legs" / f"{variant}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                # Flushed before this variant's manifest entry replaces its invalid marker,
                # so the ordering the marking scheme relies on is enforced, not assumed.
                handle.flush()
                os.fsync(handle.fileno())
            shortfalls[variant] = short
            empty_sparse[variant] = empty
            # Written now, with the leg file already on disk, so the two cannot disagree.
            entries[variant] = {
                **provenance, "legs_short_of_depth": short,
                "empty_sparse_encodings": empty, "at": utc_now(),
            }  # replaces the invalid marker only once the leg file is on disk
            write_manifest(out, entries)
            print(json.dumps({
                "event": "variant_done", "variant": variant, "queries": len(rows),
                "leg_depth": LEG_DEPTH, "legs_short_of_depth": short,
                "empty_sparse_encodings": empty,
                "elapsed_s": round(time.perf_counter() - started, 1), "at": utc_now(),
            }), flush=True)
    finally:
        for store in stores.values():
            try:
                store.close()
            except Exception:  # noqa: BLE001 - one bad close must not leak the others
                pass

    return 0


def check_provenance(out: Path, variants: Sequence[str]) -> None:
    """Refuse to fuse variants that were not all retrieved by the same system.

    Three outcomes, and the distinction between the last two is the point:

      - every requested variant recorded and agreeing -> silent, verified.
      - a requested variant recorded but DISAGREEING -> raise. Fusing across two indexes or two
        encoders compares arms from different systems and reports one number.
      - a requested variant NOT recorded, while others are -> raise. An unrecorded leg file has
        unknown provenance, and the first version of this check merely SKIPPED it, which made the
        guard toothless against the exact partial re-dump it was written for: with one variant
        recorded the comparison set had one element and could never disagree.

    A manifest that is absent or unreadable verifies nothing, and says so rather than raising:
    that state is a missing record, not evidence of a mixed dump.
    """
    recorded = read_manifest(out)
    if recorded is None:
        print(json.dumps({
            "event": "provenance", "verified": False,
            "why": "no readable manifest; the dump's provenance is NOT verified",
        }), flush=True)
        return

    absent: list[str] = []
    unrecorded: list[str] = []
    in_flight: list[str] = []
    for variant in variants:
        entry = recorded.get(variant)
        if entry is None:
            (absent if not (out / "legs" / f"{variant}.jsonl").exists()
             else unrecorded).append(variant)
        elif entry.get("provenance_invalid"):
            in_flight.append(variant)
    if absent:
        raise RuntimeError(f"these variants were never dumped into {out}: {absent}")
    if in_flight:
        raise RuntimeError(
            f"{in_flight} were being re-dumped when that run stopped, so their leg files may be "
            f"a mixture of two runs. Re-dump them, or point at a clean --output-dir."
        )
    if unrecorded:
        raise RuntimeError(
            f"the dump records provenance for {sorted(recorded)} but not for {unrecorded}, whose "
            f"leg files ARE present. An unrecorded leg file could have come from any index or "
            f"encoder. Re-dump {unrecorded}, or point at a clean --output-dir."
        )
    # json.dumps, not the raw value: a provenance key holding a list or dict is unhashable and
    # would raise TypeError out of `set()`, killing a fusion whose leg files are intact.
    seen = {
        v: tuple(json.dumps(recorded[v].get(k), sort_keys=True, default=str)
                 for k in PROVENANCE_KEYS)
        for v in variants
    }
    if len(set(seen.values())) > 1:
        raise RuntimeError(
            f"the dump's variants disagree on how they were retrieved: {seen}. Fusing them would "
            f"compare arms across different indexes or encoders and report it as one result. "
            f"Re-dump the odd variant, or point at a clean --output-dir."
        )


def load_legs(out: Path, variants: Sequence[str]) -> dict[str, dict[str, dict[str, list[str]]]]:
    """`{task_id: {variant: {leg: ranking}}}`, plus the query texts, from a dump."""
    check_provenance(out, variants)
    by_task: dict[str, dict[str, dict[str, list[str]]]] = {}
    for variant in variants:
        path = out / "legs" / f"{variant}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                by_task.setdefault(str(row["task_id"]), {})[variant] = {
                    leg: [str(c) for c in row.get(leg, [])] for leg in LEGS
                }
    return by_task


def load_queries(out: Path, variant: str) -> dict[str, str]:
    texts: dict[str, str] = {}
    with (out / "legs" / f"{variant}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                texts[str(row["task_id"])] = str(row["query"])
    return texts


# --------------------------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    """Require the reconstructed control to match a live `HybridRetriever` EXACTLY.

    This is the gate. `mq_last` is configuration-identical to `hybrid_splade`, so if rebuilding it
    from the dumped legs does not reproduce what RE-call itself returns, every number this harness
    goes on to report is a number RE-call would never compute.

    Exact, not approximate: the whole pipeline is deterministic (fixed HNSW index, fixed
    `ef_search`, deterministic encoders), so unlike the GPU/CPU float comparison in
    `rerank_offload.validate` there is no tolerance to allow here.
    """
    from recall.embeddings import FastEmbedEmbedder
    from recall.retriever import HybridRetriever
    from recall.sparse import SpladeEncoder
    from recall.store import PgVectorStore

    out = args.output_dir.resolve()
    dsn = load_dsn(args.dsn_env_file)
    embedder = FastEmbedEmbedder(EMBEDDER_MODEL)
    encoder = SpladeEncoder.from_pretrained(args.sparse_model or DEFAULT_SPARSE_MODEL)
    legs = load_legs(out, ["last"])
    texts = load_queries(out, "last")
    domains = {}
    with (out / "legs" / "last.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                domains[str(row["task_id"])] = str(row["domain"])

    control = next(a for a in MQ_ARMS if a.name == "mq_last")
    if args.sample < 1:
        raise ValueError(
            f"--sample must be >= 1, got {args.sample}. A zero sample prints verdict MATCH "
            f"having compared nothing, and this project has already shipped one gate that could "
            f"never fail; that is exactly as useless as one that could never pass."
        )
    sample = sorted(legs)[: args.sample]
    failures = []
    stores = {}
    try:
        for task_id in sample:
            domain = domains[task_id]
            if domain not in stores:
                stores[domain] = PgVectorStore(
                    dsn, embedder.dim, table=table_name(args.table_prefix, domain)
                )
            retriever = HybridRetriever(
                stores[domain], embedder, reranker=None, candidate_k=LEG_DEPTH,
                use_dense=True, use_sparse=True, sparse_backend="splade",
                sparse_encoder=encoder,
            )
            live = [h.chunk.id for h in retriever.search(texts[task_id], k=EVAL_K).hits]
            rebuilt = fuse_arm(control, legs[task_id])[:EVAL_K]
            if live != rebuilt:
                first = next(
                    (i for i, (a, b) in enumerate(zip(live, rebuilt, strict=False)) if a != b),
                    min(len(live), len(rebuilt)),
                )
                failures.append({"task_id": task_id, "first_divergent_rank": first,
                                 "live": live[first: first + 3],
                                 "rebuilt": rebuilt[first: first + 3]})
    finally:
        for store in stores.values():
            store.close()

    verdict = "MATCH" if not failures else "MISMATCH"
    sampled_domains: dict[str, int] = {}
    for task_id in sample:
        sampled_domains[domains[task_id]] = sampled_domains.get(domains[task_id], 0) + 1
    print(json.dumps({"event": "validate", "sampled": len(sample), "arm": control.name,
                      "sampled_domains": sampled_domains,
                      "failures": failures[:5], "failure_count": len(failures),
                      "verdict": verdict}, indent=2), flush=True)
    return 0 if not failures else 1


# --------------------------------------------------------------------------------------------
# fuse
# --------------------------------------------------------------------------------------------


def cmd_fuse(args: argparse.Namespace) -> int:
    """Compute every declared arm from the dumped legs and score it against the dev qrels.

    Every arm reads the same dumped rankings, so no contrast between them can be contaminated by
    retrieval jitter -- the only thing that varies is the fusion.
    """
    root = args.mtrag_root.resolve()
    out = args.output_dir.resolve()
    qrels_by_domain = load_qrels(root, "dev")
    qrels: dict[str, set[str]] = {}
    for domain in DOMAINS:
        qrels.update(qrels_by_domain[domain])

    wanted = sorted({v for arm in MQ_ARMS for v in arm.variants})
    legs = load_legs(out, wanted)
    shared = sorted(set(legs) & set(qrels))
    print(json.dumps({"event": "fuse_setup", "queries": len(shared),
                      "variants": wanted}), flush=True)

    if not shared:
        raise RuntimeError(
            "no query is present in both the dump and the dev qrels; check --mtrag-root points "
            "at the release the dump was taken from"
        )

    rankings: dict[str, dict[str, list[str]]] = {}
    summaries = []
    for arm in MQ_ARMS:
        per_query = {}
        pool_sizes: list[int] = []
        metrics: dict[str, list[float]] = {
            "nDCG@5": [], "R@5": [], "R@10": [], "R@100": []}
        for task_id in shared:
            complete = fuse_arm(arm, legs[task_id])
            # The REALISED pool, measured before truncation. `pool_bound()` is a no-overlap upper
            # bound and overstates this badly wherever the variants agree, which is exactly where
            # a pool-limited R@100 would otherwise pass unnoticed.
            pool_sizes.append(len(complete))
            fused = complete[:EVAL_K]
            per_query[task_id] = fused
            relevant = qrels[task_id]
            metrics["nDCG@5"].append(ndcg_at(fused, relevant, 5))
            metrics["R@5"].append(recall_at(fused, relevant, 5))
            metrics["R@10"].append(recall_at(fused, relevant, 10))
            metrics["R@100"].append(recall_at(fused, relevant, EVAL_K))
        rankings[arm.name] = per_query
        summary = {
            "arm": asdict(arm), "queries": len(shared), "pool_bound": arm.pool_bound(),
            "realised_pool_sizes": realised_pool_sizes(pool_sizes),
            **{name: sum(vals) / len(vals) for name, vals in metrics.items()},
        }
        summaries.append(summary)
        print(json.dumps({"event": "arm_scored", "arm": arm.name,
                          **{k: round(v, 4) for k, v in summary.items()
                             if isinstance(v, float)}}), flush=True)

    with (out / "rankings.json").open("w", encoding="utf-8") as handle:
        json.dump(rankings, handle)
    (out / "arm_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    ceilings = pool_ceilings(legs, qrels, shared)
    (out / "pool_ceilings.json").write_text(json.dumps(ceilings, indent=2), encoding="utf-8")
    print(json.dumps({"event": "pool_ceilings", **ceilings}, indent=2), flush=True)
    return 0


def realised_pool_sizes(sizes: Sequence[int]) -> dict[str, Any]:
    """Distribution of how many distinct documents an arm's fusion actually held.

    🔑 `pool_below_cutoff` is the number that matters: on those queries the arm's R@100 is capped
    by how much it retrieved, not by how well it ranked, so the metric is measuring the pool. An
    arm with a large count here is being compared unfairly against one with none, and neither
    `pool_bound()` nor any score in the summary would show it.
    """
    ordered = sorted(sizes)
    n = len(ordered)
    return {
        "min": ordered[0] if n else None,
        "p25": ordered[n // 4] if n else None,
        "median": ordered[n // 2] if n else None,
        "max": ordered[-1] if n else None,
        "eval_k": EVAL_K,
        "pool_below_cutoff": sum(1 for s in ordered if s < EVAL_K),
    }


def pool_ceilings(
    legs: dict[str, dict[str, dict[str, list[str]]]],
    qrels: dict[str, set[str]],
    shared: Sequence[str],
) -> dict[str, Any]:
    """Diagnostic, NOT a preregistered contrast: how much gold is in the pool at all?

    An arm's R@100 is bounded by the gold in the candidate pool it truncates. Reporting the union
    coverage of each variant, and of all three together, separates two very different failures:

      - the three-variant union is barely above the single-variant one -> the reformulations find
        the SAME documents, and no fusion rule can help. The lever is dead for a reason no
        weighting scheme addresses.
      - the union is far above what any arm achieves -> the raw material is there and the fusion
        rule is what is losing it, which is a tractable problem.

    This is the same question `hybrid_splade`'s "full-pool ceiling" answered for the SPLADE leg,
    and it costs nothing once the legs are on disk.
    """
    def union_recall(task_id: str, variants: Sequence[str]) -> float:
        pool: set[str] = set()
        for variant in variants:
            for leg in LEGS:
                pool.update(legs[task_id][variant].get(leg, []))
        relevant = qrels[task_id]
        return len(pool & relevant) / len(relevant) if relevant else 0.0

    out: dict[str, Any] = {"queries": len(shared)}
    for label, variants in (
        ("last", ("last",)), ("full", ("full",)), ("rewrite", ("rewrite",)),
        ("last+rewrite", ("last", "rewrite")),
        ("all_three", ("last", "full", "rewrite")),
    ):
        values = [union_recall(t, variants) for t in shared]
        out[f"pool_recall_{label}"] = sum(values) / len(values)
    out["headroom_over_last"] = out["pool_recall_all_three"] - out["pool_recall_last"]
    return out


# --------------------------------------------------------------------------------------------
# contrast
# --------------------------------------------------------------------------------------------


def decide_verdict(
    primary: dict[str, Any],
    mechanism: dict[str, Any],
    vetoes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The preregistered decision rule, applied to P1, M1 and the ranking vetoes. Pure, testable.

    P1 ships only on all of: point estimate at or above the bar, CI excluding zero, Holm
    significance, AND no ranking veto tripped. A significant NEGATIVE P1 is reported as a
    REGRESSION rather than folded in with "real but sub-threshold", which would file a measured
    harm under a label that reads like an encouraging result one more push would clear.

    A veto trips when a ranking metric's whole 95% CI sits BELOW zero, i.e. the arm is
    established to degrade that metric. A merely negative point estimate does not trip it: a point
    estimate is not a result in either direction.

    M1 decides the MECHANISM, and it is reported whatever P1 does. Gold rewrite is an unattainable
    ceiling for an LLM rewriter, so a fusion that cannot beat it offers no headroom an LLM could
    reach -- which is why a large P1 with a null M1 closes the lever instead of opening it.
    """
    tripped = sorted(
        metric for metric, stats in (vetoes or {}).items() if stats["ci_high"] < 0
    )
    established = primary["ci_excludes_zero"] and primary["holm_significant"]
    if established and primary["mean_delta"] >= SHIP_BAR and not tripped:
        verdict = "SHIPS"
    elif established and primary["mean_delta"] >= SHIP_BAR:
        # Coverage cleared the bar and the ranking paid for it. Reporting this as SHIPS would be
        # the exact failure the veto was added to prevent.
        verdict = "BLOCKED_BY_RANKING_REGRESSION"
    elif established and primary["mean_delta"] > 0:
        verdict = "INCONCLUSIVE_REAL_BUT_SUB_THRESHOLD"
    elif established:
        verdict = "REGRESSES"
    else:
        verdict = "DOES_NOT_SHIP"

    fusion = mechanism["ci_excludes_zero"] and mechanism["mean_delta"] > 0
    return {
        "verdict": verdict,
        "mechanism_verdict": "FUSION" if fusion else "REWRITING_QUALITY_OR_UNRESOLVED",
        "vetoes_tripped": tripped,
        "gpu_rental_justified": fusion and verdict in (
            "SHIPS", "INCONCLUSIVE_REAL_BUT_SUB_THRESHOLD", "BLOCKED_BY_RANKING_REGRESSION"
        ),
    }


def coverage_destination(
    rankings: dict[str, dict[str, list[str]]],
    qrels: dict[str, set[str]],
    shared: Sequence[str],
    treatment: str,
    control: str,
) -> dict[str, Any]:
    """Where does the gold that fusion ADDS actually land in the fused ranking?

    The same question closed the reranker lever: SPLADE's extra gold was real, and both
    cross-encoders buried ~90 of 123 documents below rank 10, which is why more coverage stopped
    being worth chasing through ranking.

    ⚠️ Read with the caveat this function prints. These ranks are from the RAW fused ordering. On
    2026-08-06 the SPLADE-only gold also looked reasonable raw and was then buried by two
    different cross-encoders, so a healthy distribution here is NOT evidence the documents survive
    reranking. Establishing that needs the new pairs scored on a GPU.
    """
    ranks: list[int] = []
    queries_with_extra = 0
    for task_id in shared:
        relevant = qrels[task_id]
        treated = rankings[treatment][task_id]
        extra = (set(treated) & relevant) - set(rankings[control][task_id])
        if extra:
            queries_with_extra += 1
            ranks.extend(treated.index(doc) for doc in extra)
    ranks.sort()
    n = len(ranks)
    return {
        "arm": treatment, "vs": control,
        "queries_with_added_gold": queries_with_extra,
        "added_gold_documents": n,
        "rank": {
            "min": ranks[0] if n else None,
            "p25": ranks[n // 4] if n else None,
            "median": ranks[n // 2] if n else None,
            "p75": ranks[3 * n // 4] if n else None,
            "max": ranks[-1] if n else None,
        },
        "reaching_top_5": sum(1 for r in ranks if r < 5),
        "reaching_top_10": sum(1 for r in ranks if r < 10),
        "buried_below_10": sum(1 for r in ranks if r >= 10),
        "caveat": (
            "raw fused ranking. SPLADE-only gold looked comparable here on 2026-08-06 and was "
            "then buried below rank 10 by BOTH cross-encoders, so this is not evidence the "
            "documents survive reranking; that needs the added pairs scored on a GPU."
        ),
    }


def cmd_contrast(args: argparse.Namespace) -> int:
    """The preregistered contrasts: paired bootstrap CI, sign-flip permutation, Holm at 0.05.

    `paired_stats` and `holm` are imported from `analyse_contrasts` unchanged. The standards
    require a specific procedure and a second implementation of it is a second thing that can be
    wrong, not a second opinion.

    Two means are reported per contrast. The FULL-SET mean over all judged queries is what the
    decision rule reads, because it is what would ship. The DECIDING-CELL mean covers only the
    queries the contrast can actually move; 102 of 777 dev queries carry byte-identical variants,
    where most contrasts' delta is 0.0 by construction, so the full-set mean is diluted by about
    13% and reporting only it would understate the mechanism while overstating the deployable
    gain.

    Which cell that is depends on the contrast, and `CONTRASTS` carries it per row. B1 is the
    exception: it compares truncated legs against untruncated ones, so its delta does not vanish
    on identical variants and its cell is all 777.
    """
    root = args.mtrag_root.resolve()
    out = args.output_dir.resolve()
    qrels_by_domain = load_qrels(root, "dev")
    qrels: dict[str, set[str]] = {}
    for domain in DOMAINS:
        qrels.update(qrels_by_domain[domain])
    with (out / "rankings.json").open(encoding="utf-8") as handle:
        rankings = json.load(handle)

    variant_texts = {v: load_queries(out, v) for v in VARIANT_FILES}
    shared = sorted(rankings["mq_last"])

    def distinct_variants(task_id: str) -> int:
        return len({variant_texts[v][task_id] for v in VARIANT_FILES})

    by_name = {arm.name: arm for arm in MQ_ARMS}

    def deciding(task_id: str, treatment: str, control: str) -> bool:
        """Can this contrast's delta be non-zero on this query, even in principle?

        Derived from the UNION of the variants the two arms actually fuse, rather than from a
        fixed "all three variants" rule, so a contrast added later over a different variant set
        gets the right cell instead of inheriting this one's.

        ⚠️ The boundary is ">= 2 distinct within that union", not "all 3 distinct". Where
        `rewrite` equals `last` but `full` differs, R1 (`mq_nested2` vs `mq_nested3`) still has a
        non-zero delta, because the arm on the other side of that contrast fuses `full` and `full`
        is the thing that differs. Tightening to "all 3 distinct" would drop 107 queries on which
        the contrast genuinely acts.
        """
        variants = set(by_name[treatment].variants) | set(by_name[control].variants)
        return len({variant_texts[v][task_id] for v in variants}) > 1

    rng = random.Random(20260806)
    results: dict[str, dict[str, Any]] = {}
    details: dict[str, dict[str, Any]] = {}
    for cid, treatment, control, question, cell in CONTRASTS:
        deltas, cell_deltas, distinct3_deltas = [], [], []
        for task_id in shared:
            relevant = qrels[task_id]
            delta = (recall_at(rankings[treatment][task_id], relevant, EVAL_K)
                     - recall_at(rankings[control][task_id], relevant, EVAL_K))
            deltas.append(delta)
            if cell == "all" or deciding(task_id, treatment, control):
                cell_deltas.append(delta)
            if distinct_variants(task_id) == 3:
                distinct3_deltas.append(delta)
        results[cid] = paired_stats(deltas, rng)
        details[cid] = {
            "treatment": treatment, "control": control, "question": question,
            "metric": DECISION_METRIC, "deciding_cell": cell,
            "deciding_cell_n": len(cell_deltas),
            "deciding_cell_mean_delta": sum(cell_deltas) / len(cell_deltas) if cell_deltas else None,
            # Third population, contributed by a parallel session that independently derived the
            # same 209 / 102 / 568 split before it was stopped. Strictly narrower than the
            # deciding cell: it asks what the contrast does when all three reformulations are
            # genuinely different, which is the cleanest reading of the MECHANISM.
            "all_three_distinct_n": len(distinct3_deltas),
            "all_three_distinct_mean_delta": (
                sum(distinct3_deltas) / len(distinct3_deltas) if distinct3_deltas else None
            ),
        }
    holm(results)

    for cid, *_rest in CONTRASTS:
        print(json.dumps({"event": "contrast", "id": cid, **details[cid], **results[cid]}),
              flush=True)

    # Ranking vetoes, on the SAME arm pair as P1. Uncorrected on purpose: see VETO_METRICS.
    p1_treatment, p1_control = CONTRASTS[0][1], CONTRASTS[0][2]
    vetoes: dict[str, dict[str, Any]] = {}
    for metric in VETO_METRICS:
        kind, _, depth = metric.partition("@")
        k = int(depth)
        score = ndcg_at if kind == "nDCG" else recall_at
        vetoes[metric] = paired_stats(
            [score(rankings[p1_treatment][t], qrels[t], k)
             - score(rankings[p1_control][t], qrels[t], k) for t in shared],
            rng,
        )
        print(json.dumps({"event": "veto", "metric": metric, "treatment": p1_treatment,
                          "control": p1_control, **vetoes[metric]}), flush=True)

    destination = coverage_destination(rankings, qrels, shared, p1_treatment, p1_control)
    print(json.dumps({"event": "coverage_destination", **destination}, indent=2), flush=True)

    primary = results["P1"]
    decision = {
        "event": "decision", "metric": DECISION_METRIC, "ship_bar": SHIP_BAR,
        "primary": {"id": "P1", **details["P1"], **primary},
        **decide_verdict(primary, results["M1"], vetoes),
        "vetoes": vetoes,
        "coverage_destination": destination,
        "contrasts": {cid: {**details[cid], **results[cid]} for cid, *_ in CONTRASTS},
        "at": utc_now(),
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in decision.items() if k != "contrasts"}, indent=2), flush=True)
    return 0


def cmd_posthoc(args: argparse.Namespace) -> int:
    """Score the POST-HOC arms and contrast each against `mq_last`. Reported separately, labelled.

    Same statistics as the frozen family, deliberately NOT merged into it: adding an arm to a
    preregistered family after seeing its results is how a post-hoc rescue gets published as a
    prediction. Everything this writes is marked `post_hoc: true`.
    """
    root = args.mtrag_root.resolve()
    out = args.output_dir.resolve()
    qrels_by_domain = load_qrels(root, "dev")
    qrels: dict[str, set[str]] = {}
    for domain in DOMAINS:
        qrels.update(qrels_by_domain[domain])

    legs = load_legs(out, sorted({v for arm in POST_HOC_ARMS for v in arm.variants}))
    with (out / "rankings.json").open(encoding="utf-8") as handle:
        control_rankings = json.load(handle)["mq_last"]
    shared = sorted(set(legs) & set(qrels) & set(control_rankings))
    if not shared:
        raise RuntimeError("no query is present in the dump, the qrels and mq_last's rankings")

    rng = random.Random(20260806)
    rows = []
    for arm in POST_HOC_ARMS:
        fused = {t: fuse_arm(arm, legs[t])[:EVAL_K] for t in shared}
        metrics = {
            "nDCG@5": [ndcg_at(fused[t], qrels[t], 5) for t in shared],
            "R@5": [recall_at(fused[t], qrels[t], 5) for t in shared],
            "R@10": [recall_at(fused[t], qrels[t], 10) for t in shared],
            "R@100": [recall_at(fused[t], qrels[t], EVAL_K) for t in shared],
        }
        deltas = [recall_at(fused[t], qrels[t], EVAL_K)
                  - recall_at(control_rankings[t], qrels[t], EVAL_K) for t in shared]
        # The SAME ranking veto the frozen family applies. Reporting a post-hoc coverage gain
        # without it would be exactly the one-sided reading the veto was added to prevent, and
        # being post-hoc is a reason for MORE scrutiny, not less.
        vetoes = {}
        for metric in VETO_METRICS:
            kind, _, depth = metric.partition("@")
            k = int(depth)
            score = ndcg_at if kind == "nDCG" else recall_at
            vetoes[metric] = paired_stats(
                [score(fused[t], qrels[t], k) - score(control_rankings[t], qrels[t], k)
                 for t in shared], rng,
            )
        tripped = sorted(m for m, s in vetoes.items() if s["ci_high"] < 0)
        delta_stats: dict[str, Any] = paired_stats(deltas, rng)
        row = {
            "post_hoc": True, "arm": asdict(arm), "queries": len(shared),
            **{name: sum(vals) / len(vals) for name, vals in metrics.items()},
            "vs_mq_last_R@100": delta_stats,
            "vs_mq_last_vetoes": vetoes,
            "vetoes_tripped": tripped,
        }
        rows.append(row)
        print(json.dumps({"event": "post_hoc_arm", "arm": arm.name, "post_hoc": True,
                          **{k: round(v, 4) for k, v in row.items() if isinstance(v, float)},
                          "delta_R@100": round(delta_stats["mean_delta"], 4),
                          "ci": [round(delta_stats["ci_low"], 4),
                                 round(delta_stats["ci_high"], 4)],
                          "vetoes_tripped": tripped}), flush=True)
    (out / "posthoc.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("dump", "validate", "fuse", "contrast", "posthoc"):
        p = sub.add_parser(name)
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--mtrag-root", type=Path,
                       required=(name in ("dump", "fuse", "contrast", "posthoc")),
                       default=Path("."))
        if name in ("dump", "validate"):
            p.add_argument("--dsn-env-file", type=Path)
            p.add_argument("--table-prefix", default="recall_mtrag_bge_v1")
            p.add_argument("--sparse-model", default=None)
        if name == "dump":
            p.add_argument("--variants", nargs="*", choices=list(VARIANT_FILES))
        if name == "validate":
            p.add_argument("--sample", type=int, default=25)
    args = parser.parse_args(argv)
    return {"dump": cmd_dump, "validate": cmd_validate, "fuse": cmd_fuse,
            "contrast": cmd_contrast, "posthoc": cmd_posthoc}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
