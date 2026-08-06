"""Does the multi-query coverage gain CONVERT under reranking? Equal-width, on frozen pools.

Preregistration: `docs/superpowers/specs/2026-08-06-multi-query-rerank-design.md`.

Prior work (searched 2026-08-06, `docs_search(source_type="memory")`, no gap warning):

  - [[project-recall-splade-learned-sparse-measured-2026-08-06]] -- the precedent this exists to
    check against: SPLADE moved R@100 +0.0303 and reranked nDCG@5 came out BEHIND the control.
  - [[closed-hypothesis-recall-rerank-pool-interaction-2026-08-05]] -- the SAME MiniLM gets WORSE
    as the pool widens. This is why every arm here is reranked at an IDENTICAL width.
  - [[reference-validation-standards]] -- bootstrap CI, permutation, Holm. Imported, not rewritten.

🔑 **Equal width, and what it rules out.** RE-call reranks the whole fused pool and truncates
after. The arms' realised pools differ by nearly 2x (`mq_last` median 167, `mq_nested3` 315), and
width alone is known to move this result, so a whole-pool comparison would hand the primary arm
more rope and report the consequence as an arm effect. Every arm is therefore reranked over
exactly the top `EVAL_K` of its own fused ranking.

⚠️ The direct consequence: **R@100 is INVARIANT under this design.** Permuting a fixed
100-document set cannot change its membership, so reranked R@100 equals raw R@100 by construction
for every arm. It is asserted here rather than reported, precisely so nobody quotes it as evidence
that the coverage converted. The decision metric is nDCG@5.

⚠️ This is a CONTROLLED CONTRAST, not an end-to-end system measurement. Production reranks the
whole pool; a whole-pool run would answer a different question and its numbers are not comparable.

    pairs     -- extract the top-EVAL_K of each arm, fetch passage texts, emit the scoring payload
    apply     -- fold scores back in, reorder, and run the preregistered contrasts
    validate  -- require the offloaded ordering to match a live CrossEncoderReranker

Scoring itself is `scripts/score_pairs.py`, unchanged: it already pins RE-call's default
cross-encoder and takes no DSN.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from benchmarks.mtrag.analyse_contrasts import holm, ndcg_at, paired_stats, recall_at
from benchmarks.mtrag.multiquery import EVAL_K, VARIANT_FILES, load_queries
from benchmarks.mtrag.rerank_offload import rerank_order
from benchmarks.mtrag.run import DOMAINS, load_dsn, load_qrels, table_name, utc_now

#: Frozen 2026-08-06 before any pair was scored. `mq_nested2_nogold` was post-hoc in the raw run
#: and is preregistered here, because it is the only arm that could actually be deployed.
RERANK_ARMS = ("mq_last", "mq_nested3", "mq_nested2_nogold")

#: Decision metric. NOT R@100: see the module docstring, it is invariant by construction.
DECISION_METRIC = "nDCG@5"

#: Reported alongside, not in the Holm family.
SECONDARY_METRICS = ("nDCG@10", "R@5", "R@10")

#: The preregistered Holm family, all on `DECISION_METRIC`.
RERANK_CONTRASTS = (
    ("C1", "mq_nested3", "mq_last", "does the coverage convert"),
    ("C2", "mq_nested2_nogold", "mq_last", "does the deployable arm's ranking cost survive"),
    ("C3", "mq_nested3", "mq_nested2_nogold", "what gold is worth once reranked"),
)

#: "Materially converts" bar on C1, in nDCG@5.
CONVERSION_BAR = 0.010


def metric_fn(name: str):
    kind, _, depth = name.partition("@")
    k = int(depth)
    score = ndcg_at if kind == "nDCG" else recall_at
    return lambda ranked, relevant: score(ranked, relevant, k)


def load_arm_rankings(mq_dir: Path, whole_pool: bool = False) -> dict[str, dict[str, list[str]]]:
    """The frozen rankings from the raw run's artifacts.

    `whole_pool=False` (default) gives the top-`EVAL_K` of each arm: the EQUAL-WIDTH design, which
    is the preregistered primary. `whole_pool=True` gives each arm's complete fused ranking, which
    is what RE-call itself would rerank in production and is reported as a SECONDARY, confounded
    with pool width by construction.

    `mq_nested2_nogold` lives in the post-hoc file rather than `rankings.json`, so it is rebuilt
    from the same archived legs with the same `fuse_arm` the raw run used. Rebuilding rather than
    re-fusing differently is the point: this stage must rerank exactly what was published.
    """
    from benchmarks.mtrag.multiquery import MQ_ARMS, POST_HOC_ARMS, fuse_arm, load_legs

    declared = {a.name: a for a in (*MQ_ARMS, *POST_HOC_ARMS)}
    undeclared = [name for name in RERANK_ARMS if name not in declared]
    if undeclared:
        raise RuntimeError(
            f"arms {undeclared} are in RERANK_ARMS but declared in neither MQ_ARMS nor "
            f"POST_HOC_ARMS, so they cannot be rebuilt from the legs dump"
        )
    if whole_pool:
        legs = load_legs(mq_dir, list(VARIANT_FILES))
        return {
            name: {t: fuse_arm(declared[name], legs[t]) for t in legs} for name in RERANK_ARMS
        }

    with (mq_dir / "rankings.json").open(encoding="utf-8") as handle:
        rankings = json.load(handle)
    out = {name: rankings[name] for name in RERANK_ARMS if name in rankings}
    missing = [name for name in RERANK_ARMS if name not in out]
    if missing:
        legs = load_legs(mq_dir, list(VARIANT_FILES))
        for name in missing:
            out[name] = {t: fuse_arm(declared[name], legs[t])[:EVAL_K] for t in legs}
    for name, per_query in out.items():
        deep = [t for t, r in per_query.items() if len(r) > EVAL_K]
        if deep:
            raise RuntimeError(
                f"arm {name} has rankings longer than {EVAL_K} (e.g. {deep[:3]}); the equal-width "
                f"design requires every arm to hand the cross-encoder the same number of documents"
            )
    return out


def cmd_pairs(args: argparse.Namespace) -> int:
    """Emit the scoring payload: queries, passage texts, and the unique pairs to score.

    Passage TEXT is not in the legs dump (it records ids only), so it is fetched from the same
    tables the run retrieved from. A document is fetched from the table of the domain whose query
    surfaced it, which is how the four corpora stay separate.
    """
    from recall.store import PgVectorStore

    mq_dir = args.mq_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dsn = load_dsn(args.dsn_env_file)

    rankings = load_arm_rankings(mq_dir)
    # One scoring run must cover BOTH analyses, or answering the obvious follow-up costs a second
    # GPU rental. The whole-pool pairs are a superset of the equal-width ones, so scoring their
    # union once yields the primary contrast and the end-to-end system number together.
    whole = load_arm_rankings(mq_dir, whole_pool=True) if args.include_whole_pool else {}
    queries = load_queries(mq_dir, "last")
    domains: dict[str, str] = {}
    with (mq_dir / "legs" / "last.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                domains[str(row["task_id"])] = str(row["domain"])

    # Every arm reranks the query it was BUILT from, which is the last turn for all three declared
    # arms (the other variants feed retrieval, not the cross-encoder's query side).
    wanted: dict[str, set[str]] = {domain: set() for domain in DOMAINS}
    pairs: set[tuple[str, str]] = set()
    for source in (rankings, whole):
        for per_query in source.values():
            for task_id, ranked in per_query.items():
                for doc_id in ranked:
                    wanted[domains[task_id]].add(doc_id)
                    pairs.add((task_id, doc_id))

    texts: dict[str, str] = {}
    for domain in DOMAINS:
        ids = sorted(wanted[domain])
        if not ids:
            continue
        with PgVectorStore(dsn, args.dim, table=table_name(args.table_prefix, domain)) as store:
            found = 0
            for chunk in store.iter_chunks():
                if chunk.id in wanted[domain]:
                    texts[chunk.id] = chunk.text
                    found += 1
            print(json.dumps({"event": "texts", "domain": domain, "wanted": len(ids),
                              "found": found}), flush=True)

    absent = sorted({d for _q, d in pairs} - set(texts))
    if absent:
        raise RuntimeError(
            f"{len(absent)} retrieved documents have no text in the corpus tables, e.g. "
            f"{absent[:3]}. Scoring them as empty strings would produce plausible logits for "
            f"passages the reranker never saw."
        )

    for name, payload in (
        ("queries.jsonl", [{"qid": q, "text": queries[q]} for q in sorted(queries)]),
        ("docs.jsonl", [{"doc_id": d, "text": texts[d]} for d in sorted(texts)]),
        ("pairs.jsonl", [{"qid": q, "doc_id": d} for q, d in sorted(pairs)]),
    ):
        with (out / name).open("w", encoding="utf-8") as handle:
            for row in payload:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Freeze BOTH ranking sets next to the payload. `apply` must rerank exactly what was scored,
    # and re-deriving them later from a possibly-moved legs dump is how a mismatch creeps in.
    (out / "rankings_equal_width.json").write_text(json.dumps(rankings), encoding="utf-8")
    pool_stats = {}
    if whole:
        (out / "rankings_whole_pool.json").write_text(json.dumps(whole), encoding="utf-8")
        for name, per_query in whole.items():
            sizes = sorted(len(r) for r in per_query.values())
            pool_stats[name] = {"min": sizes[0], "median": sizes[len(sizes) // 2],
                                "max": sizes[-1]}

    print(json.dumps({
        "event": "pairs_done", "arms": sorted(rankings), "queries": len(queries),
        "unique_docs": len(texts), "pairs_to_score": len(pairs),
        "eval_k": EVAL_K, "includes_whole_pool": bool(whole),
        "whole_pool_sizes": pool_stats, "at": utc_now(),
    }, indent=2), flush=True)
    return 0


def _load_scores(path: Path) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Scores, and the header naming the model that produced them.

    The header is RETURNED rather than printed and dropped: it carries the model and revision, and
    a decision file that does not say which cross-encoder produced it cannot be told apart from
    one produced by the other.
    """
    scores: dict[str, dict[str, float]] = {}
    headers: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("_header"):
                headers.append(row)
                print(json.dumps({"event": "scores_header", **row}), flush=True)
                continue
            scores.setdefault(str(row["qid"]), {})[str(row["doc_id"])] = float(row["score"])
    # An absent header is not an empty one. `scripts/score_pairs.py` always writes exactly one, so
    # zero means the file was filtered or hand-assembled and cannot be attributed to a model;
    # recording `{}` would put an unattributable decision on disk looking like an attributed one.
    if not headers:
        raise RuntimeError(
            f"{path} has no _header line, so the decision it produces could not name the model "
            f"that scored it. `scripts/score_pairs.py` writes one; this file was filtered or "
            f"assembled by hand."
        )
    if len(headers) > 1:
        models = sorted({str(h.get("model")) for h in headers})
        raise RuntimeError(
            f"{path} has {len(headers)} _header lines ({models}); it is a concatenation of "
            f"separate scoring runs, and one model name over a mixed score set is a fiction."
        )
    return scores, headers[0]


def frozen_rankings(out: Path, mq_dir: Path, whole_pool: bool) -> dict[str, dict[str, list[str]]]:
    """The rankings the scores were produced against, preferring the file `cmd_pairs` froze.

    Re-deriving them from `--mq-dir` is a real hazard, not a theoretical one: a re-fused or
    re-dumped legs directory that merely REORDERS the same 100 documents would leave the reranked
    numbers untouched while the `raw_` baselines silently describe a ranking that was never
    published. Falling back is still allowed, for a payload built before these files existed.
    """
    name = "rankings_whole_pool.json" if whole_pool else "rankings_equal_width.json"
    path = out / name
    if path.exists():
        try:
            with path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
        except json.JSONDecodeError as exc:
            # `cmd_pairs` writes these with a non-atomic write_text, so an interrupted run leaves
            # a partial file. A truncated frozen ranking is not an absent one and must not read as
            # agreement, so this refuses rather than quietly re-deriving.
            raise RuntimeError(
                f"{path} exists but is not valid JSON ({exc}); it was probably written by an "
                f"interrupted `pairs` run. Delete it to re-derive, or re-run `pairs`."
            ) from exc
        # The preferred path must be at least as strict as the fallback it replaced. An arm
        # MISSING gives a bare KeyError three loops later; an EXTRA arm is worse, because `shared`
        # intersects over every arm present and would silently shrink the query population that
        # every published number is computed on.
        declared, present = set(RERANK_ARMS), set(loaded)
        if present != declared:
            raise RuntimeError(
                f"{path} holds arms {sorted(present)} but this run declares "
                f"{sorted(declared)}. Missing: {sorted(declared - present)}; "
                f"unexpected: {sorted(present - declared)}. Re-run `pairs` against the current "
                f"RERANK_ARMS rather than mixing the two."
            )
        return loaded
    print(json.dumps({"event": "frozen_rankings_absent", "expected": str(path),
                      "falling_back_to": str(mq_dir)}), flush=True)
    return load_arm_rankings(mq_dir, whole_pool=whole_pool)


def cmd_apply(args: argparse.Namespace) -> int:
    """Reorder each arm by cross-encoder score and run the preregistered contrasts."""
    mq_dir = args.mq_dir.resolve()
    out = args.output_dir.resolve()
    # BUG-006: without this, a fresh --output-dir crashes on the final write, AFTER every
    # bootstrap and permutation test has run.
    out.mkdir(parents=True, exist_ok=True)
    qrels_by_domain = load_qrels(args.mtrag_root.resolve(), "dev")
    qrels: dict[str, set[str]] = {}
    for domain in DOMAINS:
        qrels.update(qrels_by_domain[domain])

    rankings = frozen_rankings(out, mq_dir, args.whole_pool)
    scores, header = _load_scores(args.scores)
    shared = sorted(
        set.intersection(*(set(rankings[name]) for name in RERANK_ARMS)) & set(qrels)
    )
    if not shared:
        raise RuntimeError("no query is present in every arm's rankings and the qrels")

    reranked: dict[str, dict[str, list[str]]] = {}
    for name, per_query in rankings.items():
        reranked[name] = {
            t: rerank_order(per_query[t], scores.get(t, {})) for t in shared
        }

    metrics = (DECISION_METRIC, *SECONDARY_METRICS, "R@100")
    summaries = []
    any_pool_limited = False
    for name in RERANK_ARMS:
        row: dict[str, Any] = {"arm": name, "queries": len(shared)}
        for metric in metrics:
            fn = metric_fn(metric)
            row[f"raw_{metric}"] = sum(
                fn(rankings[name][t], qrels[t]) for t in shared) / len(shared)
            row[f"reranked_{metric}"] = sum(
                fn(reranked[name][t], qrels[t]) for t in shared) / len(shared)
        # 🔑 R@100 invariance is a PROPERTY OF THE DESIGN, not something an R@100 comparison can
        # evidence. At depth <= EVAL_K, `rerank_order` returns a permutation and `recall_at` takes
        # `set(ranked[:EVAL_K])`, so raw and reranked R@100 are bit-identical NO MATTER WHAT the
        # scores say. Two successive versions of this code tried to infer the property from that
        # comparison; both were unsatisfiable, i.e. gates that could not fire, which is the exact
        # thing this module argues against. So the flag is now read off the depth, which is what
        # actually carries the claim, and the number is reported beside it.
        deepest = max(len(rankings[name][t]) for t in shared)
        row["max_ranking_length"] = deepest
        row["r@100_invariant"] = deepest <= EVAL_K
        row["r@100_invariance_basis"] = (
            "guaranteed: ranking is no deeper than the cutoff" if deepest <= EVAL_K
            else "NOT guaranteed: ranking is deeper than the cutoff, so reranking CAN move "
                 "documents across it. Whether it did is the raw vs reranked R@100 beside this."
        )
        if deepest > EVAL_K:
            any_pool_limited = True

        # The deterministic gate the R@100 comparison was standing in for: did the scores actually
        # reorder anything? Unlike "R@100 moved", this cannot be satisfied by luck and cannot fire
        # on legitimate data where a real reordering happens not to cross the cutoff.
        reordered = sum(1 for t in shared if reranked[name][t] != rankings[name][t])
        row["queries_reordered"] = reordered
        row["queries_deeper_than_cutoff"] = sum(
            1 for t in shared if len(rankings[name][t]) > EVAL_K)
        row["r@100_delta"] = row["reranked_R@100"] - row["raw_R@100"]
        if reordered == 0:
            raise RuntimeError(
                f"arm {name}: reranking changed the order of NONE of {len(shared)} queries. The "
                f"scores file does not correspond to these rankings, or every score is equal. "
                f"Publishing this would report the raw ordering as a reranked result."
            )
        summaries.append(row)
        print(json.dumps({"event": "arm_reranked",
                          **{k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in row.items()}}), flush=True)

    rng = random.Random(20260806)
    results: dict[str, dict[str, Any]] = {}
    details: dict[str, dict[str, Any]] = {}
    for cid, treatment, control, question in RERANK_CONTRASTS:
        fn = metric_fn(DECISION_METRIC)
        results[cid] = paired_stats(
            [fn(reranked[treatment][t], qrels[t]) - fn(reranked[control][t], qrels[t])
             for t in shared], rng)
        secondary = {}
        for metric in SECONDARY_METRICS:
            sfn = metric_fn(metric)
            secondary[metric] = paired_stats(
                [sfn(reranked[treatment][t], qrels[t]) - sfn(reranked[control][t], qrels[t])
                 for t in shared], rng)
        details[cid] = {"treatment": treatment, "control": control, "question": question,
                        "metric": DECISION_METRIC, "secondary": secondary}
    holm(results)
    for cid, *_ in RERANK_CONTRASTS:
        print(json.dumps({"event": "contrast", "id": cid,
                          **{k: v for k, v in details[cid].items() if k != "secondary"},
                          **results[cid]}), flush=True)
        for metric, stats in details[cid]["secondary"].items():
            print(json.dumps({"event": "secondary", "id": cid, "metric": metric,
                              "mean_delta": stats["mean_delta"], "ci_low": stats["ci_low"],
                              "ci_high": stats["ci_high"],
                              "ci_excludes_zero": stats["ci_excludes_zero"]}), flush=True)

    c1 = results["C1"]
    established = c1["ci_excludes_zero"] and c1["holm_significant"]
    if established and c1["mean_delta"] >= CONVERSION_BAR:
        verdict = "MATERIALLY_CONVERTS"
    elif established and c1["mean_delta"] > 0:
        verdict = "CONVERTS_BUT_BELOW_BAR"
    elif established:
        verdict = "REVERSES"
    else:
        verdict = "DOES_NOT_CONVERT"

    decision = {
        "event": "decision", "metric": DECISION_METRIC, "conversion_bar": CONVERSION_BAR,
        "design": (
            "WHOLE POOL: every arm reranked over its full fused pool. SECONDARY, and CONFOUNDED "
            "with pool width (medians differ 1.9x). Not in the Holm family."
            if args.whole_pool else
            "equal width: every arm reranked over its own top-%d" % EVAL_K
        ),
        "whole_pool": bool(args.whole_pool),
        # Which cross-encoder produced this. The runbook scores TWO of them into one directory,
        # so a decision that does not name its scores file is indistinguishable from the other's.
        "scores_file": str(args.scores.resolve()),
        "scores_header": header,
        # Derived from the measured numbers above, never hardcoded: under --whole-pool the
        # rankings run deeper than the cutoff and R@100 is NOT invariant.
        "r@100_invariant_for_every_arm": all(r["r@100_invariant"] for r in summaries),
        "some_arm_ranks_deeper_than_the_cutoff": any_pool_limited,
        "verdict": verdict,
        "primary": {"id": "C1", **details["C1"], **c1},
        "arms": summaries,
        "contrasts": {cid: {**details[cid], **results[cid]} for cid, *_ in RERANK_CONTRASTS},
        "at": utc_now(),
    }
    # Derive the name from the DESIGN. `rerank_offload` learned this the same way: two runs
    # writing one filename overwrite each other and the second result looks like the first. Here
    # the loser would be the preregistered primary, replaced by the width-confounded secondary.
    stem = args.scores.stem
    if "__" in stem:
        raise ValueError(
            f"scores file stem {stem!r} contains '__', which this naming scheme uses as its "
            f"separator; rename it so the design suffix stays unambiguous."
        )
    # `__whole_pool`, not `_whole_pool`: with a single underscore, `--scores scores_whole_pool.jsonl`
    # without the flag produces the same filename as `--scores scores.jsonl` WITH it, and the two
    # would overwrite each other while disagreeing about the design.
    suffix = "__whole_pool" if args.whole_pool else ""
    filename = f"rerank_decision__{stem}{suffix}.json"
    target = out / filename
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8")).get("scores_file")
        if previous and previous != str(args.scores.resolve()):
            raise RuntimeError(
                f"{target} already holds a decision produced from {previous}, and this run used "
                f"{args.scores.resolve()}. Same basename, different file: overwriting would "
                f"destroy a result whose replacement records an identical-looking provenance."
            )
    target.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in decision.items() if k not in ("contrasts", "arms")},
                     indent=2), flush=True)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Require the offloaded ordering to match a live `CrossEncoderReranker` where metrics cut."""
    from recall.rerank import CrossEncoderReranker
    from recall.types import Chunk, ScoredChunk

    from benchmarks.mtrag.rerank_offload import ORDER_EXACT_K, SCORE_TOLERANCE

    if args.sample < 1:
        raise ValueError(
            f"--sample must be >= 1, got {args.sample}. A zero sample prints verdict MATCH having "
            f"compared nothing, and this verdict is the gate on believing anything the rental "
            f"produced."
        )
    mq_dir = args.mq_dir.resolve()
    out = args.output_dir.resolve()
    scores, _header = _load_scores(args.scores)
    docs: dict[str, str] = {}
    with (out / "docs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                docs[str(row["doc_id"])] = str(row["text"])
    queries = load_queries(mq_dir, "last")
    all_rankings = frozen_rankings(out, mq_dir, args.whole_pool)
    probe = "mq_nested3"
    if probe not in all_rankings:
        raise RuntimeError(f"{probe} is absent from the rankings; nothing to validate against")
    rankings = {t: r for t, r in all_rankings[probe].items() if r}
    if not rankings:
        raise RuntimeError(f"{probe} has no non-empty ranking to validate")
    sample_ids = sorted(rankings)[: args.sample]
    needed = {c for t in sample_ids for c in rankings[t]}
    if not needed <= set(docs):
        raise RuntimeError(
            f"{len(needed - set(docs))} candidates have no text in {out / 'docs.jsonl'} "
            f"(e.g. {sorted(needed - set(docs))[:3]}). The payload and the --whole-pool flag "
            f"disagree: a whole-pool validation needs a payload built with "
            f"`pairs --include-whole-pool`."
        )
    unqueried = [t for t in sample_ids if t not in queries]
    if unqueried:
        raise RuntimeError(
            f"{len(unqueried)} sampled tasks have no query text in {mq_dir} (e.g. "
            f"{unqueried[:3]}); --mq-dir does not match the frozen rankings. Checked before the "
            f"cross-encoder is loaded, because on rented hardware that load is paid minutes."
        )
    unscored = {c for t in sample_ids for c in rankings[t] if c not in scores.get(t, {})}
    if unscored:
        raise RuntimeError(
            f"{len(unscored)} candidates have no score (e.g. {sorted(unscored)[:3]}); the scores "
            f"file does not cover the rankings being validated."
        )

    # The gate must instantiate the reranker the SCORES came from. Validating BGE scores against a
    # MiniLM ordering fails for a reason that has nothing to do with the offload, and the same
    # GPU run produces both files.
    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    if args.revision:
        kwargs["revision"] = args.revision
    reranker = CrossEncoderReranker(**kwargs)
    failures, ties = [], []
    worst = 0.0
    for task_id in sample_ids:
        candidates = rankings[task_id]
        hits = [ScoredChunk(chunk=Chunk(id=c, source="s", text=docs[c], metadata={}), score=0.0)
                for c in candidates]
        offloaded_scores = scores.get(task_id, {})
        local = reranker._model.predict([(queries[task_id], h.chunk.text) for h in hits])
        local_by_id = {c: float(v) for c, v in zip(candidates, local, strict=True)}
        worst = max(worst, max(abs(offloaded_scores[c] - local_by_id[c]) for c in candidates))
        local_order = [h.chunk.id for h in reranker.rerank(queries[task_id], hits)]
        offloaded_order = rerank_order(candidates, offloaded_scores)
        if local_order[:ORDER_EXACT_K] != offloaded_order[:ORDER_EXACT_K]:
            failures.append({"task_id": task_id, "why": f"top-{ORDER_EXACT_K} order differs"})
        elif set(local_order[:EVAL_K]) != set(offloaded_order[:EVAL_K]):
            # What R@100 COUNTS, which order within the cut does not affect. Omitting this was an
            # equal-width assumption: capped, the top-100 set is the whole ranking and cannot
            # differ. Under --whole-pool a near-tie straddling rank 100 changes the set, and
            # `rerank_offload` records a real case of two candidates swapping on a 9.5e-07 gap.
            failures.append({"task_id": task_id, "why": f"top-{EVAL_K} set differs"})
        elif local_order != offloaded_order:
            ties.append(task_id)

    verdict = "MATCH" if not failures and worst < SCORE_TOLERANCE else "MISMATCH"
    print(json.dumps({"event": "validate", "sampled": min(args.sample, len(rankings)),
                      "model": args.model or "recall default (MiniLM pin)",
                      "max_score_delta": worst, "score_tolerance": SCORE_TOLERANCE,
                      "failures": failures[:5], "deep_tie_count": len(ties),
                      "verdict": verdict}, indent=2), flush=True)
    return 0 if verdict == "MATCH" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("pairs", "apply", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--mq-dir", type=Path, required=True, help="the raw run's --output-dir")
        p.add_argument("--output-dir", type=Path, required=True)
        if name == "pairs":
            p.add_argument("--dsn-env-file", type=Path)
            p.add_argument("--table-prefix", default="recall_mtrag_bge_v1")
            p.add_argument("--dim", type=int, default=384)
            p.add_argument("--include-whole-pool", action="store_true",
                           help="also emit pairs for each arm's FULL fused pool, so the "
                                "end-to-end system number comes out of the same scoring run")
        else:
            p.add_argument("--scores", type=Path, required=True)
        if name == "apply":
            p.add_argument("--mtrag-root", type=Path, required=True)
        if name in ("apply", "validate"):
            p.add_argument(
                "--whole-pool", action="store_true",
                help="rerank each arm's FULL fused pool instead of its top-%d. SECONDARY and "
                     "CONFOUNDED with pool width; R@100 is NOT invariant under it." % EVAL_K,
            )
        if name == "validate":
            p.add_argument("--sample", type=int, default=20)
            p.add_argument("--model", default=None,
                           help="reranker the scores came from; omit for RE-call's pinned default")
            p.add_argument("--revision", default=None)
    args = parser.parse_args(argv)
    return {"pairs": cmd_pairs, "apply": cmd_apply, "validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
