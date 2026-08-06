"""Derive the committable artifact for the Mem0 head-to-head (FINDINGS §9d).

`results/ARTIFACTS.md` maps every published figure to the file that produced it. §9d was the
exception: grep that map for `mem0`, `9d` or `mcnemar` and there are no hits, and no path
under `results/` contains `mem0`. The five-row table, the loudest claim in the README, was
the only one a reader could not check. This closes that.

Prior work (searched: docs_search(source_type="memory", "LOCOMO head-to-head Mem0 paired
McNemar artifact provenance committed results"), no gap warning):

  * [[project-recall-memory-benchmark-and-voyage-2026-07-24]] -- the run itself, and the
    fact that `benchmarks/analyze.py` is where the paired analysis lives. So this module
    **delegates to `analyze.paired_mcnemar`** rather than carrying a second implementation
    of the same test. A first draft of this file re-implemented it, which would have been
    a third McNemar in the tree and, worse, a different one: analyze.py switches to
    chi-square with a continuity correction above 25 discordant pairs, and every row here
    has ~440-520, so an independently written exact binomial silently answers a slightly
    different question. It disagreed with the published table on row 4.
  * [[project-recall-mem0-locomo-article-adversarial-review-2026-07-25]] -- the desk review
    that recomputed these numbers from the raw dumps independently of analyze.py. That
    independence was deliberate there and is not wanted here: the artifact's job is to be
    the same computation as the published one, pinned.

Why derived rather than raw: the runs are ~123 MB of generated answers and retrieved
contexts, which does not belong in git. What the claim actually rests on is small:

  * `outcomes/<row>.jsonl` -- one line per paired question, `{q, recall, mem0}`, which is
    enough for anyone to recompute the paired test without the raw runs or an API key;
  * `paired_accuracy.json` -- aggregates, the full 2x2, the p-value and the Holm
    adjustment, each recomputed here rather than copied, plus the sha256 of the two raw
    runs each row came from.

Verification is unconditional and BLOCKING: every derived figure must reproduce §9d at
the precision §9d publishes, and nothing is written if one does not. It is not behind a
flag, because the run that most needs checking is the one nobody remembered to flag.

    python benchmarks/h2h_artifact.py --runs /path/to/raw/results --out results/head_to_head
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from benchmarks.analyze import paired_mcnemar
from benchmarks.claim_gate import matches

OutcomeMap = dict[str, bool]


@dataclass(frozen=True)
class Row:
    """One published §9d row: two runs that differ only in the memory layer."""

    row_id: str
    generator: str
    budget: str
    judge: str
    recall_run: str
    mem0_run: str
    #: The figures as published, **as strings**, to the digits published.
    #:
    #: Strings because the digits ARE the claim: `str(0.370)` is `"0.37"`, so holding these
    #: as floats silently drops a decimal place and checks the cell at lower precision than
    #: it was published at. That is not hypothetical -- it let `0.370` pass --verify while
    #: the value was 0.36948, which rounds to 0.369.
    published_recall: str
    published_mem0: str
    published_p: str


#: The five rows of the §9d table, in table order. Filenames are the raw runs.
PUBLISHED_ROWS: tuple[Row, ...] = (
    Row(
        "gpt4o-mini_item_judge-mini", "gpt-4o-mini", "item (k=10/10)", "gpt-4o-mini",
        "recall_openai-gpt-4o-mini_10conv_20260724T110019Z.json",
        "mem0_openai-gpt-4o-mini_10conv_20260724T120442Z.json",
        "0.416", "0.378", "0.0059",
    ),
    Row(
        "gpt4o-mini_item_judge-4o", "gpt-4o-mini", "item (k=10/10)", "gpt-4o",
        "recall_10conv_rejudged_gpt4o.json",
        "mem0_10conv_rejudged_gpt4o.json",
        "0.466", "0.412", "0.00018",
    ),
    Row(
        "gpt4o-mini_token_judge-mini", "gpt-4o-mini", "token (k=10/20)", "gpt-4o-mini",
        "recall_openai-gpt-4o-mini_10conv_20260724T110019Z.json",
        "mem0_openai-gpt-4o-mini_10conv_20260724T151540Z.json",
        "0.416", "0.369", "0.00077",
    ),
    Row(
        "gpt4o-mini_token_judge-4o", "gpt-4o-mini", "token (k=10/20)", "gpt-4o",
        "recall_10conv_rejudged_gpt4o.json",
        "mem0_k20_rejudged_gpt4o.json",
        # §9d published 0.00018 for this row. The value is 1.7484e-4, which rounds to
        # 0.00017; 0.00018 is the row ABOVE's p (1.8261e-4), evidently copied down. Found
        # by this script's --verify on the first run it was ever able to make, which is
        # the argument for the artifact existing: the table had nothing to be checked
        # against for two weeks. FINDINGS.md is corrected in the same commit. Nothing
        # downstream moves -- both values are orders below any threshold, and the Holm
        # maximum is set by a different row.
        "0.466", "0.411", "0.00017",
    ),
    Row(
        "gpt4o_token_judge-4o", "gpt-4o", "token (k=10/20)", "gpt-4o",
        "recall_openai-gpt-4o_10conv_20260725T172349Z.json",
        "mem0_openai-gpt-4o_10conv_20260725T172547Z.json",
        "0.484", "0.444", "0.0065",
    ),
)

#: The "Mem0 as-shipped" paragraph. RE-call was run TWICE here, 25 seconds apart, with
#: byte-identical configs, and the replicates disagree by 1.04 accuracy points. The README
#: quotes `0.42`, which is the higher one. Both are recorded: the published margin
#: "+0.046 to +0.056" IS these two replicates against the single Mem0 run, so the range was
#: honest, but a reader shown only the point estimate cannot tell a replicate existed.
AS_SHIPPED_RECALL_REPLICATES = (
    "recall_openai-gpt-4o-mini_10conv_20260725T214534Z.json",
    "recall_openai-gpt-4o-mini_10conv_20260725T214559Z.json",
)
AS_SHIPPED_MEM0 = "mem0_openai-gpt-4o-mini_10conv_20260726T083247Z.json"


def load_run(path: Path) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """The raw outcome list (handed to analyze.py untouched) plus a config summary.

    The list is NOT pre-filtered: `paired_mcnemar` selects answerable questions itself and
    refuses arms whose adversarial labelling disagrees. Filtering here would defeat that
    check, which is the check that catches two runs scored against different question sets.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = {
        "arm": payload["arm"],
        "generator": payload["model"],
        "judge": payload.get("rejudge", {}).get("judge_model") or payload["model"],
        "k": payload["config"]["k"],
        "conversations": payload["conversations"],
        "embedder": _embedder_of(payload),
        "aggregate_as_recorded": payload["aggregate"]["answerable_accuracy"],
    }
    return payload["outcomes"], config


def answerable(outcomes: Sequence[Mapping[str, Any]]) -> OutcomeMap:
    """`{question_id: correct}` over answerable questions, for the outcome vectors."""
    return {
        str(o["question_id"]): bool(o["correct"])
        for o in outcomes
        if not o["is_adversarial"]
    }


def _embedder_of(payload: Mapping[str, Any]) -> str | None:
    """The embedder each arm used -- the variable the as-shipped comparison controls.

    The arms record it in different shapes: RE-call under `system.embedder.name`, Mem0
    under `system.embedder.config.model`, because each records what its own library was
    handed. An earlier version PREFERRED `name`, which is a provider (`fastembed`) where
    the other arm's field is a model (`BAAI/bge-small-en-v1.5`) -- so a field called
    `embedder` held two different kinds of thing and a reader could not check from the
    artifact that the rows controlled the embedder at all. Both are kept, joined, so the
    comparison is legible.

    Any shape that is not the expected nesting returns None. An unreadable config should
    report the documented "unlabelled" outcome, not abort the build with a traceback about
    string indices: `'name' in embedder` is a SUBSTRING test when `embedder` is a str, so
    the old form could pass that test and then subscript a string with a string key.
    """
    config = payload.get("config")
    system = config.get("system") if isinstance(config, Mapping) else None
    embedder = system.get("embedder") if isinstance(system, Mapping) else None
    if not isinstance(embedder, Mapping):
        return None
    # THREE places, because the two arms nest it differently and RE-call nests it
    # differently again from what a first fix assumed:
    #   RE-call : {"name": "fastembed", "model": "BAAI/bge-small-en-v1.5"}   <- sibling key
    #   Mem0    : {"provider": "huggingface", "config": {"model": "BAAI/..."}}
    # Reading only `config.model` returned "fastembed" for RE-call against a full model id
    # for Mem0, i.e. the asymmetry this function exists to remove, just one level down.
    inner = embedder.get("config")
    model = embedder.get("model")
    if model is None and isinstance(inner, Mapping):
        model = inner.get("model")
    name = embedder.get("name") or embedder.get("provider")
    parts = [str(x) for x in (name, model) if x is not None]
    return ":".join(dict.fromkeys(parts)) or None


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input's order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def _rate(outcomes: OutcomeMap) -> float:
    return sum(outcomes.values()) / len(outcomes)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _agrees(derived: float, published: str) -> bool:
    """Does `derived` round to `published` at the digits `published` carries?

    Delegated to `claim_gate.matches`, which is the rule the gated documents are already
    checked under, rather than a second rounding convention living here. It rounds
    half-to-even, and a first draft of this file used half-up: on this table the two
    disagree on exactly one cell, which is enough to report a mismatch that belongs to the
    checker rather than to the number.
    """
    return matches(published, derived)


def build(runs: Path) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Derive the artifact and its outcome vectors. Writes NOTHING.

    Separated from writing because the docstring at the top of this module promises the
    build "fails instead of quietly emitting different numbers under the same filename",
    and an earlier version did not keep that promise: it wrote every vector inside the row
    loop and `paired_accuracy.json` immediately after, then verified. A non-reproducing run
    had already overwritten the committed artifact and signalled only through an exit code,
    and an exception mid-loop left fresh vectors beside a stale summary. A guard that runs
    after the damage is a report, not a guard.
    """
    rows_out: list[dict[str, Any]] = []
    vectors: dict[str, list[str]] = {}
    raw_p: list[float] = []
    for row in PUBLISHED_ROWS:
        recall_outcomes, recall_cfg = load_run(runs / row.recall_run)
        mem0_outcomes, mem0_cfg = load_run(runs / row.mem0_run)
        # Raises if the arms did not answer the same questions, or labelled adversarial
        # differently. A paired claim over an implicit intersection is not a paired claim.
        test = paired_mcnemar(recall_outcomes, mem0_outcomes)
        raw_p.append(test["p_value"])
        recall_map, mem0_map = answerable(recall_outcomes), answerable(mem0_outcomes)
        rows_out.append(
            {
                "row_id": row.row_id,
                "generator": row.generator,
                "budget": row.budget,
                "judge": row.judge,
                "recall": {"accuracy": _rate(recall_map), "run": row.recall_run,
                           "sha256": _digest(runs / row.recall_run), "config": recall_cfg},
                "mem0": {"accuracy": _rate(mem0_map), "run": row.mem0_run,
                         "sha256": _digest(runs / row.mem0_run), "config": mem0_cfg},
                "mcnemar": dict(test),
                "published": {"recall": row.published_recall, "mem0": row.published_mem0,
                              "p": row.published_p},
            }
        )
        _record(vectors, row.row_id, _outcome_vector(recall_map, mem0_map))

    for row_out, adjusted in zip(rows_out, holm(raw_p)):
        row_out["mcnemar"]["p_holm_adjusted"] = adjusted

    artifact: dict[str, Any] = {
        "_provenance": {
            "status": "current",
            "backs": ["results/FINDINGS.md §9d", "README.md head-to-head bullet"],
            "generated_by": "benchmarks/h2h_artifact.py",
            "raw_runs": "NOT in this repo (~123 MB of generated answers and contexts). Each "
                        "row carries the sha256 of the two runs it was derived from.",
            "note": "Derived. Every figure is recomputed from the raw runs through "
                    "benchmarks/analyze.py -- the same code that produced the published "
                    "table -- and the build refuses to write unless the two agree at "
                    "published precision.",
        },
        "benchmark": "LOCOMO",
        "metric": "LLM-judged answer accuracy, answerable questions only",
        "protocol": "Paired. Identical question set, generator and judge across arms; the "
                    "memory layer is the only variable. Adversarial questions are excluded "
                    "by analyze.paired_mcnemar -- the abstention contest is §9b, a "
                    "different claim.",
        "test": "benchmarks.analyze.paired_mcnemar; chi-square with continuity correction "
                "at these discordant counts (all >> its exact-binomial threshold of 25), "
                "then Holm-Bonferroni across the five rows",
        # Keyed by row_id, not a list: `claim_gate.lookup` walks dotted paths through
        # dicts only, so a list here would leave every cell of the table unciteable and the
        # numbers unmarkable -- which is the condition this whole artifact exists to end.
        # JSON objects preserve insertion order, so table order survives.
        "rows": {row["row_id"]: row for row in rows_out},
        # Scalar mirrors of quantities the prose quotes. `claim_gate.lookup` cannot index a
        # list, so a figure that lives only inside `margin_range` is unciteable, and an
        # unciteable figure is one the gate forces into `citation-pending` -- i.e. into the
        # bucket for numbers that have no artifact, when this one does.
        "margin_min": min(r["recall"]["accuracy"] - r["mem0"]["accuracy"] for r in rows_out),
        "margin_max": max(r["recall"]["accuracy"] - r["mem0"]["accuracy"] for r in rows_out),
        "as_shipped_embedder": _as_shipped(runs, vectors),
    }
    # Declared last: `_as_shipped` adds the replicate vectors, so anything written
    # earlier would list only the table rows.
    artifact["outcome_vectors"] = sorted(vectors)
    return artifact, vectors


def write_artifact(out: Path, artifact: Mapping[str, Any], vectors: Mapping[str, list[str]]) -> None:
    """Write the verified artifact. Called only after `verify_against_published` is clean.

    Explicit `newline="\n"` on BOTH files. `Path.write_text` uses the platform default, so
    on Windows the summary was written CRLF while the vectors beside it were LF -- and the
    sibling `scripts/generate_claims_baseline.py` documents that exact invariant for its own
    artifact, because a working copy that does not match its own git blob byte for byte
    breaks every sha256 taken over it.
    """
    outcomes = out / "outcomes"
    outcomes.mkdir(parents=True, exist_ok=True)
    for key, lines in vectors.items():
        with (outcomes / f"{key}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            fh.write("".join(line + "\n" for line in lines))
    # The directory is a FUNCTION of the build, not an accumulation of every build that ever
    # ran. Renaming a key used to leave its predecessor behind, backing nothing: the first
    # commit of this artifact shipped nine vectors for seven declared rows, because an
    # earlier key derivation had produced two of them under different names. A stale vector
    # is worse than a missing one -- it reads as evidence, and nothing distinguishes it from
    # the live file beside it.
    for stale in sorted(set(outcomes.glob("*.jsonl")) - {outcomes / f"{k}.jsonl" for k in vectors}):
        stale.unlink()
    with (out / "paired_accuracy.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(artifact, indent=2) + "\n")


def _outcome_vector(recall_map: OutcomeMap, mem0_map: OutcomeMap) -> list[str]:
    """One line per paired question, so the test is recomputable without the raw runs.

    This is the point of the artifact. A p-value nobody else can recompute is a number the
    reader has to take on trust, which is the thing this repository keeps promising not to
    ask for.

    Returns the lines rather than writing them: nothing may reach `results/` until the
    whole build has been verified against the published table.
    """
    return [
        json.dumps(
            {"q": question_id,
             "recall": int(recall_map[question_id]),
             "mem0": int(mem0_map[question_id])},
            separators=(",", ":"),
        )
        for question_id in sorted(set(recall_map) & set(mem0_map))
    ]


def _record(vectors: dict[str, list[str]], key: str, lines: list[str]) -> None:
    """Add a vector under `key`, refusing to overwrite one already recorded this build.

    The as-shipped section holds two replicates of ONE configuration, and losing one turns
    a measured spread back into a point estimate -- the exact defect this artifact exists
    to document. A silent `open("w")` on a colliding key is how that loss happens, so the
    collision is an error rather than a last-writer-wins.
    """
    if key in vectors:
        raise ValueError(
            f"two outcome vectors claim the key {key!r}; one would overwrite the other. "
            f"Give them distinct keys rather than letting a build drop a replicate."
        )
    vectors[key] = lines


def _as_shipped(runs: Path, vectors: dict[str, list[str]]) -> dict[str, Any]:
    """The `text-embedding-3-small` paragraph, with BOTH RE-call replicates."""
    mem0_outcomes, mem0_cfg = load_run(runs / AS_SHIPPED_MEM0)
    mem0_map = answerable(mem0_outcomes)
    replicates: list[dict[str, Any]] = []
    for name in AS_SHIPPED_RECALL_REPLICATES:
        outcomes, cfg = load_run(runs / name)
        outcome_map = answerable(outcomes)
        replicates.append(
            {
                "run": name,
                "sha256": _digest(runs / name),
                "accuracy": _rate(outcome_map),
                "margin_over_mem0": _rate(outcome_map) - _rate(mem0_map),
                "mcnemar": dict(paired_mcnemar(outcomes, mem0_outcomes)),
                "config": cfg,
            }
        )
        # Keyed by the run FILENAME's stem, which is unique per run by construction. An
        # earlier version used "whatever follows the last underscore", which happened to be
        # the timestamp for these two filenames and is not the timestamp in general -- two
        # runs sharing a trailing segment would have collided, and the collision would have
        # been silent. `_record` now refuses a repeat outright.
        _record(vectors, f"as_shipped__{Path(name).stem}", _outcome_vector(outcome_map, mem0_map))
    margins = sorted(float(r["margin_over_mem0"]) for r in replicates)
    return {
        "embedder": "openai/text-embedding-3-small (Mem0's documented default), both arms",
        "mem0": {"run": AS_SHIPPED_MEM0, "sha256": _digest(runs / AS_SHIPPED_MEM0),
                 "accuracy": _rate(mem0_map), "config": mem0_cfg},
        "recall_replicates": replicates,
        "margin_range": [margins[0], margins[-1]],
        "margin_min": margins[0],
        "margin_max": margins[-1],
        "replicate_low_accuracy": min(float(r["accuracy"]) for r in replicates),
        "replicate_high_accuracy": max(float(r["accuracy"]) for r in replicates),
        "replicate_spread": margins[-1] - margins[0],
        #: The same spread in accuracy POINTS, because that is the unit the prose uses and a
        #: figure quoted in one unit cannot be gated against an artifact holding another.
        "replicate_spread_points": 100 * (
            max(float(r["accuracy"]) for r in replicates)
            - min(float(r["accuracy"]) for r in replicates)
        ),
        "caveat": "Temperature 0 does not make this deterministic: generator and judge run "
                  "through a hosted API. These two replicates share a byte-identical config, "
                  "ran 25 seconds apart, and differ by 1.04 accuracy points -- about a "
                  "quarter of the reported margin. Every single-run figure in this artifact "
                  "carries at least that much run-to-run noise, and none of the five table "
                  "rows was replicated at all.",
    }


def verify_against_published(artifact: Mapping[str, Any]) -> list[str]:
    """Every derived figure must reproduce the published one at its published precision."""
    problems = []
    for row in artifact["rows"].values():
        published = row["published"]
        for arm in ("recall", "mem0"):
            if not _agrees(row[arm]["accuracy"], published[arm]):
                problems.append(
                    f"{row['row_id']}: {arm} derived {row[arm]['accuracy']:.4f} does not "
                    f"round to published {published[arm]}"
                )
        if not _agrees(row["mcnemar"]["p_value"], published["p"]):
            problems.append(
                f"{row['row_id']}: p derived {row['mcnemar']['p_value']:.6f} does not round "
                f"to published {published['p']}"
            )
    return problems


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the §9d head-to-head artifact. Verification is not optional: "
                    "nothing is written unless every derived figure reproduces the published "
                    "table."
    )
    parser.add_argument("--runs", type=Path, required=True,
                        help="directory holding the raw benchmark run JSONs")
    parser.add_argument("--out", type=Path, default=Path("results/head_to_head"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    artifact, vectors = build(args.runs)

    for row in artifact["rows"].values():
        test = row["mcnemar"]
        print(f"{row['row_id']:30s} RE-call {row['recall']['accuracy']:.4f}  "
              f"Mem0 {row['mem0']['accuracy']:.4f}  "
              f"p={test['p_value']:.6g}  Holm={test['p_holm_adjusted']:.6g}")
    shipped = artifact["as_shipped_embedder"]
    print(f"\nas-shipped: Mem0 {shipped['mem0']['accuracy']:.4f}; RE-call replicates "
          + ", ".join(f"{r['accuracy']:.4f}" for r in shipped["recall_replicates"])
          + f"  (spread {shipped['replicate_spread']:.4f})")

    problems = verify_against_published(artifact)
    if problems:
        print("\nDOES NOT REPRODUCE §9d - nothing written:")
        for problem in problems:
            print(f"  {problem}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    write_artifact(args.out, artifact, vectors)
    print(f"\nAll {len(artifact['rows'])} §9d rows reproduce at their published precision; "
          f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
