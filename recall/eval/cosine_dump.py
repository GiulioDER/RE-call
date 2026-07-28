"""Retain the top-cosine distributions the abstention results rest on, and chart them.

Why this exists
---------------
`RESULTS.md` §7b and `FINDINGS.md` §9b/§10c report abstention as *rates* — 0.00 on LOCOMO's
adversarial 446, 1.00 on a bounded corpus — and the artifacts behind them
(`results/locomo/postfix_abstention.json`) keep only those rates. The claim underneath them is
about the *distributions*: an on-topic adversarial scores as high as an answerable question, so
no threshold separates them, while a genuinely off-topic query falls in its own band.

Nothing in this repo retained the values needed to check that claim directly, which also meant
the obvious figure could not be drawn from a published artifact — only from quantiles, i.e. by
inventing shape. This module regenerates the distributions and retains them.

It measures the same quantity the abstention decision reads: `locomo_abstention._top_cosine`,
the max cosine over the hits returned at `k`, which is exactly what `trust.evaluate` thresholds.
Both corpora go through that one function so the two panels of the chart are comparable.

Three classes, on two corpora:

  LOCOMO           answerable (cat 1-4)  vs  adversarial (cat 5) — the near-miss class, at n=446
  14-doc reference answerable  vs  far-gap unanswerable  vs  near-miss

The reference corpus carries all three at once, so the far-gap/near-miss boundary is visible
without a corpus change confounding it — but its classes are small (14/5/10) and the near-miss
row in particular cannot carry a conclusion on its own; see the interval in the report.

Run the LOCOMO benchmark first — this reuses the `locomo_chunks` tables it indexes, and does not
re-index them::

    python -m recall.eval.locomo      --data locomo10.json
    python -m recall.eval.cosine_dump --data locomo10.json \\
        --out results/cosine/distributions.json --chart-dir results/cosine
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recall.calibration import separability, separability_interval
from recall.embeddings import Embedder
from recall.eval.harness import EVAL_DIR, _throwaway_store
from recall.eval.provenance import model_stack
from recall.eval.locomo import _make_embedder
from recall.eval.locomo_abstention import _partition_questions, _top_cosine
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore


def _reference_bucket(entry: dict[str, Any]) -> str | None:
    """Class for one `queries.json` entry, or None when the entry is not part of this split.

    Mirrors the rule in `recall.eval.calibrate.measure_top_cosines`, which is the single source
    of the sampling rule for this corpus: validity-sensitive (`trust`) entries carry no
    answerable label and are skipped.

    Reading the label with a falsy ``entry.get("answerable")`` instead — the obvious shortcut —
    buckets all six trust entries as unanswerable, giving a far-gap class of 11 against a true 5
    and contaminating the exact class this artifact exists to characterise. It does not raise;
    it returns a plausible, wrong distribution. Pinned by `tests/test_eval_cosine_dump.py`.
    """
    if entry.get("trust"):
        return None
    return "answerable" if entry["answerable"] else "far_gap"


def _summarise(name: str, answerable: list[float], other: list[float]) -> dict[str, Any]:
    """Separability of one class pair, with the interval the certification rule reads."""
    auc = separability(answerable, other)
    row: dict[str, Any] = {"pair": name, "n_answerable": len(answerable), "n_other": len(other)}
    if auc is None:
        return {**row, "separability": None, "ci95": None}
    lo, hi = separability_interval(auc, len(answerable), len(other))
    return {**row, "separability": round(auc, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def dump_reference(dsn: str, embedder: Embedder, k: int) -> dict[str, list[float]]:
    """Top cosines for the 14-doc reference corpus, split answerable / far-gap / near-miss."""
    queries = json.loads((EVAL_DIR / "queries.json").read_text(encoding="utf-8"))
    nearmiss = json.loads((EVAL_DIR / "near_miss.json").read_text(encoding="utf-8"))
    out: dict[str, list[float]] = {"answerable": [], "far_gap": [], "near_miss": []}

    # `_throwaway_store` rather than a local index call: it uuid-names the table, drops it on the
    # way out, and is the one indexing path this package maintains. A second path with a fixed
    # table name would double its corpus on a re-run and depress every score without erroring.
    with _throwaway_store(dsn, embedder, EVAL_DIR / "corpus", "cosdump_") as store:
        retriever = HybridRetriever(store, embedder)
        for entry in queries:
            bucket = _reference_bucket(entry)
            if bucket is None:
                continue
            cosine = _top_cosine(retriever, entry["query"], k)
            if cosine is not None:
                out[bucket].append(cosine)
        for entry in nearmiss:
            cosine = _top_cosine(retriever, entry["query"], k)
            if cosine is not None:
                out["near_miss"].append(cosine)
    return out


def dump_locomo(
    dsn: str,
    data_path: Path,
    embedder: Embedder,
    *,
    k: int,
    answerable_sample: int,
    seed: int,
) -> dict[str, list[float]]:
    """Top cosines for LOCOMO, split answerable (cat 1-4) / adversarial (cat 5).

    Reuses `_partition_questions` and advances one `Random(seed)` across conversations exactly as
    `locomo_abstention.run` does, so the answerable sample is the same one the published rates
    were scored on rather than a fresh draw.
    """
    conversations = json.loads(data_path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    out: dict[str, list[float]] = {"answerable": [], "adversarial": []}

    for i, conv in enumerate(conversations):
        sample_id = conv.get("sample_id") or f"conv{i}"
        answerable, adversarial = _partition_questions(
            conv.get("qa") or [], answerable_sample, rng
        )
        with PgVectorStore(
            dsn, dim=embedder.dim, tenant=f"locomo-{sample_id}", table="locomo_chunks"
        ) as store:
            retriever = HybridRetriever(store, embedder)
            ans = [
                c
                for q in answerable
                if (c := _top_cosine(retriever, q["question"], k)) is not None
            ]
            adv = [
                c
                for q in adversarial
                if (c := _top_cosine(retriever, q["question"], k)) is not None
            ]
        out["answerable"] += ans
        out["adversarial"] += adv
        print(f"  [{i + 1}/{len(conversations)}] {sample_id}: ans={len(ans)} adv={len(adv)}",
              flush=True)
    return out


# dataviz reference palette, light mode, slots 1/2/3 — validated all-pairs (worst CVD dE 9.2,
# worst normal-vision dE 24.0). Slot 3 sits under 3:1 on this surface, so every row is
# direct-labelled rather than identified by colour alone.
_ANSWERABLE = "#2a78d6"
_NEAR_MISS = "#eb6834"
_FAR_GAP = "#1baf7a"
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#dededa"


def save_cosine_overlap_chart(report: dict[str, Any], out_dir: Path) -> Path:
    """Two panels: LOCOMO's superimposed classes, and the reference corpus's three classes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    loc, ref = report["locomo"], report["reference_14doc"]

    def label(pair: str) -> str:
        row = next(r for r in report["separability"] if r["pair"] == pair)
        if row["separability"] is None:
            return "separability n/a"
        lo, hi = row["ci95"]
        return f"separability {row['separability']:.2f}  [{lo:.2f}, {hi:.2f}]"

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.4, 5.0), dpi=150, gridspec_kw={"width_ratios": [1.15, 1]}
    )
    fig.patch.set_facecolor(_SURFACE)

    values = [*loc["answerable"], *loc["adversarial"], *(v for c in ref.values() for v in c)]
    pad = (max(values) - min(values)) * 0.06
    xlim = (min(values) - pad, max(values) + pad)

    bins = np.linspace(*xlim, 34)
    for vals, color, name in (
        (loc["answerable"], _ANSWERABLE, f"answerable  (n={len(loc['answerable'])})"),
        (loc["adversarial"], _NEAR_MISS, f"adversarial  (n={len(loc['adversarial'])})"),
    ):
        weights = np.ones(len(vals)) / len(vals)
        ax1.hist(vals, bins=bins, weights=weights, color=color, alpha=0.55, label=name)
        ax1.hist(vals, bins=bins, weights=weights, histtype="step", color=color, linewidth=2)
    ax1.set_title("LOCOMO — abstention scores 0.00 here", fontsize=13, color=_INK,
                  fontweight="bold", loc="left", pad=12)
    ax1.text(0, 1.015, f"on-topic adversarials · {label('locomo_answerable_vs_adversarial')}",
             transform=ax1.transAxes, fontsize=9.5, color=_INK_2)
    ax1.set_ylabel("share of class", fontsize=10, color=_INK_2)
    ax1.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=_INK_2)

    rng = np.random.default_rng(0)
    rows = (
        ("answerable", ref["answerable"], _ANSWERABLE),
        ("near-miss", ref["near_miss"], _NEAR_MISS),
        ("far-gap\nunanswerable", ref["far_gap"], _FAR_GAP),
    )
    for i, (name, vals, color) in enumerate(rows):
        ax2.scatter(vals, i + rng.uniform(-0.13, 0.13, len(vals)), s=64, color=color,
                    alpha=0.85, edgecolors=_SURFACE, linewidths=1.6, zorder=3)
        ax2.text(xlim[0] + (xlim[1] - xlim[0]) * 0.005, i + 0.30, f"{name}  (n={len(vals)})",
                 fontsize=9.5, color=_INK_2, va="bottom")

    gap_lo, gap_hi = max(ref["far_gap"]), min(ref["answerable"])
    if gap_hi > gap_lo:
        ax2.axvspan(gap_lo, gap_hi, color=_FAR_GAP, alpha=0.12, zorder=0)
        inside = [v for v in ref["near_miss"] if gap_lo <= v <= gap_hi]
        ax2.annotate(
            f"clear air: separates answerable\nfrom far-gap — but {len(inside)} near-misses\n"
            f"land inside it",
            xy=((gap_lo + gap_hi) / 2, -0.30), fontsize=8.5, color=_INK_2, ha="center",
            va="center",
        )
    ax2.set_ylim(-0.75, 2.75)
    ax2.set_yticks([])
    ax2.set_title("14-doc corpus — three classes, one clean split", fontsize=13, color=_INK,
                  fontweight="bold", loc="left", pad=12)
    ax2.text(
        0, 1.015,
        "far gap " + label("reference_answerable_vs_far_gap").replace("separability ", "sep. ")
        + "   ·   near-miss "
        + label("reference_answerable_vs_near_miss").replace("separability ", "sep. "),
        transform=ax2.transAxes, fontsize=9.5, color=_INK_2,
    )

    for ax in (ax1, ax2):
        ax.set_facecolor(_SURFACE)
        ax.set_xlim(*xlim)
        ax.set_xlabel("top cosine of the retrieved hit", fontsize=10, color=_INK_2)
        ax.grid(axis="x", color=_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(_GRID)
        ax.tick_params(colors=_INK_2, labelsize=9.5, length=0)

    fig.suptitle(
        "Abstention is bounded-domain: a threshold separates a far gap, not a near-miss",
        fontsize=14.5, color=_INK, fontweight="bold", x=0.007, ha="left", y=0.985,
    )
    fig.text(0.007, 0.028,
             f"{report['embedder']}, max cosine over hits at k={report['k']} — the quantity the "
             "trust layer thresholds.  Separability = Mann-Whitney AUC "
             "(recall.calibration); a usable gate needs 0.90.",
             fontsize=8.5, color=_INK_2, ha="left")
    fig.text(0.007, 0.004,
             "The exclusion rests on LOCOMO's n: 0.60 with the interval well clear of 0.90. The "
             "14-doc near-miss row is n=10 — its interval reaches 1.00 and establishes nothing "
             "on its own.",
             fontsize=8.5, color=_INK_2, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))

    path = out_dir / "cosine_overlap.png"
    fig.savefig(path, facecolor=_SURFACE, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def run(
    data_path: Path,
    *,
    dsn: str,
    embedder_name: str,
    k: int,
    answerable_sample: int,
    seed: int,
) -> dict[str, Any]:
    started = time.time()
    embedder = _make_embedder(embedder_name)

    print("14-doc reference corpus...", flush=True)
    reference = dump_reference(dsn, embedder, k)
    print("  " + "  ".join(f"{name}={len(v)}" for name, v in reference.items()), flush=True)

    print("LOCOMO...", flush=True)
    locomo = dump_locomo(
        dsn, data_path, embedder, k=k, answerable_sample=answerable_sample, seed=seed
    )

    return {
        # Emitted, not stamped on afterwards: a regenerated artifact must carry its own
        # provenance, or the next `--out` silently produces an unlabelled file. Schema and the
        # index it feeds: `results/ARTIFACTS.md`.
        "_provenance": {
            "generation": "post-#81/#84",
            "status": "current",
            "superseded_by": None,
            "backs": ["RESULTS §12 — cosine distributions"],
            "note": (
                "Sparse leg firing (#81/#82) and the dense scan widened (#84). This is the "
                "configuration the published tables describe."
            ),
        },
        "artifact": "top-cosine distributions behind the abstention results",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedder": embedder_name,
        "k": k,
        "answerable_sample_per_conv": answerable_sample,
        "seed": seed,
        "elapsed_s": round(time.time() - started, 1),
        "stack": model_stack(),
        "quantity": (
            "max cosine over the hits returned at k — exactly what trust.evaluate thresholds, "
            "via locomo_abstention._top_cosine"
        ),
        "separability": [
            _summarise(
                "locomo_answerable_vs_adversarial", locomo["answerable"], locomo["adversarial"]
            ),
            _summarise(
                "reference_answerable_vs_far_gap", reference["answerable"], reference["far_gap"]
            ),
            _summarise(
                "reference_answerable_vs_near_miss",
                reference["answerable"],
                reference["near_miss"],
            ),
        ],
        "locomo": locomo,
        "reference_14doc": reference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump and chart the top-cosine distributions behind the abstention results."
    )
    parser.add_argument("--data", type=Path, required=True, help="path to locomo10.json")
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--embedder", default="fastembed", help="fastembed | voyage | hashing")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--answerable-sample", type=int, default=40,
        help="answerable questions per conversation (matches the published abstention run)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True, help="write the JSON artifact here")
    parser.add_argument(
        "--chart-dir", type=Path, default=None, help="also render the chart into this directory"
    )
    args = parser.parse_args()

    import os

    dsn = args.dsn or os.environ.get(
        "RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall"
    )
    report = run(
        args.data,
        dsn=dsn,
        embedder_name=args.embedder,
        k=args.k,
        answerable_sample=args.answerable_sample,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"-> {args.out}")

    for row in report["separability"]:
        ci = f"[{row['ci95'][0]:.2f}, {row['ci95'][1]:.2f}]" if row["ci95"] else "n/a"
        print(f"  {row['pair']:38s} {row['separability']}  {ci}"
              f"  n={row['n_answerable']}/{row['n_other']}")

    if args.chart_dir is not None:
        print(f"-> {save_cosine_overlap_chart(report, args.chart_dir)}")


if __name__ == "__main__":
    main()
