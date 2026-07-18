"""`python -m recall.eval` — run the full ablation matrix and write results/RESULTS.md + charts."""
from __future__ import annotations

import os
from pathlib import Path

from recall.embeddings import Embedder
from recall.eval.harness import (
    nearmiss_results_to_markdown,
    results_to_markdown,
    run_ablations,
    run_nearmiss_eval,
    run_trust_eval,
    save_charts,
    save_nearmiss_chart,
    save_trust_chart,
    trust_results_to_markdown,
)

DEFAULT_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")


def _build_embedders() -> list[Embedder]:
    from recall.embeddings import HashingEmbedder

    embedders: list[Embedder] = [HashingEmbedder(dim=64)]
    try:
        from recall.embeddings import FastEmbedEmbedder

        embedders.append(FastEmbedEmbedder())
    except Exception as exc:
        print(f"skip fastembed: {exc}")
    for cls_name, keyenv in [
        ("VoyageEmbedder", "VOYAGE_API_KEY"),
    ]:
        if not os.environ.get(keyenv):
            print(f"skip {cls_name}: no {keyenv} set")
            continue
        try:
            import recall.embeddings as e

            embedders.append(getattr(e, cls_name)())
        except Exception as exc:
            print(f"skip {cls_name}: {exc}")
    return embedders


def main() -> None:
    from recall._env import load_dotenv

    load_dotenv()  # pick up VOYAGE_API_KEY from a gitignored .env if present
    embedders = _build_embedders()
    print(f"embedders: {[e.name for e in embedders]}")
    results = run_ablations(DEFAULT_DSN, embedders)
    trust_results = run_trust_eval(DEFAULT_DSN, embedders)
    nearmiss_results = []
    try:
        from recall.entailment import QnliEntailmentJudge

        nearmiss_results = run_nearmiss_eval(DEFAULT_DSN, embedders, QnliEntailmentJudge())
    except ImportError as exc:
        print(f"skip near-miss stage: {exc}")
    out = Path("results")
    out.mkdir(exist_ok=True)
    md = results_to_markdown(results)
    trust_md = trust_results_to_markdown(trust_results)
    nearmiss_md = nearmiss_results_to_markdown(nearmiss_results) if nearmiss_results else ""
    (out / "RESULTS.md").write_text(
        "# recall — retrieval evaluation\n\n"
        "Reproduce the local (key-free) rows with `make eval` — needs Docker + the local "
        "embedder only. The Voyage cloud row appears when `VOYAGE_API_KEY` is set.\n\n"
        + md
        + "\n\n## Trust layer — superseded/expired memories vs plain search\n\n"
        "STR = superseded-trust rate: how often a stale memory was presented as the answer "
        "on the validity-sensitive queries (lower is better). The final two columns verify "
        "the trust layer does not change ordinary answerable retrieval.\n\n"
        + trust_md
        + (
            "\n\n## Entailment abstention — near-miss queries (arms A/B/C)\n\n"
            "Near-miss = a high-similarity memory that does NOT answer the query — the class a "
            "cosine threshold passes by construction. Arms: `threshold` = calibrated cosine "
            "threshold (status quo), `threshold+entail` = threshold plus the QNLI judge, "
            "`entail-only` = judge alone (ablation). The judge is identical across embedders — "
            "no per-embedder recalibration.\n\n" + nearmiss_md
            if nearmiss_md
            else ""
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        charts = save_charts(results, out)
        charts.append(save_trust_chart(trust_results, out))
        if nearmiss_results:
            charts.append(save_nearmiss_chart(nearmiss_results, out))
        print(f"charts: {[str(c) for c in charts]}")
    except Exception as exc:
        print(f"charts skipped: {exc}")
    print(f"\nwrote {out / 'RESULTS.md'} ({len(results)} ablations)\n")
    print(md)
    print()
    print(trust_md)
    if nearmiss_md:
        print()
        print(nearmiss_md)


if __name__ == "__main__":
    main()
