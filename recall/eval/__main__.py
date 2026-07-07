"""`python -m recall.eval` — run the full ablation matrix and write results/RESULTS.md + charts."""
from __future__ import annotations

import os
from pathlib import Path

from recall.embeddings import Embedder
from recall.eval.harness import results_to_markdown, run_ablations, save_charts

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
        ("OpenAIEmbedder", "OPENAI_API_KEY"),
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
    embedders = _build_embedders()
    print(f"embedders: {[e.name for e in embedders]}")
    results = run_ablations(DEFAULT_DSN, embedders)
    out = Path("results")
    out.mkdir(exist_ok=True)
    md = results_to_markdown(results)
    (out / "RESULTS.md").write_text(
        "# recall — retrieval evaluation\n\n"
        "Reproduce the local (key-free) rows with `make eval` — needs Docker + the local "
        "embedder only. Cloud rows appear when `VOYAGE_API_KEY`/`OPENAI_API_KEY` are set.\n\n"
        + md
        + "\n",
        encoding="utf-8",
    )
    try:
        charts = save_charts(results, out)
        print(f"charts: {[str(c) for c in charts]}")
    except Exception as exc:
        print(f"charts skipped: {exc}")
    print(f"\nwrote {out / 'RESULTS.md'} ({len(results)} ablations)\n")
    print(md)


if __name__ == "__main__":
    main()
