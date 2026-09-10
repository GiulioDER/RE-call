from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recall.types import Chunk
import recall_mcp.service as service


class Readiness:
    ready = True
    graph_fingerprint = "graph-a"


class GenerationStore:
    tenant = "benchmark"
    generation_id = "generation-a"

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def active_generation_id(self) -> str:
        return self.generation_id

    def graph_readiness(self) -> Readiness:
        return Readiness()

    def iter_chunks(self):
        return iter(self.chunks)

    def supersession_all(self):
        return {}, frozenset(), {}


def nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if args.samples < 20:
        raise SystemExit("--samples must be at least 20")

    chunks = [
        Chunk(
            f"chunk-{index}",
            f"source-{index // 10}.md",
            f"decision {index}: owner and rollout detail",
            {"file": f"source-{index // 10}.md"},
        )
        for index in range(100)
    ]
    store = GenerationStore(chunks)
    cold: list[float] = []
    warm: list[float] = []
    for _ in range(args.samples):
        service._reset_graph_projection_cache()
        started = time.perf_counter_ns()
        service._store_graph(store, include_text=True, policy_fingerprint="policy-a")
        cold.append((time.perf_counter_ns() - started) / 1_000_000)

        started = time.perf_counter_ns()
        service._store_graph(store, include_text=True, policy_fingerprint="policy-a")
        warm.append((time.perf_counter_ns() - started) / 1_000_000)

    print(
        json.dumps(
            {
                "samples": args.samples,
                "cold_ms": {"p50": nearest_rank(cold, 0.50), "p95": nearest_rank(cold, 0.95)},
                "warm_ms": {"p50": nearest_rank(warm, 0.50), "p95": nearest_rank(warm, 0.95)},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
