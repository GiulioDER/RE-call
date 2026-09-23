"""Measure what replacing the voyageai SDK with `recall._voyage_http` changes.

Pre-registered in `docs/preregistrations/2026-09-23-voyage-http-client.md`. Two modes:

``parity``: calls the live Voyage API with the SDK and with RE-call's client on fixed inputs and
compares the results bit for bit. The SDK is also called twice against itself, which is the
baseline: a provider that is not deterministic cannot be held to bit equality by any client.
Needs ``VOYAGE_API_KEY`` and the ``voyage`` extra. Makes 21 small API calls.

``cost``: runs each arm in a FRESH interpreter, alternating arms, and reports wall time and peak
resident memory from process start to a constructed Voyage embedder. The network is replaced by
a canned probe response, so no API call is made and no key is needed. Arms:

* ``sdk``: what `VoyageEmbedder.__init__` did before P2: ``import voyageai`` and one
  ``voyageai.Client(...).embed(["probe"], ...)``.
* ``http``: ``VoyageEmbedder(...)`` as it is now.

    python scripts/measure_voyage_http_client.py parity > parity.json
    python scripts/measure_voyage_http_client.py cost --samples 5 > cost.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from typing import Any

PARITY_INPUTS = {
    "texts": [
        "def retry_with_backoff(fn, attempts=3): ...",
        "The calibration threshold is fitted per generation and certified before serving.",
        "Erasure rewrites the corpus fingerprint in the same transaction as the deletion.",
        "x",
    ],
    "group": [
        "RE-call serves memory through an MCP server over ssh stdio.",
        "Each generation is immutable once promoted, except under erasure.",
        "The trust layer abstains when no hit clears the calibrated threshold.",
    ],
    "query": "how does erasure change the corpus fingerprint",
    "documents": [
        "Erasure rewrites corpus_fingerprint in the same transaction as the deletion.",
        "The reranker is cached once per process configuration.",
        "Voyage clients are built with a bounded timeout.",
        "HNSW ef_search widens with k on the production store.",
        "Atomic rescue only runs on unscoped searches.",
        "The docs tenant is off by default.",
    ],
}


def _parity() -> dict[str, object]:
    import voyageai

    from recall import _voyage_http

    key = os.environ["VOYAGE_API_KEY"]
    sdk_a = voyageai.Client(api_key=key, max_retries=0, timeout=60.0)
    sdk_b = voyageai.Client(api_key=key, max_retries=0, timeout=60.0)
    ours = _voyage_http.Client(api_key=key, timeout=60.0)
    cases: dict[str, tuple[str, dict[str, Any]]] = {
        "embed code-3 untyped": ("embed", {"texts": PARITY_INPUTS["texts"], "model": "voyage-code-3"}),
        "embed code-3 document": (
            "embed",
            {"texts": PARITY_INPUTS["texts"], "model": "voyage-code-3", "input_type": "document"},
        ),
        "embed code-3 query": (
            "embed",
            {"texts": [PARITY_INPUTS["query"]], "model": "voyage-code-3", "input_type": "query"},
        ),
        "context-4 document group": (
            "contextualized_embed",
            {"inputs": [PARITY_INPUTS["group"]], "model": "voyage-context-4",
             "input_type": "document", "output_dimension": 1024, "output_dtype": "float"},
        ),
        "context-4 query": (
            "contextualized_embed",
            {"inputs": [PARITY_INPUTS["query"]], "model": "voyage-context-4",
             "input_type": "query", "output_dimension": 1024, "output_dtype": "float"},
        ),
        "embed voyage-4 untyped": ("embed", {"texts": PARITY_INPUTS["texts"], "model": "voyage-4"}),
        "rerank-2.5": (
            "rerank",
            {"query": PARITY_INPUTS["query"], "documents": PARITY_INPUTS["documents"],
             "model": "rerank-2.5", "top_k": 6},
        ),
    }

    def flat(result: object) -> list[float]:
        if hasattr(result, "embeddings"):
            return [v for row in result.embeddings for v in row]
        values: list[float] = []
        for item in result.results:  # type: ignore[attr-defined]
            if hasattr(item, "embeddings"):
                values.extend(v for row in item.embeddings for v in row)
            else:
                values.extend((float(item.index), float(item.relevance_score)))
        return values

    report: dict[str, object] = {}
    for name, (method, kwargs) in cases.items():
        a = flat(getattr(sdk_a, method)(**kwargs))
        b = flat(getattr(sdk_b, method)(**kwargs))
        c = flat(getattr(ours, method)(**kwargs))
        report[name] = {
            "values": len(a),
            "sdk_vs_sdk_bit_equal": a == b,
            "sdk_vs_http_bit_equal": len(a) == len(c) and a == c,
            "sdk_vs_sdk_max_abs": max((abs(x - y) for x, y in zip(a, b)), default=0.0),
            "sdk_vs_http_max_abs": max((abs(x - y) for x, y in zip(a, c)), default=0.0),
        }
    return report


_ARM = r'''
import base64, json, struct, sys, time
T0 = time.perf_counter()
import requests
from requests.structures import CaseInsensitiveDict

class _Response:
    status_code = 200
    headers = CaseInsensitiveDict({"Content-Type": "application/json"})
    content = json.dumps({
        "data": [{"embedding": base64.b64encode(struct.pack("<1024f", *([0.0] * 1024))).decode()}],
        "usage": {"total_tokens": 1},
    }).encode()

requests.Session.request = lambda self, *a, **k: _Response()
import recall.embeddings as embeddings
KEY = "pa-CHANGEME-placeholder"
if sys.argv[1] == "sdk":
    import voyageai
    voyageai.Client(api_key=KEY, max_retries=0).embed(["probe"], model="voyage-code-3")
else:
    embeddings.VoyageEmbedder(api_key=KEY, model="voyage-code-3")
elapsed = time.perf_counter() - T0

def peak_mb():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        # argtypes and restype stated: without them ctypes truncates the pseudo-handle to an int
        # and the call fails, which returned 0 MB silently on the first run.
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = PMC(); counters.cb = ctypes.sizeof(PMC)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return counters.PeakWorkingSetSize / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

print(json.dumps({
    "arm": sys.argv[1],
    "seconds": elapsed,
    "peak_mb": peak_mb(),
    "loaded": {m: m in sys.modules for m in ("voyageai", "torch", "transformers")},
}))
'''


def _cost(samples: int) -> dict[str, object]:
    runs: list[dict[str, Any]] = []
    for index in range(samples * 2):
        # Alternate, starting with the other arm on odd samples, so neither arm always runs warm.
        arm = ("sdk", "http")[(index + index // 2) % 2]
        completed = subprocess.run(
            [sys.executable, "-c", _ARM, arm], capture_output=True, text=True, check=True, timeout=600
        )
        runs.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    summary: dict[str, object] = {"runs": runs}
    for arm in ("sdk", "http"):
        mine = [r for r in runs if r["arm"] == arm]
        summary[arm] = {
            "n": len(mine),
            "seconds_median": statistics.median(r["seconds"] for r in mine),
            "seconds_range": [min(r["seconds"] for r in mine), max(r["seconds"] for r in mine)],
            "peak_mb_median": statistics.median(r["peak_mb"] for r in mine),
            "peak_mb_range": [min(r["peak_mb"] for r in mine), max(r["peak_mb"] for r in mine)],
            "loaded_torch_in": sum(bool(r["loaded"]["torch"]) for r in mine),
            "loaded_voyageai_in": sum(bool(r["loaded"]["voyageai"]) for r in mine),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("parity")
    cost = sub.add_parser("cost")
    cost.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    result = _parity() if args.mode == "parity" else _cost(args.samples)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
