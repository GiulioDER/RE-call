"""The LLM half of `run_generation_eval.py`, on a file whose algorithmic half is already done.

The official entrypoint runs the algorithmic judges and then the LLM judges in one process. The
algorithmic half needs a GPU (BertScore over `roberta-large`: ~26 h per file on VPS2's CPU, 4 min on
a 5090) and the LLM half needs a credential that must never go on rented hardware. So they run in
different places, and this script is the second half.

Stage order is copied EXACTLY from `run_generation_eval.py`, including the in-place
`output -> output` threading. Reordering would be silent: `get_idk_conditioned_metrics` rewrites
RB_alg/RL_F/RB_llm using the IDK judgement, so running it before the IDK judge would emit
plausible, wrong, publishable numbers.

WARNING: works on a COPY. The algorithmic output cost GPU time and every stage here writes in place.

Env: AZURE_OPENAI_API_KEY (the OpenRouter key), OPENAI_AZURE_HOST (any non-empty string; unused on
the compat path), OPENAI_COMPAT_BASE_URL, OPENAI_COMPAT_MODEL_PREFIX. The judge model itself stays
the hard-coded `gpt-4o-mini-2024-07-18`, which is what the published table used; changing it would
produce numbers that only LOOK comparable.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/var/tmp/re_call_mtrag_20260803/mt-rag-benchmark/scripts/evaluation")

import warnings

warnings.filterwarnings("ignore")

import judge_wrapper as jw  # noqa: E402


def main(argv: list[str]) -> int:
    src, dst = Path(argv[0]), Path(argv[1])
    key = os.environ["AZURE_OPENAI_API_KEY"]
    host = os.environ.get("OPENAI_AZURE_HOST", "https://unused.invalid")

    if not dst.exists():
        shutil.copy2(src, dst)
        print(f"[copy] {src.name} -> {dst.name}", flush=True)
    else:
        print(f"[resume] {dst.name} already exists, continuing on it", flush=True)

    out = str(dst)
    stages = [
        ("underspecified", lambda: jw.underspecified_eval("openai", out, out)),
        ("idk_judge", lambda: jw.run_idk_judge("openai", "", out, out)),
        ("ragas_faithfulness", lambda: jw.run_ragas_judges_openai(out, out, key, host)),
        ("radbench", lambda: jw.run_radbench_judge("openai", "", out, out)),
        ("idk_conditioned", lambda: jw.get_idk_conditioned_metrics(out, out)),
    ]

    for name, fn in stages:
        started = time.perf_counter()
        print(f"[start] {name}", flush=True)
        fn()
        print(f"[done ] {name} in {time.perf_counter() - started:.1f}s", flush=True)

    print(f"[complete] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
