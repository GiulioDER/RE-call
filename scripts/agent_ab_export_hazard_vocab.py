#!/usr/bin/env python3
"""Write the df<=N hazard vocabulary to a file the write-time hook can read.

    python scripts/agent_ab_export_hazard_vocab.py --dsn <dsn> --out benchmarks/artifacts/agent_ab/hazard-vocab.json

The hook records `vocabulary_would_fire` on every injection and never acts on it, so a gated
variant is an offline re-analysis rather than a second A/B. That only works if the field is
populated: without `RECALL_HOOK_VOCAB` the hook writes `null`, and stage A's first run wrote
`null` 31 times out of 31, which would have made the registration's "BUILD WITH A GATE" cell
unreachable from its own evidence.

`df_max=2` is the trigger the screen selected on the Pareto front (`T2_vocab_df2`): coverage 9 of
10 reachable sessions at a third of the false-trigger rate of the literal alternative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agent_ab_trigger_screen import hazard_vocabulary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--df-max", type=int, default=2,
                        help="a term is in the vocabulary when at most this many memos use it")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    vocabulary = sorted(hazard_vocabulary(args.dsn, args.df_max))
    if not vocabulary:
        # An empty file is indistinguishable from an unset variable at the hook, and both write
        # null. Refusing here is the difference between a wiring bug and a silent one.
        print("REFUSED: the vocabulary came back empty; the hook would record null either way.")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vocabulary, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(vocabulary)} terms at df<={args.df_max} -> {out}")
    print("sample:", ", ".join(vocabulary[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
