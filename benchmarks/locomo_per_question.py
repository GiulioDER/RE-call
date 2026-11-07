"""Run `recall.eval.locomo` and also write its per-question records, for a paired comparison.

::

    LOCOMO_PER_QUESTION_OUT=q.json python -m benchmarks.locomo_per_question <locomo.py args>

`recall.eval.locomo.run` drops each conversation's `questions` list from the report it returns, so
two runs can only be compared by their aggregate rates. This wraps `run_conversation` to keep a
copy of every result, in conversation order, and writes them as a JSON list of
``{"conversation": <index>, "questions": [...]}``. The scoring path is untouched.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import recall.eval.locomo as locomo


def main() -> int:
    out = Path(os.environ["LOCOMO_PER_QUESTION_OUT"])
    records: list[dict[str, Any]] = []
    original = locomo.run_conversation

    def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        records.append({"conversation": len(records), "questions": result["questions"]})
        return result

    locomo.run_conversation = capture
    status = locomo.main()
    out.write_text(json.dumps(records), encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
