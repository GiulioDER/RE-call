"""Offline half of the C9 window format check: does the per-track detector sort both corpora?

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-window-format.md

No model and no database. Every LoCoMo session (272 Adds, as the reader-dates collects sent them)
and every Agent Memory Bench coding session (196 Adds, one message per event, as the Coding check
sent them) is classified by ``recall_aml.window_format.looks_like_coding``. For each Add, the raw
window texts ``build_chunks`` writes with ``per_track_windows`` on are compared with the texts of
the renderer the Add's track was measured with on 2026-09-24: timestamp, role and content for
LoCoMo (arm C), content-only for Coding (arm K0). Where the texts are identical, the per-track
arm inherits that arm's measured result.

    python scripts/aml_c9_window_format_check.py --locomo locomo10.json \\
        --amb-root <agent-memory-bench> --out window-format-offline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aml_c7_qualification import load_frozen_corpus  # noqa: E402
from aml_c9_coding_window_check import event_messages  # noqa: E402
from aml_locomo_route_compare import build_corpus  # noqa: E402
from recall_aml.models import AddRequest  # noqa: E402
from recall_aml.service import build_chunks  # noqa: E402
from recall_aml.variants import variant  # noqa: E402
from recall_aml.window_format import looks_like_coding  # noqa: E402

C9 = variant("C9_routed_specialists_grounded_graph_atomic")


def window_texts(request: AddRequest, *, content_only: bool, per_track: bool) -> list[str]:
    """The raw window texts C9's Add would write, before any embedding."""
    return [
        chunk.text
        for chunk in build_chunks(
            request,
            [],
            word_window_size=C9.word_window_size,
            word_window_stride=C9.word_window_stride,
            content_only_windows=content_only,
            stable_window_identity=C9.stable_window_order,
            per_track_windows=per_track,
        )
        if chunk.metadata.get("record_type") == "raw"
    ]


def check(requests: list[AddRequest], *, measured_content_only: bool) -> dict[str, Any]:
    flagged = [looks_like_coding(request.messages) for request in requests]
    identical = [
        window_texts(request, content_only=True, per_track=True)
        == window_texts(request, content_only=measured_content_only, per_track=False)
        for request in requests
    ]
    return {
        "adds": len(requests),
        "flagged_coding": sum(flagged),
        "windows_identical_to_measured_arm": sum(identical),
        "not_identical_session_ids": [
            request.session_id for request, same in zip(requests, identical, strict=True) if not same
        ][:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--amb-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    adds, _ = build_corpus(json.loads(args.locomo.read_bytes()), "windowformat", None)
    locomo = [AddRequest.model_validate(add) for add in adds]
    corpus = load_frozen_corpus(args.amb_root)
    coding = [
        AddRequest.model_validate(
            {
                "request_id": f"windowformat-{position:04d}",
                "messages": event_messages(corpus.corpus_root / relative),
                "user_id": "windowformat",
                "session_id": relative,
            }
        )
        for position, relative in enumerate(sorted(corpus.sessions), start=1)
    ]
    result = {
        "preregistration": "docs/preregistrations/2026-09-25-aml-c9-window-format.md",
        "locomo": check(locomo, measured_content_only=False),
        "coding": check(coding, measured_content_only=True),
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
