"""Test helper, run as a separate process: take a lite lane, log what happened. Not a test.

argv: db lane purpose hold_s wait_s log tag coalesce(0|1) [release_file]
The holder keeps the lane for hold_s, or until release_file exists when one is given: a fresh
process takes over a second to start on Windows, so a fixed hold can end before a waiter queues.
LITE_LANE_EQUAL_POLL=1 gives every waiter the same poll interval (see W2).
Each event is one JSON line appended to `log`: start, acquired, released, busy, coalesced.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    db, lane, purpose, hold_s, wait_s, log, tag, coalesce = sys.argv[1:9]
    release = Path(sys.argv[9]) if len(sys.argv) > 9 else None
    from recall.lite.writer_lock import LiteWriterBusy, LiteWriterCoalesced, lite_lane

    if os.environ.get("LITE_LANE_EQUAL_POLL") == "1":
        # Test-only: every waiter polls at one rate, so only the queue's head check can keep
        # arrival order (with the head polling faster, it would win most races anyway).
        import recall.lite.writer_lock as wl

        wl.POLL_HEAD_S = wl.POLL_START_S = wl.POLL_MAX_S = 0.02

    def emit(event: str) -> None:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"tag": tag, "event": event, "t": time.time()}) + "\n")

    emit("start")
    try:
        with lite_lane(db, lane=lane, purpose=purpose, wait_s=float(wait_s), coalesce=coalesce == "1"):  # type: ignore[arg-type]
            emit("acquired")
            deadline = time.monotonic() + float(hold_s)
            while time.monotonic() < deadline and not (release is not None and release.exists()):
                time.sleep(0.02)
            emit("released")
    except LiteWriterCoalesced:
        emit("coalesced")
    except LiteWriterBusy:
        emit("busy")


if __name__ == "__main__":
    main()
