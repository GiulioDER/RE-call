"""Test helper, run as a separate process: take a lite lane, log what happened. Not a test.

argv: db lane purpose hold_s wait_s log tag coalesce(0|1)
Each event is one JSON line appended to `log`: start, acquired, released, busy, coalesced.
"""
from __future__ import annotations

import json
import sys
import time


def main() -> None:
    db, lane, purpose, hold_s, wait_s, log, tag, coalesce = sys.argv[1:9]
    from recall.lite.writer_lock import LiteWriterBusy, LiteWriterCoalesced, lite_lane

    def emit(event: str) -> None:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"tag": tag, "event": event, "t": time.time()}) + "\n")

    emit("start")
    try:
        with lite_lane(db, lane=lane, purpose=purpose, wait_s=float(wait_s), coalesce=coalesce == "1"):  # type: ignore[arg-type]
            emit("acquired")
            time.sleep(float(hold_s))
            emit("released")
    except LiteWriterCoalesced:
        emit("coalesced")
    except LiteWriterBusy:
        emit("busy")


if __name__ == "__main__":
    main()
