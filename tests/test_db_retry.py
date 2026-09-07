from __future__ import annotations

import pytest

from recall.ops.db_retry import run_read, run_write


class OperationalError(Exception):
    pass


def test_read_retries_transient_failover() -> None:
    calls = 0

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OperationalError("failover")
        return "ok"

    assert run_read(operation, backoff_seconds=0) == "ok"
    assert calls == 3


def test_write_refuses_retry_without_idempotency() -> None:
    with pytest.raises(ValueError):
        run_write(lambda: "never", idempotency_key=None, backoff_seconds=0)
