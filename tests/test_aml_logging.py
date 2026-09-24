"""The hosted AML journal must carry the structured fields its log calls supply."""

from __future__ import annotations

import io
import json
import logging

import pytest

from recall_aml.__main__ import configure_logging
from recall_aml.compiler import _log_diagnostics


@pytest.fixture
def journal():
    # Production calls configure_logging on a root logger with no handlers. Reproduce that, then
    # hand pytest's own capture handlers back so only the handler under test is added.
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    root.handlers.clear()
    stream = io.StringIO()
    configure_logging(stream)
    ours = list(root.handlers)
    root.handlers[:] = saved_handlers + ours
    yield stream
    for handler in ours:
        root.removeHandler(handler)
    root.setLevel(saved_level)


def test_hosted_journal_lines_carry_extra_fields_once(journal) -> None:
    """Every ``extra`` field must reach the journal line, and the compiler's JSON exactly once.

    Invariant: the production handler renders ``extra`` as JSON after the unchanged prefix. The
    official Multimodal run ``teval_dcc1109c4331c3e3`` logged ``hosted_add_complete`` and
    ``hosted_request_failed`` with no latency, route or error class because of the drop.

    Red proof (mutation): replacing ``ExtraFieldsFormatter`` with ``logging.Formatter`` in
    ``recall_aml.__main__.configure_logging``, which is the pre-fix ``basicConfig`` rendering,
    fails the first equality below with the bare ``INFO:recall_aml:hosted_search_complete`` line.
    Removing the ``record.message.endswith(rendered)`` check in ``formatMessage`` fails the
    ``compiler_provider_usage`` equality with the JSON object printed twice.
    """
    log = logging.getLogger("recall_aml")
    log.info(
        "hosted_search_complete",
        extra={"latency_ms": 12.5, "specialist_route": "context"},
    )
    log.error("hosted_request_failed", extra={"error_class": "TimeoutError"})
    _log_diagnostics("compiler_provider_usage", {"prompt_tokens": 300, "total_tokens": 381})

    lines = journal.getvalue().splitlines()

    assert lines[0] == (
        'INFO:recall_aml:hosted_search_complete {"latency_ms":12.5,"specialist_route":"context"}'
    )
    assert lines[1] == 'ERROR:recall_aml:hosted_request_failed {"error_class":"TimeoutError"}'
    assert lines[2] == (
        'INFO:recall_aml:compiler_provider_usage {"prompt_tokens":300,"total_tokens":381}'
    )
    assert json.loads(lines[2].split(" ", 1)[1]) == {"prompt_tokens": 300, "total_tokens": 381}


def test_a_record_without_extra_fields_is_unchanged(journal) -> None:
    """A plain record keeps the exact pre-fix format, so existing journal searches still match.

    Red proof (mutation): appending the JSON object unconditionally in
    ``ExtraFieldsFormatter.formatMessage`` fails the equality with a trailing ``{}``.
    """
    logging.getLogger("httpx2").info("HTTP Request: POST https://example.invalid")

    assert journal.getvalue().splitlines() == [
        "INFO:httpx2:HTTP Request: POST https://example.invalid"
    ]
