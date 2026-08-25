"""A strict refusal is the gate working, and it must not arrive looking like a crash.

`recall search` used to let `TrustRefusal` escape to the top of the interpreter, so the single
most common outcome of a fresh install -- documented as exactly that on the troubleshooting page
-- printed an eleven-frame Python traceback ending in
`recall.trust_policy.TrustRefusal: INDEX_NOT_READY: refused in strict trust mode`.

Two costs, and the second is the expensive one. A traceback reads as a bug in the tool rather than
a decision it made deliberately, and the one line saying what to do next is buried under frames.

These tests need no database: `refusal_message` is a pure function of the exception.
"""
from __future__ import annotations

import pytest

from recall.cli_commands.index_search import refusal_message
from recall.trust_policy import TrustFailureCode, TrustRefusal

ALL_CODES = list(TrustFailureCode)


def _refusal(code: TrustFailureCode, **kw) -> TrustRefusal:
    fields = {
        "calibration_status": "missing",
        "tenant_id": "default",
        "generation_id": None,
    }
    fields.update(kw)
    return TrustRefusal(code=code, **fields)


class TestRefusalMessage:
    @pytest.mark.parametrize("code", ALL_CODES, ids=lambda c: c.value)
    def test_every_code_is_named_verbatim(self, code: TrustFailureCode) -> None:
        """The codes are a published interface, so the operator-facing text must carry them.

        Parametrised over the enum rather than a hand-written list: a seventh code added later
        gets this coverage without anyone remembering to extend the test.
        """
        assert code.value in refusal_message(_refusal(code))

    @pytest.mark.parametrize("code", ALL_CODES, ids=lambda c: c.value)
    def test_every_code_states_that_no_decision_was_possible(self, code) -> None:
        """The distinction the codes exist for must survive into the rendered text.

        "The gate could not run" and "the gate ran and found nothing" are the same shape on the
        wire and opposite in meaning; `advice` is where that is spelled out, so the renderer has
        to include it rather than paraphrase.
        """
        assert "no trustworthy decision" in refusal_message(_refusal(code)).lower()

    def test_the_identity_is_shown(self) -> None:
        message = refusal_message(
            _refusal(TrustFailureCode.CALIBRATION_STALE, tenant_id="acme", generation_id="gen_x")
        )
        assert "acme" in message
        assert "gen_x" in message

    def test_a_missing_identity_renders_as_a_dash_not_the_word_none(self) -> None:
        """`generation=None` in operator-facing text reads as a value, not as an absence."""
        message = refusal_message(_refusal(TrustFailureCode.INDEX_NOT_READY, generation_id=None))
        assert "generation         -" in message
        assert "None" not in message

    def test_development_mode_is_offered_as_inspection_and_not_as_the_fix(self) -> None:
        """An error that recommends relaxing a gate must say what relaxing it costs.

        A relaxed gate has no failure mode once it is unnecessary: it stops erroring and quietly
        stamps `degraded` on answers that had earned `trusted`. Naming the cost in the same breath
        is what stops the workaround outliving the problem.
        """
        message = refusal_message(_refusal(TrustFailureCode.CALIBRATION_MISSING))
        assert "RECALL_TRUST_MODE=development" in message
        assert "degraded" in message
        assert "inspection, not for serving" in message


class TestARefusalCarriesNoCorpusBytes:
    """`TrustRefusal` is built only from system-controlled fields, and this pins that it stays so.

    The guarantee is structural -- the exception is never given chunk text, source names, previews
    or the query -- so this asserts the property rather than a filter, which is the point: there
    is no sanitiser here that anyone could forget to call.
    """

    def test_the_query_is_never_echoed(self) -> None:
        message = refusal_message(_refusal(TrustFailureCode.INDEX_NOT_READY))
        assert "caching layer" not in message  # a query string is not among the inputs at all

    def test_terminal_escapes_in_a_tenant_name_are_stripped(self) -> None:
        """`--tenant` is operator-supplied, and a terminal executes what it is sent.

        `\\x1b[2K\\r` erases the line just written, so a tenant name carrying it could scroll the
        refusal away while the command still exited non-zero. Same class of defect
        `tests/test_cli_terminal_injection.py` covers for corpus-controlled strings.
        """
        message = refusal_message(
            _refusal(TrustFailureCode.INDEX_NOT_READY, tenant_id="acme\x1b[2K\rgone")
        )
        assert "\x1b" not in message
        assert "\r" not in message
