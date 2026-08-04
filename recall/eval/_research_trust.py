"""The trust policy research harnesses run under, and why it is not the production one.

A benchmark measures *retrieval quality*. It runs against LoCoMo, LongMemEval, BEIR and synthetic
corpora that have no tenant, no generation, and no published calibration artifact bound to them,
and that is not a defect in the harness: the whole point of a sweep is to score a corpus nobody
has certified yet. Under the production policy every one of those calls would refuse, and the
research path would be unable to measure the thing it exists to measure.

So the harnesses opt into `TrustPolicy.development()` **explicitly, here, once**. Three properties
matter and none of them are accidents:

1. It is a *separate callable*, not a default argument somewhere in `recall.trust`. A reader of
   `harness.py` sees `research_search` and can ask why; a silent default would have looked like
   ordinary search and quietly reopened the fail-open hole this session closed.
2. It lives outside `recall_mcp` and outside the adapters, so no serving path can reach it.
3. It still degrades honestly. Development mode does not invent a verdict: every hit comes back
   `unverified` with `trust_state=degraded`, so a harness that reads verdicts gets the truth that
   nothing was certified, rather than `ok` computed against the 0.50 floor.

If you are writing a *serving* path and you find yourself importing this, that is the bug.
"""

from __future__ import annotations

from typing import Any

from recall.trust import trusted_search
from recall.trust_policy import TrustPolicy
from recall.types import TrustedResult

#: One shared instance; `TrustPolicy` is frozen, so sharing it is safe.
RESEARCH_POLICY = TrustPolicy.development()


def research_search(*args: Any, **kwargs: Any) -> TrustedResult:
    """`trusted_search` in development mode, for benchmark and evaluation harnesses only.

    A caller that passes its own `policy` keeps it, so a harness that deliberately wants to
    measure strict-mode refusal behaviour can still do so.
    """
    kwargs.setdefault("policy", RESEARCH_POLICY)
    return trusted_search(*args, **kwargs)
