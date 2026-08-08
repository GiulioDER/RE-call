"""Local competitive-benchmark harness (RE-call vs Mem0).

Not shipped in the wheel. But `benchmarks/__init__.py` and `benchmarks/evidence_injection.py`
ARE on CI's import path — `tests/test_evidence_injection.py` imports the latter and the test
job runs the whole suite — so both must stay free of `bench`-extra dependencies. A module-level
import of one there is a CI-breaking change, not a local-only one. The rest is local-only.
"""
