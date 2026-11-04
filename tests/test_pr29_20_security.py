"""SEC-001/SEC-002: refuse unsafe destinations before any connection.

Red proof: baseline 418511e98, Linux, 2026-11-03. The initial six nodes failed
their admission assertions with unchanged guard implementations. The IPv6
identity case also fails against the pre-canonicalization repair. Receipts
preserve their node IDs, source hashes and subsequent green results.
"""
import os
import pytest
from tests import conftest
from recall.store import require_secure_dsn

@pytest.mark.parametrize("dsn, production, override", [
    ("host=db.example.com dbname=throwaway user=tester password=example", None, None),
    ("postgresql://tester:example@localhost/throwaway?host=db.example.com", None, None),
    ("postgresql://tester:example@localhost/throwaway?hostaddr=192.0.2.23", None, None),
    ("postgresql://tester:example@db.example.com/throwaway", None, "0"),
    ("postgresql://tester:example@localhost/production?application_name=tests", "postgresql://tester:example@localhost/production", None),
    ("host=::1 port=5432 dbname=production user=tester password=example", "host=0:0:0:0:0:0:0:1 port=5432 dbname=production user=tester password=example", None),
])
def test_unsafe_test_destination_is_refused(monkeypatch, dsn, production, override):
    for name in list(os.environ):
        if name.startswith("PG"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_TEST_DSN", dsn)
    for name, value in (("RECALL_DSN", production), ("RECALL_ALLOW_REMOTE_TEST_DB", override)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    refused = False
    try:
        conftest._reject_unsafe_test_dsn()
    except RuntimeError:
        refused = True
    assert refused, "unsafe destructive-test destination was admitted"


def test_default_credentials_follow_environment_destination(monkeypatch):
    for name in list(os.environ):
        if name.startswith("PG"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("RECALL_ALLOW_INSECURE_DSN", raising=False)
    monkeypatch.setenv("PGHOSTADDR", "192.0.2.23")
    refused = False
    try:
        require_secure_dsn("postgresql://recall:recall@localhost/recall")
    except PermissionError:
        refused = True
    assert refused, "environment-selected remote default credentials were admitted"
