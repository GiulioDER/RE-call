"""Repair controls: unresolved services and distinct local destinations."""
import os
import pytest
from recall.store import require_secure_dsn
from tests import conftest

@pytest.fixture
def service_environment(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("PG"):
            monkeypatch.delenv(name, raising=False)
    service = tmp_path / "pg_service.conf"
    service.write_text("[local]\nhost=localhost\nhostaddr=127.0.0.1\n[remote]\nhost=db.example.com\nhostaddr=192.0.2.23\n")
    monkeypatch.setenv("PGSERVICEFILE", str(service))
    monkeypatch.setenv("PGSERVICE", "local")
    monkeypatch.delenv("RECALL_ALLOW_INSECURE_DSN", raising=False)
    monkeypatch.delenv("RECALL_ALLOW_REMOTE_TEST_DB", raising=False)
    monkeypatch.delenv("RECALL_DSN", raising=False)


def test_explicit_service_cannot_inherit_another_services_local_safety(service_environment):
    refused = False
    try:
        require_secure_dsn("service=remote user=recall password=recall")
    except PermissionError:
        refused = True
    assert refused, "an unresolved explicit service inherited ambient local safety"


def test_destructive_service_destination_is_refused(service_environment, monkeypatch):
    monkeypatch.setenv("RECALL_TEST_DSN", "service=remote user=tester password=example dbname=throwaway")
    refused = False
    try:
        conftest._reject_unsafe_test_dsn()
    except RuntimeError:
        refused = True
    assert refused, "an unresolved test service destination was admitted"


@pytest.mark.parametrize("test_host,production_host", [("/tmp/test-cluster", "/tmp/production-cluster"), ("127.0.0.2", "127.0.0.3")])
def test_distinct_local_destinations_are_preserved(monkeypatch, test_host, production_host):
    for name in list(os.environ):
        if name.startswith("PG"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_TEST_DSN", f"host={test_host} port=5432 dbname=recall")
    monkeypatch.setenv("RECALL_DSN", f"host={production_host} port=5432 dbname=recall")
    rejected = False
    try:
        conftest._reject_unsafe_test_dsn()
    except RuntimeError:
        rejected = True
    assert not rejected, "distinct local clusters were collapsed into one destination"
