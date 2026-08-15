"""Configuration for eval calibration fleet tests.

The fleet tests use ScriptedEmbedder and do not need the database bootstrap that validates a
chunks table with a specific embedding dimension. This conftest overrides the parent autouse
bootstrap fixture with a no-op version, allowing fleet tests to run without triggering schema
validation against the existing database.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import pytest

if TYPE_CHECKING:
    pass


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_default_test_schema() -> Iterator[None]:
    """Override parent bootstrap to skip database setup for fleet tests.

    Fleet tests use ScriptedEmbedder(dim=4) which neither reads nor writes the database
    directly; they call _score_config from the real eval harness, which does hit the DB via
    the retriever. However, the ScriptedEmbedder and QueryKeyedStore are entirely deterministic
    and do not depend on the current schema state.
    """
    yield
