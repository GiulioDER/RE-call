"""Every deliberate product exception derives from RecallError.

Sixty-five exception classes existed with no common base, so a library consumer could not
catch package errors distinctly from Python built-ins. This walks the product packages and
asserts the property structurally, so a new exception family cannot silently opt out.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import recall
import recall_mcp
from recall.errors import RecallError

#: Modules whose import needs optional extras or heavy runtimes; their exceptions are
#: covered when the extras are installed, and skipping the import here keeps this test
#: runnable in the minimal environment.
_OPTIONAL_PREFIXES = (
    "recall.desktop.ui",
    "recall.desktop.install_ui",
    "recall.desktop.main",
    "recall.integrations",
    "recall.eval",
    "recall.wizard.llm",
)

#: Exceptions that are deliberately NOT RecallError: private one-module signalling types.
_EXEMPT = {"_ArtifactRefusal", "_TooDeep"}


def _product_modules() -> list[str]:
    names: list[str] = []
    for pkg in (recall, recall_mcp):
        names.append(pkg.__name__)
        for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
            if not info.name.startswith(_OPTIONAL_PREFIXES):
                names.append(info.name)
    return names


def test_every_product_exception_is_a_recall_error() -> None:
    offenders: list[str] = []
    for name in _product_modules():
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue  # an optional dependency this environment lacks
        for cls_name, cls in vars(module).items():
            if (
                inspect.isclass(cls)
                and issubclass(cls, Exception)
                and not issubclass(cls, RecallError)
                and cls.__module__ == name
                and cls_name not in _EXEMPT
            ):
                offenders.append(f"{name}.{cls_name}")
    assert not offenders, (
        "product exceptions outside the RecallError hierarchy (add RecallError as a base, "
        f"or add a justified exemption here): {sorted(set(offenders))}"
    )


def test_the_builtin_bases_survive_reparenting() -> None:
    """Call sites catch RuntimeError/ValueError today; re-parenting must not break them."""
    from recall.schema import SchemaError, SchemaIncompatible
    from recall_mcp.limits import RateLimited

    assert issubclass(SchemaError, RuntimeError)
    assert issubclass(SchemaIncompatible, ValueError)
    assert issubclass(RateLimited, RuntimeError)
    with pytest.raises(RecallError):
        raise RateLimited("x", retry_after_seconds=1.0)
