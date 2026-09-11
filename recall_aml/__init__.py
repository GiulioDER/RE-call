"""Hosted vendor API for RE-call.

The Starlette dependency is intentionally lazy so importing the package from a core RE-call
installation still works and points the operator at the hosted extra only when the API is used.
"""

from typing import Any

from recall_aml.config import HostedSettings


def create_app(*args: Any, **kwargs: Any) -> Any:
    from recall_aml.app import create_app as factory

    return factory(*args, **kwargs)


__all__ = ["HostedSettings", "create_app"]
