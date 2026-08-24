"""The common base for every exception this package raises on purpose.

Sixty-five exception classes existed with no shared root, so a library consumer could not
write `except RecallError` and had to enumerate families or catch Python built-ins. Each
family keeps its historical built-in base (`RuntimeError` or `ValueError`) as the FIRST
base, because call sites inside and outside this repository already catch those; RecallError
is added as a second base, so both `except RuntimeError` and `except RecallError` keep
working. New exception types should inherit from an existing family root, or from this
class directly alongside the built-in that best matches their meaning.
"""

from __future__ import annotations


class RecallError(Exception):
    """Base for every deliberate recall/recall_mcp exception. See the module docstring."""
