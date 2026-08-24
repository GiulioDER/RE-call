"""Check prior work declarations on newly added experiment modules."""

from __future__ import annotations

import ast
from pathlib import Path


def check_prior_work_declarations(paths: list[Path]) -> list[str]:
    """Return errors for Python modules without a first statement prior work declaration."""

    errors: list[str] = []
    for path in paths:
        if path.suffix.lower() != ".py":
            errors.append(f"{path} is not a Python experiment module")
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"{path} cannot be read: {exc}")
            continue
        try:
            module = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{path} has invalid Python syntax: {exc}")
            continue
        if not module.body or not isinstance(module.body[0], ast.Expr):
            errors.append(f"{path} must start with a module docstring containing Prior work:")
            continue
        value = module.body[0].value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            errors.append(f"{path} must start with a module docstring containing Prior work:")
            continue
        has_declaration = any(
            line.strip().startswith("Prior work:") for line in value.value.splitlines()
        )
        if not has_declaration:
            errors.append(f"{path} module docstring must contain a Prior work: declaration")
    return errors
