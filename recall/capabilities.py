"""Capability planning and dependency diagnostics without loading optional models."""

from __future__ import annotations

import importlib
import importlib.util
import platform
import shutil
from dataclasses import dataclass
from typing import Literal

CapabilityKind = Literal["python", "executable"]


@dataclass(frozen=True)
class CapabilityRequirement:
    """One optional capability needed by a selected pipeline."""

    name: str
    kind: CapabilityKind
    target: str
    package: str | None
    remediation: str


@dataclass(frozen=True)
class DependencyDiagnostic:
    """A stable, operator safe explanation of a capability result."""

    code: str
    component: str
    phase: str
    target: str
    detail: str
    remediation: str
    platform: str

    @property
    def ok(self) -> bool:
        return self.code == "capability.available"

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "code": self.code,
            "component": self.component,
            "phase": self.phase,
            "target": self.target,
            "detail": self.detail,
            "remediation": self.remediation,
            "platform": self.platform,
            "ok": self.ok,
        }

    def render(self) -> str:
        return (
            f"{self.component} {self.phase}: {self.code}: {self.detail}. "
            f"{self.remediation}"
        )


def requirements_for(
    embedder: str,
    *,
    documents: bool = False,
    rerank: bool = False,
    entailment: bool = False,
) -> tuple[CapabilityRequirement, ...]:
    """Plan only the capabilities a route actually needs.

    This function performs no imports and no model construction. It is safe for startup checks,
    dry runs, and environments where the optional extras are intentionally absent.
    """

    backend = embedder.split(":", 1)[0].lower()
    requirements: list[CapabilityRequirement] = []
    if backend == "fastembed":
        requirements.extend(
            [
                CapabilityRequirement(
                    "fastembed", "python", "fastembed", "fastembed", 'pip install "recall-rag[fastembed]"'
                ),
                CapabilityRequirement(
                    "onnxruntime", "python", "onnxruntime", "onnxruntime", 'pip install "recall-rag[fastembed]"'
                ),
            ]
        )
    elif backend in {"st", "sfr-code"}:
        requirements.append(
            CapabilityRequirement(
                "sentence transformers", "python", "sentence_transformers", "rerank", 'pip install "recall-rag[rerank]"'
            )
        )
    elif backend == "voyage":
        requirements.append(
            CapabilityRequirement(
                "voyageai", "python", "voyageai", "voyage", 'pip install "recall-rag[voyage]"'
            )
        )
    elif backend in {"openai", "openrouter"}:
        requirements.append(
            CapabilityRequirement(
                "openai", "python", "openai", "openai", 'pip install "recall-rag[openai]"'
            )
        )

    if documents:
        requirements.extend(
            CapabilityRequirement(name, "python", module, "documents", 'pip install "recall-rag[documents]"')
            for name, module in (
                ("pypdf", "pypdf"),
                ("pdfplumber", "pdfplumber"),
                ("python docx", "docx"),
                ("openpyxl", "openpyxl"),
                ("python pptx", "pptx"),
                ("beautiful soup", "bs4"),
            )
        )
        requirements.append(
            CapabilityRequirement(
                "libreoffice", "executable", "libreoffice", None,
                "install LibreOffice and ensure its executable is on PATH",
            )
        )
    if rerank:
        requirements.append(
            CapabilityRequirement(
                "cross encoder", "python", "sentence_transformers", "rerank", 'pip install "recall-rag[rerank]"'
            )
        )
    if entailment:
        requirements.append(
            CapabilityRequirement(
                "entailment", "python", "sentence_transformers", "entail", 'pip install "recall-rag[entail]"'
            )
        )
    return tuple(requirements)


def diagnose_exception(
    component: str,
    phase: str,
    exc: BaseException,
    *,
    target: str | None = None,
    remediation: str | None = None,
) -> DependencyDiagnostic:
    """Classify missing packages, native loader failures, and executable failures consistently."""

    target_name = target or component
    missing_name = getattr(exc, "name", None)
    cause = exc.__cause__
    native = isinstance(exc, OSError) or isinstance(cause, OSError)
    text = str(exc).replace("\r", " ").replace("\n", " ").strip()
    lower = text.lower()
    native = native or any(token in lower for token in ("dll", "shared object", "undefined symbol"))
    if isinstance(exc, FileNotFoundError):
        code = "dependency.executable_missing"
        detail = f"executable {target_name!r} was not found"
    elif isinstance(exc, ModuleNotFoundError) or (
        isinstance(exc, ImportError) and missing_name
    ):
        code = "dependency.missing"
        detail = f"package {missing_name or target_name!r} is not importable"
    elif native:
        code = "dependency.native_load_failed"
        detail = f"native dependency could not load: {text or type(exc).__name__}"
    else:
        code = "capability.failed"
        detail = f"{type(exc).__name__}: {text or 'no further detail'}"
    return DependencyDiagnostic(
        code=code,
        component=component,
        phase=phase,
        target=target_name,
        detail=detail,
        remediation=remediation or "inspect the dependency installation and retry",
        platform=platform.platform(aliased=True),
    )


def probe(requirement: CapabilityRequirement, *, load: bool = False) -> DependencyDiagnostic:
    """Probe a requirement, optionally importing Python modules to catch native loader failures."""

    try:
        if requirement.kind == "executable":
            if shutil.which(requirement.target) is None:
                raise FileNotFoundError(requirement.target)
        elif load:
            importlib.import_module(requirement.target)
        elif importlib.util.find_spec(requirement.target) is None:
            error = ModuleNotFoundError(requirement.target)
            error.name = requirement.target
            raise error
    except Exception as exc:  # BROAD-CATCH: error-translation; capability probes return diagnostics
        return diagnose_exception(
            requirement.name,
            "probe",
            exc,
            target=requirement.target,
            remediation=requirement.remediation,
        )
    return DependencyDiagnostic(
        code="capability.available",
        component=requirement.name,
        phase="probe",
        target=requirement.target,
        detail="available" + (" and importable" if load and requirement.kind == "python" else ""),
        remediation="none",
        platform=platform.platform(aliased=True),
    )


__all__ = [
    "CapabilityRequirement",
    "DependencyDiagnostic",
    "diagnose_exception",
    "probe",
    "requirements_for",
]
