"""Create a fail-closed, secret-safe release manifest for RE-call Hosted 1.0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from recall_aml.compiler import facet_prompt_digest, prompt_digest
from recall_aml.config import (
    EMBEDDING_PROFILE,
    GENERATION_MODEL,
    GENERATION_PROVIDER,
    PRODUCT_NAME,
    PRODUCT_VERSION,
    RERANK_MODEL,
    RETRIEVAL_PROFILE,
    SCHEMA_VERSION,
)
from recall_aml.variants import variant


BOUND_REPOSITORY_ARTIFACTS = {
    "cloudflare_ingress_template": Path("infra/systemd/cloudflared-ingress.example.yml"),
    "compiler_source": Path("recall_aml/compiler.py"),
    "environment_template": Path("infra/systemd/hosted.env.example"),
    "lockfile": Path("uv.lock"),
    "preregistration": Path("docs/preregistrations/2026-09-12-aml-hosted-industry-v1.md"),
    "project_metadata": Path("pyproject.toml"),
    "service_unit": Path("infra/systemd/recall-aml.service"),
}
SECRET_VARIABLE_NAMES = (
    "OPENROUTER_API_KEY",
    "RECALL_AML_API_KEY",
    "RECALL_AML_DATABASE_URL",
    "VOYAGE_API_KEY",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_repository(repo_root: Path, expected_commit: str) -> str:
    """Return HEAD only when it is exact and the tracked checkout is clean."""
    head = _git(repo_root, "rev-parse", "HEAD")
    if head != expected_commit:
        raise RuntimeError(f"HEAD {head} does not match expected commit {expected_commit}")
    status = _git(repo_root, "status", "--porcelain")
    if status:
        raise RuntimeError("release repository has checkout changes")
    return head


def build_manifest(
    *,
    repo_root: Path,
    wheel_path: Path,
    commit: str,
    variant_name: str,
) -> dict[str, Any]:
    """Build a manifest from public identities and artifact digests only."""
    selected = variant(variant_name)
    artifacts: dict[str, dict[str, object]] = {}
    paths = {"wheel": wheel_path, **BOUND_REPOSITORY_ARTIFACTS}
    for name, raw_path in sorted(paths.items()):
        path = raw_path if raw_path.is_absolute() else repo_root / raw_path
        if not path.is_file():
            raise FileNotFoundError(f"required release artifact is missing: {path}")
        artifacts[name] = {
            "path": str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    return {
        "manifest_schema": "recall-hosted-release-v1",
        "product": PRODUCT_NAME,
        "product_version": PRODUCT_VERSION,
        "git_commit": commit,
        "schema_version": SCHEMA_VERSION,
        "embedding_profile": EMBEDDING_PROFILE,
        "retrieval_profile": RETRIEVAL_PROFILE,
        "generation_provider": GENERATION_PROVIDER,
        "generation_model": GENERATION_MODEL,
        "reranker": RERANK_MODEL,
        "compiler_prompt_sha256": prompt_digest(),
        "facet_prompt_sha256": facet_prompt_digest(),
        "variant": {
            "name": selected.name,
            "raw": selected.raw,
            "compiler": selected.compiler,
            "facets": selected.facets,
            "reranker": selected.reranker,
            "pack": selected.pack,
            "context_chars": selected.context_chars,
        },
        "secret_policy": {
            "values_included": False,
            "required_variable_names": list(SECRET_VARIABLE_NAMES),
        },
        "artifacts": artifacts,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite release manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--variant", default="A4_pack_7000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    verify_repository(repo_root, args.commit)
    manifest = build_manifest(
        repo_root=repo_root,
        wheel_path=args.wheel,
        commit=args.commit,
        variant_name=args.variant,
    )
    write_manifest(args.output.resolve(), manifest)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
