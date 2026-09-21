"""Static safety proofs for the isolated VPS2 experiment launcher."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "aml_experience_vps2_setup.sh"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_experiment_unit_controls_every_unit_file_and_systemd_operation() -> None:
    """A C7 launch must not overwrite or restart the existing C6 unit."""
    source = _source()

    assert 'RECALL_AML_EXPERIMENT_UNIT:-recall-aml-experiment.service' in source
    assert '^recall-aml-[a-z0-9][a-z0-9-]*\\.service$' in source
    assert 'readonly unit_path="${unit_dir}/${service_unit}"' in source
    assert 'unit_tmp="$(mktemp "${unit_dir}/${service_unit}.XXXXXX")"' in source
    assert 'systemctl --user enable "$service_unit"' in source
    assert 'systemctl --user restart "$service_unit"' in source
    assert 'systemctl --user status "$service_unit"' in source
    assert source.count("recall-aml-experiment.service") == 1


def test_c6_and_c7_have_distinct_runtime_state_and_three_worker_capacity() -> None:
    """The two candidates need separate state while preserving the registered worker count."""
    source = _source()

    assert 'readonly runtime_env="${runtime_dir}/code4-exact-official.env"' in source
    assert 'readonly table="recall_aml_code4_exact_official_chunks"' in source
    assert 'readonly runtime_env="${runtime_dir}/routed-specialists.env"' in source
    assert 'readonly table="recall_aml_routed_specialists_chunks"' in source
    concurrency_branch = source[source.index('if [[ "$selected_variant" == "C5_code4_bm25"') :]
    concurrency_branch = concurrency_branch[: concurrency_branch.index("    else")]
    assert '"$selected_variant" == "C6_code4_exact_bm25"' in concurrency_branch
    assert '"$selected_variant" == "C7_routed_specialists"' in concurrency_branch
    assert "RECALL_AML_ADD_CONCURRENCY=3" in concurrency_branch
    assert "RECALL_AML_SEARCH_CONCURRENCY=3" in concurrency_branch


def test_comparison_service_can_use_an_isolated_runtime_environment_file() -> None:
    """A private C6 launch must not rewrite the environment file used by public C6."""
    source = _source()

    assert 'RECALL_AML_RUNTIME_ENV_NAME:-${runtime_env##*/}' in source
    assert '^recall-aml-[a-z0-9][a-z0-9-]*\\.env$' in source
    assert 'readonly runtime_env_path="${runtime_dir}/${runtime_env_name}"' in source
    assert 'EnvironmentFile=$runtime_env_path' in source


def test_experiment_launcher_does_not_mutate_the_public_route() -> None:
    """Installing a private C7 unit must not edit or restart the public tunnel."""
    source = _source().lower()

    for forbidden in ("cloudflared", ".cloudflared", "config.yml", "memory.pred-markets.com"):
        assert forbidden not in source


def test_experiment_launcher_has_valid_bash_syntax() -> None:
    if sys.platform == "win32":
        pytest.skip("the Windows bash shim does not provide a Linux shell")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed on this platform")

    subprocess.run([bash, "-n", str(SCRIPT)], check=True)
