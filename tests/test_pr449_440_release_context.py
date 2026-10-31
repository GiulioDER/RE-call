"""DEPLOY-001: release operations need context in the checkout-free job."""
import re
from pathlib import Path

import yaml


def test_signed_attachment_commands_have_repository_context_without_checkout():
    """Baseline160a23a3f fails repository-context assertion, no workflow is dispatched."""
    workflow = yaml.safe_load(Path(".github/workflows/windows-installer.yml").read_text())
    job = workflow["jobs"]["attach"]
    checked_out = False
    observed = []
    for step in job["steps"]:
        if str(step.get("uses", "")).startswith("actions/checkout@"):
            checked_out = True
        script = str(step.get("run", ""))
        env = {**workflow.get("env", {}), **job.get("env", {}), **step.get("env", {})}
        for line in script.splitlines():
            if re.search(r"\bgh release (view|upload)\b", line):
                observed.append(line)
                assert checked_out or env.get("GH_REPO") == "${{ github.repository }}" or (
                    "--repo " in line or "-R " in line
                ), "release command cannot infer a repository in a checkout-free job"
    assert len(observed) == 2
    assert job["permissions"] == {"contents": "write"}
