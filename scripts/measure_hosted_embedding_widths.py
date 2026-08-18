#!/usr/bin/env python
"""Re-measure the width and L2 norm of every registered hosted embedding profile.

This is the re-measure command that `recall/embedding_registry.py` promises beside its declared
hosted dimensions. A declared width is a CLAIM about a provider that can change it without
telling anyone, so the claim needs a way to be re-checked cheaply rather than a date and a hope.

    python scripts/measure_hosted_embedding_widths.py

Needs a key per provider in the environment (`OPENROUTER_API_KEY`, `VOYAGE_API_KEY`); profiles
whose key is absent are reported as SKIPPED rather than silently passing. Costs one embedding
call per profile, which is a fraction of a cent in total.

Exit status is 1 if any measured width or normalization disagrees with what the registry
declares, so this is usable as a periodic check and not only as a human-read report.
"""
from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# The repository root, ahead of everything else. Running a file under `scripts/` puts the SCRIPT's
# directory on `sys.path[0]`, not the working directory, so a bare `import recall` resolves through
# the editable install instead, and on this project that install is shared by every worktree, so
# the script would report the registry of a DIFFERENT checkout while appearing to work. Observed
# on 2026-08-18: it imported `C:/Users/gde00/Documents/recall/recall` from a worktree whose own
# registry it was meant to be measuring.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recall.embedding_registry import REGISTERED_PROFILES, RegisteredProfile  # noqa: E402

TIMEOUT_SECONDS = 60
#: How far a vector's L2 norm may sit from 1.0 while `normalization="l2"` is still a true claim.
#: Loose enough for float32 round-tripping through JSON (the measured spread on OpenAI's models is
#: ~4e-4), tight enough to catch a genuinely unnormalised Matryoshka width (measured 0.694).
NORM_TOLERANCE = 1e-2


def _post(url: str, payload: dict, key: str) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.load(response)


def measure(entry: RegisteredProfile, key: str) -> tuple[int, float]:
    """Return the (width, L2 norm) one probe string embeds to under this profile."""
    if entry.backend == "voyage":
        body = _post(
            "https://api.voyageai.com/v1/embeddings",
            {"model": entry.model_name, "input": ["probe"]},
            key,
        )
    else:
        assert entry.base_url is not None
        payload: dict[str, object] = {
            "model": entry.model_name,
            "input": ["probe"],
            "encoding_format": "float",
        }
        if entry.output_dimensions is not None:
            payload["dimensions"] = entry.output_dimensions
        body = _post(f"{entry.base_url}/embeddings", payload, key)
    vector = body["data"][0]["embedding"]
    return len(vector), math.sqrt(sum(float(x) * float(x) for x in vector))


def main() -> int:
    hosted = [e for e in REGISTERED_PROFILES.values() if e.hosted]
    if not hosted:
        print("no hosted profiles registered")
        return 0
    failures = 0
    print(f"{'profile':34s} {'declared':>8s} {'measured':>8s} {'norm':>9s}  verdict")
    for entry in sorted(hosted, key=lambda e: e.profile_id):
        key = os.environ.get(entry.api_key_env, "")
        if not key:
            print(f"{entry.profile_id:34s} {entry.dimension:8d} {'-':>8s} {'-':>9s}  "
                  f"SKIPPED ({entry.api_key_env} unset)")
            continue
        try:
            width, norm = measure(entry, key)
        except (urllib.error.URLError, KeyError, IndexError, ValueError) as exc:
            failures += 1
            print(f"{entry.profile_id:34s} {entry.dimension:8d} {'-':>8s} {'-':>9s}  "
                  f"ERROR {type(exc).__name__}: {exc}")
            continue
        problems = []
        if width != entry.dimension:
            problems.append(f"width {width} != declared {entry.dimension}")
        if entry.normalization == "l2" and abs(norm - 1.0) > NORM_TOLERANCE:
            problems.append(f"declares l2 but norm is {norm:.4f}")
        verdict = "ok" if not problems else "MISMATCH: " + "; ".join(problems)
        failures += bool(problems)
        print(f"{entry.profile_id:34s} {entry.dimension:8d} {width:8d} {norm:9.6f}  {verdict}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
