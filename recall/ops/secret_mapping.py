"""Which environment variables may be filled from AWS Secrets Manager, and parsing the map.

A leaf shared by `recall_mcp.settings` and `recall.ops.secrets`, which imported each other.
`recall_mcp.settings` re-exports both names.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

# These are the only process environment destinations that may receive a value fetched from AWS.
# In particular, deployment mode, trust policy, authentication selectors and OIDC policy are not
# secret destinations and cannot be overridden by a secret payload.
SECRET_DESTINATIONS = frozenset(
    {
        "RECALL_SERVING_DSN",
        "RECALL_MIGRATION_DSN",
        "RECALL_FACT_WRITE_DSN",
        "RECALL_REDIS_URL",
        "VOYAGE_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "RECALL_REASONING_EXPANSION_API_KEY",
        "RECALL_REASONING_ANSWER_API_KEY",
    }
)


def secret_mapping_from_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if source is None else source
    raw = str(source.get("RECALL_AWS_SECRET_MAPPING", "")).strip()
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("RECALL_AWS_SECRET_MAPPING must be a JSON object") from exc
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value.strip()
        for key, value in decoded.items()
    ):
        raise ValueError("RECALL_AWS_SECRET_MAPPING must map environment names to secret names")
    unknown = sorted(set(decoded) - SECRET_DESTINATIONS)
    if unknown:
        allowed = ", ".join(sorted(SECRET_DESTINATIONS))
        raise ValueError(
            "RECALL_AWS_SECRET_MAPPING contains forbidden destination(s): "
            f"{', '.join(unknown)}; allowed destinations are {allowed}"
        )
    return {str(key): str(value) for key, value in decoded.items()}
