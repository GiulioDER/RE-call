"""Which code produced each MTRAG arm artifact, established from the artifacts themselves.

The per-arm `.metrics.json` carries no code version. `preregistered_manifest.json` does, but it is
written per RUN to a fixed filename, so a second run into the same output directory overwrites the
first run's provenance. This reports what survived, and says plainly what did not.
"""

import glob
import json
import re
from pathlib import Path

D = "/var/tmp/mtrag_taskA_dev_20260807"

print("=== manifest on disk (belongs to the LAST run written here) ===")
man = json.loads(Path(f"{D}/preregistered_manifest.json").read_text(encoding="utf-8"))
print("  started_at:", man.get("started_at"))
print("  recall rev:", (man.get("revisions") or {}).get("recall"))

print()
print("=== run.log: every recall revision it ever recorded ===")
log = Path(f"{D}/run.log").read_text(encoding="utf-8", errors="replace")
revs = sorted(set(re.findall(r'"recall":\s*"([0-9a-f]{7,40})"', log)))
print("  revisions found:", revs or "NONE (the dev run's manifest event is not in run.log)")
# [A-Za-z0-9_]+, not [a-z_]+: arm names carry digits (recall_default_recent3,
# recall_rerank_recent3), and [a-z_]+ consumes "recall_default_recent", then needs a closing quote,
# finds "3", and fails the match outright. A provenance report that silently omits an arm reads as
# "that arm never started", which is the opposite of what it is for.
starts = re.findall(r'"event":\s*"arm_start",\s*"arm":\s*"([A-Za-z0-9_]+)"', log)
print("  arm_start events:", starts or "none")

print()
print("=== per-arm metrics: what each artifact can prove about itself ===")
for p in sorted(glob.glob(f"{D}/*.metrics.json")):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    name = Path(p).name.replace(".metrics.json", "")
    # `retried_queries` was added in f47d927. Its ABSENCE dates the artifact to before that commit,
    # which is inference from a missing key, not provenance.
    dated = "pre-f47d927" if "retried_queries" not in d else "f47d927 or later"
    o = d["scores"]["overall"]
    print(
        f"  {name:22} rev={'ABSENT':8} dated_by_schema={dated:16} "
        f"nDCG@5={o['nDCG@5']:.4f} R@100={o['Recall@100']:.4f}"
    )

print()
print("=== voyage (in flight) ===")
vlog = Path(f"{D}/voyage.log").read_text(encoding="utf-8", errors="replace")
done = re.findall(r'"completed":\s*(\d+)', vlog)
print("  progress:", (done[-1] if done else "?"), "/ 777")
print("  retries :", len(re.findall(r'"event":\s*"query_retry"', vlog)))
