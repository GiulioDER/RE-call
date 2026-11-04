"""Scratch only: apply one named mutation to the checkout for red proof. Removed before review."""

import pathlib
import sys

MUTATIONS = {
    "none": [],
    "no-scrub-guard": [(
        "recall/control_plane.py",
        "        if conn.execute(\"SELECT to_regclass('recall_migration_events')\").fetchone()[0] is None:\n            return 0\n",
        "",
    )],
    "no-superseded-branch": [(
        "recall/drift.py",
        "        superseded = _superseded_on(repository, generation_id)\n",
        "        superseded = None\n",
    )],
    "no-doctor-generation": [(
        "recall/doctor.py",
        "    generation = _active_generation(conn, tenant) if generation_route else None\n",
        "    generation = None\n",
    )],
    "no-doctor-calibration": [(
        "recall/doctor.py",
        "    status = GenerationManager(dsn, tenant).calibration_status_for(generation_id)\n",
        "    status = GenerationManager(dsn, tenant).calibration_status_for(generation_id)\n    return _calibration_check(\"hashing\", strict=strict, dsn=dsn)\n",
    )],
}

for path, old, new in MUTATIONS[sys.argv[1]]:
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, old)
    p.write_text(text.replace(old, new), encoding="utf-8")
    print(f"mutated {path}")
