"""Join our predictions onto the original MTRAG task rows, producing the SCORER's input format.

Two different formats, and conflating them is how a leak or a crash happens:

  SUBMISSION  (what `format_checker.py` validates, what a leaderboard receives)
      task_id, Collection, input, contexts, predictions -- and deliberately NO gold.
  SCORING     (what `run_algorithmic.py` consumes, per the official README: "a file in our
              generation format ... which for each task also includes an additional new
              `predictions` attribute")
      the ORIGINAL task row, gold `targets` and all, plus `predictions`.

⚠️ The gold belongs in the SCORER's input and must never reach the GENERATOR's. That is not a
contradiction, it is the whole design: the generation run read `reference.jsonl` through a path
that touches only `speaker`/`text`, and this join happens strictly afterwards, on its output.
`run_algorithmic.py` fails with KeyError: 'targets' on a submission file, which is the scorer
telling you it has nothing to score against.

Also note the official `sample_data/responses-10.jsonl` carries `targets` AND `answerability`:
that file is an example of the SCORING format, not of a submission.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                rows[row["task_id"]] = row
    return rows


def main(argv: list[str]) -> int:
    tasks_path, preds_path, out_path = (Path(a) for a in argv[:3])
    tasks, preds = load(tasks_path), load(preds_path)

    missing = [t for t in preds if t not in tasks]
    if missing:
        raise SystemExit(f"{len(missing)} predicted task_ids absent from {tasks_path.name}")

    written = no_pred = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for task_id, task in tasks.items():
            row = dict(task)
            pred = preds.get(task_id)
            if pred is None:
                # Scoring a task we never answered would compare gold against nothing and quietly
                # count as a zero. Better to leave it out and report the gap than to score air.
                no_pred += 1
                continue
            # OUR contexts and OUR answer replace the release's, because that is what is being
            # judged. Everything else (targets, Answerability, Question Type) stays as shipped:
            # the scorer needs it and the generator never saw it.
            row["predictions"] = pred["predictions"]
            row["contexts"] = pred.get("contexts", row.get("contexts"))
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(json.dumps({
        "tasks": len(tasks), "predictions": len(preds), "written": written,
        "tasks_without_prediction": no_pred,
        "has_targets": all("targets" in json.loads(l) for l in out_path.open(encoding="utf-8")),
        "output": str(out_path),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
