"""Why does MM-1 (dual scope) lose on MobileMem's text-evidence questions? Free: stored data only.

For every question: the first rank of an item from an evidence session in B and in D, whether that
rank falls inside the prefix the reader was shown (the 30-image cap), how many images D showed
(any image switches MobileMem to its multimodal prompt), and B's and D's correctness.
"""
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import aml_x1_sources as x1  # noqa: E402
from aml_x1_mm_answers import capped, image_refs, items_for, rows  # noqa: E402

X = Path("/home/sentiment/mm1-mm3")
draw = json.loads((X / "x1/draw.json").read_text(encoding="utf-8"))
evidence = {}
for tenant in x1.tenants_for("mobilemem_omni", X / "x1/data", draw, "x1"):
    for q in tenant.questions:
        ev = q.get("evidence") or {}
        evidence[str(q["question_id"])] = {"sessions": set(ev.get("sessions") or []),
                                           "image": bool(q.get("image_evidence"))}
answers = collections.defaultdict(dict)
for line in (X / "x1b/heldout-answers.jsonl").read_text(encoding="utf-8").split("\n"):
    if line.strip():
        r = json.loads(line)
        if r["source"] == "mobilemem_omni" and r["correct"] is not None:
            answers[r["id"].split(":", 1)[1]][r["arm"]] = bool(r["correct"])


def first_rank(items, sessions):
    return next((rank for rank, item in enumerate(items, 1) if str(item.get("session_id")) in sessions), None)


table = []
for row in rows(X / "x1b/out", "mobilemem_omni"):
    qid = str(row["question_id"])
    ev = evidence[qid]
    if not ev["sessions"] or "B" not in answers[qid] or "D" not in answers[qid]:
        continue
    per = {}
    for arm in ("B", "D"):
        items = items_for(arm, row)
        shown, _ = capped(items)
        rank = first_rank(items, ev["sessions"])
        per[arm] = {"rank": rank, "shown_len": len(shown), "in_shown": rank is not None and rank <= len(shown),
                    "images": sum(len(image_refs(i)) for i in shown),
                    "text_items_shown": sum(1 for i in shown if not image_refs(i))}
    table.append({"qid": qid, "image_evidence": ev["image"], "B_ok": answers[qid]["B"], "D_ok": answers[qid]["D"],
                  "B2_ok": answers[qid].get("B2"), **{f"{a}_{k}": v for a in per for k, v in per[a].items()}})

out = {"questions_with_evidence_sessions": len(table)}
for kind, keep in (("text_evidence", lambda t: not t["image_evidence"]), ("image_evidence", lambda t: t["image_evidence"])):
    sub = [t for t in table if keep(t)]
    wins = [t for t in sub if t["D_ok"] and not t["B_ok"]]
    losses = [t for t in sub if t["B_ok"] and not t["D_ok"]]
    def summary(group):
        if not group:
            return {"n": 0}
        return {"n": len(group),
                "D_prompt_multimodal": sum(t["D_images"] > 0 for t in group),
                "evidence_in_shown_B": sum(t["B_in_shown"] for t in group),
                "evidence_in_shown_D": sum(t["D_in_shown"] for t in group),
                "evidence_rank_worse_in_D": sum((t["D_rank"] or 999) > (t["B_rank"] or 999) for t in group),
                "median_rank_B": statistics.median([t["B_rank"] or 999 for t in group]),
                "median_rank_D": statistics.median([t["D_rank"] or 999 for t in group]),
                "median_text_items_shown_B": statistics.median([t["B_text_items_shown"] for t in group]),
                "median_text_items_shown_D": statistics.median([t["D_text_items_shown"] for t in group]),
                "B2_also_wrong_where_D_lost": sum(t["B2_ok"] is False for t in group)}
    out[kind] = {"all": summary(sub), "D_wins": summary(wins), "D_losses": summary(losses)}
    switched = [t for t in sub if t["D_images"] > 0]
    kept = [t for t in sub if t["D_images"] == 0]
    out[kind]["accuracy_by_D_prompt"] = {
        "D_multimodal_prompt": {"n": len(switched), "B": round(sum(t["B_ok"] for t in switched) / max(1, len(switched)), 4),
                                "D": round(sum(t["D_ok"] for t in switched) / max(1, len(switched)), 4)},
        "D_text_prompt": {"n": len(kept), "B": round(sum(t["B_ok"] for t in kept) / max(1, len(kept)), 4),
                          "D": round(sum(t["D_ok"] for t in kept) / max(1, len(kept)), 4)}}
print(json.dumps(out, indent=2))
