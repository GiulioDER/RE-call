"""F1 storyline replay: the pre-registered readout plus the Amendment 4 noise floor."""
import json
import random
import statistics
import sys
from pathlib import Path

RUN = Path(sys.argv[1])
ARMS = ["r0", "r0b", "story", "digests", "story_digests"]


def load(name):
    path = RUN / name
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []


def boot(deltas, seed=20260925, n=10_000):
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(deltas, k=len(deltas))) for _ in range(n))
    return round(means[int(0.025 * n)], 3), round(means[int(0.975 * n) - 1], 3)


judged = {a: {r["id"]: r for r in load(f"judged-{a}.jsonl")} for a in ARMS}
answers = {a: {r["id"]: r for r in load(f"answers-{a}.jsonl")} for a in ARMS}
summ = sorted(i for i, r in judged["r0"].items() if r["type"] == "summarization")
print(f"summarization questions: {len(summ)}")

print("\n== summarization, per arm")
for a in ARMS:
    s = [judged[a][i]["score"] for i in summ if judged[a][i]["score"] is not None]
    empty = sum(1 for i in summ if not answers[a][i]["answer"].strip())
    unscored = sum(1 for i in summ if judged[a][i]["score"] is None)
    words = statistics.median(len(answers[a][i]["answer"].split()) for i in summ)
    print(f"{a:14s} mean {statistics.fmean(s):.4f}  n {len(s)}  unscored {unscored}  empty {empty}  "
          f"answer words p50 {words}")


def paired(a, b):
    ids = [i for i in summ if judged[a][i]["score"] is not None and judged[b][i]["score"] is not None]
    d = [judged[a][i]["score"] - judged[b][i]["score"] for i in ids]
    return round(statistics.fmean(d), 4), boot(d), len(d), sum(1 for v in d if v > 0), sum(1 for v in d if v < 0)


print("\n== paired deltas on summarization (mean, 95% CI, n, up, down)")
for a, b in [("r0b", "r0"), ("story", "r0"), ("digests", "r0"), ("story_digests", "r0"),
             ("story", "r0b"), ("digests", "r0b"), ("story_digests", "r0b"),
             ("story_digests", "story")]:
    print(f"{a:14s} - {b:6s} {paired(a, b)}")

# Pooled baseline: the mean of r0 and r0b per question halves the replicate noise.
print("\n== against the pooled baseline mean(r0, r0b)")
for a in ["story", "digests", "story_digests"]:
    ids = [i for i in summ if all(judged[x][i]["score"] is not None for x in (a, "r0", "r0b"))]
    d = [judged[a][i]["score"] - (judged["r0"][i]["score"] + judged["r0b"][i]["score"]) / 2 for i in ids]
    print(f"{a:14s} {round(statistics.fmean(d), 4)} {boot(d)} n {len(d)}")

print("\n== apparatus checks")
ok3 = all(answers[a][i]["answer"] == answers["r0"][i]["answer"]
          for a in ["story", "digests", "story_digests", "r0b"]
          for i, r in answers[a].items() if r.get("reused_from") == "r0")
print("3 reused answers byte-identical to r0:", ok3)
print("4 max items in any fresh answer:", max(r.get("items", 0) for a in ARMS[1:] for r in answers[a].values()))
print("5 unscored summarization per arm:",
      {a: sum(1 for i in summ if judged[a][i]["score"] is None) for a in ARMS})
other = [i for i, r in judged["r0"].items() if r["type"] != "summarization"]
changed = {a: sum(1 for i in other if judged[a][i]["score"] != judged["r0"][i]["score"])
           for a in ["story", "digests", "story_digests", "r0b"]}
print("non-summary judgements that differ from r0 (must be 0):", changed)

mem = load("memory.jsonl")
print("\n== build")
print("rows", len(mem), "failed", sum(1 for r in mem if r.get("failed")),
      "compressions", sum(1 for r in mem if "compress" in r),
      "compress_failed", sum(1 for r in mem if r.get("compress_failed")))
fails = [r.get("failed", "")[:90] for r in mem if r.get("failed")]
print("failure heads:", sorted(set(fails))[:4])
last = {}
for r in mem:
    last[r["conversation"]] = r if r["chunk"] >= last.get(r["conversation"], {"chunk": -1})["chunk"] else last[r["conversation"]]
w = sorted(len(r["storyline"].split()) for r in last.values())
print("final storyline words: min", w[0], "median", statistics.median(w), "max", w[-1])
main = [r for r in mem if "usage" in r]
print("builder main call: seconds p50", statistics.median(r["seconds"] for r in main),
      "p90", round(statistics.quantiles([r["seconds"] for r in main], n=10)[-1], 2),
      "completion tokens p50", statistics.median(r["usage"]["completion_tokens"] for r in main))
comp = [r["compress"] for r in mem if "compress" in r]
if comp:
    print("compression: seconds p50", statistics.median(c["seconds"] for c in comp),
          "words after p50", statistics.median(c["words_after"] for c in comp),
          "still over 700 after", sum(1 for c in comp if c["words_after"] > 700), "of", len(comp))
usd = sum(r["usage"]["prompt_tokens"] * 0.15e-6 + r["usage"]["completion_tokens"] * 0.60e-6 for r in main)
usd += sum(c["usage"]["prompt_tokens"] * 0.15e-6 + c["usage"]["completion_tokens"] * 0.60e-6 for c in comp)
print("builder spend, final build: $", round(usd, 3))

cover = load("storycover.jsonl")
if cover:
    s = [r["score"] for r in cover if r["score"] is not None]
    print("\nstoryline-alone coverage of summarization rubric:", round(statistics.fmean(s), 4), "n", len(s))
