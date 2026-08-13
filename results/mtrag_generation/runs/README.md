# MTRAG Tasks B and C — archived run pack

> **Prior work.** This file archives runs that already exist; it proposes no new measurement, so no
> fresh search was run for it. The searches that governed these runs are recorded where the claims
> live: `../RESULTS.md` (Task B/C generation, over-abstention) and
> `../CORRECTION-idk-conditioning-2026-08-09.md`, whose search surfaced
> `[[reference-mtrag-withheld-metadata-two-shapes-2026-08-08]]` — a memo written earlier the same
> day that stated the exact fact predicting the conditioning bug, and which I had not applied.
> Governing memos: `[[project-recall-mtrag-taskbc-prompt-finding-2026-08-08]]` (numbers superseded,
> diagnosis stands), `[[reference-mtrag-official-judge-is-gpt4o-mini-2026-08-08]]`,
> `[[incident-mtrag-raw-vs-conditioned-metric-comparison-2026-08-09]]`.

Every artifact from the six generation runs is listed in `SHA256SUMS.txt`. The raw payloads and
logs are not in the CURRENT tree; restore them beside this README before replaying the commands
below. The files were written on the machine that produced them and checked again after transfer
(48/48 clean).

## 🔑 Restoring them: they are in this repository's history

They were committed once, and `f83a330` ("Externalize raw benchmark artifacts") removed them, so
every blob is still reachable from its parent. No bucket, no release asset, no external copy is
needed:

    git show f83a330^:results/mtrag_generation/runs/taskc_recall_official.predictions.jsonl.gz > taskc_recall_official.predictions.jsonl.gz
    git ls-tree --name-only f83a330^ results/mtrag_generation/runs/   # everything available

Then verify what you restored against `SHA256SUMS.txt` before reading it, which is what that file
is for. Confirmed on 2026-08-13: all six `predictions` layers and all five layers of both official
Task C runs restore and match, and the truncation scan over them is recorded in `../RESULTS.md`.

⚠️ This sentence exists because its absence cost real time. Reading "archived outside the source
tree", I searched three drives for a pack that was in `git log` the whole way, and reported it
missing. **"Not in the working tree" and "not in the repository" are different claims**, and only
the first one was ever true here.

## Why the payloads are not committed

The first public archive committed the gzipped payload pack to git because the files had no durable
home. That fixed the immediate loss risk but made the repository feel like a lab archive instead of
a library.

The source tree now keeps the index, checksums, run notes, and derived summaries. The bulky rows and
logs belong in release assets or a dataset bucket. A restored archive is still verifiable against
`SHA256SUMS.txt`, but the checksum file is not the archive itself.

**A hash proves an artifact is unchanged. It does not keep the artifact alive.** An artifact whose
only copy is a temp directory is not archived, however well its provenance is documented.

That principle stands, and history turned out to be the durable copy it was reaching for: removing
the files from the tree did not remove them from the repository, so `git clone` still carries them
and the loss risk this section worried about never materialised. If they are ever moved to a bucket
or a release asset for real, say so here WITH the location, and keep the `f83a330^` path above as
the fallback.

## The six runs

A 2×3 design: `{gold, benchmark-retrieved, RE-call-retrieved}` contexts × `{abstain, official}`
prompt, so the prompt effect and the retrieval effect separate instead of confounding. Generator is
`openai/gpt-4o` via OpenRouter on all six. Zero task failures in all six.

| stem | task | contexts | prompt |
|---|---|---|---|
| `taskb` | B | gold | abstain |
| `taskb_official` | B | gold | official (paper Appendix D.2) |
| `taskc_benchmark` | C | benchmark's own ELSER | abstain |
| `taskc_benchmark_official` | C | benchmark's own ELSER | official |
| `taskc_recall` | C | RE-call | abstain |
| `taskc_recall_official` | C | RE-call | official |

## The file layers, in the order they were produced

| suffix | what it is | cost to regenerate |
|---|---|---|
| `.predictions.jsonl` | raw generator output | paid API |
| `.scoring.jsonl` | predictions joined to the release gold | free, deterministic |
| `.algorithmic.jsonl` | + RougeL, BertScore recall / K-precision | GPU pass |
| `.scored.jsonl` | + RAGAS faithfulness and the RADBench judge | paid API |
| `.fixed.jsonl` | **+ corrected IDK conditioning. The numbers come from these.** | free, `fix_idk_conditioning.py` |
| `judge_*.log` | the judge's own stderr | — |

⚠️ **Read `.fixed.jsonl`, not `.scored.jsonl`.** The official scorer reads a lower-case
`answerability` key; the release files spell it `Answerability`. The key was never found, the
conditioning silently never ran, and every `.scored.jsonl` here therefore carries **raw** metrics
where the published baselines carry **conditioned** ones. `judge_taskb.log.gz` contains the
evidence: 2,526 `Error: answerability is None` lines and zero label matches. Full account in
`../CORRECTION-idk-conditioning-2026-08-09.md`.

Reproduce the headline number straight from the pack:

```bash
python -c "
import gzip,json,statistics
rows=[json.loads(l) for l in gzip.open('taskb_official.fixed.jsonl.gz','rt',encoding='utf-8') if l.strip()]
f=lambda v: v[0] if isinstance(v,list) else v
m=[statistics.fmean([f(r['metrics'][k+'_idk_underspecified']) for r in rows]) for k in ('RL_F','RB_llm','RB_agg')]
print(len(rows), [round(x,4) for x in m], 'harmonic', round(len(m)/sum(1/x for x in m),4))
"
```

Expect `842 [0.7793, 0.7285, 0.4573] harmonic 0.6195`.

⚠️ **Denominators are not always equal across the three metrics.** In `taskb.fixed` (Task B,
`abstain` prompt) twelve RAGAS `TimeoutError`s leave `RL_F` null, so its leg averages **832** rows
against 842 for the other two: 0.5913 all-rows, **0.5902** complete-case. Every other run is 842 on
all three. Quote the per-metric n with the number.

📎 The `.fixed.jsonl` files here were written by an earlier revision of
`../fix_idk_conditioning.py` that skipped a metric when its raw value was null instead of
recomputing it. A bug audit on 2026-08-09 caught it, and re-running the corrected script over all
six `.scored` files reproduces these artifacts with **zero** changed values and all six harmonic
means identical to four decimals. The archived files are therefore correct as they stand and have
not been regenerated; the skip only mattered for a null-raw row that was also UNANSWERABLE or
CONVERSATIONAL, and all twelve null rows are ANSWERABLE.

## Attribution

These files embed content from **MTRAG** (IBM), used under **Apache-2.0**: the gold `targets`, the
`contexts`, and the task metadata columns. Source: <https://github.com/IBM/mt-rag-benchmark>.

> Katsis, Rosenthal, Fadnis, Gunasekara, Lee, Popa, Shah, Zhu, Contractor, Danilevsky.
> *MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation
> Systems.* TACL 13:784–808, 2025. <https://doi.org/10.1162/TACL.a.19>

Only the `predictions` field and the `metrics` block are ours. Everything else in these rows is
MTRAG's, reproduced so a reader can verify a metric without reassembling the join.

⚠️ This is MTRAG's **public dev set**. It is not the sealed evaluation set, and nothing here
discloses withheld data.
