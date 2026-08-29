# Preregistration: folder scoping and a bounded folder-affinity prior

Date: 2026-08-28   Status: predicted, not yet measured

## Question

On the memory corpus, does a folder/facet layer over the existing dense retrieval recover
governing memos that plain retrieval misses, and does a bounded folder-affinity prior do so
without losing controls?

Answerable by two counts: misses rescued out of 15, controls retained out of 31.

## What is being added

Three things, deliberately separable so a null in one does not hide a gain in another.

1. **A folder filter.** A prefix predicate on `metadata->>'file'`, which the indexer already
   writes as a root-relative posix path (`recall/index.py:748`, `recall/index.py:955`). This is a
   generalization of the existing exact-match `source` filter at `recall/store.py:1565`. It needs
   no schema change and no re-embedding.
2. **A facet filter.** The memos carry `type: feedback|project|reference|user` in frontmatter and
   recall's parser discards it, keeping only `VALIDITY_KEYS` and `recall_graph`
   (`recall/frontmatter.py:20`). Recognizing it puts an authored facet in chunk metadata. It lands
   only on the next generation build, so it is measured on a rebuilt generation or not at all.
3. **A folder-affinity prior.** Per-folder centroid = mean of that folder's chunk embeddings in
   the active generation, computed in Postgres from vectors already stored, so no embedding call.
   Affinity = cosine(query, centroid). Applied as `score' = cosine + lambda * affinity`, bounded,
   reordering only. **Nothing is pruned**, which is the point: a wrong folder guess must stay
   survivable, and the certified threshold of 0.509 for this tenant was fitted on an unfiltered
   candidate pool.

## Frozen population and snapshot

The unchanged `agent-ab-skill-001` population used by the graph-first probe: **15 known misses and
31 hit controls**, prompts and initial queries fixed, labels hidden from the serving tool. Same
read-only VPS2 snapshot, tenant `memory`, embedder `voyage:voyage-4`, retrieval profile `fast`,
`k=5`. The generation is pinned at run time and recorded in the result section, not here, because
the facet arm requires a rebuild and will therefore run on a different generation from the filter
and prior arms. **That is a confound and it is named below rather than hidden.**

## Arms

Baseline is the ordinary trusted retrieval path, unchanged.

1. `filter`: oracle folder given, hard prefix scope. This is the **ceiling probe**, not a
   shippable arm: it answers "is there any signal in the folder at all" by handing the system the
   answer it would otherwise have to infer. If the ceiling arm does not clear the bar, arms 2 and
   3 cannot, and the lane closes.
2. `facet`: hard scope to the authored `type` facet, same oracle framing.
3. `prior`: no oracle, no pruning, lambda swept over {0.05, 0.10, 0.20}.

## What I predict

Written before running anything, and deliberately low, because eleven of twelve past predictions
here were falsified in the same direction, every one too high by two to four times
(`[[i-over-predict-effect-magnitudes]]`).

| Arm | Misses rescued /15 | Controls retained /31 |
|---|---:|---:|
| `filter` (oracle ceiling) | **4** | 31 |
| `facet` (oracle ceiling) | **2** | 31 |
| `prior`, best lambda | **1** | **30** |

Mechanism metric, predicted beside the outcome: on the 15 misses, the governing memo's folder is
the query's top-1 folder by centroid cosine in **6 of 15** cases. If that number is high while
rescues are low, the folder signal is real and the scoring is wrong. If it is low, the folder
signal does not exist for this miss class and no amount of lambda tuning will produce one.

I expect lambda = 0.20 to lose at least one control, and I expect the best lambda to be 0.05 or
0.10.

## What would falsify this

- `filter` rescues 0 or 1 of 15. The folder carries no signal for this miss class even when the
  correct folder is handed over, and the whole lane closes rather than being tuned.
- `prior` loses more than 1 of the 31 controls at every lambda. A reordering prior that costs
  controls is strictly worse than nothing, since the corpus already answers those.
- The mechanism metric is at or below 3 of 15 while rescues are at or above 4. That would mean
  the gain came from somewhere other than the folder, and the arm is measuring a confound.

## What I already know

- **The binding constraint is discovery, not ranking.** The task-success A/B measured +0.154 with
  the CI crossing zero over 54 pairs; search rate was 0.532 and the memo was retrieved on 60% of
  those (`[[agent-ab-task-success-result-2026-08-22]]`). Folders sharpen a scope that is already
  known. They do not tell a session that a hazard exists. So the honest prior is small.
- **Authored retrieval surfaces rescued 0 of 14** on a neighbouring population
  (`docs/preregistrations/2026-08-27-memo-discoverability-authoring.md`). A facet is an authored
  surface. That is the single strongest reason my `facet` prediction is 2 and not 6.
- The graph-first probe set its bar at 5 of 15 rescued and 30 of 31 retained. I use the same bar
  so the two lanes are comparable.

## Confounds I can name now

- **The facet arm runs on a different generation**, because the metadata only lands on a rebuild.
  A generation change moves scores on its own. The filter and prior arms must therefore be re-run
  on the rebuilt generation before `facet` is compared against them, or the comparison is between
  two corpora rather than two arms.
- **The oracle in arms 1 and 2 is not a system.** It measures a ceiling. Reporting a ceiling as an
  arm result would be the central dishonesty available here, so both are labelled ceiling probes
  in the result section too.
- **The corpus is flat today**: 317 memo files, zero subdirectories. So the `filter` arm's folders
  must be derived from something, and whatever derives them is a second untested component. If
  they are derived from the frontmatter facet, arms 1 and 2 are not independent and the table above
  is measuring one thing twice.
- **Centroid quality tracks folder size.** A folder with three memos has a centroid that is nearly
  a memo; a folder with 200 has one that is nearly the corpus mean and carries almost no
  discrimination. Folder size must be reported beside affinity, or a null will be read as "folders
  do not work" when it means "these folders were the wrong size".

## Observed results

**Status: measured 2026-08-29, and the FOLDER arm is VOID because it measured a bug.**

Snapshot: tenant `memory`, generation `gen_ec1e01b1c1324959a8a140ad65a17ed6`, embedder
`voyage:voyage-4`, k=5, all 138 responses `trust_state=trusted`. Population digest verified equal
to the registered `a87c7eeb...`. Runner: `scripts/run_scope_arms_batch.py`; artifact
`results/scope_arms/scope-arms-20260829.json`.

| arm | rescued of 15 misses | retained of 31 controls |
|---|---:|---:|
| baseline (no scope) | 0 | 19 |
| filter (oracle folder) | 0 | **0** |
| facet (oracle facet) | 0 | 23 |

**1. The folder arm is void, and the finding is a defect rather than a result.** 0 of 31 controls
retained cannot be corpus drift. `metadata->>'file'` on the PRODUCTION build path holds only the
basename (`msys-mangles-slash-patterns.md`), while the `Indexer` path stores the root-relative
path. The folder dimension is therefore empty in production: `folder=recall` matches nothing and
returns zero hits, which the tool cannot distinguish from a corpus with no answer. This measured
the bug, not the idea, and the folder ceiling remains unknown.

**2. The apparatus is degraded and no rescue verdict should be read from it.** The baseline
reproduces only 19 of 31 controls, against a population whose controls were all hits when frozen
on 2026-08-26. The corpus has moved through several generations since (r43 to r51, a full
re-derivation, and new memos), so the frozen labels no longer describe it. The registration named
this confound for the facet arm; it turns out to bind the whole run.

**3. What the facet arm does suggest, held loosely.** 0 of 15 rescues, which matches the
prediction of 2 no better than the neighbouring authored-surface probe's 0 of 14, and is the third
independent null for "make the right memo easier to find by labelling it". Retention rose from 19
to 23, which is the one positive signal here: narrowing the pool let four gold memos surface that
were crowded out of the top five unscoped. That is the prior's intended mechanism achieved by hard
filtering, and it is worth a properly-powered look once the apparatus is repaired.

**4. Gap against prediction.** I predicted 4 / 2 / 1 rescues for filter / facet / prior. Measured
0 for the two arms that ran. Consistent with `[[i-over-predict-effect-magnitudes]]`, which now has
a thirteenth instance.

**5. The prior arm was not run.** `ScopePrior` is a retriever constructor argument with no MCP
surface, so it cannot be exercised over this transport. Recorded in the artifact rather than
quietly omitted.

**What would make a rerun meaningful:** repair the production `file` value (or key the folder
predicate off `source_uri`), then re-freeze a population against a current generation. Reusing a
population frozen against a corpus three generations old is what produced the 19-of-31 baseline.

