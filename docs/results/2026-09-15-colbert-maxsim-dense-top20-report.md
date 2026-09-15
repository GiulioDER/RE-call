# ColBERT MaxSim dense top 20 selector result

Measured 2026-09-15 on the consumed 30 row empty base development cohort. The result is complete
and failed the frozen promotion gate. The preregistered plan is
[2026-09-15-colbert-maxsim-dense-top20.md](../preregistrations/2026-09-15-colbert-maxsim-dense-top20.md).

## Result

The apparatus reproduced the original dense baseline exactly. Gold source reachability at ranks
1, 3, 5, 10, and 20 was 6, 6, 7, 8, and 10. Exact span reachability was 3, 3, 5, 8, and 10. The
anchor safety gate admitted 14 answerable rows and zero controls.

ColBERT MaxSim reordered only each eligible row's original dense top 20. Gold source reachability
became 2, 7, 9, 9, and 10 at ranks 1, 3, 5, 10, and 20. Exact span reachability became 1, 6, 8,
8, and 10. Candidate membership remained identical for all 30 rows.

MaxSim changed rank one on eight eligible answerable rows. None of the eight new rank one chunks
contained the exact span, and none came from a gold source. Exact and gold source precision among
changed rank one candidates were therefore both 0.0%. Minimum exact rank improved for six
answerable rows, tied for six, and worsened for three.

Verdict: `STOP_GENERIC_RERANKING_ON_THIS_COHORT`.

The useful signal is below rank one. Exact top 3 improved from three rows to six and exact top 5
improved from five rows to eight, but exact rank one fell from three rows to one. Token level
MaxSim is a useful mid rank organizer on this cohort, not a safe first result selector. Candidate
movement cannot substitute for answer bearing evidence at the position the agent sees first.

Do not tune the checkpoint, pool width, score transform, dense and MaxSim mixture, eligibility
gate, or promotion thresholds on these consumed rows. Do not deploy this ordering and do not spend
a fresh holdout on it.

## Integrity and execution

The fixed collection contained 14 eligible queries, 239 unique documents, and 280 query document
pairs. The scorer completed all 280 pairs with `colbert-ir/colbertv2.0`, MIT licence, through
FastEmbed 0.8.0. The existing offload parity validator recomputed five pools with the live
`LateInteractionReranker` and returned `MATCH`, maximum score delta 0.0, no failures, and no deep
ties.

All model embedding and scoring ran on VPS2 under the shared embedding lock, an 8 GB memory cap,
zero swap, a 250% CPU quota, and lowered priority. The first scorer launch failed before scoring
because `/tmp/fastembed_cache` was owned by root. It wrote no score artifact. Repeating the same
registered arm with `TMPDIR` confined to the private experiment directory repaired only the cache
location. No model or experimental parameter changed.

Collection initially stopped on its first row before writing an artifact because the source
admission audit contains chunk text while the retrieval leg audit contains dense cosine. The
harness was repaired to join those two audited records by chunk ID. A new regression test was
proved red by substituting rank for cosine, then green after restoration. Collection restarted
from row one at source commit `23307a1f`.

Private artifacts remain outside the repository under
`C:\Users\gde00\.codex\evals\query-anchor-2026-09-15\colbert-dense-selector`. The complete private
result SHA256 is `ea03b5b13ea55dcf2d8211cda8b77bec01bbd95f85e0eb63cc86d33ed8331ee8`. The collection SHA256
is `6cc9a85025a91bbea7841766bbf5a90200458ed8aa638bf0215c000f79ac099f`. The score artifact SHA256
is `7bdd8891472a3c8f32f24b60de984bab75e5d7b39037f927497868925e9a697b`.

Reproduce collection from the experiment worktree with:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/query-anchor-f72e8951'
$env:RECALL_SOURCE_COMMIT='23307a1f'
$env:RECALL_POLICY_COMMIT='34618bf5'
python -u scripts/run_colbert_dense_selector_dev.py collect --query-pool docs/preregistrations/2026-09-14-query-anchor-spare-slot-pool.json --holdout-result docs/results/2026-09-14-query-anchor-holdout.json --artifact docs/results/2026-09-13-source-conditioning-model.json --output-dir C:\Users\gde00\.codex\evals\query-anchor-2026-09-15\colbert-dense-selector --generation-id gen_2ccf2130f6c64d99a11a6bcb6f929dd8 --calibration-id cal_e50dac493112488ea5e7cf79d86c0099 --pipeline-fingerprint 57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86 --corpus-fingerprint f737fd1ffcdb9275ceccc4092a5dd7cc873f374936d2d9d60e06618c3bec6312
```

Validate the harness with:

```powershell
python -m pytest tests/test_colbert_dense_selector_dev.py tests/test_late_interaction_rerank.py tests/test_bench_late_interaction.py -q
python -m ruff check scripts/run_colbert_dense_selector_dev.py tests/test_colbert_dense_selector_dev.py
python -m mypy --follow-imports=skip scripts/run_colbert_dense_selector_dev.py
```

## Next experiment

The next highest ROI lane is a span grounded extractive reader with an explicit null outcome over
the fixed original dense top 20. Ten of 15 answerable rows already contain exact evidence in that
pool, while both generic cross encoder and token level MaxSim selection have failed rank one
precision. The reader must identify and quote a supported answer span from one candidate or return
null. It should be evaluated first on this consumed cohort with matched controls, and it must not
generate an unconstrained answer. A separate preregistration must define span support, null
behavior, candidate selection, and the promotion gate before any reader output is produced.
