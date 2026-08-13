# MTRAG Tasks B and C — results as of 2026-08-08

> 🔴 **SUPERSEDED IN PART — read `CORRECTION-idk-conditioning-2026-08-09.md` FIRST.**
> Every harmonic mean below compares our RAW metrics against the baselines' IDK-CONDITIONED ones,
> because the official scorer reads a lower-case `answerability` key that the release files spell
> `Answerability`, so the conditioning silently never ran on our data. The corrected figures, the
> proof that the baselines are conditioned, and the re-scored pre-registration are in that file.
> The prose findings here still hold; the NUMBERS do not.

**Prior work** (searched this session, per CLAUDE.md; recorded rather than repeated):
`docs_search(source_type='memory', "MTRAG Task B Task C generation RAG answer quality GPT-4o judge
RL_F RB_llm RB_alg harmonic mean")` and
`docs_search(source_type='memory', "over-abstention false refusal prompt instruction says I don't
know on answerable questions cost")`. Binding results:
[[project-recall-mtrag-rbalg-probe-2026-08-05]] (the anchored-lift constraint, and a "models are
verbose" thesis already FALSIFIED — which is why no RB_alg gain is claimed from the 150-word cap),
[[project-recall-mtrag-retrieval-coverage-bottleneck-2026-08-06]], and
[[project-recall-abstention-bounded-domain-2026-07-24]] (⚠️ that one is RE-call's RETRIEVAL-side
abstention layer, a different component from the generator prompt studied here).

Six generation runs, 842 tasks each, `openai/gpt-4o` via OpenRouter, zero failures in all six.
A 2x2 by design: **{gold, benchmark-retrieved, RE-call-retrieved} contexts × {abstain, official}
prompt**, so the prompt effect and the retrieval effect can be separated instead of confounded.

⚠️ **Read the comparison caveat first.** Recomputing the published table from the release runs
+0.018 to +0.043 HIGH on every model ([[project-recall-mtrag-rbalg-probe-2026-08-05]]): the
aggregation formula is right, the instance set is not the published one. Every baseline below is
therefore **recomputed here with the same code that computes our row**, and the comparison is an
ANCHORED LIFT. None of these numbers may be quoted against the published leaderboard directly.

## Scored so far

| run | contexts | prompt | RL_F | RB_llm | RB_alg | **harmonic** |
|---|---|---|---|---|---|---|
| Task B | gold | `abstain` | 0.7011 | 0.6283 | 0.4117 | **0.5508** |
| **Task B** | **gold** | **`official`** | **0.7756** | **0.7696** | **0.4432** | **0.6192** |
| Task C | benchmark | `abstain` | 0.7369 | 0.6024 | 0.3999 | **0.5437** |

The `abstain` Task B RL_F is over 830 rows, not 842: 12 RAGAS `TimeoutError`s. Complete-case gives
0.5502 against 0.5508, a difference of 0.0006, so they are missing at random. The `official` run
has **842 on every metric**, no timeouts, no caveat.

### 🎯 The official prompt closed the gap almost entirely

| Task B, gold contexts | harmonic | vs their gpt-4o (0.6208) |
|---|---|---|
| ours, `abstain` | 0.5508 | −0.0700 |
| **ours, `official`** | **0.6192** | **−0.0016** |

That moves us from **9th of 10 to 3rd** in the recomputed Task B table, behind only
`llama-3.1-405b` (0.6277) and their `gpt-4o` (0.6208). The entire deficit was the prompt.

🔑 **What this buys for Task C**: our generation is now demonstrably at parity with the baselines,
so a Task C result reads as a statement about RETRIEVAL rather than about harness quality. It also
removes the excuse — if RE-call's contexts do not beat the benchmark's, there is no prompt to blame.

### Pre-registration scored: 2 of 4 correct

Registered in `PREREGISTRATION-official-prompt-2026-08-08.md` before any scored output existed.

| prediction | registered | actual | verdict |
|---|---|---|---|
| harmonic mean | 0.57 – 0.62 (point 0.595) | **0.6192** | ✅ in band, at the top |
| RL_F | 0.75 – 0.79 | 0.7756 | ✅ |
| RB_alg | 0.43 – 0.46 | 0.4432 | ✅ |
| RB_llm | 0.66 – 0.70 | **0.7696** | ❌ above band |
| false abstentions | < 40 / 709 | **61 / 709** | ❌ |
| ordering: largest gain RL_F | RL_F > RB_llm > RB_alg | **RB_llm +0.1413** > RL_F +0.0745 > RB_alg +0.0315 | ❌ |

🔑 **The diagnosis was right about what was broken and wrong about how it would repair.** I reasoned
that fixing false abstentions would mainly lift faithfulness (RL_F). It mainly lifted RB_llm, which
in hindsight follows: RB_llm carried the largest deficit (−0.133) and judges appropriateness and
completeness, which is precisely what a refusal destroys. The ordering prediction was chosen as the
one hardest to hit by luck, and it is the one that failed.

⚠️ This also retires the alarm I raised when correct abstentions fell 89.1% → 63.6%: the net was
strongly positive regardless, and I over-weighted that signal when I saw it in isolation.

### Recomputed baselines — Task B (gold contexts)

| model | RL_F | RB_llm | RB_alg | harmonic |
|---|---|---|---|---|
| *target (human reference, not a system)* | 0.8690 | 0.9519 | 0.8772 | *0.8978* |
| llama-3.1-405b-instruct | 0.7543 | 0.7416 | 0.4751 | 0.6277 |
| gpt-4o | 0.7576 | 0.7616 | 0.4547 | 0.6208 |
| qwen-2.5-72b-instruct | 0.7239 | 0.7473 | 0.4435 | 0.6031 |
| c4ai-command-r-plus | 0.7595 | 0.6938 | 0.4435 | 0.5985 |
| gpt-4o-mini | 0.7156 | 0.7499 | 0.4331 | 0.5953 |
| qwen-2.5-7b-instruct | 0.6787 | 0.7189 | 0.4358 | 0.5815 |
| llama-3.1-70b-instruct | 0.6962 | 0.6632 | 0.4422 | 0.5763 |
| mixtral_8x22b_instruct | 0.6174 | 0.6978 | 0.4202 | 0.5522 |
| llama-3.1-8b-instruct | 0.5539 | 0.5936 | 0.3706 | 0.4848 |

**Our two rows placed in that table:** `official` **0.6192** sits 3rd, between their `gpt-4o`
(0.6208) and `qwen-2.5-72b` (0.6031). `abstain` **0.5508** sat 9th, between `mixtral_8x22b`
(0.5522) and `llama-3.1-8b` (0.4848). Same model, same contexts, same tasks — six places apart on
the prompt alone.

### Recomputed baselines — Task C (retrieved contexts)

| model | RL_F | RB_llm | RB_alg | harmonic |
|---|---|---|---|---|
| *target (human reference, not a system)* | 0.6478 | 0.9467 | 0.8513 | *0.7947* |
| llama-3.1-405b-instruct | 0.7156 | 0.6826 | 0.4151 | 0.5691 |
| qwen-2.5-72b-instruct | 0.7119 | 0.6995 | 0.4002 | 0.5625 |
| gpt-4o | 0.7076 | 0.7006 | 0.3960 | 0.5591 |
| c4ai-command-r-plus | 0.7246 | 0.6370 | 0.3996 | 0.5502 |
| qwen-2.5-7b-instruct | 0.6715 | 0.6805 | 0.3924 | 0.5447 |
| **ours, benchmark ctx (`abstain`)** | 0.7369 | 0.6024 | 0.3999 | **0.5437** |
| gpt-4o-mini | 0.6801 | 0.6869 | 0.3859 | 0.5437 |
| llama-3.1-70b-instruct | 0.6764 | 0.6304 | 0.3978 | 0.5378 |
| mixtral_8x22b_instruct | 0.6157 | 0.6560 | 0.3865 | 0.5230 |
| llama-3.1-8b-instruct | 0.5587 | 0.5873 | 0.3475 | 0.4710 |

## 🔑 The finding so far: my prompt, not RE-call

Task B holds retrieval perfect, so **RE-call is not in that number at all**. Ours vs the
benchmark's own `gpt-4o` — same task, same gold contexts, same model — is therefore a clean read on
our harness:

| task | ours | their gpt-4o | gap |
|---|---|---|---|
| B (gold) | 0.5508 | 0.6208 | **−0.070** |
| C (benchmark ctx) | 0.5437 | 0.5591 | **−0.015** |

Cause, measured: `abstain` produced **83 false abstentions on 709 ANSWERABLE tasks**, and a false
abstention scores near zero (RL_F 0.0726 against 0.8901 when it answered).

🔑 **The handicap costs 4.6x more on Task B than on Task C**, from the same prompt. With gold
contexts the answer is nearly always present, so every false abstention is pure loss; with
retrieved contexts the passages genuinely often fall short, so abstaining is right more often.

### The prompt was published, and I said it was not

I asserted "MTRAG publishes no generation script, so the baselines' prompt is unknown" after one
grep. It is in the paper, arXiv 2501.03468 **Appendix D.2 "Model invocation"**. `abstain` differs
from it in three ways, not the one I first diagnosed:

1. no **150-word limit**;
2. `"say that you do not know rather than guessing"` instead of the exact string
   `"I do not have specific information"`;
3. different ordering and passage framing.

⛔ Also NOT the prompt in `conversations/conversations.json` — that one built the conversation
dataset with mixtral-8x7b and has no length limit. Taking the first prompt found in the repo would
have been the same error one level quieter.

### The behavioural mechanism, measured

Scored above; this is the behaviour underneath it, with a single consistent detector across both
files:

| | false-abstain on ANSWERABLE | UNANSWERABLE abstain | mean words |
|---|---|---|---|
| `abstain` | 89 / 709 (12.6%) | 49 / 55 (89.1%) | 51.6 |
| `official` | **61 / 709 (8.6%)** | **35 / 55 (63.6%)** | **79.9** |

The official prompt abstains **less everywhere** — it did not learn to abstain better, it became
less willing to abstain at all. Correct abstentions fell alongside false ones. When I saw this
before the scores I called it expensive, because a correct IDK is worth exactly 1.0 on all three
metrics; the scores say the trade was strongly positive anyway. 🔑 **A mechanism moving the wrong
way on one sub-population does not predict the aggregate**, and I should not have raised it as a
concern before the number existed.

Answers also got longer (51.6 → 79.9 words) because `abstain` said "concisely" and D.2 does not,
despite D.2 carrying a 150-word cap that binds on only 2.1% of answers.

⚠️ The detector here also matches "I do not have specific information", so it reads the old file as
89 where the pre-registered figure said 83. Old-vs-new is internally consistent; the registered
threshold was anchored to a narrower detector. A measurement-definition slip, and the third time in
one session that string-matching abstention misled me. ⛔ **Never quote a regex-derived abstention
rate** — every conclusion above rests on the official judge instead.

## MTRAGEval compliance

MTRAGEval withholds question type, answerability and multi-turn type at evaluation; only the
corpus domain is provided. Audited:

| path | status |
|---|---|
| prompt sent to the model | leak-free, verified across all 842 |
| submission rows | `task_id`, `Collection`, `input`, `contexts`, `predictions` only |
| the 65-task retrieval fix | selected by **set difference on task ids**, not labels |
| domain routing | uses `Collection`, which is provided |

The withheld labels ARE used post-hoc for stratification and diagnosis, which is what they are for
once a run is over. ⚠️ One honest caveat: the decision to change the prompt was *motivated* by a
label-stratified analysis. The prompt adopted is the paper's own, justified independently, and the
`neutral` variant that was derived from label stratification is not used — but no dev-set gain that
depended on label access should be claimed to transfer to the sealed set.

## Truncation audit (added 2026-08-13)

Every run above sent `--max-tokens 512` through a generator that never read `finish_reason`, so a
completion the ceiling cut off came back as an ordinary string, was written to the submission and
was judged as if the system had produced it. The code path is fixed (`generate_one` now raises
`CompletionTruncated`, and the per-task quarantine keeps such a task out of the submission), but
the fix is forward-only: these runs predate it, and **the stop reason was never recorded**, so for
the artifacts it has to be reconstructed.

Reconstructed by re-tokenising every stored answer with the generator's own encoding (gpt-4o,
`o200k_base`). This is close to exact rather than a proxy: a completion stopped by the ceiling
carries exactly 512 completion tokens, and the only edge case is the trailing whitespace token that
`.strip()` removes.

| run | rows | mean | p95 | max | at/over 512 |
|---|---|---|---|---|---|
| `taskb` (gold, abstain) | 842 | 64.4 | 156 | 330 | **0** |
| `taskb_official` (gold, official) | 842 | 99.1 | 169 | 239 | **0** |
| `taskc_benchmark` (benchmark-retrieved, abstain) | 842 | 66.4 | 157 | 304 | **0** |
| `taskc_recall` (RE-call-retrieved, abstain) | 842 | 78.0 | 194 | 402 | **0** |
| `taskc_benchmark_official` (benchmark-retrieved, official) | 842 | 97.5 | 169 | 233 | **0** |
| `taskc_recall_official` (RE-call-retrieved, official) | 842 | 101.2 | 176 | 311 | **0** |

**All six runs are clean.** Nothing came within 12 tokens of the ceiling. The longest answer
anywhere is 402 tokens, 110 below the limit, and the independent punctuation check agrees: the three
answers of 5,052 that end without terminal punctuation are all short ones. **No number in this
document needs a truncation caveat.**

### ✅ The gap, and how it closed

The two official-prompt Task C runs were UNVERIFIED for a while, and they are the pair this document
calls the comparison that actually measures RE-call, so the gap was on the worst possible pair. It
is recorded here rather than quietly deleted, because how it closed is the reusable part.

I reported that the payloads were "not on this disk" after searching for them. That was wrong twice
over, and both mistakes are ordinary ones:

* The search required `mtrag` in the FILENAME. The pack's files are named `taskb_*` and `taskc_*`,
  so it could not have found them however long it ran. ⚠️ A search that cannot match is not
  evidence of absence, and it reports exactly like one that found nothing.
* It never occurred to me to look in git HISTORY for files a commit had deleted. `runs/README.md`
  says the payloads live "outside the source tree", which is true of the current tree and hides
  that they were committed before `f83a330` ("Externalize raw benchmark artifacts") removed them.
  **The archive was in the repository the whole time.**

Restored with `git show f83a330^:results/mtrag_generation/runs/<file>` and verified against
`runs/SHA256SUMS.txt`: **10 of 10 files match**, which is the check that separates the real archive
from a plausible lookalike, and is what that file exists for. All five layers of each run
(`predictions`, `scoring`, `scored`, `algorithmic`, `fixed`) scan identically, as they should, since
they carry the same predictions.

Superseded, and kept because the reasoning was published: I argued the pair was probably clean
because its nearest neighbours were, and because the official prompt produced the tightest length
distribution. That inference held. ⚠️ It was still an inference, it was labelled as one, and it is
now a measurement. The margins are wider than the ones it reasoned from: 233 and 311 tokens against
a 512 ceiling.
#### The command, and why its flags are not ceremony

The scan is `scripts/scan_truncation.py`. It reads `predictions[].text` and needs nothing else.
This is the invocation that produced the two rows above, against the restored pack:

      python results/mtrag_generation/scripts/scan_truncation.py --expect taskc_benchmark_official,taskc_recall_official --expect-rows 842 <restored>/taskc_*_official.predictions.jsonl

  An unexpanded wildcard is globbed by the script itself, because PowerShell does not expand
  arguments and the operator would otherwise be told the pattern is ABSENT. ⛔ `--expect` is then
  REQUIRED, and it names RUNS rather than counting files. That distinction is the whole guard:
  each run restores five `.jsonl` layers (`runs/SHA256SUMS.txt`), so a count of two is
  unsatisfiable against a correct restore of both runs, while two layers of ONE run satisfy it
  perfectly and the other run is never named. A count cannot say "these two runs"; only names can.

  `--expect` is required on EVERY run, not only when the script expands a wildcard. Under bash the
  shell expands the pattern first, so a gate conditioned on that would be switched off by the
  operator's choice of shell, which is the same defect one layer out. Nothing here certifies what
  nobody claimed. Every report also ends with `runs covered: ...`, so the universe a CLEAN refers
  to is legible in the report itself rather than reconstructed from a command line.

  ⚠️ `--expect-rows 842` is required too, and for the same reason. Coverage matches on the
  BASENAME, so without a size claim a one-row file called
  `taskc_recall_official.predictions.jsonl` certifies that run CLEAN while 841 answers go
  unmeasured; a name is not a size. 842 is the `tasks` field in
  `manifests/taskc_recall_official.predictions.jsonl.manifest.json`. Because that field counts
  TASKS and a file holds LINES, the scan also requires the distinct `task_id` count to equal the
  row count, so a file padded with repeated rows cannot reach the number. `--expect-rows
  unclaimed` is the explicit opt-out, and the report then prints `rows expected unclaimed`, so a
  CLEAN can never be mistaken for a size claim nobody made. (This flag was optional for exactly
  one commit, which left the false clean it was written to close reachable by not typing it. That
  is why the opt-out is a token rather than silence.)

  What survives even with both flags, so a CLEAN is quoted for what it is:

  * A file carrying the right name and the right row count but copied from somewhere else is taken
    at face value, because rows record `task_id` and no run identity.
  * Measurement is over `predictions[].text`. A 520-token answer stored under a sibling key in a
    prediction object that also carries a readable `text` is not inspected, because in every file
    this has been pointed at a prediction's other keys are metadata.

  A CLEAN therefore says: **these named files were read in full, and nothing in their
  `predictions[].text` reached the ceiling the command stated.** It says they are the size their
  manifest claims ONLY when the command passed a number to `--expect-rows`; under
  `--expect-rows unclaimed` the verdict word is identical and the header line is the only place
  the disclaimer appears, so read the header before quoting a CLEAN. The stated ceiling is
  refused if it is outside 1 to 65,536, but within that range it is the operator's claim, not a
  fact the scan can check: a ceiling higher than the run actually sent will report no findings.

  🔑 It reports **CLEAN only when every file was fully read and every prediction in every row was
  measured**, and exits non-zero otherwise. An absent file, an empty one, a line that will not
  parse, a repeated JSON key, a row with no `predictions`, a prediction this tool cannot read even
  when a readable one sits beside it, or a scan that raised: none of these can end as CLEAN, they
  end as UNVERIFIED. A truncation at any prediction index, or a count one token under the ceiling,
  is not clean either: a finding outranks an unread condition and ends as TRUNCATION FOUND or NEAR
  THE CEILING, so that the report names what it found rather than hiding it behind the weaker
  label. A byte-order mark is the one case handled the other way round, deliberately: the file is
  opened as `utf-8-sig` so the mark is consumed and the first row is READ, rather than dropped as
  unparseable and the file taken for a shorter one.

  That list is not decoration. Every entry on it was a way an earlier version of this script
  silently reported "0 truncated" for rows it never read, each found by audit, twice over, before
  the script had been pointed at anything that mattered.

  Its detector is live rather than assumed: run it with `--ceiling 200` against `taskb` and it
  reports TRUNCATION FOUND, so the zero at 512 is a measurement and not a dead check. That claim is
  itself checked, by `scripts/check_scan_truncation.py`: a case per false clean above, plus
  assertions on the published statistics, the tokenizer and the default ceiling, plus an optional
  liveness run against a real corpus. It prints its own tally, which is the number to quote rather
  than one written here, since a count in prose drifts the moment a case is added. Run it after any
  edit to the scanner:

      python results/mtrag_generation/scripts/check_scan_truncation.py --corpus <a real .jsonl>

  Each case asserts the per-file VERDICT, not just a non-zero exit, because several inputs exit
  non-zero either way: a byte-order mark that hides an over-ceiling first row exits 1 as
  UNVERIFIED, so a case demanding only "not clean" passes whether the row was read or dropped.
  Every mutation of the scanner run against the matrix turned at least one case red, including the
  ones a weaker earlier version let through: stopping after two lines, shrinking the near band to
  one token, drifting the default tokenizer or the default ceiling, letting a readable prediction
  vouch for an unreadable sibling, and accepting a `--expect` that names no run at all.

#### Provenance

Every one of the six rows above was measured on a file restored from `f83a330^` and matched
against `runs/SHA256SUMS.txt` before it was read: **6 of 6 predictions layers verified, 0
mismatched**, and for the two official Task C runs all five layers each, 10 of 10.

An earlier version of this section carried a caveat that the four abstain-prompt and Task B rows
came from leftover intermediates in a temp directory rather than the checksummed archive, and could
not be matched against it because re-gzipping does not reproduce a byte-identical container. That
caveat is retired, and usefully so: re-measuring from the verified archive reproduced those four
rows exactly, to every digit of mean, p50, p95 and max. The leftovers were faithful, which was
worth confirming rather than assuming.

## Pending

Awaiting one GPU session (three algorithmic passes, ~8 min each) then the LLM judge on each:
`taskb_official` (judge running), `taskc_recall`, `taskc_benchmark_official`,
`taskc_recall_official`.

🔑 **The comparison that actually measures RE-call** is the fixed-prompt pair
`taskc_recall_official` vs `taskc_benchmark_official`: same harness, same prompt, same model, only
the contexts differ. It needs no baseline table to be meaningful, and it is the only row in this
document where RE-call is the variable.

✅ Both halves of that pair were the two runs the truncation audit could not reach, and both are now
measured clean: 233 and 311 tokens at the longest, against a 512 ceiling. Nothing in that comparison
is waiting on a truncation check.
