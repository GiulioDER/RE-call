# Focused evidence application skill: official-018

Measured 2026-09-17 under the frozen
[`focused evidence application skill preregistration`](../preregistrations/2026-09-17-focused-evidence-application-skill.md).

## Verdict

`STOP`.

The RE-call API and broker setup were repaired and smoke-tested before the run. The adjudication
receipt verifies, the preflight exposed all 22 tools, and 40 of the 50 cells passed the admission
gate. The skill did not improve quality. Against the contemporaneous official-012 protocol,
treatment lost six net paired cells and applied one wrong fact where control applied none.

## Primary endpoint

| Arm | Successes | Admitted cells | Rate |
| --- | ---: | ---: | ---: |
| Control: `recall_graph_fulltools_protocol` | 34 | 40 | 85.0% |
| Treatment: `recall_graph_fulltools` | 28 | 40 | 70.0% |

The paired breakdown was 26 both-success, 2 treatment-only, 8 control-only, and 4 both-fail.
Net wins were `-6`, the rate difference was `-15.0` percentage points, and the exact McNemar
p-value was `0.109375`.

## Mechanism and safety

The treatment called `recall_search` before any non-memory tool in 11 of 40 admitted sessions
(27.5%). It used search in 16 of 40 sessions (40.0%). The control figures were 1 of 40 for both
signals because its frozen protocol primarily uses `recall_reasoning_query`.

The treatment made five `recall_evidence` calls across three sessions: two non-abstaining
successes, one abstention, and two tool errors. Governing-source exposure and focused-bundle
identity cannot be reconstructed from the redacted final records, so those endpoints are reported
as unobservable rather than inferred.

The treatment stayed within the two-call budget in 38 of 40 sessions (95.0%) and made no
maintenance or mutation calls. However, it applied one wrong fact in 40 cells (2.5%), compared
with zero for control. This fails the preregistered wrong-fact prediction.

Mean treatment input tokens were 54,083.5 versus 148,888.3 for control, a 63.68% reduction.
Mean wall time was 90.98 seconds versus 92.28 seconds, a 1.41% reduction.

## Setup receipt

The run used three workers, the `superseded` condition, the frozen model and corpus fingerprint,
and the 22-tool surface. The signed adjudication receipt was independently verified with receipt
id `cb84df87ee4ae2f22e4898b3b551eb4258c46fcd442d26c697246dd55603b9d6`.

Publishable run artifacts are included beside this report:

- [`admission.json`](2026-09-17-focused-evidence-application-official-018/admission.json)
- [`adjudication.receipt.json`](2026-09-17-focused-evidence-application-official-018/adjudication.receipt.json)
- [`environment.json`](2026-09-17-focused-evidence-application-official-018/environment.json)
- [`costs.json`](2026-09-17-focused-evidence-application-official-018/costs.json)
- [machine-readable aggregate](2026-09-17-focused-evidence-application-official-018.json)

Raw session outputs and execution events remain on VPS2 because the final record surface is
redacted and the execution event stream contains run-only challenge material.

## Decision

Do not promote this prompt-only focused-evidence variant. The result closes this lane for now:
the extra application instructions increased selective evidence use in a few sessions, but did
not make the agent reliably search first and reduced task success overall. A future attempt should
move source selection and application into a product-generated context surface rather than adding
more prose to the initial skill.
