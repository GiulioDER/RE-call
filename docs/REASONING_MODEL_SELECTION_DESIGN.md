# Reasoning model selection in `recall setup`

Status: design, approved 2026-08-14. Implements the wizard half only. The consuming code is the
reasoning arm, which is being built separately.

## Why

The reasoning layer has no model configuration of any kind. `grep -rn "RECALL_REASONING" .` returns
nothing, `ReasoningProviderPorts.answer_provider` is never passed by `recall_mcp/service.py:1659`,
and every shipped provider implementation is deterministic. A user who wants a model backed
reasoning arm today has no supported way to say which model, which provider, or which key.

Meanwhile the wizard already collects `OPENAI_API_KEY` and `OPENROUTER_API_KEY` and then does
nothing with them beyond the embedder path. The keys are present, the feature is not.

This adds the selection step so the setting exists on the release, and pins the variable names so
the reasoning arm and the wizard cannot drift apart while they are built separately.

## Scope

In scope: a wizard step, a provider and model catalogue, a verification probe, the `.env` keys, the
tests, and the documentation lines that currently assert the opposite.

Out of scope, and deliberately so:

* Making reasoning actually call the model. `ReasoningBudget.max_model_calls` is `0`
  (`recall/reasoning_planner.py:55`, enforced at `:875`) and `recall_mcp/service.py` never supplies
  `answer_provider`. Both are documented decisions, not oversights. Lifting them belongs to the
  reasoning arm.
* A fourth ad hoc OpenAI client. `recall/truth_extraction/_openai_engine.py` already has a tested
  OpenAI compatible client with `base_url` support and an audit identity that records the endpoint
  host. The reasoning arm should reuse it. This design only writes the settings it will read.
* Cost metadata in the product. Prices change; see "No prices in the menu" below.

## Decisions

### A new `RECALL_REASONING_*` family, not the extraction one

Extraction runs on the ingest path over the whole corpus. Reasoning runs per query. Those have
different cost profiles and a user will plausibly want a cheap model for one and a better model for
the other, which a shared setting makes impossible. The name also has to stay honest: a variable
called `RECALL_EXTRACTION_MODEL` driving reasoning would be lying about its scope.

The naming follows the precedent set by `RECALL_ENTAILMENT` (`recall/entailment.py:77-95`) and
`RECALL_TRUTH_EXTRACTION` (`recall/truth_extraction/_engine.py:161-186`): a boolean named for the
feature, plus `_MODEL`, plus the provider settings.

### Keys written

```
RECALL_REASONING=1
RECALL_REASONING_MODEL=openai/gpt-4o-mini
RECALL_REASONING_BASE_URL=https://openrouter.ai/api/v1
RECALL_REASONING_API_KEY=sk-or-...
```

Answering no writes `RECALL_REASONING=0` and nothing else, so "switched off" and "never configured"
remain distinguishable in `.env`. This mirrors how `RECALL_ENTAILMENT` is seeded to `"0"` in the
`values` dict built by `run_setup_wizard` in `recall/setup.py`.

There is deliberately **no** `RECALL_REASONING_PROVIDER`. The base URL already identifies the
provider, and `_host_of()` (`recall/truth_extraction/_openai_engine.py:101-142`) already turns it
into an audit identity that records host and port and never credentials. A separate provider name
would be a second source of truth for the same fact.

Timeout and revision are not written. The consuming code should default them the way
`_openai_engine.py` does (`RECALL_EXTRACTION_TIMEOUT` defaults to `"60"`, revision to `"unpinned"`),
so `.env` stays small and the defaults live in one place.

`RECALL_REASONING_API_KEY` is written even when it duplicates a key already written as
`OPENROUTER_API_KEY`. The alternative, resolving through a fallback chain, puts resolution logic in
code that does not exist yet and makes the effective key non obvious in the file. An explicit
duplicate inside a single gitignored `.env` is the lesser cost.

### The security question hides cloud providers

Answering yes to "Is data security necessary for this installation?" already hides cloud embedders
in `embedder_choices` in `recall/setup.py`. Reasoning sends the query **and the retrieved
evidence** to the provider, which exposes more than embedding does. Someone who said their data
must not leave the machine must not be walked into sending retrieved memory to a third party three
prompts later.

Under `security_required`, the provider menu offers only the local OpenAI compatible endpoint.

### One verification probe, non fatal

After the model is chosen, send one minimal completion. A wrong key, a retired model id, or a local
endpoint that is not running are the most likely failures, and finding them now beats finding them
on the first reasoning query.

On failure the wizard prints what failed and **still writes the configuration**, so a typo can be
corrected in `.env` without re-running the whole wizard. A transient network fault must never block
an install over a choice that is probably correct.

The probe is skipped, with a printed note, when the `openai` package is not importable.

### No prices in the menu

The five OpenRouter ids below were verified against the live catalogue on 2026-08-14, at which point
`google/gemini-2.0-flash-001`, which was going to be shipped, turned out not to exist. Prices ranged
from $0.10 to $3.00 per million input tokens.

Those prices are not going into the product. A number baked into a shipped menu is a measurement
that nothing re-checks, and it goes stale on someone else's release schedule. The descriptions say
cheap, balanced, or best instead, and the manual entry option is what keeps a stale catalogue from
ever being fatal.

## The wizard step

Inserted after the entailment judge question and before "Scaffold CLAUDE.md and a memory/
directory". Three prompts, in the order the user asked for: yes or no, then provider, then model.

### 1. Yes or no

```
Enable the optional reasoning arm? [y/N]
```

Default no, matching the entailment judge (`_ask_yes_no(..., default=False)`). Answering no writes
`RECALL_REASONING=0` and skips the rest of the step.

### 2. Provider

Built by a new `reasoning_provider_choices(probe, *, security_required)` returning `list[Choice]`,
following `embedder_choices` in `recall/setup.py` in shape, including the `available` flag and an
`unavailable_note`. The note is composed locally rather than by `_why_unavailable` in
`recall/setup.py`, because that helper reports on sentence-transformers, CUDA and free disk, none
of which is why an API provider is unreachable. Reusing it would send the reader to fix the wrong
thing, which is the exact failure its own docstring warns about.

| Order | label | value | Included when | Available when |
|---|---|---|---|---|
| 1 | `local endpoint` | `local` | always | always |
| 2 | `openrouter` | `https://openrouter.ai/api/v1` | `not security_required` | `probe.internet and _module_available("openai")` |
| 3 | `openai` | `https://api.openai.com/v1` | `not security_required` | `probe.internet and _module_available("openai")` |

Only `not security_required` decides whether a cloud entry is appended to the list at all. When
security is off, both cloud entries are always appended; `probe.internet and
_module_available("openai")` then sets their `available` flag alone, which is why an unusable
cloud entry can still appear in the menu and explain, through its `unavailable_note`, why it
cannot run.

The local endpoint is first because `_choose` in `recall/setup.py` raises `ValueError(f"the first
choice for {title} must be runnable")` when `choices[0].available` is false. A local endpoint needs
no key and no internet, so it is the only entry that is unconditionally offerable.

Choosing `local` prompts for two values rather than showing a menu, because the wizard cannot know
which models have been pulled locally:

```
Base URL for the local endpoint [http://localhost:11434/v1]:
Model id:
```

A blank base URL takes the default, which is Ollama's OpenAI compatible endpoint. A blank model id
re-asks exactly once, because there is no sensible default; a second blank answer is taken as "not
now" and handled as described under "When nothing is runnable". `RECALL_REASONING_API_KEY` is written as the literal `unused-local-key` for a local endpoint,
since the OpenAI client requires a non empty key and local servers ignore its value. It is
deliberately NOT the same string as the `local` provider sentinel: while the two shared a value,
swapping one for the other at either call site would have passed every test.

### 3. Model

Only for the two cloud providers. Verified against the live OpenRouter catalogue on 2026-08-14.

| Order | label | value | description |
|---|---|---|---|
| 1 | `gpt-4o mini` | `openai/gpt-4o-mini` | `Small, fast and inexpensive, the default` |
| 2 | `deepseek chat` | `deepseek/deepseek-chat` | `Low cost general model, strong for the price` |
| 3 | `deepseek r1` | `deepseek/deepseek-r1` | `Reasoning tuned, slower and dearer than chat` |
| 4 | `llama 3.3 70b` | `meta-llama/llama-3.3-70b-instruct` | `Open weights, the cheapest option here` |
| 5 | `claude sonnet 4.5` | `anthropic/claude-sonnet-4.5` | `Best quality and the dearest, matching the truth extraction default` |
| 6 | `enter a model id` | `` | `Type an id yourself, for anything not listed` |

The table above is the **OpenRouter** menu. The `openai` provider gets a different and shorter one,
because `api.openai.com` serves only OpenAI's own models and does not accept OpenRouter's
`vendor/model` form. Offering DeepSeek or Llama there would produce a menu entry that cannot work:

| Order | label | value | description |
|---|---|---|---|
| 1 | `gpt-4o mini` | `gpt-4o-mini` | `Cheap and fast, a good default` |
| 2 | `gpt-4o` | `gpt-4o` | `Higher quality and several times the price` |
| 3 | `enter a model id` | `` | `Type an id yourself, for anything not listed` |

Those two ids could not be verified against a live catalogue the way the OpenRouter ids were, since
listing OpenAI's models requires a key. That is one more reason the manual entry exists on every
provider.

Selecting `enter a model id` prompts:

```
Model id:
```

A blank answer re-asks. The menu entry carries the empty string as its value, and the empty string
is the sentinel that triggers the prompt.

### Key capture

If a cloud provider is chosen and the corresponding key is absent from `cloud_keys`, the wizard asks
for it rather than writing a configuration that cannot work:

```
OPENROUTER_API_KEY:
```

The captured key is added to `cloud_keys`, so it is written both as the provider key and as
`RECALL_REASONING_API_KEY`.

A blank answer follows the same retry rule as the model id, through `_prompt_twice`: it is asked
once more, and if the second answer is also blank the step prints a note and writes
`RECALL_REASONING=0`, without adding anything to `cloud_keys`, rather than writing an arm that
cannot authenticate.

Note that step 1b only runs when the security question is answered no, in `run_setup_wizard` in
`recall/setup.py`, and the cloud providers are only offered in that same case, so this prompt fires
only when the user skipped the key at 1b and then asked for a cloud provider.

### When nothing is runnable

`reasoning_provider_choices` always returns at least the local endpoint, so the menu cannot be
empty. If the user selects local and then supplies no model id after being re-asked once, the step
prints a note and writes `RECALL_REASONING=0`, following the shape used when the entailment judge
is unsupported in `entailment_choices` in `recall/setup.py`, where the builder returns `[]` and the
caller prints a paragraph instead of prompting.

## Failure handling

| Situation | Behaviour |
|---|---|
| `openai` package missing | Cloud providers unavailable with `unavailable_note` naming `pip install "recall-rag[extract]"`. Local remains selectable, probe skipped with a note. |
| No internet | Cloud providers unavailable via `probe.internet`, exactly as the embedder menu does. |
| Probe fails | Print the exception text, print that the settings were written anyway, continue. Never raise. |
| Blank model id | Re-ask once, then write `RECALL_REASONING=0` with a printed note. |
| Local endpoint unreachable | This is a probe failure and is treated as one. The wizard runs before the user has necessarily started Ollama. |

The probe must catch `Exception`, not a narrower type. It runs against three different providers and
an arbitrary user supplied base URL, so the set of reachable exception types is not knowable here,
and any escape turns an optional step into a failed install.

## Testing

New unit tests for the choice builders, following `test_embedder_choices_hide_cloud_when_security_is_required`
in `tests/test_setup.py`:

* `reasoning_provider_choices` hides both cloud entries when `security_required` is true.
* `reasoning_provider_choices` marks both cloud entries unavailable, with a note, when `openai` is
  not importable. It does not hide them: an unreachable cloud provider still names itself and says
  why, the same way an unavailable embedder or reranker does.
* The first returned choice is always `available`, which is the precondition `_choose` enforces.
* The manual entry option is present for every cloud provider.

New wizard level tests, following the existing positional `iter([...])` pattern:

* Declining writes `RECALL_REASONING=0` and no other reasoning key.
* Accepting with OpenRouter and the DeepSeek default writes all four keys with the expected values.
* Accepting with a local endpoint writes the default base URL and `RECALL_REASONING_API_KEY=local`.
* A failing probe still writes the configuration and prints the failure.

A name pinning test asserting the exact four variable names, so the wizard and the reasoning arm
cannot drift while they are built separately. This is the test that makes "write the config before
the consumer exists" a safe decision rather than a hopeful one.

The probe is exercised by monkeypatching `openai.OpenAI` with a fake, following
`tests/test_truth_extraction_engine_openai.py:208-234`. No test performs a network call.

**Every existing wizard test needs its answer list updated.** `tests/test_setup.py` drives the
wizard with flat positional `iter([...])` lists, so inserting a prompt shifts every later answer.
Each list in that file carries an inline comment naming what each answer selects, which is what
keeps a shift visible in a diff rather than surfacing as an unrelated failure two prompts later.
This is the largest single cost of the change and it is mechanical, not subtle.

## Documentation

* `docs/REASONING_OPERATIONS.md:120` currently reads that no managed model provider is required by
  the default reasoning tools. That remains true of the default tools and becomes misleading once a
  provider can be configured. It needs a sentence distinguishing the default deterministic path from
  a configured reasoning arm.
* `.env.example` gains the four variables. Note that the parity gate in
  `tests/test_env_example_parity.py` is scoped to `RECALL_(AUTH|OIDC)_*`, so nothing enforces this
  automatically, and the file currently documents none of the `RECALL_EXTRACTION_*` variables
  either. Adding these without adding those would leave the file inconsistent in a new way, so the
  extraction variables should be added in the same pass.
* `docs/ENVIRONMENT.md` gains the same four.
* The setup site's wizard answer table at `site/start.html` gains a row. The site documents the
  published release, so this lands when the release ships, not before.

## Known constraints, stated so they are not rediscovered

1. Configuring a model does not make reasoning call one. `max_model_calls` is `0` and
   `answer_provider` is never wired. The wizard writes settings for a port that the shipped code
   does not yet use.
2. `_choose` requires a runnable first option and returns `choices[0]` when an unavailable option is
   selected. Any future reordering of the provider menu must keep the local endpoint first.
3. The model catalogue is a static list in a released artifact. It will go stale. The manual entry
   option is the mitigation, and it is not optional.
