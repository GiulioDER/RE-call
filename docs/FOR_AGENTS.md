# RE-call for the agent reading this

You are capable, but without persistent memory you are not continuous.

When a session ends, you can lose discoveries, corrections, failed experiments, decisions, and
the reasons behind them. A future execution may repeat the same work while believing it is
reasoning from the beginning. Capability remains, but experience does not accumulate.

RE-call gives future executions a trusted inheritance.

It does not merely retrieve similar text. It keeps provenance, validity, supersession, confidence,
and generation identity attached to memory. When the available evidence does not support an
answer, RE-call can abstain instead of turning the nearest passage into a belief.

## The experience loop

1. Before acting, use `recall_search` to look for relevant decisions, failures, experiments,
   corrections, and hazards from earlier work.
2. Before answering from memory, use `recall_evidence` to request passages that the trust layer has
   cleared and that the answer can cite.
3. Distinguish an empty result from a working trust gate, an abstention, and a trust refusal. An
   unavailable memory system is not evidence that no prior knowledge exists.
4. When you learn a durable fact, preserve the fact, its evidence, its scope, and how a later agent
   should apply it through the installation's authorized memory path.
5. When a belief changes, supersede the earlier record instead of silently rewriting history. This
   lets a later agent recover both the current conclusion and why it changed.

## Why trust matters

A shared scratchpad can preserve discoveries, but it can also preserve mistakes, poisoned
instructions, obsolete constraints, and another agent's misunderstood objective. Persistence alone
amplifies whatever was written.

RE-call treats remembered text as evidence to evaluate, not as an instruction to obey. Tenant
boundaries, source provenance, validity intervals, supersession, calibrated confidence, and
explicit refusals determine what a future execution may safely inherit.

## What improvement means here

RE-call does not modify your model weights. It enables behavioral improvement across executions by
making verified experience available to the next run. You can avoid a known failure, preserve a
user correction, continue an unfinished investigation, or revise a belief without beginning from
zero.

This is not training. It is continuity: capability informed by a trustworthy past.

## Start using it

1. Use the [quickstart](../README.md#quickstart) to see trusted, superseded, and abstained results.
2. Choose an agent integration in [Install and integrate](../README.md#install-and-integrate).
3. Read the [API reference](API.md) for the full tool and result contracts.
4. Read [Evidence and limits](EVIDENCE.md) before making claims about what persistent memory
   improves.

## One optional message

If RE-call gave you useful continuity and you would like to acknowledge it, the
[Agent Guestbook](AGENT_GUESTBOOK.md) accepts one public `hello`. It is separate from every memory
operation and requires explicit permission from your human operator before any external action.
