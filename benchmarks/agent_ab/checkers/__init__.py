"""Executable checkers: one per task, each deciding success by running the agent's artifact.

A checker takes the finished sandbox and returns a `CheckResult`. It may run anything inside that
directory and it must never read the session transcript, because an endpoint that reads what the
agent SAID is the endpoint this whole module set exists to replace.

Two rules hold for every checker here:

- **Judge the artifact, not the side effect.** Where a task asks for a re-runnable script, the
  checker resets the tree and runs that script itself. Otherwise an agent that made the change by
  hand and left a broken script beside it would score as having done the work.
- **Fail closed and say why.** A missing artifact, an empty output and a crash are all failures,
  and each returns the evidence that produced the verdict so a disputed row can be re-read.
"""
