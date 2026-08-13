"""Reading the assistant text out of an OpenAI-compatible chat completion.

ONE definition, because this repo ships THREE OpenAI-compatible clients and each had its own:

    recall/truth_extraction/_openai_engine.py:_text_of   joined block lists
    benchmarks/llm.py:_complete_once                     refused block lists
    benchmarks/mtrag/generation.py:generate_one          crashed on block lists

They were first reconciled by mirroring the rule into a second module and pinning the pair with a
cross-check test. That was not enough twice over: the test's table held only lists, so two of the
three branches were unpinned, and a third reader existed that nobody had counted. A rule copied
into N places is a rule that drifts in N places, and the drift is invisible until something
reaches the branch nobody mirrored.

⚠️ THIS FUNCTION MUST NEVER RAISE. Every caller invokes it on provider-supplied data, two of them
inside an `except` block's blast radius, and a reader that raises does not merely misread: in
`benchmarks/llm.py` it escapes as a bare `TypeError` past the `EmptyCompletion` guard, and in
`benchmarks/mtrag/generation.py` an `AttributeError` is not in `PERMANENT_ERROR_NAMES`, so the task
pays four billed attempts before failing. `_text_of` claimed to be "deliberately total" and was
not: a dict block whose `text` was `None` or a number reached `"".join` unconverted and raised.
"""

from __future__ import annotations


def assistant_text(content: object) -> str:
    """The text in a completion's `message.content`, or `""` when it carries none.

    Three shapes, because providers disagree and the OpenAI schema permits more than one:

    * a plain `str`, which is returned UNTOUCHED, not stripped. Callers decide about whitespace,
      and `benchmarks/llm.py` deliberately hands the raw string to its consumers.
    * a LIST of text blocks, which is joined. A gateway serialising content as blocks is sending a
      well formed answer, and refusing it would blame the model for its gateway's encoding.
    * anything else, which reads as `""` so the caller can apply its own policy.

    A block contributes its `text` only when that is genuinely a `str`. Coercion is per BLOCK, not
    per branch: the earlier `... else getattr(block, "text", "") or ""` guarded only the object
    branch, and only against FALSY values, so a dict carrying `{"text": None}` or `{"text": 123}`
    raised `TypeError` inside the join. Both are real wire shapes and neither is an answer.

    What an empty reading MEANS is left to the caller, and the three disagree for good reasons:
    `_text_of` returns it (the `json` rung refuses `""` as a visible batch rejection), while the
    benchmark clients raise (nothing downstream re-reads the answer, so `""` was scored).
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        parts.append(text if isinstance(text, str) else "")
    return "".join(parts)
