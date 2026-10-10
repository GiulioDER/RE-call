"""The query and document prompts each supported local model was trained with, pinned per model.

A local model encoded without its published prompt is measured outside its training setup, and
the difference is not small: on LoCoMo turn retrieval, sending the published prompts moved
voyage-4-nano by +0.138 MRR@10 and snowflake-arctic-embed-l-v2.0 by +0.174 against raw text on
both sides (1,535 questions, measured 2026-10-10). RE-call's `st:<model>` path sends neither, so
`st-prompted:<model>` exists to send them.

The table is the whole source of truth. Nothing reads a model's own
`config_sentence_transformers.json` at runtime: a repository can change its prompts under the same
revision name, and a prompt that changes is a change to every stored vector. Each entry pins the
Hugging Face revision it was read from, and the prompts are recorded in the embedder's profile, so
a different prompt is a different identity and never mixes with vectors written under this one.

`version` names the prompt set. Changing any prompt in an entry means a new version string, which
changes the embedder's name and fingerprint, so a store built under the old prompts refuses rather
than silently answering through the new ones.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSet:
    """One model's published prompts, the revision they were read from, and how to load it."""

    version: str
    query: str
    document: str
    revision: str
    remote_code: bool = False
    #: Matryoshka width to keep, or None for the model's own. The vector is renormalised after.
    dimension: int | None = None


_BGE_QUERY = "Represent this sentence for searching relevant passages: "
_QWEN3_QUERY = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)

PUBLISHED_PROMPTS: dict[str, PromptSet] = {
    "BAAI/bge-small-en-v1.5": PromptSet(
        "published-v1", _BGE_QUERY, "", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),
    "BAAI/bge-large-en-v1.5": PromptSet(
        "published-v1", _BGE_QUERY, "", "d4aa6901d3a41ba39fb536a557fa166f842b0e09"),
    "Snowflake/snowflake-arctic-embed-l-v2.0": PromptSet(
        "published-v1", "query: ", "", "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"),
    "Qwen/Qwen3-Embedding-0.6B": PromptSet(
        "published-v1", _QWEN3_QUERY, "", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
    "voyageai/voyage-4-nano": PromptSet(
        "published-v1",
        "Represent the query for retrieving supporting documents: ",
        "Represent the document for retrieval: ",
        "67fabc9bef010dabc5f6024aa1b1b6b93410426f",
        remote_code=True,
        dimension=1024,
    ),
    "nomic-ai/CodeRankEmbed": PromptSet(
        "published-v1",
        "Represent this query for searching relevant code: ",
        "",
        "3c4b60807d71f79b43f3c4363786d9493691f8b1",
        remote_code=True,
    ),
}


def prompts_for(model: str) -> PromptSet:
    """The pinned prompt set for `model`, or a ValueError naming the models that have one."""
    found = PUBLISHED_PROMPTS.get(model)
    if found is None:
        raise ValueError(
            f"st-prompted has no published prompts pinned for {model!r}; supported: "
            f"{', '.join(sorted(PUBLISHED_PROMPTS))}. Use st:{model} for raw text, or add the "
            "model's prompts to recall/embedding_prompts.py with the revision they were read from."
        )
    return found
