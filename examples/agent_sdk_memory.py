"""Example: the anti-re-litigation loop from `self_recall_agent.py`, on the Agent SDK surface.

Where `self_recall_agent.decide` calls the service layer directly, this example hands the same
memory to a real Claude Agent SDK session as in-process tools and lets the MODEL do the
consulting, which is the shape a production agent has. Needs the `agent` extra and the Claude
Code CLI:

    pip install "recall-rag[agent]"
"""
from __future__ import annotations

import anyio

from recall_agent import RecallAgentMemory


async def main() -> None:
    with RecallAgentMemory.from_env() as memory:
        options = memory.options(max_turns=4)
        from claude_agent_sdk import AssistantMessage, TextBlock, query

        prompt = (
            "Proposal: let's inject retrieved context into the prompt to boost answers. "
            "Before endorsing it, search your own memory for prior decisions about this. "
            "If a closed decision or falsified hypothesis surfaces, back off and cite it; "
            "if memory could not be consulted at all, say so and do not proceed."
        )
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)


if __name__ == "__main__":  # pragma: no cover - manual demo entry point
    anyio.run(main)
