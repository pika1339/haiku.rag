"""Rewrites the question into search queries before the agent searches.

Layer 3's middle arm: no rewrite (the base as it ships) against a model that
rewrites. The third arm, human rewrites, has no counterpart here — the corpus is
Chinese while the base's rewrite dataset is MTRAG's English one.

The instruction lands on the first request only, so it shapes how the run starts
without being repeated into every round.
"""

from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext

CAPABILITY_ID = "lit-agent-rewrite"
STATE_NAMESPACE = "rewrite"

INSTRUCTION = """Before your first search, rewrite the user's question into 2-3 \
distinct search queries. Rules:
- Use the terminology this literature actually uses; a paper is unlikely to \
contain the words of a casual question.
- Make the queries different from each other, not paraphrases: one for each \
thing the question needs, so a question asking for two facts gets a query per fact.
- Search with each of them, then answer.
Keep the rewrites to yourself; they are search queries, not part of the answer."""


@dataclass
class RewriteCapability(AbstractCapability[Any]):
    """Tells the model to rewrite the question into queries before searching."""

    @classmethod
    def from_spec(cls) -> "RewriteCapability":
        """The factory takes no configuration, so neither does the spec surface."""
        return create_capability()

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Add the rewrite instruction to the first request of the run."""
        outer = getattr(ctx.deps, "state", None)
        if isinstance(outer, dict):
            state = outer.setdefault(STATE_NAMESPACE, {})
            if state.get("injected"):
                return request_context
            state["injected"] = True

        from pydantic_ai.messages import ModelRequest, InstructionPart

        current = request_context.messages[-1]
        if isinstance(current, ModelRequest):
            current.instructions = "\n\n".join(
                part for part in (current.instructions, INSTRUCTION) if part
            )
            parameters = request_context.model_request_parameters
            from dataclasses import replace

            request_context.model_request_parameters = replace(
                parameters,
                instruction_parts=[
                    *(parameters.instruction_parts or []),
                    InstructionPart(content=INSTRUCTION, dynamic=True),
                ],
            )
        return request_context


def create_capability() -> RewriteCapability:
    """Create the capability that asks for a query rewrite before searching."""
    return RewriteCapability(
        id=CAPABILITY_ID,
        description=(
            "Instructs the model to rewrite the question into distinct search "
            "queries, using the terminology of the literature, before searching."
        ),
    )


__all__ = [
    "CAPABILITY_ID",
    "INSTRUCTION",
    "STATE_NAMESPACE",
    "RewriteCapability",
    "create_capability",
]
