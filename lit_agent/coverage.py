"""Requires the model to account for the parts of a question it cannot ground.

Retrieval here fails by returning a plausible subset: a term is inside the corpus
and simply does not come back, and nothing in the response says so. A model reading
those results has no signal that they are partial, so it answers from them.

This capability makes the gap the model's own statement. Before answering it must
call `assess_coverage` with the question broken into what it needs, each part
pointing at the chunk ids that support it. A part with no support is then a thing
the model has said out loud, and the tool's reply tells it what to do with it:
search differently, or say in the answer that the knowledge base does not cover it.

Modeled on `CitationPolicyCapability`: the same redirect-once-per-question
enforcement, the same history tag so a resumption does not ask twice.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.toolsets import FunctionToolset

from haiku.rag.capabilities.evidence import discover_evidence, question_in_progress

CAPABILITY_ID = "lit-agent-coverage"
STATE_NAMESPACE = "coverage"
ASSESS_TOOL = "assess_coverage"

COVERAGE_TAG = "[lit-agent/coverage-request]"

REDIRECT = (
    "Before you answer, call `assess_coverage`. Break the question into the parts "
    "it needs answered, and for each part give the chunk_ids from your search "
    "results that support it. A part you cannot support gets an empty list — that "
    "is a required answer, not an omission. This is not a request to change an "
    "answer you have already given: call the tool, then repeat your answer exactly "
    f"as you gave it, without a commentary on the coverage. {COVERAGE_TAG}"
)


class Need(BaseModel):
    """One part of the question, and the chunk ids that support it."""

    need: str = Field(description="A part of the question that must be answered.")
    supported_by: list[str] = Field(
        default_factory=list,
        description="Chunk ids from search results that support this part. Empty "
        "when nothing retrieved supports it.",
    )


class CoverageState(BaseModel):
    """What the model declared, per question, for hosts and evaluations to read."""

    declarations: list[dict[str, Any]] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)


def _ends_the_question(
    response: ModelResponse, request_context: ModelRequestContext
) -> bool:
    """Whether this response finishes the question rather than continuing it."""
    calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
    if not calls:
        return True
    output_tools = {
        tool.name for tool in request_context.model_request_parameters.output_tools
    }
    return any(call.tool_name in output_tools for call in calls)


def _already_asked(messages: list[ModelMessage], question: int) -> bool:
    """Whether this question has already been asked to declare coverage."""
    return any(
        isinstance(part, UserPromptPart)
        and isinstance(part.content, str)
        and COVERAGE_TAG in part.content
        for message in messages[question:]
        for part in message.parts
    )


def _declared_this_question(state: CoverageState, question: int) -> bool:
    return any(d["question"] == question for d in state.declarations)


@dataclass
class CoverageCapability(AbstractCapability[Any]):
    """Requires a per-part accounting of what the evidence does and does not cover."""

    @classmethod
    def from_spec(cls) -> "CoverageCapability":
        """The factory takes no configuration, so neither does the spec surface."""
        return create_capability()

    def _state(self, ctx: RunContext[Any]) -> CoverageState | None:
        outer = getattr(ctx.deps, "state", None)
        if not isinstance(outer, dict):
            return None
        return CoverageState.model_validate(outer.get(STATE_NAMESPACE) or {})

    def _write_state(self, ctx: RunContext[Any], state: CoverageState) -> None:
        outer = getattr(ctx.deps, "state", None)
        if isinstance(outer, dict):
            outer[STATE_NAMESPACE] = state.model_dump(mode="json")

    async def before_run(self, ctx: RunContext[Any]) -> None:
        """Publish an empty outcome, so a host can tell "none" from "not running"."""
        outer = getattr(ctx.deps, "state", None)
        if isinstance(outer, dict):
            outer.setdefault(
                STATE_NAMESPACE, CoverageState().model_dump(mode="json")
            )

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Ask once, at the last moment a question can still be redirected."""
        if not _ends_the_question(response, request_context):
            return response
        evidence = discover_evidence(ctx)
        question = question_in_progress(evidence)
        if not _has_evidence(evidence):
            return response
        state = self._state(ctx)
        if state is None:
            return response
        # A call already in flight counts as the declaration being asked for.
        if any(
            isinstance(part, ToolCallPart) and part.tool_name == ASSESS_TOOL
            for part in response.parts
        ):
            return response
        if _declared_this_question(state, question):
            return response
        if _already_asked(ctx.messages, question):
            return response
        ctx.enqueue(REDIRECT, priority="when_idle")
        return response

    def get_toolset(self) -> FunctionToolset[Any]:
        async def assess_coverage(
            ctx: RunContext[Any], needs: list[Need]
        ) -> str:
            """State which parts of the question the retrieved evidence supports.

            Call this once, right before answering. Every part of the question
            gets an entry; a part nothing supports keeps an empty
            ``supported_by``, which is how a gap is reported.

            Args:
                needs: One entry per part of the question that needs answering.
            """
            evidence = discover_evidence(ctx)
            question = question_in_progress(evidence)
            state = self._state(ctx) or CoverageState()
            rendered = [{"need": n.need, "supported_by": n.supported_by} for n in needs]
            state.declarations.append({"question": question, "needs": rendered})
            self._write_state(ctx, state)

            unsupported = [n.need for n in needs if not n.supported_by]
            if not needs:
                reply = (
                    "You declared no parts. If the question needs nothing from the "
                    "knowledge base, answer it and say so."
                )
            elif not unsupported:
                reply = (
                    f"Coverage complete: all {len(needs)} part(s) have support. "
                    "Answer now, and cite the chunk ids you named."
                )
            else:
                listed = json.dumps(unsupported, ensure_ascii=False)
                reply = (
                    f"Unsupported part(s): {listed}. You have three options, and "
                    "answering as if they were covered is not one of them: (a) search "
                    "again with different terms, or read the documents directly; "
                    "(b) answer only the supported parts and state plainly that the "
                    "knowledge base does not cover the rest; (c) if nothing supports "
                    "the question at all, say it cannot be answered from these "
                    "documents."
                )
            state.messages.append(reply)
            self._write_state(ctx, state)
            return reply

        return FunctionToolset(
            [assess_coverage], id=CAPABILITY_ID, max_retries=3, sequential=True
        )


def _has_evidence(evidence: list) -> bool:
    """Whether anything exists that an answer could have been grounded on.

    Read from the ledger rather than from search state, so a question answered
    from code execution or a document read counts too.
    """
    question = question_in_progress(evidence)
    return any(
        found.record.latest_evidence_epoch > question or found.record.occurrences
        for found in evidence
    )


def create_capability() -> CoverageCapability:
    """Create the capability that requires a per-part coverage declaration."""
    return CoverageCapability(
        id=CAPABILITY_ID,
        description=(
            "Requires the model to state which parts of the question the retrieved "
            "evidence supports and which it does not, before answering."
        ),
    )


__all__ = [
    "ASSESS_TOOL",
    "CAPABILITY_ID",
    "COVERAGE_TAG",
    "STATE_NAMESPACE",
    "CoverageCapability",
    "CoverageState",
    "Need",
    "create_capability",
]
