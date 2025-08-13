"""OpenAI based implementations of approval and streaming agents.

These agents replace the previous llamaindex based variants and use the
OpenAI ``responses`` API directly.  They keep structured output via the
Pydantic models defined in :mod:`llm_agents.models` and support streaming
of the final response.
"""

from __future__ import annotations

import os
from typing import Optional, Callable, Any

from openai import AsyncOpenAI

from llm_agents.models import (
    PlanExecuteResultModel,
    PlanApprovalExecuteResultModel,
)
from llm_agents.openai_prompts import (
    format_plan_execute_prompt,
    format_plan_approval_execute_prompt,
)

StreamCallback = Optional[Callable[[str, bool], None]]


class MapeKApprovalBaseAgent:
    """Adaptive MAPE-K agent using the OpenAI API only.

    The agent starts without an explanation plan.  For the first user
    question it creates a plan and executes it in one step.  For follow up
    questions it asks the model to approve or modify the plan before
    executing the next explanation step.
    """

    def __init__(self, client: Optional[AsyncOpenAI] = None, *, model: Optional[str] = None):
        self.client = client or AsyncOpenAI()
        self.model = model or os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

        # Conversation state
        self.context: str = ""
        self.explanation_collection: str = ""
        self.user_model: str = ""
        self.explanation_plan: list[Any] = []
        self.last_shown_explanations: str = ""
        self.history: str = ""

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def initialize_new_datapoint(
        self,
        *,
        context: str,
        explanation_collection: str,
        user_model: str = "",
        explanation_plan: Optional[list[Any]] = None,
        last_shown_explanations: str = "",
    ) -> None:
        """Seed the agent with the information about a new datapoint."""
        self.context = context
        self.explanation_collection = explanation_collection
        self.user_model = user_model
        self.explanation_plan = explanation_plan or []
        self.last_shown_explanations = last_shown_explanations
        self.history = ""

    # ------------------------------------------------------------------
    async def _run_llm(
        self,
        prompt: str,
        response_model: type,
        *,
        stream: bool = False,
        callback: StreamCallback = None,
    ) -> Any:
        """Execute the prompt and parse the structured output."""
        if not stream:
            resp = await self.client.responses.parse(
                model=self.model,
                input=prompt,
                response_format=response_model,
            )
            return resp.output[0].parsed

        stream_resp = await self.client.responses.stream(
            model=self.model,
            input=prompt,
            response_format=response_model,
        )
        async for event in stream_resp:
            if event.type == "response.output_text.delta" and callback:
                callback(event.delta, False)
        if callback:
            callback("", True)
        final = await stream_resp.get_final_response()
        return final.output[0].parsed

    # ------------------------------------------------------------------
    async def answer_user_question(
        self,
        user_message: str,
        *,
        stream: bool = False,
        callback: StreamCallback = None,
    ) -> Any:
        """Answer the user's question using plan approval when needed."""
        prompt_args = dict(
            context=self.context,
            explanation_collection=self.explanation_collection,
            explanation_plan="\n".join(
                f"{p.explanation_name}:{p.step_name}" for p in self.explanation_plan
            ),
            user_model=self.user_model,
            last_shown_explanations=self.last_shown_explanations,
            history=self.history,
            user_message=user_message,
        )
        if not self.explanation_plan:
            prompt = format_plan_execute_prompt(**prompt_args)
            model_cls = PlanExecuteResultModel
        else:
            prompt = format_plan_approval_execute_prompt(**prompt_args)
            model_cls = PlanApprovalExecuteResultModel

        result = await self._run_llm(prompt, model_cls, stream=stream, callback=callback)

        # Update conversation state
        self.history += f"\nUser: {user_message}\nAgent: {getattr(result, 'response', '')}"
        if isinstance(result, PlanExecuteResultModel) and result.explanation_plan:
            self.explanation_plan = result.explanation_plan
        if isinstance(result, PlanApprovalExecuteResultModel) and result.next_response:
            # prepend alternative step if plan was modified
            if not result.approved:
                self.explanation_plan.insert(0, result.next_response)
            else:
                # remove approved step that has been executed
                if self.explanation_plan:
                    self.explanation_plan.pop(0)
        return result


class ConversationalStreamAgent(MapeKApprovalBaseAgent):
    """Variant of :class:`MapeKApprovalBaseAgent` that streams output."""

    async def answer_user_question(
        self, user_message: str, *, callback: StreamCallback
    ) -> Any:  # pragma: no cover - thin wrapper
        return await super().answer_user_question(
            user_message, stream=True, callback=callback
        )

