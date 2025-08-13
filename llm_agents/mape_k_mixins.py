# mape_k_mixins.py

from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llm_agents.agent_utils import (timed, OPENAI_MODEL_NAME, OPENAI_MINI_MODEL_NAME, OPENAI_REASONING_MODEL_NAME)
import logging
import json

from llm_agents.events import MonitorDoneEvent, AnalyzeDoneEvent, PlanDoneEvent

# Configure logger
logger = logging.getLogger(__name__)

from llama_index.core.workflow import Context, Workflow, StartEvent, StopEvent, step
from llama_index.core.workflow.retry_policy import ConstantDelayRetryPolicy
from llm_agents.models import MonitorResultModel, AnalyzeResult, PlanResultModel, ExecuteResult, PlanApprovalModel
from llm_agents.prompt_mixins import (
    MonitorPrompt,
    AnalyzePrompt,
    PlanPrompt,
    ExecutePrompt,
    MonitorAnalyzePrompt,
    PlanExecutePrompt,
    UnifiedPrompt,
    PlanApprovalPrompt
)

# Import the new prompts and models
from llm_agents.prompt_mixins import PlanApprovalExecutePrompt
from llm_agents.models import PlanApprovalExecuteResultModel

# Import dual-mode prediction utilities
from llm_agents.utils.dual_mode_prediction import astructured_predict_with_fallback

from llm_agents.helper_mixins import (
    UserModelHelperMixin,
    ConversationHelperMixin,
    UnifiedHelperMixin,
)

# Two-step & unified prompt variants
from llm_agents.models import SinglePromptResultModel, MonitorAnalyzeResultModel, PlanExecuteResultModel

import datetime
import os
from llama_index.core import PromptTemplate
from llm_agents.llama_index_base_agent import LlamaIndexBaseAgent
from llm_agents.explanation_state import ExplanationState
from llm_agents.models import ChosenExplanationModel
from llm_agents.utils.postprocess_message import replace_plot_placeholders

# Add streaming support
import asyncio
from typing import Optional, Callable, AsyncGenerator, Dict, Any

# Reduce verbosity for OpenAI and HTTP libraries
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from llama_index.core.callbacks import CallbackManager


class BaseAgentInitMixin:
    """
    DRY mixin for common agent initialization and shared methods.
    Eliminates duplicate __init__ code and common methods across all agent classes.
    """

    def _init_agent_components(
            self,
            llm: LLM = None,
            structured_output: bool = True,
            timeout: float = 100.0,
            default_model: str = OPENAI_MODEL_NAME,
            use_mini_llm: bool = False,
            special_model: str = None,
            **workflow_kwargs
    ):

        """
        Common initialization for LLM and other agent components.

        Args:
            llm: Language model instance
            structured_output: Whether to use structured output
            timeout: Workflow timeout
            default_model: Default OpenAI model to use
            use_mini_llm: Whether this agent needs a mini_llm
            special_model: Special model (e.g., reasoning model for unified agent)
            **workflow_kwargs: Additional workflow arguments (agent-specific args filtered out)
        """
        # Filter out agent-specific kwargs - only pass known workflow arguments
        # Workflow.__init__ typically only accepts: timeout, retry_policy, verbose
        # We'll be conservative and only pass through known safe arguments
        known_workflow_args = ['retry_policy', 'verbose']
        workflow_specific_kwargs = {k: v for k, v in workflow_kwargs.items()
                                    if k in known_workflow_args}

        # Initialize Workflow with timeout and only safe workflow kwargs
        Workflow.__init__(self, timeout=timeout, **workflow_specific_kwargs)

        # Initialize StreamingMixin
        StreamingMixin.__init__(self)

        # Use an empty callback manager; logging is handled by the fallback method
        self.callback_manager = CallbackManager([])

        if special_model:
            self.llm = llm or OpenAI(
                model=special_model,
                reasoning_effort="low",
                callback_manager=self.callback_manager,
                user=str(self.experiment_id),
            )
        else:
            self.llm = llm or OpenAI(
                model=default_model,
                callback_manager=self.callback_manager,
                user=str(self.experiment_id),
            )

        if use_mini_llm:
            self.mini_llm = OpenAI(
                model=OPENAI_MINI_MODEL_NAME,
                callback_manager=self.callback_manager,
                user=str(self.experiment_id),
            )

        self.structured_output = structured_output
        # Initialize visual explanations storage
        self.visual_explanations_dict = {}
        # Buffer for CSV row items (prompt/response pairs)
        self._csv_current_run_items = []

    def log_state_snapshot(self, snapshot_type: str):
        """
        Log current user model and explanation plan state for tracking changes.
        
        Args:
            snapshot_type: Type of snapshot ('pre_execution' or 'post_execution')
        """
        import json
        
        # Capture user model state as structured JSON
        user_model_state = {}
        if hasattr(self, 'user_model') and self.user_model:
            try:
                user_model_state = self.user_model.get_state_summary(as_dict=True)
            except Exception as e:
                user_model_state = {"error": f"Error capturing user model: {e}"}
        
        # Capture explanation plan state
        explanation_plan_state = ""
        if hasattr(self, 'explanation_plan') and self.explanation_plan:
            try:
                # Convert explanation plan to readable format
                plan_items = []
                for exp in self.explanation_plan:
                    plan_items.append(f"{exp.explanation_name}:{exp.step_name}")
                explanation_plan_state = " | ".join(plan_items)
            except Exception as e:
                explanation_plan_state = f"Error capturing explanation plan: {e}"
        
        # Create state snapshot
        state_data = {
            "user_model_state": user_model_state,
            "explanation_plan_state": explanation_plan_state
        }
        
        # Log the state snapshot
        self.log_component_input_output(
            f"{snapshot_type}_state", 
            json.dumps(state_data), 
            f"State captured at {snapshot_type}"
        )

    @timed
    async def _execute_workflow(self, user_question):
        """
        Core workflow execution without logging concerns.
        Runs the workflow and extracts reasoning and response consistently.
        """
        analysis = "No reasoning available"
        response = "Sorry, please try again"
        
        try:
            result = await self.run(input=user_question)
            
            # Extract reasoning and response with fallbacks
            analysis = getattr(result, "reasoning", None) or "No reasoning available"
            response = getattr(result, "response", None) or "Sorry, please try again"
            
        except Exception as e:
            logger.error(f"Error during workflow execution: {e}", exc_info=True)
            response = f"Agent execution failed: {str(e)}"

        return analysis, response

    async def _predict_with_timing_and_logging(self, model_class, prompt, method_name, llm=None, nested=True):
        """
        Logs prompt, result, and timing. Logs errors and stack traces. Truncates long results. Logs timing as MLflow metric.
        """
        start_time = datetime.datetime.now()
        result = None
        if llm is None:
            llm = getattr(self, 'mini_llm', self.llm)

        try:
            result = await astructured_predict_with_fallback(
                llm, model_class, prompt, use_structured_output=self.structured_output
            )
        except Exception as e:
            logger.error(f"Error during {method_name}: {e}", exc_info=True)
            raise
        finally:
            end_time = datetime.datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            logger.info(f"Time taken for {method_name.replace('_', ' ').title()}: {elapsed:.2f}s")
            logger.info(
                f"{method_name.replace('_', ' ').title()} result: {result if result is not None else '[FAILED]'}\n")
        return result


# Streaming callback type
StreamCallback = Optional[Callable[[str, bool], None]]


class StreamingMixin:
    async def _predict(self, model_class, prompt, method_name):
        """
        Unified predict: streams if enabled, otherwise non-streaming, then logs result.
        
        Args:
            model_class: The Pydantic model class for structured output
            prompt: The PromptTemplate object to send to LLM
            method_name: Name for logging purposes
        """
        # Choose streaming or non-streaming
        if self.enable_streaming and self.stream_callback:
            result = await self._stream_predict_with_callback(model_class, prompt, method_name)
        else:
            result = await self._predict_with_timing_and_logging(model_class, prompt, method_name)
        # Log input and output - use formatted string if provided, otherwise extract from prompt object
        prompt_text = prompt.get_template()
        # result_text = getattr(result, "response", None) if hasattr(result, "response") else str(result)
        self.log_component_input_output(method_name, prompt_text, result)
        return result

    """Mixin to add streaming capability to agents."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_callback: StreamCallback = None
        self.enable_streaming: bool = False

    def set_stream_callback(self, callback: StreamCallback):
        """Set a callback function for streaming responses."""
        self.stream_callback = callback
        self.enable_streaming = callback is not None

    async def _execute_workflow_streaming(self, user_question: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Core streaming workflow implementation without logging concerns.
        This replaces all the duplicated streaming logic across different agent classes.

        Args:
            user_question: The user's question

        Yields:
            Dict with streaming data: {"type": "partial"|"final"|"error", "content": str, "is_complete": bool}
        """
        # Create a queue to capture streaming data
        stream_queue = asyncio.Queue()
        final_result = None

        def capture_stream(content: str, is_final: bool):
            """Capture streaming data into queue."""
            nonlocal final_result
            if is_final:
                stream_queue.put_nowait({"type": "final", "content": "", "is_complete": True})
            else:
                stream_queue.put_nowait({"type": "partial", "content": content, "is_complete": False})

        # Set up streaming callback
        self.set_stream_callback(capture_stream)

        # Start the workflow in background
        async def run_workflow():
            nonlocal final_result
            try:
                result = await self.run(input=user_question)
                final_result = result
                # Signal completion
                stream_queue.put_nowait({"type": "workflow_complete", "result": result})
            except Exception as e:
                stream_queue.put_nowait({"type": "error", "error": str(e)})

        # Start workflow task
        workflow_task = asyncio.create_task(run_workflow())

        try:
            while True:
                # Wait for stream data with timeout
                try:
                    chunk = await asyncio.wait_for(stream_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # Check if workflow is done
                    if workflow_task.done():
                        break
                    continue

                if chunk["type"] == "workflow_complete":
                    # Send final response
                    result = chunk["result"]
                    analysis = getattr(result, "reasoning", "") if hasattr(result, "reasoning") else ""
                    response = getattr(result, "response", "") if hasattr(result, "response") else ""
                    yield {
                        "type": "final",
                        "content": response,
                        "reasoning": analysis,
                        "is_complete": True
                    }
                    break
                elif chunk["type"] == "error":
                    yield {"type": "error", "content": chunk["error"], "is_complete": True}
                    break
                elif chunk["type"] == "partial":
                    yield chunk
                elif chunk["type"] == "final":
                    continue  # Wait for workflow_complete

        finally:
            # Clean up
            if not workflow_task.done():
                workflow_task.cancel()
                try:
                    await workflow_task
                except asyncio.CancelledError:
                    pass  # Expected when cancelling
                except Exception as e:
                    logger.warning(f"Error during workflow cleanup: {e}")
            else:
                # Ensure completed task exceptions are retrieved
                try:
                    workflow_task.result()
                except Exception:
                    pass  # Ignore exceptions from completed tasks
            self.set_stream_callback(None)


    async def _stream_predict_with_callback(self, model_class, prompt, method_name):
        """Real token-level streaming with improved JSON parsing."""
        try:
            logger.info(f"Starting real token streaming for {method_name}")
            start_time = datetime.datetime.now()
            # Convert structured prompt to completion prompt
            schema_prompt = self._build_schema_prompt(model_class, prompt)

            # Use normal streaming completion for real token-by-token streaming
            stream_response = await self.llm.astream_complete(schema_prompt)

            end_time = datetime.datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            logger.info(f"Time taken for {method_name} streaming: {elapsed:.2f}s")
            return await self._process_token_stream(stream_response, model_class, method_name, prompt)

        except Exception as e:
            logger.warning(f"Token streaming failed for {method_name}: {e}")
            logger.info(f"Falling back to structured prediction for {method_name}")
            try:
                fallback_result = await self.llm.astructured_predict(model_class, prompt)
                if fallback_result is None:
                    logger.error(f"Fallback structured predict returned None for {method_name}")
                return fallback_result
            except Exception as fallback_error:
                logger.error(f"Fallback structured predict also failed for {method_name}: {fallback_error}")
                return None

    def _build_schema_prompt(self, model_class, prompt):
        """Build completion prompt with better schema instructions."""
        schema = model_class.model_json_schema()
        return f"""{prompt}

Please respond in valid JSON format matching this exact schema:
{json.dumps(schema, indent=2)}

Ensure your response is properly structured JSON."""

    async def _process_token_stream(self, stream_response, model_class, method_name, prompt):
        """Process token stream with improved JSON extraction."""
        accumulated_content = ""
        last_streamed_response = ""
        response_field_started = False

        async for token_chunk in stream_response:

            new_token = self._extract_token(token_chunk)
            if not new_token:
                continue

            accumulated_content += new_token

            # Improved response field extraction
            if self._should_stream_token(accumulated_content, response_field_started):
                new_content = self._extract_streamable_content(
                    accumulated_content, last_streamed_response
                )

                if new_content:
                    # Real token-by-token streaming
                    if self.stream_callback:
                        self.stream_callback(new_content, False)
                    last_streamed_response += new_content
                    response_field_started = True

        # Final callback
        if self.stream_callback:
            self.stream_callback("", True)

        # Parse final result with better error handling
        final_result = await self._parse_final_result(accumulated_content, model_class, method_name, prompt)

        # Log streamed vs. final content for debugging
        logger.info(f"COMPARISON: Streamed response was: '{last_streamed_response}'")
        if final_result and hasattr(final_result, 'response'):
            logger.info(f"COMPARISON: Final parsed response is: '{final_result.response}'")
        else:
            logger.info("COMPARISON: Final parsed result is None or has no 'response' attribute.")
        return final_result

    def _extract_token(self, token_chunk):
        if hasattr(token_chunk, 'delta') and token_chunk.delta:
            return token_chunk.delta
        elif hasattr(token_chunk, 'text') and token_chunk.text:
            return token_chunk.text
        return ""

    def _should_stream_token(self, content, started):
        """Determine if we should stream this token."""
        return started or '"response"' in content

    def _extract_streamable_content(self, accumulated_content, last_streamed):
        """Extract new streamable content with improved parsing logic."""
        import re

        # Try complete response field first
        response_match = re.search(
            r'"response"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
            accumulated_content,
            re.DOTALL
        )

        if response_match:
            current_content = response_match.group(1)
            current_content = self._unescape_json(current_content)
            return current_content[len(last_streamed):] if current_content != last_streamed else ""

        # Handle partial response field
        partial_match = re.search(r'"response"\s*:\s*"([^"]*)', accumulated_content)
        if partial_match:
            partial_content = partial_match.group(1)
            partial_content = self._unescape_json(partial_content)
            return partial_content[len(last_streamed):] if partial_content != last_streamed else ""

        return ""

    def _unescape_json(self, content):
        """Unescape JSON string."""
        return content.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')

    async def _parse_final_result(self, accumulated_content, model_class, method_name, prompt):
        """Parse final result with better error handling."""
        import json
        import re

        try:
            # Find JSON content
            json_match = re.search(r'\{.*\}', accumulated_content, re.DOTALL)
            if json_match:
                json_content = json_match.group()
                logger.info(f"Attempting to parse JSON content for {method_name}...")
                parsed_data = json.loads(json_content)
                result = model_class(**parsed_data)
                logger.info(f"Successfully parsed streaming response for {method_name}")
                return result
            else:
                logger.warning(
                    f"No JSON found in response for {method_name}. Accumulated content: {accumulated_content}")
                # Fallback to structured prediction
                try:
                    fallback_result = await self.llm.astructured_predict(model_class, prompt)
                    if fallback_result is None:
                        logger.error(f"Structured predict returned None for {method_name} - this should not happen")
                    return fallback_result
                except Exception as fallback_error:
                    logger.error(f"Fallback structured predict failed for {method_name}: {fallback_error}")
                    return None

        except Exception as e:
            logger.warning(
                f"Failed to parse response for {method_name}: {e}. Accumulated content: {accumulated_content}")
            try:
                fallback_result = await self.llm.astructured_predict(model_class, prompt)
                if fallback_result is None:
                    logger.error(f"Structured predict returned None for {method_name} - this should not happen")
                return fallback_result
            except Exception as fallback_error:
                logger.error(f"Fallback structured predict failed for {method_name}: {fallback_error}")
                return None


class MonitorMixin(UserModelHelperMixin):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def monitor(self, ctx: Context, ev: StartEvent) -> MonitorDoneEvent:
        user_message = ev.input
        await ctx.set("user_message", user_message)

        monitor_pm = MonitorPrompt()
        template = monitor_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_context=self.get_formatted_feature_context(),
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.get_chat_history_as_xml(),
            user_message=user_message,
            understanding_displays=self.understanding_displays.as_text(),
            modes_of_engagement=self.modes_of_engagement.as_text(),
        )

        monitor_prompt = PromptTemplate(prompt_str)
        monitor_result = await self._predict(MonitorResultModel, monitor_prompt, "MonitorResult")

        # Update user model from monitor result
        self.update_user_model_from_monitor(monitor_result)

        await ctx.set("monitor_result", monitor_result)
        return MonitorDoneEvent()


class AnalyzeMixin(UserModelHelperMixin):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def analyze(self, ctx: Context, ev: MonitorDoneEvent) -> AnalyzeDoneEvent:
        user_message = await ctx.get("user_message")
        monitor_result: MonitorResultModel = await ctx.get("monitor_result", None)
        if monitor_result is None:
            raise ValueError("Monitor result is None.")

        analyze_pm = AnalyzePrompt()
        template = analyze_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_context=self.get_formatted_feature_context(),
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.get_chat_history_as_xml(),
            understanding_displays=self.understanding_displays.as_text(),
            user_model=self.user_model.get_state_summary(as_dict=False),
            last_shown_explanations=self.get_formatted_last_shown_explanations(),
            user_message=user_message,
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
        )

        analyze_prompt = PromptTemplate(prompt_str)
        analyze_result = await self._predict(AnalyzeResult, analyze_prompt, "AnalyzeResult")

        # Update user model from analyze result
        self.update_user_model_from_analyze(analyze_result)

        await ctx.set("analyze_result", analyze_result)
        return AnalyzeDoneEvent()


class MonitorAnalyzeMixin(UserModelHelperMixin):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def monitor(self, ctx: Context, ev: StartEvent) -> MonitorDoneEvent:
        # Get user message
        user_message = ev.input
        await ctx.set("user_message", user_message)

        # use modular MonitorAnalyzePrompt
        ma_pm = MonitorAnalyzePrompt()
        template = ma_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_context=self.get_formatted_feature_context(),
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            understanding_displays=self.understanding_displays.as_text(),
            modes_of_engagement=self.modes_of_engagement.as_text(),
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            explanation_plan=self.user_model.get_complete_explanation_collection(as_dict=False),
            chat_history=self.get_chat_history_as_xml(),
            user_model=self.user_model.get_state_summary(as_dict=False),
            last_shown_explanations=self.get_formatted_last_shown_explanations(),
            user_message=user_message,
        )
        prompt = PromptTemplate(prompt_str)
        result = await self._predict(MonitorAnalyzeResultModel, prompt, "MonitorAnalyzeResult")

        # Update user model from combined result using helper methods
        self.update_user_model_from_monitor(result)
        self.update_user_model_from_analyze(result)

        await ctx.set("monitor_result", result)
        return MonitorDoneEvent()


class PlanMixin(UserModelHelperMixin):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def plan(self, ctx: Context, ev: AnalyzeDoneEvent) -> PlanDoneEvent:
        user_message = await ctx.get("user_message")
        last_exp = self.get_formatted_last_shown_explanations()

        plan_pm = PlanPrompt()
        template = plan_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_context=self.get_formatted_feature_context(),
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.get_chat_history_as_xml(),
            user_model=self.user_model.get_state_summary(as_dict=False),
            user_message=user_message,
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            explanation_plan=self.explanation_plan,
            last_shown_explanations=last_exp
        )

        plan_prompt = PromptTemplate(prompt_str)
        plan_result = await self._predict(PlanResultModel, plan_prompt, "PlanResult")

        # Update user model with plan result using helper method
        self.update_explanation_plan(plan_result)

        await ctx.set("plan_result", plan_result)
        return PlanDoneEvent()


class ExecuteMixin(UserModelHelperMixin, ConversationHelperMixin):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def execute(self, ctx: Context, ev: PlanDoneEvent) -> StopEvent:
        user_message = await ctx.get("user_message")
        plan_result = await ctx.get("plan_result", None)
        if plan_result is None:
            raise ValueError("Plan result is None.")

        if not all(isinstance(exp, ChosenExplanationModel) for exp in plan_result.explanation_plan):
            raise ValueError("Invalid plan_result.explanation_plan")

        # Extract target explanations and plan reasoning from plan_result
        target_explanations = plan_result.explanation_plan if plan_result.explanation_plan else []
        plan_reasoning = plan_result.reasoning

        # Get XAI explanations from target explanations for execution
        xai_list = self.user_model.get_string_explanations_from_plan(target_explanations)

        logger.info(f"Executing with XAI list: {xai_list}")

        execute_pm = ExecutePrompt()
        template = execute_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_context=self.get_formatted_feature_context(),
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.get_chat_history_as_xml(),
            user_model=self.user_model.get_state_summary(as_dict=False),
            user_message=user_message,
            plan_result=xai_list,
            plan_reasoning=plan_reasoning,
            explanation_plan=target_explanations,
        )

        execute_prompt = PromptTemplate(prompt_str)
        execute_result = await self._predict(ExecuteResult, execute_prompt, "ExecuteResult")

        # Update user model with execute results
        self.update_user_model_from_execute(execute_result, target_explanations)

        # Process explanations after execution using universal helper (marks as shown + updates plan tracking)
        self.process_explanations_after_execution(target_explanations)

        # Update conversation history
        self.update_conversation_history(user_message, execute_result.response)

        # Process any visual explanations in the response
        execute_result.response = replace_plot_placeholders(execute_result.response, self.visual_explanations_dict)

        return StopEvent(result=execute_result)


class PlanExecuteMixin(UserModelHelperMixin, ConversationHelperMixin, StreamingMixin):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def scaffolding(self, ctx: Context, ev: MonitorDoneEvent) -> StopEvent:
        user_message = await ctx.get("user_message")
        last_exp = self.get_formatted_last_shown_explanations()

        # use modular PlanExecutePrompt for scaffolding
        pe_pm = PlanExecutePrompt()
        template = pe_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_context=self.get_formatted_feature_context(),
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            user_model=self.user_model.get_state_summary(as_dict=False),
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            chat_history=self.get_chat_history_as_xml(),
            user_message=user_message,
            explanation_plan=self.explanation_plan or "",
            last_shown_explanations=last_exp,
        )
        prompt = PromptTemplate(prompt_str)
        # Use unified prediction (streaming or not) with logging
        scaff = await self._predict(PlanExecuteResultModel, prompt, "PlanExecuteResult")

        # Handle target explanations using universal helper (respects explanations_count)
        target_explanations = self.get_target_explanations_from_execute_result(scaff)

        self.update_conversation_history(user_message, scaff.response)

        # Update datapoint and log before adding visual plots
        self.user_model.reset_understanding_displays()

        # Process any visual explanations
        scaff.response = replace_plot_placeholders(scaff.response, self.visual_explanations_dict)

        # Update explanation plan first (before processing explanations to avoid duplication)
        self.update_explanation_plan(scaff)

        # For first question (plan creation), mark explanations as shown but preserve the plan intact
        # This is different from subsequent questions where we want to modify the plan
        if target_explanations:
            from llm_agents.explanation_state import ExplanationState
            # Mark all explanations as shown in the user model
            for explanation in target_explanations:
                if explanation:
                    self.user_model.update_explanation_step_state(
                        explanation.explanation_name,
                        explanation.step_name,
                        ExplanationState.SHOWN.value
                    )

            # Update tracking but don't remove from plan (preserve for second question)
            self.update_last_shown_explanations(target_explanations)

        await ctx.set("scaffolding_result", scaff)
        return StopEvent(result=scaff)


class UnifiedMixin(UnifiedHelperMixin, StreamingMixin):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def unified_mape_k(self, ctx: Context, ev: StartEvent) -> StopEvent:
        user_message = ev.input
        await ctx.set("user_message", user_message)

        # use SinglePromptPrompt for unified call
        sp_pm = UnifiedPrompt()
        template = sp_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_context=self.get_formatted_feature_context(),
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            understanding_displays=self.understanding_displays.as_text(),
            modes_of_engagement=self.modes_of_engagement.as_text(),
            chat_history=self.get_chat_history_as_xml(),
            user_message=user_message,
            user_model=self.user_model.get_state_summary(as_dict=False),
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            explanation_plan=self.format_predefined_plan_for_prompt(),
            last_shown_explanations=self.get_formatted_last_shown_explanations(),
        )

        # Wrap the prompt string in a PromptTemplate for structured prediction
        unified_prompt = PromptTemplate(prompt_str)
        # Use unified prediction (streaming or not) with logging
        result: SinglePromptResultModel = await self._predict(SinglePromptResultModel, unified_prompt,
                                                              "SinglePromptResult")

        # Process the unified result using helper method
        # This handles all MAPE-K phases in one go
        self.process_unified_result(user_message, result)

        # Prepare final result for workflow
        final = ExecuteResult(
            reasoning=result.reasoning,
            response=replace_plot_placeholders(result.response, self.visual_explanations_dict),
        )
        return StopEvent(result=final)


# Composed agent classes

class MapeK4BaseAgent(Workflow, LlamaIndexBaseAgent, MonitorMixin, AnalyzeMixin, PlanMixin, ExecuteMixin,
                      StreamingMixin, BaseAgentInitMixin):
    """
    Full 4-step MAPE-K agent: separate Monitor, Analyze, Plan, Execute steps.
    """

    def __init__(self, llm: LLM = None, structured_output: bool = True, timeout: float = 100.0, **kwargs):
        # Initialize LlamaIndexBaseAgent with all the base parameters
        LlamaIndexBaseAgent.__init__(self, **kwargs)

        # Initialize agent-specific components
        self._init_agent_components(
            llm=llm,
            structured_output=structured_output,
            timeout=timeout,
            use_mini_llm=True,  # This agent needs mini_llm
            **kwargs
        )


class MapeK2BaseAgent(Workflow, LlamaIndexBaseAgent, MonitorAnalyzeMixin, PlanExecuteMixin, StreamingMixin,
                      BaseAgentInitMixin):
    """
    2-step MAPE-K agent: combines Monitor+Analyze and Plan+Execute steps with streaming.
    """

    def __init__(self, llm: LLM = None, structured_output: bool = True, timeout: float = 100.0, **kwargs):
        # Initialize LlamaIndexBaseAgent with all the base parameters
        LlamaIndexBaseAgent.__init__(self, **kwargs)

        # Initialize agent-specific components
        self._init_agent_components(
            llm=llm,
            structured_output=structured_output,
            timeout=timeout,
            **kwargs
        )


class MapeKUnifiedBaseAgent(Workflow, LlamaIndexBaseAgent, UnifiedMixin, StreamingMixin, BaseAgentInitMixin):
    """
    Unified MAPE-K agent: performs all MAPE-K steps in a single LLM call with streaming support.
    """

    def __init__(self, llm: LLM = None, structured_output: bool = True, timeout: float = 100.0, **kwargs):
        # Initialize LlamaIndexBaseAgent with all the base parameters
        LlamaIndexBaseAgent.__init__(self, **kwargs)

        # Initialize with special reasoning model
        self._init_agent_components(
            llm=llm,
            structured_output=structured_output,
            timeout=timeout,
            special_model=OPENAI_REASONING_MODEL_NAME,
            **kwargs
        )


class PlanApprovalMixin(UserModelHelperMixin, ConversationHelperMixin):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def plan_approval(self, ctx: Context, ev: AnalyzeDoneEvent) -> PlanDoneEvent:
        user_message = await ctx.get("user_message")
        last_exp = self.last_shown_explanations if self.last_shown_explanations else None

        # Get the predefined plan as a formatted string
        predefined_plan_str = self.format_predefined_plan_for_prompt()

        plan_approval_pm = PlanApprovalPrompt()
        template = plan_approval_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_context=self.get_formatted_feature_context(),
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.get_chat_history_as_xml(),
            user_model=self.user_model.get_state_summary(as_dict=False),
            user_message=user_message,
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            explanation_plan=predefined_plan_str,
            last_shown_explanations=last_exp,
            understanding_displays=self.understanding_displays.as_text_filtered(self.user_model),
            modes_of_engagement=self.modes_of_engagement.as_text(),
        )

        plan_approval_prompt = PromptTemplate(prompt_str)
        approval_result = await self._predict(PlanApprovalModel, plan_approval_prompt, "PlanApprovalResult")

        # Update the explanation plan based on the approval result
        # This creates the updated plan that will be passed to execute
        approval_result = self.update_explanation_plan_after_approval(approval_result)

        # Update user model with the new plan result
        self.update_explanation_plan(approval_result)

        await ctx.set("approval_result", approval_result)
        return PlanDoneEvent()


class PlanApprovalExecuteMixin(UserModelHelperMixin, ConversationHelperMixin):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def plan_approval_execute(self, ctx: Context, ev: MonitorDoneEvent) -> StopEvent:
        user_message = await ctx.get("user_message")
        last_exp = self.last_shown_explanations if self.last_shown_explanations else None

        # Get the predefined plan as a formatted string
        predefined_plan_str = self.format_predefined_plan_for_prompt()

        plan_approval_execute_pm = PlanApprovalExecutePrompt()
        template = plan_approval_execute_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_context=self.get_formatted_feature_context(),
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.get_chat_history_as_xml(),
            user_model=self.user_model.get_state_summary(as_dict=False),
            user_message=user_message,
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            explanation_plan=predefined_plan_str,
            last_shown_explanations=last_exp,
            understanding_displays=self.understanding_displays.as_text_filtered(self.user_model),
            modes_of_engagement=self.modes_of_engagement.as_text(),
        )

        plan_approval_execute_prompt = PromptTemplate(prompt_str)
        result = await self._predict(PlanApprovalExecuteResultModel, plan_approval_execute_prompt, "PlanApprovalExecuteResult", prompt_str)

        # Check if result is None and handle gracefully
        if result is None:
            logger.error("Plan Approval Execute returned None - creating fallback result")
            result = PlanApprovalExecuteResultModel(
                reasoning="Error occurred during processing - using fallback",
                approved=True,  # Default to approved to continue with existing plan
                next_response=None,
                rendered_step_names=[],
                response="I apologize, but I encountered a technical issue"
            )

        # Get target explanations from ExecuteResult's rendered_step_names (LLM determines what was shown)
        target_explanations = self.get_target_explanations_from_execute_result(result)
        
        # Defensive check to ensure target_explanations is not None
        if target_explanations is None:
            logger.warning("get_target_explanations_from_execute_result returned None - using empty list")
            target_explanations = []

        # Update user model with plan result using helper method (before plan updates)
        plan_result = self.update_explanation_plan_after_approval(result)
        self.update_explanation_plan(plan_result)

        # Process explanations after execution using universal helper
        self.process_explanations_after_execution(target_explanations)
        self.update_conversation_history(user_message, result.response)

        # Update datapoint and log before adding visual plots
        self.user_model.reset_understanding_displays()

        # Process any visual explanations
        result.response = replace_plot_placeholders(result.response, self.visual_explanations_dict)

        await ctx.set("plan_approval_execute_result", result)
        return StopEvent(result=result)


class ConditionalPlanExecuteMixin(UserModelHelperMixin, ConversationHelperMixin, StreamingMixin):
    """
    Adaptive Plan+Execute mixin that chooses between plan creation and plan approval based on conversation state.
    
    For the first question (no existing plan): Uses PlanExecute logic (create new plan and execute)
    For subsequent questions (plan exists): Uses PlanApprovalExecute logic (approve/modify plan and execute)
    """

    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def conditional_plan_execute(self, ctx: Context, ev: MonitorDoneEvent) -> StopEvent:
        """
        Conditionally route to plan creation or plan approval based on existing explanation plan state.
        """
        # Check if this is the first question (no plan exists or is empty)
        has_existing_plan = (hasattr(self, 'explanation_plan') and
                             self.explanation_plan and
                             len(self.explanation_plan) > 0)

        if not has_existing_plan:
            # First question: Create new plan and execute (PlanExecuteMixin logic)
            return await self._plan_and_execute_new(ctx, ev)
        else:
            # Subsequent questions: Approve existing plan and execute (PlanApprovalExecuteMixin logic)  
            return await self._approve_and_execute_existing(ctx, ev)

    async def _plan_and_execute_new(self, ctx: Context, ev: MonitorDoneEvent) -> StopEvent:
        """Create a new explanation plan and execute it (first question logic)."""
        from llm_agents.models import PlanExecuteResultModel
        from llm_agents.explanation_state import ExplanationState

        user_message = await ctx.get("user_message")
        last_exp = self.get_formatted_last_shown_explanations()

        # Use modular PlanExecutePrompt for scaffolding
        pe_pm = PlanExecutePrompt()
        template = pe_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_context=self.get_formatted_feature_context(),
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            user_model=self.user_model.get_state_summary(as_dict=False),
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            chat_history=self.get_chat_history_as_xml(),
            user_message=user_message,
            explanation_plan=self.explanation_plan or "",
            last_shown_explanations=last_exp,
        )

        prompt = PromptTemplate(prompt_str)
        scaff = await self._predict(PlanExecuteResultModel, prompt, "PlanExecuteResult")

        # Check if scaffolding result is None and handle gracefully
        if scaff is None:
            logger.error("Plan Execute returned None - creating fallback result")
            scaff = PlanExecuteResultModel(
                reasoning="Error occurred during processing - using fallback",
                new_explanations=[],
                explanation_plan=[],
                rendered_step_names=[],
                response="I apologize, but I encountered a technical issue. Let me try to help you understand the model's prediction."
            )

        # Get target explanations from ExecuteResult's rendered_step_names (LLM determines what was shown)
        target_explanations = self.get_target_explanations_from_execute_result(scaff)
        
        # Defensive check to ensure target_explanations is not None
        if target_explanations is None:
            logger.warning("get_target_explanations_from_execute_result returned None - using empty list")
            target_explanations = []
        self.update_conversation_history(user_message, scaff.response)

        # Update datapoint and log before adding visual plots
        self.user_model.reset_understanding_displays()

        # Process any visual explanations
        scaff.response = replace_plot_placeholders(scaff.response, self.visual_explanations_dict)

        # Update explanation plan first (save the initial plan)
        self.update_explanation_plan(scaff)

        # Process explanations after execution using universal helper (consistent with subsequent questions)
        # This ensures explanations are marked as SHOWN in user model AND removed from plan
        self.process_explanations_after_execution(target_explanations)

        await ctx.set("scaffolding_result", scaff)
        return StopEvent(result=scaff)

    async def _approve_and_execute_existing(self, ctx: Context, ev: MonitorDoneEvent) -> StopEvent:
        """Approve/modify existing plan and execute it (subsequent questions logic)."""
        user_message = await ctx.get("user_message")
        last_exp = self.last_shown_explanations if self.last_shown_explanations else None

        # Get the predefined plan as a formatted string
        predefined_plan_str = self.format_predefined_plan_for_prompt()

        plan_approval_execute_pm = PlanApprovalExecutePrompt()
        template = plan_approval_execute_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_context=self.get_formatted_feature_context(),
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.get_chat_history_as_xml(),
            user_model=self.user_model.get_state_summary(as_dict=False),
            user_message=user_message,
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            explanation_plan=predefined_plan_str,
            last_shown_explanations=last_exp,
            understanding_displays=self.understanding_displays.as_text_filtered(self.user_model),
            modes_of_engagement=self.modes_of_engagement.as_text(),
        )

        plan_approval_execute_prompt = PromptTemplate(prompt_str)

        result = await self._predict(PlanApprovalExecuteResultModel, plan_approval_execute_prompt, "PlanApprovalExecuteResult")

        # Check if result is None and handle gracefully
        if result is None:
            logger.error("Plan Approval Execute returned None - creating fallback result")
            result = PlanApprovalExecuteResultModel(
                reasoning="Error occurred during processing - using fallback",
                approved=True,  # Default to approved to continue with existing plan
                next_response=None,
                rendered_step_names=[],
                response="I apologize, but I encountered a technical issue. Let me try to continue with the existing plan."
            )

        # Get target explanations from ExecuteResult's rendered_step_names (LLM determines what was shown)
        target_explanations = self.get_target_explanations_from_execute_result(result)
        
        # Defensive check to ensure target_explanations is not None
        if target_explanations is None:
            logger.warning("get_target_explanations_from_execute_result returned None - using empty list")
            target_explanations = []

        # Update user model with plan result using helper method (before plan updates)
        # Keep original result for conversation update, get plan result separately
        plan_result = self.update_explanation_plan_after_approval(result)
        self.update_explanation_plan(plan_result)

        # Process explanations after execution using universal helper
        self.process_explanations_after_execution(target_explanations)

        self.update_conversation_history(user_message, result.response)

        # Update datapoint and log before adding visual plots
        self.user_model.reset_understanding_displays()

        # Process any visual explanations
        result.response = replace_plot_placeholders(result.response, self.visual_explanations_dict)

        await ctx.set("plan_approval_execute_result", result)
        return StopEvent(result=result)


class MapeKApprovalBaseAgent(Workflow, LlamaIndexBaseAgent, MonitorAnalyzeMixin, ConditionalPlanExecuteMixin,
                             StreamingMixin, BaseAgentInitMixin):
    """
    2-step Adaptive MAPE-K agent with conditional plan behavior:
    - First question: Uses PlanExecute logic (creates new explanation plan and executes)
    - Subsequent questions: Uses PlanApprovalExecute logic (approves/modifies existing plan and executes)
    
    This agent intelligently switches between plan creation and plan approval based on conversation state,
    providing optimal behavior for both initial and follow-up interactions.
    """

    def __init__(self, llm: LLM = None, structured_output: bool = True, timeout: float = 100.0, **kwargs):
        # Initialize LlamaIndexBaseAgent with all the base parameters
        LlamaIndexBaseAgent.__init__(self, **kwargs)

        # Initialize with specific temperature
        self._init_agent_components(
            llm=llm or OpenAI(model=OPENAI_MODEL_NAME, temperature=0.2, reasoning_effort="low"),
            structured_output=structured_output,
            timeout=timeout,
            **kwargs
        )

    def initialize_new_datapoint(
            self,
            instance,
            xai_explanations,
            xai_visual_explanations,
            predicted_class_name: str,
            opposite_class_name: str,
            datapoint_count: int,
            use_precomputed_plan: bool = False  # Override default to False for adaptive behavior
    ) -> None:
        """
        Initialize agent with a new datapoint, explicitly disabling precomputed plans for adaptive behavior.
        
        This override ensures the ConditionalPlanExecuteMixin starts with an empty plan so it can
        properly detect "first question" vs "subsequent questions" for adaptive routing.
        """
        # Call parent with precomputed plan disabled by default
        super().initialize_new_datapoint(
            instance=instance,
            xai_explanations=xai_explanations,
            xai_visual_explanations=xai_visual_explanations,
            predicted_class_name=predicted_class_name,
            opposite_class_name=opposite_class_name,
            datapoint_count=datapoint_count,
            use_precomputed_plan=use_precomputed_plan
        )


class MapeKApproval4BaseAgent(Workflow, LlamaIndexBaseAgent, MonitorMixin, AnalyzeMixin, PlanApprovalMixin,
                              ExecuteMixin, StreamingMixin, BaseAgentInitMixin):
    """
    4-step MAPE-K agent with approval mechanism: separate Monitor, Analyze, PlanApproval, Execute steps.
    This agent evaluates predefined explanation plans in a separate step and either approves them
    or modifies them before executing.
    """

    def __init__(self, llm: LLM = None, structured_output: bool = True,
                 timeout: float = 100.0, **kwargs):
        # Initialize LlamaIndexBaseAgent with all the base parameters
        LlamaIndexBaseAgent.__init__(self, **kwargs)

        # Initialize agent-specific components
        self._init_agent_components(
            llm=llm,
            structured_output=structured_output,
            timeout=timeout,
            use_mini_llm=True,  # This agent needs mini_llm
            **kwargs
        )
