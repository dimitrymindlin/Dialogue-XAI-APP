# mape_k_mixins.py

from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llm_agents.agent_utils import (timed, OPENAI_MODEL_NAME, OPENAI_MINI_MODEL_NAME, OPENAI_REASONING_MODEL_NAME)
import logging
import mlflow

from llama_index.core.workflow import Context, Event, Workflow, StartEvent, StopEvent, step
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
    LoggingHelperMixin,
    ConversationHelperMixin,
    UnifiedHelperMixin,
)

# Two-step & unified prompt variants
from llm_agents.models import SinglePromptResultModel, MonitorAnalyzeResultModel, PlanExecuteResultModel

import datetime
from llama_index.core import PromptTemplate
from llm_agents.llama_index_base_agent import LlamaIndexBaseAgent
from llm_agents.explanation_state import ExplanationState
from llm_agents.models import ChosenExplanationModel
from llm_agents.utils.postprocess_message import replace_plot_placeholders

# Add streaming support
import asyncio
from typing import Optional, Callable, AsyncGenerator, Dict, Any
import json


# Streaming callback type
StreamCallback = Optional[Callable[[str, bool], None]]


class MonitorDoneEvent(Event):
    pass


class AnalyzeDoneEvent(Event):
    pass


class PlanDoneEvent(Event):
    pass


class StreamingMixin:
    """Mixin to add streaming capability to agents."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_callback: StreamCallback = None
        self.enable_streaming: bool = False
    
    def set_stream_callback(self, callback: StreamCallback):
        """Set a callback function for streaming responses."""
        self.stream_callback = callback
        self.enable_streaming = callback is not None
    
    async def _stream_predict_with_callback(self, model_class, prompt, method_name="scaffolding"):
        """Real token-level streaming with improved JSON parsing."""
        if not self.enable_streaming or not self.stream_callback:
            return await self.llm.astructured_predict(model_class, prompt)
        
        try:
            logger.info(f"Starting real token streaming for {method_name}")
            
            # Convert structured prompt to completion prompt
            schema_prompt = self._build_schema_prompt(model_class, prompt)
            
            # Use normal streaming completion for real token-by-token streaming
            stream_response = await self.llm.astream_complete(schema_prompt)
            
            return await self._process_token_stream(stream_response, model_class, method_name, prompt)
            
        except Exception as e:
            logger.warning(f"Token streaming failed for {method_name}: {e}, using fallback")
            return await self.llm.astructured_predict(model_class, prompt)

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
                    self.stream_callback(new_content, False)
                    last_streamed_response += new_content
                    response_field_started = True
        
        # Final callback
        if self.stream_callback:
            self.stream_callback("", True)
        
        # Parse final result with better error handling
        return await self._parse_final_result(accumulated_content, model_class, method_name, prompt)

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
        
                                # Only stream new content from response field
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
                parsed_data = json.loads(json_content)
                result = model_class(**parsed_data)
                logger.info(f"Successfully parsed streaming response for {method_name}")
                return result
            else:
                logger.warning(f"No JSON found in response for {method_name}")
                # Fallback to structured prediction
                return await self.llm.astructured_predict(model_class, prompt)
            
        except Exception as e:
            logger.warning(f"Failed to parse response for {method_name}: {e}")
            return await self.llm.astructured_predict(model_class, prompt)


class MonitorMixin(LoggingHelperMixin, UserModelHelperMixin):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def monitor(self, ctx: Context, ev: StartEvent) -> MonitorDoneEvent:
        user_message = ev.input
        await ctx.set("user_message", user_message)

        # Initialize log row using helper method
        self.initialize_log_row(user_message)

        monitor_pm = MonitorPrompt()
        template = monitor_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.chat_history,
            user_message=user_message,
            understanding_displays=self.understanding_displays.as_text(),
            modes_of_engagement=self.modes_of_engagement.as_text(),
        )
        start_time = datetime.datetime.now()
        with mlflow.start_run():
            monitor_prompt = PromptTemplate(prompt_str)
            monitor_result = await astructured_predict_with_fallback(
                self.mini_llm, MonitorResultModel, monitor_prompt, use_structured_output=self.structured_output
            )
            mlflow.log_param("monitor_prompt", prompt_str)
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Monitor: {end_time - start_time}")
        logger.info(f"Monitor result: {monitor_result}.\n")

        # Update user model from monitor result
        self.update_user_model_from_monitor(monitor_result)

        # Update log with monitor results
        self.update_log("monitor", monitor_result)

        await ctx.set("monitor_result", monitor_result)
        return MonitorDoneEvent()


class AnalyzeMixin(LoggingHelperMixin, UserModelHelperMixin):
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
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.chat_history,
            understanding_displays=self.understanding_displays.as_text(),
            user_model=self.user_model.get_state_summary(as_dict=False),
            last_shown_explanations=self.last_shown_explanations,
            user_message=user_message,
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
        )

        start_time = datetime.datetime.now()
        with mlflow.start_run():
            # Log the full analyze prompt
            mlflow.log_param("analyze_prompt", prompt_str)
            analyze_prompt = PromptTemplate(prompt_str)
            analyze_result = await astructured_predict_with_fallback(
                self.mini_llm, AnalyzeResult, analyze_prompt, use_structured_output=self.structured_output
            )
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Analyze: {end_time - start_time}")
        logger.info(f"Analyze result: {analyze_result}.\n")

        # Update user model from analyze result
        self.update_user_model_from_analyze(analyze_result)

        # Update log with analyze results using helper method
        self.update_log("analyze", analyze_result)

        await ctx.set("analyze_result", analyze_result)
        return AnalyzeDoneEvent()


class MonitorAnalyzeMixin(LoggingHelperMixin, UserModelHelperMixin):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def monitor(self, ctx: Context, ev: StartEvent) -> MonitorDoneEvent:
        # Get user message
        user_message = ev.input
        await ctx.set("user_message", user_message)

        # Initialize log row using helper method
        self.initialize_log_row(user_message)

        # use modular MonitorAnalyzePrompt
        ma_pm = MonitorAnalyzePrompt()
        template = ma_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            understanding_displays=self.understanding_displays.as_text(),
            modes_of_engagement=self.modes_of_engagement.as_text(),
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            explanation_plan=self.user_model.get_complete_explanation_collection(as_dict=False),
            chat_history=self.chat_history,
            user_model=self.user_model.get_state_summary(as_dict=False),
            last_shown_explanations=self.last_shown_explanations,
            user_message=user_message,
        )
        start = datetime.datetime.now()
        # Log the combined monitor-analyze prompt
        with mlflow.start_run(nested=True):
            prompt = PromptTemplate(prompt_str)
            result = await astructured_predict_with_fallback(
                self.llm, MonitorAnalyzeResultModel, prompt, use_structured_output=self.structured_output
            )
            mlflow.log_param("monitor_analyze_prompt", prompt_str)
        end = datetime.datetime.now()
        logger.info(f"Time taken for MonitorAnalyze: {end - start}")

        # Update user model from combined result using helper methods
        self.update_user_model_from_monitor(result)
        self.update_user_model_from_analyze(result)

        # Update log with monitor results using helper method
        self.update_log("monitor", result)

        await ctx.set("monitor_result", result)
        return MonitorDoneEvent()


class PlanMixin(LoggingHelperMixin, UserModelHelperMixin):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def plan(self, ctx: Context, ev: AnalyzeDoneEvent) -> PlanDoneEvent:
        user_message = await ctx.get("user_message")
        last_exp = self.last_shown_explanations[-1] if self.last_shown_explanations else None

        plan_pm = PlanPrompt()
        template = plan_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.chat_history,
            user_model=self.user_model.get_state_summary(as_dict=False),
            user_message=user_message,
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            explanation_plan=self.explanation_plan,
            last_shown_explanations=last_exp
        )

        start_time = datetime.datetime.now()
        with mlflow.start_run():
            # Log the plan prompt
            mlflow.log_param("plan_prompt", prompt_str)
            plan_prompt = PromptTemplate(prompt_str)
            plan_result = await astructured_predict_with_fallback(
                self.llm, PlanResultModel, plan_prompt, use_structured_output=self.structured_output
            )
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Plan: {end_time - start_time}")
        logger.info(f"Plan result: {plan_result}.\n")

        # Update user model with plan result using helper method
        self.update_explanation_plan(plan_result)

        # Update log with plan results using helper method
        self.update_log("plan", plan_result)

        await ctx.set("plan_result", plan_result)
        return PlanDoneEvent()


class ExecuteMixin(LoggingHelperMixin, UserModelHelperMixin, ConversationHelperMixin):
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

        execute_pm = ExecutePrompt()
        template = execute_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.chat_history,
            user_model=self.user_model.get_state_summary(as_dict=False),
            user_message=user_message,
            plan_result=xai_list,
            plan_reasoning=plan_reasoning,
            explanation_plan=target_explanations,
        )

        start_time = datetime.datetime.now()
        with mlflow.start_run():
            # Log the execute prompt
            mlflow.log_param("execute_prompt", prompt_str)
            execute_prompt = PromptTemplate(prompt_str)
            execute_result = await astructured_predict_with_fallback(
                self.mini_llm, ExecuteResult, execute_prompt, use_structured_output=self.structured_output
            )
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Execute: {end_time - start_time}")

        # Update user model with execute results
        self.update_user_model_from_execute(execute_result, target_explanations)
        
        # Mark explanations as shown in the user model
        self.mark_explanations_as_shown(target_explanations)
        
        # Update conversation history
        self.update_conversation_history(user_message, execute_result.response)
        
        # Process any visual explanations in the response
        execute_result.response = replace_plot_placeholders(execute_result.response, self.visual_explanations_dict)
        
        # Update the last shown explanations
        self.update_last_shown_explanations(target_explanations)
        
        # Update log with execute results and finalize
        self.update_log("execute", execute_result)
        self.finalize_log_row()

        return StopEvent(result=execute_result)


class PlanExecuteMixin(LoggingHelperMixin, UserModelHelperMixin, ConversationHelperMixin, StreamingMixin):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def scaffolding(self, ctx: Context, ev: MonitorDoneEvent) -> StopEvent:
        user_message = await ctx.get("user_message")
        last_exp = self.last_shown_explanations[-1] if self.last_shown_explanations else ""

        # use modular PlanExecutePrompt for scaffolding
        pe_pm = PlanExecutePrompt()
        template = pe_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            user_model=self.user_model.get_state_summary(as_dict=False),
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            chat_history=self.chat_history,
            user_message=user_message,
            explanation_plan=self.explanation_plan or "",
            last_shown_explanations=last_exp,
        )
        start = datetime.datetime.now()
        with mlflow.start_run():
            # Log the scaffolding prompt
            mlflow.log_param("scaffolding_prompt", prompt_str)
            prompt = PromptTemplate(prompt_str)
            
            # IMPLEMENTED: Use streaming prediction with callback support
            scaff = await self._stream_predict_with_callback(PlanExecuteResultModel, prompt, "scaffolding")
            
        end = datetime.datetime.now()
        logger.info(f"Time taken for Scaffolding: {end - start}")

        # Handle target explanations
        target = scaff.explanation_plan[0] if scaff.explanation_plan else None

        self.user_model.update_explanation_step_state(
            target.explanation_name, target.step_name, ExplanationState.SHOWN.value)

        # Record shown explanations and update conversation
        self.last_shown_explanations.append(target)
        self.update_conversation_history(user_message, scaff.response)

        # Process any visual explanations
        scaff.response = replace_plot_placeholders(scaff.response, self.visual_explanations_dict)

        self.update_explanation_plan(scaff)
        self.update_user_model_from_execute(scaff, target)

        # Update datapoint and log
        self.user_model.new_datapoint()
        self.update_log("execute", scaff)
        self.finalize_log_row()

        await ctx.set("scaffolding_result", scaff)
        return StopEvent(result=scaff)

    # New streaming-enabled method for external use
    async def answer_user_question_stream(self, user_question: str, stream_callback: StreamCallback = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream-enabled version of answer_user_question.
        
        Args:
            user_question: The user's question
            stream_callback: Optional callback for streaming chunks
            
        Yields:
            Dict with streaming data: {"type": "partial"|"final", "content": str, "is_complete": bool}
        """
        if stream_callback:
            self.set_stream_callback(stream_callback)
        
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
                    analysis = getattr(result, "reasoning", None) or result.reasoning
                    response = getattr(result, "response", None) or result.response
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
            self.set_stream_callback(None)


class UnifiedMixin(UnifiedHelperMixin, StreamingMixin):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def unified_mape_k(self, ctx: Context, ev: StartEvent) -> StopEvent:
        user_message = ev.input
        await ctx.set("user_message", user_message)

        # Initialize log row using helper method
        self.initialize_log_row(user_message)

        # use SinglePromptPrompt for unified call
        sp_pm = UnifiedPrompt()
        template = sp_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            understanding_displays=self.understanding_displays.as_text(),
            modes_of_engagement=self.modes_of_engagement.as_text(),
            chat_history=self.chat_history,
            user_message=user_message,
            user_model=self.user_model.get_state_summary(as_dict=False),
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            explanation_plan=self.format_predefined_plan_for_prompt(),
            last_shown_explanations=self.last_shown_explanations,
        )

        start = datetime.datetime.now()
        # Log the unified single-prompt in a nested MLflow run to avoid param collisions
        with mlflow.start_run(nested=True):
            mlflow.log_param("unified_prompt", prompt_str)
            # Wrap the prompt string in a PromptTemplate for structured prediction
            unified_prompt = PromptTemplate(prompt_str)
            
            # Use streaming prediction with callback support
            result: SinglePromptResultModel = await self._stream_predict_with_callback(
                SinglePromptResultModel, unified_prompt, "unified"
            )
            
        end = datetime.datetime.now()
        logger.info(f"Time taken for Unified single-prompt: {end - start}")

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

class MapeK4BaseAgent(Workflow, LlamaIndexBaseAgent, MonitorMixin, AnalyzeMixin, PlanMixin, ExecuteMixin, StreamingMixin):
    """
    Full 4-step MAPE-K agent: separate Monitor, Analyze, Plan, Execute steps.
    """

    def __init__(
            self,
            llm: LLM = None,
            experiment_id: str = "",
            feature_names: str = "",
            domain_description: str = "",
            user_ml_knowledge: str = "",
            structured_output: bool = True,
            timeout: float = 100.0,
            **kwargs
    ):
        Workflow.__init__(self, timeout=timeout, **kwargs)
        LlamaIndexBaseAgent.__init__(
            self,
            experiment_id=experiment_id,
            feature_names=feature_names,
            domain_description=domain_description,
            user_ml_knowledge=user_ml_knowledge
        )
        StreamingMixin.__init__(self)  # Initialize streaming mixin
        self.llm = llm or OpenAI(model=OPENAI_MODEL_NAME)
        self.mini_llm = OpenAI(model=OPENAI_MINI_MODEL_NAME)
        self.structured_output = structured_output

    @timed
    async def answer_user_question(self, user_question):
        # Kick off the 4-step workflow
        result = await self.run(input=user_question)
        # result is your StopEvent.result, i.e. an ExecuteResult
        analysis = getattr(result, "reasoning", None) or result.reasoning
        response = getattr(result, "response", None) or result.response
        return analysis, response


class MapeK2BaseAgent(Workflow, LlamaIndexBaseAgent, MonitorAnalyzeMixin, PlanExecuteMixin, StreamingMixin):
    """
    2-step MAPE-K agent: combines Monitor+Analyze and Plan+Execute steps.
    Now with streaming support!
    """

    def __init__(
            self,
            llm: LLM = None,
            experiment_id: str = "",
            feature_names: str = "",
            domain_description: str = "",
            user_ml_knowledge: str = "",
            structured_output: bool = True,
            timeout: float = 100.0,
            **kwargs
    ):
        Workflow.__init__(self, timeout=timeout, **kwargs)
        LlamaIndexBaseAgent.__init__(
            self,
            experiment_id=experiment_id,
            feature_names=feature_names,
            domain_description=domain_description,
            user_ml_knowledge=user_ml_knowledge
        )
        StreamingMixin.__init__(self)  # Initialize streaming mixin
        self.llm = llm or OpenAI(model=OPENAI_MODEL_NAME)
        self.structured_output = structured_output

    @timed
    async def answer_user_question(self, user_question):
        result = await self.run(input=user_question)
        analysis = getattr(result, "reasoning", None) or result.reasoning
        response = getattr(result, "response", None) or result.response
        return analysis, response

    # NEW: Streaming-enabled method
    async def answer_user_question_stream(self, user_question: str, stream_callback: StreamCallback = None):
        """
        Stream-enabled version that yields partial responses.
        
        Usage:
            async for chunk in agent.answer_user_question_stream(question):
                print(chunk['content'], end='', flush=True)
        """
        async for chunk in super().answer_user_question_stream(user_question, stream_callback):
            yield chunk


class MapeKUnifiedBaseAgent(Workflow, LlamaIndexBaseAgent, UnifiedMixin, StreamingMixin):
    """
    Unified MAPE-K agent: performs all MAPE-K steps in a single LLM call.
    Now with streaming support!
    """

    def __init__(
            self,
            llm: LLM = None,
            experiment_id: str = "",
            feature_names: str = "",
            domain_description: str = "",
            user_ml_knowledge: str = "",
            structured_output: bool = True,
            timeout: float = 100.0,
            **kwargs
    ):
        Workflow.__init__(self, timeout=timeout, **kwargs)
        LlamaIndexBaseAgent.__init__(
            self,
            experiment_id=experiment_id,
            feature_names=feature_names,
            domain_description=domain_description,
            user_ml_knowledge=user_ml_knowledge
        )
        StreamingMixin.__init__(self)  # Initialize streaming mixin
        self.llm = llm or OpenAI(model=OPENAI_REASONING_MODEL_NAME, reasoning_effort="low")
        self.structured_output = structured_output

    @timed
    async def answer_user_question(self, user_question):
        result = await self.run(input=user_question)
        analysis = getattr(result, "reasoning", None) or result.reasoning
        response = getattr(result, "response", None) or result.response
        return analysis, response

    # NEW: Streaming-enabled method  
    async def answer_user_question_stream(self, user_question: str, stream_callback: StreamCallback = None):
        """Stream-enabled version for unified agent."""
        async for chunk in super().answer_user_question_stream(user_question, stream_callback):
            yield chunk


class PlanApprovalMixin(LoggingHelperMixin, UserModelHelperMixin, ConversationHelperMixin):
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
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.chat_history,
            user_model=self.user_model.get_state_summary(as_dict=False),
            user_message=user_message,
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            explanation_plan=predefined_plan_str,
            last_shown_explanations=last_exp,
            understanding_displays=self.understanding_displays.as_text_filtered(self.user_model),
            modes_of_engagement=self.modes_of_engagement.as_text(),
        )

        start_time = datetime.datetime.now()
        with mlflow.start_run():
            # Log the plan approval prompt
            mlflow.log_param("plan_approval_prompt", prompt_str)
            plan_approval_prompt = PromptTemplate(prompt_str)
            approval_result = await astructured_predict_with_fallback(
                self.llm, PlanApprovalModel, plan_approval_prompt, use_structured_output=self.structured_output
            )
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Plan Approval: {end_time - start_time}")
        logger.info(f"Plan Approval result: {approval_result}.\n")

        # Update the explanation plan based on the approval result
        # This creates the updated plan that will be passed to execute
        plan_result = self.update_explanation_plan_after_approval(approval_result)

        # Update user model with the new plan result
        self.update_explanation_plan(plan_result)

        # Update log with plan results using helper method
        self.update_log("plan_approval", approval_result)
        self.update_log("plan", plan_result)

        await ctx.set("plan_result", plan_result)
        await ctx.set("approval_result", approval_result)
        return PlanDoneEvent()


class PlanApprovalExecuteMixin(LoggingHelperMixin, UserModelHelperMixin, ConversationHelperMixin):
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
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.chat_history,
            user_model=self.user_model.get_state_summary(as_dict=False),
            user_message=user_message,
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            explanation_plan=predefined_plan_str,
            last_shown_explanations=last_exp,
            understanding_displays=self.understanding_displays.as_text_filtered(self.user_model),
            modes_of_engagement=self.modes_of_engagement.as_text(),
        )

        start_time = datetime.datetime.now()
        with mlflow.start_run():
            # Log the plan approval execute prompt
            mlflow.log_param("plan_approval_execute_prompt", prompt_str)
            plan_approval_execute_prompt = PromptTemplate(prompt_str)
            result = await astructured_predict_with_fallback(
                self.llm,
                PlanApprovalExecuteResultModel,
                plan_approval_execute_prompt,
                use_structured_output=self.structured_output
            )
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Plan Approval Execute: {end_time - start_time}")
        logger.info(f"Plan Approval Execute result: {result}.\n")

        # Determine which explanations to use based on approval decision
        target_explanations = self.get_target_explanations_from_approval(result)

        # Update the plan based on approval decision
        plan_result = self.update_explanation_plan_after_approval(result)

        # Update user model with plan result using helper method
        self.update_explanation_plan(plan_result)

        if target_explanations:
            # Get the first explanation to mark as shown (for single explanation case)
            target_explanation = target_explanations[0]
            
            # Update user model to mark explanation as shown
            self.user_model.update_explanation_step_state(
                target_explanation.explanation_name,
                target_explanation.step_name,
                ExplanationState.SHOWN.value
            )

            # Record shown explanations and update conversation
            self.last_shown_explanations.append(target_explanation)

        self.update_conversation_history(user_message, result.response)

        # Process any visual explanations
        result.response = replace_plot_placeholders(result.response, self.visual_explanations_dict)

        # Update datapoint and log
        self.user_model.reset_understanding_displays()
        self.update_log("plan_approval_execute", result)
        self.update_log("plan", plan_result)
        self.finalize_log_row()

        await ctx.set("plan_approval_execute_result", result)
        return StopEvent(result=result)


class MapeKApprovalBaseAgent(Workflow, LlamaIndexBaseAgent, MonitorAnalyzeMixin, PlanApprovalExecuteMixin, StreamingMixin):
    """
    2-step MAPE-K agent with approval mechanism: combines Monitor+Analyze and PlanApproval+Execute steps.
    This agent evaluates predefined explanation plans and either approves them or modifies them
    based on the user's current needs and understanding state.
    """

    def __init__(
            self,
            llm: LLM = None,
            experiment_id: str = "",
            feature_names: str = "",
            domain_description: str = "",
            user_ml_knowledge: str = "",
            structured_output: bool = True,
            timeout: float = 100.0,
            **kwargs
    ):
        Workflow.__init__(self, timeout=timeout, **kwargs)
        LlamaIndexBaseAgent.__init__(
            self,
            experiment_id=experiment_id,
            feature_names=feature_names,
            domain_description=domain_description,
            user_ml_knowledge=user_ml_knowledge
        )
        StreamingMixin.__init__(self)  # Initialize streaming mixin
        self.llm = llm or OpenAI(model=OPENAI_MODEL_NAME)
        self.structured_output = structured_output

    @timed
    async def answer_user_question(self, user_question):
        """
        Process user question using the approval-based MAPE-K workflow.
        
        Args:
            user_question: The user's question/input
            
        Returns:
            Tuple of (analysis, response) where analysis contains reasoning
            and response contains the final answer to the user
        """
        result = await self.run(input=user_question)

        # Extract reasoning and response from the result
        analysis = getattr(result, "reasoning", None) or "No reasoning available"
        response = getattr(result, "response", None) or "No response available"

        return analysis, response


class MapeKApproval4BaseAgent(Workflow, LlamaIndexBaseAgent, MonitorMixin, AnalyzeMixin, PlanApprovalMixin,
                              ExecuteMixin, StreamingMixin):
    """
    4-step MAPE-K agent with approval mechanism: separate Monitor, Analyze, PlanApproval, Execute steps.
    This agent evaluates predefined explanation plans in a separate step and either approves them 
    or modifies them before executing.
    """

    def __init__(
            self,
            llm: LLM = None,
            experiment_id: str = "",
            feature_names: str = "",
            domain_description: str = "",
            user_ml_knowledge: str = "",
            structured_output: bool = True,
            timeout: float = 100.0,
            **kwargs
    ):
        Workflow.__init__(self, timeout=timeout, **kwargs)
        LlamaIndexBaseAgent.__init__(
            self,
            experiment_id=experiment_id,
            feature_names=feature_names,
            domain_description=domain_description,
            user_ml_knowledge=user_ml_knowledge
        )
        StreamingMixin.__init__(self)  # Initialize streaming mixin
        self.llm = llm or OpenAI(model=OPENAI_MODEL_NAME)
        self.mini_llm = OpenAI(model=OPENAI_MINI_MODEL_NAME)
        self.structured_output = structured_output

    @timed
    async def answer_user_question(self, user_question):
        """
        Process user question using the 4-step approval-based MAPE-K workflow.
        
        Args:
            user_question: The user's question/input
            
        Returns:
            Tuple of (analysis, response) where analysis contains reasoning
            and response contains the final answer to the user
        """
        result = await self.run(input=user_question)

        # Extract reasoning and response from the result
        analysis = getattr(result, "reasoning", None) or "No reasoning available"
        response = getattr(result, "response", None) or "No response available"

        return analysis, response
