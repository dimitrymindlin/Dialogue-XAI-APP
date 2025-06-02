# mape_k_mixins.py

from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llm_agents.agent_utils import (timed, OPENAI_MODEL_NAME, OPENAI_MINI_MODEL_NAME, OPENAI_REASONING_MODEL_NAME)
import logging
import mlflow

from llama_index.core.workflow import Context, Event, Workflow, StartEvent, StopEvent, step
from llama_index.core.workflow.retry_policy import ConstantDelayRetryPolicy
from llm_agents.models import MonitorResultModel, AnalyzeResult, PlanResultModel, ExecuteResult
from llm_agents.prompt_mixins import (
    MonitorPrompt,
    AnalyzePrompt,
    PlanPrompt,
    ExecutePrompt,
    MonitorAnalyzePrompt,
    PlanExecutePrompt,
    UnifiedPrompt,
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

logger = logging.getLogger(__name__)


class MonitorDoneEvent(Event):
    pass


class AnalyzeDoneEvent(Event):
    pass


class PlanDoneEvent(Event):
    pass


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
        self.update_user_model_from_plan(plan_result)

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

        plan_reasoning = plan_result.reasoning
        xai_list = self.user_model.get_string_explanations_from_plan(plan_result.explanation_plan)

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
            explanation_plan=plan_result.explanation_plan,
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

        # Update log with execute results
        self.update_log("execute", execute_result)

        # Update conversation history
        self.update_conversation_history(user_message, execute_result.response)

        # Process visual explanations
        execute_result.response = replace_plot_placeholders(execute_result.response, self.visual_explanations_dict)

        # Update user model from execute
        self.update_user_model_from_execute(execute_result, plan_result.explanation_plan[0])

        # Finalize log row
        self.finalize_log_row()

        # Record shown explanations
        self.last_shown_explanations.extend(plan_result.next_response)

        # Update datapoint
        self.user_model.new_datapoint()
        return StopEvent(result=execute_result)


class PlanExecuteMixin(LoggingHelperMixin, UserModelHelperMixin, ConversationHelperMixin):
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
            scaff = await astructured_predict_with_fallback(
                self.llm, PlanExecuteResultModel, prompt, use_structured_output=self.structured_output
            )  # Todo: output_stream = await self.llm.astream_structured_predict(PlanExecuteResultModel, prompt) ... output_stream.response -> response to frontend,
        end = datetime.datetime.now()
        logger.info(f"Time taken for Scaffolding: {end - start}")

        # Update explanation plan from scaffolding result
        self.update_user_model_from_plan(scaff)

        # Handle target explanations
        target = scaff.explanation_plan[0] if scaff.explanation_plan else None

        self.user_model.update_explanation_step_state(
            target.explanation_name, target.step_name, ExplanationState.SHOWN.value)

        # Record shown explanations and update conversation
        self.last_shown_explanations.append(target)
        self.update_conversation_history(user_message, scaff.response)

        # Process any visual explanations
        scaff.response = replace_plot_placeholders(scaff.response, self.visual_explanations_dict)

        self.update_user_model_from_plan(scaff)
        self.update_user_model_from_execute(scaff, target)

        # Update datapoint and log
        self.user_model.new_datapoint()
        self.update_log("execute", scaff)
        self.finalize_log_row()

        await ctx.set("scaffolding_result", scaff)
        return StopEvent(result=scaff)


class UnifiedMixin(UnifiedHelperMixin):
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
            explanation_plan=self.explanation_plan,
            last_shown_explanations=self.last_shown_explanations,
        )

        start = datetime.datetime.now()
        # Log the unified single-prompt in a nested MLflow run to avoid param collisions
        with mlflow.start_run(nested=True):
            mlflow.log_param("unified_prompt", prompt_str)
            # Wrap the prompt string in a PromptTemplate for structured prediction
            unified_prompt = PromptTemplate(prompt_str)
            # Single LLM call
            result: SinglePromptResultModel = await astructured_predict_with_fallback(
                self.llm, SinglePromptResultModel, unified_prompt, use_structured_output=self.structured_output
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

class MapeK4BaseAgent(Workflow, LlamaIndexBaseAgent, MonitorMixin, AnalyzeMixin, PlanMixin, ExecuteMixin):
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


class MapeK2BaseAgent(Workflow, LlamaIndexBaseAgent, MonitorAnalyzeMixin, PlanExecuteMixin):
    """
    2-step MAPE-K agent: combines Monitor+Analyze and Plan+Execute steps.
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
        self.llm = llm or OpenAI(model=OPENAI_MODEL_NAME)
        self.structured_output = structured_output

    @timed
    async def answer_user_question(self, user_question):
        result = await self.run(input=user_question)
        analysis = getattr(result, "reasoning", None) or result.reasoning
        response = getattr(result, "response", None) or result.response
        return analysis, response


class MapeKUnifiedBaseAgent(Workflow, LlamaIndexBaseAgent, UnifiedMixin):
    """
    Unified MAPE-K agent: performs all MAPE-K steps in a single LLM call.
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
        self.llm = llm or OpenAI(model=OPENAI_REASONING_MODEL_NAME, reasoning_effort="low")
        self.structured_output = structured_output

    @timed
    async def answer_user_question(self, user_question):
        result = await self.run(input=user_question)
        analysis = getattr(result, "reasoning", None) or result.reasoning
        response = getattr(result, "response", None) or result.response
        return analysis, response


class PlanApprovalMixin(LoggingHelperMixin, UserModelHelperMixin, ConversationHelperMixin):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def plan_approval(self, ctx: Context, ev: AnalyzeDoneEvent) -> PlanDoneEvent:
        user_message = await ctx.get("user_message")
        last_exp = self.last_shown_explanations if self.last_shown_explanations else None

        # Import the new prompts and models
        from llm_agents.prompt_mixins import PlanApprovalPrompt
        from llm_agents.models import PlanApprovalModel

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

        plan_result = self.update_explanation_plan_after_approval(approval_result)

        # Update user model with plan result using helper method
        self.update_user_model_from_plan(plan_result)

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
            # Use dual-mode prediction with fallback to string parsing
            result = await astructured_predict_with_fallback(
                self.llm,
                PlanApprovalExecuteResultModel,
                plan_approval_execute_prompt,
                use_structured_output=self.structured_output
            )
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Plan Approval Execute: {end_time - start_time}")
        logger.info(f"Plan Approval Execute result: {result}.\n")

        # Determine which explanation to use based on approval decision
        target_explanation = self.get_target_explanation_from_approval(result)
        
        # Update the plan based on approval decision
        plan_result = self.update_explanation_plan_after_approval(result)
        
        # Update user model with plan result using helper method
        self.update_user_model_from_plan(plan_result)

        if target_explanation:
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
        self.user_model.new_datapoint()
        self.update_log("plan_approval_execute", result)
        self.update_log("plan", plan_result)
        self.finalize_log_row()

        await ctx.set("plan_approval_execute_result", result)
        return StopEvent(result=result)


class MapeKApprovalBaseAgent(Workflow, LlamaIndexBaseAgent, MonitorAnalyzeMixin, PlanApprovalExecuteMixin):
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
                              ExecuteMixin):
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
