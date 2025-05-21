# mape_k_mixins.py

from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llm_agents.base_agent import OPENAI_MODEL_NAME, OPENAI_MINI_MODEL_NAME, OPENAI_REASONING_MODEL_NAME
from llm_agents.base_agent import timed

# MLflow import for logging prompts
import mlflow

from llama_index.core.workflow import Context, Event, Workflow, StartEvent, StopEvent, step
from llama_index.core.workflow.retry_policy import ConstantDelayRetryPolicy
from llm_agents.models import MonitorResultModel, AnalyzeResult, PlanResultModel, ExecuteResult

# Use new prompt mixins
from llm_agents.prompt_mixins import (
    MonitorPrompt,
    AnalyzePrompt,
    PlanPrompt,
    ExecutePrompt,
    MonitorAnalyzePrompt,
    PlanExecutePrompt,
    UnifiedPrompt,
)

# Two-step & unified prompt variants
from llm_agents.models import SinglePromptResultModel, MonitorAnalyzeResultModel, PlanExecuteResultModel

import datetime
from llama_index.core import PromptTemplate
from llm_agents.base_agent import XAIBaseAgent, append_new_log_row, update_last_log_row, logger
from llm_agents.explanation_state import ExplanationState
from llm_agents.models import ChosenExplanationModel
from llm_agents.utils.postprocess_message import replace_plot_placeholders


class MonitorDoneEvent(Event):
    pass


class AnalyzeDoneEvent(Event):
    pass


class PlanDoneEvent(Event):
    pass


class MonitorMixin:
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def monitor(self, ctx: Context, ev: StartEvent) -> MonitorDoneEvent:
        user_message = ev.input
        await ctx.set("user_message", user_message)

        self.current_log_row = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_id": self.experiment_id,
            "datapoint_count": self.datapoint_count,
            "user_message": user_message,
            "monitor": "",
            "analyze": "",
            "plan": "",
            "execute": "",
            "user_model": ""
        }
        append_new_log_row(self.current_log_row, self.log_file)

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
            monitor_result = await self.mini_llm.astructured_predict(MonitorResultModel, monitor_prompt)
            mlflow.log_param("monitor_prompt", prompt_str)
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Monitor: {end_time - start_time}")
        logger.info(f"Monitor result: {monitor_result}.\n")

        self.current_log_row["monitor"] = monitor_result
        update_last_log_row(self.current_log_row, self.log_file)
        await ctx.set("monitor_result", monitor_result)
        return MonitorDoneEvent()


class AnalyzeMixin:
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def analyze(self, ctx: Context, ev: MonitorDoneEvent) -> AnalyzeDoneEvent:
        user_message = await ctx.get("user_message")
        monitor_result: MonitorResultModel = await ctx.get("monitor_result", None)
        if monitor_result is None:
            raise ValueError("Monitor result is None.")

        if monitor_result.mode_of_engagement:
            self.user_model.cognitive_state = self.modes_of_engagement.get_differentiating_description(
                monitor_result.mode_of_engagement)
        if monitor_result.explicit_understanding_displays:
            self.user_model.explicit_understanding_signals = monitor_result.explicit_understanding_displays

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
            analyze_result = await self.mini_llm.astructured_predict(output_cls=AnalyzeResult, prompt=analyze_prompt)
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Analyze: {end_time - start_time}")
        logger.info(f"Analyze result: {analyze_result}.\n")

        for change in analyze_result.model_changes:
            if isinstance(change, dict):
                exp = change["explanation_name"];
                step = change["step"];
                state = change["state"]
            else:
                exp, state, step = change
            self.user_model.update_explanation_step_state(exp, step, state)

        await ctx.set("analyze_result", analyze_result)
        self.current_log_row["analyze"] = analyze_result
        update_last_log_row(self.current_log_row, self.log_file)
        logger.info(f"User model after analyze: {self.user_model.get_state_summary(as_dict=True)}.\n")
        return AnalyzeDoneEvent()


class MonitorAnalyzeMixin:
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def monitor(self, ctx: Context, ev: StartEvent) -> MonitorDoneEvent:
        # identical row initialization to MonitorMixin
        user_message = ev.input
        await ctx.set("user_message", user_message)
        self.current_log_row = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_id": self.experiment_id,
            "datapoint_count": self.datapoint_count,
            "user_message": user_message,
            "monitor": "",
            "analyze": "",
            "plan": "",
            "execute": "",
            "user_model": ""
        }
        append_new_log_row(self.current_log_row, self.log_file)

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
            result = await self.llm.astructured_predict(MonitorAnalyzeResultModel, prompt)
            mlflow.log_param("monitor_analyze_prompt", prompt_str)
        end = datetime.datetime.now()
        logger.info(f"Time taken for MonitorAnalyze: {end - start}")

        # update user model from combined result
        if result.mode_of_engagement:
            self.user_model.cognitive_state = self.modes_of_engagement.get_differentiating_description(
                result.mode_of_engagement)
        if result.explicit_understanding_displays:
            self.user_model.explicit_understanding_signals = result.explicit_understanding_displays
        for change in result.model_changes:
            if isinstance(change, dict):
                exp, state, step = change["explanation_name"], change["state"], change["step"]
            else:
                exp, state, step = change
            self.user_model.update_explanation_step_state(exp, step, state)

        await ctx.set("monitor_result", result)
        self.current_log_row["monitor"] = result
        update_last_log_row(self.current_log_row, self.log_file)
        return MonitorDoneEvent()


class PlanMixin:
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
            plan_result = await self.llm.astructured_predict(PlanResultModel, plan_prompt)
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Plan: {end_time - start_time}")
        logger.info(f"Plan result: {plan_result}.\n")

        if plan_result.explanation_plan:
            self.explanation_plan = plan_result.explanation_plan
        if plan_result.new_explanations:
            self.user_model.add_explanations_from_plan_result(plan_result.new_explanations)

        await ctx.set("plan_result", plan_result)
        self.current_log_row["plan"] = plan_result
        update_last_log_row(self.current_log_row, self.log_file)
        return PlanDoneEvent()


class ExecuteMixin:
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
            next_exp_content=plan_result.next_response,
        )

        start_time = datetime.datetime.now()
        with mlflow.start_run():
            # Log the execute prompt
            mlflow.log_param("execute_prompt", prompt_str)
            execute_prompt = PromptTemplate(prompt_str)
            execute_result = await self.mini_llm.astructured_predict(ExecuteResult, execute_prompt)
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Execute: {end_time - start_time}")

        self.current_log_row["execute"] = execute_result
        update_last_log_row(self.current_log_row, self.log_file)
        self.append_to_history("user", user_message)
        self.append_to_history("agent", execute_result.response)

        execute_result.response = replace_plot_placeholders(execute_result.response, self.visual_explanations_dict)

        for nxt in plan_result.next_response:
            self.user_model.update_explanation_step_state(
                nxt.explanation_name, nxt.step_name, ExplanationState.UNDERSTOOD.value)

        self.current_log_row["user_model"] = self.user_model.get_state_summary(as_dict=True)
        update_last_log_row(self.current_log_row, self.log_file)
        self.user_model.new_datapoint()
        self.last_shown_explanations.extend(plan_result.next_response)

        logger.info(f"User model after execute: {self.user_model.get_state_summary(as_dict=False)}.\n")
        return StopEvent(result=execute_result)


class PlanExecuteMixin:
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
            scaff = await self.llm.astructured_predict(PlanExecuteResultModel, prompt)
        end = datetime.datetime.now()
        logger.info(f"Time taken for Scaffolding: {end - start}")

        if scaff.explanation_plan:
            self.explanation_plan = scaff.explanation_plan
        # scaff is a PlanExecuteResultModel with next_response list
        target = scaff.next_response[0] if scaff.next_response else None
        if target and target.communication_goals:
            goal = target.communication_goals.pop(0)
            self.user_model.update_explanation_step_state(
                target.explanation_name, target.step_name, ExplanationState.SHOWN.value,
                goal + " -> " + scaff.summary_sentence
            )
            if not target.communication_goals:
                self.complete_explanation_step(target.explanation_name, target.step_name)
            self.last_shown_explanations.append(target)

        self.append_to_history("user", user_message)
        self.append_to_history("agent", scaff.response)
        scaff.response = replace_plot_placeholders(scaff.response, self.visual_explanations_dict)
        self.user_model.new_datapoint()
        await ctx.set("scaffolding_result", scaff)
        self.current_log_row["execute"] = scaff
        self.current_log_row["user_model"] = self.user_model.get_state_summary(as_dict=False)
        update_last_log_row(self.current_log_row, self.log_file)
        return StopEvent(result=scaff)


class UnifiedMixin:
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def unified_mape_k(self, ctx: Context, ev: StartEvent) -> StopEvent:
        user_message = ev.input
        await ctx.set("user_message", user_message)

        # Initialize log row
        self.current_log_row = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_id": self.experiment_id,
            "datapoint_count": self.datapoint_count,
            "user_message": user_message,
            "monitor": "",
            "analyze": "",
            "plan": "",
            "execute": "",
            "user_model": ""
        }
        append_new_log_row(self.current_log_row, self.log_file)

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
            result: SinglePromptResultModel = await self.llm.astructured_predict(SinglePromptResultModel, unified_prompt)
        end = datetime.datetime.now()
        logger.info(f"Time taken for Unified single-prompt: {end - start}")

        # Monitor & Analyze from flattened fields
        if result.mode_of_engagement:
            self.user_model.cognitive_state = self.modes_of_engagement.get_differentiating_description(
                result.mode_of_engagement)
        if result.explicit_understanding_displays:
            self.user_model.explicit_understanding_signals = result.explicit_understanding_displays
        for change in result.model_changes:
            if isinstance(change, dict):
                exp, state, step = change["explanation_name"], change["state"], change["step"]
            else:
                exp, state, step = change
            self.user_model.update_explanation_step_state(exp, step, state)

        # Log monitor/analyze
        self.current_log_row["monitor"] = {
            "mode_of_engagement": result.mode_of_engagement,
            "explicit_understanding_displays": result.explicit_understanding_displays
        }
        self.current_log_row["analyze"] = result.model_changes
        update_last_log_row(self.current_log_row, self.log_file)

        # Plan phase
        if result.explanation_plan:
            self.explanation_plan = result.explanation_plan
        if result.new_explanations:
            self.user_model.add_explanations_from_plan_result(result.new_explanations)

        # Log plan
        self.current_log_row["plan"] = {
            "new_explanations": result.new_explanations,
            "explanation_plan": result.explanation_plan,
            "next_response": result.next_response
        }
        update_last_log_row(self.current_log_row, self.log_file)

        # Execute phase
        self.append_to_history("user", user_message)
        self.append_to_history("agent", result.response)
        response_with_plots = replace_plot_placeholders(result.response, self.visual_explanations_dict)

        for target in result.next_response:
            self.user_model.update_explanation_step_state(
                target.explanation_name, target.step_name, ExplanationState.UNDERSTOOD.value)
            self.last_shown_explanations.append(target)

        # Log execute & user model
        self.current_log_row["execute"] = result.response
        self.current_log_row["user_model"] = self.user_model.get_state_summary(as_dict=False)
        update_last_log_row(self.current_log_row, self.log_file)

        # Wrap into ExecuteResult for workflow
        final = ExecuteResult(
            reasoning=result.reasoning,
            response=response_with_plots,
            summary_sentence=result.summary_sentence,
        )
        return StopEvent(result=final)


# Composed agent classes

class MapeK4Agent(Workflow, XAIBaseAgent, MonitorMixin, AnalyzeMixin, PlanMixin, ExecuteMixin):
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
            timeout: float = 100.0,
            **kwargs
    ):
        Workflow.__init__(self, timeout=timeout, **kwargs)
        XAIBaseAgent.__init__(
            self,
            experiment_id=experiment_id,
            feature_names=feature_names,
            domain_description=domain_description,
            user_ml_knowledge=user_ml_knowledge
        )
        self.llm = llm or OpenAI(model=OPENAI_MODEL_NAME)
        self.mini_llm = OpenAI(model=OPENAI_MINI_MODEL_NAME)

    @timed
    async def answer_user_question(self, user_question):
        # Kick off the 4-step workflow
        result = await self.run(input=user_question)
        # result is your StopEvent.result, i.e. an ExecuteResult
        analysis = getattr(result, "reasoning", None) or result.reasoning
        response = getattr(result, "response", None) or result.response
        return analysis, response


class MapeK2Agent(Workflow, XAIBaseAgent, MonitorAnalyzeMixin, PlanExecuteMixin):
    def __init__(
            self,
            llm: LLM = None,
            experiment_id: str = "",
            feature_names: str = "",
            domain_description: str = "",
            user_ml_knowledge: str = "",
            timeout: float = 100.0,
            **kwargs
    ):
        Workflow.__init__(self, timeout=timeout, **kwargs)
        XAIBaseAgent.__init__(
            self,
            experiment_id=experiment_id,
            feature_names=feature_names,
            domain_description=domain_description,
            user_ml_knowledge=user_ml_knowledge
        )
        self.llm = llm or OpenAI(model=OPENAI_MODEL_NAME)

    @timed
    async def answer_user_question(self, user_question):
        result = await self.run(input=user_question)
        analysis = getattr(result, "reasoning", None) or result.reasoning
        response = getattr(result, "response", None) or result.response
        return analysis, response


class MapeKUnifiedAgent(Workflow, XAIBaseAgent, UnifiedMixin):
    def __init__(
            self,
            llm: LLM = None,
            experiment_id: str = "",
            feature_names: str = "",
            domain_description: str = "",
            user_ml_knowledge: str = "",
            timeout: float = 100.0,
            **kwargs
    ):
        Workflow.__init__(self, timeout=timeout, **kwargs)
        XAIBaseAgent.__init__(
            self,
            experiment_id=experiment_id,
            feature_names=feature_names,
            domain_description=domain_description,
            user_ml_knowledge=user_ml_knowledge
        )
        self.llm = llm or OpenAI(model=OPENAI_REASONING_MODEL_NAME, reasoning_effort="low")

    @timed
    async def answer_user_question(self, user_question):
        result = await self.run(input=user_question)
        analysis = getattr(result, "reasoning", None) or result.reasoning
        response = getattr(result, "response", None) or result.response
        return analysis, response
