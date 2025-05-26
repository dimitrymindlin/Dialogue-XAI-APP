from create_experiment_data.instance_datapoint import InstanceDatapoint
# Use the new imports from refactored modules
from llm_agents.agent_utils import timed, append_new_log_row, update_last_log_row
from llm_agents.openai_base_agent import OpenAIAgent
from datetime import datetime

from agents import Agent, Runner, trace
from dotenv import load_dotenv
import os

from llm_agents.prompt_mixins import (
    MonitorAgentSystemPrompt, AnalyzeAgentSystemPrompt, PlanAgentSystemPrompt, ExecuteAgentSystemPrompt,
    MonitorAnalyzeSystemPrompt, PlanExecuteSystemPrompt,
    HistoryPrompt, UserMessagePrompt, LastShownExpPrompt, NextExplanationPrompt,
    PreviousPlanPrompt, LastExplanationPrompt
)

from llm_agents.models import (
    MonitorResultModel,
    AnalyzeResult,
    MonitorAnalyzeResultModel,
    PlanResultModel,
    ExecuteResult,
    PlanExecuteResultModel,
    SinglePromptResultModel,
)

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
LLM_MODEL = os.getenv('OPENAI_MODEL_NAME')


def create_monitor_system_prompt(domain_description, feature_names, instance, predicted_class_name,
                                 understanding_displays, modes_of_engagement):
    monitor_pm = MonitorAgentSystemPrompt()
    template = monitor_pm.get_prompts()["default"].get_template()
    prompt_str = template.format(
        domain_description=domain_description,
        feature_names=feature_names,
        instance=instance,
        predicted_class_name=predicted_class_name,
        understanding_displays=understanding_displays.as_text(),
        modes_of_engagement=modes_of_engagement.as_text(),
    )
    return prompt_str


def create_monitor_user_prompt(chat_history, user_message):
    history_prompt = HistoryPrompt().get_prompts()["default"].get_template()
    history_str = history_prompt.format(
        chat_history=chat_history
    )
    user_message_prompt = UserMessagePrompt().get_prompts()["default"].get_template()
    user_message_str = user_message_prompt.format(
        user_message=user_message
    )
    return history_str + user_message_str


def create_analyze_system_prompt(domain_description, feature_names, instance, predicted_class_name,
                                 explanation_collection):
    analyze_pm = AnalyzeAgentSystemPrompt()
    template = analyze_pm.get_prompts()["default"].get_template()
    prompt_str = template.format(
        domain_description=domain_description,
        feature_names=feature_names,
        instance=instance,
        predicted_class_name=predicted_class_name,
        explanation_collection=explanation_collection,
    )
    return prompt_str


def create_analyze_user_prompt(chat_history, user_message, last_shown_explanations, user_model):
    history_prompt = HistoryPrompt().get_prompts()["default"].get_template()
    history_str = history_prompt.format(
        chat_history=chat_history
    )
    last_shown_prompt = LastShownExpPrompt().get_prompts()["default"].get_template()
    last_shown_str = last_shown_prompt.format(
        last_shown_explanations=last_shown_explanations
    )
    user_message_prompt = UserMessagePrompt().get_prompts()["default"].get_template()
    user_message_str = user_message_prompt.format(
        user_message=user_message
    )
    user_model_str = user_model.get_state_summary(as_dict=False)
    return history_str + last_shown_str + user_model_str + user_message_str


def create_plan_system_prompt(domain_description, feature_names, instance, predicted_class_name,
                              explanation_collection):
    plan_pm = PlanAgentSystemPrompt()
    template = plan_pm.get_prompts()["default"].get_template()
    prompt_str = template.format(
        domain_description=domain_description,
        feature_names=feature_names,
        instance=instance,
        predicted_class_name=predicted_class_name,
        explanation_collection=explanation_collection,
    )
    return prompt_str


def create_plan_user_prompt(chat_history, user_message, explanation_plan, user_model, last_shown_explanations):
    history_prompt = HistoryPrompt().get_prompts()["default"].get_template()
    history_str = history_prompt.format(
        chat_history=chat_history
    )
    user_message_prompt = UserMessagePrompt().get_prompts()["default"].get_template()
    user_message_str = user_message_prompt.format(
        user_message=user_message
    )
    user_model_str = user_model.get_state_summary(as_dict=False)
    plan_prompt = PreviousPlanPrompt().get_prompts()["default"].get_template()
    plan_str = plan_prompt.format(
        explanation_plan=explanation_plan
    )
    last_shown_prompt = LastExplanationPrompt().get_prompts()["default"].get_template()
    last_shown_str = last_shown_prompt.format(
        last_shown_explanations=last_shown_explanations
    )
    return history_str + user_message_str + user_model_str + plan_str + last_shown_str


def create_execute_system_prompt(domain_description, feature_names, instance, predicted_class_name):
    execute_pm = ExecuteAgentSystemPrompt()
    template = execute_pm.get_prompts()["default"].get_template()
    prompt_str = template.format(
        domain_description=domain_description,
        feature_names=feature_names,
        instance=instance,
        predicted_class_name=predicted_class_name,
    )
    return prompt_str


def create_execute_user_prompt(chat_history, user_message, next_exp_content):
    history_prompt = HistoryPrompt().get_prompts()["default"].get_template()
    history_str = history_prompt.format(
        chat_history=chat_history
    )
    user_message_prompt = UserMessagePrompt().get_prompts()["default"].get_template()
    user_message_str = user_message_prompt.format(
        user_message=user_message
    )
    next_exp_prompt = NextExplanationPrompt().get_prompts()["default"].get_template()
    next_exp_str = next_exp_prompt.format(
        next_exp_content=next_exp_content
    )
    return history_str + user_message_str + next_exp_str


# 3) Combined Monitor+Analyze
monitor_analyze_agent = Agent(
    name="monitor_analyze_agent",
    instructions=(
        "Combine Monitor + Analyze: First detect understanding displays and engagement mode, then propose any `model_changes`. Each `model_change` must include `explanation_name`, `step_name`, and `state`. Output JSON matching MonitorAnalyzeResultModel exactly."
    ),
    output_type=MonitorAnalyzeResultModel,
)

# 4) Plan only
plan_agent = Agent(
    name="plan_agent",
    instructions=(
        "You design an explanation plan for the user. Based on `user_model_state` "
        "and any `model_changes`, decide which new_explanations to introduce "
        "and the long‐term `explanation_plan`. Output JSON matching PlanResultModel."
    ),
    output_type=PlanResultModel,
)

# 5) Execute only
execute_agent = Agent(
    name="execute_agent",
    instructions=(
        "You generate the actual response. Given `new_explanations` and the full "
        "`explanation_plan`, produce a `summary_sentence` and a detailed `response` "
        "(up to three sentences per goal), styled with HTML. "
        "Output JSON matching ExecuteResult."
    ),
    output_type=ExecuteResult,
)

# 6) Combined Plan+Execute
plan_execute_agent = Agent(
    name="plan_execute_agent",
    instructions=(
        "Combine Plan + Execute: First create the new explanations & plan, "
        "then generate the user‐facing response. "
        "Output JSON matching PlanExecuteResultModel."
    ),
    output_type=PlanExecuteResultModel,
)

# 7) Single-prompt for everything
single_prompt_agent = Agent(
    name="single_prompt_agent",
    instructions=(
        "You are an XAI tutor performing all four MAPE-K steps in one. 1) Detect understanding and engagement. 2) Suggest `model_changes` (each with `explanation_name`, `step_name`, and `state`). 3) Build an explanation plan. 4) Generate the response. Output JSON matching SinglePromptResultModel exactly."
    ),
    output_type=SinglePromptResultModel,
)


# ---------------------------------------------------------------------------
# MAPE-K Agent using OpenAIAgent
# ---------------------------------------------------------------------------
class MapeK4OpenAIAgent(OpenAIAgent):
    def __init__(
            self,
            experiment_id: str,
            feature_names: str = "",
            domain_description: str = "",
            user_ml_knowledge: str = "",
    ):
        super().__init__(experiment_id, feature_names, domain_description, user_ml_knowledge)
        self.monitor_agent = None
        self.analyze_agent = None
        self.plan_agent = None
        self.execute_agent = None

    @timed
    async def answer_user_question(self, user_question: str):
        with trace("Deterministic MAPE-K flow"):
            # 1) Monitor
            mon_input = create_monitor_user_prompt(
                chat_history=self.chat_history,
                user_message=user_question
            )
            mon_res = await Runner.run(self.monitor_agent, mon_input)
            self.log_prompt("monitor", mon_input)

            # Update User model
            if mon_res.final_output.mode_of_engagement:
                self.user_model.cognitive_state = self.modes_of_engagement.get_differentiating_description(
                    mon_res.final_output.mode_of_engagement)
            if mon_res.final_output.explicit_understanding_displays:
                self.user_model.explicit_understanding_signals = mon_res.final_output.explicit_understanding_displays

            # 2) Analyze
            anl_input = create_analyze_user_prompt(
                chat_history=self.chat_history,
                user_message=user_question,
                last_shown_explanations=self.last_shown_explanations,
                user_model=self.user_model,
            )
            anl_res = await Runner.run(self.analyze_agent, anl_input)
            self.log_prompt("analyze", anl_input)

            # 3) Plan
            plan_input = create_plan_user_prompt(
                chat_history=self.chat_history,
                user_message=user_question,
                explanation_plan=self.explanation_plan or "",
                user_model=self.user_model,
                last_shown_explanations=self.last_shown_explanations,
            )
            plan_res = await Runner.run(self.plan_agent, plan_input)
            self.log_prompt("plan", plan_input)

            # Update explanation plan
            if plan_res.final_output.explanation_plan:
                self.explanation_plan = plan_res.final_output.explanation_plan

            # 4) Execute
            exec_input = create_execute_user_prompt(
                chat_history=self.chat_history,
                user_message=user_question,
                next_exp_content=plan_res.final_output.next_response[
                    0].communication_goals if plan_res.final_output.next_response else "",
            )
            exec_res = await Runner.run(self.execute_agent, exec_input)
            self.log_prompt("execute", exec_input)

            # Log to CSV
            row = {
                "timestamp": datetime.now().strftime("%d.%m.%Y_%H:%M"),
                "experiment_id": self.experiment_id,
                "datapoint_count": self.datapoint_count,
                "user_message": user_question,
                "monitor": mon_res.final_output,
                "analyze": anl_res.final_output,
                "plan": plan_res.final_output,
                "execute": exec_res.final_output,
                "user_model": self.user_model.get_state_summary(as_dict=False),
            }
            append_new_log_row(row, self.log_file)
            update_last_log_row(row, self.log_file)

            return exec_res.final_output.reasoning, exec_res.final_output.response

    def initialize_new_datapoint(
            self,
            instance: InstanceDatapoint,
            xai_explanations,
            xai_visual_explanations,
            predicted_class_name: str,
            opposite_class_name: str,
            datapoint_count: int
    ):
        # 1) call parent logic
        super().initialize_new_datapoint(
            instance,
            xai_explanations,
            xai_visual_explanations,
            predicted_class_name,
            opposite_class_name,
            datapoint_count
        )
        # 2) now initialize your LLM agents
        self.monitor_agent = Agent(
            name="monitor_agent",
            model=LLM_MODEL,
            instructions=create_monitor_system_prompt(
                domain_description=self.domain_description,
                feature_names=self.feature_names,
                instance=self.instance,
                predicted_class_name=self.predicted_class_name,
                understanding_displays=self.understanding_displays,
                modes_of_engagement=self.modes_of_engagement,
            ),
            output_type=MonitorResultModel,
        )
        self.analyze_agent = Agent(
            name="analyze_agent",
            model=LLM_MODEL,
            instructions=create_analyze_system_prompt(
                domain_description=self.domain_description,
                feature_names=self.feature_names,
                instance=self.instance,
                predicted_class_name=self.predicted_class_name,
                explanation_collection=xai_explanations,
            ),
            output_type=AnalyzeResult,
        )
        self.plan_agent = Agent(
            name="plan_agent",
            model=LLM_MODEL,
            instructions=create_plan_system_prompt(
                domain_description=self.domain_description,
                feature_names=self.feature_names,
                instance=self.instance,
                predicted_class_name=self.predicted_class_name,
                explanation_collection=xai_explanations,
            ),
            output_type=PlanResultModel,
        )
        self.execute_agent = Agent(
            name="execute_agent",
            model=LLM_MODEL,
            instructions=create_execute_system_prompt(
                domain_description=self.domain_description,
                feature_names=self.feature_names,
                instance=self.instance,
                predicted_class_name=self.predicted_class_name,
            ),
            output_type=ExecuteResult,
        )


class MapeK2OpenAIAgent(OpenAIAgent):
    def __init__(
            self,
            experiment_id: str,
            feature_names: str = "",
            domain_description: str = "",
            user_ml_knowledge: str = "",
    ):
        super().__init__(experiment_id, feature_names, domain_description, user_ml_knowledge)
        self.monitor_analyze_agent = None
        self.plan_execute_agent = None

    def initialize_new_datapoint(
            self,
            instance: InstanceDatapoint,
            xai_explanations,
            xai_visual_explanations,
            predicted_class_name: str,
            opposite_class_name: str,
            datapoint_count: int
    ):
        super().initialize_new_datapoint(
            instance,
            xai_explanations,
            xai_visual_explanations,
            predicted_class_name,
            opposite_class_name,
            datapoint_count
        )
        # Monitor+Analyze agent
        self.monitor_analyze_agent = Agent(
            name="monitor_analyze_agent",
            model=LLM_MODEL,
            instructions=MonitorAnalyzeSystemPrompt().get_prompts()["default"].get_template().format(
                domain_description=self.domain_description,
                feature_names=self.feature_names,
                instance=self.instance,
                predicted_class_name=self.predicted_class_name,
                understanding_displays=self.understanding_displays.as_text(),
                modes_of_engagement=self.modes_of_engagement.as_text(),
                explanation_collection=xai_explanations,
                chat_history=self.chat_history,
                user_model=self.user_model.get_state_summary(as_dict=False),
                last_shown_explanations=self.last_shown_explanations,
            ),
            output_type=MonitorAnalyzeResultModel,
        )
        # Plan+Execute agent
        self.plan_execute_agent = Agent(
            name="plan_execute_agent",
            model=LLM_MODEL,
            instructions=PlanExecuteSystemPrompt().get_prompts()["default"].get_template().format(
                domain_description=self.domain_description,
                feature_names=self.feature_names,
                instance=self.instance,
                predicted_class_name=self.predicted_class_name,
                explanation_collection=xai_explanations,
                chat_history=self.chat_history,
                user_model=self.user_model.get_state_summary(as_dict=False),
                explanation_plan=self.explanation_plan or "",
                last_shown_explanations=self.last_shown_explanations,
            ),
            output_type=PlanExecuteResultModel,
        )

    @timed
    async def answer_user_question(self, user_question: str):
        with trace("MAPE-K 2-step flow: Monitor+Analyze, then Plan+Execute"):
            # 1) Monitor+Analyze
            ma_input = create_monitor_user_prompt(
                chat_history=self.chat_history,
                user_message=user_question
            )
            ma_res = await Runner.run(self.monitor_analyze_agent, ma_input)
            self.log_prompt("monitor_analyze", ma_input)

            # Update user model from monitor+analyze
            if ma_res.final_output.mode_of_engagement:
                self.user_model.cognitive_state = self.modes_of_engagement.get_differentiating_description(
                    ma_res.final_output.mode_of_engagement)
            if ma_res.final_output.explicit_understanding_displays:
                self.user_model.explicit_understanding_signals = ma_res.final_output.explicit_understanding_displays
            for change in getattr(ma_res.final_output, "model_changes", []):
                if isinstance(change, dict):
                    exp, step, state = change["explanation_name"], change["step"], change["state"]
                else:
                    exp, state, step = change
                self.user_model.update_explanation_step_state(exp, step, state)

            # 2) Plan+Execute
            pe_input = create_plan_user_prompt(
                chat_history=self.chat_history,
                user_message=user_question,
                explanation_plan=self.explanation_plan or "",
                user_model=self.user_model,
                last_shown_explanations=self.last_shown_explanations,
            )
            pe_res = await Runner.run(self.plan_execute_agent, pe_input)
            self.log_prompt("plan_execute", pe_input)

            # Update explanation plan and shown explanations
            if pe_res.final_output.explanation_plan:
                self.explanation_plan = pe_res.final_output.explanation_plan
            if hasattr(pe_res.final_output, "next_response") and pe_res.final_output.next_response:
                self.last_shown_explanations.extend(pe_res.final_output.next_response)

            # Log to CSV
            row = {
                "timestamp": datetime.now().strftime("%d.%m.%Y_%H:%M"),
                "experiment_id": self.experiment_id,
                "datapoint_count": self.datapoint_count,
                "user_message": user_question,
                "monitor": ma_res.final_output,
                "analyze": ma_res.final_output,
                "plan": pe_res.final_output,
                "execute": pe_res.final_output,
                "user_model": self.user_model.get_state_summary(as_dict=False),
            }
            append_new_log_row(row, self.log_file)
            update_last_log_row(row, self.log_file)

            return getattr(pe_res.final_output, "reasoning", None), getattr(pe_res.final_output, "response", None)
