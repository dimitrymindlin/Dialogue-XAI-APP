import csv
import re
import copy

from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import (
    Event,
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
import os
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

from create_experiment_data.instance_datapoint import InstanceDatapoint
from llm_agents.base_agent import XAIBaseAgent
from llm_agents.explanation_state import ExplanationState
from llm_agents.mape_k_approach.execute_component.execute_prompt import get_execute_prompt_template, ExecuteResult
from llm_agents.mape_k_approach.analyze_component.analyze_prompt import get_analyze_prompt_template, AnalyzeResult
from llm_agents.mape_k_approach.monitor_component.icap_modes import ICAPModes
from llm_agents.mape_k_approach.plan_component.advanced_plan_prompt_multi_step import get_plan_prompt_template, \
    PlanResultModel, \
    ChosenExplanationModel, ExplanationTarget
from llm_agents.mape_k_approach.monitor_component.monitor_prompt import get_monitor_prompt_template, MonitorResultModel
from llm_agents.mape_k_approach.monitor_component.understanding_displays import UnderstandingDisplays
from llm_agents.mape_k_approach.user_model_fine_grained import UserModelFineGrained as UserModel
from llm_agents.mape_k_approach.plan_component.xai_exp_populator import XAIExplanationPopulator

from llama_index.core.workflow.retry_policy import ConstantDelayRetryPolicy

import logging

from llm_agents.utils.postprocess_message import replace_plot_placeholders

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_ORGANIZATION"] = os.getenv('OPENAI_ORGANIZATION_ID')
LLM_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME')
# Create a logger specific to the current module

logger = logging.getLogger(__name__)

# Base directory: the directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Configure a file handler
file_handler = logging.FileHandler("logfile.txt", mode="w")
file_handler.setLevel(logging.INFO)

# Define a custom formatter for more readable output
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n"  # newline for readability
)
file_handler.setFormatter(formatter)

LOG_CSV_FILE = "log_table.csv"
CSV_HEADERS = ["user_message", "monitor", "analyze", "plan", "execute", "user_model"]


def initialize_csv():
    if not os.path.isfile(LOG_CSV_FILE):
        try:
            with open(LOG_CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=';')  # Set delimiter to ';'
                writer.writerow(CSV_HEADERS)  # Write headers using the delimiter
            logger.info("Initialized log_table.csv with headers.\n")
        except Exception as e:
            logger.error(f"Failed to initialize CSV: {e}\n")


async def write_log_to_csv(ctx: Context, user_model_string):
    user_message = await ctx.get("user_message")
    monitor = await ctx.get("monitor_result")
    try:
        analyze = await ctx.get("analyze_result")
        analyze_row = analyze.reasoning + " \n->\n " + str(analyze.model_changes),
    except ValueError:  # If analyze result is not set (first message)
        analyze_row = "No analyze result"
    plan = await ctx.get("plan_result")
    execute = await ctx.get("execute_result")
    monitor_row = monitor.reasoning + "  \n->\n (" + str(
        monitor.understanding_displays) + ", " + monitor.mode_of_engagement + ")"

    plan_row = plan.reasoning + " \n->\n " + str([(exp.explanation_name, exp.step) for exp in plan.explanation_plan]) + \
               " \n->\n " + str(plan.next_explanation)

    row = [
        user_message,
        monitor_row,
        analyze_row,
        plan_row,
        execute.reasoning + "  \n->\n " + execute.response,
        user_model_string
    ]
    try:
        with open(LOG_CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(row)
        logger.info("Successfully wrote workflow data to log_table.csv.\n")
    except Exception as e:
        logger.error(f"Failed to write to CSV: {e}\n")


# Optional: Add a console handler with the same or a simpler format
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(fmt="%(levelname)s - %(message)s"))

# Set up the logger with both handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


# Define custom events

class PrepEvent(Event):
    pass


class MonitorDoneEvent(Event):
    pass


class AnalyzeDoneEvent(Event):
    pass


class PlanDoneEvent(Event):
    pass


class AugmentResult(BaseModel):
    new_user_input: str


class MapeKXAIWorkflowAgent(Workflow, XAIBaseAgent):
    def __init__(
            self,
            llm: LLM = None,
            feature_names="",
            domain_description="",
            user_ml_knowledge="",
            **kwargs
    ):
        super().__init__(timeout=100.0, **kwargs)
        initialize_csv()
        self.feature_names = feature_names
        self.domain_description = domain_description
        self.xai_explanations = None
        self.predicted_class_name = None
        self.instance = None
        self.llm = llm or OpenAI(model=LLM_MODEL_NAME)
        self.understanding_displays = UnderstandingDisplays(
            os.path.join(base_dir, "monitor_component", "understanding_displays_definition.json"))
        self.modes_of_engagement = ICAPModes(
            os.path.join(base_dir, "monitor_component", "icap_modes_definition.json")
        )

        # Mape K specific setup user understanding notepad
        self.user_model = UserModel(user_ml_knowledge)  # TODO: Use the knowlede!!!
        self.populator = None
        # Chat history
        self.chat_history = ""
        self.explanation_plan = None
        self.last_shown_explanations = []  # save tuples of explanation and step
        self.visual_explanations_dict = None

    # Method to initialize a new datapoint
    async def initialize_new_datapoint(self,
                                 instance: InstanceDatapoint,
                                 xai_explanations,
                                 xai_visual_explanations,
                                 predicted_class_name,
                                 opposite_class_name):
        # If the user_model is not empty, store understood and not understood concept information in the user model
        # and reset the rest to not_explained
        self.user_model.persist_knowledge()

        self.instance = instance.displayable_features
        self.predicted_class_name = predicted_class_name
        self.opposite_class_name = opposite_class_name

        self.chat_history = ""

        # Set user model
        self.populator = XAIExplanationPopulator(
            template_dir=".",
            template_file="llm_agents/mape_k_approach/plan_component/explanations_model.yaml",
            xai_explanations=xai_explanations,
            predicted_class_name=predicted_class_name,
            opposite_class_name=opposite_class_name,
            instance_dict=self.instance
        )
        # Populate the YAML
        self.populator.populate_yaml()
        # Validate substitutions
        self.populator.validate_substitutions()
        # Optionally, retrieve as a dictionary
        populated_yaml_dict = self.populator.get_populated_yaml(as_dict=True)
        self.user_model.set_model_from_summary(populated_yaml_dict)
        self.visual_explanations_dict = xai_visual_explanations
        self.last_shown_explanations = []

    def complete_explanation_step(self, explanation_name, step_name):
        # Delete this step from the explanation plan
        for exp in self.explanation_plan:
            if exp.explanation_name == explanation_name and exp.step == step_name:
                self.explanation_plan.remove(exp)
                break

    # Step to handle new user message
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=2))
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        # Get user input
        user_input = ev.input
        """# Augment user message by history to create stand alone message
        if len(self.chat_history) > 0:
            augment_prompt = get_augment_user_question_prompt_template().format(
                chat_history=self.chat_history,
                new_user_input=user_input)
            augment_prompt = PromptTemplate(augment_prompt)
            augmented_user_input = await self.llm.astructured_predict(AugmentResult, augment_prompt)
            augmented_user_input = augmented_user_input.new_user_input
            print(f"Augmented user input: {augmented_user_input}")
        else:"""
        augmented_user_input = user_input
        logger.info(f"Augmented User message: {augmented_user_input}.\n")

        await ctx.set("user_message", augmented_user_input)
        return PrepEvent()

    ### Monitor Step:  Interpret data from sensors, Monitoring low-level understanding, Interpreting user’s cognitive state
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=2))
    async def monitor(self, ctx: Context, ev: PrepEvent) -> MonitorDoneEvent:
        user_message = await ctx.get("user_message")

        monitor_prompt = PromptTemplate(get_monitor_prompt_template().format(
            chat_history=self.chat_history,
            user_message=user_message,
            understanding_displays=self.understanding_displays.get_displays_as_text(),
            modes_of_engagement=self.modes_of_engagement.get_modes_as_text(),
        ))
        monitor_result = await self.llm.astructured_predict(MonitorResultModel, monitor_prompt)

        logger.info(f"Monitor result: {monitor_result}.\n")
        await ctx.set("monitor_result", monitor_result)

        return MonitorDoneEvent()

    ### Analyze Step: Assessing user’s cognitive state in explanation context, updating level of understanding, verifying expectations
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=2))
    async def analyze(self, ctx: Context, ev: MonitorDoneEvent) -> AnalyzeDoneEvent:
        # Skip analyze step if this is the first message
        """if self.chat_history == "":
            return AnalyzeDoneEvent()"""

        # Get user message
        user_message = await ctx.get("user_message")

        # Monitor the user's understanding
        monitor_result: MonitorResultModel = await ctx.get("monitor_result", None)

        self.user_model.cognitive_state = monitor_result.mode_of_engagement
        self.user_model.current_understanding_signals = monitor_result.understanding_displays

        # Analyze the user's understanding
        analyze_prompt = PromptTemplate(get_analyze_prompt_template().format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.chat_history,
            understanding_displays=self.understanding_displays.get_displays_as_text(),
            user_model=self.user_model.get_state_summary(as_dict=False),
            last_shown_explanations=self.last_shown_explanations,
            user_message=user_message,
            explanation_plan=self.user_model.get_explanation_plan(as_dict=False)
        ))

        analyze_result = await self.llm.astructured_predict(output_cls=AnalyzeResult, prompt=analyze_prompt)
        logger.info(f"Analyze result: {analyze_result}.\n")

        # UPDATE USER MODEL

        # Get model changes and update user model
        for change_entry in analyze_result.model_changes:
            if isinstance(change_entry, tuple):
                exp, change, step = change_entry
            elif isinstance(change_entry, dict):
                try:
                    exp = change_entry["explanation_name"]
                    change = change_entry["state"]
                    step = change_entry["step"]
                except KeyError:
                    raise ValueError(f"Invalid change entry: {change_entry} of type {type(change_entry)}")
            # Check if variables are set
            if exp is None or change is None or step is None:
                raise ValueError(f"Invalid change entry: {change_entry}")
            self.user_model.update_explanation_step_state(exp, step, change)

        # Set cognitive state in user model
        self.user_model.cognitive_state = monitor_result.mode_of_engagement

        await ctx.set("analyze_result", analyze_result)

        # Update user model
        logger.info(f"User model after analyze: {self.user_model.get_state_summary(as_dict=True)}.\n")
        return AnalyzeDoneEvent()

    ### Plan Step: General adaptation plans • Choosing explanation strategy and moves
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=2))
    async def plan(self, ctx: Context, ev: AnalyzeDoneEvent) -> PlanDoneEvent:
        # Get user message
        user_message = await ctx.get("user_message")

        monitor_result: MonitorResultModel = await ctx.get("monitor_result")

        las_exp = self.last_shown_explanations[-1] if len(self.last_shown_explanations) > 0 else None
        # Plan the next steps
        plan_prompt = PromptTemplate(get_plan_prompt_template().format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.chat_history,
            user_model=self.user_model.get_state_summary(as_dict=False),
            user_message=user_message,
            explanation_plan=self.user_model.get_explanation_plan(as_dict=False),
            previous_plan=self.explanation_plan,
            last_explanation=las_exp
        ))

        plan_result = await self.llm.astructured_predict(PlanResultModel, plan_prompt)

        # Update Explanation Plan
        if len(plan_result.explanation_plan) > 0:
            self.explanation_plan = plan_result.explanation_plan

        if plan_result.next_explanation is not None:
            self.user_model.add_explanations_from_plan_result(plan_result.next_explanation)

        logger.info(f"Plan result: {plan_result}.\n")
        await ctx.set("plan_result", plan_result)
        return PlanDoneEvent()

    ### Execute Step: Determining realization of explanation moves, Performing selected action
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=2))
    async def execute(self, ctx: Context, ev: PlanDoneEvent) -> StopEvent:
        # Get user message
        user_message = await ctx.get("user_message")

        # Plan the next steps
        plan_result = await ctx.get("plan_result", None)

        if plan_result is None:
            raise ValueError("Plan result is None.")

        # Check the types of elements in the list
        if not all(isinstance(exp, ChosenExplanationModel) for exp in plan_result.explanation_plan):
            raise ValueError(
                f"Invalid plan result: Expected a list of ChosenExplanationModel, got {type(plan_result.next_explanations)}.")

        explanation_target: ExplanationTarget = plan_result.next_explanation
        next_explanation = explanation_target.communication_goals[0]

        xai_explanations_from_plan = self.user_model.get_string_explanations_from_plan(plan_result.explanation_plan)

        # Execute the plan
        execute_prompt = PromptTemplate(get_execute_prompt_template().format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.chat_history,
            user_model=self.user_model.get_state_summary(as_dict=False),
            user_message=user_message,
            plan_result=xai_explanations_from_plan,
            next_exp_content=next_explanation.goal,
        ))

        execute_result = await self.llm.astructured_predict(ExecuteResult, execute_prompt)
        await ctx.set("execute_result", copy.deepcopy(execute_result))
        logger.info(f"Execute result: {execute_result}.")

        # Check if result has placeholders and replace them
        execute_result.response = replace_plot_placeholders(execute_result.response, self.visual_explanations_dict)

        # Update Explanandum state and User Model
        # 1. Update user model
        exp = explanation_target.explanation_name
        exp_step = explanation_target.step_name
        exp_step_content = next_explanation.goal
        extended_exp_step_content = exp_step_content + "->" + execute_result.summary_sentence
        self.user_model.update_explanation_step_state(exp, exp_step, ExplanationState.SHOWN.value,
                                                      extended_exp_step_content)

        # 2. Update current explanation and explanation plan
        explanation_target.communication_goals.pop(0)
        if len(explanation_target.communication_goals) == 0:
            self.complete_explanation_step(exp, exp_step)

        self.last_shown_explanations.append((exp, exp_step, extended_exp_step_content))

        """# Get explanation from plan and update user model
        self.last_shown_explanations = []  # Reset and fill with new plan
        suggested_explanations = plan_result.next_explanations
        for suggestion in suggested_explanations:
            exp = suggestion.explanation_name
            exp_step = suggestion.step
            self.last_shown_explanations.append((exp, exp_step))
            self.user_model.update_explanation_step_state(exp, exp_step, ExplanationState.SHOWN.value)"""

        # Log new user model
        logger.info(f"User model after execute: {self.user_model.get_state_summary(as_dict=False)}.\n")

        # Write the collected log data to CSV
        await write_log_to_csv(ctx, self.user_model.get_state_summary(as_dict=False))

        self.chat_history += f"User: {user_message}\n"
        self.chat_history += f"System: {execute_result.response}\n"
        return StopEvent(result=execute_result)

    # Method to answer user question
    async def answer_user_question(self, user_question):
        ret = await self.run(input=user_question)
        analysis = ret.reasoning
        response = ret.response
        return analysis, response


from llama_index.utils.workflow import (
    draw_all_possible_flows,
)

if __name__ == "__main__":
    # Draw all
    draw_all_possible_flows(MapeKXAIWorkflowAgent, filename="mapek_xaiagent_flow_all.html")
