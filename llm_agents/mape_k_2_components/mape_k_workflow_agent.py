import csv

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
from llm_agents.mape_k_2_components.monitor.monitor_prompt import get_monitor_prompt_template, MonitorResultModel
from llm_agents.mape_k_2_components.scaffold.scaffold_prompt import get_scaffolding_prompt_template, \
    ScaffoldingResultModel

from llama_index.core.workflow.retry_policy import ConstantDelayRetryPolicy

import logging

from llm_agents.mape_k_approach.monitor_component.definition_wrapper import DefinitionWrapper
from llm_agents.mape_k_approach.plan_component.xai_exp_populator import XAIExplanationPopulator
from llm_agents.mape_k_approach.user_model_fine_grained import UserModelFineGrained as UserModel
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
    scaffolding = await ctx.get("scaffolding_result")
    monitor_row = monitor.reasoning + "  \n->\n (" + str(
        monitor.explicit_understanding_displays) + ", " + monitor.mode_of_engagement + ")"

    scaffolding_row = scaffolding.plan_reasoning + " \n->\n " + str(
        [(exp.explanation_name, exp.step) for exp in scaffolding.explanation_collection]) + \
                      " \n->\n " + str(scaffolding.next_explanation)

    row = [
        user_message,
        monitor_row,
        scaffolding_row,
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


class MapeK2Component(Workflow, XAIBaseAgent):
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
        self.understanding_displays = DefinitionWrapper(
            os.path.join(base_dir, "..", "mape_k_approach", "monitor_component",
                         "understanding_displays_definition.json"))
        self.modes_of_engagement = DefinitionWrapper(
            os.path.join(base_dir, "..", "mape_k_approach", "monitor_component", "icap_modes_definition.json")
        )
        self.explanation_questions = DefinitionWrapper(
            os.path.join(base_dir, "..", "mape_k_approach", "monitor_component",
                         "explanation_questions_definition.json")
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
        augmented_user_input = user_input
        logger.info(f"Augmented User message: {augmented_user_input}.\n")

        await ctx.set("user_message", augmented_user_input)
        return PrepEvent()

    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=2))
    async def monitor(self, ctx: Context, ev: PrepEvent) -> MonitorDoneEvent:
        user_message = await ctx.get("user_message")
        monitor_prompt = PromptTemplate(get_monitor_prompt_template().format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            understanding_displays=self.understanding_displays.as_text(),
            modes_of_engagement=self.modes_of_engagement.as_text(),
            explanation_questions=self.explanation_questions.as_text(),
            explanation_plan=self.user_model.get_complete_explanation_collection(as_dict=False),
            chat_history=self.chat_history,
            user_model=self.user_model.get_state_summary(as_dict=False),
            last_shown_explanations=self.last_shown_explanations,
            user_message=user_message,
        ))
        monitor_result = await self.llm.astructured_predict(MonitorResultModel, monitor_prompt)
        logger.info(f"Monitor result: {monitor_result}.\n")
        # Update the user model with cognitive state and explanation state changes:
        self.user_model.cognitive_state = monitor_result.mode_of_engagement
        self.user_model.current_understanding_signals = monitor_result.understanding_displays
        for change_entry in monitor_result.model_changes:
            if isinstance(change_entry, tuple):
                exp, change, step = change_entry
            elif isinstance(change_entry, dict):
                exp = change_entry.get("explanation_name")
                change = change_entry.get("state")
                step = change_entry.get("step")
            if exp is None or change is None or step is None:
                raise ValueError(f"Invalid change entry: {change_entry}")
            self.user_model.update_explanation_step_state(exp, step, change)
        await ctx.set("monitor_result", monitor_result)
        return MonitorDoneEvent()

    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=2))
    async def scaffolding(self, ctx: Context, ev: MonitorDoneEvent) -> StopEvent:
        user_message = await ctx.get("user_message")
        last_ex = self.last_shown_explanations[-1] if self.last_shown_explanations else ""
        # Use a default next explanation focus if none is available.
        next_exp_content = "explain the concept further"
        scaffolding_prompt = PromptTemplate(get_scaffolding_prompt_template().format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            user_model=self.user_model.get_state_summary(as_dict=False),
            explanation_plan=self.user_model.get_complete_explanation_collection(as_dict=False),
            chat_history=self.chat_history,
            user_message=user_message,
            previous_plan=self.explanation_plan if self.explanation_plan else "",
            last_explanation=last_ex,
            next_exp_content=next_exp_content,
        ))
        scaffolding_result = await self.llm.astructured_predict(ScaffoldingResultModel, scaffolding_prompt)
        logger.info(f"Scaffolding result: {scaffolding_result}.\n")
        if scaffolding_result.explanation_plan:
            self.explanation_plan = scaffolding_result.explanation_plan
        # Update user model using the generated next explanation target:
        explanation_target = scaffolding_result.next_explanation
        if explanation_target.communication_goals:
            next_goal = explanation_target.communication_goals[0]
            extended_content = next_goal.goal + "->" + scaffolding_result.summary_sentence
            self.user_model.update_explanation_step_state(
                explanation_target.explanation_name,
                explanation_target.step_name,
                ExplanationState.SHOWN.value,
                extended_content
            )
            explanation_target.communication_goals.pop(0)
            if not explanation_target.communication_goals:
                self.complete_explanation_step(explanation_target.explanation_name, explanation_target.step_name)
            self.last_shown_explanations.append(
                (explanation_target.explanation_name, explanation_target.step_name, extended_content))
        self.chat_history += f"User: {user_message}\n"
        self.chat_history += f"System: {scaffolding_result.response}\n"
        scaffolding_result.response = replace_plot_placeholders(scaffolding_result.response,
                                                                self.visual_explanations_dict)
        await ctx.set("scaffolding_result", scaffolding_result)
        await write_log_to_csv(ctx, self.user_model.get_state_summary(as_dict=False))
        return StopEvent(result=scaffolding_result)

    # Method to answer user question
    async def answer_user_question(self, user_question):
        ret = await self.run(input=user_question)
        analysis = ret.plan_reasoning
        response = ret.response
        return analysis, response


from llama_index.utils.workflow import (
    draw_all_possible_flows,
)

if __name__ == "__main__":
    # Draw all
    draw_all_possible_flows(MapeK2Component, filename="mapek_2comp_flow_all.html")
