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

from llm_agents.utils.definition_wrapper import DefinitionWrapper
from llm_agents.mape_k_approach.plan_component.xai_exp_populator import XAIExplanationPopulator
from llm_agents.mape_k_approach.user_model.user_model_fine_grained import UserModelFineGrained as UserModel
from llm_agents.utils.postprocess_message import replace_plot_placeholders
import os
import datetime

LOG_FOLDER = "mape-k-logs"
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

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
CSV_HEADERS = ["timestamp", "experiment_id", "datapoint_count", "user_message", "monitor", "scaffolding", "user_model"]


def generate_log_file_name(experiment_id: str) -> str:
    timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H:%M")
    return os.path.join(LOG_FOLDER, f"{timestamp}_{experiment_id}.csv")


def initialize_csv(log_file: str):
    if not os.path.isfile(log_file):
        try:
            with open(log_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(CSV_HEADERS)
            logger.info(f"Initialized {log_file} with headers.\n")
        except Exception as e:
            logger.error(f"Failed to initialize CSV: {e}\n")


def append_new_log_row(row: dict, log_file: str):
    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow([row.get(header, "") for header in CSV_HEADERS])


def update_last_log_row(row: dict, log_file: str):
    with open(log_file, 'r', newline='', encoding='utf-8') as file:
        reader = list(csv.reader(file, delimiter=';'))
    if len(reader) < 2:
        return  # Nothing to update yet.
    last_row = reader[-1]
    for i, header in enumerate(CSV_HEADERS):
        last_row[i] = str(row.get(header, last_row[i]))
    reader[-1] = last_row
    with open(log_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(reader)


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
            experiment_id="",
            **kwargs
    ):
        super().__init__(timeout=100.0, **kwargs)
        self.experiment_id = experiment_id
        self.log_file = generate_log_file_name(self.experiment_id)
        initialize_csv(self.log_file)
        self.feature_names = feature_names
        self.domain_description = domain_description
        self.xai_explanations = None
        self.predicted_class_name = None
        self.instance = None
        self.datapoint_count = None
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
        self.user_model = UserModel(user_ml_knowledge)
        self.populator = None
        # Chat history
        self.chat_history = self.reset_history()
        self.explanation_plan = None
        self.last_shown_explanations = []  # save tuples of explanation and step
        self.visual_explanations_dict = None

    def reset_history(self):
        self.chat_history = "No history available, beginning of the chat."
        return self.chat_history

    def append_to_history(self, role, msg):
        if role == "user":
            msg = "User: " + msg + "\n"
        elif role == "agent":
            msg = "Agent: " + msg + "\n"

        if self.chat_history == "No history available, beginning of the chat.":
            self.chat_history = msg
        else:
            self.chat_history += msg

    # Method to initialize a new datapoint
    def initialize_new_datapoint(self,
                                 instance: InstanceDatapoint,
                                 xai_explanations,
                                 xai_visual_explanations,
                                 predicted_class_name,
                                 opposite_class_name,
                                 datapoint_count):
        # If the user_model is not empty, store understood and not understood concept information in the user model
        # and reset the rest to not_explained
        self.user_model.persist_knowledge()
        self.instance = instance.displayable_features
        self.predicted_class_name = predicted_class_name
        self.opposite_class_name = opposite_class_name
        self.datapoint_count = datapoint_count + 1
        self.reset_history()

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
        populated_yaml_json = self.populator.get_populated_json(as_dict=True)
        self.user_model.set_model_from_summary(populated_yaml_json)
        self.visual_explanations_dict = xai_visual_explanations
        self.last_shown_explanations = []

    def complete_explanation_step(self, explanation_name, step_name):
        # Delete this step from the explanation plan
        for exp in self.explanation_plan:
            if exp.explanation_name == explanation_name and exp.step == step_name:
                self.explanation_plan.remove(exp)
                break

    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=2))
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

        monitor_prompt = PromptTemplate(get_monitor_prompt_template().format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            understanding_displays=self.understanding_displays.as_text(),
            modes_of_engagement=self.modes_of_engagement.as_text(),
            explanation_plan=self.user_model.get_complete_explanation_collection(as_dict=False),
            chat_history=self.chat_history,
            user_model=self.user_model.get_state_summary(as_dict=False),
            last_shown_explanations=self.last_shown_explanations,
            user_message=user_message,
        ))
        monitor_result = await self.llm.astructured_predict(MonitorResultModel, monitor_prompt)
        logger.info(f"Monitor result: {monitor_result}.\n")
        if monitor_result.mode_of_engagement != "":
            self.user_model.cognitive_state = self.modes_of_engagement.get_differentiating_description(
                monitor_result.mode_of_engagement)

        self.user_model.explicit_understanding_signals = monitor_result.explicit_understanding_displays

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
        self.current_log_row["monitor"] = monitor_result
        update_last_log_row(self.current_log_row, self.log_file)
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
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
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
        self.append_to_history("user", user_message)
        self.append_to_history("agent", scaffolding_result.response)
        scaffolding_result.response = replace_plot_placeholders(scaffolding_result.response,
                                                                self.visual_explanations_dict)
        self.user_model.new_datapoint()
        await ctx.set("scaffolding_result", scaffolding_result)
        self.current_log_row["scaffolding"] = scaffolding_result
        self.current_log_row["user_model"] = self.user_model.get_state_summary(as_dict=False)
        update_last_log_row(self.current_log_row, self.log_file)
        return StopEvent(result=scaffolding_result)

    # Method to answer user question
    async def answer_user_question(self, user_question):
        start_time = datetime.datetime.now()
        ret = await self.run(input=user_question)
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print(f"Duration of 2 component mape-k: {duration}")
        analysis = ret.plan_reasoning
        response = ret.response
        return analysis, response


from llama_index.utils.workflow import (
    draw_all_possible_flows,
)

if __name__ == "__main__":
    # Draw all
    draw_all_possible_flows(MapeK2Component, filename="mapek_2comp_flow_all.html")
