import csv
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
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

from create_experiment_data.instance_datapoint import InstanceDatapoint
from llm_agents.base_agent import XAIBaseAgent
from llm_agents.explanation_state import ExplanationState
from llm_agents.mape_k_approach.execute_component.execute_prompt import get_execute_prompt_template, ExecuteResult
from llm_agents.mape_k_approach.analyze_component.analyze_prompt import get_analyze_prompt_template, AnalyzeResult
from llm_agents.utils.definition_wrapper import DefinitionWrapper
from llm_agents.mape_k_approach.plan_component.advanced_plan_prompt_multi_step import get_plan_prompt_template, \
    PlanResultModel, \
    ChosenExplanationModel
from llm_agents.mape_k_approach.monitor_component.monitor_prompt import get_monitor_prompt_template, MonitorResultModel
from llm_agents.mape_k_approach.user_model.user_model_fine_grained import UserModelFineGrained as UserModel
from llm_agents.mape_k_approach.plan_component.xai_exp_populator import XAIExplanationPopulator

from llama_index.core.workflow.retry_policy import ConstantDelayRetryPolicy

import logging

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
OPENAI_MINI_MODEL_NAME = os.getenv('OPENAI_MINI_MODEL_NAME')
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
CSV_HEADERS = ["timestamp", "experiment_id", "datapoint_count", "user_message", "monitor", "analyze", "plan", "execute",
               "user_model"]


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
        self.mini_llm = OpenAI(model=OPENAI_MINI_MODEL_NAME)
        self.understanding_displays = DefinitionWrapper(
            os.path.join(base_dir, "monitor_component", "understanding_displays_definition.json"))
        self.modes_of_engagement = DefinitionWrapper(
            os.path.join(base_dir, "monitor_component", "icap_modes_definition.json")
        )
        self.explanation_questions = DefinitionWrapper(
            os.path.join(base_dir, "monitor_component", "explanation_questions_definition.json")
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
        logger.info(f"User model after initialization: {self.user_model.get_state_summary(as_dict=True)}.\n")
        self.visual_explanations_dict = xai_visual_explanations
        self.last_shown_explanations = []

    def complete_explanation_step(self, explanation_name, step_name):
        # Delete this step from the explanation plan
        for exp in self.explanation_plan:
            if exp.explanation_name == explanation_name and exp.step == step_name:
                self.explanation_plan.remove(exp)
                break

    ### Monitor Step:  Interpret data from sensors, Monitoring low-level understanding, Interpreting user’s cognitive state
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
            chat_history=self.chat_history,
            user_message=user_message,
            understanding_displays=self.understanding_displays.as_text(),
            modes_of_engagement=self.modes_of_engagement.as_text(),
        ))

        start_time = datetime.datetime.now()
        monitor_result = await self.mini_llm.astructured_predict(MonitorResultModel, monitor_prompt)
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Monitor: {end_time - start_time}")

        logger.info(f"Monitor result: {monitor_result}.\n")
        # Update the 'monitor' cell
        self.current_log_row["monitor"] = monitor_result
        update_last_log_row(self.current_log_row, self.log_file)
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
        if monitor_result is None:
            raise ValueError("Monitor result is None.")

        if monitor_result.mode_of_engagement != "":
            self.user_model.cognitive_state = self.modes_of_engagement.get_differentiating_description(
                monitor_result.mode_of_engagement)

        if len(monitor_result.explicit_understanding_displays) > 0:
            self.user_model.explicit_understanding_signals = monitor_result.explicit_understanding_displays

        # Analyze the user's understanding
        analyze_prompt = PromptTemplate(get_analyze_prompt_template().format(
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
        ))

        start_time = datetime.datetime.now()
        analyze_result = await self.llm.astructured_predict(output_cls=AnalyzeResult, prompt=analyze_prompt)
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Analyze: {end_time - start_time}")
        logger.info(f"Analyze result: {analyze_result}.\n")

        if not isinstance(analyze_result, AnalyzeResult):
            raise ValueError(f"Invalid analyze result: {analyze_result} of type {type(analyze_result)}")

        # UPDATE USER MODEL
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

        await ctx.set("analyze_result", analyze_result)

        # Update user model
        logger.info(f"User model after analyze: {self.user_model.get_state_summary(as_dict=True)}.\n")
        self.current_log_row["analyze"] = analyze_result
        update_last_log_row(self.current_log_row, self.log_file)
        return AnalyzeDoneEvent()

    ### Plan Step: General adaptation plans • Choosing explanation strategy and moves
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=2))
    async def plan(self, ctx: Context, ev: AnalyzeDoneEvent) -> PlanDoneEvent:
        # Get user message
        user_message = await ctx.get("user_message")

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
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            previous_plan=self.explanation_plan,
            last_explanation=las_exp
        ))
        start_time = datetime.datetime.now()
        plan_result = await self.llm.astructured_predict(PlanResultModel, plan_prompt)
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Plan: {end_time - start_time}")

        # Update Explanation Plan
        if len(plan_result.explanation_plan) > 0:
            self.explanation_plan = plan_result.explanation_plan

        if len(plan_result.new_explanations) > 0:
            self.user_model.add_explanations_from_plan_result(plan_result.new_explanations)

        logger.info(f"Plan result: {plan_result}.\n")
        await ctx.set("plan_result", plan_result)
        self.current_log_row["plan"] = plan_result
        update_last_log_row(self.current_log_row, self.log_file)
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

        plan_reasoning = plan_result.reasoning

        xai_explanations_from_plan = self.user_model.get_string_explanations_from_plan(
            plan_result.explanation_plan)

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
            plan_reasoning=plan_reasoning,
            next_exp_content=plan_result.next_response,
        ))
        start_time = datetime.datetime.now()
        execute_result = await self.llm.astructured_predict(ExecuteResult, execute_prompt)
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Execute: {end_time - start_time}")
        await ctx.set("execute_result", copy.deepcopy(execute_result))
        logger.info(f"Execute result: {execute_result}.")

        # Log and save history before replacing placeholders
        self.current_log_row["execute"] = execute_result
        update_last_log_row(self.current_log_row, self.log_file)
        self.append_to_history("user", user_message)
        self.append_to_history("agent", execute_result.response)

        # Check if result has placeholders and replace them
        execute_result.response = replace_plot_placeholders(execute_result.response, self.visual_explanations_dict)

        # Update Explanandum state and User Model
        # 1. Update user model
        for next_explanation in plan_result.next_response:
            explanation_target = next_explanation
            exp = explanation_target.explanation_name
            exp_step = explanation_target.step_name
            self.user_model.update_explanation_step_state(exp, exp_step, ExplanationState.UNDERSTOOD.value)

        self.current_log_row["user_model"] = self.user_model.get_state_summary(as_dict=True)
        update_last_log_row(self.current_log_row, self.log_file)
        self.user_model.new_datapoint()

        """# 2. Update current explanation and explanation plan
        explanation_target.communication_goals.pop(0)
        if len(explanation_target.communication_goals) == 0:
            self.complete_explanation_step(exp, exp_step)"""

        for next_explanation in plan_result.next_response:
            self.last_shown_explanations.append(next_explanation)

        # Log new user model
        logger.info(f"User model after execute: {self.user_model.get_state_summary(as_dict=False)}.\n")
        return StopEvent(result=execute_result)

    # Method to answer user question
    async def answer_user_question(self, user_question):
        start_time = datetime.datetime.now()
        ret = await self.run(input=user_question)
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Mape-K Loop with 4 Calls: {end_time - start_time}")
        analysis = ret.reasoning
        response = ret.response
        return analysis, response


from llama_index.utils.workflow import (
    draw_all_possible_flows,
)

if __name__ == "__main__":
    # Draw all
    draw_all_possible_flows(MapeKXAIWorkflowAgent, filename="mapek_xaiagent_flow_all.html")
