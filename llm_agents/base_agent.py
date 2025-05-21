import csv
import os
from abc import ABC, abstractmethod
from datetime import datetime
import functools
from flask import current_app


def timed(fn):
    @functools.wraps(fn)
    async def wrapper(self, *args, **kwargs):
        start = datetime.now()
        result = await fn(self, *args, **kwargs)
        elapsed = datetime.now() - start
        current_app.logger.info(f"{self.__class__.__name__} answered in {elapsed}")
        return result

    return wrapper


from llama_index.core.workflow.workflow import WorkflowMeta
import logging

from create_experiment_data.instance_datapoint import InstanceDatapoint
from llm_agents.mape_k_approach.plan_component.xai_exp_populator import XAIExplanationPopulator
from llm_agents.mape_k_approach.user_model.user_model_fine_grained import UserModelFineGrained as UserModel
from llm_agents.utils.definition_wrapper import DefinitionWrapper

from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_ORGANIZATION"] = os.getenv("OPENAI_ORGANIZATION_ID")

OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_MINI_MODEL_NAME = os.getenv("OPENAI_MINI_MODEL_NAME")
OPENAI_REASONING_MODEL_NAME = os.getenv("OPENAI_REASONING_MODEL_NAME")

# Optional: Add a console handler with the same or a simpler format
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(fmt="%(levelname)s - %(message)s"))

# Create a logger specific to the current module
# Configure a file handler
file_handler = logging.FileHandler("logfile.txt", mode="w")
file_handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Base directory: the directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define a custom formatter for more readable output
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n"  # newline for readability
)
file_handler.setFormatter(formatter)

LOG_FOLDER = "mape-k-logs"
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER, exist_ok=True)

LOG_CSV_FILE = "log_table.csv"
CSV_HEADERS = ["timestamp", "experiment_id", "datapoint_count", "user_message", "monitor", "analyze", "plan", "execute",
               "user_model"]


def generate_log_file_name(experiment_id: str) -> str:
    timestamp = datetime.now().strftime("%d.%m.%Y_%H:%M")
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


class XAIBaseAgentMeta(WorkflowMeta, type(ABC)):
    pass


class XAIBaseAgent(ABC, metaclass=XAIBaseAgentMeta):

    def __init__(
            self,
            experiment_id: str,
            feature_names: str = "",
            domain_description: str = "",
            user_ml_knowledge: str = "",
            **kwargs
    ):
        # common metadata + logging setup
        self.experiment_id = experiment_id
        self.log_file = generate_log_file_name(experiment_id)
        initialize_csv(self.log_file)
        # Prepare a separate file for storing full prompts
        self.prompt_log_file = self.log_file.replace('.csv', '_prompts.txt')
        # Create or clear the prompt log
        with open(self.prompt_log_file, 'w', encoding='utf-8'):
            pass

        # common context
        self.feature_names = feature_names
        self.domain_description = domain_description
        self.xai_explanations = None
        self.predicted_class_name = None
        self.opposite_class_name = None
        self.instance = None
        self.datapoint_count = None

        # shared definitions
        self.understanding_displays = DefinitionWrapper(
            os.path.join(base_dir, "mape_k_approach", "monitor_component", "understanding_displays_definition.json")
        )
        self.modes_of_engagement = DefinitionWrapper(
            os.path.join(base_dir, "mape_k_approach", "monitor_component", "icap_modes_definition.json")
        )
        self.explanation_questions = DefinitionWrapper(
            os.path.join(base_dir, "mape_k_approach", "monitor_component", "explanation_questions_definition.json")
        )

        # user‐model + history
        self.user_model = UserModel(user_ml_knowledge)
        self.populator = None
        self.chat_history = self.reset_history()
        self.explanation_plan = []
        self.last_shown_explanations = []
        self.visual_explanations_dict = None

    def log_prompt(self, component: str, prompt_str: str):
        """
        Append the full prompt for the given component (e.g. 'monitor') to the prompt log file.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        entry = f"{timestamp} - {component.upper()} prompt:\n{prompt_str}\n\n"
        with open(self.prompt_log_file, 'a', encoding='utf-8') as f:
            f.write(entry)
        logger.info(f"{component.capitalize()} prompt recorded to {self.prompt_log_file}")

    def initialize_new_datapoint(
            self,
            instance: InstanceDatapoint,
            xai_explanations,
            xai_visual_explanations,
            predicted_class_name: str,
            opposite_class_name: str,
            datapoint_count: int
    ):
        self.instance = instance.displayable_features
        self.predicted_class_name = predicted_class_name
        self.opposite_class_name = opposite_class_name
        self.datapoint_count = datapoint_count + 1
        self.reset_history()

        self.populator = XAIExplanationPopulator(
            template_dir="/Users/dimitrymindlin/UniProjects/Dialogue-XAI-APP",
            template_file="llm_agents/mape_k_approach/plan_component/explanations_model.yaml",
            xai_explanations=xai_explanations,
            predicted_class_name=predicted_class_name,
            opposite_class_name=opposite_class_name,
            instance_dict=self.instance
        )
        self.populator.populate_yaml()
        self.populator.validate_substitutions()
        populated = self.populator.get_populated_json(as_dict=True)
        self.user_model.set_model_from_summary(populated)
        logger.info(
            f"User model after initialization: "
            f"{self.user_model.get_state_summary(as_dict=True)}.\n"
        )

        self.visual_explanations_dict = xai_visual_explanations
        self.last_shown_explanations = []

    def reset_history(self):
        self.chat_history = "No history available, beginning of the chat."
        return self.chat_history

    def append_to_history(self, role: str, msg: str):
        prefix = {"user": "User: ", "agent": "Agent: "}.get(role, "")
        entry = f"{prefix}{msg}\n"
        if self.chat_history.startswith("No history available"):
            self.chat_history = entry
        else:
            self.chat_history += entry

    def complete_explanation_step(self, explanation_name, step_name):
        for exp in self.explanation_plan:
            if exp.explanation_name == explanation_name and exp.step_name == step_name:
                self.explanation_plan.remove(exp)
                break

    @timed
    @abstractmethod
    async def answer_user_question(self, user_question):
        """
        Answer the user's question based on the initialized data point.
        Returns a tuple: (analysis, response, recommend_visualization)
        """
        pass
