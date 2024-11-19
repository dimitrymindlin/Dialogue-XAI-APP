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
from llm_agents.base_agent import XAIBaseAgent
from llm_agents.mape_k_approach.mape_k_prompts import get_monitor_prompt_template, get_analyze_prompt_template, \
    get_plan_prompt_template, get_execute_prompt_template
from llm_agents.mape_k_approach.user_model import UserModel

from llm_agents.xai_utils import process_xai_explanations, extract_instance_information, \
    get_xai_explanations_as_goal_notepad, get_xai_explanations_as_goal_user_model
from llm_agents.xai_prompts import get_augment_user_question_prompt_template
import logging

# Create a logger specific to the current module
logger = logging.getLogger(__name__)

# Configure a file handler
file_handler = logging.FileHandler("logfile.txt", mode="w")
file_handler.setLevel(logging.INFO)

# Define a custom formatter for more readable output
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n"  # newline for readability
)
file_handler.setFormatter(formatter)

# Optional: Add a console handler with the same or a simpler format
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(fmt="%(levelname)s - %(message)s"))

# Set up the logger with both handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


def turn_xai_explanations_to_notepad(xai_explanations):
    pass


explanation_plan = ["TopFeatureImportance",
                    "FeatureInfluences",
                    "CeterisParibus most important feature",
                    "CeterisParibus second most important feature",
                    "Counterfactuals",
                    "Anchors",
                    "FeatureStatistics of top feature",
                    "FeatureStatistics of second top feature", ]


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


class MonitorResult(BaseModel):
    reasoning: str
    monitor_result: str


class AnalyzeResult(BaseModel):
    reasoning: str
    model_changes: list


class PlanResult(BaseModel):
    reasoning: str
    next_explanation: str


class ExecuteResult(BaseModel):
    reasoning: str
    response: str


class MapeKXAIWorkflowAgent(Workflow, XAIBaseAgent):
    def __init__(
            self,
            llm: LLM = None,
            feature_names="",
            domain_description="",
            **kwargs
    ):
        super().__init__(timeout=50.0, **kwargs)
        self.feature_names = feature_names
        self.domain_description = domain_description
        self.xai_explanations = None
        self.predicted_class_name = None
        self.instance = None
        self.llm = llm or OpenAI()

        # Mape K specific setup user understanding notepad
        self.user_model = UserModel()

        # Chat history
        self.chat_history = ""

    # Method to initialize a new datapoint
    def initialize_new_datapoint(self,
                                 instance_information,
                                 xai_explanations,
                                 predicted_class_name,
                                 opposite_class_name):
        self.xai_explanations = process_xai_explanations(xai_explanations, predicted_class_name, opposite_class_name)
        """self.user_knowledge_base["user_understanding_notepad"][
            "not explained yet"] = get_xai_explanations_as_goal_notepad(xai_explanations)"""
        get_xai_explanations_as_goal_user_model(xai_explanations, self.user_model)
        self.instance = extract_instance_information(instance_information)
        self.predicted_class_name = predicted_class_name
        self.opposite_class_name = opposite_class_name
        self.chat_history = ""

    # Step to handle new user message
    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        # Get user input
        user_input = ev.input
        # Augment user message by history to create stand alone message
        if len(self.chat_history) > 0:
            augment_prompt = get_augment_user_question_prompt_template().format(
                chat_history=self.chat_history,
                new_user_input=user_input)
            augment_prompt = PromptTemplate(augment_prompt)
            augmented_user_input = await self.llm.astructured_predict(AugmentResult, augment_prompt)
            augmented_user_input = augmented_user_input.new_user_input
            print(f"Augmented user input: {augmented_user_input}")
        else:
            augmented_user_input = user_input
            logger.info(f"Augmented User message: {augmented_user_input}.\n")

        await ctx.set("user_message", augmented_user_input)
        return PrepEvent()

    ### Monitor Step:  Interpret data from sensors, Monitoring low-level understanding, Interpreting user’s cognitive state
    @step
    async def monitor(self, ctx: Context, ev: PrepEvent) -> MonitorDoneEvent:
        user_message = await ctx.get("user_message")

        monitor_prompt = PromptTemplate(get_monitor_prompt_template().format(
            chat_history=self.chat_history,
            user_message=user_message
        ))
        monitor_result = await self.llm.astructured_predict(MonitorResult, monitor_prompt)

        logger.info(f"Monitor result: {monitor_result}.\n")
        await ctx.set("monitor_result", monitor_result.monitor_result)
        return MonitorDoneEvent()

    ### Analyze Step: Assessing user’s cognitive state in explanation context, updating level of understanding, verifying expectations
    @step()
    async def analyze(self, ctx: Context, ev: MonitorDoneEvent) -> AnalyzeDoneEvent:
        # Skip analyze step if this is the first message
        if self.chat_history == "":
            return AnalyzeDoneEvent()
        # Get user message
        user_message = await ctx.get("user_message")

        # Get user model
        user_model = self.user_model.get_summary()

        # Monitor the user's understanding
        monitor_result = await ctx.get("monitor_result", None)

        # Analyze the user's understanding
        analyze_prompt = PromptTemplate(get_analyze_prompt_template().format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.chat_history,
            user_model=user_model,
            user_question=user_message,
            monitor_result=monitor_result
        ))

        response = await self.llm.astructured_predict(AnalyzeResult, analyze_prompt)
        print(f"Analyze result: {response}")
        for change_entry in response.model_changes:
            if isinstance(change_entry, tuple):
                exp, change = change_entry
            elif isinstance(change_entry, dict):
                try:
                    exp = change_entry["explanation_type"]
                    change = change_entry["change"]
                except KeyError:
                    raise ValueError(f"Invalid change entry: {change_entry} of type {type(change_entry)}")
            if change == "understood":
                self.user_model.is_understood(exp)
            elif change == "misunderstood":
                self.user_model.is_misunderstood(exp)

        logger.info(f"Analysis result: {response}.\n")
        # Update user model
        logger.info(f"User model after analyze: {self.user_model.get_summary()}.\n")
        return AnalyzeDoneEvent()

    ### Plan Step: General adaptation plans • Choosing explanation strategy and moves
    @step()
    async def plan(self, ctx: Context, ev: AnalyzeDoneEvent) -> PlanDoneEvent:
        # Get user message
        user_message = await ctx.get("user_message")

        # Get user model
        user_model = self.user_model.get_summary()

        # Plan the next steps
        plan_prompt = PromptTemplate(get_plan_prompt_template().format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            xai_explanations=self.xai_explanations,
            chat_history=self.chat_history,
            user_model=user_model,
            user_question=user_message,
            explanation_plan=explanation_plan
        ))
        plan_result = await self.llm.astructured_predict(PlanResult, plan_prompt)
        logger.info(f"Plan result: {plan_result}.\n")
        await ctx.set("plan_result", plan_result)
        return PlanDoneEvent()

    ### Execute Step: Determining realization of explanation moves, Performing selected action
    @step()
    async def execute(self, ctx: Context, ev: PlanDoneEvent) -> StopEvent:
        # Get user message
        user_message = await ctx.get("user_message")

        # Get user model
        user_model = self.user_model.get_summary()

        # Plan the next steps
        plan_result = await ctx.get("plan_result", None)

        # Get explanation from plan and update user model
        suggested_explanation = plan_result.next_explanation
        self.user_model.mark_as_shown(suggested_explanation)

        # Execute the plan
        execute_prompt = PromptTemplate(get_execute_prompt_template().format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            xai_explanations=self.xai_explanations,
            chat_history=self.chat_history,
            user_model=user_model,
            user_question=user_message,
            plan_result=plan_result
        ))
        execute_result = await self.llm.astructured_predict(ExecuteResult, execute_prompt)
        logger.info(f"Execute result: {execute_result}.")
        logger.info("...\n")
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
    draw_most_recent_execution,
)

if __name__ == "__main__":
    # Draw all
    draw_all_possible_flows(MapeKXAIWorkflowAgent, filename="mapek_xaiagent_flow_all.html")
