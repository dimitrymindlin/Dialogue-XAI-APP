import csv
import copy
import json
from typing import List, Dict, Any, Optional

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
from pydantic import BaseModel, Field

from create_experiment_data.instance_datapoint import InstanceDatapoint
from llm_agents.base_agent import XAIBaseAgent
from llm_agents.explanation_state import ExplanationState
from llama_index.core.workflow.retry_policy import ConstantDelayRetryPolicy

import logging

from llm_agents.utils.definition_wrapper import DefinitionWrapper
from llm_agents.mape_k_approach.plan_component.xai_exp_populator import XAIExplanationPopulator
from llm_agents.mape_k_approach.user_model.user_model_fine_grained import UserModelFineGrained as UserModel
from llm_agents.utils.postprocess_message import replace_plot_placeholders
from llm_agents.merged_prompts import get_merged_prompts
import os
import datetime
import traceback

LOG_FOLDER = "mape-k-logs"
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

# Performance logging folder
PERFORMANCE_LOG_FOLDER = "performance-logs"
if not os.path.exists(PERFORMANCE_LOG_FOLDER):
    os.makedirs(PERFORMANCE_LOG_FOLDER)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_ORGANIZATION"] = os.getenv('OPENAI_ORGANIZATION_ID')
LLM_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME')
# Create a logger specific to the current module

logger = logging.getLogger(__name__)

# Base directory: the directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Configure a file handler
file_handler = logging.FileHandler("unified_logfile.txt", mode="w")
file_handler.setLevel(logging.INFO)

# Define a custom formatter for more readable output
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n"  # newline for readability
)
file_handler.setFormatter(formatter)

LOG_CSV_FILE = "unified_log_table.csv"
CSV_HEADERS = ["timestamp", "experiment_id", "datapoint_count", "user_message", "monitor", "analyze", "plan", "execute", "user_model", "processing_time"]

# Performance logging
performance_logger = logging.getLogger("performance_logger")
performance_file_handler = logging.FileHandler(os.path.join(PERFORMANCE_LOG_FOLDER, "performance_comparison.log"), mode="a")
performance_file_handler.setLevel(logging.INFO)
performance_formatter = logging.Formatter(fmt="%(asctime)s - %(message)s")
performance_file_handler.setFormatter(performance_formatter)
performance_logger.addHandler(performance_file_handler)
performance_logger.setLevel(logging.INFO)


def log_performance_data(agent_type: str, time_elapsed: float, query: str, experiment_id: str):
    """
    Logs performance data to a centralized performance log file for comparison.
    
    Args:
        agent_type (str): The type of agent used (e.g., "llama_index" or "openai_agents")
        time_elapsed (float): Processing time in seconds
        query (str): The query that was processed
        experiment_id (str): The experiment ID
    """
    performance_logger.info(
        f"PERFORMANCE_DATA,{agent_type},{time_elapsed:.4f}s,{experiment_id},\"{query[:50]}...\""
    )


def generate_log_file_name(experiment_id: str) -> str:
    timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H:%M")
    return os.path.join(LOG_FOLDER, f"{timestamp}_unified_{experiment_id}.csv")


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


# Define Pydantic models for structured output

class NewExplanationModel(BaseModel):
    """
    Data model for a new explanation concept to be added.
    """
    name: str = Field(..., description="The name of the explanation concept")
    description: str = Field("", description="Brief justification for this explanation")
    dependencies: List[str] = Field(default_factory=list, description="List of dependencies")
    is_optional: bool = Field(False, description="Whether this explanation is optional")


class ChosenExplanationModel(BaseModel):
    """
    Data model for a chosen explanation concept to be added to the explanation plan.
    """
    explanation_name: str = Field(..., description="The name of the explanation concept.")
    step: str = Field(..., description="The name or label of the step of the explanation.")
    description: str = Field("", description="Brief justification for this explanation")
    dependencies: List[str] = Field(default_factory=list, description="List of dependencies")
    is_optional: bool = Field(False, description="Whether this explanation is optional")


class CommunicationGoal(BaseModel):
    """
    Data model for communication goals
    """
    goal: str = Field(..., description="The goal description")
    type: str = Field(..., description="The type of communication goal")


class ExplanationTarget(BaseModel):
    """
    Data model for explanation targets
    """
    reasoning: str = Field(..., description="Reasoning for this explanation target")
    explanation_name: str = Field(..., description="Name of the explanation")
    step_name: str = Field(..., description="Step name of the explanation")
    communication_goals: List[CommunicationGoal] = Field(default_factory=list, description="Communication goals for this explanation")


class ModelChange(BaseModel):
    """
    Model for representing a change to the user model
    """
    explanation_name: str = Field(..., description="The name of the explanation")
    step: str = Field(..., description="The step in the explanation process")
    state: str = Field(..., description="The new state value")


class MAPE_K_ResultModel(BaseModel):
    """
    Complete MAPE-K workflow result model.
    """
    # Monitor stage
    monitor_reasoning: str = Field("", description="Reasoning about the user's understanding")
    explicit_understanding_displays: List[str] = Field(default_factory=list, description="List of explicit understanding displays from the user message")
    mode_of_engagement: str = Field("", description="The cognitive mode of engagement from the user message")
    
    # Analyze stage
    analyze_reasoning: str = Field("", description="Reasoning behind analyzing user model changes")
    model_changes: List[ModelChange] = Field(default_factory=list, description="List of changes to make to the user model")
    
    # Plan stage
    plan_reasoning: str = Field("", description="Reasoning behind the explanation plan")
    new_explanations: List[NewExplanationModel] = Field(default_factory=list, description="List of new explanations to add")
    explanation_plan: List[ChosenExplanationModel] = Field(default_factory=list, description="List of chosen explanations for the plan")
    next_response: List[ExplanationTarget] = Field(default_factory=list, description="List of explanation targets for the next response")
    
    # Execute stage
    execute_reasoning: str = Field("", description="Reasoning behind the constructed response")
    response: str = Field("", description="The HTML-formatted response to the user")


class ExecuteResult(BaseModel):
    """
    Model for execution result to maintain compatibility with existing code.
    """
    reasoning: str = Field(..., description="The reasoning behind the response.")
    response: str = Field(..., description="The HTML-formatted response to the user.")


class UnifiedMapeKAgent(Workflow, XAIBaseAgent):
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
        self.opposite_class_name = None
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
        self.explanation_plan = []
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
        logger.info(f"User model after initialization: {self.user_model.get_state_summary(as_dict=True)}.\n")
        self.visual_explanations_dict = xai_visual_explanations
        self.last_shown_explanations = []

    def complete_explanation_step(self, explanation_name, step_name):
        # Delete this step from the explanation plan
        for exp in self.explanation_plan:
            if exp.explanation_name == explanation_name and exp.step == step_name:
                self.explanation_plan.remove(exp)
                break

    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=2))
    async def unified_mape_k(self, ctx: Context, ev: StartEvent) -> StopEvent:
        """
        Unified MAPE-K workflow step that performs Monitor, Analyze, Plan, and Execute in a single LLM call.
        """
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
            "user_model": "",
            "processing_time": ""
        }
        append_new_log_row(self.current_log_row, self.log_file)

        # Create the unified prompt with all necessary context
        prompt_template_str = get_merged_prompts()
        prompt_template_str = prompt_template_str.format(
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
            last_shown_explanations=self.last_shown_explanations
        )
        
        start_time = datetime.datetime.now()
        
        # Make single LLM call to get structured JSON response
        response_text = await self.llm.acomplete(prompt_template_str)
        logger.info(f"Raw LLM response: {response_text.text}")
        
        try:
            # Extract the JSON part from the LLM response
            json_text = response_text.text.strip()
            # Sometimes LLM might wrap JSON in markdown code blocks, so handle that
            if json_text.startswith("```json"):
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif json_text.startswith("```"):
                json_text = json_text.split("```")[1].split("```")[0].strip()
                
            result_json = json.loads(json_text)
            
            # Parse the response into the MAPE_K_ResultModel
            mape_k_result = MAPE_K_ResultModel(
                # Monitor stage
                monitor_reasoning="Monitor analysis completed",
                explicit_understanding_displays=result_json["Monitor"]["understanding_displays"],
                mode_of_engagement=result_json["Monitor"]["cognitive_state"],
                
                # Analyze stage
                analyze_reasoning="Analyze process completed",
                model_changes=[
                    ModelChange(
                        explanation_name=exp_name,
                        step="", # Will be populated in the processing loop
                        state=state
                    ) 
                    for exp_name, state in result_json["Analyze"]["updated_explanation_states"].items()
                ],
                
                # Plan stage
                plan_reasoning=result_json["Plan"]["reasoning"],
                new_explanations=[],  # Will be populated from next_explanations
                explanation_plan=[
                    ChosenExplanationModel(
                        explanation_name=exp.get("name"),
                        step=exp.get("dependencies", [""])[0] if exp.get("dependencies") else "",
                        description=exp.get("description", ""),
                        dependencies=exp.get("dependencies", []),
                        is_optional=exp.get("is_optional", False)
                    )
                    for exp in result_json["Plan"]["next_explanations"]
                ],
                next_response=[],  # Will be populated during processing
                
                # Execute stage
                execute_reasoning="Response generated successfully",
                response=result_json["Execute"]["html_response"]
            )
            
            # Process model changes to fully populate the state
            analyze_model_changes = []
            for exp_name, new_state in result_json["Analyze"]["updated_explanation_states"].items():
                # Get all steps for this explanation
                explanation = self.user_model.explanations.get(exp_name)
                if explanation:
                    # Then update all steps for this explanation
                    for step in explanation.explanation_steps:
                        self.user_model.update_explanation_step_state(exp_name, step.step_name, new_state)
                        analyze_model_changes.append(ModelChange(
                            explanation_name=exp_name, 
                            step=step.step_name, 
                            state=new_state
                        ))
            
            # Update model_changes with fully populated changes
            mape_k_result.model_changes = analyze_model_changes
            
            # Create explanation targets for execute phase
            from llm_agents.mape_k_approach.plan_component.advanced_plan_prompt_multi_step import (
                ExplanationTarget as OrigExplanationTarget, 
                CommunicationGoal as OrigCommunicationGoal
            )
            
            next_response_targets = []
            for chosen_exp in mape_k_result.explanation_plan:
                exp_target = ExplanationTarget(
                    reasoning="From unified MAPE-K call",
                    explanation_name=chosen_exp.explanation_name,
                    step_name=chosen_exp.step,
                    communication_goals=[
                        CommunicationGoal(
                            goal=f"Explain {chosen_exp.explanation_name} {chosen_exp.step}",
                            type="provide_information"
                        )
                    ]
                )
                next_response_targets.append(exp_target)
            
            # Update next_response with created targets
            mape_k_result.next_response = next_response_targets
            
            # Update log with processed results
            self.current_log_row["monitor"] = {
                "reasoning": mape_k_result.monitor_reasoning,
                "explicit_understanding_displays": mape_k_result.explicit_understanding_displays,
                "mode_of_engagement": mape_k_result.mode_of_engagement
            }
            
            self.current_log_row["analyze"] = {
                "reasoning": mape_k_result.analyze_reasoning,
                "model_changes": [change.dict() for change in mape_k_result.model_changes]
            }
            
            self.current_log_row["plan"] = {
                "reasoning": mape_k_result.plan_reasoning,
                "explanation_plan": [exp.dict() for exp in mape_k_result.explanation_plan],
                "new_explanations": [exp.dict() for exp in mape_k_result.new_explanations],
                "next_response": [target.dict() for target in mape_k_result.next_response]
            }
            
            self.current_log_row["execute"] = {
                "reasoning": mape_k_result.execute_reasoning,
                "response": mape_k_result.response
            }
            
            # Calculate and log time elapsed
            end_time = datetime.datetime.now()
            time_elapsed = (end_time - start_time).total_seconds()
            self.current_log_row["processing_time"] = str(time_elapsed)
            
            update_last_log_row(self.current_log_row, self.log_file)
            
            # Log execution time
            logger.info(f"Time taken for Unified MAPE-K: {end_time - start_time}")
            
            # Log performance data for comparison
            log_performance_data("llama_index", time_elapsed, user_message, self.experiment_id)
            
            # Update chat history
            self.append_to_history("user", user_message)
            self.append_to_history("agent", mape_k_result.response)
            
            # Replace plot placeholders with actual visual content
            response_with_plots = replace_plot_placeholders(mape_k_result.response, self.visual_explanations_dict)
            
            # Update user model with explanations that were presented
            for next_explanation in mape_k_result.next_response:
                exp = next_explanation.explanation_name
                exp_step = next_explanation.step_name
                self.user_model.update_explanation_step_state(exp, exp_step, ExplanationState.UNDERSTOOD.value)
                # Here, convert the Pydantic model to the original type expected by the code
                orig_exp_target = OrigExplanationTarget(
                    reasoning=next_explanation.reasoning,
                    explanation_name=next_explanation.explanation_name,
                    step_name=next_explanation.step_name,
                    communication_goals=[
                        OrigCommunicationGoal(
                            goal=cg.goal,
                            type=cg.type
                        ) for cg in next_explanation.communication_goals
                    ]
                )
                self.last_shown_explanations.append(orig_exp_target)
            
            # Update explanation plan
            if mape_k_result.explanation_plan:
                self.explanation_plan = mape_k_result.explanation_plan
            
            # Log updated user model
            self.current_log_row["user_model"] = self.user_model.get_state_summary(as_dict=True)
            update_last_log_row(self.current_log_row, self.log_file)
            logger.info(f"User model after unified MAPE-K: {self.user_model.get_state_summary(as_dict=False)}.\n")
            
            # Create final result object for the workflow
            final_result = ExecuteResult(
                reasoning=mape_k_result.execute_reasoning,
                response=response_with_plots
            )
            
            return StopEvent(result=final_result)
            
        except (json.JSONDecodeError, KeyError) as e:
            # Calculate time even for error
            end_time = datetime.datetime.now()
            time_elapsed = (end_time - start_time).total_seconds()
            
            # Log performance data even for errors
            log_performance_data("llama_index_error", time_elapsed, user_message, self.experiment_id)
            
            logger.error(f"Error processing unified MAPE-K response: {e}\nResponse text: {response_text.text}")
            # Provide a fallback response when processing fails
            fallback_result = ExecuteResult(
                reasoning="Error processing unified MAPE-K response",
                response="I apologize, but I encountered a technical issue processing your request. Could you please try rephrasing your question?"
            )
            return StopEvent(result=fallback_result)

    # Method to answer user question - maintains compatibility with existing interface
    async def answer_user_question(self, user_question):
        """
        Public method to answer a user question using the unified MAPE-K workflow.
        """
        start_time = datetime.datetime.now()
        result = await self.run(input=user_question)
        end_time = datetime.datetime.now()
        
        time_elapsed = (end_time - start_time).total_seconds()
        logger.info(f"Total time for Unified MAPE-K workflow: {end_time - start_time}")
        
        # Log overall performance
        performance_logger.info(f"TOTAL_TIME,llama_index,{time_elapsed:.4f}s,{self.experiment_id},\"{user_question[:50]}...\"")
        
        # Return analysis and response to maintain compatibility with caller
        return result.reasoning, result.response


# For testing/visualization purposes
if __name__ == "__main__":
    from llama_index.utils.workflow import draw_all_possible_flows
    # Draw workflow diagram
    draw_all_possible_flows(UnifiedMapeKAgent, filename="unified_mape_k_flow.html") 