import csv
import copy
import json
import os
import datetime
import logging
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model

# OpenAI Agents SDK imports
from agents import Agent, Runner, function_tool

from llm_agents.base_agent import XAIBaseAgent
from llm_agents.explanation_state import ExplanationState
from llm_agents.utils.definition_wrapper import DefinitionWrapper
from llm_agents.mape_k_approach.plan_component.xai_exp_populator import XAIExplanationPopulator
from llm_agents.mape_k_approach.user_model.user_model_fine_grained import UserModelFineGrained as UserModel
from llm_agents.utils.postprocess_message import replace_plot_placeholders
from llm_agents.merged_prompts import get_merged_prompts
from create_experiment_data.instance_datapoint import InstanceDatapoint

# Configure logging
LOG_FOLDER = "mape-k-logs"
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

# Set up environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_ORGANIZATION"] = os.getenv('OPENAI_ORGANIZATION_ID')
LLM_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME', 'gpt-4')

# Create a logger
logger = logging.getLogger(__name__)

# Base directory: the directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Configure a file handler
file_handler = logging.FileHandler("openai_unified_logfile.txt", mode="w")
file_handler.setLevel(logging.INFO)

# Define a custom formatter for more readable output
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n"  # newline for readability
)
file_handler.setFormatter(formatter)

# CSV logging configuration
LOG_CSV_FILE = "openai_unified_log_table.csv"
CSV_HEADERS = ["timestamp", "experiment_id", "datapoint_count", "user_message", "monitor", "analyze", "plan", "execute", "user_model"]

# Optional: Add a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(fmt="%(levelname)s - %(message)s"))

# Set up the logger with both handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


def generate_log_file_name(experiment_id: str) -> str:
    timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H:%M")
    return os.path.join(LOG_FOLDER, f"{timestamp}_openai_unified_{experiment_id}.csv")


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


# Pydantic models for structured data
class ChosenExplanationModel(BaseModel):
    """
    Data model for a chosen explanation concept to be added to the explanation plan.
    """
    explanation_name: str = Field(..., description="The name of the explanation concept.")
    step: str = Field(..., description="The name or label of the step of the explanation.")
    description: str = Field("", description="Brief justification for this explanation")
    dependencies: list = Field(default_factory=list, description="List of dependencies")
    is_optional: bool = Field(False, description="Whether this explanation is optional")


class UnifiedMapeKResult(BaseModel):
    """
    Unified MAPE-K workflow result model.
    """
    # Monitor step results
    understanding_displays: list = Field(default_factory=list, 
                                       description="A list of explicit understanding displays from the user message")
    cognitive_state: str = Field("", description="The cognitive mode of engagement from the user message")
    
    # Analyze step results
    updated_explanation_states: dict = Field(default_factory=dict, 
                                          description="Dictionary of updated explanation states")
    
    # Plan step results
    next_explanations: list = Field(default_factory=list, 
                                 description="List of next explanations planned")
    reasoning: str = Field("", description="Reasoning behind the chosen explanations")
    
    # Execute step results
    html_response: str = Field("", description="Final HTML response to the user")


class ExecuteResult(BaseModel):
    """
    Model for execution result to maintain compatibility with existing code.
    """
    reasoning: str = Field(..., description="The reasoning behind the response.")
    response: str = Field(..., description="The HTML-formatted response to the user.")


# Pydantic models for function tools
class UnderstandingDisplay(BaseModel):
    display: str = Field(..., description="The understanding display detected")

class CognitiveState(BaseModel):
    state: str = Field(..., description="The cognitive mode of engagement")

class ExplanationState(BaseModel):
    name: str = Field(..., description="The name of the explanation")
    state: str = Field(..., description="The updated state of the explanation")

class UserModelUpdate(BaseModel):
    understanding_displays: List[str] = Field(default_factory=list, description="List of understanding displays detected in the user message")
    cognitive_state: str = Field("", description="The cognitive mode of engagement detected in the user message")
    explanation_states: Dict[str, str] = Field(default_factory=dict, description="Dictionary of explanation names and their updated states")


class ExplanationItem(BaseModel):
    name: str = Field(..., description="The name of the explanation")
    description: str = Field("", description="Brief justification for this explanation")
    dependencies: List[str] = Field(default_factory=list, description="List of dependencies")
    is_optional: bool = Field(False, description="Whether this explanation is optional")


class ResponseGeneration(BaseModel):
    next_explanations: List[ExplanationItem] = Field(..., description="List of next explanations to be included in the response")
    reasoning: str = Field("", description="The reasoning behind the chosen explanations")


# Function tools for the agent
@function_tool
def update_user_model(
    model_update: UserModelUpdate
) -> str:
    """
    Updates the user model based on the agent's analysis.
    
    Args:
        model_update: Object containing understanding displays, cognitive state, and explanation states
        
    Returns:
        A confirmation message indicating the user model was updated
    """
    return "User model updated successfully"


@function_tool
def generate_html_response(
    response_data: ResponseGeneration
) -> str:
    """
    Generates an HTML response based on the planned explanations.
    
    Args:
        response_data: Object containing next explanations and reasoning
        
    Returns:
        HTML-formatted response to be shown to the user
    """
    return f"<p>Generated HTML response based on {len(response_data.next_explanations)} explanations</p>"


class OpenAIUnifiedMapeKAgent(XAIBaseAgent):
    """
    Unified MAPE-K agent implementation using OpenAI Agents SDK.
    This agent performs the Monitor, Analyze, Plan, and Execute steps in a single workflow.
    """
    def __init__(
            self,
            feature_names="",
            domain_description="",
            user_ml_knowledge="",
            experiment_id="",
            **kwargs
    ):
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
        
        # Setup wrapper definitions
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

        # Setup user model
        self.user_model = UserModel(user_ml_knowledge)
        self.populator = None
        
        # Chat history
        self.chat_history = self.reset_history()
        self.explanation_plan = []
        self.last_shown_explanations = []  # save tuples of explanation and step
        self.visual_explanations_dict = None
        
        # Create OpenAI agent
        self.agent = self._create_agent()
        
        # Current log entry
        self.current_log_row = None

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

    def initialize_new_datapoint(self,
                                 instance: InstanceDatapoint,
                                 xai_explanations,
                                 xai_visual_explanations,
                                 predicted_class_name,
                                 opposite_class_name,
                                 datapoint_count):
        """Initialize a new datapoint for analysis."""
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
        
        # Recreate the agent with updated context
        self.agent = self._create_agent()

    def complete_explanation_step(self, explanation_name, step_name):
        """Delete a completed explanation step from the explanation plan."""
        for exp in self.explanation_plan:
            if exp.explanation_name == explanation_name and exp.step == step_name:
                self.explanation_plan.remove(exp)
                break

    def _create_agent(self):
        """Create an OpenAI Agent with appropriate instructions and tools."""
        # Create an agent with appropriate instructions
        instructions = f"""
        You are an advanced Explainable AI (XAI) assistant that manages interactions with a user about a machine learning model's predictions.
        You will perform 4 consecutive tasks: Monitor, Analyze, Plan, Execute.
        
        DOMAIN CONTEXT:
        - Domain Description: {self.domain_description}
        - Model Features: {self.feature_names}
        - Current Explained Instance: {self.instance}
        - Predicted Class by AI Model: {self.predicted_class_name}
        - Opposite Class: {self.opposite_class_name}
        
        USER MODEL:
        {self.user_model.get_state_summary(as_dict=False) if hasattr(self, 'user_model') else "No user model available yet."}
        
        UNDERSTANDING DISPLAYS:
        {self.understanding_displays.as_text()}
        
        COGNITIVE ENGAGEMENT MODES:
        {self.modes_of_engagement.as_text()}
        
        Your goal is to provide explanations that help the user understand the model's predictions.
        Always consider the user's knowledge level, cognitive state, and previous interactions.
        """
        
        mape_k_agent = Agent(
            name="MAPE-K XAI Assistant",
            instructions=instructions,
            tools=[
                update_user_model,
                generate_html_response
            ]
        )
        
        return mape_k_agent

    async def _process_unified_mape_k(self, user_message):
        """
        Process a user message through the unified MAPE-K workflow.
        """
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
        
        start_time = datetime.datetime.now()
        
        # Create input message with all necessary context
        input_message = f"""
        Please process this user message: "{user_message}"
        
        Current chat history: {self.chat_history}
        
        EXPLANATION COLLECTION:
        {self.user_model.get_complete_explanation_collection(as_dict=False)}
        
        EXPLANATION PLAN:
        {self.explanation_plan}
        
        LAST SHOWN EXPLANATIONS:
        {self.last_shown_explanations}
        
        Perform the full MAPE-K workflow:
        
        1. MONITOR: Analyze the user message for understanding displays and cognitive engagement mode.
        2. ANALYZE: Update the explanation states based on the user's message and current understanding.
        3. PLAN: Choose the next explanations to show based on the updated user model.
        4. EXECUTE: Create an HTML response based on the planned explanations.
        
        Return a structured JSON with the results from each step.
        """
        
        try:
            # Run the agent
            result = await Runner.run(self.agent, input=input_message)
            logger.info(f"Raw agent response: {result.final_output}")
            
            # Extract the JSON part from the response
            response_text = result.final_output
            json_text = None
            
            # Try to extract JSON from the text
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].strip()
            else:
                # Try to find JSON directly in the text
                import re
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                else:
                    json_text = response_text
            
            # Parse the JSON response
            result_json = json.loads(json_text)
            
            # Extract and process Monitor results
            monitor_result = {
                "reasoning": "Unified MAPE-K processing",
                "explicit_understanding_displays": result_json["Monitor"]["understanding_displays"],
                "mode_of_engagement": result_json["Monitor"]["cognitive_state"],
                "model_changes": []
            }
            
            # Update user cognitive state based on monitor results
            if monitor_result["mode_of_engagement"]:
                self.user_model.cognitive_state = self.modes_of_engagement.get_differentiating_description(
                    monitor_result["mode_of_engagement"])
            
            # Update explicit understanding signals
            if monitor_result["explicit_understanding_displays"]:
                self.user_model.explicit_understanding_signals = monitor_result["explicit_understanding_displays"]
            
            # Process Analyze changes
            analyze_result = {
                "reasoning": "Unified MAPE-K processing",
                "model_changes": []
            }
            
            # Update user model based on analyze results
            for exp_name, new_state in result_json["Analyze"]["updated_explanation_states"].items():
                # Get all steps for this explanation
                explanation = self.user_model.explanations.get(exp_name)
                if explanation:
                    # Then update all steps for this explanation
                    for step in explanation.explanation_steps:
                        self.user_model.update_explanation_step_state(exp_name, step.step_name, new_state)
                        analyze_result["model_changes"].append({
                            "explanation_name": exp_name, 
                            "step": step.step_name, 
                            "state": new_state
                        })
            
            # Process Plan results
            from llm_agents.mape_k_approach.plan_component.advanced_plan_prompt_multi_step import (
                ExplanationTarget, 
                CommunicationGoal
            )
            
            plan_result = {
                "reasoning": result_json["Plan"]["reasoning"],
                "explanation_plan": [],
                "new_explanations": [],
                "next_response": []
            }
            
            # Create chosen explanations from plan results
            for explanation in result_json["Plan"]["next_explanations"]:
                chosen_exp = ChosenExplanationModel(
                    explanation_name=explanation["name"],
                    step=explanation.get("dependencies", [""])[0] if explanation.get("dependencies") else "",
                    description=explanation.get("description", ""),
                    dependencies=explanation.get("dependencies", []),
                    is_optional=explanation.get("is_optional", False)
                )
                plan_result["explanation_plan"].append(chosen_exp)
                
                # Create explanation targets for execute phase
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
                plan_result["next_response"].append(exp_target)
            
            # Update explanation plan
            if plan_result["explanation_plan"]:
                self.explanation_plan = plan_result["explanation_plan"]
            
            # Process Execute result
            execute_result = {
                "reasoning": "Unified MAPE-K processing",
                "response": result_json["Execute"]["html_response"]
            }
            
            # Update log with processed results
            self.current_log_row["monitor"] = monitor_result
            self.current_log_row["analyze"] = analyze_result
            self.current_log_row["plan"] = plan_result
            self.current_log_row["execute"] = execute_result
            update_last_log_row(self.current_log_row, self.log_file)
            
            # Log execution time
            end_time = datetime.datetime.now()
            logger.info(f"Time taken for Unified MAPE-K: {end_time - start_time}")
            
            # Update chat history
            self.append_to_history("user", user_message)
            self.append_to_history("agent", execute_result["response"])
            
            # Replace plot placeholders with actual visual content
            response_with_plots = replace_plot_placeholders(execute_result["response"], self.visual_explanations_dict)
            
            # Update user model with explanations that were presented
            for next_explanation in plan_result["next_response"]:
                exp = next_explanation.explanation_name
                exp_step = next_explanation.step_name
                self.user_model.update_explanation_step_state(exp, exp_step, ExplanationState.UNDERSTOOD.value)
                self.last_shown_explanations.append(next_explanation)
            
            # Log updated user model
            self.current_log_row["user_model"] = self.user_model.get_state_summary(as_dict=True)
            update_last_log_row(self.current_log_row, self.log_file)
            logger.info(f"User model after unified MAPE-K: {self.user_model.get_state_summary(as_dict=False)}.\n")
            
            # Create final result object
            final_result = ExecuteResult(
                reasoning="Unified MAPE-K workflow completed successfully",
                response=response_with_plots
            )
            
            return final_result
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error processing unified MAPE-K response: {e}")
            # Provide a fallback response when processing fails
            fallback_result = ExecuteResult(
                reasoning="Error processing unified MAPE-K response",
                response="I apologize, but I encountered a technical issue processing your request. Could you please try rephrasing your question?"
            )
            return fallback_result

    async def answer_user_question(self, user_question):
        """
        Public method to answer a user question using the unified MAPE-K workflow.
        Maintains compatibility with the existing XAIBaseAgent interface.
        """
        start_time = datetime.datetime.now()
        result = await self._process_unified_mape_k(user_question)
        end_time = datetime.datetime.now()
        logger.info(f"Total time for Unified MAPE-K workflow: {end_time - start_time}")
        
        # Return analysis and response to maintain compatibility with caller
        return result.reasoning, result.response


# For testing purposes
if __name__ == "__main__":
    import asyncio
    
    async def test_agent():
        agent = OpenAIUnifiedMapeKAgent(
            feature_names="feature1, feature2, feature3",
            domain_description="A sample domain for testing",
            user_ml_knowledge="beginner",
            experiment_id="test_experiment"
        )
        
        # Test with a sample question
        reasoning, response = await agent.answer_user_question("How does feature1 affect the prediction?")
        print(f"Reasoning: {reasoning}")
        print(f"Response: {response}")
    
    asyncio.run(test_agent()) 