"""
Enhanced MAPE-K agent implementation using explicit OpenAI function calling.

This implementation improves on the original MAPE-K agent by using modern OpenAI function
calling features with properly typed schemas and better error handling.
"""

import csv
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
import uuid

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Import the prompt templates from monitor_analyze_combined
from llm_agents.mape_k_2_components.monitor_analyze_combined import (
    get_monitor_analyze_prompt_template,
    get_monitor_analyze_prompt_template_short,
    get_monitor_analyze_prompt_template_streamlined,
    MonitorAnalyzeResultModel
)

# Import the prompt templates from plan_execute_combined
from llm_agents.mape_k_2_components.plan_execute_combined import (
    get_plan_execute_prompt_template,
    get_plan_execute_prompt_template_short,
    PlanExecuteResultModel
)

# Import the DefinitionWrapper for scientific labels
from llm_agents.utils.definition_wrapper import DefinitionWrapper
from llm_agents.mape_k_approach.user_model.user_model_fine_grained import UserModelFineGrained as UserModel
from llm_agents.mape_k_approach.plan_component.xai_exp_populator import XAIExplanationPopulator

# Configure logging
LOG_FOLDER = "mape-k-logs"
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mape-k-enhanced")

# Add file handler
file_handler = logging.FileHandler(os.path.join(LOG_FOLDER, "enhanced-mape-k-agent.log"), mode="w")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# CSV logging setup
CSV_HEADERS = ["timestamp", "experiment_id", "datapoint_count", "user_message", "monitor_analyze", "plan_execute",
               "response", "performance"]

load_dotenv()
# Environment variables for API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ORGANIZATION_ID = os.getenv('OPENAI_ORGANIZATION_ID')


# Define Pydantic models for the agent's input and output schemas
class MonitorState(BaseModel):
    """State for the Monitor phase"""
    goal: str = Field(..., description="The goal the user is trying to achieve")
    context: str = Field(..., description="The context of the conversation")
    user_ml_knowledge: str = Field(..., description="The user's knowledge of machine learning")
    xai_report: str = Field(..., description="Report from the XAI system")


class AnalyzeResult(BaseModel):
    """Result from the Analyze phase"""
    analysis: str = Field(..., description="Analysis of the user's question and context")
    requirements: List[str] = Field(..., description="Requirements for an effective explanation")
    suitable_explanations: List[str] = Field(..., description="List of suitable explanation methods")


class PlanStrategy(BaseModel):
    """Strategy for the Plan phase"""
    explanation_method: str = Field(..., description="The explanation method to use")
    explanation_details: str = Field(..., description="Details about the explanation")
    rationale: str = Field(..., description="Rationale for choosing this explanation")


class ExecuteResult(BaseModel):
    """Result from the Execute phase"""
    response: str = Field(..., description="The response to provide to the user")
    reasoning: str = Field(..., description="Reasoning behind the response")
    visualization: Optional[str] = Field(None, description="Any visualization instructions")


def generate_log_file_name(experiment_id: str) -> str:
    """Generate a timestamped log file name for the experiment"""
    timestamp = datetime.now().strftime("%d.%m.%Y_%H:%M")
    return os.path.join(LOG_FOLDER, f"{timestamp}_{experiment_id}.csv")


def initialize_csv(log_file: str):
    """Initialize a CSV file with headers if it doesn't exist"""
    if not os.path.isfile(log_file):
        try:
            with open(log_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(CSV_HEADERS)
            logger.info(f"Initialized {log_file} with headers.\n")
        except Exception as e:
            logger.error(f"Failed to initialize CSV: {e}\n")


def append_new_log_row(row: dict, log_file: str):
    """Append a new row to the CSV log file"""
    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow([row.get(header, "") for header in CSV_HEADERS])


def update_last_log_row(row: dict, log_file: str):
    """Update the last row in the CSV log file"""
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


class MapeKXAIWorkflowAgentEnhanced:
    """
    Enhanced implementation of the 2-component MAPE-K XAI workflow agent using explicit function calling.
    
    This agent implements the Monitor-Analyze-Plan-Execute-Knowledge (MAPE-K) loop
    for XAI workflows using the OpenAI API.
    """

    def __init__(
            self,
            feature_names: str,
            domain_description: str,
            user_ml_knowledge: str = "beginner",
            experiment_id: str = None,
            model: str = "gpt-4-turbo",
            mini_model: str = "gpt-3.5-turbo",
            include_monitor_reasoning: bool = True,
            include_analyze_reasoning: bool = True,
            include_plan_reasoning: bool = False,
            include_execute_reasoning: bool = False,
            verbose: bool = False
    ):
        """
        Initialize the enhanced MAPE-K XAI workflow agent.
        
        Args:
            feature_names: Names of features in the dataset
            domain_description: Description of the application domain
            user_ml_knowledge: User's level of ML knowledge
            experiment_id: Identifier for the experiment
            model: OpenAI model to use for main calls
            mini_model: OpenAI model to use for less complex calls
            include_monitor_reasoning: Whether to include reasoning from the monitor phase
            include_analyze_reasoning: Whether to include reasoning from the analyze phase
            include_plan_reasoning: Whether to include reasoning from the plan phase
            include_execute_reasoning: Whether to include reasoning from the execute phase
            verbose: Whether to print verbose output
        """
        self.client = AsyncOpenAI()
        self.feature_names = feature_names
        self.domain_description = domain_description
        self.user_ml_knowledge = user_ml_knowledge
        self.experiment_id = experiment_id or str(uuid.uuid4())
        self.log_file = generate_log_file_name(self.experiment_id)
        initialize_csv(self.log_file)
        self.model = model
        self.mini_model = mini_model
        self.include_monitor_reasoning = include_monitor_reasoning
        self.include_analyze_reasoning = include_analyze_reasoning
        self.include_plan_reasoning = include_plan_reasoning
        self.include_execute_reasoning = include_execute_reasoning
        self.verbose = verbose

        # Agent state variables
        self.current_datapoint = None
        self.xai_report = None
        self.visual_explanation_dict = {}
        self.current_prediction = None
        self.opposite_class_name = None
        self.datapoint_count = 0
        self.chat_history = "No history available, beginning of the chat."
        self.current_log_row = None

        # Performance tracking
        self.query_count = 0
        self.total_latency = 0

        # Initialize scientific labels
        # Get base directory for paths
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Initialize understanding displays and modes of engagement from JSON files
        try:
            self.understanding_displays = DefinitionWrapper(
                os.path.join(base_dir, "..", "mape_k_approach", "monitor_component",
                             "understanding_displays_definition.json")
            )
            self.modes_of_engagement = DefinitionWrapper(
                os.path.join(base_dir, "..", "mape_k_approach", "monitor_component", "icap_modes_definition.json")
            )
            logger.info("Successfully loaded understanding displays and modes of engagement from JSON files")
        except Exception as e:
            logger.warning(f"Could not load JSON definition files: {e}. Using fallback definitions.")
            # We'll fall back to the hardcoded definitions in monitor_analyze if these aren't available

        # Initialize user model for tracking explanation understanding
        self.user_model = UserModel(self.user_ml_knowledge)
        self.last_shown_explanations = []

    def initialize_new_datapoint(
            self,
            instance: 'InstanceDatapoint',
            xai_explanations,
            xai_visual_explanations,
            predicted_class_name: str,
            opposite_class_name: str,
            datapoint_count: int
    ):
        """
        Initialize the agent with a new datapoint.
        
        Args:
            instance: The InstanceDatapoint object containing feature information
            xai_explanations: Explanations from XAI system
            xai_visual_explanations: Dictionary of visual explanations
            predicted_class_name: Current model prediction
            opposite_class_name: Name of the opposite class
            datapoint_count: Datapoint count in the dataset
        """
        # Extract displayable features from instance
        self.current_datapoint = instance.displayable_features
        self.xai_report = xai_explanations
        self.visual_explanation_dict = xai_visual_explanations
        self.current_prediction = predicted_class_name
        self.opposite_class_name = opposite_class_name
        self.datapoint_count = datapoint_count + 1
        self.chat_history = "No history available, beginning of the chat."

        # Get base directory for paths
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Set up user model using XAIExplanationPopulator
        try:
            self.populator = XAIExplanationPopulator(
                template_dir="/Users/dimitrymindlin/UniProjects/Dialogue-XAI-APP",
                template_file="llm_agents/mape_k_approach/plan_component/explanations_model.yaml",
                xai_explanations=xai_explanations,
                predicted_class_name=predicted_class_name,
                opposite_class_name=opposite_class_name,
                instance_dict=self.current_datapoint
            )
            # Populate the YAML
            self.populator.populate_yaml()
            # Validate substitutions
            self.populator.validate_substitutions()
            # Retrieve as a dictionary
            populated_yaml_json = self.populator.get_populated_json(as_dict=True)
            # Initialize user model from populated YAML
            self.user_model.set_model_from_summary(populated_yaml_json)
        except Exception as e:
            logger.error(f"Error initializing explanation model: {str(e)}")
            raise ValueError(f"Failed to initialize user model: {str(e)}")

        # Reset last shown explanations
        self.last_shown_explanations = []

        logger.info(f"Initialized new datapoint {self.datapoint_count}")
        logger.info(f"Current prediction: {predicted_class_name}")
        logger.info(f"Opposite class: {opposite_class_name}")
        logger.info(f"User model after initialization: {self.user_model.get_state_summary(as_dict=True)}")

    def append_to_history(self, role: str, msg: str):
        """
        Append a message to the chat history.
        
        Args:
            role: Role of the message sender ("user" or "agent")
            msg: The message content
        """
        if role == "user":
            msg = "User: " + msg + "\n"
        elif role == "agent":
            msg = "Agent: " + msg + "\n"

        if self.chat_history == "No history available, beginning of the chat.":
            self.chat_history = msg
        else:
            self.chat_history += msg

    def get_explanation_collection(self) -> str:
        """
        Get a text representation of all available explanations for the current instance.
        
        Returns:
            A formatted string of explanation types and their content
            
        Raises:
            AttributeError: If user model isn't properly initialized or missing required methods
        """
        if hasattr(self, 'user_model') and hasattr(self.user_model, 'get_complete_explanation_collection'):
            return self.user_model.get_complete_explanation_collection(as_dict=False)

        # Instead of using a fallback, throw an appropriate error
        raise AttributeError(
            "User model not properly initialized or missing get_complete_explanation_collection method")

    def _get_monitor_analyze_tools(self) -> List[Dict[str, Any]]:
        """
        Get the tools for the Monitor-Analyze component.
        
        Returns:
            List of tool definitions for the Monitor-Analyze component
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "monitor_analyze_result",
                    "description": "Monitor and analyze the user's question and understanding",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "string",
                                "description": "Short reasoning for the classification of the user message."
                            },
                            "explicit_understanding_displays": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "A list of explicitly stated understanding displays by the user"
                            },
                            "mode_of_engagement": {
                                "type": "string",
                                "description": "The cognitive mode of engagement that the user message exhibits"
                            },
                            "analysis_reasoning": {
                                "type": "string",
                                "description": "The reasoning behind the classification of the user message."
                            },
                            "model_changes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "explanation_name": {
                                            "type": "string",
                                            "description": "Name of the explanation"
                                        },
                                        "step": {
                                            "type": "string",
                                            "description": "Step in the explanation plan that was provided to the user"
                                        },
                                        "state": {
                                            "type": "string",
                                            "description": "State of understanding (not_yet_explained, partially_understood, understood, not_understood)"
                                        }
                                    },
                                    "required": ["explanation_name", "step", "state"]
                                },
                                "description": "List of changes to the user model"
                            }
                        },
                        "required": ["analysis_reasoning", "model_changes"]
                    }
                }
            }
        ]

    def _get_plan_execute_tools(self) -> List[Dict[str, Any]]:
        """
        Get the tools for the Plan-Execute component.
        
        Returns:
            List of tool definitions for the Plan-Execute component
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "plan_and_execute",
                    "description": "Plan and execute the XAI workflow",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "explanation_method": {
                                "type": "string",
                                "description": "The explanation method to use"
                            },
                            "explanation_details": {
                                "type": "string",
                                "description": "Details about the explanation"
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Rationale for choosing this explanation"
                            },
                            "response": {
                                "type": "string",
                                "description": "The response to provide to the user"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Reasoning behind the response"
                            }
                        },
                        "required": [
                            "explanation_method",
                            "explanation_details",
                            "rationale",
                            "response",
                            "reasoning"
                        ]
                    }
                }
            }
        ]

    async def monitor_analyze(self, user_question: str) -> Dict[str, Any]:
        """
        Run the Monitor and Analyze phases of the MAPE-K loop.
        
        Args:
            user_question: The user's question
            
        Returns:
            Dictionary containing the results of the Monitor and Analyze phases
            
        Raises:
            ValueError: If required components aren't initialized or API response is invalid
            AttributeError: If user model methods are missing
            RuntimeError: If the API call fails or returns unexpected data
        """
        start_time = time.time()

        # Create log row entry
        self.current_log_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_id": self.experiment_id,
            "datapoint_count": self.datapoint_count,
            "user_message": user_question,
            "monitor_analyze": "",
            "plan_execute": "",
            "response": "",
            "performance": ""
        }

        append_new_log_row(self.current_log_row, self.log_file)

        # Check required components
        if not hasattr(self, 'user_model'):
            raise AttributeError("User model not properly initialized")

        if not hasattr(self.user_model, 'get_state_summary'):
            raise AttributeError("User model is missing required method get_state_summary")

        try:
            # Define the values for scientifically-derived structures
            if not hasattr(self, 'understanding_displays'):
                raise AttributeError("Understanding displays not properly initialized")

            understanding_displays = self.understanding_displays.as_text()

            if not hasattr(self, 'modes_of_engagement'):
                raise AttributeError("Modes of engagement not properly initialized")

            modes_of_engagement = self.modes_of_engagement.as_text()

            # Build the explanation collection
            explanation_collection = self.get_explanation_collection()

            # Get user model state
            user_model = self.user_model.get_state_summary(as_dict=False)

            if not hasattr(self, 'last_shown_explanations'):
                raise AttributeError("Last shown explanations not initialized")

            last_shown_explanations = self.last_shown_explanations

            # Validate required instance data
            if self.current_datapoint is None:
                raise ValueError("Current datapoint not initialized")

            if self.current_prediction is None:
                raise ValueError("Current prediction not initialized")

            # Format the monitor-analyze prompt using the streamlined template
            monitor_analyze_prompt = get_monitor_analyze_prompt_template_streamlined().format(
                domain_description=self.domain_description,
                feature_names=self.feature_names,
                instance=self.current_datapoint,
                predicted_class_name=self.current_prediction,
                chat_history=self.chat_history,
                user_message=user_question,
                understanding_displays=understanding_displays,
                modes_of_engagement=modes_of_engagement,
                explanation_collection=explanation_collection,
                user_model=user_model,
                last_shown_explanations=last_shown_explanations
            )

            # Call the OpenAI API with function calling
            response = await self.client.chat.completions.create(
                model=self.mini_model,
                messages=[
                    {"role": "system", "content": "You are an AI assistant analyzing user understanding of ML models."},
                    {"role": "user", "content": monitor_analyze_prompt}
                ],
                tools=self._get_monitor_analyze_tools(),
                tool_choice={"type": "function", "function": {"name": "monitor_analyze_result"}}
            )

            # Track API usage
            self.query_count += 1
            elapsed_time = time.time() - start_time
            self.total_latency += elapsed_time

            # Log performance
            logger.info(f"Time taken for Monitor-Analyze: {elapsed_time:.2f}s")

            # Parse the response
            if not response.choices or not response.choices[0].message or not response.choices[0].message.tool_calls:
                raise ValueError("Invalid response format from OpenAI API")

            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name != "monitor_analyze_result":
                raise ValueError(f"Unexpected function call: {tool_call.function.name}")

            # Extract the function arguments
            try:
                function_args = json.loads(tool_call.function.arguments)

                # Validate required fields
                required_fields = ["analysis_reasoning", "model_changes"]
                for field in required_fields:
                    if field not in function_args:
                        raise ValueError(f"Missing required field in response: {field}")

                # Create a result object with the fields from the MonitorAnalyzeResultModel
                result = {
                    "reasoning": function_args.get("reasoning", ""),
                    "explicit_understanding_displays": function_args.get("explicit_understanding_displays", []),
                    "mode_of_engagement": function_args.get("mode_of_engagement", ""),
                    "analysis_reasoning": function_args["analysis_reasoning"],
                    "model_changes": function_args["model_changes"]
                }

                # Update the user model based on monitor results
                # Apply model changes from the monitor-analyze step
                for change in result["model_changes"]:
                    if all(key in change for key in ["explanation_name", "step", "state"]):
                        self.user_model.update_explanation_step_state(
                            change["explanation_name"],
                            change["step"],
                            change["state"]
                        )
                    else:
                        logger.warning(f"Skipping invalid model change: {change}")

                # Log the monitor-analyze result
                if self.include_monitor_reasoning or self.include_analyze_reasoning:
                    logger.info(f"Monitor-Analyze result: {result}")
                else:
                    filtered_result = {**result}
                    if not self.include_monitor_reasoning:
                        filtered_result["reasoning"] = "[monitor reasoning excluded]"
                    if not self.include_analyze_reasoning:
                        filtered_result["analysis_reasoning"] = "[analyze reasoning excluded]"
                    logger.info(f"Monitor-Analyze result (filtered reasoning): {filtered_result}")

                # Update log row
                self.current_log_row["monitor_analyze"] = json.dumps(result)
                update_last_log_row(self.current_log_row, self.log_file)

                return result

            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON from function call arguments: {str(e)}")

        except Exception as e:
            logger.error(f"Error in monitor_analyze: {str(e)}")
            # Re-throw the exception after logging it
            raise

    async def plan_execute(self,
                           user_question: str,
                           monitor_analyze_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the Plan and Execute phases of the MAPE-K loop.
        
        Args:
            user_question: The user's question
            monitor_analyze_result: Results from the Monitor and Analyze phases
            
        Returns:
            Dictionary containing the results of the Plan and Execute phases
            
        Raises:
            ValueError: If required components aren't initialized or API response is invalid
            RuntimeError: If the planning or execution fails
        """
        start_time = time.time()

        # Check required components
        if self.xai_report is None:
            raise ValueError("XAI report not initialized")

        if user_question is None or not isinstance(user_question, str):
            raise ValueError("Invalid user question provided")

        if not monitor_analyze_result or not isinstance(monitor_analyze_result, dict):
            raise ValueError("Invalid monitor_analyze_result")

        # Validate required fields in monitor_analyze_result
        required_monitor_fields = ["reasoning", "mode_of_engagement", "analysis_reasoning", "model_changes"]
        for field in required_monitor_fields:
            if field not in monitor_analyze_result:
                raise ValueError(f"Missing required field in monitor_analyze_result: {field}")

        try:
            # Build the explanation collection
            explanation_collection = self.get_explanation_collection()

            # Get user model state
            user_model = self.user_model.get_state_summary(as_dict=False)

            # Format the plan-execute prompt using the template
            plan_execute_prompt = get_plan_execute_prompt_template_short().format(
                domain_description=self.domain_description,
                feature_names=self.feature_names,
                instance=self.current_datapoint,
                predicted_class_name=self.current_prediction,
                chat_history=self.chat_history,
                user_message=user_question,
                explanation_collection=explanation_collection,
                user_model=user_model
            )

            # Log the plan-execute prompt
            logger.info(f"Plan-Execute prompt: {plan_execute_prompt}")

            # Call the OpenAI API with function calling
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an AI assistant for explainable AI."},
                    {"role": "user", "content": plan_execute_prompt}
                ],
                tools=self._get_plan_execute_tools(),
                tool_choice={"type": "function", "function": {"name": "plan_and_execute"}}
            )

            # Track API usage
            self.query_count += 1
            elapsed_time = time.time() - start_time
            self.total_latency += elapsed_time

            # Log performance
            logger.info(f"Time taken for Plan-Execute: {elapsed_time:.2f}s")

            # Parse the response
            if not response.choices or not response.choices[0].message or not response.choices[0].message.tool_calls:
                raise ValueError("Invalid response format from OpenAI API")

            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name != "plan_and_execute":
                raise ValueError(f"Unexpected function call: {tool_call.function.name}")

            # Extract the function arguments
            try:
                function_args = json.loads(tool_call.function.arguments)

                # Validate required fields
                required_fields = ["explanation_method", "explanation_details", "rationale", "response", "reasoning"]
                missing_fields = [field for field in required_fields if field not in function_args]
                if missing_fields:
                    raise ValueError(f"Missing required fields in response: {', '.join(missing_fields)}")

                result = {
                    "explanation_method": function_args["explanation_method"],
                    "explanation_details": function_args["explanation_details"],
                    "rationale": function_args["rationale"],
                    "response": function_args["response"],
                    "reasoning": function_args["reasoning"]
                }

                # Log the plan-execute result
                if self.include_plan_reasoning and self.include_execute_reasoning:
                    logger.info(f"Plan-Execute result: {result}")
                else:
                    filtered_result = {**result}
                    if not self.include_plan_reasoning:
                        filtered_result["rationale"] = "[plan reasoning excluded]"
                    if not self.include_execute_reasoning:
                        filtered_result["reasoning"] = "[execute reasoning excluded]"
                    logger.info(f"Plan-Execute result (filtered reasoning): {filtered_result}")

                # Add agent's response to chat history
                self.append_to_history("agent", result["response"])

                # Update log cells
                self.current_log_row["plan_execute"] = json.dumps(result)
                self.current_log_row["response"] = result["response"]

                # Add performance metrics
                performance_metrics = self.get_performance_metrics()
                self.current_log_row["performance"] = json.dumps(performance_metrics)

                update_last_log_row(self.current_log_row, self.log_file)

                return result

            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON from function call arguments: {str(e)}")

        except Exception as e:
            logger.error(f"Error in plan_execute: {str(e)}")
            # Re-throw the exception after logging it
            raise

    async def answer_user_question(self, user_question: str) -> Tuple[str, str]:
        """
        Process a user question through the complete MAPE-K workflow.
        
        Args:
            user_question: The user's question
            
        Returns:
            Tuple of (analysis, response)
        """
        try:
            # Track overall performance
            overall_start_time = time.time()

            # Append user message to chat history
            self.append_to_history("user", user_question)

            # Check if we have necessary state information
            if self.xai_report is None:
                logger.warning("XAI report not initialized. Response may be incomplete.")

            # Run the Monitor-Analyze component
            monitor_analyze_result = await self.monitor_analyze(user_question)

            # Run the Plan-Execute component
            plan_execute_result = await self.plan_execute(user_question, monitor_analyze_result)

            # Track overall performance
            overall_elapsed_time = time.time() - overall_start_time
            logger.info(f"Total time for MAPE-K workflow: {overall_elapsed_time:.2f}s")

            # Compose the reasoning output
            reasoning_parts = []

            if self.include_monitor_reasoning:
                reasoning_parts.append(f"MONITOR: {monitor_analyze_result['reasoning']}")

            if self.include_analyze_reasoning:
                reasoning_parts.append(f"ANALYZE: {monitor_analyze_result['analysis_reasoning']}")

            if self.include_plan_reasoning:
                reasoning_parts.append(
                    f"PLAN: Using {plan_execute_result['explanation_method']} - {plan_execute_result['rationale']}"
                )

            if self.include_execute_reasoning:
                reasoning_parts.append(f"EXECUTE: {plan_execute_result['reasoning']}")

            reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else plan_execute_result['reasoning']

            # Log the updated user model (user_model.get_state_summary(as_dict=True)}
            user_model_summary = self.user_model.get_state_summary(as_dict=True)
            logger.info(f"Updated user model: {user_model_summary}")

            # Return the analysis and response
            return reasoning, plan_execute_result['response']

        except Exception as e:
            logger.error(f"Error in answer_user_question: {str(e)}")

            error_response = f"I'm having trouble processing your question. Could you rephrase it?"
            self.append_to_history("agent", error_response)

            return "Error analyzing question", error_response

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the agent.
        
        Returns:
            Dictionary of performance metrics
        """
        return {
            "query_count": self.query_count,
            "total_latency": self.total_latency,
            "avg_latency": round(self.total_latency / self.query_count if self.query_count > 0 else 0, 2)
        }
