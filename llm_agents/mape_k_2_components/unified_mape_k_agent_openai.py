import csv
import copy
import json
import time
import os
import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
import asyncio
import traceback

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
import agents
from agents import Agent, Runner, Tool
from agents import RunResult
from agents import FunctionTool
logger = logging.getLogger(__name__)

try:

    from agents import AgentSystemMessage
    from agents import ToolCall, ToolResult
except ImportError:
    try:

        from agents import ToolCall, ToolResult
        from agents import AgentSystemMessage
    except ImportError:

        logger.warning("Could not import OpenAI Agents SDK classes - using mock implementations")
        class ToolCall:
            def __init__(self, *args, **kwargs):
                self.result = None
        
        class ToolResult:
            def __init__(self, *args, **kwargs):
                pass
        
        class AgentSystemMessage:
            def __init__(self, content=""):
                self.content = content
        
        class FunctionTool:
            def __init__(self, fn=None, name=None, description=None):
                self.fn = fn
                self.name = name or (fn.__name__ if fn else "unknown")
                self.description = description or "Mocked function tool"
            
            @staticmethod
            def from_defaults(fn=None, name=None, description=None):
                return FunctionTool(fn=fn, name=name, description=description)

from create_experiment_data.instance_datapoint import InstanceDatapoint
from llm_agents.base_agent import XAIBaseAgent
from llm_agents.explanation_state import ExplanationState
from llm_agents.utils.definition_wrapper import DefinitionWrapper
from llm_agents.mape_k_approach.plan_component.xai_exp_populator import XAIExplanationPopulator
from llm_agents.mape_k_approach.user_model.user_model_fine_grained import UserModelFineGrained as UserModel
from llm_agents.utils.postprocess_message import replace_plot_placeholders
from llm_agents.merged_prompts import get_merged_prompts

# Import the DirectOpenAIFallback as a last resort
try:
    from llm_agents.mape_k_2_components.fallback_direct_openai import DirectOpenAIFallback
    DIRECT_FALLBACK_AVAILABLE = True
except ImportError:
    logger.warning("DirectOpenAIFallback not available - won't be able to use it for fallback")
    DIRECT_FALLBACK_AVAILABLE = False

LOG_FOLDER = "mape-k-logs"
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

PERFORMANCE_LOG_FOLDER = "performance-logs"
if not os.path.exists(PERFORMANCE_LOG_FOLDER):
    os.makedirs(PERFORMANCE_LOG_FOLDER)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_ORGANIZATION"] = os.getenv('OPENAI_ORGANIZATION_ID')
LLM_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME') or "gpt-4-turbo"

# Setup logging
logger = logging.getLogger(__name__)

# Base directory: the directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Configure a file handler
file_handler = logging.FileHandler("openai_agent_logfile.txt", mode="w")
file_handler.setLevel(logging.INFO)

# Define a custom formatter for more readable output
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n"  # newline for readability
)
file_handler.setFormatter(formatter)

LOG_CSV_FILE = "openai_agent_log_table.csv"
CSV_HEADERS = ["timestamp", "experiment_id", "datapoint_count", "user_message", "monitor", "analyze", "plan", "execute", "user_model", "processing_time"]

# Performance logging
performance_logger = logging.getLogger("performance_logger")
performance_file_handler = logging.FileHandler(os.path.join(PERFORMANCE_LOG_FOLDER, "performance_comparison.log"), mode="a")
performance_file_handler.setLevel(logging.INFO)
performance_formatter = logging.Formatter(fmt="%(asctime)s - %(message)s")
performance_file_handler.setFormatter(performance_formatter)
performance_logger.addHandler(performance_file_handler)
performance_logger.setLevel(logging.INFO)


def generate_log_file_name(experiment_id: str) -> str:
    timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H:%M")
    return os.path.join(LOG_FOLDER, f"{timestamp}_openai_agent_{experiment_id}.csv")


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


# Optional: Add a console handler with the same or a simpler format
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(fmt="%(levelname)s - %(message)s"))

# Set up the logger with both handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


# Define Pydantic models for structured output (same as in unified_mape_k_agent.py)
class NewExplanationModel(BaseModel):
    """
    Data model for a new explanation concept to be added.
    """
    name: str = Field(..., description="The name of the explanation concept")
    description: str = Field(None, description="Brief justification for this explanation")
    dependencies: List[str] = Field(None, description="List of dependencies")
    is_optional: bool = Field(None, description="Whether this explanation is optional")


class ChosenExplanationModel(BaseModel):
    """
    Data model for a chosen explanation concept to be added to the explanation plan.
    """
    explanation_name: str = Field(..., description="The name of the explanation concept.")
    step: str = Field(..., description="The name or label of the step of the explanation.")
    description: str = Field(None, description="Brief justification for this explanation")
    dependencies: List[str] = Field(None, description="List of dependencies")
    is_optional: bool = Field(None, description="Whether this explanation is optional")


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
    communication_goals: List[CommunicationGoal] = Field(None, description="Communication goals for this explanation")


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
    monitor_reasoning: str = Field(None, description="Reasoning about the user's understanding")
    explicit_understanding_displays: List[str] = Field(None, description="List of explicit understanding displays from the user message")
    mode_of_engagement: str = Field(None, description="The cognitive mode of engagement from the user message")
    
    # Analyze stage
    analyze_reasoning: str = Field(None, description="Reasoning behind analyzing user model changes")
    model_changes: List[ModelChange] = Field(None, description="List of changes to make to the user model")
    
    # Plan stage
    plan_reasoning: str = Field(None, description="Reasoning behind the explanation plan")
    new_explanations: List[NewExplanationModel] = Field(None, description="List of new explanations to add")
    explanation_plan: List[ChosenExplanationModel] = Field(None, description="List of chosen explanations for the plan")
    next_response: List[ExplanationTarget] = Field(None, description="List of explanation targets for the next response")
    
    # Execute stage
    execute_reasoning: str = Field(None, description="Reasoning behind the constructed response")
    response: str = Field(None, description="The HTML-formatted response to the user")


class ExecuteResult(BaseModel):
    """
    Model for execution result to maintain compatibility with existing code.
    """
    reasoning: str = Field(..., description="The reasoning behind the response.")
    response: str = Field(..., description="The HTML-formatted response to the user.")


class UnifiedMapeKOpenAIAgent(XAIBaseAgent):
    """
    OpenAI Agents SDK implementation of the MAPE-K agent. 
    Uses the same interface as UnifiedMapeKAgent but with OpenAI Agents SDK under the hood.
    """
    def __init__(
            self,
            feature_names="",
            domain_description="",
            user_ml_knowledge="",
            experiment_id="",
            model_name=LLM_MODEL_NAME,
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
        self.model_name = model_name
        
        # Initialize OpenAI client
        self.openai_client = OpenAI()
        
        # Create the OpenAI Agent with improved error handling for different SDK versions
        try:
            # Try to create the agent with full configuration
            system_message = AgentSystemMessage("You are an advanced Explainable AI (XAI) assistant specializing in MAPE-K reasoning.")
            tool_fn = self.process_mape_k
            
            # Check if FunctionTool.from_defaults exists
            if hasattr(FunctionTool, 'from_defaults'):
                function_tool = FunctionTool.from_defaults(fn=tool_fn)
                tools = [function_tool]
            else:
                # Alternative approach if from_defaults doesn't exist
                logger.warning("FunctionTool.from_defaults not available, using alternative tool initialization")
                tools = [tool_fn]  # Many versions accept the function directly as a tool
            
            self.agent = Agent(
                name="MAPE-K-Agent",  # Always provide the name parameter
                model=model_name,
                instructions=system_message,
                tools=tools
            )
            logger.info("Successfully created agent with full configuration")
            
        except (TypeError, AttributeError) as e:
            logger.warning(f"Error creating agent with standard API: {e}. Trying simplified approach...")
            try:
                # Simplified approach with only required parameters
                self.agent = Agent(
                    name="MAPE-K-Agent",  # Always provide the name parameter
                    model=model_name,
                    instructions=system_message,
                    tools=tools
                )
                logger.info("Created agent with simplified configuration")
                
                # Try to register the tool if that method exists
                if hasattr(self.agent, 'register_tool'):
                    self.agent.register_tool(self.process_mape_k)
                    logger.info("Registered tool with agent.register_tool method")
            except Exception as e2:
                logger.error(f"Could not create agent with simplified API either: {e2}")
                raise RuntimeError(f"Failed to initialize OpenAI Agent: {e}; then: {e2}")
        
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

        # MAPE-K specific setup user understanding notepad
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
                
    def process_mape_k(self, prompt: str) -> Dict[str, Any]:
        """
        Function tool that processes a MAPE-K flow based on the provided prompt.
        This directly mimics the unified_mape_k step in the original implementation
        but uses OpenAI API instead of llama_index.
        
        Args:
            prompt (str): The user's query or input
            
        Returns:
            Dict[str, Any]: The MAPE-K result in structured format
        """
        # Create the unified prompt with all necessary context - exactly as in unified_mape_k
        prompt_template_str = get_merged_prompts()
        prompt_template_str = prompt_template_str.format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            understanding_displays=self.understanding_displays.as_text(),
            modes_of_engagement=self.modes_of_engagement.as_text(),
            chat_history=self.chat_history,
            user_message=prompt,
            user_model=self.user_model.get_state_summary(as_dict=False),
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            explanation_plan=self.explanation_plan,
            last_shown_explanations=self.last_shown_explanations
        )
        
        try:
            # Use OpenAI's structured output feature with our Pydantic model
            completion = self.openai_client.responses.parse(
                model=self.model_name,
                input=[{"role": "user", "content": prompt_template_str}],
                text_format=MAPE_K_ResultModel,
                temperature=0
            )
            
            # Get the parsed structured output
            result = completion.output_parsed
            logger.info(f"Structured output parsing processing...")
            # Convert to dictionary format for compatibility with existing code
            result_json = {
                "Monitor": {
                    "monitor_reasoning": result.monitor_reasoning or "",
                    "understanding_displays": result.explicit_understanding_displays or [],
                    "cognitive_state": result.mode_of_engagement or "active"
                },
                "Analyze": {
                    "analyze_reasoning": result.analyze_reasoning or "",
                    "updated_explanation_states": {
                        change.explanation_name: change.state for change in (result.model_changes or [])
                    }
                },
                "Plan": {
                    "reasoning": result.plan_reasoning or "",
                    "next_explanations": [
                        {
                            "name": exp.explanation_name,
                            "description": exp.description or "",
                            "dependencies": exp.dependencies or [],
                            "is_optional": exp.is_optional if exp.is_optional is not None else False
                        } for exp in (result.explanation_plan or [])
                    ],
                    "new_explanations": [
                        {
                            "name": exp.name,
                            "description": exp.description or "",
                            "dependencies": exp.dependencies or [],
                            "is_optional": exp.is_optional if exp.is_optional is not None else False
                        } for exp in (result.new_explanations or [])
                    ]
                },
                "Execute": {
                    "execute_reasoning": result.execute_reasoning or "",
                    "html_response": result.response or "I apologize, but I'm having difficulty generating a proper response."
                }
            }
            logger.info(f"Structured response parsed successfully: {result_json}")
            # Log the structured response
            return result_json
            
        except Exception as e:
            logger.error(f"Error with structured output parsing: {e}")
            logger.error(traceback.format_exc())
            
            # Fallback to traditional approach if structured parsing fails
            try:
                # Make the traditional API call
                completion = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt_template_str}],
                    temperature=0
                )
                
                # Extract the response content
                response_text = completion.choices[0].message.content
                logger.info(f"Raw LLM response (fallback): {response_text}")
                
                # Extract the JSON part from the LLM response
                json_text = response_text.strip()
                # Sometimes LLM might wrap JSON in markdown code blocks, so handle that
                if json_text.startswith("```json"):
                    json_text = json_text.split("```json")[1].split("```")[0].strip()
                elif json_text.startswith("```"):
                    json_text = json_text.split("```")[1].split("```")[0].strip()
                    
                result_json = json.loads(json_text)
                return result_json
            
            except (json.JSONDecodeError, KeyError, IndexError) as json_error:
                logger.error(f"Error processing MAPE-K response (fallback): {json_error}\nResponse text: {response_text}")
                # Return a fallback response
                return {
                    "Monitor": {
                        "understanding_displays": [],
                        "cognitive_state": "active",
                        "monitor_reasoning": "Error processing response"
                    },
                    "Analyze": {
                        "updated_explanation_states": {},
                        "analyze_reasoning": "Error processing response"
                    },
                    "Plan": {
                        "next_explanations": [],
                        "new_explanations": [],
                        "reasoning": "Error processing response"
                    },
                    "Execute": {
                        "html_response": "I apologize, but I encountered a technical issue processing your request. Could you please try rephrasing your question?",
                        "execute_reasoning": "Error fallback response"
                    }
                }

    async def run_mape_k_workflow(self, user_input: str) -> ExecuteResult:
        """
        Runs the MAPE-K workflow using the OpenAI Agents SDK
        
        Args:
            user_input (str): The user's message or query
            
        Returns:
            ExecuteResult: The final result with reasoning and response
        """
        start_time = datetime.datetime.now()
        
        # Initialize log row
        self.current_log_row = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_id": self.experiment_id,
            "datapoint_count": self.datapoint_count,
            "user_message": user_input,
            "monitor": "",
            "analyze": "",
            "plan": "",
            "execute": "",
            "user_model": "",
            "processing_time": ""
        }
        append_new_log_row(self.current_log_row, self.log_file)
        
        try:
            # Try different ways to run the agent based on SDK version
            result = None
            
            try:
                # Try the newer API first
                runner = Runner(tools=[self.process_mape_k])
                result = await runner.run(self.agent, user_input)
                logger.info("Successfully ran agent with newer runner.run API")
            except (TypeError, AttributeError, ValueError) as e:
                logger.warning(f"Error running agent with newer API: {e}. Trying alternatives...")
                try:
                    # Alternative run method
                    result = await Runner.run_sync(
                        self.agent,
                        user_input,
                        tools=[self.process_mape_k]
                    )
                    logger.info("Successfully ran agent with Runner.run_sync API")
                except Exception as e2:
                    logger.error(f"Error with alternative API: {e2}")
                    # Try a direct call to process_mape_k
                    try:
                        logger.warning("Using fallback direct call to process_mape_k...")
                        mape_k_json = self.process_mape_k(user_input)
                        
                        # Create a mock result object
                        from types import SimpleNamespace
                        result = SimpleNamespace()
                        result.final_output = "Processed via direct API call"
                        mock_tool_call = SimpleNamespace()
                        mock_tool_call.result = mape_k_json
                        result.final_tool_calls = [mock_tool_call]
                        logger.info("Created mock result from direct process_mape_k call")
                    except Exception as e3:
                        logger.error(f"Direct process_mape_k call failed: {e3}")
                        # Last resort - use DirectOpenAIFallback if available
                        if DIRECT_FALLBACK_AVAILABLE:
                            logger.warning("Attempting to use DirectOpenAIFallback as last resort...")
                            try:
                                # Create fallback object with same model
                                fallback = DirectOpenAIFallback(model_name=self.model_name)
                                
                                # Get the prompt template
                                prompt_template = self.construct_mape_k_prompt_template()
                                
                                # Process via fallback
                                reasoning, response = await fallback.direct_mape_k_call(
                                    prompt=prompt_template,
                                    user_message=user_input
                                )
                                
                                # Create a mock result
                                from types import SimpleNamespace
                                result = SimpleNamespace()
                                result.final_output = response
                                
                                # Create a simplified MAPE-K result
                                mape_k_result = {
                                    "Monitor": {"monitor_reasoning": "Processed via DirectOpenAIFallback"},
                                    "Analyze": {"analyze_reasoning": "Processed via DirectOpenAIFallback"},
                                    "Plan": {"reasoning": "Processed via DirectOpenAIFallback"},
                                    "Execute": {
                                        "execute_reasoning": reasoning,
                                        "html_response": response
                                    }
                                }
                                
                                mock_tool_call = SimpleNamespace()
                                mock_tool_call.result = mape_k_result
                                result.final_tool_calls = [mock_tool_call]
                                logger.info("Successfully used DirectOpenAIFallback")
                            except Exception as e4:
                                logger.error(f"DirectOpenAIFallback failed too: {e4}")
                                raise RuntimeError("All methods to run the agent failed")
                        else:
                            raise RuntimeError("All methods to run the agent failed and DirectOpenAIFallback not available")
            
            # Check if result is still None after all attempts
            if result is None:
                raise RuntimeError("All methods to run the agent failed")
            
            # Check if we have valid tool calls and results
            has_tool_calls = hasattr(result, 'final_tool_calls') and result.final_tool_calls
            has_tool_result = has_tool_calls and len(result.final_tool_calls) > 0 and hasattr(result.final_tool_calls[0], 'result')
            
            if not has_tool_calls or not has_tool_result or not result.final_tool_calls[0].result:
                # If no tool was called, use the final output as a fallback
                mape_k_result = MAPE_K_ResultModel(
                    monitor_reasoning="No valid tool result",
                    mode_of_engagement="active",
                    analyze_reasoning="No valid tool result",
                    plan_reasoning="No valid tool result",
                    execute_reasoning="Fallback response",
                    response=getattr(result, 'final_output', None) or "I apologize, but I couldn't process your request properly."
                )
            else:
                # Get the result from the tool call
                mape_k_json = result.final_tool_calls[0].result
                
                # Parse into MAPE_K_ResultModel
                mape_k_result = MAPE_K_ResultModel(
                    # Monitor stage
                    monitor_reasoning=mape_k_json["Monitor"].get("monitor_reasoning", ""),
                    explicit_understanding_displays=mape_k_json["Monitor"].get("understanding_displays", []),
                    mode_of_engagement=mape_k_json["Monitor"].get("cognitive_state", ""),
                    
                    # Analyze stage
                    analyze_reasoning=mape_k_json["Analyze"].get("analyze_reasoning", ""),
                    model_changes=[],  # Will be populated below
                    
                    # Plan stage
                    plan_reasoning=mape_k_json["Plan"].get("reasoning", ""),
                    new_explanations=[],  # Will be populated from next_explanations
                    explanation_plan=[
                        ChosenExplanationModel(
                            explanation_name=exp.get("name"),
                            step=exp.get("dependencies", [""])[0] if exp.get("dependencies") else "",
                            description=exp.get("description", ""),
                            dependencies=exp.get("dependencies", []),
                            is_optional=exp.get("is_optional", False)
                        )
                        for exp in mape_k_json["Plan"].get("next_explanations", [])
                    ],
                    next_response=[],  # Will be populated during processing
                    
                    # Execute stage
                    execute_reasoning=mape_k_json["Execute"].get("execute_reasoning", ""),
                    response=mape_k_json["Execute"].get("html_response", "")
                )
                
                # Process model changes to fully populate the state
                analyze_model_changes = []
                for exp_name, new_state in mape_k_json["Analyze"].get("updated_explanation_states", {}).items():
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
                        reasoning="From OpenAI Agent MAPE-K call",
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
            
            end_time = datetime.datetime.now()
            time_elapsed = (end_time - start_time).total_seconds()
            self.current_log_row["processing_time"] = str(time_elapsed)
            
            update_last_log_row(self.current_log_row, self.log_file)
            
            # Log execution time
            logger.info(f"Time taken for OpenAI Agent MAPE-K: {end_time - start_time}")
            
            # Log performance data for comparison
            log_performance_data("openai_agents", time_elapsed, user_input, self.experiment_id)
            
            # Update chat history
            self.append_to_history("user", user_input)
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
            logger.info(f"User model after OpenAI Agent MAPE-K: {self.user_model.get_state_summary(as_dict=False)}.\n")
            
            # Create final result object for the workflow
            final_result = ExecuteResult(
                reasoning=mape_k_result.execute_reasoning,
                response=response_with_plots
            )
            
            return final_result
            
        except Exception as e:
            end_time = datetime.datetime.now()
            time_elapsed = (end_time - start_time).total_seconds()
            logger.error(f"Error in OpenAI Agent MAPE-K: {e}")
            logger.error(traceback.format_exc())
            
            # Log performance data even for errors
            log_performance_data("openai_agents_error", time_elapsed, user_input, self.experiment_id)
            
            # Provide a fallback response when processing fails
            fallback_result = ExecuteResult(
                reasoning="Error in OpenAI Agent MAPE-K",
                response="I apologize, but I encountered a technical issue processing your request. Could you please try rephrasing your question?"
            )
            return fallback_result

    async def answer_user_question(self, user_question):
        """
        Public method to answer a user question using the OpenAI Agent MAPE-K workflow.
        This directly mimics the approach in UnifiedMapeKAgent.answer_user_question.
        """
        start_time = datetime.datetime.now()
        
        try:
            # Directly call process_mape_k with the user question
            # This mimics the Workflow.run(input=user_question) in the original
            result_json = self.process_mape_k(user_question)
            
            # Parse the result json into a structured result object
            # In the original code, this would be the StopEvent(result=...) result
            result = ExecuteResult(
                reasoning=result_json["Execute"].get("execute_reasoning", "Execution complete"),
                response=result_json["Execute"].get("html_response", "")
            )
            
            end_time = datetime.datetime.now()
            time_elapsed = (end_time - start_time).total_seconds()
            logger.info(f"Total time for OpenAI Agent MAPE-K workflow: {end_time - start_time}")
            
            # Log performance data for comparison
            log_performance_data("openai_agents", time_elapsed, user_question, self.experiment_id)
            
            # Update chat history
            self.append_to_history("user", user_question)
            self.append_to_history("agent", result.response)
            
            # Replace plot placeholders with actual visual content
            from llm_agents.utils.postprocess_message import replace_plot_placeholders
            response_with_plots = replace_plot_placeholders(result.response, self.visual_explanations_dict)
            
            # Return reasoning and response to match the expected interface
            return result.reasoning, response_with_plots
            
        except Exception as e:
            end_time = datetime.datetime.now()
            time_elapsed = (end_time - start_time).total_seconds()
            
            logger.error(f"Error in answer_user_question: {e}")
            logger.error(traceback.format_exc())
            
            # Log performance for errors
            performance_logger.info(f"TOTAL_TIME,openai_agents_error,{time_elapsed:.4f}s,{self.experiment_id},\"{user_question[:50]}...\"")
            
            # Create a simple fallback response
            fallback_message = "I apologize, but I encountered a technical issue. Please try again or rephrase your question."
            
            # Update chat history even for errors
            self.append_to_history("user", user_question)
            self.append_to_history("agent", fallback_message)
            
            return "Error processing request", fallback_message

    def construct_mape_k_prompt_template(self) -> str:
        """
        Constructs a prompt template for the MAPE-K framework that can be used
        by the fallback mechanism.
        
        Returns:
            str: A formatted prompt template string
        """
        # Get the base prompts
        merged_prompts = get_merged_prompts(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            predicted_class_name=self.predicted_class_name or "unknown",
            opposite_class_name=self.opposite_class_name or "unknown",
            instance=str(self.instance) if self.instance is not None else "No instance provided"
        )
        
        # Construct a simplified template
        template = f"""
You are an advanced Explainable AI (XAI) assistant specializing in the MAPE-K framework.

DOMAIN DESCRIPTION:
{self.domain_description}

FEATURE NAMES:
{self.feature_names}

USER'S MESSAGE:
{{user_message}}

Please provide a response following the MAPE-K framework:

1. MONITOR: Analyze the user's query to understand their needs
2. ANALYZE: Determine what explanations are needed
3. PLAN: Create a plan for providing explanations
4. EXECUTE: Generate the final response for the user

Format your response as a JSON object with the following structure:
{{
  "Monitor": {{
    "monitor_reasoning": "Your reasoning for the monitoring phase"
  }},
  "Analyze": {{
    "analyze_reasoning": "Your reasoning for the analysis phase"
  }},
  "Plan": {{
    "reasoning": "Your reasoning for the planning phase"
  }},
  "Execute": {{
    "execute_reasoning": "Your reasoning for the execution phase",
    "html_response": "Your final response to the user in HTML format"
  }}
}}
"""
        return template


# For demonstration purposes
async def test_agent():
    agent = UnifiedMapeKOpenAIAgent(
        feature_names="age, workclass, education, marital_status, occupation",
        domain_description="This dataset contains census data",
        user_ml_knowledge="low",
        experiment_id="test"
    )
    
    result = await agent.answer_user_question("How does this model work?")
    print(f"Response: {result[1]}")


if __name__ == "__main__":
    asyncio.run(test_agent()) 