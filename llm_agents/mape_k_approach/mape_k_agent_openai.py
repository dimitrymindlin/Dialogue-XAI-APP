import asyncio, logging, csv, yaml, json, os, datetime, copy
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from openai import AsyncOpenAI

aclient = AsyncOpenAI()

from create_experiment_data.instance_datapoint import InstanceDatapoint
from llm_agents.base_agent import XAIBaseAgent
from llm_agents.explanation_state import ExplanationState
from llm_agents.mape_k_approach.plan_component.xai_exp_populator import XAIExplanationPopulator
from llm_agents.mape_k_approach.user_model.user_model_fine_grained import UserModelFineGrained as UserModel
from llm_agents.utils.definition_wrapper import DefinitionWrapper
from llm_agents.utils.postprocess_message import replace_plot_placeholders

# Import necessary prompt templates
from llm_agents.mape_k_approach.monitor_component.monitor_prompt import get_monitor_prompt_template, MonitorResultModel
from llm_agents.mape_k_approach.analyze_component.analyze_prompt import get_analyze_prompt_template, AnalyzeResult
from llm_agents.mape_k_approach.plan_component.advanced_plan_prompt_multi_step import get_plan_prompt_template, PlanResultModel, ChosenExplanationModel
from llm_agents.mape_k_approach.execute_component.execute_prompt import get_execute_prompt_template, ExecuteResult

# Configure logging
LOG_FOLDER = "mape-k-logs"
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MAPE_K_Workflow")

# Add file handler
file_handler = logging.FileHandler("logfile.txt", mode="w")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# CSV logging setup
LOG_CSV_FILE = "mape_log.csv"
CSV_HEADERS = ["timestamp", "experiment_id", "datapoint_count", "user_message", "monitor", "analyze", "plan", "execute", "user_model"]

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_ORGANIZATION"] = os.getenv('OPENAI_ORGANIZATION_ID')
OPENAI_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME')
OPENAI_MINI_MODEL_NAME = os.getenv('OPENAI_MINI_MODEL_NAME')

# Base directory: the directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

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

class MapeKXAIWorkflowAgent(XAIBaseAgent):
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
        self.explanation_plan = []
        self.last_shown_explanations = []  # save tuples of explanation and step
        self.visual_explanations_dict = None
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

    async def monitor(self, user_message):
        """Monitor step: Interpret data from sensors, monitoring understanding, interpreting user's cognitive state"""

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

        monitor_prompt = get_monitor_prompt_template().format(
            chat_history=self.chat_history,
            user_message=user_message,
            understanding_displays=self.understanding_displays.as_text(),
            modes_of_engagement=self.modes_of_engagement.as_text(),
        )

        start_time = datetime.datetime.now()

        # Use OpenAI API directly instead of LlamaIndex
        messages = [
            {"role": "system", "content": "You are an AI monitoring agent. Analyze the user's message."},
            {"role": "user", "content": monitor_prompt}
        ]

        # Define the function calling format for structured output
        functions = [{
            "name": "monitor_result",
            "description": "Return the monitoring results",
            "parameters": MonitorResultModel.schema()
        }]

        response = await aclient.chat.completions.create(model=OPENAI_MINI_MODEL_NAME,
        messages=messages,
        functions=functions,
        function_call={"name": "monitor_result"})

        # Extract and parse the function call result
        function_call = response.choices[0].message.function_call
        monitor_result = MonitorResultModel.parse_raw(function_call.arguments)

        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Monitor: {end_time - start_time}")
        logger.info(f"Monitor result: {monitor_result}.\n")

        # Update the 'monitor' cell - use json() to properly serialize the Pydantic object
        self.current_log_row["monitor"] = monitor_result.json()
        update_last_log_row(self.current_log_row, self.log_file)

        return monitor_result

    async def analyze(self, user_message, monitor_result):
        """Analyze step: Assessing user's cognitive state in explanation context, updating level of understanding"""

        # Update user model based on monitor results
        if monitor_result.mode_of_engagement != "":
            self.user_model.cognitive_state = self.modes_of_engagement.get_differentiating_description(
                monitor_result.mode_of_engagement)

        if len(monitor_result.explicit_understanding_displays) > 0:
            self.user_model.explicit_understanding_signals = monitor_result.explicit_understanding_displays

        # Create the analyze prompt
        analyze_prompt = get_analyze_prompt_template().format(
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
        )

        start_time = datetime.datetime.now()

        # Use OpenAI API directly
        messages = [
            {"role": "system", "content": "You are an AI analysis agent. Analyze the user's understanding."},
            {"role": "user", "content": analyze_prompt}
        ]

        # Define the function calling format for structured output
        functions = [{
            "name": "analyze_result",
            "description": "Return the analysis results",
            "parameters": AnalyzeResult.schema()
        }]

        response = await aclient.chat.completions.create(model=OPENAI_MODEL_NAME,
        messages=messages,
        functions=functions,
        function_call={"name": "analyze_result"})

        # Extract and parse the function call result
        function_call = response.choices[0].message.function_call
        analyze_result = AnalyzeResult.parse_raw(function_call.arguments)

        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Analyze: {end_time - start_time}")
        logger.info(f"Analyze result: {analyze_result}.\n")

        # UPDATE USER MODEL
        for change_entry in analyze_result.model_changes:
            try:
                exp = change_entry["explanation_name"]
                change = change_entry["state"]
                step = change_entry["step"]
                self.user_model.update_explanation_step_state(exp, step, change)
            except KeyError:
                logger.error(f"Invalid change entry: {change_entry}")
                continue

        # Update the 'analyze' cell - use json() to properly serialize the Pydantic object
        self.current_log_row["analyze"] = analyze_result.json()
        update_last_log_row(self.current_log_row, self.log_file)

        # Log new user model
        logger.info(f"User model after analyze: {self.user_model.get_state_summary(as_dict=True)}.\n")

        return analyze_result

    async def plan(self, user_message):
        """Plan step: General adaptation plans, choosing explanation strategy and moves"""
        
        # Get last explanation if available
        last_exp = self.last_shown_explanations[-1] if len(self.last_shown_explanations) > 0 else None
        
        # Create the plan prompt
        plan_prompt = get_plan_prompt_template().format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.chat_history,
            user_model=self.user_model.get_state_summary(as_dict=False),
            user_message=user_message,
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
            previous_plan=self.explanation_plan,
            last_explanation=last_exp
        )
        
        start_time = datetime.datetime.now()
        
        # Use OpenAI API directly
        messages = [
            {"role": "system", "content": "You are an AI planning agent. Create a plan for explaining concepts to the user."},
            {"role": "user", "content": plan_prompt}
        ]
        
        # Define the function calling format for structured output, using nested schemas for the ExplanationTarget model
        functions = [{
            "name": "plan_result",
            "description": "Return the planning results",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "The reasoning behind the decision for new explanations and which explanations to include in the next steps."
                    },
                    "new_explanations": {
                        "type": "array",
                        "description": "List of new explanations to be added to the explanation plan.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "explanation_name": {
                                    "type": "string",
                                    "description": "The name of the new explanation concept."
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Description of the new explanation concept."
                                },
                                "explanation_steps": {
                                    "type": "array",
                                    "description": "List of steps for the new explanation concept.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "step_name": {
                                                "type": "string",
                                                "description": "The name of the explanation step."
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "Description of the explanation step."
                                            },
                                            "dependencies": {
                                                "type": "array",
                                                "description": "List of dependencies for the explanation step.",
                                                "items": {
                                                    "type": "string"
                                                }
                                            },
                                            "is_optional": {
                                                "type": "boolean",
                                                "description": "Whether the explanation step is optional or not."
                                            }
                                        },
                                        "required": ["step_name", "description", "dependencies", "is_optional"]
                                    }
                                }
                            },
                            "required": ["explanation_name", "description", "explanation_steps"]
                        }
                    },
                    "explanation_plan": {
                        "type": "array",
                        "description": "List of explanations or scaffolding indicating long term steps to explain to the user.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "explanation_name": {
                                    "type": "string",
                                    "description": "The name of the explanation concept."
                                },
                                "step": {
                                    "type": "string", 
                                    "description": "The name or label of the step of the explanation."
                                }
                            },
                            "required": ["explanation_name", "step"]
                        }
                    },
                    "next_response": {
                        "type": "array",
                        "description": "A list of explanations and steps to include in the next response to answer the user's question.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "reasoning": {
                                    "type": "string",
                                    "description": "The reasoning behind the choice of the current explanandum."
                                },
                                "explanation_name": {
                                    "type": "string",
                                    "description": "The name of the current explanation concept."
                                },
                                "step_name": {
                                    "type": "string",
                                    "description": "Step name that is the current explanandum."
                                },
                                "communication_goals": {
                                    "type": "array",
                                    "description": "List of atomic goals while communicating the complete explanation to the user",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "goal": {
                                                "type": "string",
                                                "description": "Each goal should focus on a specific aspect"
                                            }
                                        },
                                        "required": ["goal"]
                                    }
                                }
                            },
                            "required": ["reasoning", "explanation_name", "step_name", "communication_goals"]
                        }
                    }
                },
                "required": ["reasoning", "new_explanations", "explanation_plan", "next_response"]
            }
        }]
        
        # Use the aclient instance with the updated API format
        response = await aclient.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=messages,
            functions=functions,
            function_call={"name": "plan_result"}
        )
        
        # Extract and parse the function call result - updated to match the new API format
        function_call = response.choices[0].message.function_call
        plan_result = PlanResultModel.parse_raw(function_call.arguments)
        
        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Plan: {end_time - start_time}")
        logger.info(f"Plan result: {plan_result}.\n")
        
        # Update Explanation Plan
        if len(plan_result.explanation_plan) > 0:
            self.explanation_plan = plan_result.explanation_plan

        # Add new explanations if any
        if len(plan_result.new_explanations) > 0:
            self.user_model.add_explanations_from_plan_result(plan_result.new_explanations)
            
        # Update the 'plan' cell - use json() to properly serialize the Pydantic object
        self.current_log_row["plan"] = plan_result.json()
        update_last_log_row(self.current_log_row, self.log_file)
        
        return plan_result

    async def execute(self, user_message, plan_result):
        """Execute step: Determining realization of explanation moves, performing selected action"""

        # Validate plan result structure
        if not hasattr(plan_result, 'explanation_plan') or not all(isinstance(exp, ChosenExplanationModel) for exp in plan_result.explanation_plan):
            logger.warning(f"Invalid plan result structure: {plan_result}")
            # Handle gracefully by creating empty list if needed
            if not hasattr(plan_result, 'explanation_plan'):
                plan_result.explanation_plan = []

        plan_reasoning = plan_result.reasoning

        # Get explanations from the plan
        xai_explanations_from_plan = self.user_model.get_string_explanations_from_plan(
            plan_result.explanation_plan)

        # Create the execute prompt
        execute_prompt = get_execute_prompt_template().format(
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
        )

        start_time = datetime.datetime.now()

        # Use OpenAI API directly
        messages = [
            {"role": "system", "content": "You are an AI execution agent. Generate a response to the user."},
            {"role": "user", "content": execute_prompt}
        ]

        # Define the function calling format for structured output using explicit schema
        functions = [{
            "name": "execute_result",
            "description": "Return the execution results",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "The reasoning behind the response generation."
                    },
                    "response": {
                        "type": "string",
                        "description": "The response to the user's question about the shown instance and prediction."
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Whether the execution was successful.",
                        "default": True
                    }
                },
                "required": ["reasoning", "response"]
            }
        }]

        response = await aclient.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=messages,
            functions=functions,
            function_call={"name": "execute_result"}
        )

        # Extract and parse the function call result
        function_call = response.choices[0].message.function_call
        execute_result = ExecuteResult.parse_raw(function_call.arguments)

        end_time = datetime.datetime.now()
        logger.info(f"Time taken for Execute: {end_time - start_time}")
        logger.info(f"Execute result: {execute_result}.\n")

        # Log and save history before replacing placeholders
        self.current_log_row["execute"] = execute_result.json()
        update_last_log_row(self.current_log_row, self.log_file)

        # Only append the agent's response to chat history (user message was added at the beginning)
        self.append_to_history("agent", execute_result.response)

        # Check if result has placeholders and replace them
        execute_result.response = replace_plot_placeholders(execute_result.response, self.visual_explanations_dict)

        # Update Explanandum state and User Model
        # Update user model based on next explanations
        for next_explanation in plan_result.next_response:
            explanation_target = next_explanation
            exp = explanation_target.explanation_name
            exp_step = explanation_target.step_name
            self.user_model.update_explanation_step_state(exp, exp_step, ExplanationState.UNDERSTOOD.value)

        # Add next explanations to last shown explanations
        for next_explanation in plan_result.next_response:
            self.last_shown_explanations.append(next_explanation)

        # Use json.dumps for user_model since it's not a Pydantic object but returns a dict from get_state_summary
        self.current_log_row["user_model"] = json.dumps(self.user_model.get_state_summary(as_dict=True))
        update_last_log_row(self.current_log_row, self.log_file)
        self.user_model.new_datapoint()

        # Log new user model
        logger.info(f"User model after execute: {self.user_model.get_state_summary(as_dict=False)}.\n")

        return execute_result

    # Main method to run the full MAPE-K workflow
    async def answer_user_question(self, user_question):
        """Run the full MAPE-K workflow to answer a user question"""
        start_time = datetime.datetime.now()

        # Append user message to chat history at the beginning of the workflow
        # This ensures all components have access to the updated chat history
        self.append_to_history("user", user_question)

        # Step 1: Monitor
        monitor_result = await self.monitor(user_question)

        # Step 2: Analyze
        analyze_result = await self.analyze(user_message=user_question, monitor_result=monitor_result)

        # Step 3: Plan
        plan_result = await self.plan(user_message=user_question)

        # Step 4: Execute
        execute_result = await self.execute(user_message=user_question, plan_result=plan_result)

        end_time = datetime.datetime.now()
        logger.info(f"Time taken for full MAPE-K Loop: {end_time - start_time}")

        analysis = execute_result.reasoning
        response = execute_result.response

        return analysis, response