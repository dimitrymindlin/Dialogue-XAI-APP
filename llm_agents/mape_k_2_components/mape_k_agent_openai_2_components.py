import csv
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List

# Add fast Pydantic adapter import
from pydantic import TypeAdapter

aclient = AsyncOpenAI()

from create_experiment_data.instance_datapoint import InstanceDatapoint
from llm_agents.base_agent import XAIBaseAgent
from llm_agents.explanation_state import ExplanationState
from llm_agents.mape_k_approach.plan_component.xai_exp_populator import XAIExplanationPopulator
from llm_agents.mape_k_approach.user_model.user_model_fine_grained import UserModelFineGrained as UserModel
from llm_agents.utils.definition_wrapper import DefinitionWrapper
from llm_agents.utils.postprocess_message import replace_plot_placeholders

# Import necessary prompt templates
from llm_agents.mape_k_approach.monitor_component.monitor_prompt import MonitorResultModel
from llm_agents.mape_k_approach.analyze_component.analyze_prompt import AnalyzeResult
from llm_agents.mape_k_approach.plan_component.advanced_plan_prompt_multi_step import PlanResultModel
from llm_agents.mape_k_approach.execute_component.execute_prompt import ExecuteResult
# Import the combined monitor-analyze model and prompt
from llm_agents.mape_k_2_components.monitor_analyze_combined import get_monitor_analyze_prompt_template, MonitorAnalyzeResultModel
# Import the new combined plan-execute model and prompt
from llm_agents.mape_k_2_components.plan_execute_combined import get_plan_execute_prompt_template, PlanExecuteResultModel

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

class MapeKXAIWorkflowAgent(XAIBaseAgent):
    def __init__(
            self,
            feature_names="",
            domain_description="",
            user_ml_knowledge="",
            experiment_id="",
            include_monitor_reasoning=False,
            include_analyze_reasoning=False,
            include_plan_reasoning=True,
            include_execute_reasoning=True,
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
            os.path.join(base_dir, "../mape_k_approach", "monitor_component", "understanding_displays_definition.json"))
        self.modes_of_engagement = DefinitionWrapper(
            os.path.join(base_dir, "../mape_k_approach", "monitor_component", "icap_modes_definition.json")
        )
        self.explanation_questions = DefinitionWrapper(
            os.path.join(base_dir, "../mape_k_approach", "monitor_component", "explanation_questions_definition.json")
        )

        # Flags to control reasoning inclusion in responses
        self.include_monitor_reasoning = include_monitor_reasoning
        self.include_analyze_reasoning = include_analyze_reasoning
        self.include_plan_reasoning = include_plan_reasoning
        self.include_execute_reasoning = include_execute_reasoning

        # Pre-compile JSON adapters for faster parsing
        self._monitor_adapter = TypeAdapter(MonitorAnalyzeResultModel)
        self._plan_adapter = TypeAdapter(PlanExecuteResultModel)

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
            template_dir="/Users/dimitrymindlin/UniProjects/Dialogue-XAI-APP",
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

    def _apply_plan_execute_results(self, result: PlanExecuteResultModel):
        """Side-effect handler: update explanation plan and user model based on plan_execute results."""
        if result.explanation_plan:
            self.explanation_plan = result.explanation_plan
            logger.info(f"Updated explanation plan with {len(result.explanation_plan)} items.")
        if result.new_explanations:
            self.user_model.add_explanations_from_plan_result(result.new_explanations)
            logger.info(f"Added {len(result.new_explanations)} new explanations to user model.")

    def _apply_monitor_results(self, result: MonitorAnalyzeResultModel):
        """Side-effect handler: update user_model based on monitor results."""
        if result.mode_of_engagement:
            self.user_model.cognitive_state = self.modes_of_engagement.get_differentiating_description(
                result.mode_of_engagement
            )
        if result.explicit_understanding_displays:
            self.user_model.explicit_understanding_signals = result.explicit_understanding_displays
        logger.info(
            f"Applied monitor results: mode_of_engagement={result.mode_of_engagement}, "
            f"displays={result.explicit_understanding_displays}"
        )

    async def monitor_analyze(self, user_message):
        """Combined Monitor and Analyze steps: Process the user's cognitive state in a single API call"""

        self.current_log_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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

        # Create the combined prompt
        monitor_analyze_prompt = get_monitor_analyze_prompt_template().format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.chat_history,
            understanding_displays=self.understanding_displays.as_text(),
            modes_of_engagement=self.modes_of_engagement.as_text(),
            user_model=self.user_model.get_state_summary(as_dict=False),
            last_shown_explanations=self.last_shown_explanations,
            user_message=user_message,
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
        )

        start_time = datetime.now()

        # Use OpenAI API directly
        messages = [
            {"role": "system", "content": "You are an AI agent analyzing user understanding. Perform both monitor and analyze tasks."},
            {"role": "user", "content": monitor_analyze_prompt}
        ]

        # Define the function calling format for structured output
        functions = [
            {
                "name": "monitor_analyze_result",
                "description": "Return the combined monitoring and analysis results",
                "parameters": MonitorAnalyzeResultModel.schema()
            },
            {
                "name": "apply_monitor_results",
                "description": "Apply monitoring results to user model",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode_of_engagement": {"type": "string"},
                        "explicit_understanding_displays": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "reasoning": {"type": "string"}
                    },
                    "required": ["mode_of_engagement"]
                }
            }
        ]

        response = await aclient.chat.completions.create(
            model=OPENAI_MINI_MODEL_NAME,
            messages=messages,
            functions=functions,
            function_call="auto",
            stream=True
        )

        # Stream and accumulate function_call arguments for faster first-byte time
        func_args_json = ""
        async for chunk in response:
            cd = chunk.choices[0].delta
            fc = getattr(cd, "function_call", None)
            if fc and getattr(fc, "arguments", None):
                func_args_json += fc.arguments
        # Parse directly via Pydantic's fast JSON parser
        combined_result = self._monitor_adapter.validate_json(func_args_json)

        # Invoke side-effect tool
        self._apply_monitor_results(combined_result)

        end_time = datetime.now()
        logger.info(f"Time taken for Combined Monitor-Analyze: {end_time - start_time}")

        # Conditionally log the result based on reasoning inclusion flags
        if self.include_monitor_reasoning and self.include_analyze_reasoning:
            logger.info(f"Combined Monitor-Analyze result: {combined_result}.\n")
        else:
            # Create a modified version without reasoning if it should be excluded
            log_result = combined_result.dict()
            if not self.include_monitor_reasoning:
                log_result["reasoning"] = "[reasoning excluded]"
            if not self.include_analyze_reasoning:
                log_result["analysis_reasoning"] = "[reasoning excluded]"
            logger.info(f"Combined Monitor-Analyze result (filtered reasoning): {log_result}.\n")

        # Update the user model based on analyze results
        for change_entry in combined_result.model_changes:
            try:
                exp = change_entry["explanation_name"]
                change = change_entry["state"]
                step = change_entry["step"]
                self.user_model.update_explanation_step_state(exp, step, change)
            except KeyError:
                logger.error(f"Invalid change entry: {change_entry}")
                continue

        # Create separate results for logging purposes with optional reasoning
        monitor_result = MonitorResultModel(
            reasoning=combined_result.reasoning if self.include_monitor_reasoning else "[reasoning excluded]",
            explicit_understanding_displays=combined_result.explicit_understanding_displays,
            mode_of_engagement=combined_result.mode_of_engagement
        )

        analyze_result = AnalyzeResult(
            reasoning=combined_result.analysis_reasoning if self.include_analyze_reasoning else "[reasoning excluded]",
            model_changes=combined_result.model_changes
        )

        # Update the log cells
        self.current_log_row["monitor"] = monitor_result.json()
        self.current_log_row["analyze"] = analyze_result.json()
        update_last_log_row(self.current_log_row, self.log_file)

        # Log new user model
        logger.info(f"User model after combined monitor-analyze: {self.user_model.get_state_summary(as_dict=True)}.\n")

        return monitor_result, analyze_result

    def _normalize_plan_execute_args(self, function_args: dict) -> dict:
        """
        Ensure required fields exist and provide fallbacks for plan_execute function_args.
        """
        # Required top-level fields
        required_fields = [
            "planning_reasoning", "new_explanations", "explanation_plan",
            "next_response", "execution_reasoning", "response", "success"
        ]
        for field in required_fields:
            if field not in function_args:
                # lists for these fields
                if field in ["new_explanations", "explanation_plan", "next_response"]:
                    function_args[field] = []
                elif field == "success":
                    function_args[field] = True
                else:
                    function_args[field] = ""
            # convert None to empty list for list fields
            elif field in ["new_explanations", "explanation_plan", "next_response"] and function_args[field] is None:
                function_args[field] = []

        # Check and ensure all required fields are present in next_response
        if isinstance(function_args.get("next_response"), list):
            valid_next_responses = []
            for i, item in enumerate(function_args.get("next_response", [])):
                # If item is not a dict, create a fallback item
                if not isinstance(item, dict):
                    logger.warning(f"next_response item {i} is not a dict: {item}")
                    valid_next_responses.append({
                        "reasoning": "[auto-generated reasoning]",
                        "explanation_name": "ModelPredictionConfidence",
                        "step_name": "Concept",
                        "communication_goals": [{"goal": "Provide a response to the user"}]
                    })
                    continue

                # Handle case where we only have reasoning but missing other required fields
                valid_item = dict(item)  # Create a copy to modify

                # Check if reasoning should be included
                if not self.include_plan_reasoning and "reasoning" in valid_item:
                    valid_item["reasoning"] = "[reasoning excluded]"

                # Ensure all required fields are present
                if "explanation_name" not in valid_item or not valid_item.get("explanation_name"):
                    valid_item["explanation_name"] = "ModelPredictionConfidence"
                    logger.warning(f"Added missing explanation_name to next_response item {i}")

                if "step_name" not in valid_item or not valid_item.get("step_name"):
                    valid_item["step_name"] = "Concept"
                    logger.warning(f"Added missing step_name to next_response item {i}")

                if "communication_goals" not in valid_item or not valid_item.get("communication_goals"):
                    valid_item["communication_goals"] = [{"goal": "Provide a response to the user"}]
                    logger.warning(f"Added missing communication_goals to next_response item {i}")

                valid_next_responses.append(valid_item)

            # Replace with valid items
            function_args["next_response"] = valid_next_responses

            # If we ended up with an empty list, add a default item
            if not function_args["next_response"]:
                function_args["next_response"] = [{
                    "reasoning": "[auto-generated reasoning]",
                    "explanation_name": "ModelPredictionConfidence",
                    "step_name": "Concept",
                    "communication_goals": [{"goal": "Provide a response to the user"}]
                }]
        else:
            # If next_response is not a list, create a default one
            function_args["next_response"] = [{
                "reasoning": "[auto-generated reasoning]",
                "explanation_name": "ModelPredictionConfidence",
                "step_name": "Concept",
                "communication_goals": [{"goal": "Provide a response to the user"}]
            }]

        # Ensure next_response has at least one fallback item when empty
        if not function_args["next_response"]:
            function_args["next_response"] = [{
                "reasoning": "[auto-generated reasoning]",
                "explanation_name": "FallbackExplanation",
                "step_name": "FallbackStep",
                "communication_goals": [{"goal": "Provide a response to the user"}]
            }]

        return function_args

    async def plan_execute(self, user_message):
        """Combined Plan and Execute steps: Plan the explanation strategy and execute it in a single API call"""

        # Get last explanation if available
        last_exp = self.last_shown_explanations[-1] if len(self.last_shown_explanations) > 0 else None

        # Create the combined plan-execute prompt
        plan_execute_prompt = get_plan_execute_prompt_template().format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.chat_history,
            user_model=self.user_model.get_state_summary(as_dict=False),
            user_message=user_message,
            explanation_collection=self.user_model.get_complete_explanation_collection(as_dict=False),
        )

        start_time = datetime.now()

        # Use OpenAI API directly
        messages = [
            {"role": "system", "content": "You are an AI agent that plans and executes explanation strategies. Generate a plan and execute it in one step."},
            {"role": "user", "content": plan_execute_prompt}
        ]

        # Define the function calling format for structured output
        functions = [
            {
                "name": "plan_execute_result",
                "description": "Return the combined planning and execution results",
                "parameters": PlanExecuteResultModel.schema()
            },
            {
                "name": "apply_plan_execute_results",
                "description": "Apply planning and execution results to agent state",
                "parameters": PlanExecuteResultModel.schema()
            }
        ]

        try:
            response = await aclient.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=messages,
                functions=functions,
                function_call="auto",
                stream=True
            )

            # Stream and accumulate function_call arguments for faster first-byte time
            func_args_json = ""
            async for chunk in response:
                cd = chunk.choices[0].delta
                fc = getattr(cd, "function_call", None)
                if fc and getattr(fc, "arguments", None):
                    func_args_json += fc.arguments
            # Parse directly via Pydantic's fast JSON parser
            combined_result = self._plan_adapter.validate_json(func_args_json)
            # Invoke side-effect tool
            self._apply_plan_execute_results(combined_result)

            end_time = datetime.now()
            logger.info(f"Time taken for Combined Plan-Execute: {end_time - start_time}")

            # Conditionally log the result based on reasoning inclusion flags
            if self.include_plan_reasoning and self.include_execute_reasoning:
                logger.info(f"Combined Plan-Execute result: {combined_result}.\n")
            else:
                # Create a modified version without reasoning for logging
                log_result = combined_result.dict()
                if not self.include_plan_reasoning:
                    log_result["planning_reasoning"] = "[planning reasoning excluded]"
                    for item in log_result.get("next_response", []):
                        if isinstance(item, dict) and "reasoning" in item:
                            item["reasoning"] = "[reasoning excluded]"
                if not self.include_execute_reasoning:
                    log_result["execution_reasoning"] = "[execution reasoning excluded]"
                logger.info(f"Combined Plan-Execute result (filtered reasoning): {log_result}.\n")

            # Create separate results for logging purposes
            try:
                # Convert all Pydantic models to dictionaries to ensure JSON serializability
                plan_result = {
                    "reasoning": combined_result.planning_reasoning if self.include_plan_reasoning else "[reasoning excluded]",
                    "new_explanations": [exp.dict() if hasattr(exp, 'dict') else exp for exp in combined_result.new_explanations],
                    "explanation_plan": [exp.dict() if hasattr(exp, 'dict') else exp for exp in combined_result.explanation_plan],
                    "next_response": [resp.dict() if hasattr(resp, 'dict') else resp for resp in combined_result.next_response]
                }
            except Exception as e:
                logger.error(f"Error converting plan result to dict: {str(e)}")
                plan_result = {
                    "reasoning": "[reasoning excluded]",
                    "new_explanations": [],
                    "explanation_plan": [],
                    "next_response": []
                }

            execute_result = ExecuteResult(
                reasoning=combined_result.execution_reasoning if self.include_execute_reasoning else "[reasoning excluded]",
                response=combined_result.response,
                success=combined_result.success
            )

            # Update the log cells - use json.dumps for the plan result
            try:
                import json
                self.current_log_row["plan"] = json.dumps(plan_result)
            except TypeError as e:
                logger.error(f"JSON serialization error for plan_result: {str(e)}")
                self.current_log_row["plan"] = json.dumps({"error": "Failed to serialize plan result"})

            self.current_log_row["execute"] = execute_result.json()
            update_last_log_row(self.current_log_row, self.log_file)

            # Add agent's response to chat history
            self.append_to_history("agent", execute_result.response)

            # Check if result has placeholders and replace them
            execute_result.response = replace_plot_placeholders(execute_result.response, self.visual_explanations_dict)

            # Update Explanandum state and User Model
            try:
                # Update user model based on next explanations
                for next_explanation in combined_result.next_response:
                    if isinstance(next_explanation, str):
                        logger.error(f"next_explanation is a string, not an object: {next_explanation}")
                        continue

                    explanation_target = next_explanation
                    exp = explanation_target.explanation_name
                    exp_step = explanation_target.step_name
                    self.user_model.update_explanation_step_state(exp, exp_step, ExplanationState.UNDERSTOOD.value)

                # Add next explanations to last shown explanations
                for next_explanation in combined_result.next_response:
                    if not isinstance(next_explanation, str):
                        self.last_shown_explanations.append(next_explanation)
            except Exception as e:
                logger.error(f"Error updating user model: {str(e)}")

            # Use json.dumps for user_model since it's not a Pydantic object
            import json
            self.current_log_row["user_model"] = json.dumps(self.user_model.get_state_summary(as_dict=True))
            update_last_log_row(self.current_log_row, self.log_file)
            self.user_model.new_datapoint()

            # Log new user model
            logger.info(f"User model after combined plan-execute: {self.user_model.get_state_summary(as_dict=False)}.\n")

            return plan_result, execute_result

        except Exception as e:
            logger.error(f"Error in combined plan-execute step: {str(e)}")
            # Return minimal valid results as fallback
            fallback_plan = {
                "reasoning": "Error occurred during planning. Using fallback." if self.include_plan_reasoning else "[reasoning excluded]",
                "new_explanations": [],
                "explanation_plan": [],
                "next_response": [{
                    "reasoning": "Using fallback response due to error." if self.include_plan_reasoning else "[reasoning excluded]",
                    "explanation_name": "ErrorHandling",
                    "step_name": "Fallback",
                    "communication_goals": [{"goal": "Provide a general response to the user."}]
                }]
            }

            fallback_execute = ExecuteResult(
                reasoning="Error occurred during execution. Using fallback." if self.include_execute_reasoning else "[reasoning excluded]",
                response="I'm sorry, I wasn't able to process your question properly. Could you please rephrase or try asking something else?",
                success=False
            )

            import json
            self.current_log_row["plan"] = json.dumps(fallback_plan)
            self.current_log_row["execute"] = fallback_execute.json()
            update_last_log_row(self.current_log_row, self.log_file)
            self.append_to_history("agent", fallback_execute.response)

            return fallback_plan, fallback_execute

    # Updated main method to use the combined monitor-analyze and plan-execute steps
    async def answer_user_question(self, user_question):
        """Run the fully optimized MAPE-K workflow to answer a user question, combining all steps into just 2 API calls"""
        start_time = datetime.now()

        # Append user message to chat history at the beginning of the workflow
        self.append_to_history("user", user_question)

        # Step 1: Combined Monitor and Analyze in a single API call
        monitor_result, analyze_result = await self.monitor_analyze(user_question)

        # Step 2: Combined Plan and Execute in a single API call
        plan_result, execute_result = await self.plan_execute(user_question)

        end_time = datetime.now()
        logger.info(f"Time taken for fully optimized MAPE-K Loop with just 2 API calls: {end_time - start_time}")
        
        analysis = execute_result.reasoning
        response = execute_result.response
        
        return analysis, response