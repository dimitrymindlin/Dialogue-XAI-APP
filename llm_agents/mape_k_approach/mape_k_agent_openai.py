import logging, csv, json, os, datetime
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
# Import the new combined monitor-analyze model and prompt
from llm_agents.mape_k_2_components.monitor_analyze_combined import get_monitor_analyze_prompt_template, MonitorAnalyzeResultModel

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
            include_monitor_reasoning=False,
            include_analyze_reasoning=False, 
            include_plan_reasoning=False,
            include_execute_reasoning=False,
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

        # Flags to control reasoning inclusion in responses
        self.include_monitor_reasoning = include_monitor_reasoning
        self.include_analyze_reasoning = include_analyze_reasoning
        self.include_plan_reasoning = include_plan_reasoning
        self.include_execute_reasoning = include_execute_reasoning

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
        
        # Define the function calling format for structured output
        functions = [{
            "name": "plan_result",
            "description": "Return the planning results",
            "parameters": PlanResultModel.schema()
        }]
        
        try:
            # Use the aclient instance with the updated API format
            response = await aclient.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=messages,
                functions=functions,
                function_call={"name": "plan_result"}
            )
            
            # Extract the function call result
            function_call = response.choices[0].message.function_call
            function_args = json.loads(function_call.arguments)
            
            # Fix new_explanations field if it's missing or None
            if "new_explanations" not in function_args or function_args["new_explanations"] is None:
                function_args["new_explanations"] = []
                
            # Process each explanation to ensure it has required fields
            if function_args["new_explanations"]:
                fixed_explanations = []
                for exp in function_args["new_explanations"]:
                    # Skip invalid explanations
                    if not isinstance(exp, dict) or "explanation_name" not in exp or "description" not in exp:
                        logger.warning(f"Skipping invalid new explanation: {exp}")
                        continue
                    
                    # If explanation_steps is missing, create a default one
                    if "explanation_steps" not in exp or not isinstance(exp["explanation_steps"], list) or not exp["explanation_steps"]:
                        # Create a default explanation step with Concept
                        exp["explanation_steps"] = [
                            {
                                "step_name": "Concept",
                                "description": f"Basic concept of {exp['explanation_name']}",
                                "dependencies": [],
                                "is_optional": False
                            }
                        ]
                    else:
                        # Ensure each step has all required fields
                        fixed_steps = []
                        for step in exp["explanation_steps"]:
                            if not isinstance(step, dict):
                                continue
                                
                            # Ensure step has required fields
                            fixed_step = {
                                "step_name": step.get("step_name", "Concept"),
                                "description": step.get("description", f"Step for {exp['explanation_name']}"),
                                "dependencies": step.get("dependencies", []),
                                "is_optional": step.get("is_optional", False)
                            }
                            
                            # Check mandatory fields
                            if not fixed_step["step_name"] or not fixed_step["description"]:
                                continue
                                
                            fixed_steps.append(fixed_step)
                            
                        if fixed_steps:
                            exp["explanation_steps"] = fixed_steps
                        else:
                            # No valid steps were found, create a default one
                            exp["explanation_steps"] = [
                                {
                                    "step_name": "Concept",
                                    "description": f"Basic concept of {exp['explanation_name']}",
                                    "dependencies": [],
                                    "is_optional": False
                                }
                            ]
                    
                    fixed_explanations.append(exp)
                
                function_args["new_explanations"] = fixed_explanations
            
            # Check if reasoning should be included
            if not self.include_plan_reasoning:
                function_args["reasoning"] = "[reasoning excluded]"
                # Also remove reasoning from next_response items
                for item in function_args.get("next_response", []):
                    if "reasoning" in item:
                        item["reasoning"] = "[reasoning excluded]"
            
            # Convert back to JSON string
            fixed_args = json.dumps(function_args)
            
            # Parse with the Pydantic model
            plan_result = PlanResultModel.parse_raw(fixed_args)
            
            end_time = datetime.datetime.now()
            logger.info(f"Time taken for Plan: {end_time - start_time}")
            
            # Conditionally log the result based on reasoning inclusion flag
            if self.include_plan_reasoning:
                logger.info(f"Plan result: {plan_result}.\n")
            else:
                # Create a modified version without reasoning for logging
                log_result = plan_result.dict()
                log_result["reasoning"] = "[reasoning excluded]"
                for item in log_result.get("next_response", []):
                    item["reasoning"] = "[reasoning excluded]"
                logger.info(f"Plan result (filtered reasoning): {log_result}.\n")
            
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
            
        except Exception as e:
            logger.error(f"Error in plan step: {str(e)}")
            # Return a minimal valid plan result as fallback
            fallback_plan = PlanResultModel(
                reasoning="Error occurred during planning. Using fallback." if self.include_plan_reasoning else "[reasoning excluded]",
                new_explanations=[],
                explanation_plan=[],
                next_response=[{
                    "reasoning": "Using fallback response due to error." if self.include_plan_reasoning else "[reasoning excluded]",
                    "explanation_name": "ErrorHandling",
                    "step_name": "Fallback",
                    "communication_goals": [{"goal": "Provide a general response to the user."}]
                }]
            )
            self.current_log_row["plan"] = fallback_plan.json()
            update_last_log_row(self.current_log_row, self.log_file)
            return fallback_plan

    async def monitor_analyze(self, user_message):
        """Combined Monitor and Analyze steps: Process the user's cognitive state in a single API call"""
        
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

        start_time = datetime.datetime.now()
        
        # Use OpenAI API directly
        messages = [
            {"role": "system", "content": "You are an AI agent analyzing user understanding. Perform both monitor and analyze tasks."},
            {"role": "user", "content": monitor_analyze_prompt}
        ]
        
        # Define the function calling format for structured output
        functions = [{
            "name": "monitor_analyze_result",
            "description": "Return the combined monitoring and analysis results",
            "parameters": MonitorAnalyzeResultModel.schema()
        }]
        
        response = await aclient.chat.completions.create(
            model=OPENAI_MODEL_NAME,  # Use the full model for the combined call
            messages=messages,
            functions=functions,
            function_call={"name": "monitor_analyze_result"}
        )
        
        # Extract and parse the function call result
        function_call = response.choices[0].message.function_call
        combined_result = MonitorAnalyzeResultModel.parse_raw(function_call.arguments)
        
        end_time = datetime.datetime.now()
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
        
        # Update the user model based on monitor results
        if combined_result.mode_of_engagement != "":
            self.user_model.cognitive_state = self.modes_of_engagement.get_differentiating_description(
                combined_result.mode_of_engagement)

        if len(combined_result.explicit_understanding_displays) > 0:
            self.user_model.explicit_understanding_signals = combined_result.explicit_understanding_displays

        # Process and validate model changes
        validated_model_changes = []
        for change_entry in combined_result.model_changes:
            if isinstance(change_entry, dict):
                # Fix the step_name vs step issue
                if "step_name" in change_entry and not "step" in change_entry:
                    # Handle case where step_name contains a state name (invalid)
                    states = ["not_yet_explained", "understood", "not_understood", "partially_understood"]
                    if change_entry["step_name"] in states:
                        # Use "Concept" as a default step when receiving a state as step_name
                        change_entry["step"] = "Concept"
                    else:
                        # Otherwise use the provided step_name as step
                        change_entry["step"] = change_entry["step_name"]
                    del change_entry["step_name"]
                
                # Ensure all required fields exist with non-empty values
                if (("explanation_name" in change_entry and change_entry["explanation_name"]) and
                    ("state" in change_entry and change_entry["state"]) and
                    ("step" in change_entry and change_entry["step"])):
                    
                    # Skip entries where step is actually a state name
                    states = ["not_yet_explained", "understood", "not_understood", "partially_understood"]
                    if change_entry["step"] not in states:
                        validated_model_changes.append(change_entry)
                    else:
                        logger.error(f"Invalid step value (contains state name): {change_entry}")
                else:
                    logger.error(f"Missing required fields in model change: {change_entry}")
            else:
                logger.error(f"Invalid model change type: {type(change_entry)}")
        
        # Apply only valid changes to the user model
        for change_entry in validated_model_changes:
            exp = change_entry["explanation_name"]
            change = change_entry["state"]
            step = change_entry["step"]
            self.user_model.update_explanation_step_state(exp, step, change)
        
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