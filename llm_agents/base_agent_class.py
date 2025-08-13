"""
Base agent functionality shared across different agent implementations.

This module provides a common base class that contains shared functionality for
the LlamaIndex-based and OpenAI-based agent implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import os
import threading

from pydantic import BaseModel

from create_experiment_data.instance_datapoint import InstanceDatapoint
from llm_agents.agent_utils import (
    timed, generate_log_file_name, initialize_csv, get_definition_paths
)
from llm_agents.mape_k_approach.plan_component.xai_exp_populator import XAIExplanationPopulator
from llm_agents.mape_k_approach.user_model.user_model_fine_grained import UserModelFineGrained as UserModel
from llm_agents.models import ChosenExplanationModel
from llm_agents.utils.definition_wrapper import DefinitionWrapper


class BaseAgent(ABC):
    """
    Abstract base class for XAI agents that implements common functionality.
    
    This class handles shared functionality such as logging, history management,
    and user model initialization.
    """

    def __init__(
            self,
            feature_names: str = "",
            feature_units: str = "",
            feature_tooltips: str = "",
            domain_description: str = "",
            user_ml_knowledge: str = "",
            **kwargs
    ):
        # Logging setup
        self.experiment_id = kwargs.get("experiment_id", None)
        # Create per-execution CSV log
        self.log_file = generate_log_file_name(self.experiment_id)
        initialize_csv(self.log_file)

        # Buffer for prompt/response pairs for a single run
        self._csv_items = []
        # Lock for thread-safe file writing
        self._log_lock = threading.Lock()

        # Feature context - consolidate all feature-related information
        self._initialize_feature_context(feature_names, feature_units, feature_tooltips)

        # Common context
        self.domain_description = domain_description
        self.user_ml_knowledge = user_ml_knowledge
        self.predicted_class_name = None
        self.opposite_class_name = None
        self.instance = None
        self.datapoint_count = None

        # Load definition wrappers
        definition_paths = get_definition_paths()
        self.understanding_displays = DefinitionWrapper(definition_paths["understanding_displays"])
        self.modes_of_engagement = DefinitionWrapper(definition_paths["modes_of_engagement"])
        self.explanation_questions = DefinitionWrapper(definition_paths["explanation_questions"])

        # User model and history
        self.user_model = UserModel(user_ml_knowledge)
        self.populator = None
        self.chat_history = self.reset_history()
        self.explanation_plan = []
        self.last_shown_explanations = []
        self.visual_explanations_dict = None

    def _initialize_feature_context(self, feature_names: str, feature_units: str, feature_tooltips: str) -> None:
        """
        Initialize feature context information in a centralized way.

        Args:
            feature_names: Comma-separated feature names
            feature_units: Comma-separated feature units
            feature_tooltips: Comma-separated feature tooltips
        """
        self.feature_names = feature_names
        self.feature_units = feature_units
        self.feature_tooltips = feature_tooltips
        self.feature_context = self._create_feature_context()

    def _create_feature_context(self) -> Dict[str, Any]:
        """
        Create a context dictionary with feature-related information.

        Returns:
            A dictionary containing feature names, units, and tooltips
        """
        return {
            "feature_names": self.feature_names,
            "feature_units": self.feature_units,
            "feature_tooltips": self.feature_tooltips
        }

    def get_feature_context(self) -> Dict[str, Any]:
        """
        Get the feature context dictionary.

        Returns:
            The feature context dictionary
        """
        return self.feature_context

    def clean_html_content(self, content: str) -> str:
        """
        Clean HTML content by replacing <img> tags with [PLOT] placeholder,
        removing only bold/italic tags (<b>, <strong>, <i>, <em>), and decoding HTML entities.
        All other HTML/XML tags are preserved.

        Args:
            content: String that may contain HTML content and entities

        Returns:
            Cleaned string with only the specified tags removed and entities decoded
        """
        import re
        import html

        if not content or not isinstance(content, str):
            return content

        # First decode HTML entities (&lt; &gt; &amp; etc.)
        content = html.unescape(content)

        # Replace <img> tags with [PLOT] placeholder
        content = re.sub(r'<img[^>]*>', '[PLOT]', content, flags=re.IGNORECASE)

        # Remove only bold/italic tags (<b>, <strong>, <i>, <em>) but keep their inner text
        content = re.sub(r'</?(?:b|strong|i|em)\b[^>]*>', '', content, flags=re.IGNORECASE)
        
        return content

    def log_component_input_output(self, component_name: str, input: str, output: Union[str, BaseModel]) -> None:
        """
        Log a prompt to the buffer for CSV.
        """
        # Convert input to string if it's not already (handles PromptTemplate objects)
        if not isinstance(input, str):
            # Handle PromptTemplate objects properly
            if hasattr(input, 'get_template'):
                # Use get_template() method for PromptTemplate objects
                input = input.get_template()
            elif hasattr(input, 'template'):
                # Fallback to direct template attribute
                input = input.template
            else:
                # Generic fallback for other objects
                input = str(input)


        
        # Turn output to string if pydantic model
        if isinstance(output, BaseModel):
            try:
                # Use model_dump_json() for Pydantic v2 compatibility
                if hasattr(output, 'model_dump_json'):
                    output = output.model_dump_json()
                else:
                    # Fallback for Pydantic v1
                    output = output.json()
            except Exception as e:
                # If JSON conversion fails, fall back to string representation
                output = str(output)
        elif not isinstance(output, str):
            output = str(output)
        
        # Clean HTML content from both input and output
        input = self.clean_html_content(input)
        output = self.clean_html_content(output)
        
        # Buffer the prompt for final CSV
        self._csv_items.append({"component_name": component_name, "input": input, "output": output})

    def initialize_new_datapoint(
            self,
            instance: InstanceDatapoint,
            xai_explanations: Any,
            xai_visual_explanations: Any,
            predicted_class_name: str,
            opposite_class_name: str,
            datapoint_count: int,
            use_precomputed_plan: bool = True
    ) -> None:
        """
        Initialize agent with a new datapoint and its explanations.

        Args:
            instance: The instance data
            xai_explanations: Explanation data from XAI methods
            xai_visual_explanations: Visual explanation data
            predicted_class_name: Name of the predicted class
            opposite_class_name: Name of the alternative class
            datapoint_count: The current datapoint count
            use_precomputed_plan: If True, load predefined plan; if False, start with empty plan
        """
        # Set instance data
        self.instance = instance.displayable_features
        self.predicted_class_name = predicted_class_name
        self.opposite_class_name = opposite_class_name
        self.datapoint_count = datapoint_count + 1
        self.reset_history()

        # Retrieve understood concepts from the previous user model, if it exists
        understood_concepts = []
        if hasattr(self, 'user_model') and self.user_model is not None:
            understood_concepts = self.user_model.persist_knowledge()

        # Initialize explanations
        self.populator = XAIExplanationPopulator(
            template_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
            template_file="llm_agents/mape_k_approach/plan_component/explanations_model.yaml",
            xai_explanations=xai_explanations,
            predicted_class_name=predicted_class_name,
            opposite_class_name=opposite_class_name,
            instance_dict=self.instance
        )
        self.populator.populate_yaml()
        self.populator.validate_substitutions()
        populated = self.populator.get_populated_json(as_dict=True, include_predefined_plan=use_precomputed_plan)

        # Pass understood concepts to the new user model
        self.user_model = UserModel(self.user_ml_knowledge, initial_understood_concepts=understood_concepts)
        self.user_model.set_model_from_summary(populated)

        print("User model initialized with understood concepts:", understood_concepts)

        # Initialize explanation plan based on use_precomputed_plan flag
        if use_precomputed_plan:
            # Load predefined plan if available and requested
            predefined_plan = populated.get("predefined_plan", [])
            if predefined_plan:
                # Convert predefined plan to explanation plan format
                self.explanation_plan = []
                for plan_item in predefined_plan:
                    for child in plan_item.get("children", []):
                        step_name = child.get("step_name") or child.get("title", "")
                        self.explanation_plan.append(ChosenExplanationModel(
                            explanation_name=plan_item["title"],
                            step_name=step_name
                        ))
            else:
                # No predefined plan available
                self.explanation_plan = []
        else:
            # Explicitly start with empty plan for adaptive agents
            self.explanation_plan = []

        # Set visual explanations
        self.visual_explanations_dict = xai_visual_explanations
        self.last_shown_explanations = []

    def reset_history(self) -> str:
        """
        Reset the chat history.

        Returns:
            The initial chat history string
        """
        self.chat_history = "No history available, beginning of the chat."
        self.last_shown_explanations = ["No explanations shown yet, beginning of the chat."]
        return self.chat_history

    def append_to_history(self, role: str, msg: str) -> None:
        """
        Append a message to the chat history.

        Args:
            role: The role of the message sender ('user' or 'agent')
            msg: The message content
        """
        prefix = "User: " if role == "user" else "Agent: "
        entry = f"{prefix}{msg}\n"

        if self.chat_history.startswith("No history available"):
            self.chat_history = entry
        else:
            self.chat_history += entry

    def complete_explanation_step(self, explanation_name: str, step_name: str) -> None:
        """
        Mark an explanation step as complete.

        Args:
            explanation_name: The name of the explanation
            step_name: The name of the step within the explanation
        """
        for exp in self.explanation_plan:
            if exp.explanation_name == explanation_name and exp.step_name == step_name:
                self.explanation_plan.remove(exp)
                break

    def get_common_context(self) -> Dict[str, Any]:
        """
        Get context information common to all agent components.

        Returns:
            A dictionary with shared context information
        """
        return {
            "domain_description": self.domain_description,
            "feature_names": self.feature_names,
            "feature_units": self.feature_units,
            "feature_tooltips": self.feature_tooltips,
            "instance": self.instance,
            "predicted_class_name": self.predicted_class_name,
            "chat_history": self.chat_history,
            "user_model_state": self.user_model.get_state_summary(as_dict=False),
            "last_shown_explanations": self.last_shown_explanations,
        }

    def get_formatted_feature_context(self) -> str:
        """
        Get an XML-formatted string representation of the feature context for prompts.

        Returns:
            A concatenation of <feature ...> elements (no outer <features> wrapper).
            Descriptions (from tooltips) are preserved as provided (only whitespace-normalized),
            without truncation or ellipses. Units, when present, are kept as the unit attribute.
        """
        import re

        def _normalize(text: str) -> str:
            """Collapse whitespace; do not truncate."""
            if not text:
                return ""
            return re.sub(r"\s+", " ", str(text)).strip()

        def _xml_escape(s: str) -> str:
            if s is None:
                return ""
            # Minimal XML escaping for attribute/text safety
            return (
                str(s)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;")
            )

        if not self.feature_names:
            # No outer <features> wrapper; return empty string if no features
            return ""

        # ------- Names -------
        if isinstance(self.feature_names, list):
            if self.feature_names and isinstance(self.feature_names[0], dict):
                names = [item.get("feature_name", "").strip() for item in self.feature_names]
            else:
                names = [str(name).strip() for name in self.feature_names if str(name).strip()]
        else:
            names = [n.strip() for n in str(self.feature_names).split(",") if n.strip()]

        # De-duplicate while preserving order
        names = list(dict.fromkeys(names))

        # ------- Units -------
        units_dict = {}
        if isinstance(self.feature_units, dict):
            units_dict = {str(k): (v or "") for k, v in self.feature_units.items()}
        elif isinstance(self.feature_units, list):
            for i, unit in enumerate(self.feature_units):
                if i < len(names):
                    units_dict[names[i]] = (unit or "").strip()
        elif self.feature_units:
            unit_list = [u.strip() for u in str(self.feature_units).split(",")]
            for i, unit in enumerate(unit_list):
                if i < len(names):
                    units_dict[names[i]] = unit

        # ------- Tooltips / Descriptions -------
        tooltips_dict = {}
        if isinstance(self.feature_tooltips, dict):
            tooltips_dict = {str(k): (v or "") for k, v in self.feature_tooltips.items()}
        elif isinstance(self.feature_tooltips, list):
            for i, tooltip in enumerate(self.feature_tooltips):
                if i < len(names):
                    tooltips_dict[names[i]] = (tooltip or "").strip()
        elif self.feature_tooltips:
            tooltip_list = [t.strip() for t in str(self.feature_tooltips).split(",")]
            for i, tooltip in enumerate(tooltip_list):
                if i < len(names):
                    tooltips_dict[names[i]] = tooltip

        # ------- Build XML (no outer <features>) -------
        lines = []
        for name in names:
            raw_unit = units_dict.get(name, "")
            raw_tip = tooltips_dict.get(name, "")

            unit_attr = f' unit="{_xml_escape(raw_unit)}"' if raw_unit else ""
            desc = _normalize(raw_tip)

            if desc:
                # Keep child <description> for compatibility, with whitespace-normalized text only
                line = (
                    f"<feature name=\"{_xml_escape(name)}\"{unit_attr}>\n"
                    f"    <description>{_xml_escape(desc)}</description>\n"
                    f"</feature>"
                )
            else:
                # Self-closing when no description is available
                line = f"<feature name=\"{_xml_escape(name)}\"{unit_attr} />"

            lines.append(line)

        return "\n".join(lines)

    @timed
    @abstractmethod
    async def answer_user_question(self, user_question: str):
        """
        Process a user question through the MAPE-K workflow and return an answer.

        Args:
            user_question: The user's question text

        Returns:
            Implementation-specific result object
        """
        pass

    def finalize_log(self):
        """Write one CSV row per run with timestamp, experiment ID, and all prompts/responses."""
        import csv, json, datetime
        # Acquire lock before writing to file
        with self._log_lock:
            # Prepare a JSON array of buffered items
            items = json.dumps(self._csv_items)
            timestamp = datetime.datetime.utcnow().isoformat()
            # Append to the CSV file
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([timestamp, self.experiment_id, items])
            # Clear buffer for next run
            self._csv_items.clear()
