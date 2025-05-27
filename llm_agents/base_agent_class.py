"""
Base agent functionality shared across different agent implementations.

This module provides a common base class that contains shared functionality for
the LlamaIndex-based and OpenAI-based agent implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

from create_experiment_data.instance_datapoint import InstanceDatapoint
from llm_agents.agent_utils import (
    timed, generate_log_file_name, initialize_csv,
    log_prompt, get_definition_paths
)
from llm_agents.mape_k_approach.plan_component.xai_exp_populator import XAIExplanationPopulator
from llm_agents.mape_k_approach.user_model.user_model_fine_grained import UserModelFineGrained as UserModel
from llm_agents.utils.definition_wrapper import DefinitionWrapper


class BaseAgent(ABC):
    """
    Abstract base class for XAI agents that implements common functionality.
    
    This class handles shared functionality such as logging, history management,
    and user model initialization.
    """

    def __init__(
            self,
            experiment_id: str,
            feature_names: str = "",
            domain_description: str = "",
            user_ml_knowledge: str = "",
            **kwargs
    ):
        # Logging setup
        self.experiment_id = experiment_id
        self.log_file = generate_log_file_name(experiment_id)
        initialize_csv(self.log_file)

        # Common context
        self.feature_names = feature_names
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

    def log_prompt(self, component: str, prompt_str: str) -> None:
        """
        Log a prompt to the prompt log file.
        
        Args:
            component: The component name (e.g., 'monitor', 'analyze')
            prompt_str: The raw prompt content
        """
        log_prompt(self.prompt_log_file, component, prompt_str)

    def initialize_new_datapoint(
            self,
            instance: InstanceDatapoint,
            xai_explanations: Any,
            xai_visual_explanations: Any,
            predicted_class_name: str,
            opposite_class_name: str,
            datapoint_count: int
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
        """
        # Set instance data
        self.instance = instance.displayable_features
        self.predicted_class_name = predicted_class_name
        self.opposite_class_name = opposite_class_name
        self.datapoint_count = datapoint_count + 1
        self.reset_history()

        # Initialize explanations
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

        # Initialize explanation plan from predefined plan if available
        predefined_plan = populated.get("predefined_plan", [])
        if predefined_plan:
            # Convert predefined plan to explanation plan format
            from llm_agents.mape_k_approach.plan_component.advanced_plan_prompt_multi_step import ChosenExplanationModel
            self.explanation_plan = []
            for plan_item in predefined_plan:
                for child in plan_item.get("children", []):
                    step_name = child.get("step_name") or child.get("title", "")
                    self.explanation_plan.append(ChosenExplanationModel(
                        explanation_name=plan_item["title"],
                        step=step_name
                    ))
        else:
            # Fallback to empty plan
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
            "instance": self.instance,
            "predicted_class_name": self.predicted_class_name,
            "chat_history": self.chat_history,
            "user_model_state": self.user_model.get_state_summary(as_dict=False),
            "last_shown_explanations": self.last_shown_explanations,
        }

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
