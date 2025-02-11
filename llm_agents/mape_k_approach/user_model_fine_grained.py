# Configure logger
import logging
from collections import defaultdict
from enum import Enum
from typing import List, Tuple, Optional, Dict, Union

from pydantic import PrivateAttr

from llm_agents.explanation_state import ExplanationState
from llm_agents.mape_k_approach.plan_component.advanced_plan_prompt import ExplanationStepModel, NewExplanationModel, \
    ChosenExplanationModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust to DEBUG/INFO as needed

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Adjust level if needed

# Add formatter for pretty print
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)


# Custom Classes inheriting from Pydantic Models
class ExplanationStep(ExplanationStepModel):
    _state: ExplanationState = PrivateAttr(default=ExplanationState.NOT_YET_EXPLAINED)
    _explained_content_list: PrivateAttr = PrivateAttr(default=[])

    def update_state(self,
                     new_state: ExplanationState,
                     content_piece: str = None) -> None:
        if isinstance(new_state, ExplanationState):
            self._state = new_state
            if content_piece:
                self._explained_content_list.append(content_piece)
        else:
            raise ValueError(f"Invalid state: {new_state}")

    @property
    def state(self) -> ExplanationState:
        return self._state

    @property
    def explained_content_list(self) -> List[str]:
        return self._explained_content_list

    def __repr__(self):
        return f"{self.step_name}: {self.description} | State: {self.state.value}"


class Explanation(NewExplanationModel):
    def __init__(self, **data):
        super().__init__(**data)
        self.explanation_steps = [ExplanationStep(**step.dict()) for step in self.explanation_steps]

    def add_explanation(self, step_name: str, description: str, dependencies: Optional[List[str]] = None,
                        is_optional: bool = False):
        if not any(ex.step_name == step_name for ex in self.explanation_steps):
            explanation = ExplanationStep(
                step_name=step_name,
                description=description,
                dependencies=dependencies or [],
                is_optional=is_optional
            )
            self.explanation_steps.append(explanation)
            logger.info(f"Added step '{step_name}' to explanation '{self.explanation_name}'.")
        else:
            logger.warning(f"Explanation step '{step_name}' already exists in '{self.explanation_name}'.")

    def update_state_of_all(self, new_state: ExplanationState):
        for explanation in self.explanation_steps:
            explanation.update_state(new_state)

    def update_state(self,
                     step_name: str,
                     new_state: ExplanationState,
                     content_piece: str = None) -> None:
        exp_step = self._get_explanation_step(step_name)
        if exp_step:
            exp_step.update_state(new_state, content_piece)
        else:
            logger.warning(f"Explanation step '{step_name}' not found in '{self.explanation_name}'.")

    def _get_explanation_step(self, step_name: str) -> Optional[ExplanationStep]:
        for ex in self.explanation_steps:
            if ex.step_name == step_name:
                return ex
        logger.warning(f"Explanation step '{step_name}' not found in '{self.explanation_name}'.")
        return None

    def __repr__(self):
        return f"Explanation({self.explanation_name}, Steps: {self.explanation_steps})"


class UserModelFineGrained:
    def __init__(self, user_ml_knowledge):
        self.explanations: Dict[str, Explanation] = {}
        self.cognitive_state: Optional[str] = None
        self.current_understanding_signals: List[str] = []
        self.current_explanation_request: List[str] = []  # TODO: Not used for now.
        self.user_ml_knowledge = user_ml_knowledge

    def set_cognitive_state(self, cognitive_state: str):
        """Set the cognitive state of the user model."""
        self.cognitive_state = cognitive_state

    def get_user_info(self):
        """Return the user's cognitive state and ML knowledge as a string"""
        info = f"The user is in a {self.cognitive_state} cognitive state and their ML knowledge is: {self.user_ml_knowledge}. With the last message they signalled {self.current_understanding_signals}."
        """if len(self.current_explanation_request) > 0:
            info += f" and showed the following explanation requests {self.current_explanation_request}."
        else:
            info += . """
        return info

    def add_explanation(self, explanation_name: str, description: str):
        """Add a new explanation, overriding if it already exists."""
        explanation = Explanation(
            explanation_name=explanation_name,
            description=description,
            explanation_steps=[]  # Updated field name
        )
        self.explanations[explanation_name] = explanation
        logger.info(f"Added or updated explanation '{explanation_name}'.")

    def add_explanation_step(
            self,
            explanation_name: str,
            step_name: str,
            description: str,
            dependencies: Optional[List[str]] = None,
            is_optional: bool = False
    ) -> None:
        """Add an explanation step under a specific explanation."""
        explanation = self._get_explanation(explanation_name)
        if explanation:
            explanation.add_explanation(step_name, description, dependencies, is_optional)
        else:
            logger.warning(f"Explanation '{explanation_name}' not found in the model.")

    def _get_explanation(self, explanation_name: str) -> Optional[Explanation]:
        """Retrieve an explanation by its name."""
        explanation = self.explanations.get(explanation_name)
        if not explanation:
            logger.warning(f"Explanation '{explanation_name}' not found in the model.")
        return explanation

    def _get_explanation_step(self, explanation_name: str, step_name: str) -> Optional[ExplanationStep]:
        """Retrieve a specific explanation step."""
        explanation = self._get_explanation(explanation_name)
        if explanation:
            return explanation._get_explanation_step(step_name)
        return None

    def update_explanation_step_state(self,
                                      exp_name: str,
                                      exp_step: str,
                                      new_state: Union[ExplanationState, str],
                                      content_piece: str = None) -> None:
        """Update the state of a specific explanation step."""
        # Convert string to ExplanationState if necessary
        if isinstance(new_state, str):
            try:
                new_state = ExplanationState(new_state)
            except ValueError:
                logger.warning(f"Unknown state: {new_state}")
                return

        explanation = self._get_explanation(exp_name)
        if explanation:
            explanation.update_state(exp_step, new_state, content_piece)
        else:
            logger.warning(f"Explanation '{exp_name}' not found in the model.")

    def mark_entire_exp_as_understood(self, exp_name):
        """Mark all explanation steps as 'understood'."""
        explanation = self._get_explanation(exp_name)
        if explanation:
            explanation.update_state_of_all(ExplanationState.UNDERSTOOD)

    def mark_entire_exp_as_misunderstood(self, exp_name):
        """Mark all explanation steps as 'misunderstood'."""
        explanation = self._get_explanation(exp_name)
        if explanation:
            explanation.update_state_of_all(ExplanationState.NOT_UNDERSTOOD)

    def reset_explanation_state(self, exp_name):
        """Get the explanation and reset all steps to "not explained" apart from Concept step"""
        explanation = self._get_explanation(exp_name)
        for step in explanation.explanation_steps:
            if step.step_name != "Concept":
                step.update_state(ExplanationState.NOT_YET_EXPLAINED)

    def set_model_from_summary(self, summary: Dict) -> None:
        """
        Initialize the UserModel from a summary dictionary.
        The dictionary should include keys: 'xai_explanations', each containing 'explanation_name', 'description', and 'explanation_steps'.
        """
        explanations = summary.get("xai_explanations", [])
        for exp_dict in explanations:
            explanation_name = exp_dict["explanation_name"]
            description = exp_dict["description"]
            self.add_explanation(explanation_name, description)
            for ex in exp_dict.get("explanation_steps", []):  # Updated field name
                step_name = ex["step_name"]
                description = ex["description"]
                dependencies = ex.get("dependencies", [])
                is_optional = ex["is_optional"]
                self.add_explanation_step(explanation_name, step_name, description, dependencies, is_optional)

    def add_explanations_from_plan_result(self, exp_dict_list: List[NewExplanationModel]) -> None:
        """
        Add explanations to the user model from a list of NewExplanation objects.
        """
        if not isinstance(exp_dict_list, list):
            logger.error("Expected a list of NewExplanation objects.")
            return
        if not all(isinstance(exp, NewExplanationModel) for exp in exp_dict_list):
            logger.error("All items in the list must be NewExplanation objects.")
            return

        for new_exp in exp_dict_list:
            explanation_name = new_exp.explanation_name
            description = new_exp.description
            self.add_explanation(explanation_name, description)
            for ex in new_exp.explanation_steps:  # Updated field name
                step_name = ex.step_name
                description = ex.description
                dependencies = ex.dependencies
                is_optional = ex.is_optional
                self.add_explanation_step(explanation_name, step_name, description, dependencies, is_optional)

    def get_summary(self) -> Dict[str, Dict]:
        """Return a summary of all explanations and their steps, excluding 'ScaffoldingStrategy'."""
        excluded_explanations = {"ScaffoldingStrategy"}

        summary = {
            exp_name: {
                "description": exp.description,
                "steps": {
                    step.step_name: {
                        "state": step.state.value,
                        "dependencies": step.dependencies,
                        "is_optional": step.is_optional,
                    } for step in exp.explanation_steps
                }
            }
            for exp_name, exp in self.explanations.items()
            if exp_name not in excluded_explanations
        }
        return summary

    def get_state_summary(self, as_dict: bool = False) -> Union[Dict[str, Dict[str, List[str]]], str]:
        """
        Return the user model including the cognitive state, ml knowledge, as well
        as a collection of explanation states and which explanations and steps are in there, either as a dictionary
        or a string representation suitable for LLM prompts.

        Args:
            as_dict (bool): If True, return the summary as a dictionary. Otherwise, return a string.

        Returns:
            Union[Dict[str, Dict[str, List[str]]], str]: The state summary in the desired format.
        """
        excluded_explanations = {"ScaffoldingStrategy"}
        summary = defaultdict(lambda: defaultdict(list))

        for exp_name, exp_object in self.explanations.items():
            if exp_name in excluded_explanations:
                continue
            for step in exp_object.explanation_steps:
                summary[step.state.name][exp_name].append((step.step_name, step.explained_content_list))

        if as_dict:
            # Convert defaultdict to a regular dict with regular nested dicts
            return {state: dict(exps) for state, exps in summary.items()}

        # Generate a string representation using list comprehensions and join
        # Initialize an empty list to store all the formatted states
        formatted_states = []

        # Loop through each state and its explanations in the summary
        for state, exps in summary.items():
            # Initialize a list to store the explanations for the current state
            formatted_explanations = []

            # Loop through each explanation name and its steps
            for exp_name, steps in exps.items():
                # Format the steps as "step_name (substep1, substep2, ...)"
                formatted_steps = ", ".join(
                    f"{step[0]} ({', '.join(step[1])})" for step in steps
                )
                # Create the formatted explanation line
                formatted_explanations.append(f"  - Explanation: {exp_name}, Explanation Steps: [{formatted_steps}]")

            # Combine the state with its explanations
            formatted_states.append(f"\nState: {state}\n" + "\n".join(formatted_explanations))

        # Combine all the formatted states into the final prompt string
        prompt_lines = "\n".join(formatted_states)

        # Prepend the cognitive state and ml knowledge to the prompt
        prompt_lines = f"{self.get_user_info()}\n\n" + prompt_lines

        return prompt_lines

    def get_explanation_plan(self, as_dict: bool = False) -> Union[Dict[str, Dict], str]:
        """
        Return the explanation plan for all explanations, including states,
        dependencies, and optionality, either as a dictionary or a string representation.

        Args:
            as_dict (bool): If True, return the explanation plan as a dictionary.
                            Otherwise, return a formatted string.

        Returns:
            Union[Dict[str, Dict], str]: The explanation plan in the desired format.
        """
        excluded_explanations = {"ScaffoldingStrategy"}

        # Construct the explanation_plan dictionary using comprehensions
        explanation_plan = {
            exp_name: {
                "description": exp_object.description,
                "steps": [
                    {
                        "step_name": step.step_name,
                        "description": step.description.strip(),
                        "state": step.state.name,
                        "dependencies": step.dependencies,
                        "is_optional": step.is_optional,
                    }
                    for step in exp_object.explanation_steps
                ]
            }
            for exp_name, exp_object in self.explanations.items()
            if exp_name not in excluded_explanations
        }

        if as_dict:
            return explanation_plan

        # Generate a string representation using list comprehensions
        plan_str = "\n".join([
            f"Explanation: **{exp_name}**\n"
            f"- Description: {details['description']}\n"
            f"- Steps:\n" + "\n".join([
                f"    - **{step['step_name']}**\n"
                f"      - Description: {step['description']}\n"
                f"      - Dependent on Step: {', '.join(step['dependencies']) if step['dependencies'] else 'None'}\n"
                f"      - Optional: {'Yes' if step['is_optional'] else 'No'}"
                for step in details["steps"]
            ])
            for exp_name, details in explanation_plan.items()
        ])
        return plan_str

    def get_string_explanations_from_plan(self, explanation_plan: List[ChosenExplanationModel]) -> str:
        """
        Get a list of explanations as string from the explanation plan to use in llm prompts.

        Args:
            explanation_plan (List[ChosenExplanationModel]): A list of chosen explanations.

        Returns:
            str: A string listing the explanations and steps.
        """
        explanations = [
            f"- Use the step **{choosen_exp.step}** of the explanation **{choosen_exp.explanation_name}**: " \
            f"{self._get_explanation_step(choosen_exp.explanation_name, choosen_exp.step).description}."
            for choosen_exp in explanation_plan
        ]
        return "\n".join(explanations)

    def persist_knowledge(self):
        # Reset all explanations that are not Concept explanations
        for state, explanations_dict in self.get_state_summary(as_dict=True).items():
            if state != "NOT_YET_EXPLAINED":
                for exp_name in explanations_dict:
                    self.reset_explanation_state(exp_name)

        # Print user model after reset
        # TODO: ONLY FOR DEBUGGING.
        print("PERSISTED KNOWLEDGE:")
        print(self.get_state_summary())

    def __repr__(self):
        return f"User_Model({self.explanations})"
