# Configure logger
import json
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import List, Optional, Dict, Union

from llm_agents.explanation_state import ExplanationState
from llm_agents.models import ExplanationStepModel, NewExplanationModel, ChosenExplanationModel
from llm_agents.utils.definition_wrapper import DefinitionWrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust to DEBUG/INFO as needed

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Adjust level if needed

# Add formatter for pretty print
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)


class Explanation(NewExplanationModel):
    def __init__(self, **data):
        super().__init__(**data)
        self.explanation_steps = [ExplanationStepModel(**step.dict()) for step in self.explanation_steps]

    def add_explanation(self, step_name: str, description: str):
        if not any(ex.step_name == step_name for ex in self.explanation_steps):
            explanation = ExplanationStepModel(
                step_name=step_name,
                description=description
            )
            self.explanation_steps.append(explanation)
            logger.info(f"Added step '{step_name}' to explanation '{self.explanation_name}'.")
        else:
            logger.warning(f"Explanation step '{step_name}' already exists in '{self.explanation_name}'.")

    def update_state_of_all(self, new_state: ExplanationState):
        for explanation in self.explanation_steps:
            explanation.state = new_state

    def update_state(self,
                     step_name: str,
                     new_state: ExplanationState) -> None:
        exp_step = self._get_explanation_step(step_name)
        if exp_step:
            exp_step.state = new_state
        else:
            logger.warning(f"Explanation step '{step_name}' not found in '{self.explanation_name}'.")

    def _get_explanation_step(self, step_name: str) -> Optional[ExplanationStepModel]:
        for ex in self.explanation_steps:
            if ex.step_name == step_name:
                return ex
        logger.warning(f"Explanation step '{step_name}' not found in '{self.explanation_name}'.")
        return None

    def __repr__(self):
        return f"Explanation({self.explanation_name}, Steps: {self.explanation_steps})"


class UserModelFineGrained:
    def __init__(self, user_ml_knowledge,
                 initial_understood_concepts: Optional[List[str]] = None):
        self.explanations: Dict[str, Explanation] = {}
        self.cognitive_state: Optional[str] = ""
        self.explicit_understanding_signals: List[str] = []
        self.current_explanation_request: List[str] = []  # TODO: Not used for now.
        self.user_ml_knowledge = self.set_user_ml_knowledge(user_ml_knowledge)
        self.initial_understood_concepts = initial_understood_concepts

    def set_cognitive_state(self, cognitive_state: str):
        """Set the cognitive state of the user model."""
        self.cognitive_state = cognitive_state

    def set_explicit_understanding_signals(self, explicit_understanding_signals: List[str]):
        """Set the explicit understanding signals of the user model."""
        self.explicit_understanding_signals = explicit_understanding_signals

    def set_user_ml_knowledge(self, user_ml_knowledge: str):
        """Set the user's ML knowledge"""
        if user_ml_knowledge == "anonymous":
            return None
        # 1 Load Ml knowledge concepts from json file
        file_path = "llm_agents/mape_k_approach/user_model/user_ml_knowledge_definitions.json"
        definitions = DefinitionWrapper(file_path)
        # Get the differentiating_description for the user's ML knowledge
        definition = definitions.get_differentiating_description(user_ml_knowledge)
        return definition

    def get_user_info(self, as_dict=False):
        """Return the user's cognitive state and ML knowledge as a string."""
        if as_dict:
            return {
                "Cognitive State": self.cognitive_state,
                "ML Knowledge": self.user_ml_knowledge,
                "Explicit Understanding Signals": self.explicit_understanding_signals,
            }
        parts = []
        if self.cognitive_state != "":
            parts.append(f"The user's cognitive state is {self.cognitive_state}.")
        if self.user_ml_knowledge is not None:
            parts.append(f"The user's ML knowledge is {self.user_ml_knowledge}.")
        if self.explicit_understanding_signals != []:
            parts.append(f"With the last message they signalled {self.explicit_understanding_signals}.")
        return "\n".join(parts)

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
            description: str
    ) -> None:
        """Add an explanation step under a specific explanation."""
        explanation = self._get_explanation(explanation_name)
        if explanation:
            explanation.add_explanation(step_name, description)
        else:
            logger.warning(f"Explanation '{explanation_name}' not found in the model.")

    def _get_explanation(self, explanation_name: str) -> Optional[Explanation]:
        """Retrieve an explanation by its name."""
        explanation = self.explanations.get(explanation_name)
        if not explanation:
            logger.warning(f"Explanation '{explanation_name}' not found in the model.")
        return explanation

    def _get_explanation_step(self, explanation_name: str, step_name: str) -> Optional[ExplanationStepModel]:
        """Retrieve a specific explanation step."""
        explanation = self._get_explanation(explanation_name)
        if explanation:
            return explanation._get_explanation_step(step_name)
        return None

    def update_explanation_step_state(self,
                                      exp_name: str,
                                      exp_step: str,
                                      new_state: Union[ExplanationState, str]) -> None:
        """Update the state of a specific explanation step."""
        logger.info(f"🔄 UPDATE_EXPLANATION_STEP_STATE: {exp_name}.{exp_step} -> {new_state}")
        
        # Convert string to ExplanationState if necessary
        if isinstance(new_state, str):
            try:
                new_state = ExplanationState(new_state)
            except ValueError:
                logger.warning(f"Unknown state: {new_state}")
                return

        explanation = self._get_explanation(exp_name)
        if explanation:
            logger.info(f"  - Found explanation '{exp_name}' with {len(explanation.explanation_steps)} steps")
            step_names = [s.step_name for s in explanation.explanation_steps]
            logger.info(f"  - Available step names: {step_names}")
            explanation.update_state(exp_step, new_state)
        else:
            logger.warning(f"Explanation '{exp_name}' not found in the model.")

    def reset_explanation_state(self, exp_name):
        """Get the explanation and reset all steps to "not explained" apart from Concept step"""
        explanation = self._get_explanation(exp_name)
        for step in explanation.explanation_steps:
            if step.step_name != "Concept":
                step.state = ExplanationState.NOT_YET_EXPLAINED

    def set_model_from_summary(self, summary: Dict) -> None:
        """
        Initialize the UserModel from a JSON summary.
        The summary is expected to be a dictionary with the key 'xai_explanations'
        containing a list of explanation dictionaries, each including 'explanation_name',
        'description', and 'steps'.
        """
        # Debug logging to understand what data we received
        logger.info(f"🏗️ SET_MODEL_FROM_SUMMARY DEBUG:")
        logger.info(f"  - Summary keys: {list(summary.keys())}")
        logger.info(f"  - Has predefined_plan: {'predefined_plan' in summary}")
        logger.info(f"  - predefined_plan value: {summary.get('predefined_plan', 'NOT_FOUND')}")
        
        if 'xai_explanations' in summary:
            logger.info(f"  - Number of xai_explanations: {len(summary['xai_explanations'])}")
            # Log first explanation to see its structure
            if summary['xai_explanations']:
                first_exp = summary['xai_explanations'][0]
                logger.info(f"  - First explanation keys: {list(first_exp.keys())}")
                logger.info(f"  - First explanation explanation_steps: {first_exp.get('explanation_steps', 'NOT_FOUND')}")
                logger.info(f"  - First explanation children: {first_exp.get('children', 'NOT_FOUND')}")
        
        # If a predefined plan exists, load only the planned explanations and their steps
        plan = summary.get("predefined_plan")
        logger.info(f"  - Taking path: {'PREDEFINED_PLAN' if plan else 'ALL_EXPLANATIONS'}")
        
        if plan:
            # Clear any existing explanations
            self.explanations.clear()
            for item in plan:
                exp_name = item["title"]
                # Add explanation with placeholder description or retrieve from summary map
                desc = next(
                    (e["description"] for e in summary.get("xai_explanations", []) if
                     e["explanation_name"] == exp_name),
                    ""
                )
                self.add_explanation(exp_name, desc)
                # Add only the two planned steps
                for step in item.get("children", []):
                    step_name = step.get("step_name") or step.get("title")
                    step_desc = step.get("description", "")
                    self.add_explanation_step(exp_name, step_name, step_desc)
            return
        else:
            understood_exp_concepts = self.get_understood_concepts()
            xai_summary = summary.get("xai_explanations", [])
            for exp_data in xai_summary:
                explanation_name = exp_data["explanation_name"]
                description = exp_data["description"]
                self.add_explanation(explanation_name, description)
                # Handle both field names - "children" (from get_populated_json) or "explanation_steps" (direct YAML)
                steps_data = exp_data.get("children", exp_data.get("explanation_steps", []))
                logger.info(f"  - Loading {len(steps_data)} steps for {explanation_name}")
                for step in steps_data:
                    step_name = step["step_name"]
                    step_description = step["description"]
                    self.add_explanation_step(explanation_name, step_name, step_description)
            for exp_name in understood_exp_concepts:
                self.update_explanation_step_state(exp_name, "Concept", ExplanationState.UNDERSTOOD)

            # Apply initial understood concepts if provided
            if self.initial_understood_concepts:
                for exp_name in self.initial_understood_concepts:
                    self.update_explanation_step_state(exp_name, "Concept", ExplanationState.UNDERSTOOD)

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
                self.add_explanation_step(explanation_name, step_name, description)

    def get_state_summary(self, as_dict: bool = False) -> Union[Dict[str, Dict[str, List[str]]], str]:
        """
        Return the user model including the cognitive state, ml knowledge, as well
        as a collection of explanation states and which explanations and steps are in there, either as a dictionary
        or an XML-formatted string representation suitable for LLM prompts.

        Args:
            as_dict (bool): If True, return the summary as a dictionary. Otherwise, return an XML-formatted string.

        Returns:
            Union[Dict[str, Dict[str, List[str]]], str]: The state summary as dictionary or XML string.
        """
        excluded_explanations = {"ScaffoldingStrategy"}
        summary = defaultdict(lambda: defaultdict(list))

        for exp_name, exp_object in self.explanations.items():
            if exp_name in excluded_explanations:
                continue
            for step in exp_object.explanation_steps:
                summary[step.state.name][exp_name].append(step.step_name)

        if as_dict:
            # Convert defaultdict to a regular dict with regular nested dicts
            state_dict = {state: dict(exps) for state, exps in summary.items()}
            state_dict["User Info"] = self.get_user_info(as_dict=True)
            return state_dict

        # Generate a string representation using list comprehensions and join
        # Initialize an empty list to store all the formatted states
        formatted_states = []

        # Loop through each state and its explanations in the summary
        for state, exps in summary.items():
            # Initialize a list to store the explanations for the current state
            formatted_explanations = []

            # Loop through each explanation name and its steps
            for exp_name, steps in exps.items():
                # Format the steps as a simple comma-separated list
                formatted_steps = ", ".join(steps)
                # Create the formatted explanation line
                formatted_explanations.append(f"  - Explanation: {exp_name}, Explanation Steps: [{formatted_steps}]")

            # Combine the state with its explanations
            formatted_states.append(f"\nState: {state}\n" + "\n".join(formatted_explanations))

        # Combine all the formatted states into the final prompt string
        prompt_lines = "\n".join(formatted_states)

        # Prepend the cognitive state and ml knowledge to the prompt
        prompt_lines = f"{self.get_user_info()}\n The user's understanding about the explanations is detailed here with the keys what the user UNDERSTOOD, NOT_UNDERSTOOD, or NOT_YET_EXPLAINED" + prompt_lines

        return self._convert_state_summary_to_xml(summary)

    def _convert_state_summary_to_xml(self, summary: Dict) -> str:
        """
        Convert the state summary to XML format with better naming conventions.
        
        Args:
            summary: Dictionary containing explanation states and their associated explanations
            
        Returns:
            XML formatted string representing the user model state
        """
        # Map state names to more descriptive XML tag names
        state_name_mapping = {
            "UNDERSTOOD": "understood_explanations",
            "NOT_UNDERSTOOD": "confused_explanations",
            "NOT_YET_EXPLAINED": "not_yet_explained_explanations",
        }

        xml_parts = ["<user_model_state>"]

        # Add user information section
        user_info = self.get_user_info(as_dict=True)
        xml_parts.append("    <user_profile>")

        if user_info.get("Cognitive State"):
            xml_parts.append(
                f'        <cognitive_state>The users cognitive state is {user_info["Cognitive State"]}</cognitive_state>')

        if user_info.get("ML Knowledge"):
            xml_parts.append(
                f'        <ml_knowledge_level>The users machine learning knowledge is {user_info["ML Knowledge"]}</ml_knowledge_level>')

        if user_info.get("Explicit Understanding Signals"):
            signals = user_info["Explicit Understanding Signals"]
            if signals:
                xml_parts.append("        <recent_understanding_signals>")
                for signal in signals:
                    xml_parts.append(f"            <signal>{signal}</signal>")
                xml_parts.append("        </recent_understanding_signals>")

        xml_parts.append("    </user_profile>")

        # Add explanation states section
        xml_parts.append("    <explanation_states>")

        for state, explanations_dict in summary.items():
            # Use mapped name or fallback to original state name
            xml_tag = state_name_mapping.get(state, state.lower().replace('_', '_'))

            if explanations_dict:  # Only add section if there are explanations
                xml_parts.append(f"        <{xml_tag}>")

                for exp_name, steps in explanations_dict.items():
                    xml_parts.append(f'            <explanation name="{exp_name}">{"  |  ".join(steps)}</explanation>')

                xml_parts.append(f"        </{xml_tag}>")

        # Add sections for states with no explanations but to give context of all states
        for original_state, xml_tag in state_name_mapping.items():
            if original_state not in summary:
                # Add empty tag for states that have no explanations
                xml_parts.append(f"        <{xml_tag} />")
            elif not summary[original_state]:
                # Add empty tag for states that exist but have no explanations
                xml_parts.append(f"        <{xml_tag} />")

        xml_parts.append("    </explanation_states>")
        xml_parts.append("</user_model_state>")

        return "\n".join(xml_parts)

    def get_complete_explanation_collection(self, as_dict: bool = False, include_exp_content=True,
                                            output_format: str = "xml") -> Union[Dict[str, Dict], str]:
        """
        Return the explanation plan for all explanations in the specified format.
        
        Args:
            as_dict: If True and output_format is "json", returns a dictionary instead of JSON string
            include_exp_content: Whether to include step descriptions
            output_format: Format of the output - "json" or "xml"
        
        Returns:
            Dictionary if as_dict=True and output_format="json", otherwise formatted string
        """
        excluded_explanations = {"ScaffoldingStrategy"}

        explanation_plan = {
            exp_name: {
                "description": exp_object.description,
                "steps": [
                    {
                        "step_name": step.step_name,
                        **({"description": step.description.strip()} if include_exp_content else {}),
                    }
                    for step in exp_object.explanation_steps
                ]
            }
            for exp_name, exp_object in self.explanations.items()
            if exp_name not in excluded_explanations
        }

        if output_format.lower() == "xml":
            return self._convert_to_xml(explanation_plan)
        elif as_dict:
            return explanation_plan
        else:
            return json.dumps(explanation_plan, indent=2)

    def _convert_to_xml(self, explanation_plan: Dict[str, Dict]) -> str:
        """
        Convert explanation plan dictionary to XML format.
        
        Args:
            explanation_plan: Dictionary containing explanation data
            
        Returns:
            XML formatted string
        """
        # Create root element
        root = ET.Element("explanations")

        for exp_name, exp_data in explanation_plan.items():
            # Create explanation element
            explanation_elem = ET.SubElement(root, "explanation")
            explanation_elem.set("name", exp_name)

            # Add description
            description_elem = ET.SubElement(explanation_elem, "description")
            description_elem.text = exp_data.get("description", "")

            # Add steps
            steps_elem = ET.SubElement(explanation_elem, "steps")
            for step in exp_data.get("steps", []):
                step_elem = ET.SubElement(steps_elem, "step")
                step_elem.set("name", step.get("step_name", ""))

                # Add step description if present
                if "description" in step:
                    step_description_elem = ET.SubElement(step_elem, "description")
                    step_description_elem.text = step["description"]

        # Convert to string with proper formatting
        ET.indent(root, space="  ", level=0)
        xml_string = ET.tostring(root, encoding="unicode")
        
        # Decode HTML entities to provide clean XML to the LLM
        import html
        return html.unescape(xml_string)

    def get_string_explanations_from_plan(self, explanation_plan: List[ChosenExplanationModel]) -> str:
        """
        Get a list of explanations as string from the explanation plan to use in llm prompts.

        Args:
            explanation_plan (List[ChosenExplanationModel]): A list of chosen explanations.

        Returns:
            str: A string listing the explanations and steps.
        """
        try:
            explanations = [
                f"- Use the step **{choosen_exp.step_name}** of the explanation **{choosen_exp.explanation_name}**: " \
                f"{self._get_explanation_step(choosen_exp.explanation_name, choosen_exp.step_name).description}."
                for choosen_exp in explanation_plan
            ]
        except AttributeError:
            # Scaffolding Strategies
            explanations = [
                f"- Use the step **{choosen_exp.step_name}** of the explanation **{choosen_exp.explanation_name}**."
                for choosen_exp in explanation_plan
            ]
            # return "No explanations found in the explanation plan."
        return "\n".join(explanations)

    def persist_knowledge(self):
        # Reset all explanations that are not Concept explanations
        for state, explanations_dict in self.get_state_summary(as_dict=True).items():
            # Only reset the explanations that have been shown
            if state != "User Info":
                if state != "NOT_YET_EXPLAINED":
                    for exp_name in explanations_dict:
                        if exp_name != "PossibleClarifications":  # don't reset the clarifications
                            if exp_name != "Concept":  # don't reset the Concept step
                                self.reset_explanation_state(exp_name)
        # Return understood concepts
        return self.get_understood_concepts()

    def get_understood_concepts(self):
        """
        Get the concepts that the user has understood so far.
        """
        understood_concepts = []
        for state, explanations_dict in self.get_state_summary(as_dict=True).items():
            # Only reset the explanations that have been shown
            if state != "User Info":
                if state == "UNDERSTOOD":
                    for exp_name, concepts_list in explanations_dict.items():
                        for concept in concepts_list:
                            if concept == "Concept":
                                understood_concepts.append(exp_name)
        return understood_concepts

    def reset_understanding_displays(self):
        self.explicit_understanding_signals = []

    def __repr__(self):
        return f"User_Model({self.explanations})"
