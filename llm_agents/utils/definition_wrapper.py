import json


class DefinitionWrapper:
    def __init__(self, json_file_path):
        """
        Initializes the Wrapper with data from a JSON file.
        :param json_file_path: Path to the JSON file describing the concepts.
        """
        with open(json_file_path, 'r', encoding='utf-8') as file:
            self.displays = json.load(file)

    def as_text(self):
        """
        Provides a textual summary of all the possible concepts and their descriptions, along with examples.
        :return: A string summary suitable for use in an LLM prompt.
        """
        summary = []
        for display in self.displays:
            name = display.get("name", "Unnamed display")
            description = display.get("description", "No description available.")
            example = display.get("example", "No example provided.")
            differentiating_description = display.get("differentiating_description", None)
            if differentiating_description is not None:
                summary.append(f"- {name}: {description} {example} {differentiating_description}")
            else:
                summary.append(f"- {name}: {description} {example}")
        return "\n".join(summary)

    def as_text_filtered(self, user_model=None):
        """
        Provides a filtered textual summary containing only the name and implication fields.
        If user_model is provided, only returns understanding displays that are currently 
        assigned to the user (from their explicit_understanding_signals).
        This is used for Plan and Execute components to reduce information while maintaining
        the essential understanding signals.
        :param user_model: Optional user model containing explicit_understanding_signals
        :return: A string summary with only names and implications.
        """
        summary = []
        
        # If user model is provided and has understanding signals, filter to only those
        if (user_model and 
            hasattr(user_model, 'explicit_understanding_signals') and 
            user_model.explicit_understanding_signals):
            
            user_signals = user_model.explicit_understanding_signals
            for display in self.displays:
                name = display.get("name", "Unnamed display")
                if name in user_signals:
                    implication = display.get("implication", "No implication available.")
                    summary.append(f"- {name}: {implication}")
        else:
            # Fallback to showing all displays if no user model or no signals
            for display in self.displays:
                name = display.get("name", "Unnamed display")
                implication = display.get("implication", "No implication available.")
                summary.append(f"- {name}: {implication}")
        
        return "\n".join(summary)

    def get_differentiating_description(self, concept_name):
        """
        Provides the differentiating description for a given concept.
        :param concept_name: The name of the concept.
        :return: The differentiating description for the concept, or None if none is available.
        """
        for display in self.displays:
            if display["name"] == concept_name:
                return display.get("differentiating_description", None)
        return None

    def get_names(self):
        """
        Returns a list of all names from the definitions.
        :return: List of names from the JSON definitions.
        """
        return [display.get("name", "") for display in self.displays]
