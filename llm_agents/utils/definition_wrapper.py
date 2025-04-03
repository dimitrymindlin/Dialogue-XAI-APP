import json


class DefinitionWrapper:
    def __init__(self, json_file_path):
        """
        Initializes the Wrapper with data from a JSON file.
        :param json_file_path: Path to the JSON file describing the concepts.
        """
<<<<<<< HEAD
        with open(json_file_path, 'r') as file:
=======
        with open(json_file_path, 'r', encoding='utf-8') as file:
>>>>>>> main
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
