import json


class DefinitionWrapper:
    def __init__(self, json_file_path):
        """
        Initializes the Wrapper with data from a JSON file.
        :param json_file_path: Path to the JSON file describing the concepts.
        """
        with open(json_file_path, 'r') as file:
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
            summary.append(f"- {name}: {description} {example}")
        return "\n".join(summary)
