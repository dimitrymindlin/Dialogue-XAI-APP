import json


class ICAPModes:
    def __init__(self, json_file_path):
        """
        Initializes the ICAPModesWrapper with data from a JSON file.
        :param json_file_path: Path to the JSON file describing ICAP modes of engagement.
        """
        with open(json_file_path, 'r') as file:
            self.modes = json.load(file)

    def get_modes_as_text(self):
        """
        Provides a textual summary of all ICAP modes and their descriptions, along with examples.
        :return: A string summary suitable for use in an LLM prompt.
        """
        summary = []
        for mode in self.modes:
            name = mode.get("name", "Unnamed mode")
            description = mode.get("description", "No description available.")
            example = mode.get("example", "No example provided.")
            summary.append(f"- {name}: {description} {example}")
        return "\n".join(summary)
