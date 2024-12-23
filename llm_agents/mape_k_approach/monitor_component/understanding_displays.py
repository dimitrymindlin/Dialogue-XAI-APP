import json


class UnderstandingDisplays:
    def __init__(self, json_file_path):
        """
        Initializes the UnderstandingDisplayWrapper with data from a JSON file.
        :param json_file_path: Path to the JSON file describing understanding displays.
        """
        with open(json_file_path, 'r') as file:
            self.displays = json.load(file)

    def get_displays_as_text(self):
        """
        Provides a textual summary of all the possible displays and their descriptions, along with examples.
        :return: A string summary suitable for use in an LLM prompt.
        """
        summary = []
        for display in self.displays:
            name = display.get("name", "Unnamed display")
            description = display.get("description", "No description available.")
            example = display.get("example", "No example provided.")
            summary.append(f"- {name}: {description} {example}")
        return "\n".join(summary)
