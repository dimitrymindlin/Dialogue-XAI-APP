import yaml
from jinja2 import Environment, FileSystemLoader


class XAIExplanationPopulator:
    def __init__(self, template_dir, template_file, xai_explanations, predicted_class_name, opposite_class_name,
                 instance_dict):
        """
        Initializes the XAIExplanationPopulator with necessary data.

        :param template_dir: Directory where the YAML template resides.
        :param template_file: Filename of the YAML template.
        :param xai_explanations: Dictionary containing xAI explanations.
        :param predicted_class_name: The name of the predicted class.
        :param opposite_class_name: The name of the opposite class.
        :param instance_dict: Dictionary of feature names and their values for the instance.
        """
        self.template_dir = template_dir
        self.template_file = template_file
        self.xai_explanations = xai_explanations
        self.predicted_class_name = predicted_class_name
        self.opposite_class_name = opposite_class_name
        self.instance_dict = instance_dict
        self.substitution_dict = self.process_xai_explanations()
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=False
        )
        self.template = self.env.get_template(self.template_file)
        self.populated_yaml_content = None

    def process_xai_explanations(self):
        """
        Processes the xai_explanations and creates a substitution dictionary
        mapping placeholders to their actual explanations.

        :return: A dictionary mapping placeholders to substitution values.
        """
        substitution_dict = {}

        # Process Feature Importance
        feature_importances = self.xai_explanations.get("feature_importance", [{}])[0]
        feature_names_to_values = {k.lower().replace(" ", ""): v for k, v in self.instance_dict.items()}

        # Calculate total absolute importance for percentage calculation
        total_importance = sum(abs(val[0]) for val in feature_importances.values())
        if total_importance == 0:
            total_importance = 1  # Prevent division by zero

        # Sort features by absolute importance descending
        sorted_features = sorted(feature_importances.items(), key=lambda item: abs(item[1][0]), reverse=True)

        in_favour_of = []
        against_predicted = []

        for idx, (feature_name, importance) in enumerate(sorted_features):
            feature_key = feature_name.lower().replace(" ", "")
            feature_value = feature_names_to_values.get(feature_key, "unknown")
            importance_percentage = round((abs(importance[0]) / total_importance) * 100)

            if importance[0] > 0:
                # in favor of predicted class
                description = f"{feature_name} (value: {feature_value}) supports the prediction of ‘{self.predicted_class_name}’ " \
                              f"with an importance of {importance_percentage}%, ranking #{idx + 1} overall."
                in_favour_of.append(description)
            else:
                # against predicted class
                description = f"{feature_name} (value: {feature_value}) supports the alternative class ‘{self.opposite_class_name}’ " \
                              f"with an importance of {importance_percentage}%, ranking #{idx + 1} overall."
                against_predicted.append(description)

        substitution_dict["feature_importance"] = {
            f"features_in_favour_of_{self.predicted_class_name.replace(' ', '_')}": " ".join(
                f"{desc}" for desc in in_favour_of),
            f"features_in_favour_of_{self.opposite_class_name.replace(' ', '_')}": " ".join(
                f"{desc}" for desc in against_predicted),
            "feature_influences_plot_url": "https://yourdomain.com/path/to/feature_influences_plot.png"
        }

        # Process Counterfactuals
        counterfactuals = self.xai_explanations.get("counterfactuals", {})
        substitution_dict["counterfactuals"] = {
            "possible_counterfactuals": counterfactuals,
            "single_feature_cf": "Placeholder ... No way to determine single feature cf yet."
        }

        # Process Feature Statistics
        feature_statistics = self.xai_explanations.get("feature_statistics", {})
        substitution_dict["feature_statistics"] = {
            "feature_statistics": " ".join(
                f"{k}: {v}" for k, v in feature_statistics.items()
            )
        }

        # Process Ceteris Paribus with PossibleClassFlips and ImpossibleClassFlips
        ceteris_paribus = self.xai_explanations.get("ceteris_paribus", {})
        impossible_flips = [flip for flip in ceteris_paribus if "No changes" in flip]
        possible_flips = [flip for flip in ceteris_paribus if "No changes" not in flip]
        substitution_dict["ceteris_paribus"] = {
            "possible_class_flips": " ".join(f"{flip}" for flip in possible_flips),
            "impossible_class_flips": " ".join(f"{flip}" for flip in impossible_flips)
        }

        return substitution_dict

    def populate_yaml(self):
        """
        Populates the YAML template by substituting placeholders with actual explanations.
        Sets the populated YAML content internally.
        """
        try:
            self.populated_yaml_content = self.template.render(self.substitution_dict)
            # Optional: Check for any remaining placeholders (indicated by '{{' and '}}')
            if "{{" in self.populated_yaml_content and "}}" in self.populated_yaml_content:
                print("Warning: Some placeholders were not substituted.")
        except Exception as e:
            print(f"Error during YAML population: {e}")
            raise

    def get_populated_yaml(self, as_dict=False):
        """
        Returns the populated YAML content.

        :param as_dict: If True, returns the YAML content as a Python dictionary. Otherwise, returns as a string.
        :return: Populated YAML as a string or dictionary.
        """
        if self.populated_yaml_content is None:
            self.populate_yaml()

        if as_dict:
            return yaml.safe_load(self.populated_yaml_content)
        else:
            return self.populated_yaml_content

    def save_populated_yaml(self, output_yaml_path):
        """
        Saves the populated YAML content to the specified file path.

        :param output_yaml_path: Path where the populated YAML should be saved.
        """
        if self.populated_yaml_content is None:
            self.populate_yaml()

        with open(output_yaml_path, "w") as file:
            file.write(self.populated_yaml_content)

        print(f"Populated YAML saved to {output_yaml_path}")

    def validate_substitutions(self):
        """
        Validates that all placeholders have been substituted.
        Raises an error if any remain.
        """
        if self.populated_yaml_content is None:
            raise ValueError("YAML content has not been populated yet.")

        remaining_placeholders = set()
        for line in self.populated_yaml_content.splitlines():
            if "{{" in line and "}}" in line:
                start = line.find("{{")
                end = line.find("}}", start)
                if start != -1 and end != -1:
                    placeholder = line[start:end + 2]
                    remaining_placeholders.add(placeholder)

        if remaining_placeholders:
            raise ValueError(f"Some placeholders were not substituted: {remaining_placeholders}")
        else:
            print("All placeholders were successfully substituted.")
