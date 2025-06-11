import json

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

    def clean_html_tags(self, text):
        """
        Removes common HTML tags from text and returns cleaned text.
        Carefully preserves word boundaries by adding spaces where needed.
        
        :param text: Text that may contain HTML tags.
        :return: Cleaned text with HTML tags removed and proper spacing.
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        cleaned_text = text
        
        # First, handle line break tags by replacing them with spaces
        line_break_tags = ["<br>", "<br/>", "<br />"]
        for tag in line_break_tags:
            cleaned_text = cleaned_text.replace(tag, " ")
        
        # Handle block-level tags that should add spaces around them
        block_tags = [
            "<p>", "</p>", "<div>", "</div>", "<h1>", "</h1>", "<h2>", "</h2>",
            "<h3>", "</h3>", "<h4>", "</h4>", "<h5>", "</h5>", "<h6>", "</h6>",
            "<ul>", "</ul>", "<ol>", "</ol>", "<li>", "</li>", 
            "<blockquote>", "</blockquote>", "<pre>", "</pre>"
        ]
        for tag in block_tags:
            # Add space before and after block tags to prevent word merging
            cleaned_text = cleaned_text.replace(tag, f" {tag} ")
        
        # Handle inline tags that might need spacing
        inline_tags = [
            "<b>", "</b>", "<i>", "</i>", "<strong>", "</strong>", "<em>", "</em>",
            "<span>", "</span>", "<u>", "</u>", "<a>", "</a>", "<code>", "</code>",
            "<small>", "</small>", "<sup>", "</sup>", "<sub>", "</sub>"
        ]
        
        # For inline tags, be more careful about spacing
        for tag in inline_tags:
            # Check if the tag is at word boundaries
            import re
            # Replace tags that are not already surrounded by whitespace
            pattern = rf'(\S){re.escape(tag)}(\S)'
            cleaned_text = re.sub(pattern, rf'\1 {tag} \2', cleaned_text)
            # Now remove the tags themselves
            cleaned_text = cleaned_text.replace(tag, "")
        
        # Remove any remaining tags we might have missed
        remaining_block_tags = [
            "<p>", "</p>", "<div>", "</div>", "<h1>", "</h1>", "<h2>", "</h2>",
            "<h3>", "</h3>", "<h4>", "</h4>", "<h5>", "</h5>", "<h6>", "</h6>",
            "<ul>", "</ul>", "<ol>", "</ol>", "<li>", "</li>", 
            "<blockquote>", "</blockquote>", "<pre>", "</pre>"
        ]
        for tag in remaining_block_tags:
            cleaned_text = cleaned_text.replace(tag, "")
        
        # Clean up multiple spaces and normalize whitespace
        import re
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text.strip()

    def format_pdp_as_xml(self, pdp_dict):
        """
        Formats PDP (Partial Dependence Plot) data as XML structure.
        
        :param pdp_dict: Dictionary containing PDP explanations for each feature
        :return: XML-formatted string with PDP results (YAML-safe)
        """
        if not pdp_dict:
            return "<pdp_results />"
        
        formatted_pdp = []
        for feature, pdp_text in pdp_dict.items():
            # Clean HTML tags from the PDP text
            cleaned_text = self.clean_html_tags(pdp_text) if isinstance(pdp_text, str) else str(pdp_text)
            
            # Escape any characters that might cause YAML parsing issues
            cleaned_text = cleaned_text.replace('"', '&quot;').replace("'", '&apos;')
            
            # Create XML structure for each PDP result with proper escaping
            pdp_xml = f'      <pdp_feature name="{feature}">'
            pdp_xml += f' <pdp_result_summary>{cleaned_text}</pdp_result_summary>'
            pdp_xml += ' </pdp_feature>'
            
            formatted_pdp.append(pdp_xml)
        
        # Return as a single line to avoid YAML multiline issues
        return "<pdp_results> " + " ".join(formatted_pdp) + " </pdp_results>"

    def format_clarifications_as_xml(self, clarifications_dict):
        """
        Formats clarifications data as XML structure.
        
        :param clarifications_dict: Dictionary containing clarification questions and answers
        :return: XML-formatted string with clarifications (YAML-safe)
        """
        if not clarifications_dict:
            return "<clarifications />"
        
        formatted_clarifications = []
        for question, answer in clarifications_dict.items():
            # Clean HTML tags from both question and answer
            cleaned_question = self.clean_html_tags(question) if isinstance(question, str) else str(question)
            cleaned_answer = self.clean_html_tags(answer) if isinstance(answer, str) else str(answer)
            
            # Escape any characters that might cause YAML parsing issues
            cleaned_question = cleaned_question.replace('"', '&quot;').replace("'", '&apos;')
            cleaned_answer = cleaned_answer.replace('"', '&quot;').replace("'", '&apos;')
            
            # Create XML structure for each clarification
            clarification_xml = f'      <clarification>'
            clarification_xml += f' <question>{cleaned_question}</question>'
            clarification_xml += f' <answer>{cleaned_answer}</answer>'
            clarification_xml += ' </clarification>'
            
            formatted_clarifications.append(clarification_xml)
        
        # Return as a single line to avoid YAML multiline issues
        return "<clarifications> " + " ".join(formatted_clarifications) + " </clarifications>"

    def format_feature_statistics_as_xml(self, feature_statistics_dict):
        """
        Formats feature statistics data as XML structure.
        
        :param feature_statistics_dict: Dictionary containing feature statistics
        :return: XML-formatted string with feature statistics (YAML-safe)
        """
        if not feature_statistics_dict:
            return "<feature_statistics />"
        
        formatted_stats = []
        for feature_name, stat_text in feature_statistics_dict.items():
            # Clean HTML tags from the statistics text
            cleaned_text = self.clean_html_tags(stat_text) if isinstance(stat_text, str) else str(stat_text)
            
            # Escape any characters that might cause YAML parsing issues
            cleaned_text = cleaned_text.replace('"', '&quot;').replace("'", '&apos;')
            
            # Create XML structure for each feature's statistics
            stat_xml = f'      <feature_stat name="{feature_name}">'
            stat_xml += f' <stat_summary>{cleaned_text}</stat_summary>'
            stat_xml += ' </feature_stat>'
            
            formatted_stats.append(stat_xml)
        
        # Return as a single line to avoid YAML multiline issues
        return "<feature_statistics> " + " ".join(formatted_stats) + " </feature_statistics>"

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
        }

        # Process Feature Statistics
        feature_statistics = self.xai_explanations.get("feature_statistics", {})
        substitution_dict["feature_statistics"] = {
            "feature_statistics": self.format_feature_statistics_as_xml(feature_statistics)
        }

        # Process Anchor explanation
        anchor = self.xai_explanations.get("anchors", {})
        substitution_dict["anchor"] = {
            "anchor_text": self.clean_html_tags(anchor)
        }

        # Process Ceteris Paribus with PossibleClassFlips and ImpossibleClassFlips
        ceteris_paribus = self.xai_explanations.get("ceteris_paribus", {})
        impossible_flips = [flip for flip in ceteris_paribus if "No changes" in flip]
        possible_flips = [flip for flip in ceteris_paribus if "No changes" not in flip]
        substitution_dict["ceteris_paribus"] = {
            "possible_class_flips": " ".join(f"{self.clean_html_tags(flip)}" for flip in possible_flips),
            "impossible_class_flips": " ".join(f"{self.clean_html_tags(flip)}" for flip in impossible_flips)
        }

        # Get pdp textual explanation
        pdp = self.xai_explanations.get("pdp", {})

        substitution_dict["pdp"] = {
            "all_pdp_text": self.format_pdp_as_xml(pdp),
        }

        # Get followup clarifications
        followup_clarifications = self.xai_explanations.get("static_clarifications", {})
        substitution_dict["static_clarifications"] = {
            "all_clarifications": self.format_clarifications_as_xml(followup_clarifications)
        }

        # Model confidence
        model_confidence = self.xai_explanations.get("model_confidence", "No model confidence provided.")
        substitution_dict["model_confidence"] = {
            "confidence_description": model_confidence
        }

        # Add new template fillers for the dynamic placeholders
        # Extract possible classes from predicted and opposite class names
        possible_classes = f"{self.predicted_class_name} or {self.opposite_class_name}"

        # Extract SHAP base value from xai_explanations if available
        shap_base_value = 0.5  # Default neutral probability
        try:
            # Get SHAP base value directly from the xai_explanations dict
            shap_base_value = self.xai_explanations.get("shap_base_value", 0.5)
        except (KeyError, TypeError):
            pass  # Use default value

        # Get class names in proper order for SHAP bias interpretation
        class_names = self.xai_explanations.get("class_names", [self.opposite_class_name, self.predicted_class_name])

        # Determine which class the SHAP initial bias favors
        # If base_value > 0.5, it favors the positive class (class 1)
        # If base_value <= 0.5, it favors the negative class (class 0)
        # Note: This is independent of the current instance's prediction
        if shap_base_value > 0.5:
            # Favors positive class (class 1)
            shap_initial_bias = class_names[1] if len(class_names) > 1 else "positive class"
        else:
            # Favors negative class (class 0) 
            shap_initial_bias = class_names[0] if len(class_names) > 0 else "negative class"

        # Add the new dynamic placeholders to substitution dictionary
        substitution_dict.update({
            "possible_classes": possible_classes,
            "negative_class": self.opposite_class_name,
            "shap_base_value": round(shap_base_value, 3),
            "shap_initial_bias": shap_initial_bias
        })

        # Pass over all xai_explanations and clean HTML tags in explanations
        for key, value in self.xai_explanations.items():
            if isinstance(value, dict):
                # Only process if the key exists in substitution_dict and is a dict
                if key in substitution_dict and isinstance(substitution_dict[key], dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, str) and sub_key in substitution_dict[key]:
                            cleaned_value = self.clean_html_tags(sub_value)
                            substitution_dict[key][sub_key] = cleaned_value
            elif isinstance(value, str):
                # Only update if the key exists in substitution_dict
                if key in substitution_dict:
                    cleaned_value = self.clean_html_tags(value)
                    substitution_dict[key] = cleaned_value

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
            result = yaml.safe_load(self.populated_yaml_content)
            for node in result.get("xai_explanations", []):
                node["children"] = node.pop("explanation_steps", [])
                # Add ID field if it doesn't exist
                if "id" not in node:
                    node["id"] = node["explanation_name"].replace(" ", "").lower()

            # Build predefined plan: exclude ScaffoldingStrategy explanations
            result["predefined_plan"] = []
            for node in result.get("xai_explanations", []):
                # Skip scaffolding strategies when generating the predefined plan
                if node["explanation_name"] == "ScaffoldingStrategy":
                    continue

                # take the first two children (Concept and next step)
                first_two = node["children"][:2]
                node_id = node.get("id", node["explanation_name"].replace(" ", "").lower())
                result["predefined_plan"].append({
                    "id": node_id,
                    "title": node["explanation_name"],
                    "children": first_two
                })
            return result
        else:
            return self.populated_yaml_content

    def get_populated_json(self, as_dict=False):
        if self.populated_yaml_content is None:
            self.populate_yaml()

        data = yaml.safe_load(self.populated_yaml_content)
        for node in data.get("xai_explanations", []):
            node["children"] = node.pop("explanation_steps", [])
            # Add ID field if it doesn't exist
            if "id" not in node:
                node["id"] = node["explanation_name"].replace(" ", "").lower()

        # Build predefined plan for JSON output: exclude ScaffoldingStrategy explanations
        data["predefined_plan"] = []
        for node in data.get("xai_explanations", []):
            # Skip scaffolding strategies when generating the predefined plan
            if node["explanation_name"] == "ScaffoldingStrategy":
                continue

            first_two = node["children"][:2]
            # Generate ID from explanation_name if 'id' field doesn't exist
            node_id = node.get("id", node["explanation_name"].replace(" ", "").lower())
            data["predefined_plan"].append({
                "id": node_id,
                "title": node["explanation_name"],
                "children": first_two
            })

        if as_dict:
            return data
        else:
            return json.dumps(data)

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
