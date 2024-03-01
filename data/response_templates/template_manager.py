import json

from data.response_templates.feature_display_names import FeatureDisplayNames


class TemplateManager:
    def __init__(self,
                 conversation,
                 encoded_col_mapping_path=None):
        self.conversation = conversation
        self.feature_display_names = FeatureDisplayNames(self.conversation)
        self.encoded_col_mapping = self._load_label_encoded_feature_dict(encoded_col_mapping_path)

    def _load_label_encoded_feature_dict(self, encoded_col_mapping_path):
        """
        Function to load label encoded feature dictionary
        :return: label encoded feature dictionary
        """
        if encoded_col_mapping_path is None:
            return None
        with open(encoded_col_mapping_path, "r") as f:
            return json.load(f)

    def get_encoded_feature_name(self, feature_name, feature_value):
        """
        Function to get label encoded feature name
        :param feature_name: feature name
        :param feature_value: feature value
        :return: label encoded feature name
        """
        # feature_value is a string. turn into int if float but as as string
        if feature_value.isdigit():
            feature_value = int(feature_value)
            feature_value = str(feature_value)
        elif "." in feature_value:
            feature_value = feature_value.split(".")[0]

        try:
            return self.encoded_col_mapping.get(feature_name).get(feature_value)
        except AttributeError:
            # give a warning that the feature value is not encoded
            warning = f"Feature value {feature_value} for feature {feature_name} is not encoded"
            print(warning)
            return feature_value

    def get_feature_display_name_by_name(self, feature_name):
        """
        Function to get display feature name
        :param feature_name: feature name
        :return: display feature name
        """
        return self.feature_display_names.get_by_name(feature_name)
