import re


class FeatureDisplayNames:
    def __init__(self, conversation):
        self.conversation = conversation
        self.feature_names = self._camel_case_to_title_case()
        self.feature_name_to_display_name = self._get_feature_name_to_display_name_dict()

    def _camel_case_to_title_case(self):
        """
        Function to get feature display names for the dataset used in the experiment.
        :return: dict of feature display
        """

        feature_names = list(self.conversation.get_var("dataset").contents['X'].columns)
        feature_names = {feature_id: feature_name for feature_id, feature_name in enumerate(feature_names)}
        # change feature names from camelCase to Title Case
        for feature_id, feature_name in feature_names.items():
            # TODO: Put these exceptions in config
            if feature_name == "BMI":
                feature_name = "Body Mass Index"
            else:
                feature_name = re.sub(r"(\w)([A-Z])", r"\1 \2", feature_name)  # split camelCase
                feature_name = feature_name.title()  # convert to Title Case
            feature_names[feature_id] = feature_name
        return feature_names

    def _get_feature_name_to_display_name_dict(self):
        """
        Mehod returns a dict that maps feature names to display names
        """
        feature_name_to_display_name_dict = {}
        for feature_id, display_name in self.feature_names.items():
            original_feature_name = self.conversation.get_var("dataset").contents['X'].columns[feature_id]
            feature_name_to_display_name_dict[original_feature_name] = display_name
        return feature_name_to_display_name_dict

    def get_by_id(self, feature_id):
        """
        Function to get feature display name by feature id
        :param feature_id: feature id
        :return: feature display name
        """
        return self.feature_names.get(feature_id)

    def get_by_name(self, feature_name):
        return self.feature_name_to_display_name[feature_name]
