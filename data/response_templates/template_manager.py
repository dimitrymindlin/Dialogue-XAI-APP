import json

from data.response_templates.feature_display_names import FeatureDisplayNames


class TemplateManager:
    def __init__(self,
                 conversation,
                 encoded_col_mapping_path=None,
                 categorical_mapping=None):
        self.conversation = conversation
        self.feature_display_names = FeatureDisplayNames(self.conversation)
        self.encoded_col_mapping = self._load_label_encoded_feature_dict(encoded_col_mapping_path)
        self.categorical_mapping = categorical_mapping

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

    def decode_numeric_columns_to_names(self, df):
        """
        Function to turn integer dataframe to feature names
        :param df: dataframe
        :return: dataframe with feature names
        """
        if self.encoded_col_mapping is None:
            return df
        for col in df.columns:
            if col in self.encoded_col_mapping:
                df[col] = df[col].apply(lambda x: self.encoded_col_mapping[col].get(str(x), x))
        return df

    def apply_categorical_mapping(self, instances, is_dataframe=False):
        """
        Apply categorical mapping to instances.

        Args:
            instances (dict or DataFrame): The instances to apply categorical mapping on.
            is_dataframe (bool): Flag to indicate if the instances are in a DataFrame. Default is False.

        Returns:
            The instances with applied categorical mapping.
        """
        if self.categorical_mapping is None:
            if is_dataframe:
                return instances.astype(float)
            else:
                return instances

        if is_dataframe:
            # Iterate only over columns that have a categorical mapping.
            for column_index, column_mapping in self.categorical_mapping.items():
                column_name = instances.columns[column_index]
                old_values_copy = int(instances[column_name].values)
                if column_name in instances.columns:
                    # Prepare a mapping dictionary for the current column.
                    mapping_dict = {i: category for i, category in enumerate(column_mapping)}
                    # Replace the entire column values based on mapping_dict.
                    instances[column_name] = instances[column_name].replace(mapping_dict)
                    if old_values_copy == instances[column_name].values[0]:
                        raise ValueError(f"Column {column_name} was not replaced with categorical mapping.")
        else:
            for i, (feature_name, val) in enumerate(instances.items()):
                index_as_str = str(i)
                if index_as_str in self.categorical_mapping:
                    try:
                        instances[feature_name] = self.categorical_mapping[index_as_str][val]
                    except KeyError:
                        raise ValueError(f"Value {val} not found in categorical mapping for feature {feature_name}.")

        return instances
