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
        self.rounding_precision = 2

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
        df_copy = df.copy()  # Create a copy of the DataFrame
        if self.encoded_col_mapping is None:
            return df_copy
        for col in df_copy.columns:
            if col in self.encoded_col_mapping:
                df_copy[col] = df_copy[col].apply(lambda x: self.encoded_col_mapping[col].get(str(x), x))
        return df_copy

    def apply_categorical_mapping(self, instance, is_dataframe=False):
        """
        Apply categorical mapping to instances.

        Args:
            instance (dict or DataFrame): The instance to apply categorical mapping on.
            is_dataframe (bool): Flag to indicate if the instances are in a DataFrame. Default is False.

        Returns:
            The instances with applied categorical mapping.
        """
        if self.categorical_mapping is None:
            if is_dataframe:
                return instance.astype(float)
            else:
                return instance

        if is_dataframe:
            # Iterate only over columns that have a categorical mapping.
            for column_index, column_mapping in self.categorical_mapping.items():
                column_name = instance.columns[column_index]
                old_values_copy = int(instance[column_name].values)
                if column_name in instance.columns:
                    # Prepare a mapping dictionary for the current column.
                    mapping_dict = {i: category for i, category in enumerate(column_mapping)}
                    # Replace the entire column values based on mapping_dict.
                    instance[column_name] = instance[column_name].replace(mapping_dict)
                    if old_values_copy == instance[column_name].values[0]:
                        raise ValueError(f"Column {column_name} was not replaced with categorical mapping.")
        else:
            for i, (feature_name, val) in enumerate(instance.items()):
                index_as_str = str(i)
                if index_as_str in self.categorical_mapping:
                    mapping = self.categorical_mapping[index_as_str]
                    if isinstance(val, dict):
                        for key in ['current', 'old']:
                            if key in val:
                                try:
                                    val[key] = mapping[int(val[key])]
                                except KeyError:
                                    raise ValueError(
                                        f"Value {val[key]} not found in categorical mapping for feature {feature_name} under key '{key}'."
                                    )
                        instance[feature_name] = val
                    else:
                        try:
                            instance[feature_name] = mapping[int(val)]
                        except KeyError:
                            raise ValueError(
                                f"Value {int(val)} not found in categorical mapping for feature {feature_name}."
                            )

        return instance

    def replace_feature_names_by_display_names(self, instance):
        """
        Replace feature names by display names in instance.

        Args:
            instance (dict): The instance to replace feature names by display names.

        Returns:
            The instance with replaced feature names by display names.
        """
        keys = list(instance.keys())
        for feature_name in keys:
            feature_value = instance[feature_name]
            display_name = self.get_feature_display_name_by_name(feature_name)
            if display_name != feature_name:  # Avoids deleting a col when name is the same as display name
                instance[display_name] = feature_value
                del instance[feature_name]
        return instance

    def get_feature_display_value(self, feature_value):
        """
        Return displayable value by trying to turn to int or float with rounding precisiong and then to string
        """
        if feature_value.isdigit():
            feature_value = int(feature_value)
        elif "." in feature_value:
            float_value = float(feature_value)
            if float_value.is_integer():
                feature_value = int(float_value)
            else:
                feature_value = round(float_value, self.rounding_precision)
        return str(feature_value)
