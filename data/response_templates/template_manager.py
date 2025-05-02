import json
import os

from data.response_templates.feature_display_names import FeatureDisplayNames


class TemplateManager:
    def __init__(self,
                 conversation,
                 encoded_col_mapping_path=None,
                 categorical_mapping=None,
                 feature_name_mapping_path=None):
        self.conversation = conversation
        self.feature_name_mapping = self._load_feature_name_mapping(feature_name_mapping_path)
        self.feature_display_names = FeatureDisplayNames(conversation=self.conversation, feature_name_mapping=self.feature_name_mapping)
        self.encoded_col_mapping = self._load_label_encoded_feature_dict(encoded_col_mapping_path)
        self.categorical_mapping = categorical_mapping
        self.rounding_precision = 2

    def _load_feature_name_mapping(self, feature_name_mapping_path):
        """
        Load the mapping between original and renamed feature names
        :param feature_name_mapping_path: path to the feature name mapping JSON file
        :return: dictionary with original_to_renamed and renamed_to_original mappings
        """
        if feature_name_mapping_path is None or not os.path.exists(feature_name_mapping_path):
            return None
        try:
            with open(feature_name_mapping_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load feature name mapping from {feature_name_mapping_path}: {e}")
            return None

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
        return self.feature_display_names.get_display_name(feature_name)

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
        Apply categorical mapping to instances, using the mapping provided in the categorical_mapping.
        This replaces numeric encoded values with their categorical string values.

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
            # Handle DataFrame instances
            for column_index_str, column_mapping in self.categorical_mapping.items():
                try:
                    column_index = int(column_index_str)
                    if column_index < len(instance.columns):
                        column_name = instance.columns[column_index]
                        if column_name in instance.columns:
                            # Create a mapping dictionary for this column
                            mapping_dict = {i: category for i, category in enumerate(column_mapping)}
                            # Replace values using the mapping
                            instance[column_name] = instance[column_name].map(lambda x: mapping_dict.get(int(x), x))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Error mapping column {column_index_str}: {e}")
        else:
            # Handle dictionary instances
            # We need to map features by their position in the instance, not necessarily by direct key lookup
            feature_names = list(instance.keys())
            
            for i, feature_name in enumerate(feature_names):
                index_as_str = str(i)
                if index_as_str in self.categorical_mapping:
                    mapping = self.categorical_mapping[index_as_str]
                    value = instance[feature_name]
                    
                    if isinstance(value, dict):
                        # Handle case where value is a dict (e.g., {'current': 1, 'old': 0})
                        for key in ['current', 'old']:
                            if key in value:
                                try:
                                    # Map the numeric value to its categorical equivalent
                                    value[key] = mapping[int(value[key])]
                                except (KeyError, ValueError):
                                    print(f"Warning: Value {value[key]} not found in mapping for feature {feature_name} under key '{key}'.")
                        instance[feature_name] = value
                    else:
                        try:
                            # Map the numeric value to its categorical equivalent
                            mapped_value = mapping[int(value)]
                            instance[feature_name] = mapped_value
                        except (KeyError, ValueError):
                            # If mapping fails, leave the value as is
                            print(f"Warning: Value {value} not found in mapping for feature {feature_name}.")

        return instance

    def replace_feature_names_by_display_names(self, instance):
        """
        Replace feature names by display names in instance, preventing duplicate entries.

        Args:
            instance (dict): The instance to replace feature names by display names.

        Returns:
            The instance with replaced feature names by display names.
        """
        # Create a new dictionary to hold the display names and values
        display_instance = {}
        processed_display_names = set()  # Track display names we've already seen
        
        # Get a list of features in the preferred display order
        feature_ordering = None
        try:
            feature_ordering = self.conversation.get_var("feature_ordering")
        except:
            feature_ordering = []
            
        # Process features in the order specified by feature_ordering if available
        if feature_ordering:
            # First process features in the specified order
            for display_name in feature_ordering:
                # Find the original feature name that maps to this display name
                found = False
                for feature_name in instance.keys():
                    if self.get_feature_display_name_by_name(feature_name) == display_name:
                        display_instance[display_name] = instance[feature_name]
                        processed_display_names.add(display_name)
                        found = True
                        break
                
                if not found:
                    # This can happen if feature_ordering includes a feature not in this instance
                    continue
        
        # Process any remaining features not in the ordering
        for feature_name in instance.keys():
            display_name = self.get_feature_display_name_by_name(feature_name)
            
            # Skip if we've already processed this display name
            if display_name in processed_display_names:
                continue
                
            display_instance[display_name] = instance[feature_name]
            processed_display_names.add(display_name)
            
        return display_instance

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
