import json
import logging
import os
import re


class FeatureDisplayNames:
    def __init__(self, conversation=None, feature_names=None, feature_name_mapping=None):
        """
        Initialize the FeatureDisplayNames class using the column_details from model config.
        
        Args:
            conversation: The conversation object (optional)
            feature_names: List of feature names (optional)
            feature_name_mapping: Optional mapping between original feature names and display names
        """
        self.conversation = conversation
        self.dataset_name = None
        self.config = {}

        # Initialize mappings
        self.feature_name_to_display_name = {}  # Maps original_name -> display_name (string only)
        self.display_name_to_feature_name = {}  # Maps display_name -> original_name
        self.feature_units = {}  # Maps original_name -> unit
        self.feature_tooltips = {}  # Maps original_name -> tooltip

        # Load dataset name if conversation is provided
        if conversation:
            try:
                self.dataset_name = self.conversation.describe.dataset_name
                # Load config to get column_details
                self.config = self._load_model_config()
            except:
                pass

        # Set mappings from provided mapping or config
        if feature_name_mapping:
            self._set_mappings_from_data(feature_name_mapping)
        elif self.config and "column_details" in self.config:
            self._set_mappings_from_data(self.config["column_details"])
        elif feature_names:
            # Generate display names for provided feature names
            self._generate_display_names(feature_names)

    def _set_mappings_from_data(self, data):
        """
        Set all mappings from input data, properly separating display names, units, and tooltips
        
        Args:
            data: List of [original_name, value] pairs, where value is either a display name string or a [display_name, unit, tooltip] list
        """
        # Expect list-of-pairs format for column_details
        if not isinstance(data, list):
            return
        items = data

        for item in items:
            # Unpack original name and associated value
            if isinstance(item, tuple):
                orig_name, value = item
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                orig_name, value = item[0], item[1]
            else:
                # Skip malformed entries
                continue

            # Determine display name, unit, and tooltip
            if isinstance(value, list):
                display_name = value[0]
                self.feature_units[orig_name] = value[1] if len(value) > 1 else ""
                self.feature_tooltips[orig_name] = value[2] if len(value) > 2 else ""
            else:
                display_name = value
                self.feature_units[orig_name] = ""
                self.feature_tooltips[orig_name] = ""

            # Store mapping
            self.feature_name_to_display_name[orig_name] = display_name

        # Create inverse mapping
        self.display_name_to_feature_name = {v: k for k, v in self.feature_name_to_display_name.items()}

    def _generate_display_names(self, feature_names):
        """Generate display names for a list of feature names"""
        for name in feature_names:
            display_name = self._generate_display_name(name)
            self.feature_name_to_display_name[name] = display_name
            self.feature_units[name] = ""
            self.feature_tooltips[name] = ""

        # Create inverse mapping
        self.display_name_to_feature_name = {v: k for k, v in self.feature_name_to_display_name.items()}

    def _generate_display_name(self, column_name):
        """
        Generate a user-friendly display name for a column
        """
        # Replace special characters with spaces
        name = re.sub(r'[._-]', ' ', column_name)

        # Convert camelCase to space-separated words
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)

        # Capitalize first letter of each word
        name = ' '.join(word.capitalize() for word in name.split())

        return name

    def _load_model_config(self):
        """Load the model config file which contains column_details mapping"""
        if not self.dataset_name:
            return {}

        # Try to load the model config file
        config_path = f"data/{self.dataset_name}/{self.dataset_name}_model_config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load model config from {config_path}: {e}")

        return {}

    def get_display_name(self, feature_name):
        """
        Get display name for a feature
        
        Args:
            feature_name: Original name of the feature
            
        Returns:
            Display name of the feature or the original name if not found
        """
        if feature_name in self.feature_name_to_display_name:
            return self.feature_name_to_display_name[feature_name]

        # Try with lowercase (for case insensitive matching)
        lowercase_feature_name = feature_name.lower()
        for key, value in self.feature_name_to_display_name.items():
            if key.lower() == lowercase_feature_name:
                return value

        return feature_name

    def get_original_name(self, display_name):
        """
        Get original feature name from display name
        
        Args:
            display_name: Display name of the feature
            
        Returns:
            Original name of the feature or the display name if not found
        """
        if display_name in self.display_name_to_feature_name:
            return self.display_name_to_feature_name[display_name]

        # Try with lowercase (for case insensitive matching)
        lowercase_display_name = display_name.lower()
        for key, value in self.display_name_to_feature_name.items():
            if key.lower() == lowercase_display_name:
                return value

        return display_name

    def get_unit(self, feature_name):
        """
        Get unit for a feature
        
        Args:
            feature_name: Original name of the feature
            
        Returns:
            Unit of the feature or empty string if not found
        """
        return self.feature_units.get(feature_name, "")

    def get_tooltip(self, feature_name):
        """
        Get tooltip for a feature
        
        Args:
            feature_name: Original name of the feature
            
        Returns:
            Tooltip of the feature or empty string if not found
        """
        return self.feature_tooltips.get(feature_name, "")

    def update_config(self, columns, target_col=None):
        """
        Update the mappings with all columns,
        generating display names for those not explicitly defined.
        
        Args:
            columns (list): List of column names
            target_col (str, optional): Target column to exclude
            
        Returns:
            Self for method chaining
        """
        # Add automatic display names for columns not already specified
        for col in columns:
            # Skip target column
            if col == target_col:
                continue

            # Only generate for columns not already specified
            if col not in self.feature_name_to_display_name:
                display_name = self._generate_display_name(col)
                self.feature_name_to_display_name[col] = display_name
                self.feature_units[col] = ""
                self.feature_tooltips[col] = ""
                self.display_name_to_feature_name[display_name] = col

        return self

    def apply_column_renaming(self, dataframe, in_place=False):
        """
        Apply column renaming to a dataframe
        
        Args:
            dataframe: DataFrame to rename columns for
            in_place: Whether to modify dataframe in place or return a copy
            
        Returns:
            DataFrame with renamed columns
        """
        # Use our mapping directly
        rename_mapping = self.feature_name_to_display_name

        if in_place:
            dataframe.rename(columns=rename_mapping, inplace=True)
            return dataframe
        else:
            return dataframe.rename(columns=rename_mapping)

    def save_to_config(self, config, save_path=None):
        """
        Save the feature mappings to a config
        
        Args:
            config: Config dictionary to update
            save_path: Optional path to save the updated config
            
        Returns:
            Updated config dictionary
        """
        # Create column_details with proper structure [display_name, unit, tooltip]
        column_details = {}

        for orig_name in self.feature_name_to_display_name:
            display_name = self.feature_name_to_display_name[orig_name]
            unit = self.feature_units.get(orig_name, "")
            tooltip = self.feature_tooltips.get(orig_name, "")

            # Store as a list with the triple structure
            column_details[orig_name] = [display_name, unit, tooltip]

        # Update config with our structured data
        config["column_details"] = column_details

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(config, f, indent=4)

        return config
