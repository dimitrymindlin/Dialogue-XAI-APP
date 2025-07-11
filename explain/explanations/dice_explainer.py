import warnings
from typing import Any

import dice_ml
import gin
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.response_templates.dice_template import textual_cf
from explain.explanation import Explanation


@gin.configurable
class TabularDice(Explanation):
    """Tabular dice counterfactual explanations."""

    def __init__(self,
                 model,
                 data: pd.DataFrame,
                 num_features: list[str],
                 num_cfes_per_instance: int = 10,
                 num_in_short_summary: int = 3,
                 desired_class: str = "opposite",
                 cache_location: str = "./cache/dice-tabular.pkl",
                 class_names: dict = None,
                 categorical_mapping: dict = None,
                 background_dataset=None,
                 final_cfe_amount: int = 5,
                 features_to_vary="all",
                 max_features_to_change: int = None):
        """Init.

        Arguments:
            model: the sklearn style model, where model.predict(data) returns the predictions
                   and model.predict_proba returns the prediction probabilities
            data: the pandas df data
            num_features: The *names* of the numerical features in the dataframe
            num_cfes_per_instance: The total number of cfes to generate per instance
            num_in_short_summary: The number of cfes to include in the short summary
            desired_class: Set to "opposite" to compute opposite class
            cache_location: Location to store cache.
            class_names: The map between class names and text class description.
            max_features_to_change: Maximum number of features allowed to change in counterfactuals.
                                   If None, no limit is imposed. If set, only counterfactuals that 
                                   change <= this many features will be returned.
        """
        super().__init__(cache_location, class_names)
        self.temp_outcome_name = 'y'
        self.model = self.wrap(model)
        self.num_features = num_features
        self.desired_class = desired_class
        self.num_cfes_per_instance = num_cfes_per_instance
        self.num_in_short_summary = num_in_short_summary
        self.categorical_mapping = categorical_mapping
        self.rounding_precision = 2  # Default rounding precision for numeric values
        self.dice_model = dice_ml.Model(model=self.model, backend="sklearn")
        self.permitted_range_dict = None
        self.background_data = background_dataset
        self.final_cfe_amount = final_cfe_amount
        self.features_to_vary = features_to_vary
        self.max_features_to_change = max_features_to_change
        self.ids_without_cfes = []

        # Format data in dice accepted format
        predictions = self.model.predict(data)
        if self.model.predict_proba(data).shape[1] > 2:
            self.non_binary = True
        else:
            self.non_binary = False
        data[self.temp_outcome_name] = predictions

        self.classes = np.unique(predictions)
        self.dice_data = dice_ml.Data(dataframe=data,
                                      continuous_features=self.num_features,
                                      outcome_name=self.temp_outcome_name)

        data.pop(self.temp_outcome_name)

        self.exp = dice_ml.Dice(
            self.dice_data, self.dice_model, method="random")

    def wrap(self, model: Any):
        """Wraps model, converting pd to df to silence dice warnings"""

        class Model:
            def __init__(self, m):
                self.model = m

            def predict(self, X):
                return self.model.predict(X.values.astype(int))

            def predict_proba(self, X):
                return self.model.predict_proba(X.values.astype(int))

        return Model(model)

    def run_explanation(self,
                        data: pd.DataFrame,
                        desired_class: str = None):
        """Generate tabular dice explanations.

        Arguments:
            data: The data to generate explanations for in pandas df.
            desired_class: The desired class of the cfes. If None, will use the default provided
                           at initialization.
        Returns:
            explanations: The generated cf explanations.
        """
        if self.temp_outcome_name in data:
            raise NameError(f"Target Variable {self.temp_outcome_name} should not be in data.")

        if desired_class is None:
            desired_class = self.desired_class

        # Calculate permitted range for each feature
        if not self.permitted_range_dict:
            self.permitted_range_dict = {}
            for feature in self.num_features:
                self.permitted_range_dict[feature] = [self.background_data[feature].min(),
                                                      self.background_data[feature].max()]

        cfes = {}
        for d in tqdm(list(data.index)):
            # dice has a few function calls that are going to be deprecated
            # silence warnings for ease of use now
            current_instance = data.loc[[d]]
            current_class = self.model.predict(current_instance)[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.non_binary and desired_class == "opposite":
                    # TODO: Change to explain for EVERY other class
                    desired_class_tmp = int(
                        np.random.choice([p for p in self.classes if p != current_class]))
                else:
                    desired_class_tmp = desired_class

                cur_cfe = self.exp.generate_counterfactuals(current_instance,
                                                            total_CFs=self.num_cfes_per_instance,
                                                            desired_class=desired_class_tmp,
                                                            features_to_vary=self.features_to_vary,
                                                            permitted_range=self.permitted_range_dict)
            if cur_cfe.cf_examples_list[0].final_cfs_df is None:
                self.ids_without_cfes.append(d)
                continue
            cfes[d] = cur_cfe
        return cfes

    def get_change_string(self, cfe: Any, original_instance: Any, template_manager=None):
        """Builds a string describing the changes between the cfe and original instance."""
        cfe_features = list(cfe.columns)
        original_features = list(original_instance.columns)
        message = "CFE features and Original Instance features are different!"
        assert set(cfe_features) == set(original_features), message

        change_string = ""
        feature_display_names_dict = {}
        
        # Get display names dictionary from template_manager
        if template_manager and hasattr(template_manager, 'feature_display_names'):
            feature_display_names_dict = template_manager.feature_display_names.feature_name_to_display_name
            # Also get feature ordering if available
            feature_ordering = None
            try:
                if hasattr(template_manager.conversation, 'feature_ordering'):
                    feature_ordering = template_manager.conversation.feature_ordering
            except:
                pass
        
        for feature in cfe_features:
            # Get the actual column index from the original dataset for categorical mapping
            # This is different from the position in the current DataFrame
            try:
                # Try to get the column index from the dice_data if available
                if hasattr(self, 'dice_data') and hasattr(self.dice_data, 'data_df'):
                    feature_column_index = self.dice_data.data_df.columns.get_loc(feature)
                else:
                    # Fallback: use position in current DataFrame (original behavior)
                    feature_column_index = cfe_features.index(feature)
            except (AttributeError, KeyError):
                # Fallback: use position in current DataFrame 
                feature_column_index = cfe_features.index(feature)
            
            orig_f = original_instance[feature].values[0]
            cfe_f = cfe[feature].values[0]

            if isinstance(cfe_f, str):
                try:
                    cfe_f = float(cfe_f)
                except ValueError:
                    pass  # If it can't be converted to float, keep as string

            if orig_f != cfe_f:
                if isinstance(cfe_f, (int, float)) and isinstance(orig_f, (int, float)) and cfe_f > orig_f:
                    inc_dec = "Increasing"
                elif isinstance(cfe_f, (int, float)) and isinstance(orig_f, (int, float)):
                    inc_dec = "Decreasing"
                else:
                    inc_dec = "Changing"
                    
                # Turn feature to categorical name if possible
                if self.categorical_mapping is not None:
                    try:
                        # Convert to int if possible and use the correct column index
                        if isinstance(cfe_f, (int, float)):
                            # Use string key for categorical_mapping lookup
                            feature_column_index_str = str(feature_column_index)
                            if feature_column_index_str in self.categorical_mapping:
                                cfe_f = self.categorical_mapping[feature_column_index_str][int(cfe_f)]
                                inc_dec = "Changing"
                    except (KeyError, IndexError, ValueError) as e:
                        # Log error with more details for debugging
                        print(f"Error in categorical mapping for feature '{feature}' (column index {feature_column_index}) with value {cfe_f}: {e}")
                
                # Process the value for display
                if isinstance(cfe_f, float):
                    cfe_f = round(cfe_f, self.rounding_precision)
                    if cfe_f.is_integer():
                        cfe_f = int(cfe_f)
                
                # Find the display name for the feature
                feature_display_name = feature
                if feature in feature_display_names_dict:
                    feature_display_name = feature_display_names_dict[feature]
                else:
                    # Try case-insensitive matching
                    for key, value in feature_display_names_dict.items():
                        if key.lower() == feature.lower():
                            feature_display_name = value
                            break
                    
                    # If still not found, check if feature is in feature ordering
                    if not feature_display_name == feature and feature_ordering:
                        for display_name in feature_ordering:
                            # Convert display name to feature name format (removing spaces)
                            potential_feature_name = display_name.replace(" ", "")
                            if potential_feature_name.lower() == feature.lower():
                                feature_display_name = display_name
                                break
                
                change_string += f"{inc_dec} <b>{feature_display_name}</b> to <b>{cfe_f}</b>"
                change_string += " and "
                
        # Strip off last and
        change_string = change_string[:-5] if change_string else "No changes found"
        return change_string

    def get_final_cfes(self, data, ids, ids_to_regenerate=None, save_to_cache=False):
        """
        Returns the final cfes as pandas df and their ids for a given data instance.
        """
        explanation = self.get_explanations(ids,
                                            data,
                                            ids_to_regenerate=ids_to_regenerate,
                                            save_to_cache=save_to_cache)

        cfe = explanation[ids[0]]
        final_cfes = cfe.cf_examples_list[0].final_cfs_df
        desired_class = cfe.cf_examples_list[0].desired_class
        final_cfe_ids = list(final_cfes.index)

        if self.temp_outcome_name in final_cfes.columns:
            final_cfes.pop(self.temp_outcome_name)

        if len(final_cfe_ids) <= self.final_cfe_amount:
            return final_cfes, final_cfe_ids

        # Pick diverse cfes (don't use same combination of features twice)
        cfe_feature_mentions_list = []
        diverse_cfe_ids = []
        for index, row in final_cfes.iterrows():
            if len(diverse_cfe_ids) == self.final_cfe_amount:
                break
            # Check which features in the current row are different from the original instance
            cfe_feature_mentions = set()
            for feature in row.index:
                if feature == "y":
                    continue
                if row[feature] != data.loc[ids[0]][feature]:
                    cfe_feature_mentions.add(feature)
            
            # Apply feature change limit if specified
            if self.max_features_to_change is not None and len(cfe_feature_mentions) > self.max_features_to_change:
                continue
                
            # Check if such a set of features has already been used
            if cfe_feature_mentions not in cfe_feature_mentions_list:
                diverse_cfe_ids.append(index)
                cfe_feature_mentions_list.append(cfe_feature_mentions)
        return final_cfes.loc[diverse_cfe_ids], diverse_cfe_ids, desired_class

    def summarize_explanations(self,
                               data: pd.DataFrame,
                               ids_to_regenerate: list[int] = None,
                               filtering_text: str = None,
                               save_to_cache: bool = False,
                               template_manager=None):
        """Summarizes explanations for dice tabular.

        Arguments:
            data: pandas df containing data.
            ids_to_regenerate:
            filtering_text:
            save_to_cache:
        Returns:
            summary: a string containing the summary.
        """

        if ids_to_regenerate is None:
            ids_to_regenerate = []
        if data.shape[0] > 1:
            return ("", "I can only compute how to flip predictions for single instances at a time."
                        " Please narrow down your selection to a single instance. For example, you"
                        " could specify the id of the instance to want to figure out how to change.")

        ids = list(data.index)
        key = ids[0]

        final_cfes, final_cfe_ids, desired_class = self.get_final_cfes(data, ids,
                                                                       ids_to_regenerate=ids_to_regenerate,
                                                                       save_to_cache=save_to_cache)

        original_instance = data.loc[[key]]

        # Get all cfe strings and remove duplicates
        cfe_strings = [
            self.get_change_string(final_cfes.loc[[c_id]], original_instance, template_manager=template_manager) for
            c_id in final_cfe_ids]
        cfe_strings = list(set(cfe_strings))

        response = textual_cf(cfe_strings)

        return response, desired_class

    def summarize_cfe_for_given_attribute(self,
                                          cfe: pd.DataFrame,
                                          data: pd.DataFrame,
                                          attribute_to_vary: str,
                                          template_manager=None):
        """Summarizes explanations for a given counterfactual by dice tabular.

        Arguments:
            cfe: CounterfactualExample object.
            data: pandas df containing data.
            attribute_to_vary: The attribute to vary.
            template_manager: The template manager containing feature display names.
        Returns:
            summary: a string containing the summary.
        """

        if data.shape[0] > 1:
            return ("", "I can only compute how to flip predictions for single instances at a time."
                        " Please narrow down your selection to a single instance. For example, you"
                        " could specify the id of the instance to want to figure out how to change.")

        ids = list(data.index)
        key = ids[0]

        final_cfes = cfe[key].cf_examples_list[0].final_cfs_df
        if final_cfes is None:
            return (
                f"There are no changes possible to the chosen attribute alone that would result in a different prediction."), 0
        final_cfe_ids = list(final_cfes.index)

        if self.temp_outcome_name in final_cfes.columns:
            final_cfes.pop(self.temp_outcome_name)

        original_instance = data.loc[[key]]
        # Get all cfe strings and remove duplicates
        cfe_strings = [self.get_change_string(final_cfes.loc[[c_id]], original_instance, template_manager=template_manager) for c_id in final_cfe_ids]
        cfe_strings = list(set(cfe_strings))

        response = textual_cf(cfe_strings)

        return response
