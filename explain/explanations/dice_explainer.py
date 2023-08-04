import warnings
from typing import Any

import dice_ml
import gin
import numpy as np
import pandas as pd
from tqdm import tqdm

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
                 background_dataset=None):
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
        """
        super().__init__(cache_location, class_names)
        self.temp_outcome_name = 'y'
        self.model = self.wrap(model)
        self.num_features = num_features
        self.desired_class = desired_class
        self.num_cfes_per_instance = num_cfes_per_instance
        self.num_in_short_summary = num_in_short_summary
        self.categorical_mapping = categorical_mapping
        self.dice_model = dice_ml.Model(model=self.model, backend="sklearn")
        self.permitted_range_dict = None
        self.background_data = background_dataset

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
                        desired_class: str = None,
                        features_to_vary="all"):
        """Generate tabular dice explanations.

        Arguments:
            data: The data to generate explanations for in pandas df.
            desired_class: The desired class of the cfes. If None, will use the default provided
                           at initialization.
            features_to_vary: Either a string "all" or a list of feature names to vary.
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.non_binary and desired_class == "opposite":
                    desired_class = int(
                        np.random.choice([p for p in self.classes if p != self.model.predict(data.loc[[d]])[0]]))
                cur_cfe = self.exp.generate_counterfactuals(data.loc[[d]],
                                                            total_CFs=self.num_cfes_per_instance,
                                                            desired_class=desired_class,
                                                            features_to_vary=features_to_vary,
                                                            permitted_range=self.permitted_range_dict)
            cfes[d] = cur_cfe
        return cfes

    def get_change_string(self, cfe: Any, original_instance: Any):
        """Builds a string describing the changes between the cfe and original instance."""
        cfe_features = list(cfe.columns)
        original_features = list(original_instance.columns)
        message = "CFE features and Original Instance features are different!"
        assert set(cfe_features) == set(original_features), message

        change_string = ""
        for feature in cfe_features:
            feature_index = cfe_features.index(feature)
            orig_f = original_instance[feature].values[0]
            cfe_f = cfe[feature].values[0]

            if isinstance(cfe_f, str):
                cfe_f = float(cfe_f)

            if orig_f != cfe_f:
                if cfe_f > orig_f:
                    inc_dec = "Increasing"
                else:
                    inc_dec = "Decreasing"
                # Turn feature to categorical name if possible
                if self.categorical_mapping is not None:
                    try:
                        cfe_f = self.categorical_mapping[feature_index][int(cfe_f)]
                        inc_dec = "Changing"
                    except KeyError:
                        pass  # feature is numeric and not in categorical mapping
                    except IndexError:
                        print("Index error in DICE explanation encountered...")
                # round cfe_f if it is float and turn to string to print
                if isinstance(cfe_f, float):
                    cfe_f = str(round(cfe_f, self.rounding_precision))
                change_string += f"{inc_dec} <b>{feature}</b> to </b>{cfe_f}</b>"
                change_string += " and "
        # Strip off last and
        change_string = change_string[:-5]
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
        final_cfe_ids = list(final_cfes.index)

        if self.temp_outcome_name in final_cfes.columns:
            final_cfes.pop(self.temp_outcome_name)
        return final_cfes, final_cfe_ids

    def summarize_explanations(self,
                               data: pd.DataFrame,
                               ids_to_regenerate: list[int] = None,
                               filtering_text: str = None,
                               save_to_cache: bool = False):
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

        final_cfes, final_cfe_ids = self.get_final_cfes(data, ids,
                                                        ids_to_regenerate=ids_to_regenerate,
                                                        save_to_cache=save_to_cache)

        original_instance = data.loc[[key]]

        if filtering_text is not None and len(filtering_text) > 0:
            filtering_description = f"For instances where <b>{filtering_text}</b>"
        else:
            filtering_description = ""
        # output_string = f"{filtering_description}, the original prediction is "
        # output_string += f"<em>{original_label}</em>. "
        output_string = ""
        output_string += "Here are some changes where the model would predict the opposite class:"
        output_string += "<br><br>"

        additional_options = "Here are some more options to change the prediction of"
        additional_options += f" instance id {str(key)}.<br><br>"

        output_string += ""
        #transition_words = ["Further,", "Also,", "In addition,", "Furthermore,"]

        # Get all cfe strings and remove duplicates
        cfe_strings = [self.get_change_string(final_cfes.loc[[c_id]], original_instance) for c_id in final_cfe_ids]
        cfe_strings = list(set(cfe_strings))

        for i, cfe_string in enumerate(cfe_strings):
            # Stop the summary in case its getting too large
            if i < self.num_in_short_summary:
                #if i != 0:
                #    output_string += f"{np.random.choice(transition_words)} if you <em>"
                output_string += cfe_string
                # new_prediction = self.get_label_text(new_predictions[i])
                # output_string += f"</em>, the model will predict {new_prediction}.<br><br>"
                output_string += f".<br><br>"
            else:
                additional_options += "If you <em>"
                additional_options += cfe_string
                #new_prediction = self.get_label_text(new_predictions[i])
                #additional_options += f"</em>, the model will predict {new_prediction}.<br><br>"
                #additional_options += f"</em>, the model will predict the opposite class.<br><br>"

        # output_string += "If you want some more options, just ask &#129502"

        return additional_options, output_string

    def summarize_cfe(self,
                      cfe: pd.DataFrame,
                      data: pd.DataFrame):
        """Summarizes explanations for a given counterfactual by dice tabular.

        Arguments:
            cfe: CounterfactualExample object.
            data: pandas df containing data.
            save_to_cache:
        Returns:
            summary: a string containing the summary.
        """

        if data.shape[0] > 1:
            return ("", "I can only compute how to flip predictions for single instances at a time."
                        " Please narrow down your selection to a single instance. For example, you"
                        " could specify the id of the instance to want to figure out how to change.")

        ids = list(data.index)
        key = ids[0]

        original_prediction = self.model.predict(data)[0]
        original_label = self.get_label_text(original_prediction)

        final_cfes = cfe.cf_examples_list[0].final_cfs_df
        if final_cfes is None:
            return (
                       f"There are no changes possible to the chosen attribute alone that would result in a different prediction."), 0
        final_cfe_ids = list(final_cfes.index)

        if self.temp_outcome_name in final_cfes.columns:
            final_cfes.pop(self.temp_outcome_name)

        new_predictions = self.model.predict(final_cfes)

        original_instance = data.loc[[key]]

        # output_string = f"The original prediction is "
        # output_string += f"<em>{original_label}</em>. "
        output_string = ""
        output_string += "Here is how you could switch the attribute values to flip the prediction."
        output_string += "<br><br>"

        output_string += "If you <em>"
        transition_words = ["Further,", "Also,", "In addition,", "Furthermore,"]

        for i, c_id in enumerate(final_cfe_ids):
            # Stop the summary in case its getting too large
            if i < self.num_in_short_summary:
                if i != 0:
                    output_string += f"{np.random.choice(transition_words)} if you <em>"
                output_string += self.get_change_string(final_cfes.loc[[c_id]], original_instance)
                new_prediction = self.get_label_text(new_predictions[i])
                output_string += f"</em>, the model will predict {new_prediction}.<br><br>"

        return output_string, 1
