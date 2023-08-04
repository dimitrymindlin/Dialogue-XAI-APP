from typing import List

import numpy as np
import pandas as pd


class ExperimentHelper:

    def __init__(self, conversation, categorical_mapping, categorical_features):
        self.conversation = conversation
        self.categorical_mapping = categorical_mapping
        self.categorical_features = categorical_features

    def get_counterfactual_instances(self,
                                     original_instance,
                                     features_to_vary="all"):
        dice_tabular = self.conversation.get_var('tabular_dice').contents
        # Turn original intstance into a dataframe
        if not isinstance(original_instance, pd.DataFrame):
            original_instance = pd.DataFrame.from_dict(original_instance["values"], orient="index").transpose()
        cfes = dice_tabular.run_explanation(original_instance, "opposite", features_to_vary=features_to_vary)
        original_instance_id = original_instance.index[0]
        final_cfs_df = cfes[original_instance_id].cf_examples_list[0].final_cfs_df
        # drop y column
        final_cfs_df = final_cfs_df.drop(columns=["y"])
        return final_cfs_df

    def get_similar_instance(self,
                             original_instance,
                             model,
                             actionable_features: List[int] = list(),
                             max_features_to_vary=3):
        result_instance = None
        changed_features = 0
        for feature_name in actionable_features:
            # randomly decide if this feature should be changed
            if np.random.randint(0, 3) == 0:  # 66% chance to change
                continue
            tmp_instance = original_instance.copy() if result_instance is None else result_instance.copy()

            # Get random change value for this feature
            if feature_name in self.categorical_features:
                max_feature_value = len(
                    self.categorical_mapping[original_instance.columns.get_loc(feature_name)])
                random_change = np.random.randint(0, max_feature_value)
                tmp_instance.at[tmp_instance.index[0], feature_name] += random_change
                tmp_instance.at[tmp_instance.index[0], feature_name] %= max_feature_value
            else:
                # Sample around mean for numerical features
                feature_mean = np.mean(self.conversation.get_var('dataset').contents["X"][feature_name])
                feature_std = np.std(self.conversation.get_var('dataset').contents["X"][feature_name])
                # Sample around feature mean
                random_change = np.random.normal(loc=feature_mean, scale=feature_std)
                while random_change == tmp_instance.at[tmp_instance.index[0], feature_name]:
                    random_change = np.random.normal(loc=feature_mean, scale=feature_std)
                tmp_instance.at[tmp_instance.index[0], feature_name] = random_change

            # Check if prediction stays the same
            if model.predict(tmp_instance)[0] == model.predict(original_instance)[0]:
                result_instance = tmp_instance.copy()
                changed_features += 1
            if changed_features == max_features_to_vary:
                break
        return result_instance
