from typing import List

import numpy as np
import pandas as pd

from create_experiment_data.similar_instances import SimilarInstancesByExplanation


class ExperimentHelper:

    def __init__(self, conversation):
        self.conversation = conversation

    def get_counterfactual_instances(self,
                                     original_instance,
                                     features_to_vary="all"):
        dice_tabular = self.conversation.get_var('tabular_dice').contents
        # Turn original intstance into a dataframe
        instance_df = pd.DataFrame.from_dict(original_instance["values"], orient="index").transpose()
        instance_df.index = [original_instance["id"]]
        cfes = dice_tabular.run_explanation(instance_df, "opposite", features_to_vary=features_to_vary)
        return cfes[original_instance["id"]].cf_examples_list[0].final_cfs_df

    def get_similar_instances(self,
                              original_instance,
                              model,
                              changeable_features: List[int] = list(),
                              max_features_to_vary=3):
        result_instance = None
        changed_features = 0
        for feature_name in changeable_features:
            # randomly decide if this feature should be changed
            if np.random.randint(0, 3) == 0:  # 66% chance to change
                continue
            tmp_instance = original_instance.copy() if result_instance is None else result_instance.copy()

            # Get random change value for this feature
            max_feature_value = len(
                self.conversation.categorical_mapping[original_instance.columns.get_loc(feature_name)])
            random_change = np.random.randint(0, max_feature_value)
            tmp_instance.at[tmp_instance.index[0], feature_name] += random_change
            tmp_instance.at[tmp_instance.index[0], feature_name] %= max_feature_value
            # Check if prediction stays the same
            if model.predict(tmp_instance)[0] == model.predict(original_instance)[0]:
                result_instance = tmp_instance.copy()
                changed_features += 1
            if changed_features == max_features_to_vary:
                break
        return result_instance
