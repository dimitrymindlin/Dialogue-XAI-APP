import decimal
from typing import List

import numpy as np
import pandas as pd


class ExperimentHelper:
    def __init__(self, conversation,
                 categorical_mapping,
                 categorical_features,
                 template_manager):
        self.conversation = conversation
        self.categorical_mapping = categorical_mapping
        self.categorical_features = categorical_features
        self.template_manager = template_manager
        self.instances = {"data": [], "test": {}}
        self.current_instance = None
        self.current_instance_type = None
        self.instance_counters = {"train": 0, "test": 0}

    def load_instances(self):
        self._load_data_instances()
        self._load_test_instances()

    def _load_data_instances(self):
        diverse_instances = self.conversation.get_var("diverse_instances").contents
        self.instances["data"] = [self._prepare_instance_data(instance) for instance in diverse_instances]

    def _load_test_instances(self):
        test_instances = self.conversation.get_var("test_instances").contents
        self.instances["test"] = {instance_id: self._process_test_instances(instances_dict)
                                  for instance_id, instances_dict in test_instances.items()}

    def get_next_instance(self, train=True, return_probability=False):
        if train:
            if not self.instances["data"]:
                self._load_data_instances()
            instance, counter, probability = self._get_training_instance(return_probability)
            self.current_instance_type = "train"
        else:
            if not self.instances["test"]:
                self._load_test_instances()
            instance_id, instance, counter = self._get_test_instance()
            true_label = self.conversation.get_var("dataset").contents['y'].loc[instance_id]
            true_label_name = self.conversation.class_names[true_label]
            probability = None
            instance = (instance_id, instance, probability, true_label_name)
            self.current_instance_type = "test"

        if instance:
            self._round_instance_features(instance[1])
            self.template_manager.apply_categorical_mapping(instance[1])
            self.current_instance = instance
        return self.current_instance, counter, self.current_instance_type

    def _prepare_instance_data(self, instance):
        # Simplified example of preparing a data instance
        model_prediction = \
            self.conversation.get_var("model_prob_predict").contents(pd.DataFrame(instance['values'], index=[0]))[0]
        true_label = self._fetch_true_label(instance['id'])
        return instance['id'], instance['values'], model_prediction, true_label

    def _process_test_instances(self, instances_dict):
        # Example process for test instances
        return {comp: pd.DataFrame(data).to_dict('records')[0] for comp, data in instances_dict.items()}

    def _get_training_instance(self, return_probability):
        if not self.instances["data"]:
            return None, self.instance_counters["train"]
        self.current_instance = self.instances["data"].pop(0)
        self.instance_counters["train"] += 1
        if return_probability:
            return self.current_instance, self.instance_counters["train"], self.current_instance[2]
        return self.current_instance, self.instance_counters["train"]

    def _get_test_instance(self):
        if not self.instances["test"]:
            return None, self.instance_counters["test"]

        instance_key = "least_complex_instance" if self.instance_counters[
                                                       "test"] % 2 == 0 else "easy_counterfactual_instance"
        test_id, test_instances_dict = self.instances["test"].popitem()
        instance = test_instances_dict[instance_key]
        self.instance_counters["test"] += 1
        return test_id, instance, self.instance_counters["test"]

    def _round_instance_features(self, features):
        for feature, value in features.items():
            if isinstance(value, float):
                features[feature] = round(value, self.conversation.rounding_precision)

    def _fetch_true_label(self, instance_id):
        true_label = self.conversation.get_var("dataset").contents['y'].loc[instance_id]
        return self.conversation.class_names[true_label]

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
                             actionable_features: List[int],
                             max_features_to_vary=2):
        result_instance = None
        actionable_features = actionable_features if actionable_features is not None else list()
        changed_features = 0
        for feature_name in actionable_features:
            # randomly decide if this feature should be changed
            if np.random.randint(0, 3) == 0:  # 66% chance to change
                continue
            tmp_instance = original_instance.copy() if result_instance is None else result_instance.copy()

            # Get random change value for this feature
            if self.categorical_features is not None and feature_name in self.categorical_features:
                max_feature_value = len(
                    self.categorical_mapping[original_instance.columns.get_loc(feature_name)])
                random_change = np.random.randint(0, max_feature_value)
                tmp_instance.at[tmp_instance.index[0], feature_name] += random_change
                tmp_instance.at[tmp_instance.index[0], feature_name] %= max_feature_value
            else:
                # Sample around mean for numerical features
                feature_mean = np.mean(self.conversation.get_var('dataset').contents["X"][feature_name])
                feature_std = np.std(self.conversation.get_var('dataset').contents["X"][feature_name])
                feature_min = np.min(self.conversation.get_var('dataset').contents["X"][feature_name])
                feature_max = np.max(self.conversation.get_var('dataset').contents["X"][feature_name])
                # Sample around feature mean
                random_change = np.random.normal(loc=feature_mean, scale=feature_std)
                while random_change == tmp_instance.at[tmp_instance.index[0], feature_name]:
                    random_change = np.random.normal(loc=feature_mean, scale=feature_std)
                # Check if the new value is within the feature range
                if random_change < feature_min:
                    random_change = feature_min
                elif random_change > feature_max:
                    random_change = feature_max

                # Check precision of the feature and round accordingly
                # Sample feature values to determine precision
                feature_values = self.conversation.get_var('dataset').contents["X"][feature_name]

                # Check if the feature values are integers or floats
                if pd.api.types.is_integer_dtype(feature_values):
                    precision = 0  # No decimal places for integers
                elif pd.api.types.is_float_dtype(feature_values):
                    # Determine the typical number of decimal places if floats
                    # This could be based on the standard deviation, min, max, or a sample of values
                    decimal_places = np.mean([abs(decimal.Decimal(str(v)).as_tuple().exponent) for v in
                                              feature_values.sample(min(100, len(feature_values)))])
                    precision = int(decimal_places)
                else:
                    # Default or handle other types as necessary
                    precision = 2  # Default to 2 decimal places for non-numeric types or as a fallback

                random_change = round(random_change, precision)

                tmp_instance.at[tmp_instance.index[0], feature_name] = random_change

            # Check if prediction stays the same
            # if model.predict(tmp_instance)[0] == model.predict(original_instance)[0]:
            result_instance = tmp_instance.copy()
            changed_features += 1
            if changed_features == max_features_to_vary:
                break
        return result_instance
