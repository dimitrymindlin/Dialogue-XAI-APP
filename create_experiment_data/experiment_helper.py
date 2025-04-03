import decimal
from typing import List

import gin
import numpy as np
import pandas as pd
import copy

from create_experiment_data.instance_datapoint import InstanceDatapoint


@gin.configurable
class ExperimentHelper:
    def __init__(self, conversation,
                 categorical_mapping,
                 categorical_features,
                 template_manager,
                 feature_ordering=None,
                 actionable_features=None):
        self.conversation = conversation
        self.categorical_mapping = categorical_mapping
        self.categorical_features = categorical_features
        self.template_manager = template_manager
        self.instances = {"train": [], "test": {}}
        self.current_instance = None
        self.current_instance_type = None
        self.feature_ordering = feature_ordering
        self.actionable_features = actionable_features

    def load_instances(self):
        self._load_train_instances()
        self._load_test_instances()

    def _load_train_instances(self):
        diverse_instances = self.conversation.get_var("diverse_instances").contents
        # Turn into InstanceDatapoint
        self.instances["train"] = [self._prepare_train_instance(instance) for instance in diverse_instances]

    def _load_test_instances(self):
        test_instances = self.conversation.get_var("test_instances").contents
        self.instances["test"] = {instance_id: self._process_test_instances(instance_id, instances_dict)
                                  for instance_id, instances_dict in test_instances.items()}

    def _convert_values_to_string(self, instance):
        for key, value in instance.items():
            # Check if the value is a dictionary (to handle 'current' and 'old' values)
            if isinstance(value, dict):
                # Iterate through the inner dictionary and convert its values to strings
                for inner_key, inner_value in value.items():
                    # Turn floats to strings, converting to int first if no decimal part
                    if isinstance(inner_value, float) and inner_value.is_integer():
                        inner_value = int(inner_value)
                    value[inner_key] = str(inner_value)
            else:
                # Handle non-dictionary values as before
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                instance[key] = str(value)

    def _make_displayable_instance(self, instance,
                                   return_probabilities=False):
        # Round instance features
        instance_features = copy.deepcopy(instance.instance_as_dict)
        instance_features = self.template_manager.apply_categorical_mapping(instance_features)
        self._round_instance_features(instance_features)

        # Turn instace features to display names
        instance.displayable_features = instance_features  # Copy categorical feature values first
        instance.displayable_features = self.template_manager.replace_feature_names_by_display_names(
            instance.displayable_features)  # then add display names

        # Order instance features and values according to the feature ordering
        if self.feature_ordering is not None:
            # Order instance features according to the feature ordering
            instance.displayable_features = {feature: instance.displayable_features[feature] for feature in
                                             self.feature_ordering}
        else:  # alphabetically order features
            instance.displayable_features = dict(sorted(instance.displayable_features.items()))

        # Make sure all values are strings
        self._convert_values_to_string(instance.displayable_features)
        return instance

    def get_next_instance(self, instance_type, datapoint_count, return_probability=False):
        old_instance = None
        load_instance_methods = {"train": self._load_train_instances, "test": self._load_test_instances}
        get_instance_methods = {
            "train": lambda: self._get_training_instance(return_probability, datapoint_count),
            "test": lambda: self._get_test_instance(
                self.current_instance.instance_id if self.current_instance else None,
                datapoint_count),
            "final-test": lambda: self._get_final_test_instance(datapoint_count),
            "intro-test": lambda: self._get_intro_test_instance(datapoint_count)
        }

        if self.instances.get(instance_type, None) is None:
            self.load_instances()

        if not self.instances.get(instance_type, []):
            load_instance_methods.get(instance_type, lambda: None)()

        if instance_type != "train":
            instance_id, instance = get_instance_methods[instance_type]()
            old_instance = self.current_instance
        else:  # "train"
            instance = get_instance_methods[instance_type]()

        self.current_instance_type = instance_type

        if old_instance and not instance_type in ["final-test", "intro-test"]:
            for key, value in instance.instance_as_dict.items():
                if value != old_instance.instance_as_dict[key]:
                    instance.instance_as_dict[key] = {"old": old_instance.instance_as_dict[key], "current": value}

        instance = self._make_displayable_instance(instance)

        instance.counter = datapoint_count
        instance.instance_type = self.current_instance_type
        self.current_instance = instance
        return instance

    def print_all_labels(self):
        # Make a table with datapoint count as columns and train, test, intro-test and final-test as rows
        # 1. Create empty table
        df = pd.DataFrame(columns=["train", "test", "intro-test", "final-test"])
        # 2. Iterate over all datapoints and get the instance
        for i in range(len(self.instances["train"])):
            train_instance = self.get_next_instance("train", i)
            test_instance = self.get_next_instance("test", i)
            intro_test_instance = self.get_next_instance("intro-test", i)
            final_test_instance = self.get_next_instance("final-test", i)
            # 3. Add the labels to the table
            df.loc[i] = [train_instance.model_predicted_label_string,
                         test_instance.model_predicted_label_string,
                         intro_test_instance.model_predicted_label_string,
                         final_test_instance.model_predicted_label_string]
        print(df)

    def _prepare_train_instance(self, instance):
        # Simplified example of preparing a data instance
        model_prediction = \
            self.conversation.get_var("model_prob_predict").contents(pd.DataFrame(instance['values'], index=[0]))[0]
        # true_label = self._fetch_true_label(instance['id'])
        # Turn to InstanceDatapoint
        instance = InstanceDatapoint(instance_id=instance['id'],
                                     instance_as_dict=instance['values'],
                                     class_probabilities=model_prediction,
                                     model_predicted_label_string=self.conversation.class_names[
                                         np.argmax(model_prediction)],
                                     model_predicted_label=np.argmax(model_prediction),
                                     instance_type='train')
        return instance

    def _process_test_instances(self, train_datapoint_id, instances_dict):
        # Turn to InstanceDatapoint
        instances_dict_new = {}
        instance_dicts = {comp: pd.DataFrame(data).to_dict('records')[0] for comp, data in instances_dict.items()}
        for instance_naming, instance_dict in instance_dicts.items():
            if "label" in instance_dict:
                del instance_dict["label"]
            class_probabilities = self.conversation.get_var("model_prob_predict").contents(
                pd.DataFrame(instance_dict, index=[0]))
            predicted_label_index = np.argmax(class_probabilities)
            model_predicted_label = self.conversation.class_names[predicted_label_index]
            instances_dict_new[instance_naming] = InstanceDatapoint(instance_id=train_datapoint_id,
                                                                    instance_as_dict=instance_dict,
                                                                    class_probabilities=class_probabilities,
                                                                    model_predicted_label_string=model_predicted_label,
                                                                    model_predicted_label=predicted_label_index,
                                                                    instance_type="test")
        return instances_dict_new

    def _get_training_instance(self, return_probability, datapoint_count, instance_type="train"):
        if not self.instances[instance_type]:
            return None
        self.current_instance = self.instances[instance_type][datapoint_count]
        return self.current_instance

    def _get_test_instance(self, train_instance_id, datapoint_count, instance_type="test"):
        if not self.instances[instance_type]:
            return None

        instance_key = "least_complex_instance" if datapoint_count % 2 == 0 else "easy_counterfactual_instance"
        test_instances_dict = self.instances[instance_type][train_instance_id]
        instance = test_instances_dict[instance_key]
        return train_instance_id, instance

    def _get_final_test_instance(self, datapoint_count, instance_type="final-test"):
        if not self.instances["test"]:
            return None

        instance_key = "most_complex_instance"
        # Get final test instance based on train instance id
        train_instance_id = self.instances["train"][datapoint_count].instance_id
        test_instances_dict = self.instances["test"][train_instance_id]
        instance = test_instances_dict[instance_key]
        return train_instance_id, instance

    def _get_intro_test_instance(self, datapoint_count, instance_type="intro-test"):
        if not self.instances["test"]:
            # Load test instances if not already loaded
            self._load_test_instances()
        if not self.instances["train"]:
            # Load training instances if not already loaded
            self._load_train_instances()

        instance_key = "most_complex_instance"  # Random for now.
        # Get intro test instance based on train instance id
        train_instance_id = self.instances["train"][datapoint_count].instance_id
        test_instances_dict = self.instances["test"][train_instance_id]
        instance = test_instances_dict[instance_key]
        return train_instance_id, instance

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
        cfes = dice_tabular.run_explanation(original_instance, "opposite")
        try:
            original_instance_id = original_instance.index[0]
        except IndexError:
            original_instance_id = 0

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
                try:
                    max_feature_value = len(
                        self.categorical_mapping[original_instance.columns.get_loc(feature_name)])
                except KeyError:
                    raise KeyError(f"Feature {feature_name} is not in the categorical mapping.")
                random_change = np.random.randint(1, max_feature_value)
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

            # After modifying the feature, check if it has actually changed compared to the original_instance
            if not tmp_instance.at[tmp_instance.index[0], feature_name] == original_instance.at[
                original_instance.index[0], feature_name]:
                result_instance = tmp_instance.copy()
                changed_features += 1  # Increment only if the feature has actually changed
            else:
                continue  # Skip to the next feature if no change was detected

            # Removed the redundant increment of changed_features and assignment of result_instance here
            if changed_features == max_features_to_vary:
                break

        return result_instance, changed_features
