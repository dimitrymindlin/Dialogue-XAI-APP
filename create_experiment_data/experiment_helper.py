import decimal
from typing import List

import gin
import numpy as np
import pandas as pd
import copy

from create_experiment_data.instance_datapoint import InstanceDatapoint


@gin.configurable
class ExperimentHelper:
    def __init__(self,
                 actionable_features=None,
                 conversation=None,
                 categorical_mapping=None,
                 categorical_features=None,
                 template_manager=None):
        self.conversation = conversation
        self.categorical_mapping = categorical_mapping
        self.categorical_features = categorical_features
        self.template_manager = template_manager
        self.instances = {"train": [], "test": {}}
        self.current_instance = None
        self.current_instance_type = None

        # Get feature ordering from column_details to preserve the order specified in config
        if hasattr(template_manager.feature_display_names,
                   'config') and 'column_details' in template_manager.feature_display_names.config:
            # Extract display names in the order they appear in column_details
            self.feature_ordering = []

            for item in template_manager.feature_display_names.config['column_details']:
                if isinstance(item, list) and len(item) >= 2:
                    orig_name = item[0]
                    if isinstance(item[1], list) and len(item[1]) > 0:
                        display_name = item[1][0]  # First item is display name
                        self.feature_ordering.append(display_name)
                    else:
                        # Fallback to original name
                        self.feature_ordering.append(orig_name)
        else:
            # Fallback to existing behavior
            self.feature_ordering = list(template_manager.feature_display_names.feature_name_to_display_name.values())

        self.actionable_features = actionable_features

        # Cluster-related attributes
        self.cluster_data = {}  # Will store the original cluster structure
        self.use_clusters = False  # Flag to indicate if we're using cluster-based selection

    def load_instances(self):
        self._load_train_instances()
        self._load_test_instances()

    def _load_train_instances(self):
        import random
        diverse_instances = self.conversation.get_var("diverse_instances").contents

        # Handle both new dictionary format and legacy list format
        if isinstance(diverse_instances, dict):
            # New format: Dict[int, List[int]] - use cluster-aware selection
            self.cluster_data = diverse_instances
            self.use_clusters = True
            # Create a round-robin ordered list from clusters
            instanc_dicts = self._create_cluster_round_robin_order(diverse_instances)
        else:
            # Legacy format: List[int] - no clusters
            self.cluster_data = {0: diverse_instances}  # Put all in cluster 0
            self.use_clusters = False
            instanc_dicts = diverse_instances

        # Turn into InstanceDatapoint
        train_instances = [self._turn_to_instance_datapoint(instance) for instance in instanc_dicts]
        random.shuffle(train_instances)
        self.instances["train"] = train_instances

    def _load_test_instances(self):
        test_instance_dict = self.conversation.get_var("test_instances").contents

        # turn test_instance_dict into InstanceDatapoint
        new_test_instance_dict = {}
        for train_id in test_instance_dict.keys():
            new_test_instance_dict[train_id] = {}  # Initialize the nested dict for this train_id
            for phase, test_instance_df in test_instance_dict[train_id].items():
                # Turn df to dict
                if isinstance(test_instance_df, pd.DataFrame):
                    test_instance = test_instance_df.to_dict('records')[0]
                else:
                    test_instance = test_instance_df
                # Process each test instance
                test_instance["id"] = train_id  # Ensure the ID is set correctly
                processed_test_instances = self._turn_to_instance_datapoint(test_instance, instance_type=phase)
                new_test_instance_dict[train_id][phase] = processed_test_instances

        self.instances["test"] = new_test_instance_dict

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

    def _make_displayable_instance(self, instance, return_probabilities=False):
        # Round instance features
        if isinstance(instance, InstanceDatapoint):
            instance_features = copy.deepcopy(instance.instance_as_dict)
        else:
            instance_features = copy.deepcopy(instance)
        instance_features = self.template_manager.apply_categorical_mapping(instance_features)
        self._round_instance_features(instance_features)

        # Turn instance features to display names
        instance.displayable_features = instance_features  # Copy categorical feature values first
        instance.displayable_features = self.template_manager.replace_feature_names_by_display_names(
            instance.displayable_features)  # then add display names

        # Order instance features and values according to the feature ordering
        if self.feature_ordering is not None:
            # Make a new dict with only the features that exist in the instance AND in the feature_ordering
            # This ensures no duplicates and proper ordering
            ordered_features = {}
            seen_features = set()  # Keep track of features we've already processed

            for feature in self.feature_ordering:
                if feature in instance.displayable_features and feature not in seen_features:
                    ordered_features[feature] = instance.displayable_features[feature]
                    seen_features.add(feature)
                elif feature not in instance.displayable_features:
                    print(f"Warning: Feature '{feature}' from feature_ordering not found in instance")

            # Only add features that weren't in the ordering if needed
            for feature, value in instance.displayable_features.items():
                if feature not in seen_features:
                    ordered_features[feature] = value
                    seen_features.add(feature)

            instance.displayable_features = ordered_features
        else:  # alphabetically order features
            instance.displayable_features = dict(sorted(instance.displayable_features.items()))

        # Make sure all values are strings
        self._convert_values_to_string(instance.displayable_features)
        return instance

    def get_next_instance(self, instance_type, datapoint_count, return_probability=False):
        # Make sure train instances are loaded before any other type
        if len(self.instances.get("train", [])) == 0:
            self.load_instances()

        old_instance = None
        
        if instance_type == "train":
            instance = self._get_training_instance(datapoint_count)
        else:
            # For all test types, get the train instance ID first
            train_instance_id = self.instances["train"][datapoint_count].instance_id
            # Use the corresponding train instance as the baseline for comparison
            old_instance = self.instances["train"][datapoint_count]
            
            # Map instance types to phase names in the test dictionary
            phase_mapping = {
                "test": "test", 
                "final-test": "final-test",
                "intro-test": "intro_test"
            }
            phase = phase_mapping.get(instance_type, instance_type)
            
            # Directly get the instance from the nested dict structure
            instance = self.instances["test"][train_instance_id][phase]

        self.current_instance_type = instance_type

        # Compare with old instance to track changes (except for final/intro tests)
        if old_instance and instance_type not in ["final-test", "intro-test"]:
            for key, value in instance.instance_as_dict.items():
                if value != old_instance.instance_as_dict[key]:
                    instance.instance_as_dict[key] = {"old": old_instance.instance_as_dict[key], "current": value}

        try:
            instance = self._make_displayable_instance(instance)
        except Exception as e:
            print(f"Error in making instance displayable: {e}, using normal instance")

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

    def _turn_to_instance_datapoint(self, instance, instance_type='train'):
        if 'values' in instance:
            instance_values = instance['values'].copy()
        else:
            instance_values = {k: v for k, v in instance.items() if k not in ('id', 'label')}

        model_prediction = \
            self.conversation.get_var("model_prob_predict").contents(pd.DataFrame(instance_values, index=[0]))[0]
        # true_label = self._fetch_true_label(instance['id'])

        # Round instance values before creating InstanceDatapoint
        self._round_instance_features(instance_values)
        # Turn to InstanceDatapoint
        instance = InstanceDatapoint(instance_id=instance['id'],
                                     instance_as_dict=instance_values,
                                     class_probabilities=model_prediction,
                                     model_predicted_label_string=self.conversation.class_names[
                                         np.argmax(model_prediction)],
                                     model_predicted_label=np.argmax(model_prediction),
                                     instance_type=instance_type)
        return instance

    def _process_test_instances(self, train_datapoint_id, instances_dict):
        # Turn to InstanceDatapoint
        instances_dict_new = {}
        instance_dicts = {comp: pd.DataFrame(data).to_dict('records')[0] for comp, data in instances_dict.items()}
        for instance_naming, instance_dict in instance_dicts.items():
            if "label" in instance_dict:
                del instance_dict["label"]

            # Round instance_dict values before creating InstanceDatapoint
            self._round_instance_features(instance_dict)

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

    def _get_training_instance(self, datapoint_count, instance_type="train"):
        if not self.instances[instance_type]:
            return None
        return self.instances[instance_type][datapoint_count]


    def _round_instance_features(self, features):
        for feature, value in features.items():
            # Handle various numeric types including numpy types
            if isinstance(value, (float, np.floating)):
                features[feature] = round(float(value), self.conversation.rounding_precision)
            elif isinstance(value, (np.integer)):
                # Convert numpy integers to regular integers
                features[feature] = int(value)
            elif isinstance(value, dict):
                # Handle nested dictionaries (for 'current' and 'old' values)
                for inner_key, inner_value in value.items():
                    if isinstance(inner_value, (float, np.floating)):
                        value[inner_key] = round(float(inner_value), self.conversation.rounding_precision)
                    elif isinstance(inner_value, (np.integer)):
                        value[inner_key] = int(inner_value)

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

        # Use the same flow as regular counterfactual explanations to ensure feature limits are applied
        ids = list(original_instance.index)
        final_cfes, final_cfe_ids, desired_class = dice_tabular.get_final_cfes(
            original_instance,
            ids,
            ids_to_regenerate=None,
            save_to_cache=False
        )

        # Log feature changes for each counterfactual
        for idx, (cfe_id, row) in enumerate(final_cfes.iterrows()):
            changed_features = []
            for feature in row.index:
                if feature == "y":
                    continue
                if row[feature] != original_instance.loc[ids[0]][feature]:
                    changed_features.append(feature)
        # drop y column if it exists
        if "y" in final_cfes.columns:
            final_cfes = final_cfes.drop(columns=["y"])

        return final_cfes

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

    def _create_cluster_round_robin_order(self, cluster_data):
        """
        Create a round-robin ordering of instances across clusters.
        This ensures balanced representation from all clusters.
        
        Args:
            cluster_data: Dict[int, List[int]] mapping cluster IDs to instance lists
            
        Returns:
            List[int]: Ordered list of instance IDs with round-robin cluster selection
        """
        if not cluster_data:
            return []

        # Get all cluster instance lists
        cluster_lists = list(cluster_data.values())
        max_cluster_size = max(len(cluster_list) for cluster_list in cluster_lists)

        round_robin_instances = []

        # Round-robin through clusters
        for i in range(max_cluster_size):
            for cluster_list in cluster_lists:
                if i < len(cluster_list):  # If this cluster still has instances at this position
                    round_robin_instances.append(cluster_list[i])

        if self.use_clusters:
            # Log the cluster distribution in the final order
            cluster_positions = {}
            for idx, instance_id in enumerate(round_robin_instances):
                for cluster_id, cluster_instances in cluster_data.items():
                    if instance_id in cluster_instances:
                        if cluster_id not in cluster_positions:
                            cluster_positions[cluster_id] = []
                        cluster_positions[cluster_id].append(idx)
                        break
        return round_robin_instances

    def get_current_instance_cluster_info(self):
        """
        Get cluster information for the current instance.
        
        Returns:
            dict: Contains cluster_id, cluster_size, position_in_cluster, total_clusters
        """
        if not self.use_clusters or not self.current_instance:
            return {"cluster_id": 0, "cluster_size": len(self.instances["train"]),
                    "position_in_cluster": 0, "total_clusters": 1}

        instance_id = self.current_instance.instance_id

        # Find which cluster this instance belongs to
        for cluster_id, cluster_instances in self.cluster_data.items():
            if instance_id in cluster_instances:
                position_in_cluster = cluster_instances.index(instance_id)
                return {
                    "cluster_id": cluster_id,
                    "cluster_size": len(cluster_instances),
                    "position_in_cluster": position_in_cluster,
                    "total_clusters": len(self.cluster_data)
                }

        # Fallback if not found
        return {"cluster_id": -1, "cluster_size": 0, "position_in_cluster": -1,
                "total_clusters": len(self.cluster_data)}

    def get_cluster_summary(self):
        """
        Get a summary of all clusters.
        
        Returns:
            dict: Summary of cluster information
        """
        if not self.use_clusters:
            return {"total_instances": len(self.instances["train"]), "clusters": 1, "cluster_info": {}}

        cluster_info = {}
        total_instances = 0

        for cluster_id, cluster_instances in self.cluster_data.items():
            cluster_info[cluster_id] = {
                "size": len(cluster_instances),
                "instances": cluster_instances
            }
            total_instances += len(cluster_instances)

        return {
            "total_instances": total_instances,
            "clusters": len(self.cluster_data),
            "cluster_info": cluster_info,
            "using_round_robin": True
        }
