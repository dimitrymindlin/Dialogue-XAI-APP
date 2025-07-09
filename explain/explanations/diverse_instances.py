import os
import time
from typing import List, Dict, Union
import pickle as pkl
import gin
import pandas as pd
from create_experiment_data.feature_difference_utils import (
    is_meaningful_change,
    count_feature_differences
)


def load_cache(cache_location: str):
    """Loads the cache."""
    if os.path.isfile(cache_location):
        with open(cache_location, 'rb') as file:
            cache = pkl.load(file)
    else:
        cache = {}
    return cache


def _convert_legacy_cache(cache):
    """Convert old list format to new dict format for backward compatibility."""
    if isinstance(cache, list):
        print("Converting legacy diverse instances format (list) to new cluster-based format (dict)")
        return {0: cache}
    return cache


@gin.configurable
class DiverseInstances:
    """This class finds DiverseInstances by random or submodular pick."""

    def __init__(self,
                 cache_location: str = "./cache/diverse-instances.pkl",
                 dataset_name: str = "german",
                 instance_amount: int = 5,
                 feature_explainer=None,
                 categorical_features: List[str] = None,
                 actionable_features: List[str] = None,
                 n_clusters=None):
        cached_data = load_cache(cache_location)
        self.diverse_instances = _convert_legacy_cache(cached_data)
        self.cache_location = cache_location
        self.feature_explainer = feature_explainer
        self.instance_amount = instance_amount
        self.dataset_name = dataset_name
        self.categorical_features = categorical_features or []
        self.actionable_features = actionable_features
        self.feature_thresholds = None  # Will be set when data is available
        self.n_clusters = n_clusters

    def get_all_instance_ids(self) -> List[int]:
        all_instances = []
        
        # Collect all cluster lists, handling scalars
        cluster_lists = []
        for cluster_id, cluster_instances in self.diverse_instances.items():
            if isinstance(cluster_instances, list):
                cluster_lists.append(cluster_instances)
            else:
                # Handle case where cluster_instances is a scalar (corrupted cache)
                cluster_lists.append([cluster_instances])
        
        # If we have multiple clusters, interleave them using zip logic
        if len(cluster_lists) > 1:
            from itertools import zip_longest
            # Interleave instances from all clusters
            for instances_tuple in zip_longest(*cluster_lists, fillvalue=None):
                for instance in instances_tuple:
                    if instance is not None:
                        all_instances.append(instance)
        elif len(cluster_lists) == 1:
            # Single cluster, just extend normally
            all_instances.extend(cluster_lists[0])
        
        return all_instances

    def _meaningful_feature_difference_count(self, row1, row2, data):
        return count_feature_differences(
            row1, row2, self.actionable_features or data.columns, self.feature_thresholds, self.categorical_features or []
        )

    def _is_meaningful_change(self, feature_name, val1, val2, data):
        return is_meaningful_change(
            feature_name, val1, val2, self.feature_thresholds, self.categorical_features or []
        )

    def _find_neighbors(self, data, seed_idx, used_indices, max_neighbors, min_feature_diff=2, max_feature_diff=3):
        neighbors = []
        seed_row = data.loc[seed_idx]
        for idx, row in data.iterrows():
            if idx == seed_idx or idx in used_indices:
                continue
            diff_count = self._meaningful_feature_difference_count(seed_row, row, data)
            if min_feature_diff <= diff_count <= max_feature_diff:
                neighbors.append(idx)
            if len(neighbors) >= max_neighbors:
                break
        return neighbors

    def get_instance_ids_to_show(self,
                                 data: pd.DataFrame,
                                 model,
                                 y_values: List[int],
                                 save_to_cache=True,
                                 submodular_pick=False) -> Union[List[int], Dict[int, List[int]]]:
        """
        Returns diverse instances for the given data set using random or submodular pick.
        """
        if len(self.diverse_instances) > 0:
            return self.get_all_instance_ids()

        if not self.categorical_features:
            self.categorical_features = self._auto_detect_categorical_features(data)
        
        # Calculate feature thresholds if not already set
        if self.feature_thresholds is None:
            from create_experiment_data.feature_difference_utils import calculate_feature_thresholds
            self.feature_thresholds = calculate_feature_thresholds(data, self.categorical_features)

        if submodular_pick:
            if not self.n_clusters:
                raise ValueError("DiverseInstances.n_clusters must be specified in config for submodular pick.")
            diverse_instances = self.feature_explainer.get_diverse_instance_ids(data.values, self.n_clusters)
            # Convert all indices to Python integers to avoid numpy.int64 issues
            diverse_instances = [int(data.index[i]) for i in diverse_instances]
            print(f"Found {len(diverse_instances)} diverse instances using submodular pick.")
            # Assign each seed to a cluster (round robin if not provided)
            cluster_to_seeds = {i: [diverse_instances[i]] for i in range(len(diverse_instances))}
            used_indices = set(diverse_instances)
            # If we need more, fill by alternating between clusters
            cluster_indices = list(cluster_to_seeds.keys())
            cluster_ptr = 0
            while len(used_indices) < self.instance_amount:
                cluster = cluster_indices[cluster_ptr % len(cluster_indices)]
                seed_idx = cluster_to_seeds[cluster][0]
                neighbors = self._find_neighbors(data, seed_idx, used_indices, max_neighbors=1)
                if neighbors:
                    neighbor_idx = neighbors[0]
                    cluster_to_seeds[cluster].append(neighbor_idx)
                    used_indices.add(neighbor_idx)
                cluster_ptr += 1
                # If no more neighbors can be found for any cluster, break
                if cluster_ptr > len(cluster_indices) * self.instance_amount:
                    break
            # Preserve cluster structure instead of flattening
            self.diverse_instances = cluster_to_seeds

        else:
            # Random selection
            dynamic_seed = int(time.time() * 1e6) % (2 ** 32 - 1)
            get_instances_amount = min(self.instance_amount * 50, len(data))
            diverse_instances_pandas_indices = data.sample(get_instances_amount, random_state=dynamic_seed).index.tolist()
            # Optionally filter by class balance or other logic here if needed
            self.diverse_instances[0] = diverse_instances_pandas_indices[:self.instance_amount]

        if save_to_cache:
            self.save_diverse_instances(self.diverse_instances)
        return self.get_all_instance_ids()

    def save_diverse_instances(self, diverse_instances):
        with open(self.cache_location, 'wb') as file:
            pkl.dump(diverse_instances, file)

    def _auto_detect_categorical_features(self, data: pd.DataFrame) -> List[str]:
        categorical_features = []
        dataset_categoricals = {
            "adult": ["workclass", "education", "marital-status", "occupation",
                      "relationship", "race", "sex", "native-country"],
            "german": ["Status", "Duration", "Credit_history", "Purpose",
                       "Savings", "Employment", "Personal_status", "Other_parties",
                       "Property_magnitude", "Other_payment_plans", "Housing",
                       "Job", "Telephone", "foreign_worker"],
            "diabetes": [],
            "compas": ["sex", "race", "c_charge_degree", "score_text"],
            "titanic": ["Sex", "Embarked", "Pclass"]
        }
        if self.dataset_name in dataset_categoricals:
            categorical_features.extend(dataset_categoricals[self.dataset_name])
        for col in data.columns:
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                if col not in categorical_features:
                    categorical_features.append(col)
        return [col for col in categorical_features if col in data.columns]
