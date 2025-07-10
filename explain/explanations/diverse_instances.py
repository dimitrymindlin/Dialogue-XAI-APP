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
        With submodular pick, enforces 0.5 class balance and uses 2*n_clusters seeds.
        """
        if len(self.diverse_instances) > 0:
            return self.get_all_instance_ids()

        self._prepare_features(data)

        if submodular_pick:
            self._create_submodular_instances(data, model)
        else:
            self._create_random_instances(data, model)

        # Validation checks to ensure selection criteria are met
        self._validate_selection_criteria(data, model, submodular_pick)
        
        if save_to_cache:
            self.save_diverse_instances(self.diverse_instances)
        return self.get_all_instance_ids()

    def _prepare_features(self, data: pd.DataFrame):
        """Prepare categorical features and feature thresholds."""
        if not self.categorical_features:
            self.categorical_features = self._auto_detect_categorical_features(data)
        
        if self.feature_thresholds is None:
            from create_experiment_data.feature_difference_utils import calculate_feature_thresholds
            self.feature_thresholds = calculate_feature_thresholds(data, self.categorical_features)

    def _create_submodular_instances(self, data: pd.DataFrame, model):
        """Create diverse instances using SP-LIME with enforced class balance."""
        if not self.n_clusters:
            raise ValueError("DiverseInstances.n_clusters must be specified in config for submodular pick.")
        
        # Get diverse seeds from SP-LIME
        diverse_instances = self._get_diverse_seeds(data, model)
        
        # Select balanced seeds
        selected_seeds = self._select_balanced_seeds(data, model, diverse_instances)
        
        # Build clusters with neighbors
        self._build_balanced_clusters(data, model, selected_seeds)

    def _get_diverse_seeds(self, data: pd.DataFrame, model):
        """Get diverse instance seeds from SP-LIME."""
        target_seeds = 2 * self.n_clusters
        print(f"Using SP-LIME with SHAP to find {target_seeds} seeds for {self.n_clusters} clusters")
        print(f"Dataset size: {len(data)}, Target seeds: {target_seeds}")
        
        diverse_instance_indices = self.feature_explainer.get_diverse_instance_ids(data.values, num_instances=target_seeds)
        print(f"SP-LIME returned {len(diverse_instance_indices)} diverse indices (max target: {target_seeds})")
        
        # Convert array indices to DataFrame indices
        diverse_instances = [data.index[i] for i in diverse_instance_indices if i < len(data)]
        print(f"Found {len(diverse_instances)} diverse instances using submodular pick.")
        
        # Check minimum requirements
        if len(diverse_instances) < self.n_clusters:
            raise ValueError(
                f"Insufficient diverse instances for {self.n_clusters} clusters. "
                f"SP-LIME found only {len(diverse_instances)} diverse instances, need at least {self.n_clusters}. "
                f"Consider reducing n_clusters or using a different dataset."
            )
        
        return diverse_instances

    def _select_balanced_seeds(self, data: pd.DataFrame, model, diverse_instances: List[int]):
        """Select n_clusters seeds with enforced 0.5 class balance."""
        predictions = self._get_class_predictions(diverse_instances, data, model)
        unique_classes = sorted(set(predictions))
        print(f"Available classes in predictions: {unique_classes}")
        
        # Calculate seeds per class for balance
        target_distribution = self._calculate_balanced_distribution(self.n_clusters, len(unique_classes))
        
        # Group seeds by class
        seeds_by_class = {cls: [] for cls in unique_classes}
        for idx, pred_class in zip(diverse_instances, predictions):
            seeds_by_class[pred_class].append(idx)
        
        # Validate class balance is possible
        for i, cls in enumerate(unique_classes):
            min_needed = target_distribution[i]
            if len(seeds_by_class[cls]) < min_needed:
                raise ValueError(
                    f"Cannot achieve 0.5 class balance for {self.n_clusters} clusters. "
                    f"Class {cls} has only {len(seeds_by_class[cls])} diverse instances, need at least {min_needed}. "
                    f"Available per class: {[(cls, len(instances)) for cls, instances in seeds_by_class.items()]}."
                )
        
        # Select balanced seeds
        selected_seeds = []
        for i, cls in enumerate(unique_classes):
            target_count = target_distribution[i]
            selected_seeds.extend(seeds_by_class[cls][:target_count])
        
        selected_seeds = selected_seeds[:self.n_clusters]
        print(f"Selected {len(selected_seeds)} seeds with enforced 0.5 class balance")
        
        # Log seed distribution
        class_counts = self._calculate_class_distribution(selected_seeds, data, model)
        print(f"Class distribution in selected seeds: {class_counts}")
        
        return selected_seeds

    def _build_balanced_clusters(self, data: pd.DataFrame, model, selected_seeds: List[int]):
        """Build clusters with neighbors while maintaining overall class balance."""
        seed_predictions = self._get_class_predictions(selected_seeds, data, model)
        unique_classes = sorted(set(seed_predictions))
        
        # Calculate target instances per class for perfect balance
        target_per_class = self._calculate_balanced_distribution(self.instance_amount, len(unique_classes))
        # Convert to class-indexed dictionary
        class_targets = {unique_classes[i]: target for i, target in target_per_class.items()}
        print(f"Target instances per class for 0.5 balance: {class_targets}")
        
        # Initialize clusters and group by class
        cluster_to_seeds = {i: [selected_seeds[i]] for i in range(len(selected_seeds))}
        clusters_by_class = {}
        for cluster_id, (cluster_seeds, seed_class) in enumerate(zip(cluster_to_seeds.values(), seed_predictions)):
            clusters_by_class.setdefault(seed_class, []).append(cluster_id)
        
        used_indices = set(selected_seeds)
        
        # Build clusters maintaining class balance
        for cls in unique_classes:
            self._expand_class_clusters(data, model, cls, clusters_by_class[cls], 
                                      cluster_to_seeds, class_targets[cls], used_indices)
        
        # Show results
        total_instances = sum(len(cluster_seeds) for cluster_seeds in cluster_to_seeds.values())
        print(f"Total instances across all clusters: {total_instances} (target: {self.instance_amount})")
        
        # Final class distribution
        all_instances = [idx for cluster_seeds in cluster_to_seeds.values() for idx in cluster_seeds]
        final_class_counts = self._calculate_class_distribution(all_instances, data, model)
        print(f"Final class distribution: {final_class_counts}")
        
        self.diverse_instances = cluster_to_seeds

    def _expand_class_clusters(self, data: pd.DataFrame, model, target_class: int, 
                             class_cluster_ids: List[int], cluster_to_seeds: Dict, 
                             class_target_total: int, used_indices: set):
        """Expand clusters of a specific class to reach target total instances."""
        base_size = class_target_total // len(class_cluster_ids)
        extra = class_target_total % len(class_cluster_ids)
        
        for i, cluster_id in enumerate(class_cluster_ids):
            cluster_target_size = base_size + (1 if i < extra else 0)
            cluster_seeds = cluster_to_seeds[cluster_id]
            seed_idx = cluster_seeds[0]
            
            attempts = 0
            max_attempts = len(data) // 2
            
            while len(cluster_seeds) < cluster_target_size and attempts < max_attempts:
                neighbors = self._find_neighbors(data, seed_idx, used_indices, max_neighbors=5)
                added_any = False
                
                for neighbor_idx in neighbors:
                    if len(cluster_seeds) >= cluster_target_size:
                        break
                    
                    neighbor_class = model.predict(data.loc[[neighbor_idx]])[0]
                    if neighbor_class == target_class:
                        cluster_seeds.append(neighbor_idx)
                        used_indices.add(neighbor_idx)
                        added_any = True
                
                if not added_any:
                    break
                attempts += 1
            
            print(f"Cluster {cluster_id}: {len(cluster_seeds)} instances (target: {cluster_target_size}, seed class: {target_class})")

    def _create_random_instances(self, data: pd.DataFrame, model):
        """Create random instances with enforced class balance."""
        dynamic_seed = int(time.time() * 1e6) % (2 ** 32 - 1)
        get_instances_amount = min(self.instance_amount * 50, len(data))
        sampled_data = data.sample(get_instances_amount, random_state=dynamic_seed)
        
        predictions = model.predict(sampled_data)
        unique_classes = sorted(set(predictions))
        
        # Calculate balanced distribution
        target_distribution = self._calculate_balanced_distribution(self.instance_amount, len(unique_classes))
        
        balanced_instances = []
        for i, cls in enumerate(unique_classes):
            class_indices = sampled_data[predictions == cls].index.tolist()
            target_count = target_distribution[i]
            balanced_instances.extend(class_indices[:target_count])
        
        # Fill remaining slots if needed
        while len(balanced_instances) < self.instance_amount:
            for cls in unique_classes:
                if len(balanced_instances) >= self.instance_amount:
                    break
                class_indices = sampled_data[predictions == cls].index.tolist()
                remaining_of_class = [idx for idx in class_indices if idx not in balanced_instances]
                if remaining_of_class:
                    balanced_instances.append(remaining_of_class[0])
        
        balanced_instances = balanced_instances[:self.instance_amount]
        print(f"Random selection with enforced 0.5 class balance: {len(balanced_instances)} instances")
        
        self.diverse_instances[0] = balanced_instances

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

    def _validate_selection_criteria(self, data: pd.DataFrame, model, submodular_pick: bool):
        """
        Validates that the selected diverse instances meet the specified criteria.
        Throws detailed errors if validation fails.
        """
        all_instances = self.get_all_instance_ids()
        
        # Basic validation: Check if we have instances
        if not all_instances:
            raise ValueError("No diverse instances were selected")
        
        # Get predictions for all selected instances
        class_counts = self._calculate_class_distribution(all_instances, data, model)
        unique_classes = sorted(class_counts.keys())
        
        print(f"Validation: Selected {len(all_instances)} total instances")
        print(f"Validation: Class distribution: {class_counts}")
        
        if submodular_pick and self.n_clusters:
            # Submodular pick specific validation
            
            # 1. Check if we have the correct number of clusters
            if len(self.diverse_instances) != self.n_clusters:
                raise ValueError(
                    f"Expected {self.n_clusters} clusters but got {len(self.diverse_instances)}. "
                    f"Clusters found: {list(self.diverse_instances.keys())}"
                )
            
            # 2. Check class balance (should be as close to 0.5 as possible)
            if len(unique_classes) == 2:
                class_ratio = min(class_counts.values()) / max(class_counts.values())
                if class_ratio < 0.4:  # Allow some tolerance for small numbers
                    raise ValueError(
                        f"Class balance validation failed. Expected ~0.5 ratio, got {class_ratio:.3f}. "
                        f"Class distribution: {class_counts}"
                    )
            
            # 3. Validate each cluster maintains same class prediction
            for cluster_id, cluster_instances in self.diverse_instances.items():
                if not cluster_instances:
                    raise ValueError(f"Cluster {cluster_id} is empty")
                
                cluster_predictions = model.predict(data.loc[cluster_instances])
                cluster_unique_classes = set(cluster_predictions)
                
                if len(cluster_unique_classes) > 1:
                    cluster_class_counts = {cls: list(cluster_predictions).count(cls) for cls in cluster_unique_classes}
                    raise ValueError(
                        f"Cluster {cluster_id} validation failed: Contains multiple classes {cluster_class_counts}. "
                        f"Each cluster should contain instances with the same predicted class."
                    )
                
                print(f"Validation: Cluster {cluster_id} - {len(cluster_instances)} instances, class {cluster_predictions[0]}")
            
            # 4. Check for meaningful feature differences within clusters
            for cluster_id, cluster_instances in self.diverse_instances.items():
                if len(cluster_instances) > 1:
                    seed_idx = cluster_instances[0]
                    seed_row = data.loc[seed_idx]
                    
                    for neighbor_idx in cluster_instances[1:]:
                        neighbor_row = data.loc[neighbor_idx]
                        diff_count = self._meaningful_feature_difference_count(seed_row, neighbor_row, data)
                        
                        if diff_count < 2:
                            print(f"Warning: Cluster {cluster_id} - Instance {neighbor_idx} has only {diff_count} meaningful differences from seed {seed_idx}")
                        elif diff_count > 3:
                            print(f"Warning: Cluster {cluster_id} - Instance {neighbor_idx} has {diff_count} meaningful differences from seed {seed_idx} (more than expected)")
        
        else:
            # Random selection validation
            
            # Check class balance for random selection
            if len(unique_classes) == 2:
                class_ratio = min(class_counts.values()) / max(class_counts.values())
                if class_ratio < 0.4:  # Allow some tolerance
                    raise ValueError(
                        f"Random selection class balance validation failed. Expected ~0.5 ratio, got {class_ratio:.3f}. "
                        f"Class distribution: {class_counts}"
                    )
        
        print("✓ All validation checks passed!")

    def _calculate_class_distribution(self, instances: List[int], data: pd.DataFrame, model) -> Dict[int, int]:
        """Calculate class distribution for given instances."""
        predictions = model.predict(data.loc[instances])
        unique_classes = sorted(set(predictions))
        return {cls: list(predictions).count(cls) for cls in unique_classes}

    def _calculate_balanced_distribution(self, total_amount: int, num_classes: int) -> Dict[int, int]:
        """Calculate target distribution for perfect class balance."""
        instances_per_class = total_amount // num_classes
        remaining_instances = total_amount % num_classes
        return {
            i: instances_per_class + (1 if i < remaining_instances else 0)
            for i in range(num_classes)
        }

    def _get_class_predictions(self, instances: List[int], data: pd.DataFrame, model) -> List[int]:
        """Get class predictions for instances, cached to avoid repeated calls."""
        return model.predict(data.loc[instances])
