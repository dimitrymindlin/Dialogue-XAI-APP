import os
import time
from typing import List, Dict, Union
import pickle as pkl
import gin
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances


def load_cache(cache_location: str):
    """Loads the cache."""
    if os.path.isfile(cache_location):
        with open(cache_location, 'rb') as file:
            cache = pkl.load(file)
    else:
        # Return empty dict for new structure, convert old list format if found
        cache = {}
    return cache


def _convert_legacy_cache(cache):
    """Convert old list format to new dict format for backward compatibility."""
    if isinstance(cache, list):
        print("Converting legacy diverse instances format (list) to new cluster-based format (dict)")
        # Put all instances in cluster 0 for legacy compatibility
        return {0: cache}
    return cache


@gin.configurable
class DiverseInstances:
    """This class finds DiverseInstances by using LIMEs submodular pick."""

    def __init__(self,
                 cache_location: str = "./cache/diverse-instances.pkl",
                 dataset_name: str = "german",
                 instance_amount: int = 5,
                 lime_explainer=None,
                 use_clustering: bool = True,
                 categorical_features: List[str] = None,
                 n_clusters: int = None):
        """

        Args:
            cache_location: location to save the cache
            lime_explainer: lime explainer to use for finding diverse instances (from MegaExplainer)
            use_clustering: whether to use cluster-based selection for diverse instances
            categorical_features: list of categorical feature names for preprocessing
            n_clusters: number of clusters to use (defaults to instance_amount * 2, max data_size // 2)
        """
        cached_data = load_cache(cache_location)
        # Convert legacy format and ensure it's a dict
        self.diverse_instances = _convert_legacy_cache(cached_data)
        
        self.cache_location = cache_location
        self.lime_explainer = lime_explainer
        self.instance_amount = instance_amount
        self.dataset_name = dataset_name
        self.use_clustering = use_clustering
        self.categorical_features = categorical_features or []
        self.n_clusters = n_clusters
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def _preprocess_data_for_clustering(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess data for clustering by encoding categoricals and scaling."""
        processed_data = data.copy()
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in processed_data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    processed_data[col] = self.label_encoders[col].fit_transform(processed_data[col].astype(str))
                else:
                    # Handle unknown categories in transform
                    try:
                        processed_data[col] = self.label_encoders[col].transform(processed_data[col].astype(str))
                    except ValueError:
                        # If there are unknown categories, refit the encoder
                        self.label_encoders[col] = LabelEncoder()
                        processed_data[col] = self.label_encoders[col].fit_transform(processed_data[col].astype(str))
        
        # Handle missing values
        processed_data = processed_data.fillna(processed_data.mean(numeric_only=True))
        
        # Scale all features
        return self.scaler.fit_transform(processed_data)
    
    def set_n_clusters(self, n_clusters: int):
        """
        Set the number of clusters to use for clustering-based diverse instance selection.
        
        Args:
            n_clusters: Number of clusters to use
        """
        self.n_clusters = n_clusters
        print(f"Number of clusters set to: {n_clusters}")

    def get_total_instance_count(self) -> int:
        """Get total number of instances across all clusters."""
        return sum(len(instances) for instances in self.diverse_instances.values())

    def get_cluster_info(self) -> Dict[int, int]:
        """Get information about cluster sizes."""
        return {cluster_id: len(instances) for cluster_id, instances in self.diverse_instances.items()}

    def get_instances_by_cluster(self, cluster_id: int) -> List[int]:
        """Get all instances for a specific cluster."""
        return self.diverse_instances.get(cluster_id, [])

    def get_all_instance_ids(self) -> List[int]:
        """Get all instance IDs as a flat list for backward compatibility."""
        all_instances = []
        for cluster_instances in self.diverse_instances.values():
            all_instances.extend(cluster_instances)
        return all_instances

    def _get_cluster_representatives(self, data: pd.DataFrame, n_clusters: int = None) -> Dict[int, List[int]]:
        """
        Use K-Means clustering to find representative instances from different data regions.
        Selects up to instance_amount instances from each cluster.
        
        Args:
            data: DataFrame to cluster
            n_clusters: Number of clusters (defaults to self.n_clusters or instance_amount)
            
        Returns:
            Dict mapping cluster IDs to lists of instance indices from that cluster
        """
        if len(data) < 2:
            print(f"Data too small ({len(data)} instances), returning all available instances")
            return {0: data.index.tolist()[:self.instance_amount]}
            
        # Use provided n_clusters, or self.n_clusters, or calculate default
        if n_clusters is None:
            if self.n_clusters is not None:
                n_clusters = self.n_clusters
            else:
                # Default to instance_amount if not specified, but cap at data size
                n_clusters = min(self.instance_amount, len(data) // 2, len(data))
        
        # Ensure we don't have more clusters than data points
        n_clusters = min(n_clusters, len(data))
        
        print(f"Attempting to create {n_clusters} clusters from {len(data)} data instances to select {self.instance_amount} instances per cluster")
        print(f"Data index type: {type(data.index)}, length: {len(data.index)}")
        print(f"Data shape: {data.shape}")
        
        try:
            # Preprocess data for clustering
            processed_data = self._preprocess_data_for_clustering(data)
            print(f"Processed data shape: {processed_data.shape}")
            
            # Apply K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(processed_data)
            print(f"Cluster labels shape: {cluster_labels.shape}, unique labels: {np.unique(cluster_labels)}")
            
            # Organize instances by cluster and sort by distance to centroid
            cluster_data_organized = {}
            clusters_with_data = 0
            
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                if not cluster_mask.any():
                    print(f"Warning: Cluster {cluster_id} is empty")
                    continue
                    
                clusters_with_data += 1
                cluster_data = processed_data[cluster_mask]
                cluster_indices = data.index[cluster_mask]
                
                print(f"Cluster {cluster_id}: {len(cluster_indices)} instances (mask sum: {cluster_mask.sum()})")
                
                # Debug: Check if we have the right indices
                print(f"  Cluster indices type: {type(cluster_indices)}, shape: {cluster_indices.shape if hasattr(cluster_indices, 'shape') else 'N/A'}")
                print(f"  First few indices: {cluster_indices[:5].tolist() if len(cluster_indices) > 0 else 'None'}")
                
                # Calculate distances to centroid for all instances in this cluster
                centroid = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
                distances = pairwise_distances(centroid, cluster_data)[0]  # [0] to get the first (and only) row
                
                print(f"  Calculated {len(distances)} distances for {len(cluster_indices)} instances")
                
                # Sort indices by distance to centroid (closest first)
                sorted_indices_distances = sorted(zip(cluster_indices, distances), key=lambda x: x[1])
                sorted_indices = [idx for idx, _ in sorted_indices_distances]
                
                print(f"  Sorted indices length: {len(sorted_indices)}")
                
                cluster_data_organized[cluster_id] = sorted_indices
            
            print(f"Successfully created {clusters_with_data} non-empty clusters")
            
            # Select instance_amount instances from each cluster
            cluster_instance_distribution = {}
            total_instances_selected = 0
            
            for cluster_id in cluster_data_organized.keys():
                cluster_instances = cluster_data_organized[cluster_id]
                
                # Select up to instance_amount instances from this cluster
                instances_to_select = min(self.instance_amount, len(cluster_instances))
                selected_instances = cluster_instances[:instances_to_select]
                
                cluster_instance_distribution[cluster_id] = selected_instances
                total_instances_selected += len(selected_instances)
                
                print(f"Selected {len(selected_instances)} instances from cluster {cluster_id} (out of {len(cluster_instances)} available)")
                
                # Log if we couldn't get the full amount from this cluster
                if len(selected_instances) < self.instance_amount:
                    print(f"Warning: Cluster {cluster_id} only had {len(cluster_instances)} instances, needed {self.instance_amount}")
            
            # Remove empty clusters from result
            cluster_instance_distribution = {k: v for k, v in cluster_instance_distribution.items() if v}
            
            print(f"Final cluster distribution: {[(cluster_id, len(instances)) for cluster_id, instances in cluster_instance_distribution.items()]}")
            print(f"Selected {total_instances_selected} diverse instances total ({self.instance_amount} per cluster target) from {len(cluster_instance_distribution)} clusters")
            
            return cluster_instance_distribution
            
        except Exception as e:
            print(f"Clustering failed: {e}, falling back to random sampling")
            random_indices = data.sample(min(self.instance_amount, len(data)), random_state=42).index.tolist()
            return {0: random_indices}

    def _balance_cluster_selection(self, data: pd.DataFrame, model, cluster_distribution: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Balance cluster-selected instances by class if we have more than needed.
        
        Args:
            data: Original data DataFrame
            model: Trained model for predictions
            cluster_distribution: Dict mapping cluster IDs to instance lists
            
        Returns:
            Balanced cluster distribution
        """
        total_instances = sum(len(instances) for instances in cluster_distribution.values())
        if total_instances <= self.instance_amount:
            return cluster_distribution
            
        try:
            # Flatten all instances for class prediction
            all_indices = []
            cluster_instance_map = {}  # Map instance to cluster
            for cluster_id, instances in cluster_distribution.items():
                all_indices.extend(instances)
                for instance in instances:
                    cluster_instance_map[instance] = cluster_id
            
            # Predict classes for all instances
            predicted_classes = model.predict(data.loc[all_indices])
            
            # Separate by class
            class_0_indices = [idx for idx, pred in zip(all_indices, predicted_classes) if pred == 0]
            class_1_indices = [idx for idx, pred in zip(all_indices, predicted_classes) if pred == 1]
            
            print(f"Class distribution in cluster representatives: Class 0: {len(class_0_indices)}, Class 1: {len(class_1_indices)}")
            
            # Balance classes
            target_per_class = self.instance_amount // 2
            remaining = self.instance_amount % 2
            
            selected_class_0 = class_0_indices[:target_per_class + remaining]
            selected_class_1 = class_1_indices[:target_per_class]
            
            # If one class doesn't have enough, take more from the other
            if len(selected_class_0) < target_per_class + remaining:
                shortage = (target_per_class + remaining) - len(selected_class_0)
                selected_class_1 = class_1_indices[:target_per_class + shortage]
                print(f"Class 0 shortage: taking {shortage} additional from class 1")
            elif len(selected_class_1) < target_per_class:
                shortage = target_per_class - len(selected_class_1)
                selected_class_0 = class_0_indices[:target_per_class + remaining + shortage]
                print(f"Class 1 shortage: taking {shortage} additional from class 0")
            
            final_balanced = (selected_class_0 + selected_class_1)[:self.instance_amount]
            print(f"Balanced cluster selection: {len(final_balanced)} instances")
            
            # Rebuild cluster distribution with balanced instances
            balanced_distribution = {}
            for instance in final_balanced:
                cluster_id = cluster_instance_map[instance]
                if cluster_id not in balanced_distribution:
                    balanced_distribution[cluster_id] = []
                balanced_distribution[cluster_id].append(instance)
            
            return balanced_distribution
            
        except Exception as e:
            print(f"Class balancing failed: {e}, returning original distribution")
            # Take first instance_amount instances while preserving cluster structure
            balanced_distribution = {}
            instances_taken = 0
            for cluster_id, instances in cluster_distribution.items():
                if instances_taken >= self.instance_amount:
                    break
                remaining_needed = self.instance_amount - instances_taken
                instances_to_take = min(len(instances), remaining_needed)
                balanced_distribution[cluster_id] = instances[:instances_to_take]
                instances_taken += instances_to_take
            return balanced_distribution

    def filter_instances_by_class(self, data, model, diverse_instances_pandas_indices,
                                  instance_amount, filter_by_additional_feature=False):
        # Step 1: Predict Classes
        predicted_classes = model.predict(data.loc[diverse_instances_pandas_indices])

        # Step 2: Separate Instances by Predicted Class
        class_0_indices, class_1_indices = [], []
        for i, pred_class in zip(diverse_instances_pandas_indices, predicted_classes):
            if pred_class == 0:
                class_0_indices.append(i)
            else:
                class_1_indices.append(i)

        # Helper function to filter instances based on alternating marital status
        def filter_by_marital_status(indices):
            filtered = []
            last_marital_status = -1
            for i in indices:
                current_marital_status = data.loc[i, "MaritalStatus"]
                if current_marital_status != last_marital_status:
                    filtered.append(i)
                    last_marital_status = current_marital_status
            return filtered

        # Step 3: Filter each class list by marital status
        if filter_by_additional_feature:
            class_0_indices = filter_by_marital_status(class_0_indices)
            class_1_indices = filter_by_marital_status(class_1_indices)

        # Step 4: Balance the classes
        min_length = min(len(class_0_indices), len(class_1_indices))
        balanced_class_0 = np.random.choice(class_0_indices, min_length, replace=False)
        balanced_class_1 = np.random.choice(class_1_indices, min_length, replace=False)

        # Step 5: Shuffle each class list
        np.random.shuffle(balanced_class_0)
        np.random.shuffle(balanced_class_1)

        # Step 6: Take half of the instances from each class
        balanced_class_0 = balanced_class_0[:int(instance_amount / 2)]
        balanced_class_1 = balanced_class_1[:int(instance_amount / 2)]

        # Combine the lists (ensured that the final list has the desired length and class balance)
        combined_instances = np.concatenate((balanced_class_0, balanced_class_1))
        # Shuffle
        np.random.shuffle(combined_instances)
        final_instances = combined_instances[:instance_amount].tolist()

        return final_instances

    def get_instance_ids_to_show(self,
                                 data: pd.DataFrame,
                                 model,
                                 y_values: List[int],
                                 save_to_cache=True,
                                 submodular_pick=False) -> Union[List[int], Dict[int, List[int]]]:
        """
        Returns diverse instances for the given data set.
        Args:
            data: pd.Dataframe the data instances to use to find diverse instances
            save_to_cache: whether to save the diverse instances to the cache
        Returns: Dict mapping cluster IDs to instance lists (new format) or List of instance IDs (backward compatibility)

        """
        if len(self.diverse_instances) > 0:
            # For backward compatibility, return list if requested by legacy code
            if not self.use_clustering:
                return self.get_all_instance_ids()
            return self.diverse_instances

        # Auto-detect categorical features if not provided
        if not self.categorical_features:
            self.categorical_features = self._auto_detect_categorical_features(data)
            if self.use_clustering:
                print(f"Auto-detected categorical features: {self.categorical_features}")

        counter = 0
        
        # For clustering mode, we want instance_amount per cluster, so skip the total count check
        # For non-clustering mode, we want exactly instance_amount total instances
        while (self.use_clustering and len(self.diverse_instances) == 0) or \
              (not self.use_clustering and self.get_total_instance_count() < self.instance_amount):

            # Generate diverse instances
            if submodular_pick:
                print(f"Using submodular pick to find {self.instance_amount} diverse instances.")
                diverse_instances = self.lime_explainer.get_diverse_instance_ids(data.values,
                                                                                 int(self.instance_amount / 10))
                if self.dataset_name == "adult":
                    filter_by_additional_feature = True
                else:
                    filter_by_additional_feature = False
                diverse_instances_pandas_indices = self.filter_instances_by_class(data,
                                                                                  model,
                                                                                  diverse_instances,
                                                                                  self.instance_amount,
                                                                                  filter_by_additional_feature=filter_by_additional_feature)
                
                # Get pandas index for the diverse instances
                diverse_instances_pandas_indices = [data.index[i] for i in diverse_instances_pandas_indices]
                
                # For submodular pick, put all instances in cluster 0
                self.diverse_instances[0] = diverse_instances_pandas_indices
                
            else:
                if self.use_clustering:
                    # Get cluster-based representative instances
                    print(f"Using clustering to find {self.instance_amount} diverse instances per cluster.")
                    if self.n_clusters:
                        print(f"Using user-specified number of clusters: {self.n_clusters}")
                    cluster_distribution = self._get_cluster_representatives(data)
                    
                    # Store the cluster distribution (no balancing needed since we want instance_amount per cluster)
                    total_instances = sum(len(instances) for instances in cluster_distribution.values())
                    print(f"Generated {total_instances} total instances ({self.instance_amount} per cluster target) across {len(cluster_distribution)} clusters")
                    
                    self.diverse_instances.update(cluster_distribution)
                    
                    # Break out of the while loop since clustering gives us the final result
                    print(f"Clustering completed with {self.get_total_instance_count()} diverse instances across {len(self.diverse_instances)} clusters")
                    break
                else:
                    # Get random instances - put in cluster 0 for non-clustering mode
                    dynamic_seed = int(time.time()) % 10000
                    get_instances_amount = self.instance_amount * 50
                    if get_instances_amount > len(data):
                        get_instances_amount = len(data)
                    diverse_instances_pandas_indices = data.sample(get_instances_amount,
                                                                   random_state=dynamic_seed).index.tolist()

                    # Apply dataset-specific filtering for random selection
                    if self.dataset_name == "adult":
                        diverse_instances_pandas_indices = self.filter_instances_by_class(data,
                                                                                          model,
                                                                                          diverse_instances_pandas_indices,
                                                                                          self.instance_amount,
                                                                                          filter_by_additional_feature=True)
                    elif self.dataset_name in ["german", "diabetes"]:
                        diverse_instances_pandas_indices = self.filter_instances_by_class(data,
                                                                                          model,
                                                                                          diverse_instances_pandas_indices,
                                                                                          self.instance_amount,
                                                                                          filter_by_additional_feature=False)

            # Only add instances for non-clustering methods (clustering handles this above)
            if not self.use_clustering or submodular_pick:
                # Ensure cluster 0 exists for non-clustering approaches
                if 0 not in self.diverse_instances:
                    self.diverse_instances[0] = []
                    
                for i in diverse_instances_pandas_indices:
                    if i not in self.diverse_instances[0]:
                        self.diverse_instances[0].append(i)

            counter += 1
            if counter > 20:
                print(f"Could not find enough diverse instances, only found {self.get_total_instance_count()}.")
                break

        if save_to_cache:
            self.save_diverse_instances(self.diverse_instances)
            
        # Return based on clustering mode
        if self.use_clustering and not submodular_pick:
            return self.diverse_instances
        else:
            # For backward compatibility with non-clustering mode
            return self.get_all_instance_ids()

    def save_diverse_instances(self, diverse_instances):
        with open(self.cache_location, 'wb') as file:
            pkl.dump(diverse_instances, file)

    def _auto_detect_categorical_features(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect categorical features based on data types and dataset name."""
        categorical_features = []
        
        # Dataset-specific categorical features
        dataset_categoricals = {
            "adult": ["workclass", "education", "marital-status", "occupation", 
                     "relationship", "race", "sex", "native-country"],
            "german": ["Status", "Duration", "Credit_history", "Purpose", 
                      "Savings", "Employment", "Personal_status", "Other_parties",
                      "Property_magnitude", "Other_payment_plans", "Housing", 
                      "Job", "Telephone", "foreign_worker"],
            "diabetes": [],  # Mostly numerical
            "compas": ["sex", "race", "c_charge_degree", "score_text"],
            "titanic": ["Sex", "Embarked", "Pclass"]
        }
        
        if self.dataset_name in dataset_categoricals:
            categorical_features.extend(dataset_categoricals[self.dataset_name])
        
        # Auto-detect based on data types
        for col in data.columns:
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                if col not in categorical_features:
                    categorical_features.append(col)
        
        # Filter to only include columns that actually exist in the data
        return [col for col in categorical_features if col in data.columns]
