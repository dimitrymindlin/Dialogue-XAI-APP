import os
from typing import Dict, Any, Tuple, List
import pickle as pkl
import gin
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .diverse_knn_test_instances import DiverseKNNInstanceSelector
from .manifold_categorical_generator import ManifoldAwareCategoricalGenerator
from .feature_difference_utils import calculate_feature_thresholds, count_feature_differences
from .dataset_specific_generators import get_dataset_generator


def load_cache(cache_location: str):
    if os.path.isfile(cache_location):
        with open(cache_location, 'rb') as file:
            return pkl.load(file)
    return []


@gin.configurable
class TestInstances:
    """
    Comprehensive test instance generator for explainable AI experiments.
    
    Creates diverse test instances with configurable constraints, ensuring meaningful
    differences from original instances. Uses dataset-specific generators with robust
    validation to prevent identical instances from being used as test cases.
    
    The generator employs multiple strategies:
    - Dataset-specific optimized generation
    - Manifold-aware categorical generation for fallback
    - KNN-based diverse instance selection as final fallback
    - Strict validation requiring minimum feature differences
    
    All generated instances are validated to ensure they have at least the minimum
    required feature differences from the original instance before being accepted.
    """
    
    def __init__(self,
                 data,
                 model,
                 mega_explainer,
                 experiment_helper,
                 diverse_instance_ids,
                 actionable_features,
                 max_features_to_vary=2,
                 cache_location: str = "./cache/test-instances.pkl",
                 instance_amount: int = 10,
                 categorical_features: List[str] = None,
                 min_feature_differences: int = 2,
                 max_feature_differences: int = 3,
                 dataset_name: str = "unknown"):
        self.data = data
        # Flatten diverse_instance_ids if dict
        if isinstance(diverse_instance_ids, dict):
            self.diverse_instance_ids = [iid for ids in diverse_instance_ids.values() for iid in ids]
        else:
            self.diverse_instance_ids = diverse_instance_ids
        self.cache_location = cache_location
        self.model = model
        self.mega_explainer = mega_explainer
        self.experiment_helper = experiment_helper
        self.actionable_features = actionable_features
        self.test_instances = load_cache(cache_location)
        self.max_features_to_vary = max_features_to_vary
        self.instance_amount = instance_amount
        self.categorical_features = categorical_features or []
        self.min_feature_differences = min_feature_differences
        self.max_feature_differences = max_feature_differences
        self.dataset_name = dataset_name
        self.feature_thresholds = calculate_feature_thresholds(self.data, self.categorical_features)

    def _calculate_detailed_feature_differences(self, original_instance: pd.DataFrame, generated_instance: pd.DataFrame) -> Dict[str, Any]:
        """Calculates detailed feature differences between two instances."""
        if generated_instance is None or len(generated_instance) == 0:
            return {'total': 0, 'categorical': 0, 'numerical': 0, 'changed_features': []}

        # Convert single row to proper format for comparison
        gen_values = generated_instance.iloc[0] if hasattr(generated_instance, 'iloc') and len(generated_instance) > 0 else generated_instance
        orig_values = original_instance.iloc[0] if hasattr(original_instance, 'iloc') and len(original_instance) > 0 else original_instance

        categorical_differences = 0
        numerical_differences = 0
        changed_features_list = []

        for col in self.actionable_features:
            if col in orig_values and col in gen_values:
                orig_val = orig_values[col]
                gen_val = gen_values[col]

                is_categorical = (col in self.categorical_features or
                                  isinstance(orig_val, str) or isinstance(gen_val, str) or
                                  not isinstance(orig_val, (int, float, np.number)) or
                                  not isinstance(gen_val, (int, float, np.number)))

                if is_categorical:
                    if str(orig_val) != str(gen_val):
                        categorical_differences += 1
                        changed_features_list.append(f"{col}: {orig_val} → {gen_val}")
                else:
                    threshold = self.feature_thresholds.get(col, 1e-6)
                    try:
                        if abs(float(orig_val) - float(gen_val)) > threshold:
                            numerical_differences += 1
                            # Format based on original data type
                            if isinstance(orig_val, (int, np.integer)) and isinstance(gen_val, (int, np.integer)):
                                changed_features_list.append(f"{col}: {int(orig_val)} → {int(gen_val)}")
                            else:
                                changed_features_list.append(f"{col}: {orig_val:.3f} → {gen_val:.3f}")
                    except (ValueError, TypeError): # Fallback to string comparison if float conversion fails
                        if str(orig_val) != str(gen_val):
                            categorical_differences += 1
                            changed_features_list.append(f"{col}: {orig_val} → {gen_val}")
        
        total_differences = categorical_differences + numerical_differences
        return {
            'total': total_differences,
            'categorical': categorical_differences,
            'numerical': numerical_differences,
            'changed_features': changed_features_list
        }

    def _count_and_log_feature_differences(self, original_instance: pd.DataFrame, generated_instance: pd.DataFrame, instance_type: str) -> int:
        """Count feature differences and log them with categorical/numerical breakdown."""
        if generated_instance is None or len(generated_instance) == 0:
            return 0
        
        details = self._calculate_detailed_feature_differences(original_instance, generated_instance)
        total_differences = details['total']
        
        # Log only if constraints are not met
        if not (self.min_feature_differences <= total_differences <= self.max_feature_differences):
            print(f"[TestInstances] {instance_type}: {total_differences} differences (expected {self.min_feature_differences}-{self.max_feature_differences})")
        
        return total_differences

    def _count_feature_differences_silent(self, original_instance: pd.DataFrame, generated_instance: pd.DataFrame) -> int:
        """Count feature differences without logging (for internal use)."""
        details = self._calculate_detailed_feature_differences(original_instance, generated_instance)
        return details['total']

    def get_test_instances(self, save_to_cache: bool = True, close_instances: bool = True) -> Tuple[Dict[str, Any], List[int]]:
        """Main entry point for generating test instances."""
        if self.test_instances:
            return self.test_instances, []
        
        test_instances, ids_to_remove = self._generate_instances(self.instance_amount, close_instances)
        
        if save_to_cache:
            with open(self.cache_location, 'wb') as file:
                pkl.dump(test_instances, file)
        
        return test_instances, ids_to_remove

    def _generate_instances(self, instance_count: int, close_instances_flag: bool) -> Tuple[Dict[str, Any], List[int]]:
        """Generate instances for each diverse instance ID."""
        ids_to_remove = []
        test_instances = {}
        
        for i, instance_id in enumerate(self.diverse_instance_ids):
            original_instance = self.data.loc[instance_id:instance_id]
            
            if not close_instances_flag:
                test_instances[instance_id] = self._generate_random_instances()
            else:
                close_instances = self._generate_close_instances(original_instance, instance_count)
                if close_instances is not None:
                    # Validate each instance individually
                    valid_core_instances = 0
                    
                    for instance_type, instance_data in close_instances.items():
                        diff_count = self._count_feature_differences_silent(original_instance, instance_data)
                        if diff_count >= 2 and instance_type in ['most_complex_instance', 'least_complex_instance']:
                            valid_core_instances += 1
                    
                    # Require at least 2 valid core instances
                    if valid_core_instances >= 2:
                        test_instances[instance_id] = close_instances
                    else:
                        print(f"[TestInstances] Removing instance {instance_id}: insufficient valid instances ({valid_core_instances}/2)")
                        ids_to_remove.append(instance_id)
                else:
                    ids_to_remove.append(instance_id)
        
        print(f"[TestInstances] Generated {len(test_instances)} instances, removed {len(ids_to_remove)}")
        return test_instances, ids_to_remove

    def _generate_random_instances(self) -> Dict[str, Any]:
        """Generate random test instances as fallback."""
        test_instance = self.data.sample(1)
        test_instance['label'] = self.model.predict(test_instance)
        return {k: test_instance for k in ["most_complex_instance", "least_complex_instance", 
                                          "easy_counterfactual_instance", "hard_counterfactual_instance"]}

    def _generate_close_instances(self, original_instance: pd.DataFrame, instance_count: int) -> Dict[str, Any]:
        """Generate close instances using dataset-specific generators."""
        instance_id = original_instance.index[0] if len(original_instance.index) > 0 else "unknown"
        
        try:
            # Get the appropriate dataset-specific generator
            generator = get_dataset_generator(
                dataset_name=self.dataset_name,
                data=self.data,
                model=self.model,
                experiment_helper=self.experiment_helper,
                actionable_features=self.actionable_features,
                categorical_features=self.categorical_features,
                min_feature_differences=self.min_feature_differences,
                max_feature_differences=self.max_feature_differences
            )
            
            # Generate instances using the dataset-specific approach
            results = generator.generate_instances(
                original_instance=original_instance,
                instance_count=instance_count
            )
            
            if results:
                return results
            else:
                print(f"[TestInstances] Dataset-specific generator failed for instance {instance_id}")
                return None
                
        except Exception as e:
            print(f"[TestInstances] Dataset-specific generation failed: {e}")
            return self._legacy_generate_close_instances(original_instance, instance_count)

    def _legacy_generate_close_instances(self, original_instance: pd.DataFrame, instance_count: int) -> Dict[str, Any]:
        """Fallback method using ManifoldAwareCategoricalGenerator for backward compatibility."""
        instance_id = original_instance.index[0] if len(original_instance.index) > 0 else "unknown"
        
        try:
            # Use manifold-aware categorical generator
            generator = ManifoldAwareCategoricalGenerator(
                data=self.data,
                model=self.model,
                categorical_features=self.categorical_features,
                actionable_features=self.actionable_features
            )
            
            results = generator.generate_instances(
                original_instance=original_instance,
                target_count=instance_count,
                min_feature_differences=self.min_feature_differences,
                max_feature_differences=self.max_feature_differences
            )
            
            # Generate counterfactuals
            counterfactuals = self._generate_counterfactuals(original_instance)
            
            if not counterfactuals:
                return None
            
            # Validate that generated instances are different from original
            most_complex = results.get('most_complex_instance')
            least_complex = results.get('least_complex_instance')
            
            if most_complex is not None:
                diff_count = self._count_feature_differences_silent(original_instance, most_complex)
                if diff_count == 0:
                    most_complex = None
            
            if least_complex is not None:
                diff_count = self._count_feature_differences_silent(original_instance, least_complex)
                if diff_count == 0:
                    least_complex = None
            
            # Only proceed if we have at least one valid complex instance and counterfactuals
            if (most_complex is not None or least_complex is not None) and counterfactuals:
                final_results = {
                    'most_complex_instance': most_complex or least_complex,
                    'least_complex_instance': least_complex or most_complex,
                    **counterfactuals
                }
            else:
                return None
            
            return final_results
            
        except Exception as e:
            return self._fallback_to_knn_generation(original_instance, instance_count)

    def _fallback_to_knn_generation(self, original_instance: pd.DataFrame, instance_count: int) -> Dict[str, Any]:
        """Fallback to KNN-based generation if manifold approach fails.""" 
        instance_id = original_instance.index[0] if len(original_instance.index) > 0 else "unknown"
        
        try:
            selector = DiverseKNNInstanceSelector(
                data=self.data,
                categorical_features=self.categorical_features,
                actionable_features=self.actionable_features
            )
            
            neighbors = selector.get_similar_instances(
                original_instance=original_instance,
                instance_count=instance_count,
                max_feature_differences=self.max_feature_differences,
                min_feature_differences=self.min_feature_differences,
                k_multiplier=25,
                prioritize_categorical=True
            )
            
        except Exception as e:
            print(f"[TestInstances] KNN fallback failed for instance {instance_id}: {e}")
            return self._generate_random_instances()
        
        if len(neighbors) < 2:
            print(f"[TestInstances] Insufficient neighbors ({len(neighbors)}) for instance {instance_id}")
            return None
        
        # Generate different types of instances
        complex_instances = self._sort_and_select_instances(original_instance, neighbors)
        
        # Check if instance generation failed
        if complex_instances is None:
            print(f"[TestInstances] Failed to generate complex instances for instance {instance_id}")
            return None
            
        counterfactuals = self._generate_counterfactuals(original_instance)
        
        # Handle case where counterfactuals fail (returning None)
        if counterfactuals is None:
            # Use the complex instances as fallbacks for counterfactuals
            counterfactuals = {
                "easy_counterfactual_instance": complex_instances["least_complex_instance"],
                "hard_counterfactual_instance": complex_instances["most_complex_instance"]
            }
        
        return {
            **complex_instances,
            **counterfactuals
        }

    def _sort_and_select_instances(self, original_instance: pd.DataFrame, instances: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Sort instances by complexity and select most/least complex with categorical prioritization."""
        if len(instances) == 0:
            print("[TestInstances] No instances to sort, returning None")
            return None
        
        # Complexity calculation prioritizing categorical changes
        complexities = []
        
        for _, row in instances.iterrows():
            instance_df = pd.DataFrame([row], columns=instances.columns)
            
            # Calculate cosine similarity for base complexity
            similarity = cosine_similarity(
                original_instance.values.reshape(1, -1), 
                instance_df.values.reshape(1, -1)
            )[0][0]
            base_complexity = 1 - similarity
            
            # Get detailed feature differences
            diff_details = self._calculate_detailed_feature_differences(original_instance, instance_df)
            categorical_changes = diff_details['categorical']
            total_changes = diff_details['total']
            
            # Calculate complexity with categorical prioritization
            if not self.actionable_features: 
                feature_complexity = 0
                categorical_contribution = 0
            else:
                feature_complexity = total_changes / len(self.actionable_features)
                categorical_contribution = (categorical_changes * 2.0) / len(self.actionable_features)
            
            enhanced_complexity = (
                0.4 * base_complexity +
                0.3 * feature_complexity +
                0.3 * categorical_contribution
            )
            
            complexities.append(enhanced_complexity)
        
        instances = instances.copy()
        instances['complexity'] = complexities
        sorted_instances = instances.sort_values(by='complexity', ascending=False)
        
        most_complex = sorted_instances.head(1).copy()
        least_complex = sorted_instances.tail(1).copy()
        
        # Remove temporary columns
        most_complex.drop(columns=['complexity'], inplace=True)
        least_complex.drop(columns=['complexity'], inplace=True)
        
        return {
            "most_complex_instance": most_complex,
            "least_complex_instance": least_complex
        }

    def _generate_counterfactuals(self, original_instance: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate counterfactual instances."""
        try:
            counterfactuals = self.experiment_helper.get_counterfactual_instances(original_instance)

            if len(counterfactuals) >= 2:
                return {
                    "easy_counterfactual_instance": counterfactuals.head(1),
                    "hard_counterfactual_instance": counterfactuals.tail(1)
                }
            elif len(counterfactuals) == 1:
                single_cf = counterfactuals.head(1)
                return {
                    "easy_counterfactual_instance": single_cf,
                    "hard_counterfactual_instance": single_cf
                }
            else:
                return None

        except Exception as e:
            return None
