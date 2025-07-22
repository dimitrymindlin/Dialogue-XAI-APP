import os
from typing import Dict, Any, Tuple, List
import pickle as pkl
import gin
import pandas as pd
import numpy as np
from .demand_driven_manager import DemandDrivenTestInstanceManager
from .demand_driven_dataset_generators import get_demand_driven_generator


def load_cache(cache_location: str):
    if os.path.isfile(cache_location):
        with open(cache_location, 'rb') as file:
            return pkl.load(file)
    return {}, {} # Return two empty dictionaries if no cache


@gin.configurable
class TestInstances:
    """
    Comprehensive test instance generator for explainable AI experiments.
    
    Creates diverse test instances with configurable constraints, ensuring meaningful
    differences from original instances. Uses dataset-specific generators with robust
    validation to prevent identical instances from being used as test cases.
    
    The generator employs multiple strategies:
    - Dataset-specific manifold-aware generation (primary for Adult/Diabetes)
    - KNN-based diverse instance selection as fallback
    - Random selection as final fallback
    - Strict validation requiring minimum feature differences
    
    All generated instances are validated to ensure they have at least the minimum
    required feature differences from the original instance before being accepted.
    """
    def __init__(self,
                 data,
                 model,
                 experiment_helper,
                 diverse_instance_ids,
                 actionable_features,
                 cache_location: str = "./cache/test-instances.pkl",
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
        self.experiment_helper = experiment_helper
        self.actionable_features = actionable_features
        
        # Load both structured test instances and structured final_test instances from cache
        self.test_instances_structured, self.final_test_instances_structured = load_cache(cache_location)

        self.categorical_features = categorical_features or []
        self.min_feature_differences = min_feature_differences
        self.max_feature_differences = max_feature_differences
        self.dataset_name = dataset_name

    def get_test_instances(self, save_to_cache: bool = True, target_balance: float = 0.5, tolerance: float = 0.1) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Main entry point for demand-driven test instance generation.
        Dynamically generates instances to achieve class balance.
        Returns a tuple of (structured_test_instances, structured_final_test_instances).
        """
        if self.test_instances_structured and self.final_test_instances_structured:
            return self.test_instances_structured, self.final_test_instances_structured

        # Prepare demand-driven generator and manager
        generator = get_demand_driven_generator(
            data=self.data,
            model=self.model,
            experiment_helper=self.experiment_helper,
            actionable_features=self.actionable_features,
            categorical_features=self.categorical_features,
            min_feature_differences=self.min_feature_differences,
            max_feature_differences=self.max_feature_differences,
            dataset_name=self.dataset_name
        )
        manager = DemandDrivenTestInstanceManager(
            model=self.model,
            generator=generator,
            data=self.data
        )

        # Generate all balanced test instances in a single call
        # This creates balanced 'test' and 'final_test' phases with intro_test=final_test
        all_instances = manager.generate_balanced_instances(
            training_instances=self.diverse_instance_ids,
            target_balance=target_balance,
            tolerance=tolerance
        )
        
        # The manager returns one dict with all phases: test, final_test, intro_test
        # For backward compatibility, return the same dict for both
        test_instances = all_instances  
        final_test_instances = all_instances

        if save_to_cache:
            with open(self.cache_location, 'wb') as file:
                pkl.dump((test_instances, final_test_instances), file)

        return test_instances, final_test_instances

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

    def _generate_random_instances(self) -> Dict[str, Any]:
        """Generate random test instances as fallback."""
        test_instance = self.data.sample(1)
        test_instance['label'] = self.model.predict(test_instance)
        return {k: test_instance for k in ["most_complex_instance", "least_complex_instance", 
                                          "easy_counterfactual_instance", "hard_counterfactual_instance"]}

    def _validate_instance(self, instance: pd.DataFrame) -> bool:
        """Validate the generated instance against the original instance."""
        try:
            # Check for minimum feature differences
            diff_count = self._count_feature_differences_silent(self.data.loc[instance.index[0]], instance)
            return diff_count >= self.min_feature_differences
        except Exception as e:
            print(f"[TestInstances] Validation failed: {e}")
            return False
