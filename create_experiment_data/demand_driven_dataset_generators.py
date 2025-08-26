"""
Demand-Driven Implementation for Existing Dataset Generators

This module provides concrete implementations of the demand-driven interface
that work with the existing Adult and Diabetes dataset generators.
"""

import pandas as pd
from typing import List, Optional
import logging

from .demand_driven_generator import DemandDrivenBaseGenerator
from .manifold_categorical_generator import ManifoldAwareCategoricalGenerator
from .manifold_numerical_generator import ManifoldAwareNumericalGenerator
from .diverse_knn_test_instances import DiverseKNNInstanceSelector
from .manifold_validator import ManifoldValidator

logger = logging.getLogger(__name__)


class DemandDrivenAdultGenerator(DemandDrivenBaseGenerator):
    """
    Demand-driven generator for Adult dataset using manifold-aware categorical generation.
    
    This class extends the existing AdultDatasetGenerator logic to support
    targeted instance generation with class constraints.
    """
    
    def __init__(self, data: pd.DataFrame, model, experiment_helper, 
                 actionable_features: List[str], categorical_features: List[str] = None,
                 min_feature_differences: int = 2, max_feature_differences: int = 3):
        super().__init__(data, model, experiment_helper, actionable_features, 
                        categorical_features, min_feature_differences, max_feature_differences)
        # Mark this as adult generator (keeps existing validation)
        self.dataset_name = "adult"
        
        # Initialize manifold generator for similar instances
        self.manifold_generator = ManifoldAwareCategoricalGenerator(
            data=data,
            model=model,
            categorical_features=categorical_features or [],
            actionable_features=actionable_features,
            min_feature_differences=min_feature_differences,
            max_feature_differences=max_feature_differences
        )
        
        # Initialize KNN selector for fallback
        self.knn_selector = DiverseKNNInstanceSelector(
            data=data,
            categorical_features=categorical_features or [],
            actionable_features=actionable_features,
            dataset_name="adult"
        )
        
        # Initialize manifold validator for consistent validation
        self.manifold_validator = ManifoldValidator(
            data=data,
            categorical_features=categorical_features or []
        )
    
    def _generate_similar_candidate(self, original_instance: pd.DataFrame, 
                                   min_feature_differences: int = None, 
                                   max_feature_differences: int = None) -> Optional[pd.DataFrame]:
        """Generate similar instance using manifold-aware categorical approach."""
        try:
            # Use manifold generator to create similar instances
            results = self.manifold_generator.generate_instances(
                original_instance=original_instance,
                target_count=4  # Generate a few candidates
            )
            
            # Return the least complex instance (most similar)
            if results and 'least_complex_instance' in results:
                return results['least_complex_instance']
            
            # Fallback to most complex if least not available
            if results and 'most_complex_instance' in results:
                return results['most_complex_instance']
            
        except Exception as e:
            logger.warning(f"Manifold generation failed: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
        
        # Fallback to KNN approach
        try:
            neighbors = self.knn_selector.get_similar_instances(
                original_instance=original_instance,
                instance_count=5,
                max_feature_differences=self.max_feature_differences,
                min_feature_differences=self.min_feature_differences,
                k_multiplier=25,
                prioritize_categorical=True
            )
            
            if len(neighbors) > 0:
                # Return the most similar neighbor (first one)
                return neighbors.iloc[0:1]
                
        except Exception as e:
            logger.warning(f"KNN fallback failed: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
        
        return None
    
    def _generate_counterfactual_candidate(self, original_instance: pd.DataFrame, 
                                         target_class: int) -> Optional[pd.DataFrame]:
        """Generate counterfactual using experiment helper with class targeting."""
        try:
            # Use experiment helper to generate counterfactuals
            counterfactuals = self.experiment_helper.get_counterfactual_instances(
                original_instance, count=5
            )
            
            if len(counterfactuals) == 0:
                return None
            
            # If we have a specific target class, try to find one that matches
            if target_class is not None:
                for i in range(len(counterfactuals)):
                    candidate = counterfactuals.iloc[i:i+1]
                    try:
                        predicted_class = self.model.predict(candidate)[0]
                        if predicted_class == target_class:
                            return candidate
                    except Exception:
                        continue
            
            # If no specific match found, return the first counterfactual
            return counterfactuals.iloc[0:1]
            
        except Exception as e:
            logger.debug(f"Counterfactual generation failed: {e}")
            return None
    
    def can_generate_target_class(self, original_instance: pd.DataFrame,
                                instance_type: str, target_class: int) -> bool:
        """Check capability for Adult dataset specifically."""
        try:
            original_class = self.model.predict(original_instance)[0]
            
            if instance_type == "similar":
                # Similar instances usually keep the same class for Adult dataset
                # But manifold generation can sometimes flip it
                return True  # We'll try and see
            elif instance_type == "counterfactual":
                # Counterfactuals are designed to flip class
                return target_class != original_class
            else:
                return False
                
        except Exception:
            return False


class DemandDrivenDiabetesGenerator(DemandDrivenBaseGenerator):
    """
    Demand-driven generator for Diabetes dataset using manifold-aware numerical generation.
    
    This class mirrors the Adult generator pattern but focuses on numerical features
    since diabetes dataset has no categorical features.
    """
    
    def __init__(self, data: pd.DataFrame, model, experiment_helper, 
                 actionable_features: List[str], categorical_features: List[str] = None,
                 min_feature_differences: int = 2, max_feature_differences: int = 3):
        super().__init__(data, model, experiment_helper, actionable_features, 
                        categorical_features, min_feature_differences, max_feature_differences)
        # Mark this as diabetes generator (keeps existing validation)
        self.dataset_name = "diabetes"
        
        # Calculate feature thresholds for the numerical generator
        feature_data = data.drop(['Y'], axis=1, errors='ignore')
        knn_selector = DiverseKNNInstanceSelector(
            data=feature_data,
            categorical_features=categorical_features or [],
            actionable_features=actionable_features,
            dataset_name="diabetes"
        )
        
        # Initialize numerical manifold generator for similar instances
        self.numerical_manifold_generator = ManifoldAwareNumericalGenerator(
            data=data,
            model=model,
            actionable_features=actionable_features,
            categorical_features=categorical_features or [],
            feature_thresholds=knn_selector.feature_thresholds,
            min_feature_differences=min_feature_differences,
            max_feature_differences=max_feature_differences
        )
        
        # Initialize KNN selector for fallback
        self.knn_selector = knn_selector
        
        # Initialize manifold validator for consistent validation
        self.manifold_validator = ManifoldValidator(
            data=data,
            categorical_features=categorical_features or []
        )
    
    def _generate_similar_candidate(self, original_instance: pd.DataFrame, 
                                   min_feature_differences: int = None, 
                                   max_feature_differences: int = None) -> Optional[pd.DataFrame]:
        """Generate similar instance using manifold-aware numerical approach."""
        try:
            # Use numerical manifold generator to create similar instances (like Adult does)
            results = self.numerical_manifold_generator.generate_instances(
                original_instance=original_instance,
                target_count=4  # Generate a few candidates
            )
            
            # Return the least complex instance (most similar)
            if results and 'least_complex_instance' in results:
                return results['least_complex_instance']
            
            # Fallback to most complex if least not available
            if results and 'most_complex_instance' in results:
                return results['most_complex_instance']
            
        except Exception as e:
            logger.warning(f"Numerical manifold generation failed: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
        
        # Fallback to KNN approach
        try:
            neighbors = self.knn_selector.get_similar_instances(
                original_instance=original_instance,
                instance_count=5,
                max_feature_differences=self.max_feature_differences,
                min_feature_differences=self.min_feature_differences,
                k_multiplier=25,
                prioritize_categorical=False  # Diabetes has no categorical features
            )
            
            if len(neighbors) > 0:
                # Return the most similar neighbor (first one)
                return neighbors.iloc[0:1]
                
        except Exception as e:
            logger.warning(f"KNN fallback failed: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
        
        return None
    
    def _generate_counterfactual_candidate(self, original_instance: pd.DataFrame, 
                                         target_class: int) -> Optional[pd.DataFrame]:
        """Generate counterfactual using experiment helper with class targeting."""
        try:
            # Use experiment helper to generate counterfactuals
            counterfactuals = self.experiment_helper.get_counterfactual_instances(
                original_instance, count=5
            )
            
            if len(counterfactuals) == 0:
                return None
            
            # If we have a specific target class, try to find one that matches
            if target_class is not None:
                for i in range(len(counterfactuals)):
                    candidate = counterfactuals.iloc[i:i+1]
                    try:
                        predicted_class = self.model.predict(candidate)[0]
                        if predicted_class == target_class:
                            return candidate
                    except Exception:
                        continue
            
            # If no specific match found, return the first counterfactual
            return counterfactuals.iloc[0:1]
            
        except Exception as e:
            logger.debug(f"Counterfactual generation failed: {e}")
            return None
    
    def can_generate_target_class(self, original_instance: pd.DataFrame,
                                instance_type: str, target_class: int) -> bool:
        """Check capability for Diabetes dataset specifically."""
        try:
            original_class = self.model.predict(original_instance)[0]
            
            if instance_type == "similar":
                # Similar instances usually keep the same class for numerical datasets
                return target_class == original_class
            elif instance_type == "counterfactual":
                # Counterfactuals are designed to flip class
                return target_class != original_class
            else:
                return False
                
        except Exception:
            return False


def get_demand_driven_generator(dataset_name: str, data: pd.DataFrame, model, 
                               experiment_helper, actionable_features: List[str], 
                               categorical_features: List[str] = None,
                               min_feature_differences: int = 2, 
                               max_feature_differences: int = 3) -> DemandDrivenBaseGenerator:
    """
    Factory function to get the appropriate demand-driven generator for a dataset.
    
    Args:
        dataset_name: Name of the dataset ('adult', 'diabetes', etc.)
        data: The dataset DataFrame
        model: The trained model
        experiment_helper: Helper for counterfactual generation
        actionable_features: List of features that can be modified
        categorical_features: List of categorical feature names
        min_feature_differences: Minimum feature differences required
        max_feature_differences: Maximum feature differences allowed
        
    Returns:
        Appropriate demand-driven generator instance
    """
    dataset_name_lower = dataset_name.lower()
    
    if 'adult' in dataset_name_lower:
        logger.info("Using DemandDrivenAdultGenerator")
        return DemandDrivenAdultGenerator(
            data=data,
            model=model,
            experiment_helper=experiment_helper,
            actionable_features=actionable_features,
            categorical_features=categorical_features,
            min_feature_differences=min_feature_differences,
            max_feature_differences=max_feature_differences
        )
    elif 'diabetes' in dataset_name_lower:
        logger.info("Using DemandDrivenDiabetesGenerator")
        return DemandDrivenDiabetesGenerator(
            data=data,
            model=model,
            experiment_helper=experiment_helper,
            actionable_features=actionable_features,
            categorical_features=categorical_features,
            min_feature_differences=min_feature_differences,
            max_feature_differences=max_feature_differences
        )
    else:
        # Default to Adult generator for unknown datasets
        logger.warning(f"Unknown dataset '{dataset_name}', using DemandDrivenAdultGenerator as default")
        return DemandDrivenAdultGenerator(
            data=data,
            model=model,
            experiment_helper=experiment_helper,
            actionable_features=actionable_features,
            categorical_features=categorical_features,
            min_feature_differences=min_feature_differences,
            max_feature_differences=max_feature_differences
        )
