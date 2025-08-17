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
from .diverse_knn_test_instances import DiverseKNNInstanceSelector

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
    
    def _generate_similar_candidate(self, original_instance: pd.DataFrame) -> Optional[pd.DataFrame]:
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
    Demand-driven generator for Diabetes dataset.
    
    Similar to Adult but may have different characteristics for numerical features.
    """
    
    def __init__(self, data: pd.DataFrame, model, experiment_helper, 
                 actionable_features: List[str], categorical_features: List[str] = None,
                 min_feature_differences: int = 2, max_feature_differences: int = 3):
        super().__init__(data, model, experiment_helper, actionable_features, 
                        categorical_features, min_feature_differences, max_feature_differences)
        # Mark this as diabetes generator to skip redundant validation
        self.dataset_name = "diabetes"
        
        # Initialize KNN selector for numerical feature handling
        # Remove target column for threshold calculation
        feature_data = data.drop(['Y'], axis=1, errors='ignore')
        self.knn_selector = DiverseKNNInstanceSelector(
            data=feature_data,
            categorical_features=categorical_features or [],
            actionable_features=actionable_features,
            dataset_name="diabetes"
        )
        # Store full dataset separately for instance search
        self.full_dataset = data
    
    def _generate_similar_candidate(self, original_instance: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Find similar instance by searching dataset for instances with 2-3 meaningful feature differences."""
        logger.debug(f"DIABETES: Starting similar candidate search for instance {original_instance.index[0]}")
        try:
            original_class = self.model.predict(original_instance)[0]
            original_features = original_instance.drop('Y', axis=1, errors='ignore')
            
            # Get all instances from dataset
            dataset = self.full_dataset
            dataset_features = dataset.drop('Y', axis=1, errors='ignore')
            
            # Calculate feature thresholds (same as KNN selector)
            feature_thresholds = self.knn_selector.feature_thresholds
            logger.debug(f"Using feature thresholds: {feature_thresholds}")
            
            logger.debug(f"Searching {len(dataset)} instances for similar candidates (target class: {original_class})")
            
            # Search with progressive relaxation
            search_attempts = [
                {"min_diff": 2, "max_diff": 3},  # Preferred range
                {"min_diff": 2, "max_diff": 4},  # Slightly relaxed
                {"min_diff": 1, "max_diff": 4},  # More relaxed
                {"min_diff": 1, "max_diff": 5},  # Most relaxed
            ]
            
            for attempt in search_attempts:
                candidates = self._search_dataset_for_similar(
                    original_instance, original_features, original_class, 
                    dataset, dataset_features, feature_thresholds,
                    attempt["min_diff"], attempt["max_diff"]
                )
                
                if candidates:
                    logger.debug(f"Found {len(candidates)} candidates with {attempt['min_diff']}-{attempt['max_diff']} differences")
                    try:
                        selected = candidates[0]
                        return selected
                    except Exception as e:
                        logger.debug(f"Similar selection failed: {str(e)}")
                        continue
                    
            logger.debug("No similar instances found in dataset search")
            return None
            
        except Exception as e:
            logger.error(f"Dataset search failed: {e}")
            return None
    
    def _search_dataset_for_similar(self, original_instance: pd.DataFrame, original_features: pd.DataFrame,
                                  original_class: int, dataset: pd.DataFrame, dataset_features: pd.DataFrame,
                                  feature_thresholds: dict, min_diff: int, max_diff: int) -> List[pd.DataFrame]:
        """Search dataset for instances meeting difference criteria."""
        candidates = []
        original_idx = original_instance.index[0] if len(original_instance.index) > 0 else None
        
        for idx, candidate_row in dataset.iterrows():
            # Skip same instance
            if original_idx is not None and idx == original_idx:
                continue
                
            # Check if same class prediction
            candidate_instance = pd.DataFrame([candidate_row], columns=dataset.columns, index=[idx])
            candidate_features = candidate_instance.drop('Y', axis=1, errors='ignore')
            
            try:
                candidate_class = self.model.predict(candidate_features)[0]
                if candidate_class != original_class:
                    continue
            except Exception:
                continue
                
            # Count meaningful feature differences
            diff_count = self._count_meaningful_differences(
                original_features.iloc[0], candidate_features.iloc[0], feature_thresholds
            )
            
            # Check if within desired range
            if min_diff <= diff_count <= max_diff:
                candidates.append(candidate_instance)
                
        return candidates
    
    def _count_meaningful_differences(self, original_row: pd.Series, candidate_row: pd.Series, 
                                    feature_thresholds: dict) -> int:
        """Count meaningful differences between two instances."""
        diff_count = 0
        
        for feature in original_row.index:
            if feature in candidate_row.index and feature in feature_thresholds:
                original_val = original_row[feature]
                candidate_val = candidate_row[feature]
                threshold = feature_thresholds[feature]
                
                # For categorical features (threshold = 0)
                if threshold == 0:
                    if original_val != candidate_val:
                        diff_count += 1
                else:
                    # For numerical features
                    if abs(original_val - candidate_val) > threshold:
                        diff_count += 1
                        
        return diff_count
    
    def _generate_counterfactual_candidate(self, original_instance: pd.DataFrame, 
                                         target_class: int) -> Optional[pd.DataFrame]:
        """Generate counterfactual for Diabetes dataset."""
        logger.debug(f"DIABETES: Starting counterfactual search for instance {original_instance.index[0]} (target class: {target_class})")
        try:
            # Try to use experiment helper for counterfactuals first
            counterfactuals = self.experiment_helper.get_counterfactual_instances(
                original_instance, count=5
            )
            
            if len(counterfactuals) > 0:
                # Try to find one that matches target class
                if target_class is not None:
                    for i in range(len(counterfactuals)):
                        candidate = counterfactuals.iloc[i:i+1]
                        try:
                            predicted_class = self.model.predict(candidate)[0]
                            if predicted_class == target_class:
                                return candidate
                        except Exception:
                            continue
                
                # Return first available counterfactual
                return counterfactuals.iloc[0:1]
            
        except Exception as e:
            logger.debug(f"Counterfactual generation via experiment helper failed: {e}")
        
        # Fallback: Search dataset for target class instances
        try:
            if target_class is not None:
                original_features = original_instance.drop('Y', axis=1, errors='ignore')
                dataset = self.full_dataset
                feature_thresholds = self.knn_selector.feature_thresholds
                
                logger.debug(f"Searching dataset for counterfactual candidates (target class: {target_class})")
                
                # Search for instances with target class prediction
                candidates = []
                original_idx = original_instance.index[0] if len(original_instance.index) > 0 else None
                
                for idx, candidate_row in dataset.iterrows():
                    # Skip same instance
                    if original_idx is not None and idx == original_idx:
                        continue
                        
                    candidate_instance = pd.DataFrame([candidate_row], columns=dataset.columns, index=[idx])
                    candidate_features = candidate_instance.drop('Y', axis=1, errors='ignore')
                    
                    try:
                        candidate_class = self.model.predict(candidate_features)[0]
                        if candidate_class == target_class:
                            # Calculate difference count for ranking
                            diff_count = self._count_meaningful_differences(
                                original_features.iloc[0], candidate_features.iloc[0], feature_thresholds
                            )
                            candidates.append((candidate_instance, diff_count))
                    except Exception:
                        continue
                
                if candidates:
                    logger.debug(f"Found {len(candidates)} counterfactual candidates")
                    
                    # Prefer candidates with reasonable number of differences (2-5)
                    preferred = [c for c, d in candidates if 2 <= d <= 5]
                    if preferred:
                        try:
                            selected = preferred[0]
                            selected_idx = selected.index[0] if len(selected.index) > 0 else "unknown"
                            logger.debug(f"Selected preferred counterfactual {selected_idx}")
                            return selected
                        except Exception as e:
                            logger.debug(f"Preferred counterfactual selection failed: {str(e)}")
                    
                    # If no preferred, sort by difference count and take middle ground
                    try:
                        candidates.sort(key=lambda x: x[1])
                        selected = candidates[len(candidates)//2][0]  # Take middle candidate
                        selected_idx = selected.index[0] if len(selected.index) > 0 else "unknown"
                        logger.debug(f"Selected middle counterfactual {selected_idx}")
                        return selected
                    except Exception as e:
                        logger.error(f"💥 DIABETES: Middle counterfactual selection failed - {str(e)}")
                        return None
            
        except Exception as e:
            logger.warning(f"Dataset search counterfactual generation failed: {e}")
        
        logger.error("No counterfactual instances found in dataset")
        return None


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
