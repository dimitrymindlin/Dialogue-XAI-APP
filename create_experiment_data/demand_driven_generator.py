"""
Demand-Driven Test Instance Generator Interface

This module provides the enhanced generator interface that supports dynamic,
targeted instance generation for achieving class balance.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DemandDrivenGeneratorInterface(ABC):
    """
    Abstract interface for demand-driven instance generation.
    
    This interface allows managers to request specific types of instances
    with optional class constraints, enabling dynamic balance achievement.
    """
    
    @abstractmethod
    def generate_similar_instance(self, original_instance: pd.DataFrame, 
                                target_class: Optional[int] = None,
                                max_attempts: int = 50) -> Optional[pd.DataFrame]:
        """
        Generate a similar instance to the original.
        
        Args:
            original_instance: The reference instance
            target_class: Optional class constraint (0 or 1)
            max_attempts: Maximum generation attempts
            
        Returns:
            Generated instance or None if failed
        """
        pass
    
    @abstractmethod
    def generate_counterfactual_instance(self, original_instance: pd.DataFrame,
                                       target_class: Optional[int] = None,
                                       max_attempts: int = 50) -> Optional[pd.DataFrame]:
        """
        Generate a counterfactual instance.
        
        Args:
            original_instance: The reference instance
            target_class: Optional class constraint (0 or 1)
            max_attempts: Maximum generation attempts
            
        Returns:
            Generated instance or None if failed
        """
        pass
    
    @abstractmethod
    def can_generate_target_class(self, original_instance: pd.DataFrame,
                                instance_type: str, target_class: int) -> bool:
        """
        Check if generator can reliably produce instances of target class.
        
        Args:
            original_instance: The reference instance
            instance_type: 'similar' or 'counterfactual'
            target_class: Target class (0 or 1)
            
        Returns:
            True if generator can produce target class reliably
        """
        pass


class DemandDrivenBaseGenerator(DemandDrivenGeneratorInterface):
    """
    Base implementation of demand-driven generator that extends existing generators.
    
    This class provides the demand-driven interface on top of existing
    dataset-specific generators.
    """
    
    def __init__(self, data: pd.DataFrame, model, experiment_helper, 
                 actionable_features: List[str], categorical_features: List[str] = None,
                 min_feature_differences: int = 2, max_feature_differences: int = 3):
        self.data = data
        self.model = model
        self.experiment_helper = experiment_helper
        self.actionable_features = actionable_features
        self.categorical_features = categorical_features or []
        self.min_feature_differences = min_feature_differences
        self.max_feature_differences = max_feature_differences
        
        # Calculate feature thresholds for validation
        self.feature_thresholds = self._calculate_feature_thresholds()
        
        # Initialize the legacy generator for fallback
        self._legacy_generator = None
        
    def _calculate_feature_thresholds(self) -> Dict[str, float]:
        """Calculate meaningful change thresholds for numerical features."""
        thresholds = {}
        for col in self.actionable_features:
            if col in self.categorical_features:
                thresholds[col] = 0  # Any change in categorical is meaningful
            elif col in self.data.columns:
                thresholds[col] = max(self.data[col].std() * 0.2, 1e-6)
            else:
                thresholds[col] = 1e-6
        return thresholds
    
    def _count_feature_differences(self, original_instance: pd.DataFrame, 
                                 generated_instance: pd.DataFrame) -> int:
        """Count meaningful feature differences between two instances."""
        if generated_instance is None or len(generated_instance) == 0:
            return 0
        
        differences = 0
        orig_values = original_instance.iloc[0]
        gen_values = generated_instance.iloc[0]
        
        for col in self.actionable_features:
            if col in orig_values and col in gen_values:
                orig_val = orig_values[col]
                gen_val = gen_values[col]
                
                if col in self.categorical_features:
                    if str(orig_val) != str(gen_val):
                        differences += 1
                else:
                    threshold = self.feature_thresholds.get(col, 1e-6)
                    try:
                        if abs(float(orig_val) - float(gen_val)) > threshold:
                            differences += 1
                    except (ValueError, TypeError):
                        if str(orig_val) != str(gen_val):
                            differences += 1
        
        return differences
    
    def _count_all_feature_differences(self, original_instance: pd.DataFrame, 
                                     generated_instance: pd.DataFrame) -> int:
        """Count ALL feature differences between two instances (no thresholds)."""
        if generated_instance is None or len(generated_instance) == 0:
            return 0
        
        differences = 0
        orig_values = original_instance.iloc[0]
        gen_values = generated_instance.iloc[0]
        
        for col in self.actionable_features:
            if col in orig_values and col in gen_values:
                orig_val = orig_values[col]
                gen_val = gen_values[col]
                
                # Count any difference (no thresholds)
                if col in self.categorical_features:
                    if str(orig_val) != str(gen_val):
                        differences += 1
                else:
                    try:
                        if float(orig_val) != float(gen_val):
                            differences += 1
                    except (ValueError, TypeError):
                        if str(orig_val) != str(gen_val):
                            differences += 1
        
        return differences
    
    def _validate_instance(self, original_instance: pd.DataFrame, 
                          generated_instance: pd.DataFrame) -> bool:
        """Validate that generated instance meets feature difference constraints."""
        if generated_instance is None or len(generated_instance) == 0:
            return False
            
        # Validate feature difference constraints for all datasets
        diff_count = self._count_feature_differences(original_instance, generated_instance)
        is_valid = self.min_feature_differences <= diff_count <= self.max_feature_differences
        
        if not is_valid:
            logger.debug(f"Validation failed: {diff_count} differences (expected {self.min_feature_differences}-{self.max_feature_differences})")
        
        return is_valid
    
    def generate_similar_instance(self, original_instance: pd.DataFrame, 
                                target_class: Optional[int] = None,
                                max_attempts: int = 50) -> Optional[pd.DataFrame]:
        """
        Generate a similar instance with optional class targeting.
        
        This method tries multiple strategies:
        1. Generate variations using manifold-aware techniques
        2. Use KNN-based selection for similar instances
        3. Apply class filtering if target_class is specified
        """
        logger.debug(f"Generating similar instance (target_class: {target_class})")
        
        for attempt in range(max_attempts):
            # Strategy 1: Try manifold-aware generation
            instance = self._generate_similar_candidate(original_instance, self.min_feature_differences, self.max_feature_differences)
            
            if instance is not None:
                # Validate feature differences
                if not self._validate_instance(original_instance, instance):
                    continue
                
                # Check class constraint if specified
                if target_class is not None:
                    try:
                        predicted_class = self.model.predict(instance)[0]
                        if predicted_class != target_class:
                            continue
                    except Exception as e:
                        logger.warning(f"Class prediction failed: {e}")
                        continue
                
                logger.debug(f"Similar instance generated successfully (attempt {attempt + 1})")
                return instance
        
        logger.warning(f"Failed to generate similar instance after {max_attempts} attempts")
        return None
    
    def generate_counterfactual_instance(self, original_instance: pd.DataFrame,
                                       target_class: Optional[int] = None,
                                       max_attempts: int = 50) -> Optional[pd.DataFrame]:
        """
        Generate a counterfactual instance with optional class targeting.
        
        Uses the experiment_helper to generate counterfactuals and applies
        class filtering if target_class is specified.
        """
        logger.debug(f"Generating counterfactual instance (target_class: {target_class})")
        
        try:
            # Use experiment helper to generate counterfactuals
            original_class = self.model.predict(original_instance)[0]
            
            # If no target class specified, use opposite of original
            if target_class is None:
                target_class = 1 - original_class
            
            # Generate counterfactual using existing logic
            counterfactual = self._generate_counterfactual_candidate(original_instance, target_class)
            
            if counterfactual is not None:
                # Validate feature differences
                if self._validate_instance(original_instance, counterfactual):
                    # Verify class constraint
                    predicted_class = self.model.predict(counterfactual)[0]
                    if predicted_class == target_class:
                        logger.debug("Counterfactual instance generated successfully")
                        return counterfactual
                    else:
                        logger.debug(f"Counterfactual class mismatch: got {predicted_class}, wanted {target_class}")
            
        except Exception as e:
            logger.warning(f"Counterfactual generation failed: {e}")
        
        logger.warning("Failed to generate counterfactual instance")
        return None
    
    def can_generate_target_class(self, original_instance: pd.DataFrame,
                                instance_type: str, target_class: int) -> bool:
        """
        Check if generator can reliably produce instances of target class.
        
        This is a heuristic check based on the original instance class and
        the requested instance type.
        """
        try:
            original_class = self.model.predict(original_instance)[0]
            
            if instance_type == "similar":
                # Similar instances usually keep the same class
                return target_class == original_class
            elif instance_type == "counterfactual":
                # Counterfactuals usually flip the class
                return target_class != original_class
            else:
                return False
                
        except Exception:
            return False
    
    def _generate_similar_candidate(self, original_instance: pd.DataFrame, 
                                   min_feature_differences: int = None, 
                                   max_feature_differences: int = None) -> Optional[pd.DataFrame]:
        """Generate a candidate similar instance. Override in subclasses."""
        # This is a placeholder - subclasses should implement specific logic
        # For now, return a copy with small random perturbations
        try:
            instance = original_instance.copy()
            
            # Add small random perturbations to numerical features
            for col in self.actionable_features:
                if col not in self.categorical_features and col in instance.columns:
                    if np.issubdtype(instance[col].dtype, np.number):
                        noise_scale = self.data[col].std() * 0.1
                        noise = np.random.normal(0, noise_scale)
                        instance.loc[instance.index[0], col] += noise
            
            return instance
        except Exception:
            return None
    
    def _generate_counterfactual_candidate(self, original_instance: pd.DataFrame, 
                                         target_class: int) -> Optional[pd.DataFrame]:
        """Generate a candidate counterfactual instance. Override in subclasses."""
        try:
            # Use experiment helper if available
            if hasattr(self.experiment_helper, 'get_counterfactual_instances'):
                counterfactuals = self.experiment_helper.get_counterfactual_instances(
                    original_instance, count=1
                )
                if len(counterfactuals) > 0:
                    return counterfactuals.iloc[0:1]
            
            # Fallback: generate with larger perturbations
            instance = original_instance.copy()
            
            # Add larger perturbations to flip class
            for col in self.actionable_features:
                if col not in self.categorical_features and col in instance.columns:
                    if np.issubdtype(instance[col].dtype, np.number):
                        noise_scale = self.data[col].std() * 0.3
                        noise = np.random.normal(0, noise_scale)
                        instance.loc[instance.index[0], col] += noise
            
            return instance
            
        except Exception:
            return None
