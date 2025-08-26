"""
Manifold-Aware Numerical Generator for Test Instance Creation

This module provides numerical instance generation that mirrors the successful 
ManifoldAwareCategoricalGenerator pattern but focuses on numerical features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import random

from .manifold_validator import ManifoldValidator

logger = logging.getLogger(__name__)


class ManifoldAwareNumericalGenerator:
    """
    Generates test instances by systematically modifying numerical features
    while ensuring the new instances remain in the data manifold and maintain
    realistic feature combinations.
    
    This mirrors the ManifoldAwareCategoricalGenerator pattern but handles
    numerical features using statistical modifications.
    """
    
    def __init__(self, data: pd.DataFrame, model, actionable_features: List[str],
                 categorical_features: List[str] = None, feature_thresholds: Dict[str, float] = None,
                 min_feature_differences: int = 2, max_feature_differences: int = 3):
        self.data = data.copy()
        self.model = model
        self.actionable_features = actionable_features or []
        self.categorical_features = categorical_features or []
        self.min_feature_differences = min_feature_differences
        self.max_feature_differences = max_feature_differences
        
        # Get numerical features (excluding categorical ones)
        self.numerical_features = [f for f in self.actionable_features 
                                 if f not in self.categorical_features and f in data.columns]
        
        # Calculate feature thresholds if not provided
        if feature_thresholds:
            self.feature_thresholds = feature_thresholds
        else:
            self.feature_thresholds = self._calculate_feature_thresholds()
        
        # Calculate feature statistics for realistic modifications
        self.feature_stats = self._calculate_feature_statistics()
        
        # Initialize manifold validator with relaxed settings for generated instances
        self.manifold_validator = ManifoldValidator(
            data=data,
            categorical_features=categorical_features or [],
            threshold_percentile=80  # More lenient than default 75
        )
        
        logger.info(f"ManifoldNumericalGenerator initialized with {len(self.numerical_features)} numerical features")
    
    def _calculate_feature_thresholds(self) -> Dict[str, float]:
        """Calculate meaningful change thresholds for numerical features."""
        thresholds = {}
        for feature in self.numerical_features:
            if feature in self.data.columns:
                feature_data = self.data[feature].dropna()
                # Use 20% of standard deviation as threshold for meaningful change
                thresholds[feature] = max(feature_data.std() * 0.2, 1e-6)
            else:
                thresholds[feature] = 1e-6
        return thresholds
    
    def _calculate_feature_statistics(self) -> Dict[str, Dict]:
        """Calculate comprehensive statistics for each numerical feature."""
        stats = {}
        for feature in self.numerical_features:
            if feature in self.data.columns:
                feature_data = self.data[feature].dropna()
                stats[feature] = {
                    'mean': feature_data.mean(),
                    'std': feature_data.std(),
                    'min': feature_data.min(),
                    'max': feature_data.max(),
                    'q25': feature_data.quantile(0.25),
                    'q75': feature_data.quantile(0.75)
                }
            else:
                stats[feature] = {'mean': 0, 'std': 1, 'min': 0, 'max': 1, 'q25': 0, 'q75': 1}
        return stats
    
    def generate_instances(self, original_instance: pd.DataFrame, target_count: int) -> Dict[str, Any]:
        """
        Generate instances by modifying numerical features within the manifold.
        
        This mirrors the interface of ManifoldAwareCategoricalGenerator.generate_instances().
        
        Returns:
            Dict with keys: 'most_complex_instance', 'least_complex_instance', etc.
        """
        logger.debug(f"Generating {target_count} numerical instances...")
        
        valid_instances = []
        attempts = 0
        max_attempts = target_count * 20  # Similar to Adult approach
        
        original_class = self.model.predict(original_instance)[0]
        
        while len(valid_instances) < target_count and attempts < max_attempts:
            attempts += 1
            
            # Generate instance with random number of feature changes (2-3)
            target_changes = random.randint(self.min_feature_differences, self.max_feature_differences)
            
            try:
                modified_instance = self._create_numerical_variant(original_instance, target_changes)
                
                if modified_instance is not None:
                    # Validate the instance
                    if self._validate_numerical_instance(original_instance, modified_instance, original_class):
                        # Calculate complexity (number of meaningful differences)
                        complexity = self._count_meaningful_differences(original_instance, modified_instance)
                        valid_instances.append({
                            'instance': modified_instance,
                            'complexity': complexity,
                            'changes': target_changes
                        })
                        
            except Exception as e:
                logger.debug(f"Instance generation attempt failed: {e}")
                continue
        
        logger.debug(f"Generated {len(valid_instances)} valid instances from {attempts} attempts")
        
        if not valid_instances:
            return {}
        
        # Sort by complexity (similar to categorical generator)
        valid_instances.sort(key=lambda x: x['complexity'])
        
        # Return in expected format
        result = {}
        if valid_instances:
            result['least_complex_instance'] = valid_instances[0]['instance']
            result['most_complex_instance'] = valid_instances[-1]['instance']
            
            # Add middle instances if available
            if len(valid_instances) >= 3:
                mid_idx = len(valid_instances) // 2
                result['medium_complex_instance'] = valid_instances[mid_idx]['instance']
        
        return result
    
    def _create_numerical_variant(self, original_instance: pd.DataFrame, 
                                target_changes: int) -> Optional[pd.DataFrame]:
        """Create a variant by modifying exactly target_changes numerical features."""
        if len(self.numerical_features) < target_changes:
            return None
        
        # Start with copy of original
        modified_instance = original_instance.copy()
        original_values = original_instance.iloc[0]
        
        # Randomly select features to modify
        features_to_modify = random.sample(self.numerical_features, target_changes)
        
        for feature in features_to_modify:
            original_val = original_values[feature]
            new_val = self._generate_meaningful_numerical_change(feature, original_val)
            
            # Preserve original dtype
            original_dtype = modified_instance[feature].dtype
            if original_dtype in ['int64', 'int32', 'int16', 'int8']:
                new_val = int(round(new_val))
            elif original_dtype in ['float64', 'float32']:
                new_val = float(new_val)
            
            # Verify change is still meaningful after dtype conversion
            threshold = self.feature_thresholds.get(feature, 1e-6)
            if abs(new_val - original_val) <= threshold:
                # Try with larger change if rounding made it too small
                new_val = self._generate_meaningful_numerical_change(feature, original_val, scale_factor=2.0)
                if original_dtype in ['int64', 'int32', 'int16', 'int8']:
                    new_val = int(round(new_val))
                elif original_dtype in ['float64', 'float32']:
                    new_val = float(new_val)
            
            modified_instance.loc[modified_instance.index[0], feature] = new_val
        
        return modified_instance
    
    def _generate_meaningful_numerical_change(self, feature: str, original_val: float, 
                                            scale_factor: float = 1.0) -> float:
        """Generate a meaningful change for a numerical feature."""
        stats = self.feature_stats.get(feature, {})
        threshold = self.feature_thresholds.get(feature, 1e-6)
        
        feature_std = stats.get('std', 1.0)
        feature_min = stats.get('min', 0.0)
        feature_max = stats.get('max', 100.0)
        
        # Ensure change is meaningful but reasonable
        min_change = max(threshold * 1.5, feature_std * 0.2) * scale_factor
        max_change = min(feature_std * 1.0, (feature_max - feature_min) * 0.1) * scale_factor
        
        # Random direction and magnitude
        direction = random.choice([-1, 1])
        magnitude = random.uniform(min_change, max_change)
        change = direction * magnitude
        
        # Apply change and clamp to valid range
        new_val = original_val + change
        new_val = max(feature_min, min(feature_max, new_val))
        
        # Ensure final change is meaningful
        if abs(new_val - original_val) < threshold * scale_factor:
            # If clamping made it too small, try the other direction
            new_val = original_val - change
            new_val = max(feature_min, min(feature_max, new_val))
        
        return new_val
    
    def _validate_numerical_instance(self, original_instance: pd.DataFrame, 
                                   modified_instance: pd.DataFrame, original_class: int) -> bool:
        """Validate that the numerical instance meets all requirements."""
        try:
            # Check meaningful differences count
            diff_count = self._count_meaningful_differences(original_instance, modified_instance)
            if not (self.min_feature_differences <= diff_count <= self.max_feature_differences):
                return False
            
            # Check realistic bounds (relaxed manifold validation)
            if not self._is_numerically_realistic(modified_instance):
                return False
            
            # Check class consistency
            try:
                predicted_class = self.model.predict(modified_instance)[0]
                if predicted_class != original_class:
                    return False
            except Exception:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _count_meaningful_differences(self, original_instance: pd.DataFrame, 
                                    modified_instance: pd.DataFrame) -> int:
        """Count meaningful differences between instances."""
        original_values = original_instance.iloc[0]
        modified_values = modified_instance.iloc[0]
        
        diff_count = 0
        for feature in self.numerical_features:
            if feature in original_values.index and feature in modified_values.index:
                threshold = self.feature_thresholds.get(feature, 1e-6)
                if abs(original_values[feature] - modified_values[feature]) > threshold:
                    diff_count += 1
        
        return diff_count
    
    def _is_numerically_realistic(self, instance: pd.DataFrame) -> bool:
        """Check if instance has realistic numerical values (relaxed validation)."""
        try:
            instance_values = instance.iloc[0]
            
            for feature in self.numerical_features:
                if feature in instance_values.index:
                    value = instance_values[feature]
                    stats = self.feature_stats.get(feature, {})
                    
                    # Check bounds with buffer
                    feature_min = stats.get('min', 0)
                    feature_max = stats.get('max', 100)
                    buffer = (feature_max - feature_min) * 0.05
                    
                    if not (feature_min - buffer <= value <= feature_max + buffer):
                        return False
                    
                    # Check z-score (relaxed outlier detection)
                    feature_mean = stats.get('mean', 0)
                    feature_std = stats.get('std', 1)
                    
                    if feature_std > 0:
                        z_score = abs(value - feature_mean) / feature_std
                        if z_score > 3.5:  # More lenient than usual 3.0
                            return False
            
            return True
            
        except Exception:
            # If validation fails, be permissive for now
            return True