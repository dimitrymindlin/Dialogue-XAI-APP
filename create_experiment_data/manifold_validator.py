"""
Standalone Manifold Validator for Test Instance Generation

This module provides a reusable ManifoldValidator class that can be used by both 
Adult and Diabetes generators to ensure generated instances remain within the 
data manifold using k-nearest neighbors distance thresholding.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ManifoldValidator:
    """
    Validates if generated instances remain within the data manifold using k-nearest neighbors.
    
    This class provides manifold validation capabilities that can be shared across different
    test instance generators, ensuring consistency in validation logic.
    """
    
    def __init__(self, data: pd.DataFrame, categorical_features: List[str] = None, 
                 manifold_k: int = None, threshold_percentile: float = 75):
        """
        Initialize the ManifoldValidator.
        
        Args:
            data: Training dataset for manifold reference
            categorical_features: List of categorical feature names
            manifold_k: Number of neighbors for KNN (auto-calculated if None)
            threshold_percentile: Percentile for distance threshold calculation
        """
        self.data = data.copy()
        self.categorical_features = categorical_features or []
        self.threshold_percentile = threshold_percentile
        
        # Adaptive k based on dataset size: scale with dataset but keep reasonable bounds
        if manifold_k is None:
            self.manifold_k = max(10, min(100, len(data) // 100))
        else:
            self.manifold_k = manifold_k
            
        logger.info(f"ManifoldValidator initialized with k={self.manifold_k}, threshold_percentile={threshold_percentile}")
        
        # Initialize manifold checking components
        self._prepare_manifold_checker()
    
    def _prepare_manifold_checker(self):
        """Prepare KNN for checking if generated instances are in data manifold."""
        # Encode categorical features for manifold distance calculation
        self.encoded_data = self.data.copy()
        self.label_encoders = {}
        
        for feature in self.categorical_features:
            if feature in self.encoded_data.columns:
                le = LabelEncoder()
                self.encoded_data[feature] = le.fit_transform(self.encoded_data[feature].astype(str))
                self.label_encoders[feature] = le
            else:
                logger.warning(f"Categorical feature '{feature}' not found in data columns!")
        
        # Scale for distance calculation
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(self.encoded_data)
        
        # Setup KNN for manifold checking
        self.manifold_knn = NearestNeighbors(n_neighbors=self.manifold_k, metric='euclidean')
        self.manifold_knn.fit(self.scaled_data)
        
        # Pre-compute manifold threshold for efficiency
        self._compute_manifold_threshold()
    
    def _compute_manifold_threshold(self):
        """Pre-compute manifold distance threshold for efficiency."""
        # Sample distances from existing data - adaptive sample size
        all_distances = []
        # Use more samples for larger datasets, but cap at reasonable limit and dataset size
        sample_size = max(500, min(2000, len(self.data) // 10))
        sample_size = min(sample_size, len(self.data))  # Ensure sample_size doesn't exceed dataset size
        sample_indices = np.random.choice(len(self.data), sample_size, replace=False)
        
        for idx in sample_indices:
            instance_encoded = self.scaled_data[idx:idx + 1]
            dists, _ = self.manifold_knn.kneighbors(instance_encoded)
            all_distances.append(np.mean(dists[0]))
        
        self.manifold_threshold = np.percentile(all_distances, self.threshold_percentile)
        logger.info(f"Computed manifold threshold: {self.manifold_threshold:.4f}")
    
    def _encode_instance(self, instance: pd.DataFrame) -> np.ndarray:
        """Encode an instance for manifold distance calculation."""
        encoded_instance = instance.copy()
        
        for feature in self.categorical_features:
            if feature in encoded_instance.columns and feature in self.label_encoders:
                try:
                    encoded_instance[feature] = self.label_encoders[feature].transform(
                        encoded_instance[feature].astype(str)
                    )
                except ValueError:
                    # Handle unseen categories by using most frequent value
                    most_frequent = self.encoded_data[feature].mode().iloc[0]
                    encoded_instance[feature] = most_frequent
                    logger.warning(f"Unseen category in feature '{feature}', using most frequent value")
        
        return self.scaler.transform(encoded_instance)
    
    def is_in_manifold(self, instance: pd.DataFrame, debug: bool = False) -> bool:
        """
        Check if an instance is within the data manifold by comparing its distance
        to k nearest neighbors with the pre-computed threshold.
        
        Args:
            instance: Instance to validate as DataFrame
            debug: Whether to log debug information
            
        Returns:
            bool: True if instance is within manifold, False otherwise
        """
        try:
            encoded_instance = self._encode_instance(instance)
            distances, _ = self.manifold_knn.kneighbors(encoded_instance)
            avg_distance = np.mean(distances[0])
            
            is_in_manifold = avg_distance <= self.manifold_threshold
            
            if debug:
                logger.debug(f"Manifold validation: distance={avg_distance:.4f}, "
                           f"threshold={self.manifold_threshold:.4f}, "
                           f"result={'PASS' if is_in_manifold else 'FAIL'}")
            
            return is_in_manifold
            
        except Exception as e:
            logger.warning(f"Error checking manifold: {e}")
            return False
    
    def get_manifold_distance(self, instance: pd.DataFrame) -> float:
        """
        Get the manifold distance for an instance (useful for ranking candidates).
        
        Args:
            instance: Instance to calculate distance for
            
        Returns:
            float: Average distance to k nearest neighbors
        """
        try:
            encoded_instance = self._encode_instance(instance)
            distances, _ = self.manifold_knn.kneighbors(encoded_instance)
            return np.mean(distances[0])
        except Exception as e:
            logger.warning(f"Error calculating manifold distance: {e}")
            return float('inf')
    
    def get_manifold_info(self) -> Dict:
        """
        Get information about the manifold validator configuration.
        
        Returns:
            dict: Configuration and statistics information
        """
        return {
            'manifold_k': self.manifold_k,
            'manifold_threshold': self.manifold_threshold,
            'threshold_percentile': self.threshold_percentile,
            'dataset_size': len(self.data),
            'categorical_features': self.categorical_features,
            'total_features': len(self.data.columns)
        }