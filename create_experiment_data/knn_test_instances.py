import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Tuple, Dict, Any
import warnings


class KNNSimilarInstanceGenerator:
    """KNN generator for similar instances with 2-3 meaningful feature differences."""
    
    def __init__(self, data: pd.DataFrame, categorical_features: List[str] = None):
        self.data = data.copy()
        self.categorical_features = categorical_features or []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self._prepare_data()
        self._calculate_feature_thresholds()
        
    def _prepare_data(self):
        """Encode categorical features and scale all features for KNN."""
        processed_data = self.data.copy()
        
        for col in self.categorical_features:
            if col in processed_data.columns:
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
                self.label_encoders[col] = le
        
        self.processed_data = pd.DataFrame(
            self.scaler.fit_transform(processed_data),
            columns=processed_data.columns,
            index=processed_data.index
        )
    
    def _preprocess_instance(self, instance: pd.DataFrame) -> np.ndarray:
        """Preprocess instance using fitted encoders and scaler."""
        processed_instance = instance.copy()
        
        for col in self.categorical_features:
            if col in processed_instance.columns and col in self.label_encoders:
                try:
                    processed_instance[col] = self.label_encoders[col].transform(
                        processed_instance[col].astype(str)
                    )
                except ValueError:
                    processed_instance[col] = 0  # Handle unseen categories
        
        return self.scaler.transform(processed_instance)
    
    def _count_feature_differences(self, instance1: pd.DataFrame, instance2: pd.DataFrame) -> int:
        """Count meaningful feature differences between two instances."""
        return sum(
            self._is_meaningful_change(col, instance1[col].iloc[0], instance2[col].iloc[0])
            for col in instance1.columns if col in instance2.columns
        )
    
    def _calculate_feature_thresholds(self):
        """Calculate meaningful change thresholds: 20% of std for numerical, 0 for categorical."""
        self.feature_thresholds = {}
        for col in self.data.columns:
            if col in self.categorical_features:
                self.feature_thresholds[col] = 0
            else:
                self.feature_thresholds[col] = max(self.data[col].std() * 0.2, 1e-6)
    
    def _is_meaningful_change(self, feature_name: str, val1: float, val2: float) -> bool:
        """Check if change between values is meaningful based on thresholds."""
        if feature_name in self.categorical_features:
            return val1 != val2
        return abs(val1 - val2) > self.feature_thresholds.get(feature_name, 1e-10)

    def get_similar_instances(self, 
                            original_instance: pd.DataFrame,
                            instance_count: int,
                            max_feature_differences: int = 3,
                            min_feature_differences: int = 2,
                            k_multiplier: int = 10) -> pd.DataFrame:
        """Find similar instances with 2-3 meaningful feature differences."""
        original_processed = self._preprocess_instance(original_instance)
        k = min(len(self.data), instance_count * k_multiplier)
        
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nbrs.fit(self.processed_data.values)
        distances, indices = nbrs.kneighbors(original_processed)
        
        candidate_instances = self.data.iloc[indices[0]]
        valid_instances = []
        original_index = original_instance.index[0] if len(original_instance.index) > 0 else None
        
        for _, candidate_row in candidate_instances.iterrows():
            if original_index is not None and candidate_row.name == original_index:
                continue
                
            candidate_df = pd.DataFrame([candidate_row], columns=candidate_instances.columns)
            diff_count = self._count_feature_differences(original_instance, candidate_df)
            
            if min_feature_differences <= diff_count <= max_feature_differences:
                valid_instances.append(candidate_df)
                if len(valid_instances) >= instance_count:
                    break
        
        if len(valid_instances) < instance_count and k < len(self.data):
            warnings.warn(f"Only found {len(valid_instances)}/{instance_count} instances with "
                         f"{min_feature_differences}-{max_feature_differences} feature differences.")
        
        return pd.concat(valid_instances, ignore_index=True).head(instance_count) if valid_instances else \
               pd.DataFrame(columns=original_instance.columns)
    
    def get_feature_difference_analysis(self, original_instance: pd.DataFrame, 
                                      similar_instances: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature differences between original and similar instances."""
        analysis = {'feature_differences_per_instance': [], 'features_changed_frequency': {}, 'average_differences': 0}
        
        for _, row in similar_instances.iterrows():
            instance_df = pd.DataFrame([row], columns=similar_instances.columns)
            diff_count = self._count_feature_differences(original_instance, instance_df)
            analysis['feature_differences_per_instance'].append(diff_count)
            
            for col in original_instance.columns:
                if col in instance_df.columns and \
                   self._is_meaningful_change(col, original_instance[col].iloc[0], instance_df[col].iloc[0]):
                    analysis['features_changed_frequency'][col] = analysis['features_changed_frequency'].get(col, 0) + 1
        
        if len(similar_instances) > 0:
            analysis['average_differences'] = sum(analysis['feature_differences_per_instance']) / len(similar_instances)
        
        return analysis


def get_knn_similar_instances(data: pd.DataFrame, original_instance: pd.DataFrame, instance_count: int,
                            categorical_features: List[str] = None, max_feature_differences: int = 3,
                            min_feature_differences: int = 2) -> pd.DataFrame:
    """Convenience function to get similar instances using KNN approach."""
    generator = KNNSimilarInstanceGenerator(data, categorical_features)
    return generator.get_similar_instances(original_instance, instance_count, 
                                         max_feature_differences, min_feature_differences)
