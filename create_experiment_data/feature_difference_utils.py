import pandas as pd
from typing import List, Set

def calculate_feature_thresholds(data: pd.DataFrame, categorical_features: List[str], dataset_name: str = "unknown") -> dict:
    """Calculate meaningful change thresholds: percentile-based for diabetes, std-based for others."""
    feature_thresholds = {}
    
    if dataset_name == "diabetes":
        # Use percentile-based thresholds for diabetes (more adaptive)
        feature_thresholds = _calculate_percentile_thresholds(data, categorical_features)
    else:
        # Use standard deviation approach for other datasets
        multiplier = 0.2   # Keep existing for adult/german (mixed features)
        for col in data.columns:
            if col in categorical_features:
                feature_thresholds[col] = 0
            else:
                feature_thresholds[col] = max(data[col].std() * multiplier, 1e-6)
    
    return feature_thresholds

def _calculate_percentile_thresholds(data: pd.DataFrame, categorical_features: List[str]) -> dict:
    """Calculate percentile-based thresholds by analyzing actual feature differences."""
    import numpy as np
    feature_thresholds = {}
    
    # Sample instance pairs to calculate difference distributions
    sample_size = min(500, len(data) * (len(data) - 1) // 2)
    np.random.seed(42)  # For reproducibility
    
    for col in data.columns:
        if col in categorical_features:
            feature_thresholds[col] = 0
        else:
            differences = []
            
            # Sample pairs of instances to calculate differences
            for _ in range(sample_size):
                idx1, idx2 = np.random.choice(len(data), 2, replace=False)
                diff = abs(data.iloc[idx1][col] - data.iloc[idx2][col])
                if diff > 0:  # Only non-zero differences
                    differences.append(diff)
            
            if differences:
                # Use 30th percentile as threshold (captures smaller but meaningful differences)
                threshold = np.percentile(differences, 30)
                feature_thresholds[col] = max(threshold, 1e-6)
            else:
                # Fallback to std approach if no differences found
                feature_thresholds[col] = max(data[col].std() * 0.2, 1e-6)
    
    return feature_thresholds


def is_meaningful_change(feature_name: str, val1, val2, feature_thresholds: dict, categorical_features: List[str]) -> bool:
    if feature_name in categorical_features:
        return val1 != val2
    return abs(val1 - val2) > feature_thresholds.get(feature_name, 1e-10)


def count_feature_differences(instance1, instance2, actionable_features: List[str], feature_thresholds: dict, categorical_features: List[str]) -> int:
    count = 0
    for col in actionable_features:
        if col in instance1 and col in instance2:
            if is_meaningful_change(col, instance1[col], instance2[col], feature_thresholds, categorical_features):
                count += 1
    return count


def get_changed_features(instance1, instance2, actionable_features: List[str], feature_thresholds: dict, categorical_features: List[str]) -> Set[str]:
    changed = set()
    for col in actionable_features:
        if col in instance1 and col in instance2 and \
           is_meaningful_change(col, instance1[col], instance2[col], feature_thresholds, categorical_features):
            changed.add(col)
    return changed
