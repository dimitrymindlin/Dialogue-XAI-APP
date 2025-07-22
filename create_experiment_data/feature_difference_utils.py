import pandas as pd
from typing import List, Set

def calculate_feature_thresholds(data: pd.DataFrame, categorical_features: List[str]) -> dict:
    """Calculate meaningful change thresholds: 20% of std for numerical, 0 for categorical."""
    feature_thresholds = {}
    for col in data.columns:
        if col in categorical_features:
            feature_thresholds[col] = 0
        else:
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
