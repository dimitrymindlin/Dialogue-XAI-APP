import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Tuple, Dict, Any, Set
import warnings
from collections import defaultdict
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiverseKNNInstanceSelector:
    """Selector for similar instances with enforced actionable feature diversity (KNN-based candidate pool, diversity-aware selection)."""

    def __init__(self, data: pd.DataFrame, categorical_features: List[str] = None,
                 actionable_features: List[str] = None, dataset_name: str = "unknown"):
        self.original_data = data.copy()  # Store original data with string categorical values
        self.data = data.copy()  # This will contain encoded data after _prepare_data()
        self.categorical_features = categorical_features or []
        self.actionable_features = actionable_features or list(data.columns)
        self.dataset_name = dataset_name
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self._prepare_data()
        self._calculate_feature_thresholds()
        self._calculate_feature_weights()

    def _prepare_data(self):
        """Encode categorical features and scale all features for KNN."""
        processed_data = self.data.copy()

        for col in self.categorical_features:
            if col in processed_data.columns:
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
                self.label_encoders[col] = le

        # Scale all features first
        scaled_data = self.scaler.fit_transform(processed_data)

        # Weights will be applied later in _apply_feature_weights after they're calculated
        self.processed_data = pd.DataFrame(
            scaled_data,
            columns=processed_data.columns,
            index=processed_data.index
        )

    def _calculate_feature_weights(self):
        """Calculate feature weights for distance calculations, prioritizing categorical features."""
        self.feature_weights = {}
        
        for feature in self.actionable_features:
            if feature in self.categorical_features:
                # High weight for categorical features to prioritize their changes
                self.feature_weights[feature] = 5.0
            elif 'age' in feature.lower() or 'instance_id' in feature.lower():
                # Low weight for age and ID features to minimize their influence
                self.feature_weights[feature] = 0.1
            else:
                # Standard weight for numerical features
                self.feature_weights[feature] = 1.0

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

    def _weighted_distance(self, instance1: np.ndarray, instance2: np.ndarray) -> float:
        """Calculate weighted Euclidean distance between two instances using feature weights."""
        differences = (instance1 - instance2) ** 2
        
        for i, feature in enumerate(self.data.columns):
            if feature in self.feature_weights:
                differences[i] *= self.feature_weights[feature]
        
        return np.sqrt(np.sum(differences))

    def _analyze_feature_changes(self, row1: pd.Series, row2: pd.Series) -> Dict[str, Any]:
        """
        Unified method to analyze all feature changes between two instances.
        Returns a dictionary with comprehensive change analysis.
        """
        changed_features = []
        total_changes = 0
        categorical_changes = 0
        numerical_changes = 0
        
        for feature in row1.index:
            if feature in row2.index:
                val1, val2 = row1[feature], row2[feature]
                
                try:
                    # Handle categorical features
                    if feature in self.categorical_features:
                        # Convert both to string for consistent comparison
                        val1_str = str(val1)
                        val2_str = str(val2)
                        
                        if val1_str != val2_str:
                            changed_features.append(feature)
                            total_changes += 1
                            categorical_changes += 1
                    
                    else:
                        # Handle numerical features
                        if isinstance(val1, (int, float, np.number)) and isinstance(val2, (int, float, np.number)):
                            if abs(val1 - val2) > self.feature_thresholds.get(feature, 1e-10):
                                changed_features.append(feature)
                                total_changes += 1
                                numerical_changes += abs(val1 - val2)
                        else:
                            # Try to convert to numeric if possible
                            try:
                                num_val1 = float(val1)
                                num_val2 = float(val2)
                                if abs(num_val1 - num_val2) > self.feature_thresholds.get(feature, 1e-10):
                                    changed_features.append(feature)
                                    total_changes += 1
                                    numerical_changes += abs(num_val1 - num_val2)
                            except (ValueError, TypeError):
                                # Fall back to string comparison
                                if str(val1) != str(val2):
                                    changed_features.append(feature)
                                    total_changes += 1
                                    categorical_changes += 1
                
                except Exception as e:
                    logger.error(f"Error comparing feature '{feature}': {e}")
                    continue
        
        return {
            'changed_features': changed_features,
            'total_changes': total_changes,
            'categorical_changes': categorical_changes,
            'numerical_changes': numerical_changes
        }

    def _count_feature_differences(self, instance1: pd.DataFrame, instance2: pd.DataFrame) -> int:
        """Count meaningful feature differences between two instances."""
        # Debug: Check if we're comparing a DataFrame vs Series
        if isinstance(instance1, pd.DataFrame) and len(instance1) > 0:
            row1 = instance1.iloc[0]
        else:
            row1 = instance1
            
        if isinstance(instance2, pd.DataFrame) and len(instance2) > 0:
            row2 = instance2.iloc[0]
        else:
            row2 = instance2
            
        result = self._analyze_feature_changes(row1, row2)
        logger.debug(f"Feature differences: {result['total_changes']} (categorical: {result['categorical_changes']}, numerical: {result['numerical_changes']})")
        return result['total_changes']

    def _get_changed_features(self, instance1: pd.DataFrame, instance2: pd.DataFrame) -> Set[str]:
        """Get set of actionable features that meaningfully changed between instances."""
        return self._analyze_feature_changes(instance1, instance2)['changed_features']

    def _calculate_feature_thresholds(self):
        """Calculate thresholds for determining meaningful changes in features."""
        self.feature_thresholds = {}
        
        if self.dataset_name == "diabetes":
            # Use percentile-based thresholds for diabetes (more adaptive)
            self.feature_thresholds = self._calculate_percentile_thresholds()
        else:
            # Keep existing standard deviation approach for adult/german
            multiplier = 0.2   # Keep existing for adult/german (mixed features)
            for col in self.data.columns:
                if col in self.categorical_features:
                    self.feature_thresholds[col] = 0
                else:
                    self.feature_thresholds[col] = max(self.data[col].std() * multiplier, 1e-6)
    
    def _calculate_percentile_thresholds(self) -> dict:
        """Calculate percentile-based thresholds by analyzing actual feature differences (diabetes only)."""
        import numpy as np
        feature_thresholds = {}
        
        # Sample instance pairs to calculate difference distributions
        sample_size = min(500, len(self.data) * (len(self.data) - 1) // 2)
        np.random.seed(42)  # For reproducibility
        
        for col in self.data.columns:
            if col in self.categorical_features:
                feature_thresholds[col] = 0
            else:
                differences = []
                
                # Sample pairs of instances to calculate differences
                for _ in range(sample_size):
                    idx1, idx2 = np.random.choice(len(self.data), 2, replace=False)
                    diff = abs(self.data.iloc[idx1][col] - self.data.iloc[idx2][col])
                    if diff > 0:  # Only non-zero differences
                        differences.append(diff)
                
                if differences:
                    # Use 50th percentile as threshold (less sensitive, more meaningful differences)
                    # Also ensure minimum threshold to avoid counting tiny changes
                    threshold = np.percentile(differences, 50)
                    min_threshold = self.data[col].std() * 0.1  # Minimum based on std
                    feature_thresholds[col] = max(threshold, min_threshold, 1e-6)
                else:
                    # Fallback to std approach if no differences found
                    feature_thresholds[col] = max(self.data[col].std() * 0.2, 1e-6)
        
        return feature_thresholds

    def _is_meaningful_change(self, feature_name: str, val1: float, val2: float) -> bool:
        """Determine if the change between two feature values is significant enough to consider."""
        if feature_name in self.categorical_features:
            return val1 != val2
        return abs(val1 - val2) > self.feature_thresholds.get(feature_name, 1e-10)

    def _calculate_diversity_score(self, instance_list: List[pd.DataFrame],
                                   original_instance: pd.DataFrame) -> float:
        """Calculate an entropy-based diversity score for feature changes across multiple instances."""
        feature_change_counts = defaultdict(int)
        total_instances = len(instance_list)

        for instance_df in instance_list:
            changes = self._analyze_feature_changes(original_instance, instance_df)
            for feature in changes['changed_features']:
                feature_change_counts[feature] += 1

        # Calculate entropy-based diversity score
        if not feature_change_counts or total_instances == 0:
            return 0.0

        probabilities = [count / total_instances for count in feature_change_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

        # Normalize by maximum possible entropy (uniform distribution)
        max_entropy = np.log2(len(self.actionable_features)) if len(self.actionable_features) > 1 else 1
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0

        return diversity_score

    def get_similar_instances(self,
                              original_instance: pd.DataFrame,
                              instance_count: int,
                              max_feature_differences: int = 3,
                              min_feature_differences: int = 2,
                              k_multiplier: int = 15,
                              diversity_weight: float = 0.5,
                              prioritize_categorical: bool = True) -> pd.DataFrame:
        """Find similar instances with controlled feature diversity and categorical feature prioritization."""
        # Step 1: Find candidate instances using KNN
        candidates = self._find_knn_candidates(original_instance, instance_count * k_multiplier)
        
        # Check if we found any candidates
        if candidates.empty:
            warnings.warn("No candidates found using KNN. Returning empty DataFrame.")
            return pd.DataFrame(columns=original_instance.columns)
    
        # Step 2: Filter candidates by feature difference constraints
        valid_candidates = self._filter_candidates_by_differences(
            original_instance, candidates, min_feature_differences, max_feature_differences, prioritize_categorical
        )
        
        if len(valid_candidates) < instance_count:
            warnings.warn(f"Only found {len(valid_candidates)}/{instance_count} valid candidates.")
            return self._candidates_to_dataframe(valid_candidates)
        
        # Step 3: Select diverse instances from valid candidates
        selected_instances = self._select_diverse_instances(
            original_instance, valid_candidates, instance_count, diversity_weight, prioritize_categorical
        )
        
        # Step 4: Return final diverse subset
        return self._finalize_selection(selected_instances, instance_count)

    def _find_knn_candidates(self, original_instance: pd.DataFrame, k: int) -> pd.DataFrame:
        """Find k nearest neighbors using weighted distance."""
        original_processed = self._preprocess_instance(original_instance)
        k = min(len(self.data), k)

        # Define custom distance metric
        def custom_distance(x, y):
            return self._weighted_distance(x.reshape(1, -1)[0], y.reshape(1, -1)[0])

        # Use KNN with custom distance metric
        nbrs = NearestNeighbors(n_neighbors=k, metric=custom_distance, algorithm='brute')
        nbrs.fit(self.processed_data.values)
        distances, indices = nbrs.kneighbors(original_processed)

        # Return original data (with string categorical values) instead of encoded data
        return self.original_data.iloc[indices[0]]

    def _filter_candidates_by_differences(self, original_instance: pd.DataFrame, 
                                          candidates: pd.DataFrame, 
                                          min_diff: int, max_diff: int,
                                          prioritize_categorical: bool) -> List[Tuple[pd.DataFrame, int]]:
        """Filter candidates based on feature differences and categorical prioritization."""
        original_index = original_instance.index[0] if len(original_instance.index) > 0 else None
        
        if prioritize_categorical:
            return self._filter_and_prioritize_categorical(
                original_instance, candidates, original_index, min_diff, max_diff, len(candidates)
            )
        else:
            # Simple filtering without categorical prioritization
            valid_candidates = []
            for _, candidate_row in candidates.iterrows():
                if original_index is not None and candidate_row.name == original_index:
                    continue

                candidate_df = pd.DataFrame([candidate_row], columns=candidates.columns)
                diff_count = self._count_feature_differences(original_instance, candidate_df)

                if min_diff <= diff_count <= max_diff:
                    valid_candidates.append((candidate_df, diff_count))
            
            return valid_candidates

    def _candidates_to_dataframe(self, candidates: List[Tuple[pd.DataFrame, int]]) -> pd.DataFrame:
        """Convert list of candidate tuples to DataFrame."""
        if not candidates:
            return pd.DataFrame()
        return pd.concat([cand[0] for cand in candidates], ignore_index=True)

    def _finalize_selection(self, selected_instances: List[pd.DataFrame], instance_count: int) -> pd.DataFrame:
        """Finalize the selection by ensuring diversity and proper formatting."""
        if not selected_instances:
            return pd.DataFrame()
        
        # Convert to DataFrame
        result_df = pd.concat(selected_instances, ignore_index=True)
        
        # If we have more than needed, select most diverse subset
        if len(result_df) > instance_count:
            result_df = self.select_most_diverse_subset(result_df, instance_count)
        
        return result_df

    def _filter_and_prioritize_categorical(self, original_instance: pd.DataFrame, 
                                           candidate_instances: pd.DataFrame, original_index,
                                           min_feature_differences: int, max_feature_differences: int,
                                           instance_count: int) -> List[Tuple[pd.DataFrame, int]]:
        """Filter candidates and prioritize those with categorical feature changes."""
        categorical_candidates = []
        numerical_candidates = []
        
        logger.debug(f"Filtering {len(candidate_instances)} candidates with diff range [{min_feature_differences}, {max_feature_differences}]")
        
        for idx, candidate_row in candidate_instances.iterrows():
            if original_index is not None and candidate_row.name == original_index:
                logger.debug(f"Skipping candidate {idx} - same index as original")
                continue

            candidate_df = pd.DataFrame([candidate_row], columns=candidate_instances.columns)
            diff_count = self._count_feature_differences(original_instance, candidate_df)
            
            logger.debug(f"Candidate {idx}: {diff_count} differences")

            if min_feature_differences <= diff_count <= max_feature_differences:
                # Check if this candidate has categorical feature changes
                changes = self._analyze_feature_changes(original_instance.iloc[0], candidate_df.iloc[0])
                has_categorical_changes = changes['categorical_changes'] > 0
                
                if has_categorical_changes:
                    categorical_candidates.append((candidate_df, diff_count))
                    logger.debug(f"Added categorical candidate {idx}")
                else:
                    numerical_candidates.append((candidate_df, diff_count))
                    logger.debug(f"Added numerical candidate {idx}")
            else:
                logger.debug(f"Candidate {idx} rejected: {diff_count} not in range [{min_feature_differences}, {max_feature_differences}]")
        
        logger.debug(f"Found {len(categorical_candidates)} categorical and {len(numerical_candidates)} numerical candidates")
        # Prioritize categorical candidates
        if len(categorical_candidates) >= instance_count:
            print(f"[DiverseKNNInstanceSelector] Found {len(categorical_candidates)} candidates with categorical changes, using only those")
            return categorical_candidates
        elif len(categorical_candidates) > 0:
            # Use all categorical + some numerical to fill the gap
            needed_numerical = min(len(numerical_candidates), instance_count - len(categorical_candidates))
            print(f"[DiverseKNNInstanceSelector] Found {len(categorical_candidates)} categorical + {needed_numerical} numerical candidates")
            return categorical_candidates + numerical_candidates[:needed_numerical]
        else:
            # Fall back to numerical only
            print(f"[DiverseKNNInstanceSelector] No categorical changes found, using {len(numerical_candidates)} numerical candidates")
            return numerical_candidates

    def _select_diverse_instances(self, original_instance: pd.DataFrame,
                                  candidates: List[Tuple[pd.DataFrame, int]],
                                  instance_count: int, diversity_weight: float,
                                  prioritize_categorical: bool = True) -> List[pd.DataFrame]:
        """Select diverse instances using a two-phase approach: categorical representation then diversity optimization."""
        if not candidates:
            return []

        selected = []
        remaining_candidates = candidates.copy()
        feature_usage_count = defaultdict(int)

        # Phase 1: Ensure categorical feature representation (if prioritizing categorical)
        if prioritize_categorical:
            categorical_features = [f for f in self.categorical_features if f in self.actionable_features]
            for feature in categorical_features:
                if len(selected) >= instance_count:
                    break
                
                # Find best candidate that changes this categorical feature
                best_candidate = None
                best_distance = float('inf')
                best_idx = -1
                
                for idx, (candidate_df, _) in enumerate(remaining_candidates):
                    changes = self._analyze_feature_changes(original_instance, candidate_df)
                    if feature in changes['changed_features']:
                        # Calculate distance for tie-breaking
                        candidate_processed = self._preprocess_instance(candidate_df)
                        original_processed = self._preprocess_instance(original_instance)
                        distance = self._weighted_distance(original_processed[0], candidate_processed[0])
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_candidate = candidate_df
                            best_idx = idx
                
                if best_candidate is not None:
                    selected.append(best_candidate)
                    changes = self._analyze_feature_changes(original_instance, best_candidate)
                    for f in changes['changed_features']:
                        feature_usage_count[f] += 1
                    remaining_candidates.pop(best_idx)

        # Phase 2: Fill remaining slots with diverse selection
        while len(selected) < instance_count and remaining_candidates:
            best_candidate = None
            best_score = -float('inf')
            best_idx = -1

            for idx, (candidate_df, _) in enumerate(remaining_candidates):
                changes = self._analyze_feature_changes(original_instance, candidate_df)
                
                # Calculate similarity score
                candidate_processed = self._preprocess_instance(candidate_df)
                original_processed = self._preprocess_instance(original_instance)
                distance = self._weighted_distance(original_processed[0], candidate_processed[0])
                similarity_score = 1.0 / (1.0 + distance)

                # Calculate diversity score based on feature usage
                diversity_score = 0.0
                for feature in changes['changed_features']:
                    if feature in self.actionable_features:
                        # Bonus for underused features
                        usage_bonus = 1.0 / (1.0 + feature_usage_count[feature])
                        # Extra bonus for categorical features
                        if feature in self.categorical_features:
                            usage_bonus *= 2.0
                        diversity_score += usage_bonus

                # Bonus for categorical changes
                categorical_bonus = changes['categorical_changes'] * 0.5

                # Combined score
                combined_score = (
                    (1 - diversity_weight) * similarity_score +
                    diversity_weight * diversity_score +
                    categorical_bonus
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate_df
                    best_idx = idx

            if best_candidate is not None:
                selected.append(best_candidate)
                changes = self._analyze_feature_changes(original_instance, best_candidate)
                for f in changes['changed_features']:
                    feature_usage_count[f] += 1
                remaining_candidates.pop(best_idx)
            else:
                break

        return selected

    def get_feature_difference_analysis(self, original_instance: pd.DataFrame,
                                        similar_instances: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature differences between original and similar instances."""
        analysis = {
            'feature_differences_per_instance': [],
            'features_changed_frequency': {},
            'average_differences': 0,
            'diversity_score': 0.0,
            'feature_representation': {}
        }

        instance_list = []
        for _, row in similar_instances.iterrows():
            instance_df = pd.DataFrame([row], columns=similar_instances.columns)
            instance_list.append(instance_df)

            diff_count = self._count_feature_differences(original_instance, instance_df)
            analysis['feature_differences_per_instance'].append(diff_count)

            for col in self.actionable_features:
                if col in instance_df.columns and \
                        self._is_meaningful_change(col, original_instance[col].iloc[0], instance_df[col].iloc[0]):
                    analysis['features_changed_frequency'][col] = analysis['features_changed_frequency'].get(col, 0) + 1

        if len(similar_instances) > 0:
            analysis['average_differences'] = sum(analysis['feature_differences_per_instance']) / len(similar_instances)
            analysis['diversity_score'] = self._calculate_diversity_score(instance_list, original_instance)

            # Calculate feature representation percentages
            total_instances = len(similar_instances)
            for feature in self.actionable_features:
                count = analysis['features_changed_frequency'].get(feature, 0)
                analysis['feature_representation'][feature] = (count / total_instances) * 100

        return analysis

    def select_most_diverse_subset(self, instances: pd.DataFrame, subset_size: int) -> pd.DataFrame:
        """
        Select the most diverse subset of instances using a greedy max-min distance approach.
        Iteratively picks instances that maximize the minimum feature difference to already selected instances.
        """
        if len(instances) <= subset_size:
            return instances.copy()
        indices = list(instances.index)

        # Helper: count feature differences between two rows
        def feature_diff(idx1, idx2):
            row1 = instances.loc[idx1]
            row2 = instances.loc[idx2]
            return sum(row1[feat] != row2[feat] for feat in self.actionable_features)

        # Step 1: Pick first instance arbitrarily
        selected = [indices[0]]
        remaining = set(indices[1:])
        # Step 2: Greedily pick next as the instance with max min-diff to all selected
        while len(selected) < subset_size and remaining:
            best_idx = None
            best_score = -1
            for idx in remaining:
                min_diff = min(feature_diff(idx, sel) for sel in selected)
                if min_diff > best_score:
                    best_score = min_diff
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
        return instances.loc[selected].copy().reset_index(drop=True)


def get_diverse_similar_instances(data: pd.DataFrame, original_instance: pd.DataFrame, instance_count: int,
                                  categorical_features: List[str] = None, actionable_features: List[str] = None,
                                  max_feature_differences: int = 3, min_feature_differences: int = 2,
                                  diversity_weight: float = 0.5, prioritize_categorical: bool = True) -> pd.DataFrame:
    """Convenience function for generating diverse similar instances with categorical feature prioritization."""
    selector = DiverseKNNInstanceSelector(data, categorical_features, actionable_features)
    return selector.get_similar_instances(
        original_instance, instance_count, max_feature_differences, min_feature_differences,
        diversity_weight=diversity_weight, prioritize_categorical=prioritize_categorical
    )
