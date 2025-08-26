import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Dict, Any
import logging

from .manifold_validator import ManifoldValidator

logger = logging.getLogger(__name__)


class ManifoldAwareCategoricalGenerator:
    """
    Generates test instances by systematically changing categorical features
    while ensuring the new instances remain in the data manifold and maintain
    the same prediction as the original instance.
    """

    def __init__(self, data: pd.DataFrame, model, categorical_features: List[str],
                 actionable_features: List[str] = None, manifold_k: int = None,
                 min_feature_differences: int = 2, max_feature_differences: int = 3):
        self.data = data.copy()
        self.model = model
        self.categorical_features = categorical_features or []
        self.actionable_features = actionable_features or list(data.columns)
        # Adaptive k based on dataset size: scale with dataset but keep reasonable bounds
        if manifold_k is None:
            self.manifold_k = max(10, min(100, len(data) // 100))
        else:
            self.manifold_k = manifold_k
        self.min_feature_differences = min_feature_differences
        self.max_feature_differences = max_feature_differences

        # Store categorical value mappings
        self.categorical_values = {}
        for feature in self.categorical_features:
            if feature in self.data.columns:
                unique_vals = sorted(self.data[feature].unique())
                self.categorical_values[feature] = unique_vals
            else:
                logger.warning(f"[ManifoldGenerator] Categorical feature '{feature}' not found in data columns!")

        if len(self.categorical_values) == 0:
            logger.warning(f"[ManifoldGenerator] No valid categorical features found!")

        # Initialize manifold validator
        self.manifold_validator = ManifoldValidator(
            data=data,
            categorical_features=categorical_features,
            manifold_k=self.manifold_k
        )


    def _has_same_prediction(self, original_instance: pd.DataFrame,
                             generated_instance: pd.DataFrame) -> bool:
        """Check if generated instance has same prediction as original."""
        try:
            orig_pred = self.model.predict(original_instance)[0]
            gen_pred = self.model.predict(generated_instance)[0]
            return orig_pred == gen_pred
        except Exception as e:
            logger.warning(f"Error checking prediction: {e}")
            return False

    def _generate_categorical_variants(self, original_instance: pd.DataFrame, max_variants: int = 200) -> List[
        pd.DataFrame]:
        """Generate variants by randomly changing 1-3 categorical features."""
        variants = []
        original_row = original_instance.iloc[0]

        # Get actionable categorical features
        categorical_actionable = [f for f in self.categorical_features
                                  if f in self.actionable_features and f in original_instance.columns]

        if len(categorical_actionable) == 0:
            logger.warning("No actionable categorical features found!")
            return variants

        # Generate random variants with 1-2 categorical changes (to allow room for numerical)
        for _ in range(max_variants):
            # Randomly choose number of features to change (1-2)
            num_features_to_change = np.random.randint(1, min(3, len(categorical_actionable) + 1))

            # Randomly select features to change
            features_to_change = np.random.choice(categorical_actionable,
                                                  num_features_to_change, replace=False)

            # Create variant by changing selected features
            variant = original_instance.copy()
            change_descriptions = []

            valid_variant = True
            for feature in features_to_change:
                original_value = original_row[feature]
                available_values = self.categorical_values.get(feature, [])
                possible_values = [v for v in available_values if v != original_value]

                if not possible_values:
                    valid_variant = False
                    break

                # Randomly select new value
                new_value = np.random.choice(possible_values)
                variant.iloc[0, variant.columns.get_loc(feature)] = new_value
                change_descriptions.append(f"{feature}: {original_value} → {new_value}")

            if valid_variant:
                change_desc = ", ".join(change_descriptions)
                variants.append((variant, change_desc))

        return variants

    def _enhance_with_numerical_noise(self, variant: pd.DataFrame,
                                      original_instance: pd.DataFrame,
                                      max_total_changes: int = 2) -> pd.DataFrame:
        """Add varied numerical changes to create complexity differentiation while staying in manifold."""
        enhanced = variant.copy()
        numerical_features = [f for f in self.actionable_features
                              if f not in self.categorical_features and f in variant.columns]

        # Count existing categorical changes to limit total changes
        existing_changes = 0
        for feature in self.categorical_features:
            if feature in variant.columns and feature in original_instance.columns:
                if variant.iloc[0][feature] != original_instance.iloc[0][feature]:
                    existing_changes += 1

        # Only add numerical changes if we haven't reached max changes
        remaining_changes = max_total_changes - existing_changes
        if remaining_changes <= 0 or len(numerical_features) == 0:
            return enhanced

        # Add noise to 1 numerical feature maximum to stay within limits
        num_to_change = min(1, remaining_changes, len(numerical_features))
        features_to_change = np.random.choice(numerical_features, num_to_change, replace=False)

        for feature in features_to_change:
            original_val = original_instance.iloc[0][feature]
            feature_std = self.data[feature].std()

            # Add moderate noise: 15-25% of std for differentiation
            noise_magnitude = np.random.uniform(0.15, 0.25) * feature_std
            noise = np.random.normal(0, noise_magnitude)
            new_val = original_val + noise

            # Ensure within reasonable bounds
            feature_min = self.data[feature].min()
            feature_max = self.data[feature].max()
            new_val = np.clip(new_val, feature_min, feature_max)

            # Preserve integer type for integer features
            if pd.api.types.is_integer_dtype(self.data[feature]):
                new_val = int(round(new_val))

            enhanced.iloc[0, enhanced.columns.get_loc(feature)] = new_val

        return enhanced

    def generate_instances(self, original_instance: pd.DataFrame, target_count: int) -> Dict[str, Any]:
        """Generate instances by changing only categorical features within the manifold."""
        # print(f"[ManifoldCategoricalGenerator] Generating {target_count} instances with categorical changes...")

        original_instance_values = original_instance.iloc[0]
        valid_instances = []

        # Generate a pool of potential variants
        variants = self._generate_categorical_variants(original_instance, max_variants=200)
        # print(f"[ManifoldCategoricalGenerator] Generated {len(variants)} categorical variants")

        attempts = 0
        rejection_reasons = {}

        for variant, change_desc in variants:
            if attempts >= target_count:
                break

            attempts += 1

            # Enhance with numerical noise to create complexity diversity (respecting max_feature_differences)
            if np.random.random() < 0.6:  # 60% chance to add numerical noise
                enhanced_variant = self._enhance_with_numerical_noise(variant, original_instance,
                                                                      max_total_changes=self.max_feature_differences)
            else:
                enhanced_variant = variant

            # Check constraints
            is_manifold = self.manifold_validator.is_in_manifold(enhanced_variant)
            same_prediction = self._has_same_prediction(original_instance, enhanced_variant)

            # Count feature differences
            diff_count = self._count_feature_differences(original_instance, enhanced_variant)

            # Log attempt
            is_valid = is_manifold and same_prediction and self.min_feature_differences <= diff_count <= self.max_feature_differences

            # Collect rejection reasons
            reasons = []
            if not is_manifold:
                reasons.append("not in manifold")
            if not same_prediction:
                reasons.append("different prediction")
            if not (self.min_feature_differences <= diff_count <= self.max_feature_differences):
                reasons.append(
                    f"diff_count={diff_count} not in [{self.min_feature_differences}, {self.max_feature_differences}]")

            # print attempt details
            if is_valid:
                valid_instances.append(enhanced_variant)
                status = "✅"
                # print(f"[ManifoldCategoricalGenerator] {status} Instance {len(valid_instances)}: {change_desc} ({diff_count} diffs)")
            else:
                status = "❌"
                # print(f"[ManifoldCategoricalGenerator] {status} Rejected: {change_desc} - {', '.join(reasons)}")
                for reason in reasons:
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        success_rate = len(valid_instances) / attempts if attempts > 0 else 0
        print(
            f"[ManifoldCategoricalGenerator] Generated {len(valid_instances)}/{target_count} valid instances from {attempts} attempts (success rate: {success_rate:.1%})")

        # Log top 3 rejection reasons if any rejections occurred
        if rejection_reasons:
            sorted_reasons = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)
            top_reasons = sorted_reasons[:3]
            print(f"[ManifoldCategoricalGenerator] Top rejection reasons: {top_reasons}")

        # Sort instances by complexity for most/least selection
        if len(valid_instances) >= 2:
            instances_df = pd.concat(valid_instances, ignore_index=True)
            complexity_results = self._sort_by_complexity(original_instance, instances_df)
            return {
                **complexity_results,
                'total_attempts': attempts,
                'success_rate': len(valid_instances) / attempts if attempts > 0 else 0
            }
        else:
            # Fallback to any valid instances we found
            result = {}
            if len(valid_instances) > 0:
                result['most_complex_instance'] = valid_instances[0]
                result['least_complex_instance'] = valid_instances[-1]
            else:
                result['most_complex_instance'] = original_instance
                result['least_complex_instance'] = original_instance

            result.update({
                'total_attempts': attempts,
                'success_rate': len(valid_instances) / attempts if attempts > 0 else 0
            })
            return result

    def _count_feature_differences(self, instance1: pd.DataFrame, instance2: pd.DataFrame) -> int:
        """Count meaningful feature differences between instances."""
        differences = 0
        row1 = instance1.iloc[0]
        row2 = instance2.iloc[0]

        for feature in self.actionable_features:
            if feature in instance1.columns and feature in instance2.columns:
                val1, val2 = row1[feature], row2[feature]

                if feature in self.categorical_features:
                    if val1 != val2:
                        differences += 1
                else:
                    # For numerical features, use threshold
                    threshold = self.data[feature].std() * 0.2
                    if abs(val1 - val2) > threshold:
                        differences += 1

        return differences

    def _sort_by_complexity(self, original_instance: pd.DataFrame,
                            instances: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Sort instances by complexity (categorical changes prioritized)."""
        complexities = []

        for idx, row in instances.iterrows():
            instance_df = pd.DataFrame([row], columns=instances.columns)

            # Count categorical vs numerical changes
            categorical_changes = 0
            numerical_changes = 0

            for feature in self.actionable_features:
                if feature in original_instance.columns and feature in instance_df.columns:
                    orig_val = original_instance.iloc[0][feature]
                    new_val = instance_df.iloc[0][feature]

                    if feature in self.categorical_features:
                        if orig_val != new_val:
                            categorical_changes += 1
                    else:
                        threshold = self.data[feature].std() * 0.2
                        if abs(orig_val - new_val) > threshold:
                            numerical_changes += 1

            # Complexity score: prioritize categorical changes heavily
            complexity = categorical_changes * 3.0 + numerical_changes * 1.0
            complexities.append(complexity)

        # Sort by complexity
        instances_with_complexity = instances.copy()
        instances_with_complexity['complexity'] = complexities
        sorted_instances = instances_with_complexity.sort_values('complexity', ascending=False)

        most_complex = pd.DataFrame([sorted_instances.iloc[0].drop('complexity')],
                                    columns=instances.columns)
        least_complex = pd.DataFrame([sorted_instances.iloc[-1].drop('complexity')],
                                     columns=instances.columns)

        return {
            'most_complex_instance': most_complex,
            'least_complex_instance': least_complex
        }


def generate_manifold_categorical_instances(data: pd.DataFrame, model,
                                            original_instance: pd.DataFrame,
                                            categorical_features: List[str],
                                            actionable_features: List[str] = None,
                                            target_count: int = 10,
                                            min_feature_differences: int = 2,
                                            max_feature_differences: int = 2) -> Dict[str, Any]:
    """Convenience function for generating manifold-aware categorical instances."""
    generator = ManifoldAwareCategoricalGenerator(
        data=data,
        model=model,
        categorical_features=categorical_features,
        actionable_features=actionable_features,
        min_feature_differences=min_feature_differences,
        max_feature_differences=max_feature_differences
    )

    return generator.generate_instances(
        original_instance=original_instance,
        target_count=target_count
    )
