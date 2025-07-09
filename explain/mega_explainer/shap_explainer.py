"""Generates SHAP explanations."""
from collections import defaultdict
import numpy as np
import shap
from scipy.special import expit
from explain.mega_explainer.base_explainer import BaseExplainer
from lime.lime_tabular import explanation
from lime.submodular_pick import SubmodularPick


class SHAPExplainer(BaseExplainer):
    """The SHAP explainer"""

    def __init__(self,
                 model,
                 data: np.ndarray,
                 link: str = 'identity',
                 method='kernel'):
        """Init.

        Args:
            model: model object
            data: pandas data frame or numpy array
            link: str, 'identity' or 'logit'
        """
        super().__init__(model)

        # Store the data
        self.method = method

        if self.method == 'tree':
            self.data = data
        else:
            self.data = shap.kmeans(data, 25)

        # Use the SHAP kernel explainer in all cases. We can consider supporting
        # domain specific methods in the future.
        if method == 'tree':
            self.explainer = shap.TreeExplainer(model.named_steps["model"], model_output="raw", feature_perturbation="interventional")
        else:
            self.explainer = shap.KernelExplainer(self.model, self.data, link=link)

    def get_explanation(self, data_x: np.ndarray, label) -> tuple[np.ndarray, float]:
        """Gets the SHAP explanation.

        Args:
            label: The label to explain.
            data_x: data sample to explain. This sample is of type np.ndarray and is of shape
                    (1, dims).
        Returns:
            final_shap_values: SHAP values [dim (shap_vals) == dim (data_x)]
        """

        def get_shap_group_indices(feature_names):
            """
            Given a list of one-hot encoded feature names, extract the indices that need to be summed together
            to reconstruct SHAP values for original features.

            Args:
                feature_names (list of str): List of feature names from OneHotEncoder.

            Returns:
                dict: A dictionary mapping original feature names to lists of indices that should be summed.
            """
            feature_groups = defaultdict(list)

            # Iterate over feature names and assign indices to their original feature group
            for idx, feature in enumerate(feature_names):
                original_feature = feature.split('_')[0]  # Extract prefix before "_"
                feature_groups[original_feature].append(idx)

            return dict(feature_groups)

        def combine_shap_values_reduced(shap_values, shap_groups):
            """
            Groups SHAP values according to shap_groups by summing their values together,
            while keeping non-grouped values unchanged. The output has a reduced dimension.

            Args:
                shap_values (list or np.array): SHAP values for all features (1D array).
                shap_groups (dict): Dictionary mapping original features to the corresponding indices.

            Returns:
                np.array: Reduced SHAP values array where grouped values are summed,
                          and non-grouped values remain unchanged.
            """
            shap_values = np.array(shap_values).flatten()  # Ensure it's a 1D NumPy array
            total_indices = set(range(len(shap_values)))  # All indices in the SHAP array
            grouped_indices = set(idx for indices in shap_groups.values() for idx in indices)  # Grouped indices

            # Compute summed SHAP values for each group
            grouped_shap_values = {feature: np.sum(shap_values[indices]) for feature, indices in shap_groups.items()}

            # Identify ungrouped indices
            ungrouped_indices = sorted(total_indices - grouped_indices)

            # Construct final reduced SHAP values array
            final_shap_values = []

            # Add grouped values first
            for feature in shap_groups.keys():
                final_shap_values.append(grouped_shap_values[feature])

            # Add ungrouped values
            for idx in ungrouped_indices:
                final_shap_values.append(shap_values[idx])

            return np.array(final_shap_values)

        # Compute the shapley values on the **single** instance
        if self.method == 'tree':
            transformed_x = self.model[:-1].transform(data_x)
            shap_vals = self.explainer.shap_values(transformed_x)
            shap_vals = shap_vals[:, :, label]  # Select the relevant class

            # Extract the OneHotEncoder from the ColumnTransformer
            encoder = self.model.named_steps['preprocessor'].named_transformers_['one_hot']

            shap_groups = get_shap_group_indices(encoder.get_feature_names_out())

            final_shap_values = combine_shap_values_reduced(shap_vals, shap_groups)
        else:
            shap_vals = self.explainer.shap_values(data_x[0], nsamples=10_000, silent=True)

            # Ensure that we select the correct label, if shap values are computed on output prob. distribution
            if len(shap_vals) > 1:
                final_shap_values = shap_vals[label]
            else:
                final_shap_values = shap_vals

        # Convert base log-odds to probability for the specified class
        base_log_odds = self.explainer.expected_value[label]
        base_probability = expit(base_log_odds)
        return final_shap_values, base_probability

    def explain_instance(self, data_row, predict_fn, num_features, top_labels=None, **kwargs):
        """
        Make SHAPExplainer compatible with LIME's SubmodularPick by returning
        a LIME-style Explanation object built from SHAP values.
        """
        # Compute SHAP values and base probability for a dummy label 0
        shap_vals, base_prob = self.get_explanation(data_row.reshape(1, -1), label=0)

        # Create a LIME Explanation wrapper
        exp = explanation.Explanation(domain_mapper=None, mode='classification')
        # Ensure class_names is set so as_pyplot_figure can index correctly
        try:
            # If model is a sklearn Pipeline, extract the final estimator's classes_
            classes = self.model.named_steps["model"].classes_
        except Exception:
            # Otherwise, try model.classes_
            classes = getattr(self.model, "classes_", None)
        exp.class_names = [str(c) for c in classes] if classes is not None else ["0"]
        # Set the predicted probabilities
        exp.local_pred = predict_fn(data_row.reshape(1, -1))[0]
        # Use base probability as intercept (optional)
        exp.intercept = float(base_prob)
        exp.top_labels = [np.argmax(exp.local_pred)]

        # available_labels just returns our dummy label
        exp.available_labels = lambda: [0]
        # as_list returns pairs of (feature_name, shap_value)
        exp.as_list = lambda label=0: [
            (str(i), float(shap_vals[i])) for i in range(len(shap_vals))
        ]
        return exp

    def get_diverse_instance_ids(self,
                                 data_x: np.ndarray,
                                 label: int,
                                 num_instances: int = 5) -> list[int]:
        """
        Get instance indices by performing Submodular Pick over SHAP explanations.
        """
        # Use LIME's SubmodularPick with this SHAPExplainer as the explainer
        print(f"Running SHAP SP Lime for {num_instances} instances.")
        sp = SubmodularPick(
            explainer=self,
            data=data_x,
            predict_fn=self.model.predict_proba,
            method='full',
            num_exps_desired=num_instances,
            num_features=data_x.shape[1]
        )

        import matplotlib
        matplotlib.use('Agg')

        # Generate and display each explanation figure
        for i, exp in enumerate(sp.sp_explanations):
            fig = exp.as_pyplot_figure(label=exp.top_labels[0])
            fig.savefig(f'diverse_explanation_{i}.png')
        return sp.V
