"""Compares many explanations to determine the best one."""
import copy
from dataclasses import dataclass
from functools import partial
from typing import Union, Any

import heapq
import numpy as np
import pandas as pd

from explain.mega_explainer.lime_explainer import Lime
from explain.mega_explainer.perturbation_methods import NormalPerturbation
from explain.mega_explainer.shap_explainer import SHAPExplainer


@dataclass
class MegaExplanation:
    """The return format for the mega explanation!"""
    list_exp: list
    score: float
    label: int
    best_explanation_type: str
    agree: bool


def conv_disc_inds_to_char_enc(discrete_feature_indices: list[int], n_features: int):
    """Converts an array of discrete feature indices to a char encoding.

    Here, the ith value in the returned array is 'c' or 'd' for whether the feature is
    continuous or discrete respectively.

    Args:
        discrete_feature_indices: An array like [0, 1, 2] where the ith value corresponds to
                                  whether the arr[i] column in the data is discrete.
        n_features: The number of features in the data.
    Returns:
        char_encoding: An encoding like ['c', 'd', 'c'] where the ith value indicates whether
                       that respective column in the data is continuous ('c') or discrete ('d')
    """
    # Check to make sure (1) feature indices are integers and (2) they are unique
    error_message = "Features all must be type int but are not"
    assert all(isinstance(f, int) for f in discrete_feature_indices), error_message
    error_message = "Features indices must be unique but there are repetitions"
    assert len(set(discrete_feature_indices)) == len(discrete_feature_indices), error_message
    # Perform conversion
    char_encoding = ['e'] * n_features
    for i in range(len(char_encoding)):
        if i in discrete_feature_indices:
            char_encoding[i] = 'd'
        else:
            char_encoding[i] = 'c'
    # In case something still went wrong
    assert 'e' not in char_encoding, 'Error in char encoding processing!'
    return char_encoding


class Explainer:
    """
    Explainer is the orchestrator class that drives the logic for selecting
    the best possible explanation from the set of explanation methods.
    """

    def __init__(self,
                 model,
                 explanation_dataset: np.ndarray,
                 predict_fn: Any,
                 feature_names: list[str],
                 discrete_features: list[int],
                 use_selection: bool = True,
                 use_tree_shap: bool = False):
        """
        Init.

        Args:
            explanation_dataset: background data, given as numpy array
            predict_fn: the callable black box model. the model should be callable via
                               explanation_model(data) to generate prediction probabilities
            feature_names: the feature names
            discrete_features: The indices of the discrete features in the dataset. Note, in the
                               rest of the repo, we adopt the terminology 'categorical features'.
                               However, in this mega_explainer sub folder, we adopt the term
                               `discrete features` to describe these features.
            use_selection: Whether to use the explanation selection. If false, uses lime.
            use_tree_shap: Whether to use tree shap or kernel shap. If true, uses tree shap.
        """
        if isinstance(explanation_dataset, pd.DataFrame):
            # Creating a copy of the explanation dataset... For large datasets, this may be an
            # issue. However, converting from pd.DataFrame to np.ndarray in this way seems
            # to overwrite the underlying dataset, causing potentially confusing issues
            explanation_dataset = copy.deepcopy(explanation_dataset)
            explanation_dataset = explanation_dataset.to_numpy()
        else:
            arr_type = type(explanation_dataset)
            message = f"Data must be pd.DataFrame or np.ndarray, not {arr_type}"
            assert isinstance(explanation_dataset, np.ndarray), message

        self.model = model  # sklearn model or pipeline
        self.data = explanation_dataset
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.use_tree_shap = use_tree_shap

        # We store a dictionary containing all the explanation methods we are going to compare
        # in order to figure out "the best" explanation. These methods are initialized and
        # stored here

        lime_template = partial(Lime,
                                model=self.predict_fn,
                                data=self.data,
                                feature_names=self.feature_names,
                                discrete_features=discrete_features)

        # Generate explanations with many lime kernels
        if self.use_tree_shap:
            # Use tree shap for exact shap values
            shap_explainer = SHAPExplainer(self.model, self.data, method='tree')
            available_explanations = {"shap": shap_explainer}
        else:
            # Use kernel shap for approximate shap values and select the best kernel width
            shap_explainer = SHAPExplainer(self.model, self.predict_fn, self.data)
            if use_selection:
                kernel_widths = [0.25, 0.50, 0.75, 1.0]
            else:
                kernel_widths = [0.75]

            available_explanations = {}
            for width in kernel_widths:
                name = f"lime_{round(width, 3)}"
                available_explanations[name] = lime_template(kernel_width=width)
            # add shap
            if use_selection:
                available_explanations["shap"] = shap_explainer

        self.explanation_methods = available_explanations

        # Can we delete this line?
        self.get_explanation_methods = {}

        # TODO(satya): change this to be inputs to __init__
        # The criteria used to perturb the explanation point and determine which explanations
        # are the most faithful
        self.perturbation_mean = 0.0
        self.perturbation_std = 0.05
        self.perturbation_flip_percentage = 0.03
        self.perturbation_max_distance = 0.4

        # This is a bit clearer, instead of making users use this representation + is the way
        # existing explanation packages (e.g., LIME do it.)
        self.feature_types = conv_disc_inds_to_char_enc(discrete_feature_indices=discrete_features,
                                                        n_features=self.data.shape[1])

        # Set up the Gaussian perturbations
        self.perturbation_method = NormalPerturbation("tabular",
                                                      mean=self.perturbation_mean,
                                                      std=self.perturbation_std,
                                                      flip_percentage=self.perturbation_flip_percentage)

    @staticmethod
    def _arr(x) -> np.ndarray:
        """Converts x to a numpy array."""
        return np.array(x)

    def _compute_faithfulness_auc(self, data, explanation, c_label, k, metric="topk"):
        """Computes AUC for faithfulness scores, perturbing top k (where k is an array)."""
        faithfulness = 0
        for k_i in k:
            # Create boolean mask for top k features
            top_k_map = np.ones(len(explanation), dtype=bool)
            top_k_indices = np.argsort(np.abs(explanation))[-k_i:]
            top_k_map[top_k_indices] = False

            if metric == "topk":
                faithfulness += self._compute_faithfulness_topk(data, c_label, top_k_map)
            else:
                faithfulness += self._compute_faithfulness_topk(data, c_label, ~top_k_map)
        return faithfulness

    def _compute_faithfulness_topk(self, x, label, top_k_mask, num_samples: int = 10_000):
        """Approximates the expected local faithfulness of the explanation in a neighborhood."""
        perturb_args = {
            "original_sample": x[0],
            "feature_mask": top_k_mask,
            "num_samples": num_samples,
            "max_distance": self.perturbation_max_distance,
            "feature_metadata": self.feature_types
        }

        # Get perturbed instances
        x_perturbed = self.perturbation_method.get_perturbed_inputs(**perturb_args)

<<<<<<< HEAD
        # TODO(satya): Could you make these lines more readable?
        y = self._arr([i[label] for i in self._arr(self.predict_fn(x.reshape(1, -1)))])
        # Model expects numpy arrays for german dataset. (Dimi)
        y_perturbed = self._arr([i[label] for i in self._arr(self.predict_fn(x_perturbed.float().numpy()))])
=======
        y = self._arr([i[label] for i in self._arr(self.predict_fn(x.reshape(1, -1)))])
        y_perturbed = self._arr([i[label] for i in self._arr(self.predict_fn(x_perturbed))])
>>>>>>> main

        return np.mean(np.abs(y - y_perturbed), axis=0)

    @staticmethod
    def check_exp_data_shape(data_x: np.ndarray) -> np.ndarray:
        """Checks to make sure the data being explained is a single instance and 1-dim."""
        # Check to make sure data_x is an individual sample
        data_x_shape = data_x.shape
        if len(data_x_shape) > 1:
            n_samples = data_x_shape[0]
            if n_samples > 1:
                message = f"Data must be individual sample, but has shape {data_x_shape}"
                assert len(data_x_shape) == 1, message
        elif len(data_x_shape) == 1:
            data_x = data_x.reshape(1, -1)
        return data_x

    def explain_instance(self,
                         data: Union[np.ndarray, pd.DataFrame],
                         top_k_starting_pct: float = 0.2,
                         top_k_ending_pct: float = 0.5,
                         epsilon: float = 1e-4,
                         return_fidelities: bool = False,
                         explain_only_most_likely=True) -> MegaExplanation:
        """Computes the explanation.

        This function computes the explanation. It calls several explanation methods, computes
        metrics over the different methods, computes an aggregate score and returns the best one.

        Args:
            return_fidelities: Whether to return explanation fidelities
            epsilon:
            top_k_ending_pct: The percentage of top k features to use for the explanation
            top_k_starting_pct: The percentage of top k features to use for the explanation
            data: The instance to explain. If given as a pd.DataFrame, will be converted to a
                  np.ndarray
            explain_only_most_likely: Whether to only explain the most likely class or all classes
        Returns:
            explanations: the final explanations, selected based on most faithful
        """
        if not isinstance(data, np.ndarray):
            try:
                data = data.to_numpy()
            except Exception as exp:
                message = f"Data not type np.ndarray, failed to convert with error {exp}"
                raise NameError(message)

        explanations, scores, fidelity_scores_topk = {}, {}, {}
        # Makes sure data is formatted correctly
        formatted_data = self.check_exp_data_shape(data)

        if explain_only_most_likely:
            # Get the prediction scores for most likely class
            predictions = self.predict_fn(formatted_data)[0]
            num_classes = predictions.shape[-1]
            class_indices = [np.argmax(predictions)]
        else:
            # Get the prediction scores for all classes
            predictions = self.predict_fn(formatted_data)[0]
            num_classes = predictions.shape[-1]
            class_indices = list(range(num_classes))

        # Gets indices of 20-50% of data
        lower_index = int(formatted_data.shape[1] * top_k_starting_pct)
        upper_index = int(formatted_data.shape[1] * top_k_ending_pct)
        k = list(range(lower_index, upper_index))

        # Iterate over all classes
        for class_index in class_indices:
            explanations[class_index], scores[class_index], fidelity_scores_topk[class_index] = {}, {}, {}

            # Explanation logic adapted for multi-class
            if len(self.explanation_methods.keys()) > 1:
                for method in self.explanation_methods.keys():
                    cur_explainer = self.explanation_methods[method]
                    cur_expl, score = cur_explainer.get_explanation(formatted_data, label=class_index)

<<<<<<< HEAD
                    explanations[class_index][method] = cur_expl.squeeze(0)
=======
                    # Handle both squeezable and non-squeezable arrays
                    if len(cur_expl.shape) > 1:
                        explanations[class_index][method] = cur_expl.squeeze()
                    else:
                        explanations[class_index][method] = cur_expl

>>>>>>> main
                    scores[class_index][method] = score
                    fidelity_scores_topk[class_index][method] = self._compute_faithfulness_auc(formatted_data,
                                                                                               explanations[class_index][
                                                                                                   method],
                                                                                               class_index,
                                                                                               k,
                                                                                               metric="topk")
            else:
                # If only one explanation method is available, use it
                method = list(self.explanation_methods.keys())[0]
                cur_explainer = self.explanation_methods[method]
                cur_expl, score = cur_explainer.get_explanation(formatted_data, label=class_index)
                
                # Handle both squeezable and non-squeezable arrays
                if len(cur_expl.shape) > 1:
                    explanations[class_index][method] = cur_expl.squeeze()
                else:
                    explanations[class_index][method] = cur_expl
                    
                scores[class_index][method] = score
<<<<<<< HEAD

        # Iterate over each explanation method and compute fidelity scores of topk
        # and non-topk features per the method
=======
>>>>>>> main

        if return_fidelities:
            return fidelity_scores_topk

        # Initialize dictionary to store the best explanation for each class
        if len(self.explanation_methods.keys()) > 1:
            best_explanations_per_class = {}

            for class_index in explanations.keys():
                class_fidelity_scores = {method: fidelity_scores_topk[class_index][method] for method in
                                         fidelity_scores_topk[class_index]}
                if len(class_fidelity_scores) >= 2:
                    top2_methods = heapq.nlargest(2, class_fidelity_scores, key=class_fidelity_scores.get)

                    diff = abs(class_fidelity_scores[top2_methods[0]] - class_fidelity_scores[top2_methods[1]])
                    if diff > epsilon:
                        best_method = top2_methods[0]
                    else:
                        # Adapt the stability computation for multi-class context if necessary
                        highest_fidelity = self.compute_stability(formatted_data,
                                                                  explanations[class_index][top2_methods[0]],
                                                                  self.explanation_methods[top2_methods[0]], class_index, k)
                        second_highest_fidelity = self.compute_stability(formatted_data,
                                                                         explanations[class_index][top2_methods[1]],
                                                                         self.explanation_methods[top2_methods[1]],
                                                                         class_index, k)

                        best_method = top2_methods[0] if highest_fidelity > second_highest_fidelity else top2_methods[1]
                else:
                    # Default to a specific method if not enough data is available for comparison
                    best_method = "lime_0.75"
                # Store the best explanation for the class
                best_explanations_per_class[class_index] = {
<<<<<<< HEAD
                    'explanation': explanations[class_index][best_method].numpy(),
=======
                    'explanation': explanations[class_index][best_method],
>>>>>>> main
                    'method': best_method,
                    'score': scores[class_index][best_method],
                    'agree': diff <= epsilon if 'diff' in locals() else True
                }
        else:
            # If only one explanation method is available, use it
            best_explanations_per_class = {}
            for class_index in explanations.keys():
                best_method = list(self.explanation_methods.keys())[0]
                best_explanations_per_class[class_index] = {
<<<<<<< HEAD
                    'explanation': explanations[class_index][best_method].numpy(),
=======
                    'explanation': explanations[class_index][best_method],
>>>>>>> main
                    'method': best_method,
                    'score': scores[class_index][best_method],
                    'agree': True
                }

        # Format return
        # Initialize a container for the final explanations for all classes
        final_explanations_for_all_classes = {}

        # Iterate over each class in best_explanations_per_class to format the final explanation
        for class_index, best_explanation_info in best_explanations_per_class.items():
            # Extract information for the current class's best explanation
            best_exp_array = best_explanation_info['explanation']
            best_method = best_explanation_info['method']
            best_method_score = best_explanation_info['score']
            agree = best_explanation_info['agree']

            # Format the explanation for the current class
            final_explanations_for_all_classes[class_index] = self._format_explanation(best_exp_array,
                                                                                       class_index,
                                                                                       best_method_score,
                                                                                       best_method,
                                                                                       agree)

        # Extract inner dict for explain_only_most_likely
        if explain_only_most_likely:
            final_explanations_for_all_classes = final_explanations_for_all_classes[class_indices[0]]
            fidelity_scores_topk = fidelity_scores_topk[class_indices[0]]

        if return_fidelities:
            return final_explanations_for_all_classes, fidelity_scores_topk
        else:
            return final_explanations_for_all_classes

    def compute_stability(self, data, baseline_explanation, explainer, label, top_k_inds):
        """Computes the AUC stability scores."""
        stability = 0
        for k_i in top_k_inds:
            stability += self.compute_stability_topk(data,
                                                   baseline_explanation,
                                                   explainer,
                                                   label,
                                                   k_i)
        return stability

    def compute_stability_topk(self, data, baseline_explanation, explainer, label, top_k, num_perturbations=100):
        """Computes the stability score."""
        perturb_args = {
            "original_sample": data[0],
            "feature_mask": np.zeros(len(baseline_explanation), dtype=bool),
            "num_samples": num_perturbations,
            "max_distance": self.perturbation_max_distance,
            "feature_metadata": self.feature_types
        }

        # Get the perturbed instances
        x_perturbed = self.perturbation_method.get_perturbed_inputs(**perturb_args)

        # Get top k indices for baseline explanation
        topk_base = np.argsort(np.abs(baseline_explanation))[-top_k:]
        stability_value = 0
        
        for perturbed_sample in x_perturbed:
            explanation_perturbed_input, _ = explainer.get_explanation(perturbed_sample[None, :],
                                                                     label=label)
            abs_expl = np.abs(explanation_perturbed_input)
            topk_perturbed = np.argsort(abs_expl)[-top_k:]

            jaccard_distance = len(np.intersect1d(topk_base, topk_perturbed)) / len(
                np.union1d(topk_base, topk_perturbed))
            stability_value += jaccard_distance

        mean_stability = stability_value / num_perturbations

        return mean_stability

    def _format_explanation(self, explanation: list, label: int, score: float, best_method: str, agree: bool):
        """Formats the explanation in LIME format to be returned."""
        list_exp = []

        # combine feature importances & features names into tuples of feature name and feature
        # importance
        for feature_name, feature_imp in zip(self.feature_names, explanation):
            list_exp.append((feature_name, feature_imp))

        # Sort the explanations so that the most important features are first
        list_exp.sort(key=lambda x: abs(x[1]), reverse=True)

        # Format the output
        return_exp = MegaExplanation(list_exp=list_exp,
                                     label=label,
                                     score=score,
                                     best_explanation_type=best_method,
                                     agree=agree)

        return return_exp
