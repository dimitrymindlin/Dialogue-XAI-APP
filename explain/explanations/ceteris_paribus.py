import warnings

import gin
import numpy as np
import pandas as pd
from anchor.anchor_explanation import AnchorExplanation
from matplotlib import pyplot as plt
from tqdm import tqdm

from explain.explanation import Explanation
import dalex as dx

def plot_cp(names, values):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the y-axis labels
    ax.set_yticklabels(names)

    # Set the y-axis ticks
    ax.set_yticks(range(len(names)))

    # Set the x-axis range
    ax.set_xlim(min(values), max(values))

    plt.subplots_adjust(left=0.5)

    # Plot the bars
    for i, val in enumerate(values):
        color = 'green' if val > 0 else 'red'
        ax.barh(i, val, color=color)
    plt.show()

@gin.configurable
class CeterisParibus(Explanation):
    """This class generates CeterisParibus explanations for tabular data."""

    def __init__(self,
                 model,
                 background_data: pd.DataFrame,
                 ys: pd.DataFrame,
                 class_names: dict,
                 cache_location: str = "./cache/ceterisparibus-tabular.pkl",
                 feature_names: list = None):
        """

        Args:
            model: The model to explain.
            data: the background dataset provided at pandas df
            value of the categorical features. Every feature that is not in
            this map will be considered as ordinal or continuous, and thus discretized.

        """
        super().__init__(cache_location, class_names)
        self.background_data = background_data
        self.model = model
        """self.categorical_names = categorical_names
        self.class_names = list(class_names.values())"""
        self.feature_names = feature_names
        self.ys = ys

    def run_explanation(self,
                        current_data: pd.DataFrame):
        """Generate tabular dice explanations.

        Arguments:
            current_data: The data to generate explanations for in pandas df.
            desired_class: The desired class of the cfes. If None, will use the default provided
                           at initialization.
        Returns:
            explanations: The generated cf explanations.
        """

        cps = {}
        for d in tqdm(list(current_data.index)):
            instance = current_data.loc[[d]]
            # Explain here
            #observation = pd.DataFrame(instance).T
            # pred = np.array([np.argmax(self.model.predict(instance))])
            exp = dx.Explainer(self.model, self.background_data, y=self.ys)
            """bd = exp.predict_parts(observation, type='break_down', label=ys.iloc[0])
            bd_interactions= exp.predict_parts(observation, type='break_down_interactions', label="John+")
            bd.plot(bd_interactions)"""
            rf_profile = exp.predict_profile(instance)
            print(rf_profile.result.head())
            # Create bar plot from rf_profile
            rf_profile.plot(variable_type='categorical')
            plt.show()
            print()

            output = self.explainer.explain_instance(data_x[0],
                                                     self.model.predict,
                                                     threshold=0.95,
                                                     max_anchor_size=3)
            return output

            cps[d] = cur_cp
        return cps

    def summarize_explanations(self,
                               data: pd.DataFrame,
                               ids_to_regenerate: list[int] = None,
                               filtering_text: str = None,
                               save_to_cache: bool = False):
        """Summarizes explanations for Anchor tabular.

        Arguments:
            data: pandas df containing data.
            ids_to_regenerate:
            filtering_text:
            save_to_cache:
        Returns:
            summary: a string containing the summary.
        """

        if ids_to_regenerate is None:
            ids_to_regenerate = []
        # Not needed in question selection case
        """if data.shape[0] > 1:
            return ("", "I can only compute Anchors for single instances at a time."
                        " Please narrow down your selection to a single instance. For example, you"
                        " could specify the id of the instance to want to figure out how to change.")"""

        ids = list(data.index)
        key = ids[0]

        explanation = self.get_explanations(ids,
                                            data,
                                            ids_to_regenerate=ids_to_regenerate,
                                            save_to_cache=save_to_cache)
        exp = explanation[key]
        output_string = ""
        output_string += "By fixing the following attributes, the prediction stays the same even though other attributes are changed:"
        output_string += "<br><br>"

        additional_options = "Here are some more options to change the prediction of"
        additional_options += f" instance id {str(key)}.<br><br>"

        output_string += ' AND <br><br>'.join(exp.names())

        return additional_options, output_string
