import warnings

import gin
import numpy as np
import pandas as pd
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
        self.feature_names = feature_names
        self.ys = ys
        self.explainer = dx.Explainer(self.model, self.background_data, y=self.ys)

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
            rf_profile = self.explainer.predict_profile(instance)
            cps[d] = rf_profile
        return cps

    def get_explanation(self, data_df, feature_name=None, as_plot=True):
        id = data_df.index[0]
        cp_data = self.get_explanations([id], self.background_data, save_to_cache=True)
        if not as_plot:
            return cp_data
        else:
            # TODO: Handle categorical features
            fig = cp_data[id].plot(variables=[feature_name], show=False)
            return fig
