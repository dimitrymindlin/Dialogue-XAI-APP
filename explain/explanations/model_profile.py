import os
import pickle
from typing import List

import gin
import pandas as pd
import dalex as dx


@gin.configurable
class PdpExplanation:
    """This class generates CeterisParibus explanations for tabular data."""

    def __init__(self,
                 model,
                 background_data: pd.DataFrame,
                 ys: pd.DataFrame,
                 cache_location: str = "./cache/pdp-tabular.pkl",
                 feature_names: list = None,
                 categorical_features: List[str] = None,
                 numerical_features: List[str] = None,
                 categorical_mapping: dict = None):
        """

        Args:
            model: The model to explain.
            background_data: The background dataset provided as a pandas df.
            ys: The target variable data.
            cache_location: The location to save the cache.
            feature_names: The names of the features.
            categorical_features: List of categorical feature names.
            numerical_features: List of numerical feature names.
            categorical_mapping: Dictionary mapping categorical feature indices to category names.
        """
        self.cache_location = cache_location
        self.cache = self.load_cache()
        self.background_data = background_data
        self.model = model
        self.feature_names = feature_names
        self.ys = ys
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.categorical_mapping = categorical_mapping

    def load_cache(self):
        """Load the cache from a file."""
        if os.path.exists(self.cache_location):
            with open(self.cache_location, 'rb') as file:
                return pickle.load(file)
        return {}

    def save_cache(self):
        """Save the current cache to a file."""
        with open(self.cache_location, 'wb') as file:
            pickle.dump(self.cache, file)

    def get_explanations(self):
        """Gets explanations corresponding to ids in data, where data is a pandas df.

        This routine will pull explanations from the cache if they exist. If
        they don't it will call run_explanation on these ids.
        """
        exp = self.load_cache()
        # Check if dict is empty
        if not exp:
            exp = self.run_explanation()
        return exp

    def run_explanation(self, use_cache: bool = True):
        """Generate ceteris paribus explanations.

        Returns:
            explanations: The generated counterfactual explanations.
        """
        self.explainer = dx.Explainer(self.model, self.background_data, y=self.ys)

        cat_profiles = self.explainer.model_profile(type='partial', variable_type='categorical', N=800,
                                                    variables=self.categorical_features)
        num_profiles = self.explainer.model_profile(type='accumulated', variable_type='numerical',
                                                    variables=self.numerical_features)

        # Replace the feature values in `_x_` with the original names from `categorical_mapping`
        if self.categorical_mapping:
            for feature in self.categorical_features:
                if feature in cat_profiles.result['_vname_'].values:
                    feature_mask = cat_profiles.result['_vname_'] == feature
                    feature_indices = cat_profiles.result.loc[feature_mask, '_x_']

                    # Get corresponding mapping list
                    feature_mapping = self.categorical_mapping.get(self.background_data.columns.get_loc(feature))

                    if feature_mapping:
                        cat_profiles.result.loc[feature_mask, '_x_'] = feature_indices.apply(
                            lambda x: feature_mapping[int(x)] if int(x) < len(feature_mapping) else x
                        )

        """cat_profiles.plot()
        num_profiles.plot()"""

        # TODO: This was plotted and analyzed by hand. Find a way to automatically analyze the graphs ... LLM prompt?

        feature_to_trend_mapping = {
            "InvestmentOutcome": "<p>On Average, the most important value of <strong>Investment Outcome</strong> is <strong>Major Gain (above 5K$)</strong>, which is most strongly linked to higher income.</p> <p><strong>Major Loss</strong> and <strong>Minor Loss</strong> are also linked to higher income, but less strongly, while <strong>No Investment</strong> and <strong>Minor Gain</strong> are slightly linked to lower income.</p>",

            "MaritalStatus": "<p>On Average, the most important value of <strong>Marital Status</strong> is <strong>Married</strong>, which is most often associated with higher income, while <strong>Single</strong> is more often linked to lower income.</p>",

            "EducationLevel": "<p>On Average, the most important value of <strong>Education Level</strong> is <strong>Bachelor’s Degree</strong>, which is most often linked to higher income.</p> <p><strong>Associate’s Degree</strong> and <strong>High School Graduate</strong> are also linked to higher income, but to a lesser extent, while <strong>Primary Education</strong> and <strong>Middle School</strong> are linked to lower income.</p> ",

            "Occupation": "<p>On Average, the most important values of <strong>Occupation</strong> are <strong>Professional</strong> and <strong>White-Collar</strong> jobs, which are most often linked to higher income.</p> <p><strong>Sales</strong> and <strong>Military</strong> jobs are also linked to slightly higher income, but less strong, while <strong>Blue-Collar</strong> and <strong>Service</strong> jobs are more often slightly linked to lower income.</p>",

            "WorkLifeBalance": "<p>On Average, the differences in <strong>Work Life Balance</strong> do not seem to have an effect on income, as all categories (<strong>Good, Fair, Poor</strong>) show no difference and impact.</p> <p>This suggests that <strong>Work Life Balance does not play a role</strong> in determining income in this dataset.</p>",

            "Age": "<p>On Average, the most important Age range for higher income is between <strong>40–50 years</strong>, where people are most likely to earn more.</p> <p>Before 40, income levels tend to increase with age, while after 50, they remain more stable.</p>",

            "WeeklyWorkingHours": "<p>On Average, the most important value of <strong>Weekly Working Hours</strong> is working <strong>more than 40 hours per week</strong>, which is strongly linked to higher income. After 40, income levels tend to increase with more hours while after 50, they remain more stable.</p>"
        }

        if use_cache:
            self.cache['pdp_dict'] = feature_to_trend_mapping
            self.save_cache()

        return feature_to_trend_mapping

    def get_explanation(self, feature_name):
        # Load the cache or run the explanation and return the result for a single feature
        exp = self.get_explanations()['pdp_dict']
        exp_string = exp[feature_name] if feature_name in exp else None
        if exp_string is None:
            raise KeyError(f"Feature {feature_name} not found in explanations.")
        return exp_string
