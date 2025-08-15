import os
import pickle
from typing import List
import json

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
                 categorical_mapping: dict = None,
                 dataset_name: str = ""):
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
            dataset_name: The name of the dataset for selecting appropriate explanations.
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
        self.dataset_name = dataset_name

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
                            lambda x: feature_mapping[int(float(x))] if int(float(x)) < len(feature_mapping) else x
                        )

        """# Plot the profiles
        if self.categorical_features and cat_profiles is not None:
            print("Plotting categorical profiles...")
            cat_profiles.plot()
        
        if self.numerical_features and num_profiles is not None:
            print("Plotting numerical profiles...")
            num_profiles.plot()"""

        # TODO: This was plotted and analyzed by hand. Find a way to automatically analyze the graphs ... LLM prompt?

        # Adult dataset explanations
        adult_feature_to_trend_mapping = {
            "InvestmentOutcome": "<p>On Average, the most important value of <strong>Investment Outcome</strong> is <strong>Major Gain (above 5K$)</strong>, which is most strongly linked to higher income.</p> <p><strong>Major Loss</strong> and <strong>Minor Loss</strong> are also linked to higher income, but less strongly, while <strong>No Investment</strong> and <strong>Minor Gain</strong> are slightly linked to lower income.</p>",

            "MaritalStatus": "<p>On Average, the most important value of <strong>Marital Status</strong> is <strong>Married</strong>, which is most often associated with higher income, while <strong>Single</strong> is more often linked to lower income.</p>",

            "EducationLevel": "<p>On Average, the most important value of <strong>Education Level</strong> is <strong>Bachelor’s Degree</strong>, which is most often linked to higher income.</p> <p><strong>Associate’s Degree</strong> and <strong>High School Graduate</strong> are also linked to higher income, but to a lesser extent, while <strong>Primary Education</strong> and <strong>Middle School</strong> are linked to lower income.</p> ",

            "Occupation": "<p>On Average, the most important values of <strong>Occupation</strong> are <strong>Professional</strong> and <strong>White-Collar</strong> jobs, which are most often linked to higher income.</p> <p><strong>Sales</strong> and <strong>Military</strong> jobs are also linked to slightly higher income, but less strong, while <strong>Blue-Collar</strong> and <strong>Service</strong> jobs are more often slightly linked to lower income.</p>",

            "WorkLifeBalance": "<p>On Average, the differences in <strong>Work Life Balance</strong> do not seem to have an effect on income, as all categories (<strong>Good, Fair, Poor</strong>) show no difference and impact.</p> <p>This suggests that <strong>Work Life Balance does not play a role</strong> in determining income in this dataset.</p>",

            "Age": "<p>On Average, the most important Age range for higher income is between <strong>40–50 years</strong>, where people are most likely to earn more.</p> <p>Before 40, income levels tend to increase with age, while after 50, they remain more stable.</p>",

            "WeeklyWorkingHours": "<p>On Average, the most important value of <strong>Weekly Working Hours</strong> is working <strong>more than 40 hours per week</strong>, which is strongly linked to higher income. After 40, income levels tend to increase with more hours while after 50, they remain more stable.</p>"
        }

        # Diabetes dataset explanations  
        diabetes_feature_to_trend_mapping = {
            "Age": "<p>On average, people between <strong>40 and 60 years old</strong> are most likely to have diabetes.</p> <p>People under 30 are less likely to have it, and after 60, the chance goes down a little but stays higher than in younger people.</p>",

            "Bmi": "<p>On average, people with a <strong>BMI over 27</strong> are more likely to have diabetes. The chance <strong>increases a lot between 27 and 45</strong>, and then stays about the same after that.</p> <p>People with a BMI under 27 are less likely to have it.</p>",

            "Glucose": "<p>On average, people with <strong>glucose (glucose) levels over 120</strong> are much more likely to have diabetes. The chance keeps going up until around <strong>160</strong>, and then stays high.</p> <p>People with lower glucose levels have a much lower chance.</p>",

            "Pregnancies": "<p>On average, people who have had <strong>more pregnancies</strong> tend to be slightly more likely to have diabetes.</p> <p>The effect is not strong, but there is a small upward trend with the number of pregnancies.</p>",

            "Bloodpressure": "<p>On average, <strong>blood pressure</strong> does not seem to change the chance of having diabetes.</p> <p>The risk stays about the same across different blood pressure levels.</p>",

            "Diabetespedigreefunction": "<p>On average, people with a <strong>higher family history score for diabetes</strong> have a slightly higher chance of getting it.</p> <p>This effect is small and levels off once the score goes above 1.0.</p>",

            "Insulin": "<p>On average, people with <strong>insulin levels above 150</strong> have a slightly higher chance of having diabetes.</p> <p>Beyond that level, the chance stays about the same, showing only a weak effect.</p>",

            "Dailywaterintake": "<p>On average, a person's <strong>daily water intake</strong> does not seem to affect the chance of having diabetes.</p> <p>The risk stays about the same regardless of how much water someone drinks.</p>"
        }

        # Select the appropriate feature mapping based on dataset name
        if 'adult' in self.dataset_name.lower():
            feature_to_trend_mapping = adult_feature_to_trend_mapping
        elif 'diabetes' in self.dataset_name.lower():
            feature_to_trend_mapping = diabetes_feature_to_trend_mapping
        else:
            # Default to diabetes mapping if dataset name is not recognized
            print(f"Warning: Unknown dataset name '{self.dataset_name}'. Using diabetes explanations as default.")
            feature_to_trend_mapping = diabetes_feature_to_trend_mapping

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


if __name__ == "__main__":
    """
    Main function to run PDP plot generation for the diabetes dataset.
    You can run this directly in PyCharm.
    """

    # Load the diabetes dataset and model
    print("Loading diabetes dataset and model...")

    # Load the trained model
    model_path = "../../data/diabetes/diabetes_model_rf.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load the background data
    background_data = pd.read_csv("../../data/diabetes/diabetes_train.csv", index_col=0)

    # Load the categorical mapping
    with open("../../data/diabetes/categorical_mapping.json", 'r') as f:
        categorical_mapping = json.load(f)

    # Convert string keys to integers for categorical_mapping
    categorical_mapping = {int(k): v for k, v in categorical_mapping.items()}

    # Prepare the target variable
    target_col = 'Y'
    ys = background_data[target_col]
    background_data_features = background_data.drop(columns=[target_col])

    # Define feature categories
    categorical_features = ['BloodGroup']
    numerical_features = ['Age', 'Pregnancies', 'Glucose', 'Bloodpressure', 'Insulin', 'Bmi', 'Diabetespedigreefunction']

    print(f"Background data shape: {background_data_features.shape}")
    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")

    # Create PDP explainer
    print("Creating PDP explainer...")
    pdp_explainer = PdpExplanation(
        model=model,
        background_data=background_data_features,
        ys=ys,
        cache_location="../../cache/diabetes-pdp-demo.pkl",
        feature_names=list(background_data_features.columns),
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        categorical_mapping=categorical_mapping
    )

    # Generate explanations and plots (this will automatically plot the profiles)
    print("Generating PDP explanations and plots...")
    explanations = pdp_explainer.run_explanation()

    print("\nPDP plotting complete!")
    print("\nGenerated explanations for features:")
    for feature_name in explanations.keys():
        print(f"- {feature_name}")
