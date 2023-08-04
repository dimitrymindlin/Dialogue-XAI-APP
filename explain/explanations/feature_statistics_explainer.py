import pandas as pd

from data.response_templates.feature_statistics_template import feature_statistics_template


class FeatureStatisticsExplainer:
    def __init__(self,
                 data: pd.DataFrame,
                 numerical_features: list,
                 feature_names: list,
                 categorical_mapping,
                 rounding_precision: int = 2,
                 feature_units: dict = None
                 ):
        self.data = data
        self.numerical_features = numerical_features
        self.feature_names = feature_names
        self.rounding_precision = rounding_precision
        self.categorical_mapping = categorical_mapping
        self.feature_units = feature_units

    def get_categorical_statistics(self, feature_name):
        """Returns a string with the frequencies of values of a categorical feature."""
        feature_value_frequencies = self.data[feature_name].value_counts()
        # Map feature indeces to feature names
        feature_id = self.data.columns.get_loc(feature_name)
        feature_value_frequencies.index = self.categorical_mapping[feature_id]
        # Sort by frequency
        feature_value_frequencies.sort_values(ascending=False, inplace=True)
        result_text = ""
        for i, (value, frequency) in enumerate(feature_value_frequencies.items()):
            result_text += f"The value <b>{value}</b> occurs <b>{frequency}</b> times.<br>"
        return result_text

    def get_numerical_statistics(self, feature_name):
        mean = round(self.data[feature_name].mean(), 2)
        std = round(self.data[feature_name].std(), 2)
        min_v = round(self.data[feature_name].min(), 2)
        max_v = round(self.data[feature_name].max(), 2)
        # make float values to strings where the . is replaced by a ,
        mean = str(mean).replace(".", ",")
        std = str(std).replace(".", ",")
        min_v = str(min_v).replace(".", ",")
        max_v = str(max_v).replace(".", ",")
        return feature_statistics_template(feature_name, mean, std, min_v, max_v, self.feature_units)

    def get_single_feature_statistic(self, feature_name):
        # Check if feature is numerical or categorical
        if feature_name in self.numerical_features:
            return self.get_numerical_statistics(feature_name)
        else:
            return self.get_categorical_statistics(feature_name)
