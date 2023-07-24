import pandas as pd


class FeatureStatisticsExplainer:
    def __init__(self,
                 data: pd.DataFrame,
                 numerical_features: list,
                 feature_names: list,
                 categorical_mapping,
                 rounding_precision: int = 2,
                 ):
        self.data = data
        self.numerical_features = numerical_features
        self.feature_names = feature_names
        self.rounding_precision = rounding_precision
        self.categorical_mapping = categorical_mapping

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
        mean = round(self.data[feature_name].mean(), self.rounding_precision)
        std = round(self.data[feature_name].std(), self.rounding_precision)
        min_v = round(self.data[feature_name].min(), self.rounding_precision)
        max_v = round(self.data[feature_name].max(), self.rounding_precision)
        return (f"Here are statistics for the feature <b>{feature_name}</b>: <br><br>"
                f"The <b>mean</b> is {mean},<br> one <b>standard deviation</b> is {std},<br>"
                f" the <b>minimum</b> value is {min_v},<br> and the <b>maximum</b> value is {max_v}.")

    def get_single_feature_statistic(self, feature_name):
        # Check if feature is numerical or categorical
        if feature_name in self.numerical_features:
            self.get_numerical_statistics(feature_name)
        else:
            return self.get_categorical_statistics(feature_name)
