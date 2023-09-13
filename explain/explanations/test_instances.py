import os
from typing import List
import pickle as pkl
import gin
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_cache(cache_location: str):
    """Loads the cache."""
    if os.path.isfile(cache_location):
        with open(cache_location, 'rb') as file:
            cache = pkl.load(file)
    else:
        cache = []
    return cache


@gin.configurable
class TestInstances:
    """This class creates test instances for the diverse instances"""

    def __init__(self,
                 data, model, mega_explainer, experiment_helper, diverse_instance_ids,
                 actionable_features,
                 cache_location: str = "./cache/test-instances.pkl"):
        """

        Args:
            cache_location: location to save the cache
            lime_explainer: lime explainer to use for finding diverse instances (from MegaExplainer)
        """
        self.data = data
        self.diverse_instance_ids = diverse_instance_ids
        self.cache_location = cache_location
        self.model = model
        self.mega_explainer = mega_explainer
        self.experiment_helper = experiment_helper
        self.actionable_features = actionable_features
        self.test_instances = load_cache(cache_location)

    def get_test_instances(self,
                           instance_count: int = 10,
                           save_to_cache=True) -> List[int]:
        """
        Returns diverse instances for the given data set.
        Args:
            data: pd.Dataframe the data instances to use to find diverse instances
            instance_count: parameter to determine how many diverse instances should be generated to choose from
                            (high value means more diverse instances but also more computation time)
            save_to_cache: whether to save the diverse instances to the cache
        Returns: List of diverse instance ids.

        """
        if len(self.test_instances) > 0:
            return self.test_instances

        test_instances = {}
        for instance_id in self.diverse_instance_ids:
            # Get the instance as a pandas dataframe
            original_instance = self.data.loc[instance_id].to_frame().transpose()
            # Get model prediction
            original_class_prediction = np.argmax(self.model.predict_proba(original_instance)[0])
            # Get the feature importances
            feature_importances = self.mega_explainer.get_feature_importances(original_instance)[0][
                original_class_prediction]
            # get a list of similar instances
            similar_instances = [
                self.experiment_helper.get_similar_instance(original_instance, self.model, self.actionable_features)
                for _
                in range(instance_count)]
            similar_instances = pd.concat(similar_instances)
            # Sort instances by complexity
            similar_instances = self.sort_instances_by_complexity(original_instance, similar_instances,
                                                                  feature_importances,
                                                                  self.model.predict_proba)
            # Get the most complex instance (i.e. the first one in the df)
            most_complex_instance = similar_instances.head(1)
            # Get the least complex instance (i.e. the last one in the df)
            least_complex_instance = similar_instances.tail(1)
            # get a list of counterfactual instance by varying the easiest instance
            counterfactual_instances = self.experiment_helper.get_counterfactual_instances(least_complex_instance)
            # Restructure counterfactual instances list to be list of one row dataframes

            # Sort instances by complexity
            counterfactual_instances = self.sort_instances_by_complexity(original_instance, counterfactual_instances,
                                                                         feature_importances, self.model.predict_proba)

            # get an easy counterfactual instance
            easy_counterfactual_instance = counterfactual_instances.tail(1)

            # get a hard counterfactual instance
            hard_counterfactual_instance = counterfactual_instances.head(1)

            # Save most_complex, least_complex and counterfactual instance in dict
            test_instances[original_instance.index[0]] = {"most_complex_instance": most_complex_instance,
                                                          "least_complex_instance": least_complex_instance,
                                                          "easy_counterfactual_instance": easy_counterfactual_instance,
                                                          "hard_counterfactual_instance": hard_counterfactual_instance}
        # Save dict to pkl file
        if save_to_cache:
            with open(self.cache_location, 'wb') as file:
                pkl.dump(test_instances, file)
        return test_instances

    def calculate_prediction_task_complexity(self,
                                             original_instance,
                                             new_instance,
                                             feature_importances,
                                             model_certainty_old,
                                             model_certainty_new,
                                             ):
        """
        Calculates the prediction task complexity for the given instance.
        The complexity is a measure that consists of the (weighted by FI) sparsity of the changes,
        the similarity of the new_instance to the original instance, and the certainty of the model prediction.
        original_instance: the original instance
        new_instance: the new instance that was created for the prediction task
        feature_importances: the feature importances for the original instance
        model_certainty_old: the certainty of the model prediction for the original instance
        model_certainty_new: the certainty of the model prediction for the new instance
        """
        # Make sure the instances are dataframes and have features in columns
        if not isinstance(original_instance, pd.DataFrame):
            original_instance = pd.DataFrame.from_dict(original_instance, orient='index')
        if not isinstance(new_instance, pd.DataFrame):
            new_instance = pd.DataFrame.from_dict(new_instance, orient='index')
        if original_instance.shape[0] != 1:
            original_instance = original_instance.transpose()
        if new_instance.shape[0] != 1:
            new_instance = new_instance.transpose()

        # Calculate similarity between original and new instance
        similarity = cosine_similarity(original_instance, new_instance)[0][0]  # cos similarity, 0 = different, 1 = same

        # Calculate sparsity of changes (i.e. how many features were changed) weighted by feature importance
        # Get the absolute values of the feature importances
        feature_importances = {feature: abs(value[0]) for feature, value in feature_importances.items()}

        max_possible_difference = {feature: self.data[feature].max() - self.data[feature].min() for feature in
                                   new_instance}

        max_weighted_sparsity = sum(
            feature_importances[feature] * max_possible_difference[feature] for feature in new_instance)

        # Calculate the original weighted sparsity as before
        weighted_sparsity = sum(
            feature_importances[feature] * abs(new_instance[feature].values - original_instance[feature].values) for
            feature in new_instance)[0]

        # Normalize
        normalized_weighted_sparsity = weighted_sparsity / max_weighted_sparsity if max_weighted_sparsity != 0 else 0
        inverse_normalized_weighted_sparsity = 1 - normalized_weighted_sparsity  # 0 = different, 1 = same

        """# Calculate difference in certainty of model prediction, considering if the class changed
        prediction_certainty_difference = abs(model_certainty_old[0] - model_certainty_new[0]) + \
                                          abs(model_certainty_old[1] - model_certainty_new[1])
    
        print(prediction_certainty_difference)"""

        # Calculate prediction task complexity
        combined_metric = (similarity + inverse_normalized_weighted_sparsity) / 2
        return combined_metric

    def sort_instances_by_complexity(self,
                                     original_instance,
                                     new_instances,
                                     feature_importances,
                                     pred_f,
                                     drop_importances=True):
        """
        Sorts the new instances by their prediction task complexity.
        Args:
            original_instance: the original instance
            new_instances: the new instances that were created for the prediction task
            feature_importances: the feature importances for the original instance
            pred_f: the prediction function of the model
            drop_importances: whether to drop the feature importances column from the new instances df
        Returns: the new instances sorted by their prediction task complexity
        """
        # turn list of dfs into df
        if isinstance(new_instances, list):
            new_instances = pd.concat(new_instances)

        original_logits = pred_f(original_instance)[0]
        instances_complexities = []
        # iterate over new instances df
        for _, new_instance in new_instances.iterrows():
            # turn new instance into df
            new_instance = new_instance.to_frame().transpose()
            # make sure new instance has same data type as original instance
            new_instance = new_instance.astype(original_instance.dtypes[0])
            try:
                new_instance_logits = pred_f(new_instance)[0]
            except ValueError:
                new_instance = new_instance.astype('float64')
                new_instance_logits = pred_f(new_instance)[0]
            # calculate prediction task complexity
            instance_complexity = self.calculate_prediction_task_complexity(original_instance, new_instance,
                                                                            feature_importances, original_logits,
                                                                            new_instance_logits)
            instances_complexities.append(instance_complexity)

        # Return sorted df
        new_instances['importance'] = instances_complexities
        new_instances = new_instances.sort_values(by='importance', ascending=False)  # most important is on top
        if drop_importances:
            new_instances = new_instances.drop(columns='importance')
        return new_instances
