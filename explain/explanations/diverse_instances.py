import os
import time
from typing import List
import pickle as pkl
import gin
import pandas as pd
import numpy as np


def load_cache(cache_location: str):
    """Loads the cache."""
    if os.path.isfile(cache_location):
        with open(cache_location, 'rb') as file:
            cache = pkl.load(file)
    else:
        cache = []
    return cache


@gin.configurable
class DiverseInstances:
    """This class finds DiverseInstances by using LIMEs submodular pick."""

    def __init__(self,
                 cache_location: str = "./cache/diverse-instances.pkl",
                 instance_amount: int = 5,
                 lime_explainer=None):
        """

        Args:
            cache_location: location to save the cache
            lime_explainer: lime explainer to use for finding diverse instances (from MegaExplainer)
        """
        self.diverse_instances = load_cache(cache_location)
        self.cache_location = cache_location
        self.lime_explainer = lime_explainer
        self.instance_amount = instance_amount

    def filter_instances_by_marital_status(self, data, diverse_instances_pandas_indices, instance_amount):
        """
        Filters instances to ensure that "marital status" alternates between instances.

        Parameters:
        - data (DataFrame): The dataset containing the instances.
        - diverse_instances_pandas_indices (list): Indices of potential instances to filter from.
        - instance_amount (int): The desired number of instances to include in the filtered list.

        Returns:
        - List of indices representing filtered instances based on marital status alternation.
        """
        filtered_instances = []
        last_marital_status = -1  # Initialize to a value not in [0, 1].

        for i in diverse_instances_pandas_indices:
            current_marital_status = data.loc[i, "MaritalStatus"]

            # Ensure alternation of marital status
            if current_marital_status != last_marital_status:
                filtered_instances.append(i)
                last_marital_status = current_marital_status

                if len(filtered_instances) >= instance_amount:
                    break

        # shuffle the instances to ensure randomness
        filtered_instances = np.random.permutation(filtered_instances).tolist()

        return filtered_instances

    def get_instance_ids_to_show(self,
                                 data: pd.DataFrame,
                                 model,
                                 y_values: List[int],
                                 save_to_cache=True,
                                 submodular_pick=False) -> List[int]:
        """
        Returns diverse instances for the given data set.
        Args:
            data: pd.Dataframe the data instances to use to find diverse instances
            instance_count: number of diverse instances to return
            save_to_cache: whether to save the diverse instances to the cache
        Returns: List of diverse instance ids.

        """
        if len(self.diverse_instances) > 0:
            return self.diverse_instances

        counter = 0
        while len(self.diverse_instances) < self.instance_amount:

            # Generate diverse instances
            if submodular_pick:
                print(f"Using submodular pick to find {self.instance_amount} diverse instances.")
                diverse_instances = self.lime_explainer.get_diverse_instance_ids(data.values, self.instance_amount)
                # Get pandas index for the diverse instances
                diverse_instances_pandas_indices = [data.index[i] for i in diverse_instances]
            else:
                # Get random instances
                dynamic_seed = int(time.time()) % 10000
                # Get 100 times more instances to filter and ensure diversity
                diverse_instances_pandas_indices = data.sample(self.instance_amount * 100,
                                                               random_state=dynamic_seed).index.tolist()

                diverse_instances_pandas_indices = self.filter_instances_by_marital_status(data,
                                                                                           diverse_instances_pandas_indices,
                                                                                           self.instance_amount)

            for i in diverse_instances_pandas_indices:
                if i not in self.diverse_instances:
                    self.diverse_instances.append(i)

            counter += 1
            if counter > 20:
                print(f"Could not find enough diverse instances, only found {len(self.diverse_instances)}.")
                break

        # TODO: This is hacky and only for Diabetes dataset. Move to data preprocessing
        """# remove instance with id 123
        diverse_instances_pandas_indices.remove(123)"""

        if save_to_cache:
            with open(self.cache_location, 'wb') as file:
                pkl.dump(diverse_instances_pandas_indices, file)
        return diverse_instances_pandas_indices
