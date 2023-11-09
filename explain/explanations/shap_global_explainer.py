import numpy as np
import pandas as pd
import shap
import pickle
import os
import gin


@gin.configurable
class ShapGlobalExplainer:
    """This class generates Shap Global explanations for tabular data with its own caching system."""

    def __init__(self, model, data: pd.DataFrame, link: str = 'identity', class_names: dict = None,
                 cache_location: str = "./cache/shap-global-tabular.pkl"):
        self.data = data
        self.model = model
        self.link = link
        self.class_names = list(class_names.values()) if class_names else []
        self.cache_location = cache_location
        self.explainer = shap.KernelExplainer(self.model.predict, shap.kmeans(data, 25), link=self.link)
        self.cache = self.load_cache()

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

    def get_explanations(self, use_cache: bool = True) -> shap.Explanation:
        """Generate SHAP global explanations and cache them."""
        # Check if cached explanations exist
        if use_cache and 'global_shap_values' in self.cache:
            print("Using cached SHAP values.")
            return self.cache['global_shap_values']

        # Compute SHAP values if not using cache or if they are not in cache
        print("Computing SHAP values...")
        shap_values = self.explainer.shap_values(self.data)
        shap_values = shap.Explanation(values=shap_values, feature_names=self.data.columns,
                                       output_names=self.class_names)

        # Cache the newly computed SHAP values
        if use_cache:
            self.cache['global_shap_values'] = shap_values
            self.save_cache()

        return shap_values
