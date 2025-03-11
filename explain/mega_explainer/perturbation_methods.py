"""TODO(satya): docstring."""
import numpy as np


def compare_torch_numpy(torch_tensor, numpy_array, message=""):
    """Compare torch tensor with numpy array and print differences."""
    if isinstance(torch_tensor, torch.Tensor):
        numpy_from_torch = torch_tensor.detach().cpu().numpy()
    else:
        numpy_from_torch = torch_tensor
        
    if not isinstance(numpy_array, np.ndarray):
        numpy_array = np.array(numpy_array)
        
    try:
        if not np.allclose(numpy_from_torch, numpy_array, equal_nan=True):
            print(f"Difference detected {message}:")
            print(f"Torch output: {numpy_from_torch}")
            print(f"Numpy output: {numpy_array}")
            print(f"Max difference: {np.max(np.abs(numpy_from_torch - numpy_array))}")
    except Exception as e:
        print(f"Error comparing arrays {message}: {str(e)}")
        print(f"Torch shape: {numpy_from_torch.shape}, dtype: {numpy_from_torch.dtype}")
        print(f"Numpy shape: {numpy_array.shape}, dtype: {numpy_array.dtype}")


class BasePerturbation:
    """Base Class for perturbation methods."""

    def __init__(self, data_format):
        """Initialize generic parameters for the perturbation method."""
        assert data_format == "tabular", "Currently, only tabular data is supported!"
        self.data_format = data_format

    def get_perturbed_inputs(self,
                            original_sample: np.ndarray,
                            feature_mask: np.ndarray,
                            num_samples: int,
                            feature_metadata: list,
                            max_distance: int = None) -> np.ndarray:
        """Logic of the perturbation methods which will return perturbed samples.

        This method should be overwritten.
        """


class NormalPerturbation(BasePerturbation):
    """TODO(satya): docstring.

    TODO(satya): Should we scale the std. based on the size of the feature? This could lead to
    some odd results if the features aren't scaled the same and we apply the same std noise
    across all the features.
    """

    def __init__(self,
                 data_format,
                 mean: float = 0.0,
                 std: float = 0.05,
                 flip_percentage: float = 0.3):
        """Init.

        Args:
            data_format: A string describing the format of the data, i.e., "tabular" for tabular
                         data.
            mean: the mean of the gaussian perturbations
            std: the standard deviation of the gaussian perturbations
            flip_percentage: The percent of features to flip while perturbing
        """
        self.mean = mean
        self.std_dev = std
        self.flip_percentage = flip_percentage
        super(NormalPerturbation, self).__init__(data_format)

    def get_perturbed_inputs(self,
                            original_sample: np.ndarray,
                            feature_mask: np.ndarray,
                            num_samples: int,
                            feature_metadata: list,
                            max_distance: int = None) -> np.ndarray:
        """Given a sample and mask, compute perturbations.

        Args:
            original_sample: The original instance
            feature_mask: the indices of the indices to mask where True corresponds to an index
                          that is to be masked. E.g., [False, True, False] means that index 1 will
                          not be perturbed while 0 and 2 _will_ be perturbed.
            num_samples: number of perturbed samples.
            feature_metadata: the list of 'c' or 'd' for whether the feature is categorical or
                              discrete.
            max_distance: the maximum distance between original sample and perturbed samples.
        Returns:
            perturbed_samples: The original_original sample perturbed with Gaussian perturbations
                               num_samples times.
        """
        feature_type = feature_metadata
        original_sample = original_sample.astype(float)  # Convert from object to float
        message = f"mask size == original sample in get_perturbed_inputs for {self.__class__}"
        assert len(feature_mask) == len(original_sample), message

        # Create feature type masks
        continuous_features = np.array([i == 'c' for i in feature_type])
        discrete_features = np.array([i == 'd' for i in feature_type])

        # Processing continuous columns
        mean = self.mean
        std_dev = self.std_dev
        perturbations = np.random.normal(mean, std_dev,
                                       [num_samples, len(feature_type)]) * continuous_features + original_sample

        # Processing discrete columns
        flip_percentage = self.flip_percentage
        p = np.full((num_samples, len(feature_type)), flip_percentage)
        perturbations = perturbations * (~discrete_features) + np.abs(
            (perturbations * discrete_features) - (np.random.binomial(1, p) * discrete_features))

        # Keeping features static based on feature mask
        perturbed_samples = np.array(original_sample) * feature_mask + perturbations * (~feature_mask)

        return perturbed_samples
