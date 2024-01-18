"""The base explanation class"""

class BaseExplainer:
    """The base explanation class."""

    def __init__(self, model):
        """Init.

        Args:
            model: The model to explain.
        """
        super(BaseExplainer, self).__init__()
        self.model = model

    @staticmethod
    def get_local_neighborhood(x):
        """
            Input : x : any given sample
            This function generates local neighbors of the input sample.
        """

    def get_explanation(self, data_x, label):
        """
        Input : x : Input sample
        Output : This function uses the explanation model to return explanations for the given sample
        TODO : 1. If we are using public implementations of the explanation methods, do we require predict function?
        """

    def evaluate_explanation(self, explanation, evaluation_metric: str,
                             ground_truth = None):
        """
        Input :-
        x : Input explanation for evaluation
        evaluation_metric : the evaluation metric to compute
        ground_truth : expected explanation
        Output : return evaluation metric value.
        """
