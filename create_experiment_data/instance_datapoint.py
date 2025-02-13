class InstanceDatapoint:
    def __init__(self, instance_id,
                 instance_as_dict,
                 class_probabilities,
                 model_predicted_label_string,
                 model_predicted_label=None):
        self.instance_id = instance_id
        self.instance_as_dict = instance_as_dict
        self.class_probabilities = class_probabilities
        self.model_predicted_label_string = model_predicted_label_string
        self.model_predicted_label = model_predicted_label
        self.displayable_features = None
        self.instance_type = None
        self.counter = -1

    def get_datapoint_as_dict_for_frontend(self):
        # return a dict with frontend-needed information
        try:
            datapoint_dict = {
                "id": self.instance_id,
                "probabilities": self.class_probabilities.tolist(),
                "ml_prediction": self.model_predicted_label_string,
                "displayable_features": self.displayable_features,
            }
        except AttributeError:
            datapoint_dict = {
                "id": self.instance_id,
                "ml_prediction": self.model_predicted_label_string,
                "displayable_features": self.displayable_features,
            }
        return datapoint_dict
