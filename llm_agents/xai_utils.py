# xai_utils.py

def process_xai_explanations(xai_explanations):
    numbered_importances = {}
    feature_importances = xai_explanations["feature_importance"][0]
    for idx, (feature_name, importance) in enumerate(feature_importances.items()):
        new_feature_name = f"{idx}_{feature_name}"
        numbered_importances[new_feature_name] = importance
    xai_explanations["feature_importance"] = numbered_importances

    # Add descriptions to feature importances
    xai_explanations["feature_importance"]['method_description'] = (
        "Positive values indicate attributes in favor of the current prediction and negative values against it. "
        "While the absolute values don't have a clear interpretation, they indicate the difference in the "
        "weight of the feature in the prediction. Do not name all the features, but the most important ones at first. "
        "Simplify the statements about feature importances with saying in favour or against the prediction and name the current prediction."
    )
    # Add descriptions to counterfactual explanations
    xai_explanations["counterfactuals"] = {"possible_counterfactuals": xai_explanations["counterfactuals"]}
    xai_explanations["counterfactuals"]['method_description'] = (
        "Counterfactual explanations show examples of feature changes that would flip the model prediction. "
        "The examples are not complete, meaning that there are more possible ways. However, the counterfactual "
        "method tries to find minimal counterfactuals first, only changing one attribute. For one feature, the "
        "output is similar to ceteris paribus explanations."
    )
    # Remove unnecessary keys
    xai_explanations.pop("model_prediction", None)
    xai_explanations.pop("instance_type", None)

    return xai_explanations


def extract_instance_information(instance_information):
    return instance_information[1]
