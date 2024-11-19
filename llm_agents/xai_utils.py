# xai_utils.py

def process_xai_explanations(xai_explanations, predicted_class_name, opposite_class_name):
    numbered_importances = {}
    feature_importances = xai_explanations["feature_importance"][0]
    for idx, (feature_name, importance) in enumerate(feature_importances.items()):
        new_feature_name = f"Importance Rank {idx + 1}: {feature_name}"
        importance_tuple = (
            importance,
            f"in favour of {predicted_class_name}" if importance[0] > 0 else f"in favour of {opposite_class_name}")
        numbered_importances[new_feature_name] = importance_tuple
    xai_explanations["feature_importance"] = numbered_importances

    # Add descriptions to feature importances
    xai_explanations["feature_importance"]['method_description'] = (
        "Positive values indicate attributes in favour of the current prediction and negative values against it. "
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

    # make feature statistics clearner, by removing html b tags and the feature name
    feature_statistics = xai_explanations["feature_statistics"]
    feature_statistics_string = ""
    for feature_name, feature_stat in feature_statistics.items():
        feature_stat = feature_stat.replace("<b>", "")
        feature_stat = feature_stat.replace("</b>", "")
        feature_stat = feature_stat.replace("<br>", "")
        if feature_name not in feature_stat:
            feature_stat = f"{feature_name}: {feature_stat}"
        feature_statistics_string += f"{feature_stat}\n"
    xai_explanations["feature_statistics"] = feature_statistics_string
    return xai_explanations


def get_xai_explanations_as_goal_notepad(xai_explanations):
    """
    Turn all the explanations into a list of goals for the user to understand.
    """
    goal_notepad = []
    for explanation_type, explanation in xai_explanations.items():
        if explanation_type == "feature_importance":
            fi_string = f"Feature Attributions - Understand the importance of the top three important features: "
            feature_names = list(explanation.keys())
            if "method_description" in feature_names:
                feature_names.remove("method_description")
            fi_string += f"{feature_names[0]}, {feature_names[1]}, {feature_names[2]}"
            goal_notepad.append(fi_string)
            fi_string = f"Feature Attributions - Understand the importance of the lesat three important features: "
            fi_string += f"{feature_names[-3]}, {feature_names[-2]}, {feature_names[-1]}"
            goal_notepad.append(fi_string)

        elif explanation_type == "counterfactuals":
            goal_notepad.append(
                "Counterfactuals - Understand how changing the features can affect the model's decision.")
        elif explanation_type == "feature_statistics":
            goal_notepad.append(
                "Feature Statistics - Understand the statistics of the top 3 most important features in the model's decision.")
        elif explanation_type == "ceteris_paribus":
            goal_notepad.append(
                "Ceteris Paribus - Understand the individual effect of changing the top 3 features on the model's decision.")
    return goal_notepad


def get_xai_explanations_as_goal_user_model(xai_explanations, user_model):
    """
    Turn all the explanations into labeled goals for the user to understand and populate the UserModel.
    """
    for explanation_type, explanation in xai_explanations.items():
        if explanation_type == "feature_importance":
            # Top three important features
            fi_string = "Feature Attributions - Understand the importance of the top three important features: "
            feature_names = list(explanation.keys())
            if "method_description" in feature_names:
                feature_names.remove("method_description")
            fi_string += f"{feature_names[0]}, {feature_names[1]}, {feature_names[2]}"
            user_model.add_not_explained_yet("TopFeatureImportance", fi_string)

            # Least three important features
            fi_string = "Feature Attributions - Understand the importance of the least three important features: "
            fi_string += f"{feature_names[-3]}, {feature_names[-2]}, {feature_names[-1]}"
            user_model.add_not_explained_yet("LeastFeatureImportance", fi_string)

        elif explanation_type == "counterfactuals":
            user_model.add_not_explained_yet(
                "Counterfactuals",
                "Understand how changing the features can affect the model's decision."
            )

        elif explanation_type == "feature_statistics":
            user_model.add_not_explained_yet(
                "FeatureStatistics",
                "Understand the statistics of the top 3 most important features in the model's decision."
            )

        elif explanation_type == "ceteris_paribus":
            user_model.add_not_explained_yet(
                "CeterisParibus",
                "Understand the individual effect of changing the top 3 features on the model's decision."
            )


def extract_instance_information(instance_information):
    return instance_information[1]
