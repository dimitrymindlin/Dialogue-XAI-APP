# xai_utils.py

def process_xai_explanations(xai_explanations, predicted_class_name, opposite_class_name, instance_dict):
    numbered_importances = {}
    explanation_text_list = []
    favor_predicted = {}
    favor_alternative = {}
    feature_importances = xai_explanations["feature_importance"][0]
    top_feature_against_predicted = next(iter(feature_importances.values()))[0] < 0
    # copy and make instance_dict keys lower case and remove whitespaces
    feature_names_to_values = {k.lower().replace(" ", ""): v for k, v in instance_dict.items()}

    for idx, (feature_name, importance) in enumerate(
            sorted(feature_importances.items(), key=lambda item: abs(item[1][0]), reverse=True)
    ):
        feature_rank = idx + 1
        feature_value_name = feature_names_to_values.get(feature_name.lower().replace(" ", ""), "unknown")
        importance_percentage = round((importance[0] / sum(abs(val[0]) for val in feature_importances.values())) * 100)

        if importance[0] > 0:
            favor_predicted[feature_name] = {
                "importance": importance_percentage,
                "rank": feature_rank,
                "description": f"{feature_name} (value: {feature_value_name}) supports the prediction of ‘{predicted_class_name}’ "
                               f"with an importance of {importance_percentage}%, ranking #{feature_rank} overall."
            }
        else:
            favor_alternative[feature_name] = {
                "value": feature_value_name,
                "importance": importance_percentage,
                "rank": feature_rank,
                "description": f"{feature_name} (value: {feature_value_name}) supports the alternative class ‘{opposite_class_name}’ "
                               f"with an importance of {importance_percentage}%, ranking #{feature_rank} overall."
            }

    xai_explanations[f"feature_importance_in_favor_of_{predicted_class_name}"] = favor_predicted
    xai_explanations[f"feature_importance_in_favour_of{opposite_class_name}"] = favor_alternative

    for idx, (feature_name, importance) in enumerate(feature_importances.items()):
        # Calculate absolute percentage for clarity
        total_importance = sum(abs(val[0]) for val in feature_importances.values())
        importance_percentage = abs(importance[0]) / total_importance * 100
        importance_percentage = {round(importance_percentage)}
        # Determine the direction of influence
        influence_direction = (
            f"the predicted class (‘{predicted_class_name}’)"
            if importance[0] > 0 else
            f"the alternative class (‘{opposite_class_name}’)"
        )

        # Assign importance_tuple for flexibility in storing raw data and text explanation
        importance_tuple = (
            f"{importance_percentage} percent",  # Percentage importance
            influence_direction  # Influence direction
        )

        # Start explanation
        explanation_text = (
            f"Importance Rank {idx + 1}: {feature_name} strongly influences the prediction "
            f"toward {influence_direction} with an importance weight of {importance_percentage} percent. "
            f"In this case, being {feature_value_name} makes the model more "
            f"likely to predict ‘{opposite_class_name if importance[0] < 0 else predicted_class_name}’. "
        )

        # Add conditional part for the top feature
        if idx == 0 and top_feature_against_predicted:
            explanation_text += (
                f"While this feature strongly suggests ‘{opposite_class_name}’, the combination of other features "
                f"outweighs this, leading to the final prediction of ‘{predicted_class_name}’."
            )

        # Add to explanations
        numbered_importances[feature_name] = explanation_text

    xai_explanations["feature_importance"] = numbered_importances

    # Add descriptions to feature importances
    xai_explanations["feature_importance"][
        "method_description"] = "Positive values indicate attributes supporting the current " \
                                "prediction, while negative values indicate support for the " \
                                "alternative class. Focus on describing the most influential " \
                                "features first, explaining whether they favor or oppose the " \
                                "current prediction. Always reference the current prediction, " \
                                "the feature’s specific value, and its influence on the decision."

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


def prepare_dynamic_explanations(xai_explanations, predicted_class_name, opposite_class_name, instance_dict):
    # Helper function to generate formatted feature importance explanations
    def format_feature_importances(feature_dict, predicted_class_name, is_positive=True):
        return [
            {
                "label": f"Feature: {feature_name}",
                "description": feature_data["description"],
            }
            for feature_name, feature_data in feature_dict.items()
        ]

    # Compute feature importance details
    favor_predicted = {}
    favor_alternative = {}
    feature_importances = xai_explanations["feature_importance"][0]
    feature_names_to_values = {k.lower().replace(" ", ""): v for k, v in instance_dict.items()}

    for idx, (feature_name, importance) in enumerate(
            sorted(feature_importances.items(), key=lambda item: abs(item[1][0]), reverse=True)
    ):
        feature_rank = idx + 1
        feature_value_name = feature_names_to_values.get(feature_name.lower().replace(" ", ""), "unknown")
        importance_percentage = round((importance[0] / sum(abs(val[0]) for val in feature_importances.values())) * 100)

        if importance[0] > 0:
            favor_predicted[feature_name] = {
                "importance": importance_percentage,
                "rank": feature_rank,
                "description": f"{feature_name} (value: {feature_value_name}) supports the prediction of ‘{predicted_class_name}’ "
                               f"with an importance of {importance_percentage}%, ranking #{feature_rank} overall."
            }
        else:
            favor_alternative[feature_name] = {
                "value": feature_value_name,
                "importance": importance_percentage,
                "rank": feature_rank,
                "description": f"{feature_name} (value: {feature_value_name}) supports the alternative class ‘{opposite_class_name}’ "
                               f"with an importance of {importance_percentage}%, ranking #{feature_rank} overall."
            }

    # Format dynamic feature importance explanations
    feature_importance_dynamic = (
            format_feature_importances(favor_predicted, predicted_class_name, is_positive=True)
            + format_feature_importances(favor_alternative, opposite_class_name, is_positive=False)
    )

    # Prepare counterfactual dynamic content
    counterfactual_dynamic = [
        {
            "label": "Possible Counterfactuals",
            "explanation": xai_explanations["counterfactuals"],
            "dependencies": ["Concept"],
        }
    ]

    # Process feature statistics into clean text
    feature_statistics_string = "\n".join(
        f"{feature_name}: {stat.replace('<b>', '').replace('</b>', '').replace('<br>', '')}"
        for feature_name, stat in xai_explanations["feature_statistics"].items()
    )

    feature_statistics_dynamic = [
        {
            "label": "Feature Statistics Overview",
            "description": feature_statistics_string,
            "dependencies": [],
        }
    ]

    # Return structured dynamic content for merging
    return {
        "Feature Importance": feature_importance_dynamic,
        "Counterfactual Plan": counterfactual_dynamic,
        "Feature Statistics": feature_statistics_dynamic,
    }


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
